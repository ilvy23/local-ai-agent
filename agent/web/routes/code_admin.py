"""The rest of the code agent's GUI: sessions (delete, prune, export), repo policy and the command risk test, checkpoints, project memory, the
coding models, evaluation runs, statistics and benchmarks.

Everything wraps the product's own modules (agent.codeagent.policy / sessionstore / timing / verified, agent.safety, agent.repoaccess, agent.undo,
agent.projectmem, agent.coding, scripts/*.py); nothing is re-implemented. Anything that loosens access needs a typed word, as in code_runs.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib.util
import io
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import agent
from agent import projectmem, repoaccess, undo
from agent.codeagent import eventlog, policy as policy_mod, sessionstore, timing
from agent.safety import RiskLevel, block_reason, classify_command
from agent.web.deps import get_client, get_config, get_store
from agent.web.routes import code_runs

router = APIRouter(prefix="/api/code", tags=["code-admin"])
REPO = Path(agent.__file__).resolve().parent.parent
_RANK = {"ask": 0, "medium": 1, "auto": 2}


def _script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    if not path.is_file():
        raise HTTPException(404, f"scripts/{name}.py is not part of this installation.")
    spec = importlib.util.spec_from_file_location(f"_agent_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)           # type: ignore[arg-type]
    sys.path.insert(0, str(REPO))
    try:
        spec.loader.exec_module(mod)                      # type: ignore[union-attr]
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(REPO))
    return mod


def _find(session_id: str) -> sessionstore.Meta:
    meta, why = sessionstore.find(session_id)
    if meta is None:
        raise HTTPException(404, why)
    return meta


def _git(root: Path, *args: str, timeout: int = 30) -> tuple[bool, str]:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return done.returncode == 0, done.stdout if done.returncode == 0 else (done.stderr or done.stdout)


# ------------------------------------------------------------------------------------------------------------------ sessions


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    meta = _find(session_id)
    if meta.state == "in progress" and sessionstore.folder_busy(Path(meta.root)):
        raise HTTPException(409, f"{meta.id} is running right now in {meta.root}.")
    if not sessionstore.delete(meta):
        raise HTTPException(500, f"Could not delete {meta.id}.")
    return {"deleted": meta.id}


class PruneIn(BaseModel):
    days: float = Field(gt=0, le=3650)
    dry_run: bool = False


@router.post("/sessions/prune")
def prune_sessions(body: PruneIn) -> dict[str, Any]:
    cut = time.time() - body.days * 86400
    if body.dry_run:
        old = [m for m in sessionstore.list_sessions() if m.updated < cut and m.state != "in progress"]
        return {"would_delete": [{"id": m.id, "title": m.title, "root": m.root} for m in old], "deleted": []}
    gone = sessionstore.prune(body.days)
    return {"deleted": [{"id": m.id, "title": m.title, "root": m.root} for m in gone], "would_delete": []}


@router.get("/sessions/{session_id}/export")
def export_session(session_id: str) -> Response:
    """One saved session as redacted JSONL (the trajectory format of scripts/export_trajectories.py: one record, home paths and secrets removed)."""
    meta = _find(session_id)
    try:
        data = sessionstore.load(meta.path)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    ex = _script("export_trajectories")
    home = str(Path.home())
    msgs = [m for m in data.get("messages", []) if isinstance(m, dict) and "role" in m]
    rec = {"id": meta.id, "source": meta.id, "task": ex.clean(meta.title, home), "model": meta.model, "state": meta.state,
           "messages": [ex.convert(m, home) for m in msgs]}
    return Response(json.dumps(rec, ensure_ascii=False) + "\n", media_type="application/x-ndjson",
                    headers={"Content-Disposition": f'attachment; filename="trajectory-{meta.id}.jsonl"'})


class ExportAllIn(BaseModel):
    confirm: str = ""


@router.post("/sessions/export-all")
def export_all(body: ExportAllIn) -> dict[str, Any]:
    """Training material from every saved session (scripts/export_trajectories.py): writes data/trajectories/{sft,pairs}.jsonl in the repository."""
    if body.confirm != "export":
        raise HTTPException(422, "Type the word 'export' to confirm.")
    ex = _script("export_trajectories")
    out = REPO / "data" / "trajectories"
    stats = ex.export(sessionstore.state_root().parent, out, str(Path.home()))
    return {**stats, "out": str(out)}


@router.get("/why-slow")
def why_slow(session: str | None = None, since: str | None = None) -> dict[str, Any]:
    """`agent code time`: where the time of a run went, and the likely causes. `session` is a saved session id (or an event-log session prefix)."""
    try:
        after = time.time() - timing.parse_since(since) if since else None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    events = timing.select(timing.read_log(), since=after)
    label = "all sessions" if since and not session else ""
    if session:
        events = [e for e in events if str(e.get("session", "")).rsplit("-", 1)[-1].startswith(session) or str(e.get("session", "")).startswith(session)]
        label = f"session {session}"
    elif not since:
        latest = events[-1]["session"] if events and events[-1].get("session") else None
        events = [e for e in events if e.get("session") == latest] if latest else []
        label = f"session {latest}" if latest else ""
    if not events:
        return {"label": label, "empty": True, "breakdown": None, "table": "", "causes": [], "parts": []}
    b = timing.breakdown(events)
    parts = [{"label": "Prefill", "pct": round(b.prefill_share * 100), "seconds": round(b.prefill_s, 1)},
             {"label": "Generation", "pct": round(b.gen_share * 100), "seconds": round(b.gen_s, 1)},
             {"label": "Tools", "pct": round(b.tool_share * 100), "seconds": round(b.tool_s, 1)},
             {"label": "Waiting", "pct": round(b.wait_share * 100), "seconds": round(b.wait_s, 1)}]
    return {"label": label, "empty": False, "breakdown": dataclasses.asdict(b), "table": b.table(), "causes": timing.why_slow(b), "parts": parts}


@router.get("/state")
async def folder_state(folder: str) -> dict[str, Any]:
    """`agent code state`: the verified state of a folder, measured now (runs the project's own tests; no model)."""
    root = code_runs.resolve_folder(folder)
    return await asyncio.to_thread(_state_text, root)


def _state_text(root: Path) -> dict[str, Any]:
    from rich.console import Console

    from agent.codeagent import verified
    from agent.codeagent.engine import CodeEngine
    from agent.codeagent.eventlog import EventLog
    from agent.codeagent.session import _NoModel
    from agent.config import load_config

    engine = CodeEngine(root, load_config(), _NoModel(), Console(file=io.StringIO()), interactive=False, events=EventLog(f"{root.name}-state-{int(time.time())}"))
    meta = sessionstore.latest(root)
    if meta is not None:
        with contextlib.suppress(ValueError, OSError, KeyError, TypeError):
            sessionstore.restore(engine, sessionstore.load(meta.path))
    return {"folder": str(root), "session": meta.id if meta else "", "text": verified.render_cli(verified.build(engine))}


# ------------------------------------------------------------------------------------------------------------------ policy


def _pol(p: policy_mod.RepoPolicy) -> dict[str, Any]:
    return dataclasses.asdict(p)


@router.get("/policy")
def get_policy() -> dict[str, Any]:
    return {"repos": [{"path": k, **_pol(v), "exists": Path(k).is_dir()} for k, v in sorted(policy_mod.all_repos().items())],
            "defaults": _pol(policy_mod.load_defaults()), "greenlist": [str(g) for g in repoaccess.greenlist()], "modes": list(policy_mod.MODES)}


class PolicyIn(BaseModel):
    folder: str
    changes: dict[str, Any]
    confirm: str | None = None


def _clean_changes(changes: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in changes.items():
        if k in ("network", "installs"):
            if not isinstance(v, bool):
                raise HTTPException(422, f"{k} must be on or off.")
            out[k] = v
        elif k in ("allow", "deny"):
            if not isinstance(v, list) or len(v) > 200:
                raise HTTPException(422, f"{k} must be a list of command patterns.")
            pats = [str(x).strip() for x in v if str(x).strip()]
            if any(len(x) > 300 or "\n" in x for x in pats):
                raise HTTPException(422, f"A {k} pattern is too long or has a line break.")
            out[k] = pats
        elif k in ("max_steps", "ctx"):
            if v in (None, "") and k == "ctx":
                out[k] = None
            else:
                out[k] = code_runs._int(v, k, 5 if k == "max_steps" else 2048, 500 if k == "max_steps" else 2_000_000)
        elif k == "mode":
            if v not in policy_mod.MODES:
                raise HTTPException(422, "mode must be ask, medium or auto.")
            out[k] = v
        elif k == "commit":
            if v not in ("never", "ask", "auto"):
                raise HTTPException(422, "commit must be never, ask or auto.")
            out[k] = v
        elif k == "think":
            if v not in ("auto", "on", "off"):
                raise HTTPException(422, "think must be auto, on or off.")
            out[k] = v
        elif k in ("model", "test_cmd"):
            out[k] = (str(v).strip() or None) if v is not None else None
        else:
            raise HTTPException(422, f"Unknown setting {k!r}.")
    return out


def _loosening(before: policy_mod.RepoPolicy, changes: dict[str, Any]) -> str | None:
    """The typed word a change needs, or None. 'auto' for full access; 'yes' for anything else that gives the agent more room."""
    if changes.get("mode") and _RANK[changes["mode"]] > _RANK[before.mode]:
        return "auto" if changes["mode"] == "auto" else "yes"
    more = (changes.get("network") is True and not before.network) or (changes.get("installs") is True and not before.installs) \
        or (changes.get("commit") == "auto" and before.commit != "auto") or bool(set(changes.get("allow", [])) - set(before.allow)) \
        or bool(set(before.deny) - set(changes.get("deny", before.deny)))
    return "yes" if more else None


@router.put("/policy")
def put_policy(body: PolicyIn) -> dict[str, Any]:
    folder, note = repoaccess.resolve_folder(body.folder)
    if folder is None:
        raise HTTPException(422, note)
    changes = _clean_changes(body.changes)
    if changes.get("mode") not in (None, "ask"):
        problem = repoaccess.folder_problem(folder)
        if problem:
            raise HTTPException(422, f"Can't grant that here: {problem}")
    before = policy_mod.for_repo(folder)
    word = _loosening(before, changes)
    if word and body.confirm != word:
        raise HTTPException(422, f"This gives the agent more room in {folder}. Type '{word}' to confirm.")
    pol = policy_mod.save_repo(folder, **changes)
    return {"path": str(folder), **_pol(pol)}


@router.delete("/policy")
def forget_policy(folder: str) -> dict[str, bool]:
    return {"removed": policy_mod.forget_repo(Path(folder).expanduser())}


class GreenIn(BaseModel):
    folder: str
    confirm: str = ""


@router.post("/policy/greenlist")
def greenlist_add(body: GreenIn) -> dict[str, str]:
    if body.confirm != "greenlist":
        raise HTTPException(422, "Type the word 'greenlist' to confirm: the folder gets full access in every future session.")
    folder, note = repoaccess.resolve_folder(body.folder)
    if folder is None:
        raise HTTPException(422, note)
    _kind, msg = repoaccess.handle_command(f"/greenlist-{folder}") or ("", "")
    return {"message": msg}


@router.delete("/policy/greenlist")
def greenlist_remove(folder: str) -> dict[str, str]:
    _kind, msg = repoaccess.handle_command(f"/ungreenlist-{folder}") or ("", "")
    return {"message": msg}


class ClassifyIn(BaseModel):
    command: str = Field(max_length=5000)
    folder: str | None = None


@router.post("/policy/classify")
def classify(body: ClassifyIn, config=Depends(get_config)) -> dict[str, Any]:
    """The command risk classifier (agent.safety) and, with a folder, what that repo's policy would do with the command."""
    level = classify_command(body.command, config)
    out: dict[str, Any] = {"command": body.command, "tier": level.name, "why": block_reason(body.command, config) if level is RiskLevel.BLOCKED else "",
                           "verdict": None, "reason": ""}
    if body.folder:
        root = code_runs.resolve_folder(body.folder)
        pol = policy_mod.for_repo(root)
        jail = policy_mod.forbidden_in_command(body.command, root) if body.command.strip() else None
        d = policy_mod.Decision("block", jail) if jail else policy_mod.check_command(pol, root, body.command, level.name)
        out.update(verdict=d.verdict, reason=d.reason, mode=pol.mode)
    return out


# ------------------------------------------------------------------------------------------------------------------ models


@router.get("/models")
def models(client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    from agent.coding import models as cm

    installed: list[str] = []
    with contextlib.suppress(Exception):
        installed = sorted(client.list_models())
    loaded: list[str] = []
    with contextlib.suppress(Exception):
        loaded = list(client.loaded_models())
    return {"coding": code_runs.effective_coding(config)["executor"]["model"], "agent": (config.get("code") or {}).get("model") or config["models"]["chat"],
            "installed": installed, "loaded": loaded, "recommended": [{"model": t, "note": n} for t, n in cm.RECOMMENDED],
            "routing_override": str((config.get("models") or {}).get("coding") or "")}


class ModelIn(BaseModel):
    model: str = Field(min_length=1, max_length=200)


@router.put("/models/coding")
def set_coding_model(body: ModelIn, config=Depends(get_config)) -> dict[str, Any]:
    """`agent code-model NAME`: the executor of the classic builder. models.coding (the routing table) wins over coding.executor.model in
    coding.config.load, so when it is set it is kept in sync, as that module's docstring requires."""
    name = body.model.strip()
    code_runs.write_coding_defaults({"executor_model": name}, config)
    if (config.get("models") or {}).get("coding"):
        import yaml

        from agent import config as config_mod

        path = config_mod.DEFAULT_CONFIG_PATH
        raw = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
        raw.setdefault("models", {})["coding"] = name
        path.write_text(yaml.dump(raw, sort_keys=False))
        config["models"]["coding"] = name
    return {"coding": name}


class ProseIn(BaseModel):
    request: str = Field(min_length=1, max_length=2000)
    model: str | None = None
    gpu: bool = False
    folder: str | None = None


@router.post("/prose")
async def prose_try(body: ProseIn) -> dict[str, str]:
    """`agent code prose "..."`: try the prose model without the agent (CPU only unless gpu)."""
    from agent.codeagent import prose

    buf = io.StringIO()
    root_arg = str(code_runs.resolve_folder(body.folder)) if body.folder else None

    def run() -> int:
        import contextlib as cl

        with cl.redirect_stdout(buf):
            return prose.prose_cli(body.request, body.model, body.gpu, root_arg)

    code = await asyncio.to_thread(run)
    return {"exit_code": str(code), "text": buf.getvalue()[-6000:]}


# ------------------------------------------------------------------------------------------------------------------ subprocess jobs (eval, benchmark)


class SubJob:
    """A child process whose output is kept as lines; start/poll/stop for the GUI."""

    def __init__(self, kind: str, cmd: list[str], label: str, model: str = "") -> None:
        self.id = f"{kind}-{int(time.time() * 1000) % 10**9:x}"
        self.kind, self.cmd, self.label, self.model = kind, cmd, label, model
        self.started, self.ended, self.exit_code = time.time(), None, None
        self.lines: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.stopped = False

    def start(self) -> None:
        self.proc = subprocess.Popen(self.cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True, name=self.id).start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.lines.append(line.rstrip())
            del self.lines[:-3000]
        self.exit_code = self.proc.wait()
        self.ended = time.time()
        _record(self)

    def stop(self) -> None:
        self.stopped = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def public(self, since: int = 0) -> dict[str, Any]:
        state = "running" if self.ended is None else "stopped" if self.stopped else "done" if self.exit_code == 0 else "failed"
        return {"id": self.id, "kind": self.kind, "label": self.label, "state": state, "started": self.started, "ended": self.ended,
                "exit_code": self.exit_code, "log": self.lines[since:], "next": len(self.lines), "model": self.model}


_JOBS: dict[str, SubJob] = {}
_JOBS_LOCK = threading.Lock()


def _history_file() -> Path:
    return sessionstore.state_root().parent / "code-jobs.json"


def _history() -> list[dict[str, Any]]:
    try:
        data = json.loads(_history_file().read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


_PASS = re.compile(r"pass@1\s+([0-9.]+)\s+\((\d+)/(\d+)")


def _record(job: SubJob) -> None:
    """Keep a short record of every finished eval / benchmark so the Evaluation and Stats tabs have a history."""
    text = "\n".join(job.lines)
    m = _PASS.search(text)
    rec = {"id": job.id, "kind": job.kind, "label": job.label, "model": job.model, "started": job.started, "ended": job.ended, "exit_code": job.exit_code,
           "stopped": job.stopped, "passed": int(m.group(2)) if m else None, "total": int(m.group(3)) if m else None,
           "seconds": round((job.ended or job.started) - job.started, 1), "tail": job.lines[-40:], "log": job.lines[-400:]}
    with contextlib.suppress(OSError):
        rows = [*_history(), rec][-60:]
        _history_file().parent.mkdir(parents=True, exist_ok=True)
        _history_file().write_text(json.dumps(rows, ensure_ascii=False) + "\n")


def _launch(job: SubJob) -> dict[str, Any]:
    with _JOBS_LOCK:
        busy = next((j for j in _JOBS.values() if j.ended is None), None)
        if busy is not None:
            raise HTTPException(409, f"{busy.label} is still running; the GPU can only do one at a time.")
    if (a := code_runs._active()) is not None:
        raise HTTPException(409, f"Run {a.id} is using the model; stop it first.")
    job.start()
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    return {"id": job.id}


class EvalIn(BaseModel):
    kind: str = "fix"                        # fix | agentic
    seeds: int = 3
    trials: int = 2
    confirm: str = ""


def eval_command(body: EvalIn) -> list[str]:
    args = ["code-eval"]
    if body.kind == "agentic":
        args += ["--agentic", "--trials", str(code_runs._int(body.trials, "trials", 1, 10))]
    else:
        args += ["--seeds", str(code_runs._int(body.seeds, "seeds", 1, 20))]
    return [sys.executable, "-c", "from agent.main import app; app()", *args]


@router.get("/evals")
def evals(config=Depends(get_config)) -> dict[str, Any]:
    hist = [h for h in _history() if h["kind"] in ("eval-fix", "eval-agentic")]
    from agent.coding.evals.tasks import TASKS

    with _JOBS_LOCK:
        live = [j.public() for j in _JOBS.values() if j.kind.startswith("eval") and j.ended is None]
    return {"runs": [{k: v for k, v in h.items() if k not in ("tail", "log")} for h in reversed(hist)], "active": live[0] if live else None,
            "tasks": [t.name for t in TASKS], "model": code_runs.effective_coding(config)["executor"]["model"]}


@router.post("/evals")
def start_eval(body: EvalIn, config=Depends(get_config)) -> dict[str, Any]:
    if body.kind not in ("fix", "agentic"):
        raise HTTPException(422, "kind must be fix or agentic.")
    if body.confirm != "run":
        raise HTTPException(422, "An evaluation runs the real coding model on the GPU for a long time. Type 'run' to confirm.")
    cmd = eval_command(body)
    return _launch(SubJob(f"eval-{body.kind}", cmd, f"Evaluation ({body.kind})", code_runs.effective_coding(config)["executor"]["model"]))


@router.get("/evals/history/{job_id}")
def eval_log(job_id: str) -> dict[str, Any]:
    for h in _history():
        if h["id"] == job_id:
            return {"id": job_id, "log": h.get("log") or h.get("tail") or []}
    raise HTTPException(404, "No such recorded run.")


@router.get("/jobs/{job_id}")
def job_state(job_id: str, since: int = 0) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job (jobs live as long as the server process; finished ones are in the history).")
    return job.public(since)


@router.post("/jobs/{job_id}/stop")
def job_stop(job_id: str) -> dict[str, bool]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    job.stop()
    return {"stopping": True}


@router.get("/model-compare")
def model_compare() -> dict[str, Any]:
    """Pass rate and median time per coding model, from the evaluation runs that were recorded here (code-model changes the executor)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for h in _history():
        if h["kind"] == "eval-fix" and h.get("total") and h.get("model"):
            by.setdefault(h["model"], []).append(h)
    rows = [{"model": m, "runs": len(rs), "pass_rate": round(100 * sum(r["passed"] for r in rs) / sum(r["total"] for r in rs)),
             "median_s": round(statistics.median(r["seconds"] for r in rs))} for m, rs in sorted(by.items())]
    return {"rows": rows}


# ------------------------------------------------------------------------------------------------------------------ checkpoints

_CKPT = re.compile(r"^refs/agent/checkpoints/[0-9]{8}-[0-9]{6}[A-Za-z0-9_.-]{0,20}$")
_BRANCH = re.compile(r"^agent-code/[0-9]+$")


def _repo_root(folder: str) -> Path:
    root = code_runs.resolve_folder(folder)
    ok, top = _git(root, "rev-parse", "--show-toplevel")
    if not ok:
        raise HTTPException(422, f"{root} is not a git repository, so it has no checkpoints.")
    return Path(top.strip())


def _ref_ok(ref: str) -> str:
    if not (_CKPT.match(ref) or _BRANCH.match(ref) or ref == "HEAD"):
        raise HTTPException(422, "Not a checkpoint or agent branch.")
    return ref


@router.get("/checkpoints")
def checkpoints(folder: str) -> dict[str, Any]:
    root = _repo_root(folder)
    ok, out = _git(root, "for-each-ref", "--sort=creatordate", "--format=%(refname)|%(objectname:short)|%(creatordate:unix)|%(subject)", "refs/agent/checkpoints", "refs/heads/agent-code")
    items = []
    for line in out.splitlines() if ok else []:
        ref, short, ts, subject = (line.split("|", 3) + ["", "", "", ""])[:4]
        if ref.startswith("refs/heads/"):
            ref = ref[len("refs/heads/"):]
        if not (_CKPT.match(ref) or _BRANCH.match(ref)):
            continue
        items.append({"ref": ref, "short": short, "when": int(ts or 0), "subject": subject, "kind": "branch" if _BRANCH.match(ref) else "checkpoint"})
    items.sort(key=lambda i: -i["when"])
    return {"root": str(root), "items": items}


@router.get("/checkpoints/detail")
def checkpoint_detail(folder: str, ref: str, against: str = "previous") -> dict[str, Any]:
    """Files and diff of a checkpoint against the previous one, the working tree ("now"), or another checkpoint."""
    root = _repo_root(folder)
    _ref_ok(ref)
    if against == "now":
        base = ["diff", ref]
    elif against == "previous":
        base = ["diff", f"{ref}^", ref]
    else:
        base = ["diff", _ref_ok(against), ref]
    ok, stat = _git(root, *base, "--numstat")
    files = []
    added = removed = 0
    for ln in stat.splitlines() if ok else []:
        a, r, name = (ln.split("\t") + ["", "", ""])[:3]
        a_n = int(a) if a.isdigit() else 0
        r_n = int(r) if r.isdigit() else 0
        added, removed = added + a_n, removed + r_n
        files.append({"path": name, "added": a_n, "removed": r_n})
    ok2, diff = _git(root, *base)
    return {"ref": ref, "against": against, "files": files, "added": added, "removed": removed, "diff": diff[:200_000] if ok2 else "", "error": "" if ok else stat[:300]}


class RestoreIn(BaseModel):
    folder: str
    ref: str
    confirm: str = ""


@router.post("/checkpoints/restore")
def restore_checkpoint(body: RestoreIn) -> dict[str, Any]:
    """`git checkout <ref> -- .` (what undo.py documents). A safety checkpoint of the current tree is taken first, so a restore can be undone."""
    root = _repo_root(body.folder)
    if not _CKPT.match(body.ref):
        raise HTTPException(422, "Only checkpoints can be restored; an agent branch is merged in the terminal.")
    if body.confirm != "restore":
        raise HTTPException(422, "Type the word 'restore' to confirm.")
    for r in list(code_runs._RUNS.values()):
        if r.active and Path(r.folder) == root:
            raise HTTPException(409, "A run is working in this folder; stop it first.")
    ok, sha = _git(root, "rev-parse", "--verify", "-q", body.ref + "^{commit}")
    if not ok:
        raise HTTPException(404, "No such checkpoint.")
    # the sha is read first: undo.git_checkpoint names refs by the second, so a safety checkpoint taken in the same second would reuse (and overwrite) the ref
    safety = undo.git_checkpoint(str(root / "x"))
    ok, msg = _git(root, "checkout", sha.strip(), "--", ".")
    if not ok:
        raise HTTPException(500, f"git checkout failed: {msg[:300]}")
    return {"restored": body.ref, "safety_checkpoint": safety}


# ------------------------------------------------------------------------------------------------------------------ project memory


def _key(folder: str) -> tuple[Path, str]:
    root = code_runs.resolve_folder(folder)
    return root, projectmem.project_key(root) or str(root)


@router.get("/memory")
def memory(folder: str, store=Depends(get_store)) -> dict[str, Any]:
    root, key = _key(folder)
    pm = projectmem.ProjectMemory(store.conn)
    return {"folder": str(root), "key": key, "notes": [{"id": i, "note": n} for i, n in pm.notes(key, limit=200)],
            "test_cmd": policy_mod.for_repo(root).test_cmd or "", "repos": [{"key": k, "name": Path(k).name} for k in _known_projects(store)]}


def _known_projects(store) -> list[str]:
    projectmem.ProjectMemory(store.conn)
    rows = store.conn.execute("SELECT project FROM project_notes GROUP BY project ORDER BY MAX(created_at) DESC").fetchall()
    return [r[0] for r in rows]


class NotesIn(BaseModel):
    folder: str
    notes: str = Field(max_length=20000)


@router.put("/memory/notes")
def set_notes(body: NotesIn, store=Depends(get_store)) -> dict[str, Any]:
    """The notes box: one note per line. Lines that vanished are forgotten, new lines are added (notes are short, 8 to 300 characters)."""
    _root, key = _key(body.folder)
    pm = projectmem.ProjectMemory(store.conn)
    want = [" ".join(ln.split()) for ln in body.notes.splitlines() if ln.strip()]
    have = {n: i for i, n in pm.notes(key, limit=200)}
    for note, i in have.items():
        if note not in want:
            store.conn.execute("DELETE FROM project_notes WHERE id = ?", (i,))
    store.conn.commit()
    added = [w for w in want if w not in have and pm.add(key, w)]
    return {"notes": [{"id": i, "note": n} for i, n in pm.notes(key, limit=200)], "added": len(added), "skipped": [w for w in want if w not in have and w not in added]}


class CheckIn(BaseModel):
    folder: str
    test_cmd: str = Field(max_length=500)


@router.put("/memory/check")
def set_check(body: CheckIn) -> dict[str, str]:
    root = code_runs.resolve_folder(body.folder)
    cmd = body.test_cmd.strip()
    if "\n" in cmd:
        raise HTTPException(422, "The check is one command.")
    policy_mod.save_repo(root, test_cmd=cmd or None)
    return {"test_cmd": cmd}


@router.delete("/memory")
def forget_project(folder: str, store=Depends(get_store)) -> dict[str, Any]:
    root, key = _key(folder)
    n = projectmem.ProjectMemory(store.conn).clear(key)
    policy_mod.save_repo(root, test_cmd=None)
    return {"forgotten": n}


# ------------------------------------------------------------------------------------------------------------------ stats


@router.get("/stats")
def stats(since: str | None = None) -> dict[str, Any]:
    """Run statistics (scripts/run_stats.py) from the event log and the saved sessions, plus the median generation speed."""
    rs = _script("run_stats")
    events = rs.load_events(eventlog.default_path())
    cutoff = rs.parse_since(since) if since else None
    if cutoff is not None:
        events = [e for e in events if e["ts"] >= cutoff]
    by: dict[str, list[dict]] = {}
    for e in events:
        by.setdefault(e["session"], []).append(e)
    metas = rs.load_metas(sessionstore.state_root())
    rows = sorted((rs.analyse(s, v, metas, 300.0) for s, v in by.items()), key=lambda r: r["start"])
    rates = [e["gen"] / e["decode_s"] for e in events if e.get("kind") == "step" and (e.get("decode_s") or 0) > 0.5 and (e.get("gen") or 0) > 30]
    done = sum(1 for r in rows if r["outcome"].startswith(("finished", "autopilot done")) or r.get("saved_state") == "finished")
    return {"overall": rs.overall(rows) if rows else None, "sessions": rows[-100:], "runs": len(rows), "finished": done,
            "median_tps": round(statistics.median(rates), 1) if rates else None}


@router.get("/stats/speed")
def speed(session: str | None = None) -> dict[str, Any]:
    """The speed-trial judgement (scripts/speed_trial.py --dry-run) of a recorded run: prefill share and steps per minute against the thresholds."""
    st = _script("speed_trial")
    events = timing.read_log()
    if not events:
        return {"empty": True, "lines": [], "ok": None}
    sess = session or events[-1].get("session")
    b = timing.breakdown(events, session=sess)
    ok, lines = st.verdict(b, 0.25, 6.0)
    return {"empty": False, "session": sess, "ok": ok, "lines": lines, "steps": b.steps, "steps_per_min": round(b.steps_per_min, 1),
            "prefill_share": round(b.prefill_share * 100)}


@router.post("/stats/cache-report")
async def cache_report() -> dict[str, Any]:
    """Size and stability of the fixed prompt prefix (scripts/prefix_report.py). Builds contexts for a generated sample repo; no model is called."""
    pr = _script("prefix_report")

    def run() -> dict[str, Any]:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, pr.isolated_state():
            repo = pr.make_sample_repo(Path(tmp) / "sample")
            rep = pr.build_report(repo)
            return {"text": pr.render(rep), "breakers": list(rep.breakers), "notes": list(rep.notes),
                    "parts": [{"name": p.name, "tokens": p.tokens} for p in rep.parts[:15]]}

    return await asyncio.to_thread(run)


class BenchIn(BaseModel):
    kind: str = "server"                    # server | dry
    model: str | None = None
    repeats: int = 3
    confirm: str = ""


@router.get("/bench")
def bench_history() -> dict[str, Any]:
    rows = [h for h in _history() if h["kind"].startswith("bench")]
    with _JOBS_LOCK:
        live = [j.public() for j in _JOBS.values() if j.kind.startswith("bench") and j.ended is None]
    return {"runs": [{k: v for k, v in h.items() if k != "log"} for h in reversed(rows)], "active": live[0] if live else None}


def bench_command(body: BenchIn) -> list[str]:
    cmd = [sys.executable, str(REPO / "scripts" / "server_bench.py"), "--repeats", str(code_runs._int(body.repeats, "repeats", 1, 10))]
    if body.model:
        cmd += ["--model", body.model.strip()]
    return cmd + (["--dry-run"] if body.kind == "dry" else ["--yes"])


@router.post("/bench")
def start_bench(body: BenchIn) -> dict[str, Any]:
    """scripts/server_bench.py: the plan only (--dry-run, sends nothing) or the real A/B benchmark (--yes, GPU busy for a long time)."""
    if body.kind not in ("server", "dry"):
        raise HTTPException(422, "kind must be server or dry.")
    if body.kind == "server" and body.confirm != "benchmark":
        raise HTTPException(422, "The benchmark sends real generation requests and keeps the GPUs busy. Type 'benchmark' to confirm.")
    if not (REPO / "scripts" / "server_bench.py").is_file():
        raise HTTPException(404, "scripts/server_bench.py is not part of this installation.")
    return _launch(SubJob(f"bench-{body.kind}", bench_command(body), "Server benchmark" + (" (plan only)" if body.kind == "dry" else ""), body.model or ""))
