"""Goal runs of the code agent, started from the GUI as background jobs.

The same two engines `agent code` has: the agent (agent.codeagent: one prompt, `--stages`, `--autopilot`) and the classic sandbox builder / fixer
(agent.coding, `--classic`). The run executes in a thread; what the engine prints is kept as log lines, its saved session feeds the existing run
timeline (`/api/code/sessions/{id}/timeline`), and anything that needs a human goes through the engine's own policy: a call the repo policy rates
"ask" blocks the job until the GUI answers (`/decide`), exactly where the terminal would have asked "run it? [y/N]". A DANGEROUS call needs the
typed word `yes`. Nothing here loosens a policy: the access level of a run can only be raised with the typed word, and the folder rules
(repoaccess.folder_problem) apply as in the CLI.
"""

from __future__ import annotations

from agent import brand

import contextlib
import copy
import dataclasses
import io
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console

from agent import repoaccess
from agent.codeagent import policy as policy_mod, sessionstore
from agent.coding import config as coding_config, models as coding_models
from agent.web.deps import get_client, get_config

router = APIRouter(prefix="/api/code", tags=["code-runs"])

APPROVAL_TIMEOUT_S = 1800          # an unanswered question is a "no" after this long (the job must never hang forever)
MAX_LOG_LINES = 3000
MAX_RUNS_KEPT = 20
_RANK = {"ask": 0, "medium": 1, "auto": 2}
EFFORTS = ("min", "mid", "max")

_RUNS: dict[str, "Run"] = {}
_LOCK = threading.RLock()


# ---------------------------------------------------------------------------------------------------------------- the job object


class _Stopped(KeyboardInterrupt):
    """Raised inside the job thread at a step boundary when the user pressed Stop (KeyboardInterrupt is what the engines already handle)."""


class _Sink(io.TextIOBase):
    """File object behind the rich Console: complete lines become log lines of the run."""

    def __init__(self, run: "Run") -> None:
        self.run, self._buf = run, ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.run.log(line)
        return len(s)

    def flush(self) -> None:
        pass


class JobConsole(Console):
    """A Console with no terminal: no spinner, and `input` is an end of file (nobody types into a server)."""

    def status(self, *_a: Any, **_k: Any):  # type: ignore[override]
        return contextlib.nullcontext()

    def input(self, *_a: Any, **_k: Any) -> str:  # type: ignore[override]
        raise EOFError


@dataclasses.dataclass
class Run:
    id: str
    engine: str                                   # agent | classic
    folder: str
    goal: str
    opts: dict[str, Any]
    started: float = dataclasses.field(default_factory=time.time)
    ended: float | None = None
    outcome: str = ""                             # "" while running | done | partial | needs_human | failed | stopped
    session_id: str = ""
    exit_code: int | None = None
    summary: str = ""
    result: dict[str, Any] = dataclasses.field(default_factory=dict)
    lines: list[tuple[int, str]] = dataclasses.field(default_factory=list)
    items: list[dict[str, Any]] = dataclasses.field(default_factory=list)       # classic runs: structured steps (agent runs use the session timeline)
    approval: dict[str, Any] | None = None
    _answer: bool | None = None
    _seq: int = 0
    stop_flag: threading.Event = dataclasses.field(default_factory=threading.Event)
    resume_flag: threading.Event = dataclasses.field(default_factory=threading.Event)
    answered: threading.Event = dataclasses.field(default_factory=threading.Event)
    paused: bool = False
    thread: threading.Thread | None = None
    obj: Any = None                               # the CodeEngine (agent runs), kept after the end so /undo works
    _tokens: int = 0

    def log(self, text: str) -> None:
        with _LOCK:
            self._seq += 1
            self.lines.append((self._seq, text.rstrip()))
            del self.lines[:-MAX_LOG_LINES]

    def item(self, kind: str, tone: str, title: str, detail: str = "") -> None:
        with _LOCK:
            self.items.append({"t": round(time.time() - self.started, 1), "kind": kind, "tone": tone, "title": title, "detail": detail})
            del self.items[:-1000]

    @property
    def state(self) -> str:
        if self.outcome:
            return self.outcome
        if self.approval is not None:
            return "waiting"
        return "paused" if self.paused else "running"

    @property
    def active(self) -> bool:
        return not self.outcome


def _public(run: Run, since: int = 0) -> dict[str, Any]:
    with _LOCK:
        lines = [{"seq": s, "text": t} for s, t in run.lines if s > since]
        eng = run.obj if run.engine == "agent" else None
        meters: dict[str, Any] = {"elapsed_s": round((run.ended or time.time()) - run.started, 1)}
        plan: dict[str, Any] | None = None
        if eng is not None:
            with contextlib.suppress(Exception):
                meters.update(steps=int(eng.ctx.step), max_steps=int(eng.budget_now), step_ceiling=int(eng.step_ceiling), budget_source=eng.budget_source, ctx_used=int(eng.ctx.estimate()), ctx_limit=int(eng.num_ctx),
                              tokens_per_s=round(float(eng.avg_tps()), 1), model=eng.model, mode=eng.pol.mode, in_turn=bool(eng.in_turn),
                              undo_available=len(eng.undo))
            with contextlib.suppress(Exception):
                from agent.codeagent import planview

                view = planview.build(eng)
                if view.stages or view.checklist:                 # nothing staged / no checklist yet: leave `plan` out entirely
                    plan = planview.to_dict(view)
        elif run.engine == "classic":
            meters.update(steps=len([i for i in run.items if i["kind"] in ("round", "attempt")]), tokens=run._tokens)
        return {"id": run.id, "engine": run.engine, "folder": run.folder, "goal": run.goal, "state": run.state, "outcome": run.outcome,
                "started": run.started, "ended": run.ended, "session_id": run.session_id, "exit_code": run.exit_code, "summary": run.summary,
                "result": run.result, "approval": run.approval, "paused": run.paused, "log": lines, "next": run._seq, "items": run.items[-300:],
                "meters": meters, "opts": run.opts, "plan": plan}


def _get(run_id: str) -> Run:
    with _LOCK:
        run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "No such run (the list only keeps the latest runs of this server process; saved sessions are under Sessions).")
    return run


def _active() -> Run | None:
    with _LOCK:
        return next((r for r in _RUNS.values() if r.active), None)


def _prune_runs() -> None:
    with _LOCK:
        done = sorted((r for r in _RUNS.values() if not r.active), key=lambda r: r.started)
        for r in done[:-MAX_RUNS_KEPT]:
            _RUNS.pop(r.id, None)


# ---------------------------------------------------------------------------------------------------------------- validation


class StartIn(BaseModel):
    folder: str
    goal: str = ""
    engine: str = "agent"                                      # agent | classic
    files: list[str] = Field(default_factory=list)             # focus files (agent) / files to fix (classic)
    test: str | None = None
    mode: str | None = None                                    # ask | medium | auto, this run only
    confirm_access: str | None = None                          # the typed word when the run is more permissive than the saved policy
    remember: bool = False                                     # also save the mode for this folder (what /mode does)
    stages: bool = False
    autopilot: bool = False
    new: bool = True
    resume: str | None = None                                  # a saved session id
    keep_model: bool = False
    init_git: bool = False                                     # classic: make a baseline commit when the folder has no repository
    model: str | None = None                                   # this run's model
    max_steps: int | None = None                               # agent: tool steps per prompt
    stage_max_minutes: int | None = None
    stage_max_steps: int | None = None
    stage_prompt_cap: int | None = None
    prose_model: str | None = None                             # "" = none
    protect: list[str] | None = None
    effort: str | None = None                                  # classic
    diagnose_first: bool | None = None
    auto_commit: bool | None = None
    max_repair_attempts: int | None = None
    escalation_enabled: bool | None = None
    escalation_model: str | None = None
    escalation_budget_s: int | None = None
    timeout_s: int | None = None
    memory_mb: int | None = None
    budget_max_actions: int | None = None
    budget_max_seconds: int | None = None
    budget_max_tokens: int | None = None


def _int(v: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise HTTPException(422, f"{name} must be a whole number from {lo} to {hi}.")
    return v


def clean_patterns(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 50:
        raise HTTPException(422, "protect patterns must be a list of at most 50 patterns.")
    out = []
    for p in value:
        p = str(p).strip()
        if not p:
            continue
        if len(p) > 200 or "\n" in p or "\x00" in p:
            raise HTTPException(422, f"Bad protect pattern: {p[:40]!r}")
        out.append(p)
    return out


def coding_overlay(body: StartIn) -> dict[str, Any]:
    """The `coding:` config keys this run changes (never written to config.yaml here). Validated like the CLI/config would."""
    o: dict[str, Any] = {}
    if body.effort is not None:
        if body.effort not in EFFORTS:
            raise HTTPException(422, "effort must be min, mid or max.")
        o.setdefault("build", {})["effort"] = body.effort
    if body.diagnose_first is not None:
        o["diagnose_first"] = bool(body.diagnose_first)
    if body.auto_commit is not None:
        o["auto_commit"] = bool(body.auto_commit)
    if body.max_repair_attempts is not None:
        o.setdefault("executor", {})["max_repair_attempts"] = _int(body.max_repair_attempts, "max repair attempts", 1, 20)
    if body.engine == "classic" and body.model:
        o.setdefault("executor", {})["model"] = body.model.strip()
    esc: dict[str, Any] = {}
    if body.escalation_enabled is not None:
        esc["tier3_enabled"] = bool(body.escalation_enabled)
    if body.escalation_model is not None:
        esc["tier3_model"] = body.escalation_model.strip()
    if body.escalation_budget_s is not None:
        esc["total_budget_seconds"] = _int(body.escalation_budget_s, "escalation budget", 10, 86400)
    if esc:
        o["escalation"] = esc
    sb: dict[str, Any] = {}
    if body.timeout_s is not None:
        sb["timeout_seconds"] = _int(body.timeout_s, "command timeout", 1, 3600)
    if body.memory_mb is not None:
        sb["memory_mb"] = _int(body.memory_mb, "memory limit", 256, 262144)
    if sb:
        o["sandbox"] = sb
    bu: dict[str, Any] = {}
    for key, v, lo, hi in (("max_actions", body.budget_max_actions, 1, 100000), ("max_seconds", body.budget_max_seconds, 10, 604800),
                           ("max_tokens", body.budget_max_tokens, 1000, 100000000)):
        if v is not None:
            bu[key] = _int(v, key.replace("_", " "), lo, hi)
    if bu:
        o["budget"] = bu
    if body.prose_model is not None:
        o["prose_model"] = body.prose_model.strip()
    if body.protect is not None:
        o["protect"] = clean_patterns(body.protect)
    for key, v, lo, hi in (("stage_max_minutes", body.stage_max_minutes, 0, 1440), ("stage_max_steps", body.stage_max_steps, 0, 5000),
                           ("stage_prompt_cap", body.stage_prompt_cap, 0, 1000000)):
        if v is not None:
            o[key] = _int(v, key.replace("_", " "), lo, hi)
    return o


def merge_coding(config: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    base = cfg.setdefault("coding", {})

    def merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(base, overlay)
    return cfg


def resolve_folder(text: str) -> Path:
    """The project folder of a run: absolute, existing, never the home directory or a system folder (repoaccess), git root when inside a repo."""
    from agent.codeagent.session import find_root

    folder, note = repoaccess.resolve_folder(text)
    if folder is None:
        raise HTTPException(422, note)
    root = find_root(folder)
    if root == Path.home() or len(root.parts) < 3:
        raise HTTPException(422, "Run this inside a project folder, not your home or a top-level folder.")
    return root


def check_files(root: Path, files: list[str]) -> list[str]:
    out = []
    for f in files[:50]:
        f = str(f).strip()
        if not f:
            continue
        p = (root / f).resolve()
        if Path(f).is_absolute() or not policy_mod.inside(p, root):
            raise HTTPException(422, f"{f!r} is outside the project folder.")
        out.append(f)
    return out


def check_access(root: Path, mode: str | None, confirm: str | None) -> None:
    """The access level of one run. More permissive than the folder's saved policy needs the typed word, and folders the CLI refuses stay refused."""
    if mode is None:
        return
    if mode not in _RANK:
        raise HTTPException(422, "mode must be ask, medium or auto.")
    if mode != "ask":
        problem = repoaccess.folder_problem(root)
        if problem:
            raise HTTPException(422, f"Can't grant {mode} here: {problem}")
    if _RANK[mode] > _RANK[policy_mod.for_repo(root).mode] and confirm != mode:
        raise HTTPException(422, f"Giving this run '{mode}' access is more than the folder has now. Type the word '{mode}' to confirm.")


# ---------------------------------------------------------------------------------------------------------------- the engines


def _release_lock(root: Path) -> None:
    from agent.codeagent import session as S

    for item in list(S._held_locks):
        if item[0] == str(root):
            with contextlib.suppress(Exception):
                item[1].close()
            S._held_locks.remove(item)


def _ask_human(run: Run, description: str, reason: str) -> bool:
    """The engine's approve_fn: blocks the job thread until the GUI decides (or Stop, or the timeout = no)."""
    verb = description.split()[0] if description else ""
    risky_cmd = verb in ("run", "tui")
    ap = {"approval_id": secrets.token_hex(3), "tool_name": "run_command" if risky_cmd else "write_file",
          "arguments": {"command": description.split(None, 1)[1] if risky_cmd and " " in description else description} if risky_cmd else {"path": description},
          "risk_level": "DANGEROUS" if "dangerous" in reason else "CAUTION", "description": description, "reason": reason, "asked": time.time()}
    run._answer = None
    run.answered.clear()
    with _LOCK:
        run.approval = ap
    run.log(f"? {description}" + (f"  ({reason})" if reason else ""))
    run.item("approval", "warn", "Waiting for your decision", description)
    try:
        run.answered.wait(timeout=APPROVAL_TIMEOUT_S)
        if run.stop_flag.is_set():
            raise _Stopped()
        ok = bool(run._answer)
    finally:
        with _LOCK:
            run.approval = None
    run.log("  " + ("approved" if ok else "declined"))
    return ok


def _step_hook(run: Run, save: Callable[[], None]) -> Callable[[], None]:
    """Runs before every model call: autosave, pause, stop."""
    def hook() -> None:
        save()
        while run.paused and not run.stop_flag.is_set():
            run.resume_flag.wait(0.2)
        if run.stop_flag.is_set():
            raise KeyboardInterrupt
    return hook


def _build_engine(run: Run, root: Path, config: dict[str, Any], client: Any, console: Console):
    """The CodeEngine of a run, built like session.open_engine but asking through the GUI. Tests replace this."""
    from agent import projectmem
    from agent.codeagent.engine import CodeEngine
    from agent.codeagent.eventlog import EventLog
    from agent.memory.store import Store
    from agent.tools.web import make_web_tools

    store = Store(config["data"]["db_path"])
    web: list = []
    with contextlib.suppress(Exception):
        web = make_web_tools(console)
    key = projectmem.project_key(root) or str(root)
    engine = CodeEngine(root, config, client, console, approve_fn=lambda d, r: _ask_human(run, d, r), web_tools=web,
                        notes=projectmem.ProjectMemory(store.conn), note_key=key, events=EventLog(f"{root.name}-{int(time.time())}"), interactive=False)
    return engine, store


def _exec_agent(run: Run, body: StartIn, root: Path, config: dict[str, Any], client: Any, console: Console) -> None:
    from agent import foreground
    from agent.codeagent import session as S
    from agent.codeagent.supervisor import should_autopilot

    base = S.with_stages(S.with_keep_model(config, body.keep_model), body.stages)
    goal = body.goal.strip()
    task = goal if not body.files else f"{goal}\n\nFocus on these files: {', '.join(body.files)}."
    autopilot = body.autopilot or should_autopilot(goal, base, root)
    chosen = None
    if body.resume:
        chosen, why = sessionstore.find(body.resume, root)
        if chosen is None:
            raise RuntimeError(why)
    run_config = S.with_autopilot(base) if autopilot else base
    ok, owner = S.acquire_lock(root)
    if not ok:
        raise RuntimeError(f"Another coding session is already working in {root} (pid {owner}). Finish or stop that one first.")
    engine = store = None
    try:
        with foreground.session(config):
            engine, store = _build_engine(run, root, run_config, client, console)
            run.obj = engine
            with contextlib.suppress(Exception):
                S.free_gpu(console, client, engine.model)
            meta = chosen or (None if body.new else S.pick_session(console, root, "latest", False))
            info = S.resume_into(engine, meta) if meta is not None else None
            if info is None:
                engine.session_id = sessionstore.new_id()
                engine.start()
            else:
                S.resume_banner(console, engine, info)
            engine.events.session = f"{root.name}-{engine.session_id}"
            run.session_id = engine.session_id
            engine.on_step = _step_hook(run, lambda: S.save_session(engine))
            if body.model:
                engine.model = body.model.strip()
            if body.max_steps:
                engine.pol = dataclasses.replace(engine.pol, max_steps=_int(body.max_steps, "steps per prompt", 5, 500), steps_set=True)
                engine.step_ceiling = engine.budget_now = engine.pol.max_steps
                engine.budget_source = "user"
            if body.test:
                engine.profile.test_cmd = body.test
            if body.mode and not autopilot:
                S.autopilot_mode(engine, body.mode)
            console.print(f"{brand.CMD} code  {root}  {engine.model}  mode {engine.pol.mode}  session {engine.session_id}")
            run.item("start", "info", "Run started", f"{engine.model} · {engine.pol.mode} mode" + (" · autopilot" if autopilot else ""))
            try:
                if autopilot:
                    code = S.run_autopilot(engine, task, mode=body.mode)
                    run.exit_code = code
                    run.outcome = {0: "done", 2: "needs_human", 3: "partial", 130: "stopped"}.get(code, "failed")
                    run.summary = {0: "All stages done.", 2: "Needs a human: a stage is blocked.", 3: "Partly done.", 130: "Stopped."}.get(code, f"Ended with code {code}.")
                else:
                    report = S.run_task_staged(engine, task)
                    if report is not None:
                        run.exit_code = 0 if report.verdict == "done" else 1
                        run.outcome = "done" if report.verdict == "done" else "partial"
                        run.summary = f"Staged run: {report.verdict}."
                    else:
                        res = engine.turn(task)
                        run.exit_code = 0 if res.stopped == "" and res.text is not None else 1
                        run.result = {"text": res.text, "files": res.files, "stopped": res.stopped, "verified": res.verified, "not_verified": res.not_verified}
                        run.outcome = "stopped" if res.stopped == "interrupted" else "done" if run.exit_code == 0 else "partial"
                        run.summary = (res.text or f"Stopped: {res.stopped or 'no answer'}")[:2000]
            finally:
                with contextlib.suppress(Exception):
                    S.save_session(engine)
                with contextlib.suppress(Exception):
                    S.release_gpu_on_exit(console, client, config)
    finally:
        _release_lock(root)
        if store is not None:
            with contextlib.suppress(Exception):
                store.close()


def _exec_classic(run: Run, body: StartIn, root: Path, config: dict[str, Any], client: Any, console: Console) -> None:
    from agent import coding_cli
    from agent.coding import build as build_mod, repair
    from agent.coding.verify import runner
    from agent.events import Events

    if runner.run(["git", "rev-parse", "HEAD"], cwd=root).ok is not True:
        if not body.init_git:
            raise RuntimeError("The classic builder needs a git repository with a commit (its sandbox is a worktree). Tick 'make a baseline commit' or run git init in the folder.")
        runner.run(["git", "init", "-q"], cwd=root)
        if coding_cli._baseline_commit(console, root) is None:
            raise RuntimeError("Refusing to auto-commit this folder (too broad). Pick the specific project folder.")
    ev = Events()

    def on(e: Any) -> None:
        if run.stop_flag.is_set():
            raise KeyboardInterrupt
        k, d = e.kind, e.data
        while run.paused and not run.stop_flag.is_set():
            run.resume_flag.wait(0.2)
        if k == "token":
            run._tokens += 1
            return
        if k == "build_round":
            run.item("round", "info", f"Round {d['n']}/{d['of']}", "writing the whole program" if d["phase"] == "generate" else "fixing checks")
        elif k == "attempt":
            run.item("attempt", "info", f"Attempt {d['n']}/{d['of']}", str(d.get("model", "")))
        elif k == "edit":
            run.item("edit", "info", "Edited", str(d["path"]))
        elif k == "check_passed":
            run.item("check", "ok", "Checks pass")
        elif k == "check_failed":
            run.item("check", "danger", "Check failed", str(d.get("stage", "")))
        elif k == "diagnose_end":
            run.item("diagnose", "info", "Diagnosis", str(d.get("text", ""))[:200])
        elif k == "escalate":
            run.item("model", "warn", "Escalating", str(d.get("to", "")))
        elif k == "stall":
            run.item("stall", "warn", "Stalled", "identical output twice")
        elif k == "auto_install":
            run.item("install", "info", "Installing", " ".join(d.get("packages") or []))
        elif k == "ctx_backoff":
            run.item("context", "neutral", "Context lowered", str(d.get("num_ctx")))
        run.log(f"{k} {d}" if k not in ("generate_start", "generate_end") else k)

    ev.subscribe(on)
    cfg = config
    model = coding_config.load(cfg)["executor"]["model"]
    with contextlib.suppress(Exception):
        base = model.split(":")[0]
        for m in [m for m in client.loaded_models() if m.split(":")[0] != base]:
            client.unload(m)
    run.item("start", "info", "Run started", f"classic · {model}")
    console.print(f"{brand.CMD} code (classic)  {root}  {model}")
    try:
        if body.files:
            out = repair.run(root, body.goal, list(body.files), config=cfg, llm=client, test_target=body.test, events=ev)
        else:
            out = build_mod.build(root, body.goal, config=cfg, llm=client, events=ev)
    finally:
        with contextlib.suppress(Exception):
            client.unload(model)
    run.result = {"ok": out.ok, "stage": out.stage, "attempts": out.attempts, "branch": out.branch, "error": (out.error or "")[:2000],
                  "diff": (out.diff or "")[:200000], "trail": list(getattr(out, "trail", []) or [])}
    run.exit_code = 0 if out.ok else 1
    run.outcome = "done" if out.ok else "partial"
    run.summary = (f"Built in {out.attempts} round(s)." if out.ok and out.stage == "built" else "The checks already pass." if out.stage == "already-passing"
                   else f"Fixed in {out.attempts} attempt(s)." if out.ok else f"{out.stage}: {(out.error or '')[:300]}")
    if out.branch:
        run.summary += f" Work is on branch {out.branch} (git merge {out.branch})."


def _thread_main(run: Run, body: StartIn, root: Path, config: dict[str, Any], client: Any) -> None:
    console = JobConsole(file=_Sink(run), force_terminal=False, color_system=None, width=140, highlight=False)
    try:
        (_exec_agent if run.engine == "agent" else _exec_classic)(run, body, root, config, client, console)
    except (KeyboardInterrupt, _Stopped):
        run.outcome, run.summary = "stopped", "Stopped. The work so far is kept."
    except Exception as exc:  # noqa: BLE001 - the job reports, the server lives on
        run.outcome, run.summary = "failed", f"{type(exc).__name__}: {str(exc)[:500]}"
        run.log(run.summary)
    finally:
        if run.stop_flag.is_set() and not run.outcome:
            run.outcome = "stopped"
        run.outcome = run.outcome or "done"
        run.ended = time.time()
        run.item("end", {"done": "ok", "stopped": "warn"}.get(run.outcome, "danger" if run.outcome == "failed" else "warn"), f"Run {run.outcome}", run.summary[:200])


# ---------------------------------------------------------------------------------------------------------------- routes


@router.post("/runs")
def start_run(body: StartIn, client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    if body.engine not in ("agent", "classic"):
        raise HTTPException(422, "engine must be agent or classic.")
    if not body.goal.strip() and not body.resume:
        raise HTTPException(422, "Write the goal first.")
    if not body.goal.strip():
        body.goal = "continue"
    if len(body.goal) > 20000:
        raise HTTPException(422, "The goal is too long (20000 characters at most).")
    if body.engine == "classic" and (body.autopilot or body.stages or body.resume):
        raise HTTPException(422, "Stages, autopilot and resume belong to the agent; the classic builder runs one goal.")
    root = resolve_folder(body.folder)
    body.files = check_files(root, body.files)
    check_access(root, body.mode, body.confirm_access)
    overlay = coding_overlay(body)
    with _LOCK:
        busy = _active()
        if busy is not None:
            raise HTTPException(409, f"Run {busy.id} is still active ({busy.folder}). Two runs would fight over the model; stop or wait for it.")
        run = Run(id=secrets.token_hex(4), engine=body.engine, folder=str(root), goal=body.goal.strip(),
                  opts=body.model_dump(exclude={"goal", "folder", "confirm_access"}, exclude_none=True))
        _RUNS[run.id] = run
        _prune_runs()
    if body.remember and body.mode:
        policy_mod.save_repo(root, mode=body.mode)
    run.thread = threading.Thread(target=_thread_main, args=(run, body, root, merge_coding(config, overlay), client), daemon=True, name=f"code-run-{run.id}")
    run.thread.start()
    return {"id": run.id, "state": run.state}


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    with _LOCK:
        runs = sorted(_RUNS.values(), key=lambda r: -r.started)
        return [{k: v for k, v in _public(r).items() if k not in ("log", "items", "opts", "result")} for r in runs]


@router.get("/runs/{run_id}")
def get_run(run_id: str, since: int = 0) -> dict[str, Any]:
    return _public(_get(run_id), since)


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, Any]:
    run = _get(run_id)
    if run.active:
        run.stop_flag.set()
        run.answered.set()          # a pending question is a no, and the job leaves its wait
        run.resume_flag.set()
    return {"state": run.state}


@router.post("/runs/{run_id}/pause")
def pause_run(run_id: str) -> dict[str, Any]:
    run = _get(run_id)
    if not run.active:
        raise HTTPException(409, "The run has ended.")
    run.resume_flag.clear()
    run.paused = True
    run.log("paused: waiting before the next model call")
    return {"state": run.state}


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: str) -> dict[str, Any]:
    run = _get(run_id)
    if run.active:
        run.paused = False
        run.resume_flag.set()
        run.log("resumed")
        return {"state": run.state, "continued": "paused run"}
    if run.engine != "agent" or not run.session_id:
        raise HTTPException(409, "This run has no saved session to resume.")
    return {"state": run.state, "resume_session": run.session_id}


class DecideIn(BaseModel):
    approve: bool
    confirm: str | None = None


@router.post("/runs/{run_id}/decide")
def decide(run_id: str, body: DecideIn) -> dict[str, Any]:
    run = _get(run_id)
    with _LOCK:
        ap = run.approval
        if ap is None:
            raise HTTPException(409, "Nothing is waiting for a decision.")
        if body.approve and ap["risk_level"] == "DANGEROUS" and body.confirm != "yes":
            raise HTTPException(422, "A dangerous command needs the typed word 'yes'.")
        run._answer = bool(body.approve)
    run.answered.set()
    return {"decided": body.approve}


@router.post("/runs/{run_id}/undo")
def undo_run(run_id: str) -> dict[str, Any]:
    run = _get(run_id)
    eng = run.obj if run.engine == "agent" else None
    if eng is None:
        raise HTTPException(409, "Undo is for agent runs.")
    if eng.in_turn:
        raise HTTPException(409, "The run is working; stop it first.")
    return {"lines": eng.undo.undo()}


@router.get("/runs/{run_id}/diff")
def run_diff(run_id: str) -> dict[str, str]:
    run = _get(run_id)
    if run.engine == "classic" and run.result.get("diff") is not None:
        return {"diff": run.result["diff"], "branch": run.result.get("branch", "")}
    return {"diff": git_diff(Path(run.folder)), "branch": ""}


def git_diff(root: Path, limit: int = 200_000) -> str:
    """Changes since HEAD, new (untracked) files included (read-only: the index is not touched)."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    out = git("diff", "HEAD")
    for rel in git("ls-files", "--others", "--exclude-standard").splitlines()[:30]:
        if len(out) < limit and (root / rel).is_file() and (root / rel).stat().st_size < 100_000:
            out += git("diff", "--no-index", "--", "/dev/null", rel).replace("b/" + rel, "b/" + rel)
    return out[:limit]


# ---------------------------------------------------------------------------------------------------------------- options + defaults


def effective_coding(config: dict[str, Any]) -> dict[str, Any]:
    c = coding_config.load(config)
    user = config.get("coding") or {}
    return {"effort": c["build"]["effort"], "diagnose_first": bool(user.get("diagnose_first", True)), "auto_commit": bool(c["auto_commit"]),
            "budget": c["budget"], "executor": {"model": c["executor"]["model"], "max_repair_attempts": c["executor"]["max_repair_attempts"]},
            "escalation": c["escalation"], "sandbox": c["sandbox"], "prose_model": str(user.get("prose_model") or ""),
            "protect": list(user.get("protect") or []), "stages": str(user.get("stages", "auto")), "autopilot": str(user.get("autopilot", "auto")),
            "stage_max_minutes": int(user.get("stage_max_minutes", 25)), "stage_max_steps": int(user.get("stage_max_steps", 180)),
            "stage_prompt_cap": int(user.get("stage_prompt_cap", 28000)), "keep_model_loaded": bool(user.get("keep_model_loaded", False))}


@router.get("/options")
def options(folder: str | None = None, client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    installed: list[str] = []
    with contextlib.suppress(Exception):
        installed = sorted(client.list_models())
    pol = None
    root = None
    if folder:
        with contextlib.suppress(HTTPException):
            root = resolve_folder(folder)
    if root is not None:
        p = policy_mod.for_repo(root)
        pol = dataclasses.asdict(p)
    default_agent_model = (pol or {}).get("model") or (config.get("code") or {}).get("model") or config["models"]["chat"]
    busy = _active()
    return {"coding": effective_coding(config), "agent_model": default_agent_model, "installed": installed,
            "recommended": [{"model": t, "note": n} for t, n in coding_models.RECOMMENDED], "modes": list(policy_mod.MODES), "efforts": list(EFFORTS),
            "policy": pol, "folder": str(root) if root else None, "busy": busy.id if busy else None}


class DefaultsIn(BaseModel):
    values: dict[str, Any]


_DEFAULT_KEYS = {
    "effort": ("build.effort", str), "diagnose_first": ("diagnose_first", bool), "auto_commit": ("auto_commit", bool),
    "budget_max_actions": ("budget.max_actions", int), "budget_max_seconds": ("budget.max_seconds", int), "budget_max_tokens": ("budget.max_tokens", int),
    "executor_model": ("executor.model", str), "max_repair_attempts": ("executor.max_repair_attempts", int), "prose_model": ("prose_model", str),
    "escalation_enabled": ("escalation.tier3_enabled", bool), "escalation_model": ("escalation.tier3_model", str),
    "escalation_budget_s": ("escalation.total_budget_seconds", int), "timeout_s": ("sandbox.timeout_seconds", int), "memory_mb": ("sandbox.memory_mb", int),
    "protect": ("protect", list), "stage_max_minutes": ("stage_max_minutes", int), "stage_max_steps": ("stage_max_steps", int),
    "stage_prompt_cap": ("stage_prompt_cap", int), "keep_model_loaded": ("keep_model_loaded", bool),
}


def write_coding_defaults(values: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Persist coding.* defaults to config.yaml (what `agent code-model` / the menu do) and to the running server's config."""
    import yaml

    from agent import config as config_mod

    path_file = config_mod.DEFAULT_CONFIG_PATH
    raw = (yaml.safe_load(path_file.read_text()) or {}) if path_file.exists() else {}
    changed = []
    file_cfg = raw
    for key, v in values.items():
        if key not in _DEFAULT_KEYS:
            raise HTTPException(422, f"Unknown default {key!r}.")
        path, kind = _DEFAULT_KEYS[key]
        if kind is bool:
            if not isinstance(v, bool):
                raise HTTPException(422, f"{key} must be true or false.")
        elif kind is int:
            _int(v, key, 0, 100_000_000)
        elif kind is list:
            v = clean_patterns(v)
        else:
            v = str(v).strip()
            if key == "effort" and v not in EFFORTS:
                raise HTTPException(422, "effort must be min, mid or max.")
        for target in (file_cfg, config):
            node = target.setdefault("coding", {})
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = v
        changed.append(key)
    path_file.write_text(yaml.dump(file_cfg, sort_keys=False))
    return changed


@router.put("/defaults")
def put_defaults(body: DefaultsIn, config=Depends(get_config)) -> dict[str, Any]:
    changed = write_coding_defaults(body.values, config)
    return {"changed": changed, "coding": effective_coding(config)}
