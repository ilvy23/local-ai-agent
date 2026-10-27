"""Checks as first-class objects: what must be TRUE, verified by the harness, never by the author's word.

Seen live (D3): the model reported "all 4 tiers unique" while 35 sentences were shared, invented death names in its
report and accepted "mostly DONE" for a character that missed the brief - its own tests were too lax. An independent
check script written by the supervisor caught all of it at once. So:

    Check = {id, description, kind, spec, source, items, last_result}
      command   spec {"command": "...", "timeout": 120}            pass = exit 0
      python    spec {"path": "checks/x.py", "timeout": 120}        a script run in a subprocess, pass = exit 0, stdout shown
      banned    spec {"patterns": [...], "regex": true, "files": ["*.py"], "objects": ["lena:LENA_OPEN"]}
                                                                    pass = none of the patterns occurs in those files / objects
      rubric    spec {"criterion": "...", "scenario": ["cmd", ...]} judged by the reviewer (reviewer.py) on the output the
                                                                    harness gets by running the scenario commands
      tui       spec {"command": "python3 game.py", "keys": ["Right", "q"], "expect": ["SCORE", "re:^Level \\d"], "size": "80x24",
                "exit_code": 0, "timeout": 10}                      the program runs on a real pty (tuismoke.py, the `tui_smoke` tool); pass =
                                                                    every expect seen, no crash / timeout, exit code 0 (or still running;
                                                                    exit_code=N requires the program to have exited with N)

Where checks come from:
  * the model: the `add_check` tool (saved with the session; it may replace its own checks),
  * the user / a supervisor: files dropped into a checks folder, picked up on every run, READ-ONLY for the model
    (it cannot replace, edit or remove them - they live outside the repo, and add_check refuses their ids):
        ~/.local/state/agent/code-sessions/<folder key>/checks/          every session of that project folder
        ~/.local/state/agent/code-sessions/<folder key>/<id>.checks/     one session
    (`agent code checks` prints both paths; `agent code checks --add FILE` copies a file in.)  File kinds:
        NAME.py   a python check (pass = exit 0)      NAME.sh   a command check (`bash NAME.sh`)
        NAME.json one check object or a list of them ({"id", "description", "kind", "spec", "items"})
Scripts run with the project folder as working directory and on PYTHONPATH, so `import lena` works, and with the
agent's own interpreter, so `from agent.codeagent import textcheck` works (sentence-level duplicates, banned phrases).

How they are used (engine.py): `mark_done` for an acceptance item is accepted only when the item's checks pass (the harness
runs them and ignores the claimed evidence); the final gate runs ALL checks and sends the model back with the short
failing output - at most CHECK_RETURNS times per check, after which the answer is delivered, labelled "NOT VERIFIED"."""

from __future__ import annotations

import hashlib
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.codeagent import tuismoke

KINDS = ("command", "python", "banned", "rubric", "tui")
CHECK_RETURNS = 3                 # final-gate returns per failing check, then the answer goes out labelled NOT VERIFIED
DEFAULT_TIMEOUT = 120
OUT_HEAD, OUT_TAIL = 500, 1200    # what the model sees of a failing check
SCENARIO_TIMEOUT = 120
PASSING = ("pass",)
FIXABLE = ("fail", "error")       # statuses that send the model back; "skipped" (policy) and "unreviewed" only label the answer

_OBJECTS_SNIPPET = r"""
import importlib, json, sys
sys.path.insert(0, ".")
def flat(x):
    if isinstance(x, dict): return [t for v in x.values() for t in flat(v)]
    if isinstance(x, (list, tuple, set)): return [t for v in x for t in flat(v)]
    return [x] if isinstance(x, str) else []
out = {}
for spec in sys.argv[1:]:
    mod, _, attr = spec.partition(":")
    obj = importlib.import_module(mod)
    for part in [p for p in attr.split(".") if p]:
        obj = getattr(obj, part)
    out[spec] = flat(obj() if callable(obj) and not isinstance(obj, type) else obj)
print(json.dumps(out))
"""


def _clip(text: str, head: int = OUT_HEAD, tail: int = OUT_TAIL) -> str:
    text = str(text or "").strip()
    return text if len(text) <= head + tail + 40 else f"{text[:head]}\n[… {len(text) - head - tail} chars …]\n{text[-tail:]}"


@dataclass
class Check:
    id: str
    description: str
    kind: str
    spec: dict[str, Any]
    source: str = "model"                      # "model" | "supervisor"
    items: list[int] = field(default_factory=list)
    last_result: dict[str, Any] | None = None
    origin: str = ""                           # the file a supervisor check came from

    @property
    def readonly(self) -> bool:
        return self.source == "supervisor"

    def status(self) -> str:
        return (self.last_result or {}).get("status", "not run")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "description": self.description, "kind": self.kind, "spec": self.spec, "source": self.source,
                "items": list(self.items), "last_result": self.last_result, "origin": self.origin}

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str | None = None) -> "Check":
        items = d.get("items") or []
        return cls(id=str(d["id"]), description=str(d.get("description") or d["id"]), kind=str(d.get("kind") or "command"),
                   spec=dict(d.get("spec") or {}), source=source or str(d.get("source") or "model"),
                   items=[int(i) for i in (items if isinstance(items, list) else [items]) if str(i).strip().lstrip("-").isdigit()],
                   last_result=d.get("last_result"), origin=str(d.get("origin") or ""))


def validate(kind: str, spec: dict[str, Any]) -> str | None:
    """None when the spec is usable, else what is missing."""
    if kind not in KINDS:
        return f"kind must be one of {', '.join(KINDS)}"
    if kind == "command" and not str(spec.get("command") or "").strip():
        return "a command check needs command"
    if kind == "python" and not str(spec.get("path") or "").strip():
        return "a python check needs path (a script; pass = exit code 0)"
    if kind == "banned":
        if not spec.get("patterns"):
            return "a banned check needs patterns (a list of phrases or regexes)"
        if not (spec.get("files") or spec.get("objects")):
            return "a banned check needs files (globs) and/or objects (\"module:NAME\")"
    if kind == "tui":
        if not str(spec.get("command") or "").strip():
            return "a tui check needs command (the terminal program), and keys / expect"
        if not (spec.get("expect") or spec.get("exit_code") is not None):
            return "a tui check needs expect (texts the screen must show) and/or exit_code"
    if kind == "rubric":
        if not str(spec.get("criterion") or "").strip():
            return "a rubric check needs criterion (one sentence the reviewer judges)"
        if not spec.get("scenario"):
            return "a rubric check needs scenario (commands whose output shows the criterion)"
    return None


# ── where supervisor checks live ────────────────────────────────────────────

def checks_dirs(root: Path | str, session_id: str = "") -> list[Path]:
    from agent.codeagent import sessionstore

    base = sessionstore.folder_dir(Path(root).resolve())
    return [base / "checks"] + ([base / f"{session_id}.checks"] if session_id else [])


def load_supervisor(root: Path | str, session_id: str = "") -> tuple[list[Check], list[str]]:
    """Supervisor-provided checks (read-only for the model), and warnings for files that could not be used."""
    out: list[Check] = []
    warn: list[str] = []
    for d in checks_dirs(root, session_id):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            try:
                if f.suffix == ".py":
                    out.append(Check(f.stem, f"supervisor check {f.name}", "python", {"path": str(f)}, "supervisor", origin=str(f)))
                elif f.suffix == ".sh":
                    out.append(Check(f.stem, f"supervisor check {f.name}", "command", {"command": f"bash {shlex.quote(str(f))}"}, "supervisor", origin=str(f)))
                elif f.suffix == ".json":
                    data = json.loads(f.read_text(encoding="utf-8"))
                    for d0 in (data if isinstance(data, list) else [data]):
                        c = Check.from_dict(d0, "supervisor")
                        c.origin = str(f)
                        why = validate(c.kind, c.spec)
                        if why:
                            warn.append(f"{f.name}: check {c.id}: {why}")
                            continue
                        out.append(c)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warn.append(f"{f.name}: {exc}")
    return out, warn


# ── running ─────────────────────────────────────────────────────────────────

@dataclass
class Runner:
    """What running a check needs from the engine (or the supervisor): exec(command, timeout) -> (exit or None, output,
    seconds) in the project folder; allow(command) -> bool for model-written commands (the repo policy); review(criterion,
    output) -> reviewer result (None: no reviewer available)."""
    root: Path
    exec: Callable[[str, int], tuple[int | None, str, float]]
    allow: Callable[[str], bool] = lambda _c: True
    review: Callable[[str, str], dict[str, Any]] | None = None
    tui: Callable[[dict[str, Any]], Any] | None = None          # spec -> tuismoke.SmokeResult (the engine: CodeTools.tui_check, with the protect snapshot)


def _python_cmd(root: Path, script: str) -> str:
    import os

    agent_src = str(Path(__file__).resolve().parents[2])           # the folder that holds the `agent` package
    pp = os.pathsep.join([str(root), agent_src])
    return f"env PYTHONPATH={shlex.quote(pp)} {shlex.quote(sys.executable)} {shlex.quote(script)}"


def run(check: Check, runner: Runner) -> dict[str, Any]:
    """Run one check; returns (and stores in check.last_result) {status, output, seconds, ts}."""
    t0 = time.monotonic()
    try:
        status, output = _run(check, runner)
    except Exception as exc:  # noqa: BLE001 - a broken check is a failed check, never a crashed run
        status, output = "error", f"{type(exc).__name__}: {exc}"
    res = {"status": status, "output": _clip(output), "seconds": round(time.monotonic() - t0, 2), "ts": round(time.time(), 1)}
    check.last_result = res
    return res


def _run(check: Check, runner: Runner) -> tuple[str, str]:
    spec, root = check.spec, Path(runner.root)
    timeout = int(spec.get("timeout") or DEFAULT_TIMEOUT)
    if check.kind == "command":
        cmd = str(spec["command"])
        if not check.readonly and not runner.allow(cmd):
            return "skipped", f"not run: the repo policy does not allow `{cmd[:120]}` without asking"
        code, out, secs = runner.exec(cmd, timeout)
        return ("pass" if code == 0 else "fail"), f"$ {cmd}\nexit {code if code is not None else 'timeout'} ({secs:.1f}s)\n{out}"
    if check.kind == "python":
        path = Path(str(spec["path"]))
        p = path if path.is_absolute() else root / path
        if not p.is_file():
            return "error", f"the check script {spec['path']} does not exist"
        code, out, secs = runner.exec(_python_cmd(root, str(p)), timeout)
        return ("pass" if code == 0 else "fail"), f"$ python {spec['path']}\nexit {code if code is not None else 'timeout'} ({secs:.1f}s)\n{out}"
    if check.kind == "banned":
        return _banned(check, runner)
    if check.kind == "tui":
        cmd = str(spec["command"])
        if not check.readonly and not runner.allow(cmd):
            return "skipped", f"not run: the repo policy does not allow `{cmd[:120]}` without asking"
        if runner.tui is None:
            return "error", "no terminal runner available for a tui check"
        res = runner.tui(spec)
        want = spec.get("exit_code")
        ok = res.passed(int(want) if str(want).lstrip("-").isdigit() else None)
        return ("pass" if ok else "fail"), tuismoke.format_result(res, 1800)
    if check.kind == "rubric":
        if runner.review is None:
            return "unreviewed", "no reviewer available"
        scen = spec.get("scenario")
        cmds = [scen] if isinstance(scen, str) else [str(c) for c in scen or []]
        parts = []
        for cmd in cmds[:6]:
            if not check.readonly and not runner.allow(cmd):
                parts.append(f"$ {cmd}\n(not run: the repo policy does not allow it)")
                continue
            code, out, _s = runner.exec(cmd, int(spec.get("timeout") or SCENARIO_TIMEOUT))
            parts.append(f"$ {cmd}\n{out.strip()}" + (f"\n(exit {code})" if code else ""))
        from agent.codeagent import reviewer

        result = runner.review(str(spec["criterion"]), "\n\n".join(parts))
        return str(result.get("status") or "unreviewed"), reviewer.summary(result)
    return "error", f"unknown kind {check.kind}"


def _banned(check: Check, runner: Runner) -> tuple[str, str]:
    from agent.codeagent import textcheck

    spec, root = check.spec, Path(runner.root)
    regex = spec.get("regex", True) is not False
    pats = [str(p) for p in spec.get("patterns") or []]
    texts: dict[str, str] = {}
    for g in spec.get("files") or []:
        for f in sorted(root.glob(str(g))):
            if f.is_file() and ".scratch" not in f.parts:
                texts[str(f.relative_to(root))] = f.read_text(encoding="utf-8", errors="replace")
    objs = [str(o) for o in spec.get("objects") or []]
    if objs:
        cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(_OBJECTS_SNIPPET)} " + " ".join(shlex.quote(o) for o in objs)
        code, out, _s = runner.exec(f"env PYTHONPATH={shlex.quote(str(root))} {cmd}", 60)
        if code != 0:
            return "error", f"could not load {', '.join(objs)}:\n{out}"
        try:
            data = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return "error", f"could not read the strings of {', '.join(objs)}:\n{out}"
        texts.update({k: "\n".join(v) for k, v in data.items()})
    if not texts:
        return "error", "no files or objects matched - nothing was checked"
    found = []
    for name, text in texts.items():
        for pat, n in textcheck.banned_hits(text, pats, regex):
            found.append(f"/{pat}/ x{n} in {name}")
    return ("fail", "banned phrases found:\n" + "\n".join(found[:30])) if found else ("pass", f"none of {len(pats)} patterns in {len(texts)} text(s)")


# ── the set of a session ────────────────────────────────────────────────────

def content_stamp(root: Path, paths: Any) -> str:
    """A hash over the sizes and mtimes on disk of the given project files (a rubric verdict is fresh only while it is unchanged)."""
    h = hashlib.sha1()
    for p in sorted(paths):
        f = Path(p) if Path(p).is_absolute() else Path(root) / p
        try:
            st = f.stat()
            h.update(f"{p}\0{st.st_size}\0{st.st_mtime_ns}\n".encode("utf-8", errors="replace"))
        except OSError:
            h.update(f"{p}\0-1\n".encode("utf-8", errors="replace"))
    return h.hexdigest()


class CheckSet:
    """The model's checks (saved with the session) plus the supervisor's (re-read from the checks folders every time)."""

    def __init__(self, root: Path, session_id: Callable[[], str]) -> None:
        self.root = Path(root)
        self.session_id = session_id
        self.model: dict[str, Check] = {}
        self._supervisor_results: dict[str, dict[str, Any]] = {}      # last results of read-only checks (they are re-read from disk)
        self.warnings: list[str] = []

    def supervisor(self) -> list[Check]:
        checks, self.warnings = load_supervisor(self.root, self.session_id())
        for c in checks:
            c.last_result = self._supervisor_results.get(c.id)
        return checks

    def all(self) -> list[Check]:
        sup = self.supervisor()
        ids = {c.id for c in sup}
        return sup + [c for c in self.model.values() if c.id not in ids]

    def get(self, cid: str) -> Check | None:
        return next((c for c in self.all() if c.id == cid), None)

    def add(self, check: Check) -> str:
        """'' when added (or the model's own check replaced), else why not."""
        if any(c.id == check.id for c in self.supervisor()):
            return f"'{check.id}' is a supervisor check: it is read-only (you cannot replace, change or remove it). Pick another id."
        why = validate(check.kind, check.spec)
        if why:
            return why
        check.source = "model"
        self.model[check.id] = check
        return ""

    def remember(self, check: Check) -> None:
        if check.readonly and check.last_result is not None:
            self._supervisor_results[check.id] = check.last_result

    def for_item(self, item_id: int) -> list[Check]:
        return [c for c in self.all() if item_id in c.items]

    def to_json(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.model.values()]

    def load_json(self, data: Any) -> None:
        self.model = {}
        for d in data or []:
            try:
                c = Check.from_dict(d, "model")
            except (KeyError, TypeError, ValueError):
                continue
            self.model[c.id] = c


def failure_block(failing: list[Check], brief: str = "") -> str:
    """The message the model gets when checks fail at the final gate: short output of each, and the brief re-shown."""
    parts = ["CHECKS FAILED - run by the harness just now; the task is not done until they pass (your own evidence does not count):"]
    for c in failing:
        r = c.last_result or {}
        parts.append(f"- {c.id} ({c.kind}{', supervisor - read-only' if c.readonly else ''}): {c.description}\n"
                     + "\n".join("    " + ln for ln in _clip(r.get("output", ""), 300, 900).splitlines()))
    parts.append("Fix the code or the texts so these pass (do not weaken or rewrite a check to make it pass), then give the final answer.")
    if brief:
        parts.append(brief)
    return "\n".join(parts)


def summary_lines(checks: list[Check]) -> list[str]:
    return [f"{c.id}: {c.status().upper()} ({c.kind}{', supervisor' if c.readonly else ''}) {c.description[:80]}" for c in checks]
