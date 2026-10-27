#!/usr/bin/env python3
"""Size and stability of the FIXED PROMPT PREFIX of `agent code` (no model is called, nothing outside a temp folder is written).

Every stage of a staged run starts in a fresh context and the model server (llama.cpp / Ollama) re-reads the prompt from the first byte that
differs from the previous request. So two things matter: how big the part is that is sent first, and whether it stays byte-identical.

    python scripts/prefix_report.py                       # a generated sample repo
    python scripts/prefix_report.py --repo PATH           # your own repo (it is only read, plus a tiny probe file in a COPY)
    python scripts/prefix_report.py --task "build ..."    # the request the brief card is made from

It builds the context exactly as the stage runner does (CodeEngine.begin_stage with the measured VERIFIED STATE, brief card, stage notes) and
prints, sorted by size, the token estimate (characters / 3.2, context.CHARS_PER_TOKEN) of: the system prompt, the PROJECT block, every tool
schema, the brief card, the repo map, the project memory, the stage goal, the stage notes, dead ends and the verified state.
Then two stability checks:
  * build twice with the clock moved by two hours: a part that differs holds a timestamp or counter,
  * a stage restart after progress (a new file, a saved project note, a few steps): which parts differ from the previous stage's context and
    where the first changed message sits. Everything before it can be served from the server's prompt cache.
and a pattern scan of the early messages for step counters, clock times and dates.
Exit code 0 when no prefix-cache breaker was found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_TASK = ('Build a small command line inventory tool. It must "keep every item in a JSON file", it must "print a table sorted by name", '
               'and the code needs tests for every command. Do not use any third party package. Quantities are never negative. '
               'Item names are unique, compared without regard to case. Every error is printed as one line on stderr with exit code 2.')
SAMPLE_GOAL = "Stage 2: add the `report` command that prints the table, with tests (stock.py, test_stock.py)."
SAMPLE_NOTES = ("- stage 1 \"skeleton\" [done]: package, models, JSON store; 9 tests pass\n- later stages, NOT yours now: 3. export to CSV")
_EARLY_PATTERNS = [
    ("step counter", re.compile(r"(?i)\bstep[ s]*\d+\b|\bat step\b|\(step \d+\)")),
    ("clock time", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")),
    ("date", re.compile(r"\b20\d\d-\d\d-\d\d\b")),
    ("seconds measured", re.compile(r"\b\d+(?:\.\d+)?s\b\)?")),
]


@dataclass
class Part:
    name: str
    text: str
    index: int | None = None         # position in the message list (None: a tool schema, sent next to it)
    role: str = ""

    @property
    def tokens(self) -> int:
        from agent.codeagent.context import CHARS_PER_TOKEN

        return int(len(self.text) / CHARS_PER_TOKEN)


@dataclass
class Report:
    parts: list[Part] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    breakers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ── a throw-away engine ─────────────────────────────────────────────────────

@contextmanager
def isolated_state():
    """Policy, events and state of the report live in a temp folder, never in ~/.config or ~/.local/state."""
    tmp = Path(tempfile.mkdtemp(prefix="prefix-report-"))
    keys = ("AGENT_CODE_POLICY", "AGENT_CODE_EVENTS", "XDG_STATE_HOME", "XDG_CONFIG_HOME")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({"AGENT_CODE_POLICY": str(tmp / "policy.json"), "AGENT_CODE_EVENTS": str(tmp / "events.jsonl"),
                       "XDG_STATE_HOME": str(tmp / "state"), "XDG_CONFIG_HOME": str(tmp / "config")})
    try:
        yield tmp
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def make_sample_repo(folder: Path) -> Path:
    """A small, deterministic Python project (package, tests, README, pyproject)."""
    (folder / "inventory").mkdir(parents=True, exist_ok=True)
    (folder / "tests").mkdir(exist_ok=True)
    (folder / "pyproject.toml").write_text('[project]\nname = "inventory"\nversion = "0.1.0"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (folder / "README.md").write_text("# inventory\n\nA tiny inventory tool.\n")
    (folder / "inventory" / "__init__.py").write_text('"""Inventory."""\n')
    for n, name in enumerate(("models", "store", "cli", "format", "search", "errors")):
        body = [f'"""{name} module."""', "", "from __future__ import annotations", ""]
        for i in range(4):
            body += [f"class {name.title()}{i}:", f'    """Part {i} of {name}."""', "", f"    def run(self, value: int = {n}) -> int:",
                     f"        return value + {i}", ""]
        (folder / "inventory" / f"{name}.py").write_text("\n".join(body))
        (folder / "tests" / f"test_{name}.py").write_text(
            f"from inventory.{name} import {name.title()}0\n\n\ndef test_{name}():\n    assert {name.title()}0().run() == {n}\n")
    return folder


def make_engine(root: Path, notes: Any = None, key: str | None = None):
    from rich.console import Console

    from agent.codeagent.engine import CodeEngine
    from agent.codeagent.eventlog import EventLog
    from agent.codeagent.session import _NoModel

    config = {"models": {"chat": "m", "background": "m", "chat_num_ctx": 32768}, "coding": {"claims_check": False, "verified_tests": False}, "autonomy": {}}
    console = Console(file=io.StringIO(), width=120, force_terminal=False)
    return CodeEngine(root, config, _NoModel(), console, interactive=False, events=EventLog("prefix-report", root.parent / "prefix-report-events.jsonl"),
                      notes=notes, note_key=key)


def stage_context(engine, task: str, goal: str, stage_notes: str) -> list[dict[str, Any]]:
    """The messages of a stage's first request, built the way stages._Runner.run_stage does (the user turn is the goal, no model call)."""
    from agent.codeagent import brief as brief_mod, verified

    card = brief_mod.make(task)
    engine.briefs = [card] if card else []
    state = verified.build(engine, run_tests=False)
    engine.begin_stage(goal, state.text, stage_notes, reset_plan=True)
    return [dict(m) for m in engine.ctx.messages]


# ── naming and measuring ────────────────────────────────────────────────────

def name_of(message: dict[str, Any]) -> str:
    text = str(message.get("content", ""))
    if message.get("role") == "user":
        return "verified state" if text.startswith("VERIFIED STATE") else "user turn"
    for prefix, name in (("REPO MAP", "repo map"), ("PROJECT MEMORY", "project memory"), ("BRIEF CARD", "brief card"), ("STAGE GOAL", "stage goal"),
                         ("EARLIER STAGES", "stage notes"), ("DEAD ENDS", "dead ends"), ("PROJECT NOTES SAVED", "session notes")):
        if text.startswith(prefix):
            return name
    return "system prompt"


def parts_of(messages: list[dict[str, Any]], schema: list[dict[str, Any]]) -> list[Part]:
    out: list[Part] = []
    for i, m in enumerate(messages):
        text = str(m.get("content", ""))
        name = name_of(m)
        if name == "system prompt" and "\n\nPROJECT " in text:                 # SYSTEM + the PROJECT block share one message
            head, tail = text.split("\n\nPROJECT ", 1)
            out += [Part("system prompt", head, i, m["role"]), Part("PROJECT block", "PROJECT " + tail, i, m["role"])]
        else:
            out.append(Part(name, text, i, m["role"]))
    for t in schema:
        fn = t["function"]
        out.append(Part(f"tool {fn['name']}", json.dumps(t, ensure_ascii=False, sort_keys=True), None, "tool"))
    return out


# ── stability checks ────────────────────────────────────────────────────────

@contextmanager
def clock_shifted(seconds: float):
    real_time, real_mono = time.time, time.monotonic
    time.time, time.monotonic = (lambda: real_time() + seconds), (lambda: real_mono() + seconds)
    try:
        yield
    finally:
        time.time, time.monotonic = real_time, real_mono


def diff_messages(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """(index in b, part name) of every message of b that is not byte-identical to the same position in a."""
    out = []
    for i, m in enumerate(b):
        if i >= len(a) or json.dumps(a[i], sort_keys=True) != json.dumps(m, sort_keys=True):
            out.append((i, name_of(m)))
    return out


def check_volatility(messages: list[dict[str, Any]], repo: Path, task: str, goal: str, notes: str, memory: Any = None, key: str | None = None) -> list[str]:
    """Build the same stage twice with the clock two hours apart: any difference is a timestamp, counter or random order."""
    with clock_shifted(7200):
        again = stage_context(make_engine(repo, memory, key), task, goal, notes)
    return [f"{name} (message {i}) differs between two builds of the same stage: it holds a timestamp, counter or unstable order"
            for i, name in diff_messages(messages, again)]


def check_restart(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Which parts a stage restart after progress changes, and which of them sit in front of the stage goal (the cacheable head)."""
    changed = diff_messages(first, second)
    goal_at = next((i for i, m in enumerate(second) if name_of(m) == "stage goal"), len(second))
    early = [(i, n) for i, n in changed if i < goal_at]
    warn = [f"{n} (message {i}) changes when a stage restarts after progress and sits before the stage goal: everything after it is re-read" for i, n in early]
    first_at = changed[0][0] if changed else len(second)
    return warn, f"first changed message on a stage restart: #{first_at} of {len(second)} ({next((n for i, n in changed), 'none')}); messages 0-{first_at - 1} stay cached"


def scan_early(messages: list[dict[str, Any]]) -> list[str]:
    """Step counters, clock times and dates inside the messages in front of the stage goal (the verified state at the end may have them)."""
    goal_at = next((i for i, m in enumerate(messages) if name_of(m) == "stage goal"), len(messages))
    out = []
    for i, m in enumerate(messages[:goal_at]):
        for label, rx in _EARLY_PATTERNS:
            hit = rx.search(str(m.get("content", "")))
            if hit:
                out.append(f"{name_of(m)} (message {i}) contains a {label}: {hit.group(0)!r}")
    return out


def build_report(repo: Path, task: str = SAMPLE_TASK, goal: str = SAMPLE_GOAL, stage_notes: str = SAMPLE_NOTES) -> Report:
    """Everything the CLI prints, as data. `repo` must be a throw-away folder: the restart simulation adds a file to it."""
    from agent import projectmem

    memory = projectmem.ProjectMemory(sqlite3.connect(":memory:"))
    key = str(repo.resolve())
    memory.add(key, "tests run with pytest -q from the repo root")
    engine = make_engine(repo, memory, key)
    engine.start()
    messages = stage_context(engine, task, goal, stage_notes)
    rep = Report(messages=messages)
    rep.parts = sorted(parts_of(messages, engine.schema), key=lambda p: -p.tokens)
    rep.breakers += check_volatility(messages, repo, task, goal, stage_notes, memory, key)
    rep.breakers += scan_early(messages)
    # progress, then the next stage of the same session: a new file, a saved project note, a few steps and a recorded session note
    (repo / "inventory" / "report.py").write_text('"""report module."""\n\n\ndef table(rows):\n    return "\\n".join(rows)\n')
    memory.add(key, "the store keeps items in inventory.json, sorted by name")
    engine.session_notes.append("the store keeps items in inventory.json, sorted by name")
    engine.ctx.step += 7
    later = stage_context(engine, task, "Stage 3: export to CSV.", stage_notes + "\n- stage 2 \"report\" [done]: report.py, 11 tests pass")
    warn, summary = check_restart(messages, later)
    rep.breakers += warn
    rep.notes.append(summary)
    return rep


def render(rep: Report, top: int = 15) -> str:
    total = sum(p.tokens for p in rep.parts)
    fixed = sum(p.tokens for p in rep.parts if p.role in ("system", "tool"))
    lines = [f"FIXED PROMPT PREFIX  (estimate: characters / 3.2; total {total} tokens, of which system messages + tool schemas {fixed})",
             f"{'tokens':>7}  {'chars':>6}  part"]
    for p in rep.parts[:top]:
        lines.append(f"{p.tokens:>7}  {len(p.text):>6}  {p.name}" + (f"  (message {p.index})" if p.index is not None else ""))
    if len(rep.parts) > top:
        rest = rep.parts[top:]
        lines.append(f"{sum(p.tokens for p in rest):>7}  {sum(len(p.text) for p in rest):>6}  {len(rest)} smaller parts")
    schema = [p for p in rep.parts if p.name.startswith("tool ")]
    lines.append(f"tool schemas together: {sum(p.tokens for p in schema)} tokens in {len(schema)} tools")
    lines += ["", *rep.notes, "", "PREFIX-CACHE BREAKERS:" if rep.breakers else "PREFIX-CACHE BREAKERS: none found"]
    lines += [f"  ! {b}" for b in rep.breakers]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", help="a repo to measure (it is COPIED to a temp folder first; nothing is written into it)")
    ap.add_argument("--task", default=SAMPLE_TASK, help="the request the brief card is made from")
    ap.add_argument("--top", type=int, default=15, help="rows to show")
    args = ap.parse_args(argv)
    with isolated_state() as tmp:
        repo = tmp / "repo"
        if args.repo:
            src = Path(args.repo).expanduser().resolve()
            shutil.copytree(src, repo, ignore=shutil.ignore_patterns(".git", ".venv", "node_modules", "__pycache__", ".scratch"))
        else:
            make_sample_repo(repo)
        rep = build_report(repo, args.task)
        print(render(rep, args.top))
    return 1 if rep.breakers else 0


if __name__ == "__main__":
    sys.exit(main())
