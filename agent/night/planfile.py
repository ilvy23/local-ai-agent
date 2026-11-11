"""The plan file: a checklist the model writes BEFORE it codes and the harness ticks AFTER every attempt. A small model loses its thread when a context ends;
a plan on disk with steps that can be checked gives the next fresh context its direction back (architect/editor idea, with measured ticks instead of claimed ones).

    .night/plans/<prompt>.md
    - [ ] 1. Add the Wren class to characters.py | check: grep -q "class Wren" characters.py
    - [ ] 2. A test loads her | check: file tests/test_wren.py
    - [ ] 3. Her lines react to the ledger | check: test tests/test_wren.py::test_ledger

`check:` is one of `file PATH` (exists), `test NODEID` (that test does not fail in the last test run) or a shell command (exit 0). The model never ticks a step itself:
`evaluate()` runs the checks and rewrites the boxes. A step without a check is shown but can only be ticked by the model writing `[x]` (kept: it is advice, not proof)."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.night.testwatch import TestState

_STEP = re.compile(r"^(\s*-\s*)\[( |x|X)\]\s*(.+?)(?:\s*\|\s*check:\s*(.+?))?\s*$")
PLAN_PROMPT = ("Before any coding, write a plan for this task as a checklist in {path}. 4 to 10 steps, each on one line in exactly this form:\n"
               "- [ ] 1. what to do | check: how the harness can verify it\n"
               "A check is `file PATH` (the file exists), `test TESTID` (that test passes) or a short shell command that exits 0 when the step is done. "
               "Every step needs a check. Do not change any other file in this step. Then stop.")


@dataclass
class Step:
    text: str
    done: bool
    check: str = ""


def path(root: Path, name: str) -> Path:
    return root / ".night" / "plans" / f"{''.join(c if c.isalnum() else '-' for c in name).strip('-').lower() or 'prompt'}.md"


def parse(text: str) -> list[Step]:
    steps = []
    for line in text.splitlines():
        m = _STEP.match(line)
        if m:
            steps.append(Step(m.group(3).strip(), m.group(2) in "xX", (m.group(4) or "").strip()))
    return steps


def render(steps: list[Step], title: str = "") -> str:
    head = "# Plan" + (": " + title if title else "") + "\n\n"
    lines = []
    for i, s in enumerate(steps, 1):
        text = re.sub(r"^\d+\.\s*", "", s.text)
        lines.append(f"- [{'x' if s.done else ' '}] {i}. {text}" + (f" | check: {s.check}" if s.check else ""))
    return head + "\n".join(lines) + "\n"


def usable(steps: list[Step]) -> bool:
    """A plan worth following: at least 3 steps and at least half of them checkable."""
    return len(steps) >= 3 and sum(1 for s in steps if s.check) * 2 >= len(steps)


def _sh(cmd: str, cwd: Path) -> bool:
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, timeout=300).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def check(step: Step, root: Path, state: TestState | None, sh: Callable[[str, Path], bool] = _sh) -> bool:
    c = step.check
    if not c:
        return step.done
    kind, _, arg = c.partition(" ")
    arg = arg.strip()
    if kind == "file" and arg:
        return not arg.startswith(("/", "..")) and (root / arg).exists()
    if kind == "test" and arg:
        return state is not None and state.ok is not None and arg not in state.failed and (state.ok or bool(state.failed))
    return sh(c, root)


def evaluate(root: Path, name: str, state: TestState | None, sh: Callable[[str, Path], bool] = _sh) -> tuple[int, int]:
    """Run the checks, rewrite the boxes (ticked ones stay ticked only while their check still passes). Returns (steps verified, steps)."""
    p = path(root, name)
    try:
        steps = parse(p.read_text())
    except OSError:
        return 0, 0
    for s in steps:
        s.done = check(s, root, state, sh)
    p.write_text(render(steps, name))
    return sum(1 for s in steps if s.done), len(steps)


def remaining(root: Path, name: str, limit: int = 4) -> str:
    try:
        steps = parse(path(root, name).read_text())
    except OSError:
        return ""
    open_ = [s.text for s in steps if not s.done]
    return "; ".join(open_[:limit]) + (f" (+{len(open_) - limit} more)" if len(open_) > limit else "")
