"""Facts about the project measured before the night: does its test command work at all, how long does it take, how many tests are there?
A night built on a test command that cannot run (exit 127, 'no tests ran', a missing module) would call every attempt red and roll everything
back, so this is checked at the start and written to .night/FACTS.md for the model as well."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.night.testwatch import TestState

UNUSABLE = ("command not found", "no such file or directory", "modulenotfounderror", "no module named", "no tests ran", "collected 0 items", "cannot find module", "error: no such command")


@dataclass
class Facts:
    command: str
    seconds: float
    state: TestState | None
    problem: str = ""

    @property
    def usable(self) -> bool:
        return not self.problem

    def text(self) -> str:
        if self.state is None:
            return "tests: none detected for this project (the harness cannot verify anything by tests)"
        status = "all green" if self.state.ok else f"{len(self.state.failed)} failing at the start"
        return f"tests: `{self.command}` takes {self.seconds:.0f} s, {self.state.passed} passing, {status}" + (f"; PROBLEM: {self.problem}" if self.problem else "")


def probe(command: str | None, run: Callable[[], TestState]) -> Facts:
    if not command:
        return Facts("", 0.0, None)
    t0 = time.monotonic()
    state = run()
    took = time.monotonic() - t0
    low = state.tail.lower()
    if state.code in (127, 126) or (not state.ok and not state.failed and any(s in low for s in UNUSABLE)):
        return Facts(command, took, state, "the test command cannot run: " + state.digest[-200:].replace("\n", " "))
    if state.ok and state.passed == 0 and "no test" in low:
        return Facts(command, took, state, "the test command ran no tests")
    return Facts(command, took, state)


def write(root: Path, facts: Facts) -> None:
    nd = root / ".night"
    nd.mkdir(exist_ok=True)
    (nd / "FACTS.md").write_text("# Facts measured by the harness\n\n" + facts.text() + "\n")


def test_timeout(facts: Facts, floor: float = 120.0, ceiling: float = 1800.0, factor: float = 4.0) -> float:
    """How long a test run may take before it counts as hanging: `factor` times what the probe measured, never less than `floor`, never more than `ceiling`."""
    return float(min(max(facts.seconds * factor, floor), ceiling)) if facts.state is not None else ceiling
