"""A canary before the night: one tiny task through the REAL chain (subprocess runner -> `nightage code --autopilot` -> model -> file written -> git), so that a night
does not start at 1 a.m. only to find out at 7 that the model never loaded. The doctor checks the parts; this checks that they work together.
Success = the attempt exited cleanly AND the file it was asked to write exists with the expected content. The canary leaves nothing behind (the folder is restored)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from agent.night import gitbox
from agent.night.runner import Attempt, Outcome

FILE = "night_canary.txt"
TASK = f"Create the file {FILE} in the project root containing exactly the word ok (nothing else), then stop. Do not change any other file."


def timeout_for(test_seconds: float, floor: float = 8 * 60, factor: float = 8.0, ceiling: float = 30 * 60) -> float:
    """The canary starts the whole coding agent, which runs the project's tests several times: on the game (54 s suite) it took 300 s, on a toy 30 s. So the allowance is
    `factor` times the measured test run, never under `floor` and never over `ceiling`."""
    return float(min(max(test_seconds * factor, floor), ceiling))


def run(root: Path, runner: Callable[[Attempt], Outcome], timeout_s: float = 8 * 60, clock: Callable[[], float] = time.monotonic) -> tuple[bool, str]:
    """(ok, detail). The attempt is allowed `timeout_s`; the canary file is removed afterwards whatever happened."""
    gitbox.ensure_repo(root)
    f = root / FILE
    f.unlink(missing_ok=True)
    t0 = clock()
    try:
        out = runner(Attempt(TASK, new=True, timeout_s=timeout_s, stall_s=timeout_s, idle_s=timeout_s, log=str(root / ".night" / "canary.log")))
        took = clock() - t0
        if out.why:
            return False, f"the canary attempt ended with {out.why} after {took:.0f} s (see .night/canary.log)"
        if not f.exists():
            return False, f"the canary attempt exited with {out.code} after {took:.0f} s but did not write {FILE} (see .night/canary.log)"
        if f.read_text(errors="replace").strip().lower() != "ok":
            return False, f"{FILE} has the wrong content: {f.read_text(errors='replace').strip()[:40]!r}"
        return True, f"the whole chain works ({took:.0f} s)"
    finally:
        f.unlink(missing_ok=True)
        gitbox.reset_to(root, "HEAD")
