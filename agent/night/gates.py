"""More than tests: the project's own build and lint commands as extra gates (profile.py finds them). A failing BUILD means the attempt is not green even if the
tests pass (a project that no longer builds is not finished); a failing LINT is advice that goes into the handoff and the report, never a block. They only run when
the tests are green (no point building a state that is already red)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from agent.night.testwatch import TestState


@dataclass
class Gate:
    name: str
    cmd: str
    required: bool


@dataclass
class GateResult:
    gate: Gate
    ok: bool
    tail: str


def from_profile(profile) -> list[Gate]:
    gates = []
    if getattr(profile, "build_cmd", None):
        gates.append(Gate("build", profile.build_cmd, True))
    if getattr(profile, "lint_cmd", None):
        gates.append(Gate("lint", profile.lint_cmd, False))
    return gates


def _sh(cmd: str, cwd: Path, timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as e:
        return 127, str(e)
    return p.returncode, (p.stdout + p.stderr)[-800:]


def run(root: Path, gates: list[Gate], runner: Callable[[str, Path, int], tuple[int, str]] = _sh, timeout: int = 900) -> list[GateResult]:
    out = []
    for g in gates:
        code, tail = runner(g.cmd, root, timeout)
        out.append(GateResult(g, code == 0, tail))
    return out


def apply(state: TestState, results: list[GateResult]) -> TestState:
    """`state` with the gates folded in: a failed required gate turns it red and says why; a failed advisory gate only adds a line."""
    if not results:
        return state
    ok, digest = state.ok, state.digest
    for r in results:
        if r.ok:
            continue
        last = " / ".join(ln for ln in r.tail.strip().splitlines() if ln.strip())[-300:]
        if r.gate.required:
            ok = False
            digest = f"{r.gate.name} FAILED (`{r.gate.cmd}`): {last}" + (f"\n{digest}" if digest else "")
        else:
            digest = (digest + "\n" if digest else "") + f"{r.gate.name} reports problems (advice only): {last}"
    return replace(state, ok=ok, digest=digest[:1400])
