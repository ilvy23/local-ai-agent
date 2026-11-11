"""What the project's own tests say, measured, never claimed: per-test failures, regressions against a baseline, flaky quarantine, and a
short digest for the model (300 lines of pytest noise cost a 28k window more than they teach).

`parse(output, code) -> TestState` understands pytest (`-q`; the short summary lines `FAILED path::name - msg`), unittest (`FAIL: name (module.Class)`),
`go test` (`--- FAIL: TestName`), `cargo test` (`test name ... FAILED`) and jest/vitest (`✕ name` / `FAIL path`). Anything else is judged by the exit code
alone (`failed` stays empty, `ok` = code 0)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_PYTEST = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+.*)?$", re.M)
_UNITTEST = re.compile(r"^(?:FAIL|ERROR):\s+(\S+)\s+\(([\w.]+)\)", re.M)
_GO = re.compile(r"^\s*--- FAIL:\s+(\S+)", re.M)
_CARGO = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED", re.M)
_JEST = re.compile(r"^\s*(?:✕|×)\s+(.+?)(?:\s+\(\d+\s*ms\))?$", re.M)
_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped)")
_ASSERT_LINE = re.compile(r"^E\s+(.*)$", re.M)


@dataclass
class TestState:
    __test__ = False                                      # pytest must not try to collect this as a test class
    ok: bool
    passed: int = 0
    failed: list[str] = field(default_factory=list)       # test ids that failed or errored
    code: int = 0
    tail: str = ""                                        # the last lines of the raw output (for logs)
    digest: str = ""                                      # what to show the model
    timed_out: bool = False                               # the test command did not finish within its limit

    def __iter__(self):                                   # `ok, tail = state` keeps working for callers that only want those two
        yield self.ok
        yield self.tail


def _failed_names(out: str) -> list[str]:
    names: list[str] = []
    for rx in (_PYTEST, _GO, _CARGO, _JEST):
        names += [m.group(1) if rx is not _JEST else m.group(1).strip() for m in rx.finditer(out)]
    names += [f"{m.group(2)}.{m.group(1)}" for m in _UNITTEST.finditer(out)]
    seen: dict[str, None] = {}
    for n in names:
        seen.setdefault(n, None)
    return list(seen)


def parse(out: str, code: int) -> TestState:
    counts = {k: int(n) for n, k in _COUNT.findall(out)}
    failed = _failed_names(out)
    return TestState(ok=code == 0, passed=counts.get("passed", 0), failed=failed, code=code, tail=out[-1500:], digest=digest(out, failed))


def digest(out: str, failed: list[str] | None = None, limit: int = 1400, first: int = 5) -> str:
    """The failing test ids, the assertion lines pytest marks with `E`, and the last lines: at most `limit` characters."""
    failed = _failed_names(out) if failed is None else failed
    parts: list[str] = []
    if failed:
        parts.append(f"{len(failed)} failing: " + ", ".join(failed[:first]) + (f" (+{len(failed) - first} more)" if len(failed) > first else ""))
    asserts = [m.group(1).strip() for m in _ASSERT_LINE.finditer(out)][:8]
    if asserts:
        parts.append("assertions: " + " | ".join(asserts))
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    parts.append("last lines: " + " / ".join(lines[-4:]))
    text = "\n".join(parts)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def regressions(before: TestState | None, after: TestState) -> list[str]:
    """Tests that fail now and did not fail in `before` (everything that fails counts when there is no baseline)."""
    known = set(before.failed) if before else set()
    return [t for t in after.failed if t not in known]


def fixed(before: TestState | None, after: TestState) -> list[str]:
    return [t for t in (before.failed if before else []) if t not in set(after.failed)]


def score(s: TestState) -> tuple[int, int, int]:
    """Higher is better: green first, then more passing tests, then fewer failing ones."""
    return (1 if s.ok else 0, s.passed, -len(s.failed))


def settle_flaky(first: TestState, rerun, times: int = 2) -> tuple[TestState, list[str]]:
    """Re-run a red result: a test that passes on ANY rerun is flaky. Returns (the state to trust, the flaky ids). `rerun() -> TestState`.
    When a rerun is fully green every failure of the first run was flaky and that green state is trusted; else the first (red) state stands."""
    if first.ok or not first.failed:
        return first, []
    flaky: set[str] = set()
    for _ in range(times):
        again = rerun()
        flaky |= set(first.failed) - set(again.failed)
        if again.ok:
            return again, sorted(first.failed)
    return first, sorted(flaky)
