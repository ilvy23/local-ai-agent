"""Why did the attempt fail, and what should the next one do about it? Decided from measurements (outcome, tests, diff, event log) - never
from the model's words. The kinds, in the order they are checked:

    server_down   the model server failed (errors in the event log / the attempt log): wait for it, then resume - the work is fine
    timeout       the attempt used its whole time slice
    idle          the work tree did not change for 30 minutes although events kept coming (a model that only reads and thinks)
    tests_hang    the project's tests did not finish in time (an infinite loop or a wait for input in the new code)
    stalled       nothing happened for a long time (hung call, dead server)
    crash         the process died with an unexpected exit code
    no_edits      it changed nothing on disk (it only read)
    loop          it repeated the same call (the autopilot's own loop guard fired)
    almost        a substantial change (50+ added lines) broke at most 3 tests that passed before: repair it, do not throw it away
    broke_tests   tests that passed before fail now (a regression)
    tests_red     tests are red, no regression against the baseline (work in progress)
    incomplete    green, but a file the prompt asked to create does not exist (deliverables.py)
    partial       the autopilot gave up on a stage (exit 2/3)
    green         tests green and exit 0 - nothing to fix

`next_strategy(kind, history)` maps a kind to the next strategy, and escalates when the same kind came twice in a row."""
from __future__ import annotations

from dataclasses import dataclass

from agent.night.handoff import DiffStat, EventSummary
from agent.night.runner import Outcome
from agent.night.testwatch import TestState, regressions

KINDS = ("server_down", "timeout", "idle", "tests_hang", "stalled", "almost", "crash", "no_edits", "loop", "broke_tests", "tests_red", "incomplete", "partial", "green")
ALMOST_MAX_TESTS, ALMOST_MIN_LINES = 3, 50                  # a big change that breaks this few tests is repaired, not rolled back
LOG_SIGNS = ("connection refused", "connecterror", "model server", "502", "503", "out of memory", "cuda error", "vulkan", "killed")

# kind -> strategies from mild to strong; a repeat of the same kind moves one step to the right (and stays on the last)
PLAN = {
    "server_down": ("resume", "resume", "fresh"),
    "timeout": ("fresh", "small", "rollback"),
    "stalled": ("fresh", "rollback", "small"),
    "idle": ("small", "fresh", "rollback"),
    "tests_hang": ("resume", "fresh", "rollback"),
    "crash": ("fresh", "fresh", "rollback"),
    "no_edits": ("small", "fresh", "rollback"),
    "loop": ("fresh", "small", "rollback"),
    "broke_tests": ("rollback", "small", "rollback"),
    "almost": ("repair", "fresh", "rollback"),
    "tests_red": ("resume", "fresh", "small", "rollback"),
    "incomplete": ("resume", "fresh", "small"),
    "partial": ("fresh", "small", "rollback"),
    "green": ("resume",),
}


@dataclass
class Verdict:
    kind: str
    detail: str


def classify(out: Outcome, before: TestState | None, after: TestState, diff: DiffStat, events: EventSummary, log_tail: str = "") -> Verdict:
    low = log_tail.lower()
    if events.model_errors >= 3 or events.server_waits or (events.steps < 3 and any(s in low for s in LOG_SIGNS)):
        return Verdict("server_down", f"{events.model_errors} model errors, {events.server_waits} server waits")
    if out.why == "timeout":
        return Verdict("timeout", "the time slice of the attempt was used up")
    if out.why == "idle":
        return Verdict("idle", "the work tree did not change for 30 minutes")
    if after.timed_out:
        return Verdict("tests_hang", "the test suite did not finish: something loops or waits for input")
    if out.why == "stalled":
        return Verdict("stalled", "no event for a long time")
    if out.why == "crash":
        return Verdict("crash", f"exit code {out.code}")
    if out.code == 0 and after.ok:
        return Verdict("green", "tests green")
    if diff.empty():
        return Verdict("no_edits", f"{events.reads} reads, nothing written")
    if events.loops >= 3:
        return Verdict("loop", f"{events.loops} loops")
    reg = regressions(before, after) if before is not None else []
    if reg:                                                  # only meaningful against a baseline: without one every failure would count
        if len(reg) <= ALMOST_MAX_TESTS and diff.added >= ALMOST_MIN_LINES:
            return Verdict("almost", f"{diff.added} lines added, only {', '.join(reg)} broke")
        return Verdict("broke_tests", "broke " + ", ".join(reg[:4]))
    if not after.ok:
        return Verdict("tests_red", after.digest[:200])
    return Verdict("partial", f"exit code {out.code}, tests green but the autopilot did not finish")


def next_strategy(kind: str, history: list[str], stats: dict[str, tuple[int, float]] | None = None) -> str:
    """`history` = the kinds of the earlier attempts of this prompt (oldest first). The k-th consecutive repeat picks the k-th entry of PLAN[kind].
    `stats` (attemptlog.rates for this kind) may overrule the default with evidence from this night."""
    plan = PLAN.get(kind, PLAN["tests_red"])
    repeats = 0
    for k in reversed(history):
        if k != kind:
            break
        repeats += 1
    default = plan[min(repeats, len(plan) - 1)]
    if not stats:
        return default
    from agent.night import attemptlog

    return attemptlog.prefer(default, list(dict.fromkeys(plan)), stats)
