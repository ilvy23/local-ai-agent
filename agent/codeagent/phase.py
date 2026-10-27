"""The verification-spiral phase and the honest forced ending.

Seen live (task 'Make a snake game.', one-shot): the game file was complete after the first write, then the model sent 11 more `run` calls
(the same smoke-test heredoc that fakes a curses screen); the loop guard warned twice and aborted - the run ended 'interrupted (loop_abort)'
WITHOUT a final answer although everything was done. Two deterministic fixes, no extra model call except ONE tool-less summary turn:

  * `VerifyPhase`: the 'verification phase' is a real (non-scratch) edit in this prompt plus >= coding.verify_phase_min (3) calls since the last
    edit / failure signal that were all `run` or read-only calls with a clean result that changed nothing. In it the loop guard's notes say
    'give your final answer now' instead of 'try a different approach' (and force no reasoning pass), and coding.verify_phase_budget (6) calls
    end the tool use (stopped='verify_budget'). A new real edit, any other write, a changed file or a failure signal (error shape, non-zero
    exit, traceback, red test, syntax error) starts the count again.
  * `facts` / `prefix` / `mechanical` / `instruction`: when the engine stops WITHOUT a model answer in a one-shot or REPL prompt, it asks for ONE
    answer with tools disabled (engine._forced_ending) from the harness-measured facts below; the text shown is prefixed with a factual harness
    line, the stored message stays clean, and a failed call falls back to the mechanical summary. Nothing here runs a test or a check.
Config (coding.*): verify_phase_min (3), verify_phase_budget (6; 0 = no budget), forced_summary (true); see docs/CODE_ENGINE.md."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.codeagent.loopguard import PHASE_NOTE  # noqa: F401 - the note's one definition (loopguard strips it when comparing copies)

VERIFY_PHASE_MIN, VERIFY_PHASE_BUDGET = 3, 6
SUMMARY_PREDICT = 400
INSTRUCTION = ("Write the final answer for the user: what was built (files), what is verified and what is not, what is left. Do not call tools.")
REASONS = {"loop": "the same kind of call repeated without progress", "verify_budget": "the post-edit verification budget was used up",
           "budget": "the step budget was used up", "announce": "the model announced a next step without doing it",
           "malformed": "too many unusable replies in a row", "stage_cap": "the context cap was reached"}
_TRACEBACK = re.compile(r"Traceback \(most recent call last\)|SYNTAX GATE FAILED|SyntaxError")
_RED = re.compile(r"\b[1-9]\d* (?:failed|errors?)\b")


class VerifyPhase:
    """Counters of the verification phase (see the module text). State is per prompt (engine.turn resets it) and is not saved."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.edited = False                 # a real edit happened in this prompt / stage
        self.clean = 0                      # clean run / read-only calls since the last edit or failure signal
        self.ended = False                  # the budget is used up: the engine ends the tool use

    def edit(self) -> None:
        self.edited, self.clean, self.ended = True, 0, False

    def restart(self) -> None:
        """A failure signal or another kind of activity: the clean count (and the budget) start again."""
        self.clean, self.ended = 0, False

    def active(self, min_calls: int = VERIFY_PHASE_MIN) -> bool:
        return self.edited and self.clean >= max(1, min_calls)

    def clean_call(self, budget: int) -> None:
        self.clean += 1
        if budget > 0 and self.edited and self.clean >= budget:
            self.ended = True


def failure_signal(name: str, result: Any, failed_run: bool) -> bool:
    """A result that is a failure signal: an error shape / non-zero exit (`failed_run`, the engine's own test), a traceback or syntax error in the
    output, or a test run that reports failures."""
    text = str(result)
    if failed_run or _TRACEBACK.search(text):
        return True
    return name in ("run_tests", "run") and bool(_RED.search(text)) and bool(re.search(r"\b(?:pytest|unittest|passed|failed)\b", text))


# ── the facts of a forced ending ────────────────────────────────────────────

def _clip(text: Any, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def reason_text(stopped: str, reason: str = "") -> str:
    key = reason if stopped == "error" and reason else stopped
    return f"{stopped}{'/' + reason if stopped == 'error' and reason else ''} - {REASONS.get(key, 'the engine stopped before a final answer')}"


def facts(engine: Any, result: Any, files: list[str]) -> dict[str, Any]:
    """Measured, never remembered, and nothing is run: the changed files, the commands since the last edit with their exit codes (the last known test /
    check results), checks, syntax / static errors and open acceptance items. `not_verified` lists what is not proven."""
    from agent.codeagent import loopguard, verified

    t = engine.tools
    ev = list(getattr(t, "evidence", []) or [])
    commands = [f"`{_clip(cmd, 50)}` exit {code}" for _s, cmd, code in ev[-4:]]
    red = [cmd for _s, cmd, code in ev if code not in (0, None)]
    tests = [(cmd, code) for _s, cmd, code in ev if loopguard.is_test_command(cmd)]
    not_verified: list[str] = []
    try:
        state = verified.build(engine, run_tests=False, run_checks=False)
        sec = state.sections
    except Exception:  # noqa: BLE001 - the facts are a best effort; a broken block must not hide the others
        sec = {}
    static_n = int((sec.get("STATIC") or {}).get("problems") or 0)
    bad = sorted({*getattr(t, "syntax_bad", ()), *getattr(t, "static_bad", ())})
    open_ids = list((sec.get("OPEN ITEMS") or {}).get("open") or [])
    failing_checks = [cid for cid, st in ((sec.get("CHECKS") or {}).get("results") or {}).items() if st not in ("pass", "skipped")]
    if files and not tests:
        not_verified.append("no test run since the last edit")
    if not ev:
        not_verified.append("nothing was run since the last edit")
    if red:
        not_verified.append(f"{len(red)} command(s) failed since the last edit")
    if bad or static_n:
        not_verified.append("syntax/static problems: " + (", ".join(bad[:3]) or str(static_n)))
    if failing_checks:
        not_verified.append("checks not passing: " + ", ".join(failing_checks[:4]))
    if open_ids:
        not_verified.append("open acceptance items: " + ", ".join(str(i) for i in open_ids[:5]))
    return {"stop": reason_text(result.stopped, getattr(result, "reason", "")), "files": files, "commands": commands, "tests": tests,
            "open": open_ids, "not_verified": not_verified, "since_edit": len(ev)}


def facts_line(f: dict[str, Any]) -> str:
    """One compact line for the harness prefix."""
    files = ", ".join(f["files"][:6]) + (f" +{len(f['files']) - 6}" if len(f["files"]) > 6 else "") if f["files"] else "none"
    run = f"{f['since_edit']} command(s) since the last edit, last: " + "; ".join(f["commands"][-2:]) if f["commands"] else "no command since the last edit"
    out = f"files changed: {files}; {run}"
    if f["not_verified"]:
        out += "; NOT VERIFIED: " + "; ".join(f["not_verified"])
    return out


def prefix(f: dict[str, Any]) -> str:
    return f"[stopped: {f['stop']} - summary below was written after the stop; verified facts: {facts_line(f)}]"


def message(task: str, f: dict[str, Any]) -> str:
    """The user turn of the summary call: the measured facts, then the fixed instruction."""
    rows = [f"The tool use has ended ({f['stop']}).", "Facts measured by the harness (trust them, do not claim more):",
            f"- files changed: {', '.join(f['files']) or 'none'}",
            f"- commands since the last edit: {'; '.join(f['commands']) or 'none'}"]
    rows += [f"- NOT VERIFIED: {x}" for x in f["not_verified"]]
    if task:
        rows.insert(1, f"The task was: {_clip(task, 300)}")
    return "\n".join(rows) + "\n\n" + INSTRUCTION


def mechanical(f: dict[str, Any]) -> str:
    """The summary without a model: files changed and the measured status."""
    lines = [f"Work stopped before a final answer ({f['stop']}).",
             "Files changed: " + (", ".join(f["files"]) if f["files"] else "none") + ".",
             "Commands since the last edit: " + ("; ".join(f["commands"]) if f["commands"] else "none") + "."]
    lines.append("Not verified: " + ("; ".join(f["not_verified"]) if f["not_verified"] else "nothing the harness could measure is open") + ".")
    lines.append("Say 'continue' to go on.")
    return "\n".join(lines)


def rel_files(engine: Any) -> list[str]:
    t = engine.tools
    return [t.rel(Path(p)) for p in t.changes.changed()]
