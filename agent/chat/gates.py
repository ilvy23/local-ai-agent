"""Stateful turn features, factored out of the terminal loop (agent.tui) so any front end can use them.

Each gate is small and testable on its own: it makes a decision and returns data (console markup, an event dict,
messages to append). It never prints, never reads stdin and never imports agent.tui. The front end shows the result
the way it likes (rich console, JSON event, both). Contract and rationale: docs/design/turn-convergence.md.

  verify      make_verify_gate / note_clean_edit / run_verification   run the project's check before accepting "done"
  plan        plan_creation_enabled / PlanGate                        nudge the model back to unconfirmed plan items
  goal        GoalReminder                                            restate the original task every N tool calls
  steps       StepBudget                                              extend the step budget on request (callback)
  changes     ChangeTracker (+ shared CHANGES / UNDO / LAST_EDITED)   per-prompt file change log, git checkpoint, undo
  access      decide_access                                           repo-access grant / web-taint decision for one call
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from agent import autonomy, changes, plan as planmod, repoaccess, undo, verify
from agent.chat.helpers import DENIED_RESULT, _mood, _one_line
from agent.research import taint
from agent.safety import RiskLevel

logger = logging.getLogger(__name__)

Say = Callable[[str], None]            # print one console-markup line
Event = Callable[..., None]            # event(kind, **data); a no-op front end passes `_no_event`


def _no_event(kind: str, /, **data: Any) -> None:
    return None


GATE_NAMES = ("plan", "goal", "step_extension", "changes", "verify")


def gate_enabled(config: dict[str, Any], name: str, default: bool = True) -> bool:
    """`chat.gates.<name>` from the config; `default` when it is not set."""
    value = (((config.get("chat") or {}).get("gates")) or {}).get(name)
    return default if value is None else bool(value)


def begin_turn() -> None:
    """Per-turn reset of the process-wide web/evidence singletons: what the web said in an earlier turn is not what
    this turn read (taint), and the evidence record starts empty. Call once at the start of a NEW turn, never on a
    resume after an approval."""
    from agent.research import evidence as turn_evidence

    taint.clear()
    turn_evidence.clear()


# --------------------------------------------------------------------------------------------- verify gate

def make_verify_gate(config: dict[str, Any], registry: Any) -> "verify.VerifyGate | None":
    """The gate for this turn, or None when verification is off or there is no `run_command` tool to run it with."""
    if verify.enabled(config) and registry.get("run_command"):
        return verify.VerifyGate(retries=int((config.get("autonomy") or {}).get("verify_retries", 2)))
    return None


def note_clean_edit(gate: "verify.VerifyGate | None", call: dict[str, Any], result: str) -> None:
    """After a tool call: a file edit that did not fail makes the file 'dirty' (to be checked before done)."""
    if gate is not None and call["name"] in changes.TRACKED_TOOLS and call["arguments"].get("path") \
            and autonomy.failure_signature(call["name"], call["arguments"], result) is None:
        gate.note_edit(str(call["arguments"]["path"]))


def run_verification(
    gate: "verify.VerifyGate", run_command: Callable[[str], str], messages: list[dict[str, Any]], answer: str, *,
    say: Say = lambda _line: None, event: Event = _no_event, width: int = 80,
    on_pass: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]] | None:
    """Run the project's check before accepting "done". `run_command(command)` runs it through the front end's own
    tool executor (risk gate included) and returns the result text. Returns the messages to retry with when the check
    failed (and tries remain), else None: passed, nothing to run, declined, or out of tries. `on_pass(command, file)`
    is called after a pass (project memory)."""
    command = gate.command()
    if not command:
        gate.give_up()                                    # nothing to check - don't ask again
        return None
    from rich.markup import escape

    say(f"[dim]verify  {escape(_one_line(command, max(width - 12, 20)))}[/dim]")
    event("status", state="verifying", label="verifying", detail=_one_line(command, 160), since=time.time())
    event("verify", state="running", command=command, tries=gate.tries, retries=gate.retries)
    return finish_verification(gate, command, run_command(command), messages, answer, say=say, event=event, on_pass=on_pass)


def finish_verification(
    gate: "verify.VerifyGate", command: str, result: str, messages: list[dict[str, Any]], answer: str | None = None, *,
    say: Say = lambda _line: None, event: Event = _no_event, on_pass: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]] | None:
    """Judge the result of the check command: the messages to retry with when it failed (and tries remain), else None.
    Split out so a front end that suspended while the command waited for an approval (the web) can finish the
    verification after the answer arrives; there `answer` (the held-back reply) is not at hand and is left out."""
    if result == DENIED_RESULT:
        gate.give_up()
        say("[dim]verify  skipped[/dim]")
        event("verify", state="skipped", command=command)
        return None
    code = verify.exit_code(result)
    if code == 0:
        edited = gate.dirty[0] if gate.dirty else ""
        gate.passed()
        say(f"[green]{_mood('test_passed')}verify  passed[/green]")
        event("verify", state="passed", command=command)
        if on_pass and edited:
            try:
                on_pass(command, edited)
            except Exception:  # noqa: BLE001 - memory is a bonus
                logger.debug("project hook failed", exc_info=True)
        return None
    if gate.failed():
        say(f"[red]{_mood('error')}verify  failed - back to work ({gate.tries}/{gate.retries})[/red]")
        event("verify", state="failed", command=command, tries=gate.tries, retries=gate.retries)
        held = [{"role": "assistant", "content": answer}] if answer is not None else []
        return [*messages, *held, {"role": "system", "content": verify.failure_message(command, result)}]
    gate.give_up()
    say(f"[red]{_mood('error')}verify  still failing after {gate.retries} fixes - treat the result as unfinished[/red]")
    event("verify", state="gave_up", command=command, tries=gate.tries, retries=gate.retries)
    return None


# -------------------------------------------------------------------------------------------------- plan gate

MAX_PLAN_NUDGES = 2


def plan_creation_enabled(user_input: str, config: dict[str, Any]) -> bool:
    """Should a plan be generated for this prompt (big build task, plan not switched off in config or env)?"""
    return bool(planmod.should_create(user_input) and (config.get("autonomy") or {}).get("plan", True)
                and os.environ.get("AGENT_PLAN", "").lower() not in ("0", "off", "false", "no"))


@dataclass
class PlanDecision:
    action: str                                   # "nudge" (go back to work) | "unconfirmed" (give up, say so)
    line: str                                     # console markup line
    event: dict[str, Any]                         # payload for the `plan` event (state, confirmed, total, nudges, items)
    reminder: str = ""                            # system text for the retry (nudge only)

    def retry_messages(self, messages: list[dict[str, Any]], content: str) -> list[dict[str, Any]]:
        return [*messages, {"role": "assistant", "content": content}, {"role": "system", "content": self.reminder}]


class PlanGate:
    """When the model says it is done but plan items are unconfirmed: nudge it back (at most twice per prompt)."""

    @staticmethod
    def reset(holder: Any) -> None:
        if holder is not None and holder.plan is not None:
            holder.plan.nudges = 0

    @staticmethod
    def check(holder: Any, content: str) -> PlanDecision | None:
        if holder is None or holder.plan is None or not content.strip() or not holder.plan.open_items():
            return None
        p = holder.plan
        items = [{"id": i.id, "text": i.text, "done": i.done} for i in p.items]
        if p.nudges < MAX_PLAN_NUDGES:
            p.nudges += 1
            return PlanDecision(
                "nudge", f"[dim]plan  {p.confirmed()} of {len(p.items)} confirmed - back to work[/dim]",
                {"state": "nudge", "confirmed": p.confirmed(), "total": len(p.items), "nudges": p.nudges, "items": items},
                p.reminder())
        return PlanDecision(
            "unconfirmed", f"[red]plan  {len(p.open_items())} of {len(p.items)} items not confirmed[/red]",
            {"state": "unconfirmed", "confirmed": p.confirmed(), "total": len(p.items), "nudges": p.nudges, "items": items})


# ------------------------------------------------------------------------------------------- goal restatement

@dataclass
class GoalDecision:
    log_text: str
    message: dict[str, Any]


class GoalReminder:
    """Every `every_n` tool calls, a reminder of the original task so the model does not drift on long runs.
    Wraps the plain `goal_state` dict (total_calls / every_n / original_task) the front ends already keep."""

    def __init__(self, goal_state: dict[str, Any] | None) -> None:
        self.state = goal_state

    def tick(self) -> GoalDecision | None:
        """Count one tool call; the reminder when this one is due, else None."""
        g = self.state
        if g is None:
            return None
        g["total_calls"] = g.get("total_calls", 0) + 1
        every_n = g.get("every_n", 10)
        original_task = g.get("original_task", "")
        if not (every_n > 0 and original_task and g["total_calls"] % every_n == 0):
            return None
        return GoalDecision(
            f"goal check (after {g['total_calls']} tool calls)",
            {"role": "system", "content": (
                f"GOAL REMINDER — {g['total_calls']} tool calls so far. "
                f"Original task: {original_task}\n"
                "Verify your next action still works toward this goal.")})


# --------------------------------------------------------------------------------------------- step extension

class StepBudget:
    """The model's step budget. When it is used up `ask_more(done)` (a console prompt, a web approval event, ...)
    decides whether to grant another `max_iterations`; with no callback the turn simply stops."""

    def __init__(self, max_iterations: int, ask_more: Callable[[int], bool] | None = None) -> None:
        self.step = max_iterations
        self.budget = max_iterations
        self.ask_more = ask_more
        self.extensions = 0

    def may_continue(self, iteration: int) -> bool:
        """`iteration` steps are done: True when step number iteration+1 may run."""
        if iteration < self.budget:
            return True
        if self.ask_more is None or not self.ask_more(iteration):
            return False
        self.budget += self.step
        self.extensions += 1
        return True


# --------------------------------------------------------------------------------------------- change tracking

CHANGES = changes.FileChangeLog()
UNDO = undo.UndoStack()
LAST_EDITED: list[str] = []           # files edited by the most recent prompt (to find the active project)


class ChangeTracker:
    """One prompt's file-change log, its git checkpoint and the undo stack, behind one object."""

    def __init__(self, log: "changes.FileChangeLog | None" = None, undo_stack: "undo.UndoStack | None" = None,
                 last_edited: list[str] | None = None, checkpoints: bool = True) -> None:
        self.checkpoints = checkpoints            # False: no git checkpoint (a front end without /undo, e.g. the web)
        self.log = CHANGES if log is None else log
        self.undo = UNDO if undo_stack is None else undo_stack
        self.last_edited = LAST_EDITED if last_edited is None else last_edited

    @classmethod
    def shared(cls) -> "ChangeTracker":
        return cls()

    @classmethod
    def private(cls, checkpoints: bool = False) -> "ChangeTracker":
        """A tracker with its own log / undo stack (one web turn; never touches the terminal singletons)."""
        return cls(changes.FileChangeLog(), undo.UndoStack(), [], checkpoints=checkpoints)

    @staticmethod
    def is_tracked(name: str, arguments: dict[str, Any]) -> bool:
        return name in changes.TRACKED_TOOLS and bool(arguments.get("path"))

    def begin(self) -> None:
        self.log.clear()

    def checkpoint_before(self, path: str) -> None:
        """First tracked edit of the prompt: take the git checkpoint undo restores from."""
        if not self.log.checkpoint_tried:
            self.log.checkpoint_tried = True
            if self.checkpoints:
                self.log.checkpoint = undo.git_checkpoint(path)

    def record(self, path: str, before: str | None, after: str | None) -> None:
        self.log.record(path, before, after)

    def finish(self) -> dict[str, tuple[str | None, str]]:
        """End of the prompt: push the edits onto the undo stack and remember which files were touched."""
        changed = self.log.changed()
        self.undo.push(changed, self.log.checkpoint)
        self.last_edited[:] = list(changed)
        return changed

    def totals(self) -> list[tuple[str, int, int, bool]]:
        return self.log.totals()

    def change_event(self) -> dict[str, Any] | None:
        """Payload of the closing `changes` event, or None when nothing changed."""
        rows = self.log.totals()
        if not rows:
            return None
        return {"files": [{"path": changes.display_path(p_), "added": a_, "removed": r_, "created": c_}
                          for p_, a_, r_, c_ in rows],
                "added": sum(r[1] for r in rows), "removed": sum(r[2] for r in rows)}

    def print_summary(self, console: Any) -> None:
        self.log.print_summary(console)

    def clear(self) -> None:
        self.log.clear()

    def undo_lines(self, list_only: bool = False) -> list[str]:
        return self.undo.describe() if list_only else self.undo.undo()


# ------------------------------------------------------------------------------------------ repo access / taint

@dataclass
class AccessDecision:
    from_web: bool                    # an argument is text copied from a web page (never auto-allowed)
    auto_allowed: bool                # a granted folder access answers the question: no approval needed


def decide_access(name: str, arguments: dict[str, Any], level: RiskLevel) -> AccessDecision:
    """For a call that would ask for approval (CAUTION/DANGEROUS): does a repo-access grant make asking unnecessary?
    Text taken from a web page is always asked about."""
    from_web = taint.copied(arguments)
    return AccessDecision(from_web, not from_web and repoaccess.allows(name, arguments, level is RiskLevel.CAUTION))
