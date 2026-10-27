"""The ONE tool executor of the shared chat engine (agent.chat.turn.run_turn): what `tui._execute_tool` does for the
terminal, with the approval step pluggable.

For every tool call: unknown tool / bad arguments come back to the model as text, the call is risk-classified
(`helpers._classify_call`), BLOCKED never runs, CAUTION/DANGEROUS ask, everything that runs is audited, file edits are
tracked (git checkpoint, change log, `edit` event with a diff) and the result goes back to the model.

Approval strategies (how "ask" is answered):

  BlockingApprover  terminal / dock: emits `awaiting_approval`, then BLOCKS on `approve_fn(ApprovalRequest)` in the
                    worker thread. DANGEROUS only accepts the exact typed word `yes` (ui.chat_adapter.resolve_approval).
                    The only strategy that honours folder access grants (repoaccess greenlists) - and never for text
                    copied from a web page (gates.decide_access).
  SuspendApprover   web: persists a pending approval (with the turn state), emits `awaiting_approval` and raises
                    ApprovalNeeded so the turn suspends; the approvals route resumes it later. Never blocks a thread and
                    never auto-allows: whatever asked before still asks.

Two event styles (ExecutorStyle): the terminal's (tool_call announced as soon as the call is classified, `risk`,
`summary`, a `status` event per tool, sources/verdict after a web search) and the web's (silent for unknown tools and
bad arguments, tool_call right before the run, `result_preview` on tool_result, no `status`, sources come from the tool
itself through the web console). `tool_result` always carries `result` (capped at TOOL_RESULT_CAP).
"""

from __future__ import annotations

import difflib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from agent import changes, toolargs
from agent.chat import gates
from agent.chat.helpers import DENIED_RESULT, RESULT_TRUNCATE_LEN, _classify_call, _format_tool_call, _one_line
from agent.chat.turn import ApprovalNeeded, plain_description
from agent.safety import RiskLevel, block_reason
from agent.tools.files import safe_text

logger = logging.getLogger(__name__)

TOOL_RESULT_CAP = 20_000          # characters of a tool result carried in a tool_result event
WEB_TEXT_SUFFIX = " - text taken from a web page"

Event = Callable[..., None]       # event(kind, **data)


# ------------------------------------------------------------------------------------------------ small helpers

def unified_diff(path: str, before: str | None, after: str, limit: int = 2000) -> tuple[str, bool]:
    lines = list(difflib.unified_diff((before or "").splitlines(True), after.splitlines(True),
                                      fromfile="/dev/null" if before is None else f"a/{path}", tofile=f"b/{path}"))
    cut = len(lines) > limit
    text = "".join(line if line.endswith("\n") else line + "\n" for line in lines[:limit])
    return text, cut


def emit_search_evidence(event: Event, seen: dict[str, Any]) -> None:
    """After a web search: `sources` and `verdict` events from the turn's evidence record (once per search).
    `seen["sig"]` remembers the last search already reported."""
    from agent.research import evidence as turn_evidence

    last = turn_evidence.last_search()
    if not last["items"]:
        return
    sig = (id(last["items"][0]), len(last["items"]), str(last["verdict"].get("label", "")))
    if sig == seen.get("sig"):
        return
    seen["sig"] = sig
    items = [{**{k: v for k, v in i.items() if k != "passage"}, "passage": str(i.get("passage", ""))[:600]}
             for i in last["items"]]
    event("sources", items=items)
    if last["verdict"]:
        event("verdict", **last["verdict"])


_TOOL_STATUS = {
    "web_search": ("searching", "query"), "web_fetch": ("fetching", "url"), "read_file": ("reading", "path"),
    "list_dir": ("listing", "path"), "search_files": ("searching files", "pattern"), "find_files": ("finding files", "pattern"),
    "run_command": ("running command", "command"), "write_file": ("writing", "path"),
    "replace_in_file": ("editing", "path"), "deep_research": ("researching", "question"),
}


def tool_status(event: Event, name: str, arguments: dict[str, Any]) -> None:
    label, key = _TOOL_STATUS.get(name, (f"running {name}", ""))
    detail = _one_line(str(arguments.get(key, "")), 160) if key else ""
    event("status", state="tool", label=label, detail=detail, tool=name, since=time.time())


def run_approved_call(
    registry: Any, store: Any, tool: Any, name: str, arguments: dict[str, Any], *, tracker: "gates.ChangeTracker | None",
    event: Event, audit_kind: str, detail: str,
) -> tuple[str, bool]:
    """Run an allowed call: track file edits (checkpoint, change log, `edit` event), clean the result, audit it and fire
    the registry's on-call hook. Shared by the executor and by the web's resume-after-approval. Returns (result, errored)."""
    tracked = tracker is not None and tracker.is_tracked(name, arguments)
    if tracked:
        tracker.checkpoint_before(str(arguments["path"]))                      # type: ignore[union-attr]
    before = changes.read_text(arguments["path"]) if tracked else None
    errored = False
    try:
        result = tool.handler(**arguments)
    except Exception as exc:  # noqa: BLE001 - handler errors go back to the model
        result = f"Error running tool '{name}': {exc}"
        errored = True
    # Backstop: a result that can't be encoded as UTF-8 kills the whole session when it is sent to the model as JSON.
    result = safe_text(result)
    if tracked:
        after = changes.read_text(arguments["path"])
        tracker.record(str(arguments["path"]), before, after)                  # type: ignore[union-attr]
        if after is not None and before != after:
            shown_path = changes.display_path(str(arguments["path"]))
            added, removed = changes.line_stats(before, after)
            diff, cut = unified_diff(shown_path, before, after)
            event("edit", path=shown_path, abs_path=str(Path(str(arguments["path"])).expanduser()),
                  added=added, removed=removed, created=before is None, diff=diff, diff_truncated=cut)
    store.add_audit_log(kind=audit_kind, detail=detail, approved=1, result=result[:RESULT_TRUNCATE_LEN])
    registry.fire_on_call(name, arguments, result)
    return result, errored


# ----------------------------------------------------------------------------------------- approval strategies

@dataclass
class PendingCall:
    """One call that needs an answer."""

    name: str
    arguments: dict[str, Any]
    level: RiskLevel
    description: str                  # one plain line (no markup)
    diff: str = ""                    # unified diff for write_file, else ""
    from_web: bool = False            # an argument is text copied from a web page
    state: dict[str, Any] = field(default_factory=dict)      # the turn state a suspended turn resumes from


class ApprovalStrategy(Protocol):
    honours_repo_access: bool

    def __call__(self, call: PendingCall) -> bool:
        """True = run it, False = denied. May raise ApprovalNeeded (the turn suspends)."""


class BlockingApprover:
    """Terminal: ask through `approve_fn` and wait. With no `approve_fn` there is nobody to ask, so the answer is no
    (never a stdin read behind a hidden console)."""

    honours_repo_access = True

    def __init__(self, event: Event, approve_fn: Callable[[Any], Any] | None) -> None:
        self.event, self.approve_fn = event, approve_fn

    def __call__(self, call: PendingCall) -> bool:
        from agent.ui.chat_adapter import ApprovalRequest, resolve_approval

        level = call.level
        plain = call.description + (WEB_TEXT_SUFFIX if call.from_web else "")
        self.event("awaiting_approval", name=call.name, risk=level.name, description=plain, arguments=call.arguments,
                   from_web=call.from_web, diff=call.diff, needs_typed_yes=level is RiskLevel.DANGEROUS)
        if self.approve_fn is None:
            return False
        request = ApprovalRequest(call.name, level, level.name, plain, dict(call.arguments), call.from_web, call.diff,
                                  level is RiskLevel.DANGEROUS)
        try:
            return resolve_approval(level, self.approve_fn(request))
        except Exception:  # noqa: BLE001 - a broken approver means no
            logger.warning("approve_fn failed", exc_info=True)
            return False


class SuspendApprover:
    """Web: save the call as a pending approval, tell the GUI and suspend the turn. Never auto-allows."""

    honours_repo_access = False

    def __init__(self, store: Any, session_id: int, event: Event) -> None:
        self.store, self.session_id, self.event = store, session_id, event

    def __call__(self, call: PendingCall) -> bool:
        approval_id = self.store.add_pending_approval(
            session_id=self.session_id, tool_name=call.name, arguments=call.arguments, risk_level=call.level.name,
            description=call.description, turn_state=call.state)
        self.event("awaiting_approval", approval_id=approval_id, tool_name=call.name, arguments=call.arguments,
                   risk_level=call.level.name, description=call.description, from_web=call.from_web)
        raise ApprovalNeeded(approval_id)


# ----------------------------------------------------------------------------------------------------- executor

@dataclass(frozen=True)
class ExecutorStyle:
    announce_first: bool          # tool_call right after classification (terminal) / right before the run (web)
    report_unrun: bool            # tool_call + tool_result for unknown tools and bad arguments (terminal) / silent (web)
    tool_status: bool             # a `status` event per tool
    result_preview: bool          # `result_preview` on tool_result (the web GUI reads it)


TERMINAL_STYLE = ExecutorStyle(announce_first=True, report_unrun=True, tool_status=True, result_preview=False)
WEB_STYLE = ExecutorStyle(announce_first=False, report_unrun=False, tool_status=False, result_preview=True)


class ToolExecutor:
    """`execute(name, arguments, state) -> result text` for agent.chat.turn.run_turn."""

    def __init__(
        self, registry: Any, store: Any, config: dict[str, Any], event: Event, approver: ApprovalStrategy, *,
        style: ExecutorStyle = TERMINAL_STYLE, tracker: "gates.ChangeTracker | None" = None,
        after_result: Callable[[], None] | None = None, result_cap: int = TOOL_RESULT_CAP,
    ) -> None:
        self.registry, self.store, self.config, self.event, self.approver = registry, store, config, event, approver
        self.style, self.tracker, self.after_result, self.result_cap = style, tracker, after_result, result_cap

    # -- reporting
    def _result_event(self, name: str, result: str, ok: bool, level: RiskLevel | None, started: float, *,
                      command: str = "", ran: bool = False) -> str:
        cap = self.result_cap
        data: dict[str, Any] = {
            "name": name, "result": result if len(result) <= cap else result[:cap], "truncated": len(result) > cap,
            "chars": len(result), "seconds": round(time.monotonic() - started, 3), "ok": ok,
            "risk": level.name if level is not None else None}
        if self.style.result_preview:
            data["result_preview"] = f"[ran: {command}]" if ran and command else result[:200]
        self.event("tool_result", **data)
        return result

    def _announce(self, name: str, arguments: dict[str, Any], level: RiskLevel | None, summary: str | None = None) -> None:
        self.event("tool_call", name=name, summary=summary if summary is not None else _one_line(_format_tool_call(name, arguments), 160),
                   arguments=arguments, risk=level.name if level is not None else None)

    # -- the call
    def __call__(self, name: str, arguments: dict[str, Any], state: dict[str, Any] | None = None) -> str:
        style, started = self.style, time.monotonic()
        tool = self.registry.get(name)
        if tool is None:
            result = f"Error: no such tool '{name}'."
            if style.report_unrun:
                self._announce(name, arguments, None, summary=name)
            self.store.add_audit_log(kind="tool", detail=json.dumps({"name": name, "arguments": arguments}), approved=0, result=result)
            return self._result_event(name, result, False, None, started) if style.report_unrun else result
        arguments, bad_arguments = toolargs.validate(tool, arguments)
        if bad_arguments:
            if not style.report_unrun:
                return bad_arguments                          # nothing ran and nothing was asked: the model just retries
            self._announce(name, arguments, None)
            return self._result_event(name, bad_arguments, False, None, started)

        is_shell = tool.risk == "dynamic"
        audit_kind = "shell" if is_shell else "tool"
        command = str(arguments.get("command", ""))
        detail = command if is_shell else json.dumps({"name": name, "arguments": arguments})
        level = _classify_call(tool, arguments, self.config)
        if style.announce_first:
            self._announce(name, arguments, level)

        if level is RiskLevel.BLOCKED:                         # never runs
            reason = block_reason(command, self.config) if is_shell else (
                "sending WhatsApp messages is switched off" if name.startswith("wa_send") else "tool declared blocked")
            result = f"BLOCKED by safety policy: {reason}"
            self.store.add_audit_log(kind=audit_kind, detail=detail, approved=0, result=result)
            return self._result_event(name, result, False, level, started)

        if level is not RiskLevel.SAFE:                        # CAUTION / DANGEROUS: ask (or a granted folder answers)
            access = gates.decide_access(name, arguments, level)
            auto_allowed = access.auto_allowed and getattr(self.approver, "honours_repo_access", False)
            if not auto_allowed:
                path = str(arguments.get("path", "")) if name in changes.TRACKED_TOOLS else ""
                diff = ""
                if path and name == "write_file":
                    diff = unified_diff(changes.display_path(path), changes.read_text(path), str(arguments.get("content", "")))[0]
                try:
                    plain = plain_description(name, arguments)
                except Exception:  # noqa: BLE001
                    plain = _one_line(_format_tool_call(name, arguments), 200)
                call = PendingCall(name, arguments, level, plain, diff, access.from_web, dict(state or {}))
                if not self.approver(call):                    # ApprovalNeeded propagates: the web turn suspends here
                    self.store.add_audit_log(kind=audit_kind, detail=detail, approved=0, result=DENIED_RESULT)
                    return self._result_event(name, DENIED_RESULT, False, level, started)
                started = time.monotonic()                     # the wait for the answer is not the tool's runtime

        if not style.announce_first:
            self._announce(name, arguments, level)
        if style.tool_status:
            tool_status(self.event, name, arguments)
        result, errored = run_approved_call(self.registry, self.store, tool, name, arguments, tracker=self.tracker,
                                            event=self.event, audit_kind=audit_kind, detail=detail)
        out = self._result_event(name, result, not errored and not result.startswith(("Error", "BLOCKED")), level, started,
                                 command=command if is_shell else "", ran=True)
        if self.after_result is not None:
            self.after_result()
        return out
