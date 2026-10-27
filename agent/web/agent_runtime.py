"""The web adapter of the chat turn engine (agent/chat/turn.py).

The engine is synchronous and reports progress through an `emit` callback; this module runs it on a worker thread, hands
its events to the SSE stream, and supplies the web flavour of tool execution: a tool that needs approval cannot block on
a terminal prompt (the approver may be on another device by the time it matters), so it is saved as a pending approval,
`awaiting_approval` is emitted and the turn suspends; `resolve_pending_approval` runs it later from the approvals route.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.chat import gates
from agent.chat.executor import WEB_STYLE, SuspendApprover, ToolExecutor, run_approved_call
from agent.chat.helpers import DENIED_RESULT
from agent.chat.turn import ApprovalNeeded, TurnGates, grounding_events, run_turn, tool_message_pair
from agent.memory.store import Store
from agent.tools.registry import ToolRegistry
from agent.web.console_shim import WebConsole

Emit = Callable[[dict[str, Any]], Awaitable[None]]

# kept under the old name: routes/approvals.py imports it
_tool_message_pair = tool_message_pair

__all__ = ["ApprovalNeeded", "begin_web_turn", "create_plan", "get_runtime", "make_executor", "resolve_pending_approval",
           "run_agent_turn_web"]


# ----------------------------------------------------------------------------------- per-session web turn runtime

WEB_GATE_DEFAULTS = {"plan": True, "goal": True, "changes": True, "verify": True, "step_extension": False}
"""`chat.gates.*` for the web: all of these change no permission (verify only asks for its check command like any other
command, and only when a project check is detected). `step_extension` stays off: asking "continue?" needs an event the
GUI does not have yet, so the web still stops at the step limit."""

_MAX_RUNTIMES = 200


def web_gate(config: dict[str, Any], name: str) -> bool:
    return gates.gate_enabled(config, name, WEB_GATE_DEFAULTS[name])


@dataclass
class WebTurnRuntime:
    """What a web turn keeps across a suspension for approval (the turn resumes on another request, maybe on another
    device): the shared gates' state, the change log, edit events produced while the approved call ran, and - when the
    suspended call was the verify command - its result. In memory only: after a server restart a pending approval still
    resumes, just without the gate state (verify, goal count, change log)."""

    tracker: Any = None                                   # gates.ChangeTracker (private to this turn)
    verify_gate: Any = None
    goal_state: dict[str, Any] | None = None
    plan_holder: Any = None
    pending_events: list[dict[str, Any]] = field(default_factory=list)
    verify_result: tuple[str, str] | None = None          # (command, result) of an approved verify command
    prepared: bool = False                                # created by begin_web_turn, not yet picked up by its run


_RUNTIMES: dict[int, WebTurnRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_runtime(session_id: int) -> WebTurnRuntime | None:
    with _RUNTIMES_LOCK:
        return _RUNTIMES.get(session_id)


def _drop_runtime(session_id: int) -> None:
    with _RUNTIMES_LOCK:
        _RUNTIMES.pop(session_id, None)


def begin_web_turn(config: dict[str, Any], session_id: int, user_input: str, *, reset: bool = True) -> WebTurnRuntime:
    """Start the runtime of a NEW turn (the send-message route calls this before it builds the registry, so the plan
    tool can share the plan holder). With `reset` it also clears the process-wide taint / evidence singletons (never on
    a resume: what the web said earlier in the turn is still what the turn read)."""
    from agent import plan as planmod

    if reset:
        gates.begin_turn()
    registry_flag = {name: web_gate(config, name) for name in WEB_GATE_DEFAULTS}
    runtime = WebTurnRuntime(
        tracker=gates.ChangeTracker.private(checkpoints=False) if registry_flag["changes"] else None,
        goal_state={"total_calls": 0, "every_n": config.get("goal_restatement_every_n_actions", 10),
                    "original_task": user_input} if registry_flag["goal"] else None,
        plan_holder=planmod.Holder() if registry_flag["plan"] else None,
        prepared=True,
    )
    with _RUNTIMES_LOCK:
        if len(_RUNTIMES) >= _MAX_RUNTIMES:
            _RUNTIMES.pop(next(iter(_RUNTIMES)))
        _RUNTIMES[session_id] = runtime
    return runtime


def create_plan(runtime: WebTurnRuntime, client: Any, config: dict[str, Any], user_input: str,
                messages: list[dict[str, Any]], emit: Callable[[dict[str, Any]], None]) -> list[dict[str, Any]]:
    """Big build task: write the acceptance list (one extra model call) and put it in front of the user's message.
    Blocking: run it on a worker thread. Returns the (possibly extended) messages."""
    from agent import plan as planmod
    from agent.tui import _insert_before_last_user

    if runtime.plan_holder is None or not gates.plan_creation_enabled(user_input, config):
        return messages
    emit({"type": "status", "state": "planning", "label": "planning", "detail": "", "since": time.strftime("%Y-%m-%dT%H:%M:%S")})
    new_plan = planmod.generate(client, config, user_input)
    if new_plan is None:
        return messages
    runtime.plan_holder.plan = new_plan
    emit({"type": "plan", "confirmed": 0, "total": len(new_plan.items), "state": "created",
          "items": [{"id": i.id, "text": i.text, "done": i.done} for i in new_plan.items]})
    return _insert_before_last_user(messages, new_plan.system_message())


def make_executor(
    registry: ToolRegistry, store: Store, emit: Callable[[dict[str, Any]], None], config: dict[str, Any], session_id: int,
    tracker: Any = None,
) -> Callable[[str, dict[str, Any], dict[str, Any]], str]:
    """Classify and run (or defer) one tool call: SAFE and BLOCKED resolve now, CAUTION/DANGEROUS persist a pending
    approval carrying the turn state, emit `awaiting_approval` and raise ApprovalNeeded. This is the shared executor
    (agent.chat.executor) with the suspend/resume approval strategy: it never blocks a thread, and it never skips an
    approval that was asked for before (folder access grants are a terminal-only shortcut)."""

    def event(kind: str, /, **data: Any) -> None:
        emit({"type": kind, **data})

    return ToolExecutor(registry, store, config, event, SuspendApprover(store, session_id, event), style=WEB_STYLE,
                        tracker=tracker)


def resolve_pending_approval(
    registry: ToolRegistry, store: Store, config: dict[str, Any], approval_id: int, approve: bool
) -> tuple[str, dict[str, Any]] | None:
    """Run (or deny) a previously-persisted tool call. Returns (result, turn_state) to resume the turn with, or None if
    the approval id doesn't exist or was already decided.

    An approved call runs through the same shared code as every other call (change tracking, edit events, audit, the
    on-call hook). The turn's change log and verify gate live in the session's WebTurnRuntime; the `edit` events cannot
    be emitted from here (no stream yet) and are queued for the resumed turn."""
    row = store.get_pending_approval(approval_id)
    if row is None or row["status"] != "pending":
        return None

    name = row["tool_name"]
    arguments = row["arguments"]
    is_shell = name == "run_command"  # matches tui.py's dynamic-risk shell tool name
    audit_kind = "shell" if is_shell else "tool"
    command = str(arguments.get("command", ""))
    detail = command if is_shell else json.dumps({"name": name, "arguments": arguments})
    runtime = get_runtime(row["session_id"])
    verify_command = ((row["turn_state"] or {}).get("verify") or {}).get("command")

    if not approve:
        store.decide_pending_approval(approval_id, "denied")
        store.add_audit_log(kind=audit_kind, detail=detail, approved=0, result=DENIED_RESULT)
        if runtime is not None and verify_command:
            runtime.verify_result = (verify_command, DENIED_RESULT)
        return DENIED_RESULT, row["turn_state"]

    tool = registry.get(name)
    if tool is None:
        result = f"Error: no such tool '{name}'."
        store.add_audit_log(kind=audit_kind, detail=detail, approved=1, result=result)
    else:
        queued: list[dict[str, Any]] = []
        result, _ = run_approved_call(
            registry, store, tool, name, arguments, tracker=runtime.tracker if runtime is not None else None,
            event=lambda kind, /, **data: queued.append({"type": kind, **data}), audit_kind=audit_kind, detail=detail)
        if runtime is not None:
            runtime.pending_events.extend(queued)
            gates.note_clean_edit(runtime.verify_gate, {"name": name, "arguments": arguments}, result)
            if verify_command:
                runtime.verify_result = (verify_command, result)
    store.decide_pending_approval(approval_id, "approved")
    return result, row["turn_state"]


async def run_agent_turn_web(
    client: Any, config: dict[str, Any], store: Store, session_id: int, registry: ToolRegistry,
    messages: list[dict[str, Any]], persona_name: str, emit: Emit, tools_called: set[str] | None = None,
    start_iteration: int = 0,
) -> tuple[str | None, set[str]]:
    """Run the shared turn engine on a worker thread and stream its events through `emit`.

    Returns (final reply or None if the turn suspended on an approval, was stopped, or hit the step limit, the names of
    every tool called so far). If the HTTP request is cancelled (tab closed) the generation is stopped.

    The shared gates (plan nudge, goal restatement, change tracking, verify) are on by default and follow
    `chat.gates.*`; their state survives a suspension in the session's WebTurnRuntime."""
    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def sync_emit(event: dict[str, Any]) -> None:
        try:
            asyncio.run_coroutine_threadsafe(emit(event), loop).result(timeout=30)
        except Exception:  # noqa: BLE001 - a dead stream must not crash the worker; `stop` ends it
            stop.set()

    resuming = start_iteration > 0
    runtime = get_runtime(session_id)
    if runtime is None or (not resuming and not runtime.prepared):
        runtime = begin_web_turn(config, session_id, _last_user_text(messages), reset=not resuming)   # no route prepared it
    runtime.prepared = False
    if runtime.verify_gate is None and not resuming and web_gate(config, "verify"):
        runtime.verify_gate = gates.make_verify_gate(config, registry)
    if resuming:
        for queued in runtime.pending_events:
            await emit(queued)                              # the `edit` events of the call that was just approved
        runtime.pending_events.clear()
        if runtime.verify_gate is not None and runtime.verify_result is not None:
            command, result = runtime.verify_result
            runtime.verify_result = None
            verdicts: list[dict[str, Any]] = []                # (we are on the loop thread: collect, then emit)
            messages = gates.finish_verification(runtime.verify_gate, command, result, messages,
                                                 event=lambda kind, /, **d: verdicts.append({"type": kind, **d})) or messages
            for queued in verdicts:
                await emit(queued)

    console = WebConsole(emit, loop)
    execute = make_executor(registry, store, sync_emit, config, session_id, tracker=runtime.tracker)
    run_gates = TurnGates(flavor="web", verify=runtime.verify_gate, plan=runtime.plan_holder, goal_state=runtime.goal_state,
                          tracker=runtime.tracker)
    try:
        outcome = await asyncio.to_thread(
            run_turn, client, config, store, session_id, registry, messages, sync_emit, execute,
            console=console, tools_called=tools_called, start_iteration=start_iteration, stop=stop, finalize=grounding_events,
            gates=run_gates,
        )
    except asyncio.CancelledError:
        stop.set()
        _drop_runtime(session_id)
        raise
    except httpx.HTTPError as exc:                          # the model server is down or refused: tell the GUI, do not just cut the stream
        _drop_runtime(session_id)
        unreachable = not isinstance(exc, httpx.HTTPStatusError)
        await emit({"type": "error", "code": "model_unreachable" if unreachable else "model_error",
                    "message": "The model server isn’t reachable." if unreachable else f"The model server answered with an error: {exc}"})
        return None, set(tools_called or ())
    if not outcome.suspended:
        _drop_runtime(session_id)
    return outcome.reply, outcome.tools_called


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""
