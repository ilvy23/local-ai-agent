"""One user turn: model call -> tool calls -> ... -> final reply, as a stream of events.

The web GUI needs exactly what the terminal loop (tui._run_agent_turn_core) does - streamed text, tool calls through
the risk gate, stuck detection, context compaction - but reports progress as JSON events instead of printing. This
module is that loop without any terminal code. A front end supplies:

  emit(event)                      called with every event dict (token, thinking, status, tool_call, tool_result,
                                   context, done, error ...; shapes in docs/design/02-event-component-contract.md)
  execute(name, arguments, state)  runs one tool call and returns its result text; it may raise ApprovalNeeded to
                                   suspend the turn until a person decides (the web executor does that)

Optional `gates=TurnGates(...)` switch on the stateful features of the terminal loop for any front end (verify gate,
plan nudge, goal restatement, step extension, change tracking with the closing `changes` event, forced compaction):
see agent/chat/gates.py and docs/design/turn-convergence.md. Tool execution lives in agent/chat/executor.py (one
executor, the approval step is pluggable: blocking for the terminal, suspend/resume for the web).

Reused unchanged from agent.chat.helpers (formerly tui.py): tool-call extraction, result framing, the stuck watcher reaction. The model client
interface is the engine seam in llm.py (chat_with_tools / chat_with_tools_stream); nothing here names Ollama.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from agent import autonomy
from agent.chat import gates as gatelib
from agent.chat.compaction import compact_context
from agent.chat.helpers import (
    DENIAL_FRAMING,
    DENIED_RESULT,
    _extract_tool_call,
    _format_tool_call,
    _frame_tool_result,
    _num_ctx,
    _react_to_stuck,
    _strip_thinking_artifacts,
)
from agent.llm import GenerationAborted

Emit = Callable[[dict[str, Any]], None]
Execute = Callable[[str, dict[str, Any], dict[str, Any]], str]

THINKING_PULSE_S = 1.5
FAKE_STREAM_CHUNK = 24


# Wording the model gets after a denial - one copy for every front end (agent.chat.helpers.DENIAL_FRAMING).


def tool_message_pair(call_name: str, arguments: dict[str, Any], result: str) -> tuple[dict, dict]:
    """(assistant call message, tool result message) for one resolved tool call, framed like the terminal loop does."""
    call_msg = {"role": "assistant", "content": json.dumps({"tool": call_name, "arguments": arguments})}
    return call_msg, {"role": "tool", "content": DENIAL_FRAMING if result == DENIED_RESULT else _frame_tool_result(call_name, result)}


def plain_description(name: str, arguments: dict[str, Any]) -> str:
    """One readable line for an approval card: `edit rps/game.py +3 −1` for file edits, the command for a shell call."""
    from agent import changes

    path = str(arguments.get("path", ""))
    if name == "send_album":
        from agent.tools.send_album import describe
        return describe(arguments)
    if name in changes.TRACKED_TOOLS and path:
        if name == "write_file":
            before = changes.read_text(path)
            added, removed = changes.line_stats(before, str(arguments.get("content", "")))
            verb = "create" if before is None else "overwrite"
        else:
            added, removed = len(str(arguments.get("replace", "")).splitlines()), len(str(arguments.get("search", "")).splitlines())
            verb = "edit"
        return f"{verb} {changes.display_path(path)} +{added} −{removed}"
    command = str(arguments.get("command", ""))
    return " ".join(command.split())[:400] if command else _format_tool_call(name, arguments)


def grounding_events(answer: str) -> list[dict[str, Any]]:
    """After a web answer: check its figures against the sources it cites and return a `verdict` event that carries the
    per-claim result (`claims`, `warnings`) so the GUI can mark statements no source supports."""
    from agent.research import evidence, verdict

    last = evidence.last_search()
    items = [i for i in last["items"] if i.get("status") in ("ok", "cached") and i.get("passage")]
    if not items or not last["verdict"]:
        return []
    checked = verdict.ground(answer, [{"n": i["n"], "text": i["passage"]} for i in items])
    return [{"type": "verdict", **last["verdict"], "claims": checked["claims"], "warnings": checked["warnings"],
             "stray_figures": checked["stray_figures"]}]


class ApprovalNeeded(Exception):
    """A tool call needs a person's yes/no: the turn stops here and resumes after the decision."""

    def __init__(self, approval_id: int) -> None:
        self.approval_id = approval_id
        super().__init__(f"awaiting approval #{approval_id}")


@dataclass
class TurnOutcome:
    reply: str | None
    tools_called: set[str] = field(default_factory=set)
    suspended: bool = False
    aborted: bool = False
    partial: str = ""                  # text produced before an abort (the terminal keeps it as the interrupted reply)


@dataclass
class TurnGates:
    """The optional gates and front-end details of a turn. `run_turn(..., gates=None)` is the plain engine the web
    started with; everything here is opt-in, so existing callers behave as before.

    flavor            "web" (default) or "terminal": cosmetics of the events (status label/detail, thinking pulses as
                      `thinking` events vs `status`, the reply as one `token` event vs 24-char chunks, the step-limit text)
    verify            a verify.VerifyGate (gates.make_verify_gate) - run the project check before accepting "done"
    project_hook      called (check command, edited file) after a verify pass (project memory)
    plan              a plan.Holder - nudge the model back to unconfirmed plan items
    goal_state        {total_calls, every_n, original_task} - restate the task every N tool calls
    ask_more          callback(done_steps) -> bool: extend the step budget (None = hard stop at the limit)
    tracker           a gates.ChangeTracker - `changes` event at the end, undo bookkeeping (the executor emits `edit`)
    compact           object with take() -> bool: a forced (/compact) pass at the next tool-step boundary
    capture_thinking  emit the reasoning text as `thinking` events
    always_done       terminal contract: every ending (stop, step limit, exception) emits `changes` + `done`
    """

    flavor: str = "web"
    verify: Any = None
    project_hook: Callable[[str, str], None] | None = None
    plan: Any = None
    goal_state: dict[str, Any] | None = None
    ask_more: Callable[[int], bool] | None = None
    tracker: Any = None
    compact: Any = None
    capture_thinking: bool = False
    always_done: bool = False


class EventSink:
    """`event(kind, **data)` over a dict emitter - the shape the gates, the stuck reaction and the executor speak."""

    def __init__(self, emit: Emit) -> None:
        self._emit = emit

    def event(self, kind: str, /, **data: Any) -> None:
        self._emit({"type": kind, **data})


class _NullConsole:
    def print(self, *args: Any, **kwargs: Any) -> None:
        return None


class _TokenGate:
    """Streams reply text unless it is (or looks like) the model's text-form tool call, which must never be shown.

    The first characters decide, as in liveview.LiveReply: JSON-ish -> stay silent; anything else -> stream."""

    def __init__(self, emit: Emit) -> None:
        from agent.liveview import _DECIDE_AFTER, _TOOLISH_RE

        self._emit, self._re, self._after = emit, _TOOLISH_RE, _DECIDE_AFTER
        self.state, self._buf, self.text = "pending", "", ""

    def feed(self, piece: str) -> None:
        if not piece:
            return
        self.text += piece
        if self.state == "tool":
            return
        if self.state == "live":
            self._emit({"type": "token", "text": piece})
            return
        self._buf += piece
        head = self._buf.lstrip()
        if not head:
            return
        if self._re.match(head):
            self.state = "tool"
        elif len(head) >= self._after or "\n" in head or (head.startswith("`") and len(head) >= 8):
            self.state = "live"
            self._emit({"type": "token", "text": self._buf})
            self._buf = ""


def _fake_stream(text: str, emit: Emit) -> None:
    for i in range(0, len(text), FAKE_STREAM_CHUNK):
        emit({"type": "token", "text": text[i:i + FAKE_STREAM_CHUNK]})


def _context_event(message: dict[str, Any], num_ctx: int, compacted: bool = False) -> dict[str, Any]:
    used = int(message.get("_prompt_eval_count", 0) or 0) + int(message.get("_eval_count", 0) or 0)
    eval_ns = int(message.get("_eval_ns", 0) or 0)
    rate = round(int(message.get("_eval_count", 0) or 0) / (eval_ns / 1e9), 1) if eval_ns else None
    return {"type": "context", "used": used, "limit": num_ctx, "percent": round(100 * used / num_ctx) if num_ctx else 0,
            "tokens_per_s": rate, "compacted": compacted}


def prompt_estimate(messages: list[dict[str, Any]], tools: Any = None) -> int:
    """Rough token count of what is about to be sent, for the 'reading the prompt' hint: message text at ~4 chars a
    token; the tool schemas are verbose JSON that tokenizes far denser (measured on a real 22-tool turn: ~12 chars a token)."""
    try:
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        if tools:
            return chars // 4 + len(json.dumps(tools, default=str)) // 12
        return chars // 4
    except Exception:  # noqa: BLE001 - a hint, never a failure
        return 0


def _model_call(client: Any, messages: list[dict], model: str, tools: list[dict], num_ctx: int, emit: Emit,
                gate: _TokenGate, stop: threading.Event, *, terminal: bool = False, capture_thinking: bool = False) -> dict[str, Any]:
    from agent.llm import is_deepseek_r1

    stream = getattr(client, "chat_with_tools_stream", None)
    if not callable(stream) or is_deepseek_r1(model):          # (R1's <think> block is part of the content: never stream it)
        return client.chat_with_tools(messages=messages, model=model, tools=tools, num_ctx=num_ctx)
    state = {"chars": 0, "last": 0.0}

    def on_content(piece: str) -> None:
        if stop.is_set():
            raise GenerationAborted("stopped")
        gate.feed(piece)

    def on_thinking(piece: str) -> None:
        if stop.is_set():
            raise GenerationAborted("stopped")
        state["chars"] += len(piece)
        if capture_thinking:
            emit({"type": "thinking", "text": piece})
        now = time.monotonic()
        if now - state["last"] >= THINKING_PULSE_S:
            state["last"] = now
            if terminal:
                emit({"type": "status", "state": "thinking", "label": "thinking", "detail": "", "tokens": state["chars"] // 4})
            else:
                emit({"type": "thinking", "tokens": state["chars"] // 4})

    extra: dict[str, Any] = {}
    params = inspect.signature(stream).parameters
    if "on_thinking" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        extra["on_thinking"] = on_thinking
    if "cancel" in params:
        extra["cancel"] = stop                                # Esc ends the call at once, even before the first token
    try:
        return stream(messages=messages, model=model, tools=tools, num_ctx=num_ctx, on_content=on_content, **extra)
    except GenerationAborted as aborted:                     # a client that lets the abort through: keep what arrived
        return {"role": "assistant", "content": gate.text, "_aborted": str(aborted) or "stopped"}
    except (httpx.TransportError, ValueError):
        if gate.state == "live":
            raise
        return client.chat_with_tools(messages=messages, model=model, tools=tools, num_ctx=num_ctx)


STEP_LIMIT_WEB = "Too many tool calls in a row. Say \"continue\" to go on, or rephrase."
STEP_LIMIT_TERMINAL = ("Stopped: step limit reached before the task was done. Say \"continue\" to go on "
                       "(the work so far is kept), or raise tools.max_iterations.")


def run_turn(
    client: Any, config: dict[str, Any], store: Any, session_id: int, registry: Any, messages: list[dict[str, Any]],
    emit: Emit, execute: Execute, *, console: Any = None, tools_called: set[str] | None = None, start_iteration: int = 0,
    stop: threading.Event | None = None, finalize: Callable[[str], list[dict[str, Any]]] | None = None,
    gates: TurnGates | None = None,
) -> TurnOutcome:
    """Run the tool loop for one user turn. Returns the outcome; a suspended turn (ApprovalNeeded) has already emitted
    its `awaiting_approval` event through `execute`.

    `gates` (TurnGates) switches on the shared gates: verify, plan nudge, goal restatement, step extension, change
    tracking (+ the closing `changes` event), forced compaction and the terminal's closing contract. Without it the
    behaviour is the plain web engine."""
    stop = stop or threading.Event()
    g = gates
    terminal = g is not None and g.flavor == "terminal"
    sink = EventSink(emit)
    model = config["models"]["chat"]
    tools_schema = registry.to_ollama_schema()
    max_iterations = config["tools"]["max_iterations"]
    num_ctx = _num_ctx(config, model)
    tools_called = set(tools_called or ())
    watcher = (autonomy.StuckWatcher(max_lookups=int((config.get("autonomy") or {}).get("max_lookups_per_turn", 3)))
               if autonomy.enabled(config) and registry.get("web_search") and (console is not None or g is not None) else None)
    stuck_console = console if console is not None else _NullConsole()
    tracker = g.tracker if g is not None else None
    verify_gate = g.verify if g is not None else None
    goal = gatelib.GoalReminder(g.goal_state if g is not None else None)
    steps = gatelib.StepBudget(max_iterations, g.ask_more if g is not None else None)
    if g is not None and start_iteration == 0:
        if tracker is not None:
            tracker.begin()
        gatelib.PlanGate.reset(g.plan)

    def close(reply: str, *, interrupted: bool = False, failed: bool = False) -> None:
        """The closing events: `changes` (if tracked) then `done`."""
        if tracker is not None:
            tracker.finish()
            change_event = tracker.change_event()
            if change_event is not None:
                sink.event("changes", **change_event)
            if terminal:
                tracker.clear()
        if terminal:
            sink.event("done", reply=reply, interrupted=interrupted, tools=sorted(tools_called), failed=failed)
        else:
            sink.event("done", reply=reply)                     # the web's contract: just the reply

    def abort(partial: str = "") -> TurnOutcome:
        if g is not None and g.always_done:
            close(partial, interrupted=True)
        return TurnOutcome(None, tools_called, aborted=True, partial=partial)

    try:
        iteration = start_iteration
        while True:
            if stop.is_set():
                return abort()
            if not steps.may_continue(iteration):
                break
            index, iteration = iteration, iteration + 1
            if terminal:
                sink.event("status", state="thinking", label="waiting for model", detail=model, since=time.time(), step=index + 1,
                           prompt_tokens=prompt_estimate(messages, tools_schema))
            else:
                sink.event("status", state="thinking", label="thinking", detail="", since=time.strftime("%Y-%m-%dT%H:%M:%S"))
            gate = _TokenGate(emit)
            message = _model_call(client, messages, model, tools_schema, num_ctx, emit, gate, stop,
                                  terminal=terminal, capture_thinking=bool(g and g.capture_thinking))
            emit(_context_event(message, num_ctx))
            call = _extract_tool_call(message)
            if message.get("_aborted") or (stop.is_set() and gate.state != "live" and call is not None):
                partial = "" if gate.state == "tool" else _strip_thinking_artifacts(str(message.get("content") or gate.text))
                return abort(partial)

            state = {"messages": messages, "tools_called": sorted(tools_called), "iteration": index}
            if call is None:
                content = _strip_thinking_artifacts(message.get("content", ""))
                if not content.strip() and tools_called:
                    continue                                    # a blank reply mid-task is a stochastic blip: re-roll
                if verify_gate is not None and content.strip() and verify_gate.due():
                    retry = gatelib.run_verification(
                        verify_gate,
                        lambda command: execute("run_command", {"command": command}, {**state, "verify": {"command": command}}),
                        messages, content, event=sink.event, on_pass=g.project_hook if g is not None else None)
                    if retry is not None:
                        messages = retry
                        continue
                decision = gatelib.PlanGate.check(g.plan if g is not None else None, content)
                if decision is not None:
                    sink.event("plan", **decision.event)
                    if decision.action == "nudge":
                        messages = decision.retry_messages(messages, content)
                        continue
                if gate.state != "live":                        # nothing was streamed (no streaming client, or JSON-looking text)
                    if terminal:
                        if content:
                            sink.event("token", text=content)
                    else:
                        _fake_stream(content, emit)
                for extra in (finalize(content) if finalize else []) or []:     # e.g. the answer-vs-sources check, before the end
                    emit(extra)
                if g is not None:
                    close(content)
                else:
                    emit({"type": "done", "reply": content})
                return TurnOutcome(content, tools_called)

            if stop.is_set():
                return abort()                                  # a tool call was announced but never ran
            result = execute(call["name"], call["arguments"], state)
            assistant_msg, tool_msg = tool_message_pair(call["name"], call["arguments"], result)
            store.add_message(session_id, "assistant", assistant_msg["content"])
            store.add_message(session_id, "tool", result)
            messages = [*messages, assistant_msg, tool_msg]
            tools_called.add(call["name"])
            gatelib.note_clean_edit(verify_gate, call, result)
            if watcher is not None:
                messages = _react_to_stuck(watcher, registry, stuck_console, messages, call, result,
                                           denied=result == DENIED_RESULT, ctl=sink)
            messages = compact_context(client, config, messages, message, num_ctx, event=sink.event,
                                       force=bool(g is not None and g.compact is not None and g.compact.take()))
            reminder = goal.tick()                              # every N tool calls: remind the model of the original task
            if reminder is not None:
                sink.event("log", text=reminder.log_text)
                messages = [*messages, reminder.message]

        sink.event("error", message=STEP_LIMIT_TERMINAL if terminal else STEP_LIMIT_WEB, code="max_steps")
        if g is not None and g.always_done:
            close("")
        return TurnOutcome(None, tools_called)
    except ApprovalNeeded:
        return TurnOutcome(None, tools_called, suspended=True)
    except BaseException as exc:
        if g is not None and g.always_done:
            from agent import friendly

            sink.event("error", **friendly.error_fields(exc, model))
            close("", failed=True)
        raise
