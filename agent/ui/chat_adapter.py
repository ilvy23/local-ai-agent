"""Chat-turn adapter: the terminal chat's turn loop (agent.tui) seen as an event stream.

Two layers live here:

* Control objects the turn core in `agent.tui` consumes (`TurnControl`, `CompactFlag`, `ApprovalRequest`,
  `resolve_approval`). This part imports nothing heavy so `agent.tui` can import it at module level.
* `ChatState` + `run_chat_turn`: everything `tui.run_repl` keeps in locals, and the pre/post-turn steps around
  the turn (context building, plan, forced /web, persistence), so a dock/worker can run one chat turn in
  a thread with a single call and no console prompts. The turn itself runs on the shared engine
  (`agent.chat.turn.run_turn`, via `run_agent_turn_engine` below); the classic `chat --classic` loop
  (`tui._run_agent_turn`) is the legacy path, kept for parity (tests/test_chat_parity.py).

Event kinds emitted (all `UiEvent(source="chat")`, payload in `data`): status, token, thinking, tool_call,
tool_result, edit, changes, sources, verdict, plan, verify, stuck, context, log, awaiting_approval, error, done.
See the module docstring of each emitter in agent/tui.py (`_execute_tool`, `_run_agent_turn_core`) and
`docs` in the report; payload shapes are pinned by tests/test_ui_chat_adapter.py.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.ui.events import UiEvent

logger = logging.getLogger(__name__)

TOOL_RESULT_CAP = 20_000   # characters of a tool result carried in a tool_result event

Emit = Callable[[UiEvent], None]


class CompactFlag:
    """Thread-safe "compact at the next tool-step boundary" request (the /compact command sets it)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    @property
    def pending(self) -> bool:
        return self._event.is_set()

    def take(self) -> bool:
        """True exactly once per request (and clears it)."""
        if self._event.is_set():
            self._event.clear()
            return True
        return False


@dataclass
class ApprovalRequest:
    """What an `approve_fn` is asked about. The function answers True/False, or a string."""

    name: str
    level: Any                       # agent.safety.RiskLevel
    risk: str                        # "CAUTION" | "DANGEROUS"
    description: str                 # one plain line (no markup)
    arguments: dict[str, Any] = field(default_factory=dict)
    from_web: bool = False           # the text of the call was copied from a web page
    diff: str = ""                   # unified diff for file edits, else ""
    needs_typed_yes: bool = False    # DANGEROUS: only the exact string "yes" approves


def resolve_approval(level: Any, answer: Any) -> bool:
    """Same rules as the console prompt. CAUTION: True or y/yes (any case). DANGEROUS: ONLY the exact,
    case-sensitive string "yes" (a bool True is not enough - the typed word is the safety step)."""
    from agent.safety import RiskLevel

    if level is RiskLevel.DANGEROUS:
        return isinstance(answer, str) and answer.strip() == "yes"
    if isinstance(answer, bool):
        return answer
    return isinstance(answer, str) and answer.strip().lower() in ("y", "yes")


@dataclass
class TurnControl:
    """Everything optional a turn can be given. Passing none of it keeps the classic terminal behaviour."""

    emit: Emit | None = None
    stop: threading.Event | None = None
    approve_fn: Callable[[ApprovalRequest], bool | str] | None = None
    suppress_print: bool = False
    capture_thinking: bool = False       # thinking TEXT is only emitted when True (never stored anywhere)
    compact: CompactFlag | None = None
    source: str = "chat"
    turn: int | None = None
    interrupted: bool = False            # set by the turn core when `stop` ended it
    partial: str = ""                    # reply text produced before the interruption
    sources_sig: Any = None              # internal: last web search already reported

    @classmethod
    def build(cls, **kw: Any) -> "TurnControl | None":
        """A control for the given options, or None when nothing was asked for (classic path)."""
        ctl = cls(**{k: v for k, v in kw.items() if k in cls.__dataclass_fields__})
        if (ctl.emit is None and ctl.stop is None and ctl.approve_fn is None and not ctl.suppress_print
                and ctl.compact is None and not ctl.capture_thinking):
            return None
        return ctl

    def event(self, kind: str, /, **data: Any) -> None:
        """Emit one event; a broken subscriber never breaks the turn."""
        if self.emit is None:
            return
        try:
            self.emit(UiEvent(kind=kind, data=data, source=self.source, turn=self.turn))
        except Exception:  # noqa: BLE001
            pass

    def stopped(self) -> bool:
        return bool(self.stop is not None and self.stop.is_set())


# --------------------------------------------------------------------------- chat state + one-call turn


@dataclass
class TurnResult:
    reply: str | None                 # final reply (or the partial text when interrupted); None if nothing/cap hit
    interrupted: bool = False
    tool_names: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)   # last context numbers: used, limit, percent, tokens_per_s
    error: str = ""                   # set when the turn ended on a connection/model error
    seconds: float = 0.0


@dataclass
class ChatState:
    """The state `run_repl` keeps in locals, as one object a worker can hold across turns."""

    client: Any
    config: dict[str, Any]
    store: Any
    registry: Any
    vectors: Any = None
    console: Any = None
    session_id: int = 0
    model: str = ""
    persona_name: str = "Agent"
    goal_state: dict[str, Any] = field(default_factory=dict)
    plan_holder: Any = None
    active_project: Any = None
    project_memory: Any = None
    pending_task: str | None = None
    last_reply: str | None = None
    tool_exchanges: list[tuple] = field(default_factory=list)
    title_pending: bool = True
    compact: CompactFlag = field(default_factory=CompactFlag)
    distiller: Any = None            # agent.memory.queue.BackgroundDistiller: saves facts while the chat is idle

    @classmethod
    def from_repl_args(
        cls, client: Any, config: dict[str, Any], console: Any = None, store: Any = None,
        session_id: int | None = None, vectors: Any = None, tool_registry: Any = None,
    ) -> "ChatState":
        """Same inputs as `tui.run_repl`. Builds the registry/vectors/session like its set-up block does but
        does NOT register web/plugin/deep-research tools bound to a console (the caller can add them)."""
        from rich.console import Console

        from agent import plan as planmod
        from agent import projectmem
        from agent.memory.vectors import VectorIndex
        from agent.tools.registry import default_registry

        console = console or Console()
        registry = tool_registry if tool_registry is not None else default_registry(config)
        if store is not None and vectors is None:
            vectors = VectorIndex(store)
        resuming = session_id is not None
        if session_id is None and store is not None:
            session_id = store.create_session()
        state = cls(
            client=client, config=config, store=store, registry=registry, vectors=vectors, console=console,
            session_id=session_id or 0, model=config["models"]["chat"],
            persona_name=config["persona"]["name"], plan_holder=planmod.Holder(),
            title_pending=not resuming,
        )
        state.reset_goal()
        if store is not None:
            state.project_memory = projectmem.ProjectMemory(store.conn)
            state.active_project = projectmem.ActiveProject()
        return state

    def reset_goal(self) -> None:
        self.goal_state = {
            "total_calls": 0,
            "every_n": self.config.get("goal_restatement_every_n_actions", 10),
            "original_task": None,
        }

    def request_compact(self) -> None:
        """Thread-safe: the running (or next) turn compacts at its next tool-step boundary."""
        self.compact.request()


def compact_now(client: Any, config: dict[str, Any], messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Force one mask + work-log pass over a tool-loop message list (what `/compact` does at a step boundary).

    Only meaningful for the message list of a RUNNING turn. Between turns the history is rebuilt from the
    database by build_context, so there is nothing to compact; use `ChatState.request_compact()` instead.
    Returns (messages, {"saved_tokens": n, "before_chars": a, "after_chars": b, "compacted": 0|1})."""
    from agent.memory import worklog

    before = sum(len(str(m.get("content", ""))) for m in messages)
    masked, saved = worklog.mask_old_observations(messages)
    if saved > 0:
        messages = masked
    compacted = worklog.compact_turn_messages(client, config, messages)
    did = compacted is not None
    if did:
        messages = compacted
    after = sum(len(str(m.get("content", ""))) for m in messages)
    return messages, {"saved_tokens": saved // 4, "before_chars": before, "after_chars": after, "compacted": int(did)}


def run_chat_turn(
    state: ChatState, prompt: str, emit: Emit | None = None, stop: threading.Event | None = None,
    approve_fn: Callable[[ApprovalRequest], bool | str] | None = None, *,
    suppress_print: bool = True, capture_thinking: bool = False, turn: int | None = None,
) -> TurnResult:
    """One user prompt, start to finish, without touching stdin: build context, optional plan, forced /web,
    the tool loop (events via `emit`), then persistence. Never raises for HTTP/model errors (they become an
    `error` event and `TurnResult.error`); a `stop` ends it cleanly with `interrupted=True` and the partial
    reply kept. Slash commands are NOT handled here - the command registry/dock does that first.

    Not re-entrant: agent.tui keeps per-turn singletons (_CHANGES, taint, turn_evidence), so run ONE chat turn
    at a time per process."""
    import os
    import time

    import httpx

    from agent import plan as planmod
    from agent import tui
    from agent.chat import gates
    from agent.memory.recall import build_context
    from agent.research import evidence as turn_evidence
    from agent.research import taint

    t0 = time.monotonic()
    ctl = TurnControl(emit=emit, stop=stop, approve_fn=approve_fn, suppress_print=suppress_print,
                      capture_thinking=capture_thinking, compact=state.compact, turn=turn)
    console = state.console
    s = state
    user_input, web_forced = tui._split_web_trigger(prompt.strip())
    if web_forced and not user_input:
        ctl.event("error", message="Add something to search for before /web.", code="empty_web")
        ctl.event("done", reply="", interrupted=False, tools=[], error="empty_web")
        return TurnResult(None, error="empty_web")
    if s.goal_state.get("original_task") is None:
        s.goal_state["original_task"] = user_input

    taint.clear()
    turn_evidence.clear()
    ctl.event("status", state="memory", label="recalling", detail="", since=time.time())
    tool_names = [t.name for t in s.registry.list()]
    messages = build_context(s.store, s.vectors, s.client, s.config, s.session_id, user_input, tool_names=tool_names)
    user_msg_id = s.store.add_message(s.session_id, "user", user_input)

    if gates.gate_enabled(s.config, "plan") and gates.plan_creation_enabled(user_input, s.config):
        ctl.event("status", state="planning", label="planning", detail="", since=time.time())
        new_plan = planmod.generate(s.client, s.config, user_input)
        if new_plan is not None:
            s.plan_holder.plan = new_plan
            ctl.event("plan", confirmed=0, total=len(new_plan.items), state="created",
                      items=[{"id": i.id, "text": i.text, "done": i.done} for i in new_plan.items])
    extra: list[str] = []
    if s.plan_holder.plan is not None:
        extra.append(s.plan_holder.plan.system_message())
    if s.active_project is not None and s.active_project.key and s.project_memory is not None:
        block = s.project_memory.system_message(s.active_project.key)
        if block:
            extra.append(block)
    for block in extra:
        messages = tui._insert_before_last_user(messages, block)

    if web_forced:
        web = s.registry.get("web_search")
        ctl.event("status", state="searching", label="searching", detail=user_input, since=time.time())
        query = tui._rewrite_search_query(user_input, s.client, s.config)
        if query != user_input:
            ctl.event("log", text=f"searching for: {query}")
        results = web.handler(query=query) if web else "web search unavailable"
        results = results[:20000]
        _emit_sources(ctl)
        s.store.add_message(s.session_id, "tool", f"Web search results for {query!r}:\n{results}")
        messages = [*messages, {"role": "system", "content": tui._WEB_INJECT + results}]
    elif (s.last_reply and s.registry.get("web_search") and tui._looks_like_disagreement(user_input)
          and tui._is_factual_correction(s.client, s.config, s.last_reply, user_input)):
        messages = [*messages, {"role": "system", "content": tui._CORRECTION_NUDGE}]

    usage: dict[str, Any] = {}

    def _tap(event: UiEvent) -> None:
        if event.kind == "context":
            usage.update(event.data)
        if emit is not None:
            emit(event)

    ctl.emit = _tap
    problem = getattr(s.distiller, "take_problem", lambda: "")() if s.distiller is not None else ""
    if problem and isinstance(problem, str):
        ctl.event("log", text=problem)
    try:
        reply, tools_used = run_agent_turn_engine(
            s.client, s.config, s.store, s.session_id, s.registry, messages, s.persona_name,
            goal_state=s.goal_state, plan=s.plan_holder, ctl=ctl,
            console=console if not suppress_print else None,
        )
        if tui._LAST_EDITED and s.active_project is not None:
            s.active_project.set_from(tui._LAST_EDITED[0])
    except httpx.HTTPError as exc:
        from agent import friendly

        line, hint = friendly.explain(exc, s.model)
        text = line
        logger.warning("chat turn failed: %r", exc)
        if not getattr(exc, "_nightage_reported", False):
            ctl.event("error", message=line, code="http", next=hint)
        ctl.event("done", reply="", interrupted=False, tools=[], error="http")
        return TurnResult(None, error=str(text), usage=usage, seconds=time.monotonic() - t0)

    s.store.touch_session(s.session_id)
    if s.title_pending:
        s.store.set_session_title(s.session_id, user_input[:tui.TITLE_MAX_LEN])
        s.title_pending = False
    tui._embed_message(s.vectors, s.client, s.config, user_msg_id, user_input)

    names = sorted(tools_used)
    if ctl.interrupted:
        partial = tui._strip_thinking_artifacts(ctl.partial or reply or "")
        if partial.strip():
            s.store.add_message(s.session_id, "assistant", partial + "\n\n[interrupted by the user]")
        return TurnResult(partial or None, True, names, usage, seconds=time.monotonic() - t0)
    if reply is None:
        return TurnResult(None, False, names, usage, seconds=time.monotonic() - t0)

    reply = tui._strip_thinking_artifacts(reply)
    s.last_reply = reply
    msg_id = s.store.add_message(s.session_id, "assistant", reply)
    tui._embed_message(s.vectors, s.client, s.config, msg_id, reply)
    if (tools_used - {"remember_this"}) or web_forced:
        s.tool_exchanges.append((user_input, reply, turn_evidence.take()))
    return TurnResult(reply, False, names, usage, seconds=time.monotonic() - t0)


def run_agent_turn_engine(
    client: Any, config: dict[str, Any], store: Any, session_id: int, registry: Any, messages: list[dict[str, Any]],
    persona_name: str = "Agent", *, goal_state: dict[str, Any] | None = None, plan: Any = None,
    ctl: TurnControl | None = None, console: Any = None, ask_more: Callable[[int], bool] | None = None,
) -> tuple[str | None, set[str]]:
    """The terminal chat's turn on the shared engine (agent.chat.turn.run_turn): same arguments and return value as
    `tui._run_agent_turn` (reply or the partial text of an interrupted turn, names of the tools called), same events.

    Nothing is printed and stdin is never read: events carry everything, approvals go through `ctl.approve_fn`
    (no `approve_fn` = no, DANGEROUS needs the typed word `yes`). `console` is only used to ask for more steps when
    the budget is used up (the classic prompt) - the dock passes none, so the turn simply stops there, as before.
    The gates (verify, plan nudge, goal restatement, step extension, change tracking) follow `chat.gates.*` in the
    config, all on by default. Not re-entrant (one turn per process): the change log, undo stack, taint and
    evidence are process-wide singletons."""
    from agent import tui
    from agent.chat import gates, turn
    from agent.chat.executor import TERMINAL_STYLE, BlockingApprover, ToolExecutor

    ctl = ctl or TurnControl()
    tracker = tui._tracker() if gates.gate_enabled(config, "changes") else None
    approver = BlockingApprover(ctl.event, ctl.approve_fn)
    execute = ToolExecutor(registry, store, config, ctl.event, approver, style=TERMINAL_STYLE, tracker=tracker,
                           after_result=lambda: tui._emit_search_evidence(ctl))
    if ask_more is None and gates.gate_enabled(config, "step_extension") and console is not None:
        ask_more = lambda done: tui._ask_more_steps(console, done)   # noqa: E731 - same prompt as the classic loop
    run_gates = turn.TurnGates(
        flavor="terminal",
        verify=gates.make_verify_gate(config, registry) if gates.gate_enabled(config, "verify") else None,
        project_hook=lambda command, edited: tui._PROJECT_HOOK(command, edited) if tui._PROJECT_HOOK else None,
        plan=plan if gates.gate_enabled(config, "plan") else None,
        goal_state=goal_state if gates.gate_enabled(config, "goal") else None,
        ask_more=ask_more, tracker=tracker, compact=ctl.compact, capture_thinking=ctl.capture_thinking, always_done=True)

    def emit(event: dict[str, Any]) -> None:
        ctl.event(event["type"], **{k: v for k, v in event.items() if k != "type"})

    outcome = turn.run_turn(client, config, store, session_id, registry, messages, emit, execute, stop=ctl.stop,
                            gates=run_gates)
    if outcome.aborted:
        ctl.interrupted, ctl.partial = True, outcome.partial
        return (outcome.partial or None), outcome.tools_called
    return outcome.reply, outcome.tools_called


def _emit_sources(ctl: TurnControl) -> None:
    from agent.tui import _emit_search_evidence

    _emit_search_evidence(ctl)
