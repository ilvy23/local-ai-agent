"""The shared chat turn engine and its web adapter: events, streaming, tool calls, approvals, stopping. No model, no network."""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.chat import turn
from agent.chat.turn import ApprovalNeeded, run_turn
from agent.memory.store import Store
from agent.tools.registry import Tool, ToolRegistry
from agent.web import agent_runtime as rt

CONFIG = {"models": {"chat": "m", "background": "m"}, "tools": {"max_iterations": 5}, "autonomy": {"web_lookup": False},
          "safety": {}}


def _tool(name="echo", risk="safe", handler=None, props=None, required=None):
    return Tool(name=name, description="d", risk=risk, handler=handler or (lambda **kw: f"ran {kw}"),
                parameters={"type": "object", "properties": props or {"text": {"type": "string"}}, "required": required or []})


def _registry(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _msg(content="", calls=None, **extra):
    m = {"role": "assistant", "content": content, "_prompt_eval_count": 800, "_eval_count": 200, "_eval_ns": 5_000_000_000, **extra}
    if calls:
        m["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in calls]
    return m


class PlainClient:
    """No streaming support: chat_with_tools only."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def chat_with_tools(self, messages, model, tools, num_ctx=None, **kw):
        self.seen.append(list(messages))
        return self.replies.pop(0)


class StreamClient(PlainClient):
    """Streams each reply as small pieces through on_content / on_thinking."""

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, **kw):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if on_thinking:
            on_thinking("hmm " * 10)
        for i in range(0, len(reply.get("content", "")), 7):
            on_content(reply["content"][i:i + 7])
        return reply


def _run(client, tmp_path, *, tools=None, execute=None, messages=None, stop=None, **kw):
    events = []
    store = Store(tmp_path / "a.db")
    sid = store.create_session()
    reg = _registry(*(tools or [_tool()]))
    ex = execute or (lambda name, args, state: reg.get(name).handler(**args))
    out = run_turn(client, CONFIG, store, sid, reg, messages or [{"role": "user", "content": "hi"}], events.append, ex, stop=stop, **kw)
    return out, events, store, sid


def _types(events):
    return [e["type"] for e in events]


def test_plain_client_reply_is_faked_into_tokens_and_ends_with_done(tmp_path):
    out, events, _, _ = _run(PlainClient(_msg("Hello there, this is a reply.")), tmp_path)
    assert out.reply == "Hello there, this is a reply." and not out.suspended
    assert "".join(e["text"] for e in events if e["type"] == "token") == out.reply
    assert _types(events)[0] == "status" and _types(events)[-1] == "done" and events[-1]["reply"] == out.reply
    ctx = next(e for e in events if e["type"] == "context")
    assert ctx["used"] == 1000 and ctx["limit"] > 0 and ctx["tokens_per_s"] == 40.0


def test_streaming_client_streams_tokens_once_and_pulses_thinking(tmp_path):
    out, events, _, _ = _run(StreamClient(_msg("A streamed answer that is long enough to pass the gate.")), tmp_path)
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == out.reply                                   # not doubled by the fake-stream fallback
    assert any(e["type"] == "thinking" for e in events)


def test_text_form_tool_call_is_never_streamed_but_is_executed(tmp_path):
    call_text = '{"tool": "echo", "arguments": {"text": "x"}}'
    client = StreamClient(_msg(call_text), _msg("done now, all fine and good."))
    out, events, store, sid = _run(client, tmp_path)
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert '"tool"' not in streamed and streamed == "done now, all fine and good."
    assert out.tools_called == {"echo"}
    roles = [m["role"] for m in store.get_messages(sid)]
    assert roles == ["assistant", "tool"]


def test_structured_tool_call_runs_and_the_model_sees_the_framed_result(tmp_path):
    client = PlainClient(_msg(calls=[("echo", {"text": "hi"})]), _msg("The tool said hi."))
    out, events, store, sid = _run(client, tmp_path)
    second_prompt = client.seen[1]
    assert second_prompt[-1]["role"] == "tool" and "ran {'text': 'hi'}" in second_prompt[-1]["content"]
    assert "real result just returned by a tool" in second_prompt[-1]["content"]          # framed, like the terminal loop
    assert out.reply == "The tool said hi." and out.tools_called == {"echo"}
    assert store.get_messages(sid)[1]["content"] == "ran {'text': 'hi'}"                   # raw result stored


def test_approval_suspends_the_turn_without_done(tmp_path):
    def execute(name, args, state):
        assert state["iteration"] == 0 and state["messages"][-1]["content"] == "hi"
        raise ApprovalNeeded(7)

    out, events, _, _ = _run(PlainClient(_msg(calls=[("echo", {})])), tmp_path, execute=execute)
    assert out.suspended and out.reply is None and "done" not in _types(events)


def test_empty_reply_after_a_tool_is_retried(tmp_path):
    client = PlainClient(_msg(calls=[("echo", {})]), _msg(""), _msg("finally an answer to give."))
    out, _, _, _ = _run(client, tmp_path)
    assert out.reply == "finally an answer to give." and len(client.seen) == 3


def test_step_limit_ends_with_an_error_event(tmp_path):
    client = PlainClient(*[_msg(calls=[("echo", {})]) for _ in range(10)])
    out, events, _, _ = _run(client, tmp_path)
    assert out.reply is None and events[-1]["type"] == "error" and "Too many tool calls" in events[-1]["message"]


def test_stop_flag_aborts_before_and_during_generation(tmp_path):
    stop = threading.Event()
    stop.set()
    out, events, _, _ = _run(PlainClient(_msg("never")), tmp_path, stop=stop)
    assert out.aborted and "done" not in _types(events)

    stop2 = threading.Event()

    class Slow(StreamClient):
        def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, **kw):
            stop2.set()
            on_content("some text that arrives after the stop flag was set")        # raises GenerationAborted

    out, events, _, _ = _run(Slow(), tmp_path, stop=stop2)
    assert out.aborted and out.reply is None


# ---------------------------------------------------------------------------------------- the web executor


def _executor(tmp_path, *tools):
    events = []
    store = Store(tmp_path / "e.db")
    sid = store.create_session()
    reg = _registry(*tools)
    return rt.make_executor(reg, store, events.append, CONFIG, sid), events, store, reg, sid


def test_safe_tool_runs_and_reports(tmp_path):
    ex, events, store, _, _ = _executor(tmp_path, _tool())
    assert ex("echo", {"text": "a"}, {}) == "ran {'text': 'a'}"
    assert _types(events) == ["tool_call", "tool_result"] and events[1]["result_preview"].startswith("ran")
    assert store.get_audit_log()[0]["approved"] == 1


def test_bad_arguments_and_unknown_tools_come_back_as_text(tmp_path):
    ex, events, _, _, _ = _executor(tmp_path, _tool(props={"n": {"type": "integer"}}, required=["n"]))
    assert "n" in ex("echo", {}, {}) and not events
    assert ex("nope", {}, {}).startswith("Error: no such tool")


def test_caution_tool_is_deferred_then_resolved(tmp_path):
    ran = []
    ex, events, store, reg, sid = _executor(tmp_path, _tool("write_file", risk="caution", handler=lambda **kw: ran.append(kw) or "written",
                                                             props={"path": {"type": "string"}}))
    with pytest.raises(ApprovalNeeded) as pending:
        ex("write_file", {"path": "/tmp/x"}, {"messages": [], "tools_called": [], "iteration": 0})
    assert not ran and events[-1]["type"] == "awaiting_approval" and events[-1]["risk_level"] == "CAUTION"
    result, state = rt.resolve_pending_approval(reg, store, CONFIG, pending.value.approval_id, True)
    assert result == "written" and ran == [{"path": "/tmp/x"}] and state["iteration"] == 0
    assert rt.resolve_pending_approval(reg, store, CONFIG, pending.value.approval_id, True) is None        # only once


def test_denial_reaches_the_model_with_the_honesty_framing(tmp_path):
    ex, _, store, reg, _ = _executor(tmp_path, _tool("write_file", risk="caution"))
    with pytest.raises(ApprovalNeeded) as pending:
        ex("write_file", {}, {"messages": [], "tools_called": [], "iteration": 0})
    result, _ = rt.resolve_pending_approval(reg, store, CONFIG, pending.value.approval_id, False)
    _, tool_msg = turn.tool_message_pair("write_file", {}, result)
    assert "APPROVAL DENIED" in tool_msg["content"]


def test_whatsapp_send_is_blocked_outright(tmp_path):
    ex, events, _, _, _ = _executor(tmp_path, _tool("wa_send_message", risk="dangerous"))
    out = ex("wa_send_message", {}, {})
    assert out.startswith("BLOCKED by safety policy") and events[-1]["type"] == "tool_result"


# ---------------------------------------------------------------------------------------- the async adapter


def test_web_adapter_streams_events_in_order_and_returns_the_reply(tmp_path):
    store = Store(tmp_path / "w.db")
    sid = store.create_session()
    reg = _registry(_tool())
    got = []

    async def emit(event):
        got.append(event)

    client = StreamClient(_msg(calls=[("echo", {"text": "q"})]), _msg("All done, here is your final answer."))
    reply, tools = asyncio.run(rt.run_agent_turn_web(client, CONFIG, store, sid, reg, [{"role": "user", "content": "go"}], "Agent", emit))
    assert reply == "All done, here is your final answer." and tools == {"echo"}
    kinds = _types(got)
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("token") < kinds.index("done")


def test_web_adapter_suspends_on_approval(tmp_path):
    store = Store(tmp_path / "w.db")
    sid = store.create_session()
    reg = _registry(_tool("write_file", risk="caution"))
    got = []

    async def emit(event):
        got.append(event)

    client = PlainClient(_msg(calls=[("write_file", {})]))
    reply, tools = asyncio.run(rt.run_agent_turn_web(client, CONFIG, store, sid, reg, [{"role": "user", "content": "go"}], "Agent", emit))
    assert reply is None and "awaiting_approval" in _types(got) and "done" not in _types(got)


# ---------------------------------------------------------------------------------------- GUI-facing additions


def test_approval_descriptions_are_one_readable_line(tmp_path):
    target = tmp_path / "game.py"
    target.write_text("a\nb\n")
    assert turn.plain_description("write_file", {"path": str(target), "content": "a\nb\nc\nd\n"}).startswith("overwrite ")
    assert turn.plain_description("write_file", {"path": str(tmp_path / "new.py"), "content": "x\ny\n"}).endswith("new.py +2 −0")
    assert turn.plain_description("replace_in_file", {"path": str(target), "search": "a", "replace": "x\ny"}).endswith("game.py +2 −1")
    assert turn.plain_description("run_command", {"command": "rm -r  build\n&& ls"}) == "rm -r build && ls"
    assert turn.plain_description("send_file", {"path": "/x"}) .startswith("send_file(")


def test_grounding_event_marks_unsupported_figures_and_needs_a_search(tmp_path):
    from agent.research import evidence

    evidence.clear()
    assert turn.grounding_events("It costs 40 euros.") == []                     # no web search this turn: nothing to check
    evidence.set_last_search([{"n": 1, "status": "ok", "passage": "Startup is 40 % faster than 3.13."}], {"label": "likely", "summary": "1 source"})
    (ev,) = turn.grounding_events("Startup is 40 % faster [1]. Builds are 43 % faster too.")
    assert ev["type"] == "verdict" and ev["label"] == "likely" and ev["summary"] == "1 source"
    assert any("43" in w for w in ev["warnings"])
    evidence.clear()


def test_the_finalize_hook_runs_before_done(tmp_path):
    seen = []
    out, events, _, _ = _run(PlainClient(_msg("A final answer that is long enough.")), tmp_path, finalize=lambda text: seen.append(text) or [{"type": "verdict", "label": "likely"}])
    assert seen == ["A final answer that is long enough."]
    kinds = _types(events)
    assert kinds.index("verdict") < kinds.index("done")


def test_an_unreachable_model_becomes_an_error_event_not_a_cut_stream(tmp_path):
    import httpx

    class Down(PlainClient):
        def chat_with_tools(self, *a, **k):
            raise httpx.ConnectError("refused")

    store = Store(tmp_path / "d.db")
    sid = store.create_session()
    got = []

    async def emit(e):
        got.append(e)

    reply, _ = asyncio.run(rt.run_agent_turn_web(Down(), CONFIG, store, sid, _registry(_tool()), [{"role": "user", "content": "x"}], "A", emit))
    assert reply is None and got[-1]["type"] == "error" and got[-1]["code"] == "model_unreachable"
