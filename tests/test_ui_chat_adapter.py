"""Event stream of the terminal chat turn (agent.tui + agent.ui.chat_adapter). Fakes only, no model."""

from __future__ import annotations

import json
import threading
from io import StringIO

import pytest
from rich.console import Console

from agent import tui
from agent.llm import GenerationAborted
from agent.memory.store import Store
from agent.research import evidence as turn_evidence
from agent.safety import RiskLevel
from agent.tools.registry import Tool, ToolRegistry
from agent.ui.chat_adapter import (
    ApprovalRequest, ChatState, CompactFlag, TurnControl, compact_now, resolve_approval, run_chat_turn,
)

CONFIG = {
    "models": {"chat": "m:1", "background": "b:1", "embed": "e:1"},
    "persona": {"name": "Agent", "style": "You are Agent."},
    "memory": {"recall_k": 6, "context_char_budget": 24000},
    "tools": {"max_iterations": 8},
}


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def say(text, **extra):
    return {"role": "assistant", "content": text, "_prompt_eval_count": 100, "_eval_count": 20,
            "_eval_ns": 2_000_000_000, **extra}


class StreamClient:
    """Streams each scripted message in 4-char pieces (thinking first), honouring GenerationAborted like llm.py."""

    def __init__(self, script, thinking="hmm"):
        self.script, self.thinking, self.calls = list(script), thinking, 0

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, **kw):
        self.calls += 1
        message = dict(self.script.pop(0))
        message.setdefault("_prompt_eval_count", 100)
        message.setdefault("_eval_count", 20)
        text, seen = message.get("content", ""), []
        try:
            if on_thinking and self.thinking:
                on_thinking(self.thinking)
            for i in range(0, len(text), 4):
                if on_content:
                    on_content(text[i:i + 4])
                seen.append(text[i:i + 4])
        except GenerationAborted as stop:
            message["content"] = "".join(seen)
            message["_aborted"] = str(stop)
        return message

    def chat_with_tools(self, messages, model, tools, **kw):
        raise AssertionError("buffered call not expected")

    def chat(self, *a, **kw):
        return iter(["- step 1\n- step 2"])


def make_registry(tmp_path, approvals=None):
    reg = ToolRegistry()
    reg.register(Tool("echo", "Echo.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                      lambda text: f"echoed: {text}"))
    reg.register(Tool("sendit", "Caution.", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                      lambda x: f"sent {x}", risk="caution"))
    reg.register(Tool("nuke", "Danger.", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                      lambda x: f"nuked {x}", risk="dangerous"))

    def write(path, content):
        from pathlib import Path
        Path(path).write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    reg.register(Tool("write_file", "Write.", {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, write, risk="caution"))

    def search(query):
        turn_evidence.set_last_search(
            [{"n": 1, "url": "https://a.example/x", "domain": "a.example", "status": "ok", "passage": "fact", "trust": {"level": "high"}}],
            {"label": "verified", "summary": "1 source"})
        return f"Web search {query}: [1] fact"

    reg.register(Tool("web_search", "Search.", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, search))
    return reg


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "agent.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _fresh_globals():
    tui._CHANGES.clear()
    turn_evidence.clear()
    yield
    tui._CHANGES.clear()


def run(store, client, registry, *, console=None, messages=None, **kw):
    console = console or Console(file=StringIO(), force_terminal=False, width=100)
    sid = store.create_session()
    events = []
    result = tui._run_agent_turn(client, CONFIG, console, store, sid, registry,
                                 messages or [{"role": "user", "content": "do it"}], "Agent",
                                 emit=events.append, **kw)
    return result, events, console


def kinds(events):
    return [e.kind for e in events]


# ------------------------------------------------------------------ plain answer


def test_plain_answer_event_order(store, tmp_path):
    client = StreamClient([say("Hello there, this is the answer.")])
    (reply, tools), events, _ = run(store, client, make_registry(tmp_path), suppress_print=True)
    assert reply == "Hello there, this is the answer."
    ks = kinds(events)
    assert ks[0] == "status" and ks[-1] == "done"
    assert ks.index("token") < ks.index("context") < ks.index("done")
    assert "".join(e.data["text"] for e in events if e.kind == "token") == reply
    ctx = next(e for e in events if e.kind == "context").data
    assert ctx["used"] == 120 and ctx["percent"] >= 0 and ctx["tokens_per_s"] == 10.0 and ctx["compacted"] is False
    done = events[-1].data
    assert done["reply"] == reply and done["interrupted"] is False


def test_thinking_text_only_with_capture_flag(store, tmp_path):
    (_, _), ev, _ = run(store, StreamClient([say("Hello there, answer here.")]), make_registry(tmp_path), suppress_print=True)
    assert "thinking" not in kinds(ev)
    (_, _), ev, _ = run(store, StreamClient([say("Hello there, answer here.")]), make_registry(tmp_path),
                        suppress_print=True, capture_thinking=True)
    assert [e.data["text"] for e in ev if e.kind == "thinking"] == ["hmm"]


# ------------------------------------------------------------------ tools


def test_tool_turn_has_full_args_result_risk(store, tmp_path):
    client = StreamClient([call("echo", text="hi"), say("It said echoed: hi okay.")])
    (reply, tools), ev, _ = run(store, client, make_registry(tmp_path), suppress_print=True)
    assert tools == {"echo"}
    ks = kinds(ev)
    assert ks.index("tool_call") < ks.index("tool_result") < ks.index("done")
    tc = next(e for e in ev if e.kind == "tool_call").data
    assert tc["name"] == "echo" and tc["arguments"] == {"text": "hi"} and tc["risk"] == "SAFE" and "echo" in tc["summary"]
    tr = next(e for e in ev if e.kind == "tool_result").data
    assert tr["result"] == "echoed: hi" and tr["ok"] is True and tr["seconds"] >= 0 and tr["truncated"] is False


def test_tool_result_is_capped(store, tmp_path):
    reg = make_registry(tmp_path)
    reg.register(Tool("big", "Big.", {"type": "object", "properties": {}}, lambda: "x" * 50_000))
    (_, _), ev, _ = run(store, StreamClient([call("big"), say("Finished with that tool call.")]), reg, suppress_print=True)
    tr = next(e for e in ev if e.kind == "tool_result").data
    assert len(tr["result"]) == 20_000 and tr["truncated"] is True and tr["chars"] == 50_000


def test_unknown_tool_and_blocked_are_reported_not_ok(store, tmp_path):
    (_, _), ev, _ = run(store, StreamClient([call("nope"), say("Could not do it, sorry about that.")]),
                        make_registry(tmp_path), suppress_print=True)
    assert next(e for e in ev if e.kind == "tool_result").data["ok"] is False


# ------------------------------------------------------------------ edit


def test_edit_event_carries_diff_and_changes_summary(store, tmp_path):
    target = tmp_path / "game.py"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")
    client = StreamClient([call("write_file", path=str(target), content="a = 1\nb = 3\nc = 4\n"), say("Done editing the file now.")])
    (_, _), ev, _ = run(store, client, make_registry(tmp_path), suppress_print=True, approve_fn=lambda r: True)
    edit = next(e for e in ev if e.kind == "edit").data
    assert edit["added"] == 2 and edit["removed"] == 1 and edit["created"] is False
    assert "-b = 2" in edit["diff"] and "+b = 3" in edit["diff"] and "+c = 4" in edit["diff"]
    ch = next(e for e in ev if e.kind == "changes").data
    assert ch["added"] == 2 and ch["removed"] == 1 and len(ch["files"]) == 1
    assert kinds(ev).index("tool_call") < kinds(ev).index("edit") < kinds(ev).index("changes") < kinds(ev).index("done")


# ------------------------------------------------------------------ web


def test_web_turn_emits_sources_and_verdict(store, tmp_path):
    client = StreamClient([call("web_search", query="x"), say("According to the source, it is so.")])
    (_, _), ev, _ = run(store, client, make_registry(tmp_path), suppress_print=True)
    ks = kinds(ev)
    assert ks.index("tool_result") < ks.index("sources") < ks.index("verdict")
    src = next(e for e in ev if e.kind == "sources").data["items"][0]
    assert src["url"] == "https://a.example/x" and src["passage"] == "fact"
    assert next(e for e in ev if e.kind == "verdict").data["label"] == "verified"
    st = [e.data for e in ev if e.kind == "status" and e.data.get("tool") == "web_search"]
    assert st and st[0]["label"] == "searching" and st[0]["detail"] == "x"
    assert ks.count("sources") == 1


# ------------------------------------------------------------------ stop


def test_stop_mid_stream_is_clean_and_keeps_partial(store, tmp_path):
    stop = threading.Event()

    class Stopper(StreamClient):
        def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, **kw):
            def wrapped(piece):
                on_content(piece)
                if len(piece) and not stop.is_set():
                    stop.set()           # the user presses Esc after the first piece
            return super().chat_with_tools_stream(messages, model, tools, num_ctx, wrapped, on_thinking, **kw)

    client = Stopper([say("This reply would have been much longer than this.")], thinking="")
    ctl = TurnControl(emit=None, stop=stop, suppress_print=True)
    events = []
    ctl.emit = events.append
    console = Console(file=StringIO(), force_terminal=False)
    sid = store.create_session()
    reply, tools = tui._run_agent_turn(client, CONFIG, console, store, sid, make_registry(tmp_path),
                                       [{"role": "user", "content": "go"}], "Agent", ctl=ctl)
    assert ctl.interrupted is True
    assert reply and "This reply would have been much longer than this." not in reply
    assert ctl.partial == reply
    done = events[-1]
    assert done.kind == "done" and done.data["interrupted"] is True and done.data["reply"] == reply


def test_stop_set_before_start_does_not_call_the_model(store, tmp_path):
    stop = threading.Event()
    stop.set()
    client = StreamClient([say("never used here at all")])
    (reply, _), ev, _ = run(store, client, make_registry(tmp_path), suppress_print=True, stop=stop)
    assert reply is None and client.calls == 0 and ev[-1].data["interrupted"] is True


def test_stop_between_tool_calls_skips_the_next_tool(store, tmp_path):
    stop = threading.Event()
    ran = []
    reg = make_registry(tmp_path)
    reg.register(Tool("echo", "Echo.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                      lambda text: (ran.append(text), stop.set(), "ok")[2]))
    client = StreamClient([call("echo", text="1"), call("echo", text="2"), say("unused final answer text")])
    (reply, tools), ev, _ = run(store, client, reg, suppress_print=True, stop=stop)
    assert ran == ["1"] and tools == {"echo"} and ev[-1].data["interrupted"] is True


# ------------------------------------------------------------------ approvals


def test_approve_fn_allow_and_deny_caution(store, tmp_path):
    seen = []

    def allow(req: ApprovalRequest):
        seen.append(req)
        return True

    (_, _), ev, _ = run(store, StreamClient([call("sendit", x="a"), say("Sent it for you just now.")]),
                        make_registry(tmp_path), suppress_print=True, approve_fn=allow)
    assert seen[0].risk == "CAUTION" and seen[0].arguments == {"x": "a"} and not seen[0].needs_typed_yes
    aw = next(e for e in ev if e.kind == "awaiting_approval").data
    assert aw["risk"] == "CAUTION" and aw["name"] == "sendit"
    assert next(e for e in ev if e.kind == "tool_result").data["result"] == "sent a"

    (_, _), ev, _ = run(store, StreamClient([call("sendit", x="a"), say("Okay, I did not do that.")]),
                        make_registry(tmp_path), suppress_print=True, approve_fn=lambda r: False)
    tr = next(e for e in ev if e.kind == "tool_result").data
    assert tr["result"] == tui.DENIED_RESULT and tr["ok"] is False


def test_dangerous_needs_the_exact_word_yes(store, tmp_path):
    for answer, expect in [(True, False), ("y", False), ("Yes", False), ("yes", True)]:
        (_, _), ev, _ = run(store, StreamClient([call("nuke", x="z"), say("Finished handling the request.")]),
                            make_registry(tmp_path), suppress_print=True, approve_fn=lambda r, a=answer: a)
        result = next(e for e in ev if e.kind == "tool_result").data["result"]
        assert (result == "nuked z") is expect, answer
        assert next(e for e in ev if e.kind == "tool_call").data["risk"] == "DANGEROUS"


def test_suppress_print_without_approve_fn_denies_and_never_reads_stdin(store, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("stdin read"))
    (_, _), ev, _ = run(store, StreamClient([call("sendit", x="a"), say("Not done since not allowed.")]),
                        make_registry(tmp_path), suppress_print=True)
    assert next(e for e in ev if e.kind == "tool_result").data["result"] == tui.DENIED_RESULT


def test_default_console_prompt_is_unchanged_without_approve_fn(store, tmp_path):
    class Scripted(Console):
        def input(self, prompt="", **kw):
            self.print(prompt, end="")
            return "y"

    console = Scripted(file=StringIO(), force_terminal=False)
    (_, _), ev, console = run(store, StreamClient([call("sendit", x="a"), say("All sent successfully now.")]),
                              make_registry(tmp_path), console=console)
    assert "Run this?" in console.file.getvalue()
    assert next(e for e in ev if e.kind == "tool_result").data["result"] == "sent a"


def test_resolve_approval_rules():
    assert resolve_approval(RiskLevel.CAUTION, True) and resolve_approval(RiskLevel.CAUTION, " Y ")
    assert not resolve_approval(RiskLevel.CAUTION, "no") and not resolve_approval(RiskLevel.CAUTION, None)
    assert resolve_approval(RiskLevel.DANGEROUS, "yes") and not resolve_approval(RiskLevel.DANGEROUS, True)


# ------------------------------------------------------------------ compaction


def test_manual_compact_request_fires_once(store, tmp_path, monkeypatch):
    from agent.memory import worklog

    calls = []

    def fake_compact(client, config, messages, keep_last=0):
        calls.append(len(messages))
        return [messages[0], {"role": "system", "content": "WORKLOG short"}, messages[-1]]

    monkeypatch.setattr(worklog, "compact_turn_messages", fake_compact)
    flag = CompactFlag()
    flag.request()
    client = StreamClient([call("echo", text="1"), call("echo", text="2"), say("All finished with the steps.")])
    (_, _), ev, _ = run(store, client, make_registry(tmp_path), suppress_print=True, compact=flag)
    assert len(calls) == 1 and not flag.pending
    compacted = [e for e in ev if e.kind == "context" and e.data["compacted"]]
    assert len(compacted) == 1 and compacted[0].data["before"] >= 0


def test_compact_flag_is_one_shot():
    f = CompactFlag()
    assert f.take() is False
    f.request(); f.request()
    assert f.pending and f.take() is True and f.take() is False


def test_compact_now_between_turns_is_just_mask_and_summary(monkeypatch):
    from agent.memory import worklog

    monkeypatch.setattr(worklog, "compact_turn_messages", lambda c, cfg, m: None)
    msgs = [{"role": "user", "content": "t"}]
    out, info = compact_now(object(), CONFIG, msgs)
    assert out == msgs and info["compacted"] == 0


# ------------------------------------------------------------------ printing / default behaviour


def test_suppress_print_prints_nothing_but_events_flow(store, tmp_path):
    target = tmp_path / "f.py"
    client = StreamClient([call("echo", text="hi"), call("write_file", path=str(target), content="x = 1\n"), say("Everything is complete now.")])
    (_, _), ev, console = run(store, client, make_registry(tmp_path), suppress_print=True, approve_fn=lambda r: True)
    assert console.file.getvalue() == ""
    assert {"tool_call", "tool_result", "edit", "changes", "token", "done"} <= set(kinds(ev))


def test_default_prints_unchanged_and_emit_alone_does_not_remove_prints(store, tmp_path):
    reg = make_registry(tmp_path)
    reg.register(Tool("read_file", "Read.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                      lambda path: "contents"))
    plain_console = Console(file=StringIO(), force_terminal=False, width=100)
    sid = store.create_session()
    msgs = [{"role": "user", "content": "go"}]

    class Buffered:
        def __init__(self): self.s = [call("read_file", path="/x"), say("The file says contents here.")]
        def chat_with_tools(self, messages, model, tools, **kw): return self.s.pop(0)

    tui._run_agent_turn(Buffered(), CONFIG, plain_console, store, sid, reg, msgs, "Agent")
    baseline = plain_console.file.getvalue()
    assert "read › " in baseline and "The file says contents here." in baseline

    emitted = Console(file=StringIO(), force_terminal=False, width=100)
    client = StreamClient([call("read_file", path="/x"), say("The file says contents here.")])
    tui._run_agent_turn(client, CONFIG, emitted, store, sid, reg, msgs, "Agent", emit=lambda e: None)
    assert emitted.file.getvalue() == baseline


def test_broken_subscriber_never_breaks_the_turn(store, tmp_path):
    def boom(ev):
        raise RuntimeError("subscriber bug")

    sid = store.create_session()
    reply, _ = tui._run_agent_turn(StreamClient([say("Fine anyway, answer is here.")]), CONFIG,
                                   Console(file=StringIO()), store, sid, make_registry(tmp_path),
                                   [{"role": "user", "content": "q"}], "Agent", emit=boom, suppress_print=True)
    assert reply == "Fine anyway, answer is here."


def test_http_error_becomes_error_event(store, tmp_path):
    import httpx

    class Bad:
        def chat_with_tools_stream(self, *a, **k): raise httpx.ConnectError("down")
        def chat_with_tools(self, *a, **k): raise httpx.ConnectError("down")

    events = []
    sid = store.create_session()
    with pytest.raises(httpx.ConnectError):
        tui._run_agent_turn(Bad(), CONFIG, Console(file=StringIO()), store, sid, make_registry(tmp_path),
                            [{"role": "user", "content": "q"}], "Agent", emit=events.append, suppress_print=True)
    assert kinds(events)[-2:] == ["error", "done"] and events[-1].data["failed"] is True


# ------------------------------------------------------------------ run_chat_turn


def _state(store, tmp_path, client):
    return ChatState.from_repl_args(client, CONFIG, Console(file=StringIO()), store, None, None, make_registry(tmp_path))


def test_run_chat_turn_persists_and_returns_result(store, tmp_path):
    state = _state(store, tmp_path, StreamClient([call("echo", text="a"), say("The tool answered with a result.")]))
    events = []
    res = run_chat_turn(state, "please echo a", events.append)
    assert res.reply == "The tool answered with a result." and res.interrupted is False and res.tool_names == ["echo"]
    assert res.usage["used"] == 120
    msgs = store.get_messages(state.session_id)
    assert [m["role"] for m in msgs][-1] == "assistant" and msgs[0]["content"] == "please echo a"
    assert state.last_reply == res.reply and state.title_pending is False
    assert state.goal_state["original_task"] == "please echo a"
    assert state.tool_exchanges and kinds(events)[-1] == "done"


def test_run_chat_turn_interrupt_keeps_partial(store, tmp_path):
    stop = threading.Event()

    class Stopper(StreamClient):
        def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, **kw):
            return super().chat_with_tools_stream(messages, model, tools, num_ctx,
                                                  lambda p: (on_content(p), stop.set()), on_thinking, **kw)

    state = _state(store, tmp_path, Stopper([say("A long answer that gets cut off early.")], thinking=""))
    res = run_chat_turn(state, "tell me", None, stop)
    assert res.interrupted is True and res.reply
    assert store.get_messages(state.session_id)[-1]["content"].endswith("[interrupted by the user]")
    assert state.last_reply is None


def test_run_chat_turn_forced_web_emits_sources(store, tmp_path):
    state = _state(store, tmp_path, StreamClient([say("The web says it is so, confirmed.")]))
    events = []
    res = run_chat_turn(state, "what is the latest /web", events.append)
    assert res.reply and "sources" in kinds(events) and "verdict" in kinds(events)
    assert any(m["role"] == "tool" and "Web search results" in m["content"] for m in store.get_messages(state.session_id))


def test_run_chat_turn_http_error_does_not_raise(store, tmp_path):
    import httpx

    class Bad(StreamClient):
        def chat_with_tools_stream(self, *a, **k): raise httpx.ConnectError("down")
        def chat_with_tools(self, *a, **k): raise httpx.ConnectError("down")

    state = _state(store, tmp_path, Bad([]))
    events = []
    res = run_chat_turn(state, "hello", events.append)
    assert res.reply is None and res.error
    assert "error" in kinds(events) and kinds(events)[-1] == "done"


def test_request_compact_reaches_the_turn(store, tmp_path, monkeypatch):
    from agent.memory import worklog

    n = []
    monkeypatch.setattr(worklog, "compact_turn_messages", lambda c, cfg, m, keep_last=0: n.append(1))
    state = _state(store, tmp_path, StreamClient([call("echo", text="a"), say("Finished without trouble at all.")]))
    state.request_compact()
    run_chat_turn(state, "echo a", None)
    assert n == [1] and not state.compact.pending
