"""run_chat_ui end to end through the real dock (pipe input, dummy output) with a fake model and a temp Store."""

import io

from prompt_toolkit.output import DummyOutput
from rich.console import Console

from agent import tui
from agent.memory.store import Store
from agent.ui import chat_app
from tests._chatfake import CONFIG, FakeClient, Piped, make_registry, roles, say, wait_for
from tests.test_tui import FakeClient as ClassicClient
from tests.test_tui import _ScriptedConsole


def ui(tmp_path, client, store, buf, **kw):
    def go(inp):
        con = Console(file=buf, width=100, force_terminal=False, color_system=None, highlight=False)
        return chat_app.run_chat_ui(CONFIG | {"models": dict(CONFIG["models"])}, plain=True, input=inp, output=DummyOutput(),
                                    client=client, store=store, console=con, banner=False, history_path=tmp_path / "hist",
                                    width=lambda: 80, live_guard="raise", **kw)
    return go


def test_full_session_prompt_stream_persist_quit_unloads_models(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "_unload_models_on_exit", lambda *a, **k: calls.append("unload"))
    monkeypatch.setattr(tui, "_distill_on_exit", lambda *a, **k: calls.append("distill"))
    calls = []
    client, store, buf = FakeClient([say("Hello from the new terminal UI.")]), Store(tmp_path / "agent.db"), io.StringIO()
    # the controller builds its own registry via setup_chat (default tools); only the model is scripted
    with Piped(ui(tmp_path, client, store, buf)) as p:
        p.type("hello there\r")
        assert wait_for(lambda: store.get_last_session_id() and len(roles(store, store.get_last_session_id())) == 2)
        p.type("/quit\r")
    sid = store.get_last_session_id()
    assert roles(store, sid) == [("user", "hello there"), ("assistant", "Hello from the new terminal UI.")]
    assert p.rc == [0] and "Agent is ready" in buf.getvalue() and "Hello from the new terminal UI." in buf.getvalue()
    assert calls == ["distill"]            # injected client => the caller owns unloading


def test_classic_session_resumes_in_the_new_ui_and_back(tmp_path):
    store = Store(tmp_path / "agent.db")
    tui.run_repl(ClassicClient(["classic answer"]), CONFIG, console=_ScriptedConsole(["classic question", "/quit"]), store=store,
                 tool_registry=make_registry([]))
    sid = store.get_last_session_id()
    assert [r for r, _ in roles(store, sid)] == ["user", "assistant"]

    client, buf = FakeClient([say("Continuing in the new UI now.")]), io.StringIO()
    with Piped(ui(tmp_path, client, store, buf, resume=True)) as p:
        p.type("and now?\r")
        assert wait_for(lambda: len(roles(store, sid)) == 4)
        p.type("/quit\r")
    out = buf.getvalue()
    assert "resuming previous conversation" in out and "you: classic question" in out and "agent: classic answer" in out
    assert roles(store, sid)[2:] == [("user", "and now?"), ("assistant", "Continuing in the new UI now.")]
    # the model saw the classic rows as history
    sent = [m["content"] for m in client.seen[0] if m["role"] in ("user", "assistant")]
    assert sent[:3] == ["classic question", "classic answer", "and now?"]

    # and the classic loop continues the session the new UI wrote to
    classic = ClassicClient(["classic again"])
    tui.run_repl(classic, CONFIG, console=_ScriptedConsole(["third", "/quit"]), store=store, session_id=sid, tool_registry=make_registry([]))
    assert roles(store, sid)[-2:] == [("user", "third"), ("assistant", "classic again")]
    assert [m["content"] for m in classic.chat_calls[0] if m["role"] != "system"][:4] == [
        "classic question", "classic answer", "and now?", "Continuing in the new UI now."]


def test_resume_without_sessions_or_with_bad_id_returns_1(tmp_path):
    store = Store(tmp_path / "agent.db")
    assert chat_app.run_chat_ui(CONFIG, resume=True, client=FakeClient(), store=store) == 1
    sid = store.create_session()
    assert chat_app.run_chat_ui(CONFIG, session_id=sid + 99, client=FakeClient(), store=store) == 1
    assert chat_app.resolve_session(store, None, False) == (None, "")
    assert chat_app.resolve_session(store, None, True) == (sid, "")
    assert chat_app.resolve_session(store, sid, False) == (sid, "")


def test_slash_commands_through_the_dock_and_busy_prompt_queues(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "_distill_on_exit", lambda *a, **k: None)
    import threading
    gate = threading.Event()
    client, store, buf = FakeClient([say("First reply to you, here."), say("Second reply to you, here.")], gate=gate), Store(tmp_path / "agent.db"), io.StringIO()
    with Piped(ui(tmp_path, client, store, buf)) as p:
        p.type("first\r")
        assert wait_for(lambda: client.calls == 1)
        p.type("second\r")
        p.type("/queue\r")
        assert wait_for(lambda: "queued (1)" in buf.getvalue())
        gate.set()
        assert wait_for(lambda: len(roles(store, store.get_last_session_id())) == 4)
        p.type("/status\r")
        assert wait_for(lambda: "persona" in buf.getvalue())
        p.type("/quit\r")
    assert [c for _, c in roles(store, store.get_last_session_id())][::2] == ["first", "second"]


def test_history_preview_is_compact():
    msgs = [{"role": "user", "content": "q " * 100}, {"role": "tool", "content": "skipped"}, {"role": "assistant", "content": "a"}]
    lines = chat_app.history_preview(msgs, 40)
    assert len(lines) == 2 and lines[0].endswith("...") and len(lines[0]) <= 40 and lines[1] == "agent: a"


def test_demo_flag_runs_the_scripted_demo(monkeypatch):
    seen = {}
    import agent.ui.demo as demo
    monkeypatch.setattr(demo, "run_demo", lambda **kw: seen.update(kw) or 0)
    assert chat_app.run_chat_ui({}, demo=True, plain=True) == 0 and seen["plain"] is True


def test_owned_client_and_store_path_unloads_and_marks_foreground(tmp_path, monkeypatch):
    from agent import foreground, llm
    client = FakeClient([say("Reply from the owned client.")])
    monkeypatch.setattr(llm, "OllamaClient", lambda: client)
    monkeypatch.setattr(tui, "_distill_on_exit", lambda *a, **k: None)
    marked = []
    monkeypatch.setattr(foreground, "mark", lambda cfg: marked.append("mark"))
    monkeypatch.setattr(foreground, "clear", lambda cfg: marked.append("clear"))
    cfg = CONFIG | {"models": dict(CONFIG["models"]), "data": {"db_path": str(tmp_path / "own.db")}}
    buf = io.StringIO()

    def go(inp):
        con = Console(file=buf, width=100, force_terminal=False, color_system=None, highlight=False)
        return chat_app.run_chat_ui(cfg, plain=True, input=inp, output=DummyOutput(), console=con, banner=False,
                                    history_path=tmp_path / "hist", width=lambda: 80, live_guard="raise")
    with Piped(go) as p:
        p.type("hi\r")
        assert wait_for(lambda: "Reply from the owned client." in buf.getvalue())
        p.type("/quit\r")
    assert client.unloaded == 1 and marked[0] == "mark" and marked[-1] == "clear"
    store = Store(tmp_path / "own.db")
    assert [r for r, _ in roles(store, store.get_last_session_id())] == ["user", "assistant"]
