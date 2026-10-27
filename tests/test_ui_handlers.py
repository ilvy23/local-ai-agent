"""ChatController + handlers wired into NightageApp, with a fake model and a temp Store. No terminal needed:
lines go through app.handle_line (the dock's entry point); the real dock is covered in test_ui_chat_app.py."""

import threading

import pytest

from agent import repoaccess, tui, undo
from tests._chatfake import (
    FakeClient, call, make_app, make_registry, make_state, nap, roles, say, wait_for,
)


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    tui._CHANGES.clear()
    monkeypatch.setattr(repoaccess, "_session", {})
    monkeypatch.setattr(repoaccess, "_greenlist_file", lambda: tmp_path / "greenlist.json")
    monkeypatch.setattr(tui, "_UNDO", undo.UndoStack())
    saved = {}
    import agent.settings as settings
    monkeypatch.setattr(settings, "set_value", lambda k, v: saved.__setitem__(k, v))
    yield saved


def idle(app):
    return wait_for(lambda: not app.worker.busy and app.worker.wait_idle(0.01))


def setup(tmp_path, script=(), gate=None, ran=None):
    client = FakeClient(script, gate=gate)
    state, cfg, store = make_state(tmp_path, client, make_registry(ran if ran is not None else []))
    app, ctl, buf = make_app(state, cfg, tmp_path)
    return app, ctl, state, store, client, buf


def test_plain_turn_persists_exactly_like_the_classic_loop(tmp_path):
    app, ctl, state, store, client, buf = setup(tmp_path, [say("Hello there, I am the answer.")])
    app.handle_line("hi there")
    assert wait_for(lambda: len(roles(store, state.session_id)) == 2) and idle(app)
    assert roles(store, state.session_id) == [("user", "hi there"), ("assistant", "Hello there, I am the answer.")]
    assert store.list_sessions()[0]["title"] == "hi there"
    assert "Hello there, I am the answer." in buf.getvalue()


def test_queue_while_busy_then_drain_in_order(tmp_path):
    gate = threading.Event()
    app, ctl, state, store, client, buf = setup(tmp_path, [say("First answer text."), say("Second answer text.")], gate=gate)
    app.handle_line("one")
    assert wait_for(lambda: app.worker.busy and client.calls == 1)
    app.handle_line("two")
    assert len(app.queue) == 1 and "queued (1)" in buf.getvalue()
    gate.set()
    assert wait_for(lambda: len(roles(store, state.session_id)) == 4) and idle(app)
    assert [r for r, _ in roles(store, state.session_id)] == ["user", "assistant", "user", "assistant"]
    assert roles(store, state.session_id)[2][1] == "two"
    assert len(app.queue) == 0


def test_model_lists_then_switches_and_persists(tmp_path, _clean):
    app, ctl, state, store, client, buf = setup(tmp_path)
    app.handle_line("/model")
    out = buf.getvalue()
    assert "1. m:1  (current)" in out and "2. m:2" in out
    app.handle_line("/model 2")
    assert state.config["models"]["chat"] == "m:2" and state.model == "m:2"
    assert _clean == {"chat_model": "m:2"}
    assert "Switched to m:2" in buf.getvalue()
    app.handle_line("/model nonsense")
    assert state.model == "m:2" and "not one of the installed models" in buf.getvalue()


def test_model_switch_is_deferred_while_busy(tmp_path):
    gate = threading.Event()
    app, ctl, state, store, client, buf = setup(tmp_path, [say("Some answer text here.")], gate=gate)
    app.handle_line("go")
    assert wait_for(lambda: app.worker.busy)
    app.handle_line("/model 2")
    assert state.model == "m:1" and len(app.queue) == 1
    gate.set()
    assert wait_for(lambda: state.model == "m:2")
    assert idle(app)


def test_clear_is_deferred_while_busy_and_starts_a_new_session(tmp_path):
    gate = threading.Event()
    app, ctl, state, store, client, buf = setup(tmp_path, [say("Answer for the old session.")], gate=gate)
    old = state.session_id
    app.handle_line("hello")
    assert wait_for(lambda: app.worker.busy)
    app.handle_line("/clear")
    assert state.session_id == old and len(app.queue) == 1          # nothing happens mid-turn
    gate.set()
    assert wait_for(lambda: state.session_id != old) and idle(app)
    assert [r for r, _ in roles(store, old)] == ["user", "assistant"]   # old session intact
    assert roles(store, state.session_id) == [] and state.title_pending is True
    assert "fresh conversation" in buf.getvalue()


def test_new_while_idle(tmp_path):
    app, ctl, state, store, client, buf = setup(tmp_path)
    old = state.session_id
    app.handle_line("/new")
    assert wait_for(lambda: state.session_id != old) and idle(app)
    assert "new chat" in buf.getvalue()


def test_undo_list_and_undo(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("after", encoding="utf-8")
    tui._UNDO.push({str(f): ("before", "after")})
    app, ctl, state, store, client, buf = setup(tmp_path)
    app.handle_line("/undo list")
    assert "1 file(s): a.txt" in buf.getvalue()
    app.handle_line("/undo")
    assert f.read_text(encoding="utf-8") == "before"
    app.handle_line("/undo")
    assert "nothing to undo" in buf.getvalue()


def test_plan_project_notes(tmp_path):
    app, ctl, state, store, client, buf = setup(tmp_path)
    app.handle_line("/plan")
    assert "plan  none" in buf.getvalue()
    app.handle_line("/project")
    assert "project  none active" in buf.getvalue()
    app.handle_line("/notes")
    assert "No notes taken" in buf.getvalue()
    store.append_working_note(state.session_id, "remember the milk")
    app.handle_line("/notes")
    assert "remember the milk" in buf.getvalue()
    app.handle_line("/notes clear")
    assert store.get_working_notes(state.session_id) in (None, "")


def test_repoaccess_family_and_access(tmp_path):
    folder = tmp_path / "proj"
    (folder / ".git").mkdir(parents=True)
    app, ctl, state, store, client, buf = setup(tmp_path)
    app.handle_line(f"/medium-{folder}")
    assert repoaccess.mode_for(folder.resolve()) == "medium"
    assert state.active_project.key
    app.handle_line("/access")
    assert str(folder.resolve()) in buf.getvalue() and "medium" in buf.getvalue()
    app.handle_line(f"/ask-{folder}")
    assert repoaccess.mode_for(folder.resolve()) == "ask"


def test_work_in_repo_holds_the_task_until_a_level_is_chosen(tmp_path):
    folder = tmp_path / "repo"
    folder.mkdir()
    app, ctl, state, store, client, buf = setup(tmp_path, [say("Started working in the repo now.")])
    app.handle_line(f"work in this repo: {folder}")
    assert wait_for(lambda: state.pending_task is not None)
    assert client.calls == 0 and roles(store, state.session_id) == []
    assert "how much may i do" in buf.getvalue().lower()
    app.handle_line(f"/medium-{folder}")
    assert wait_for(lambda: len(roles(store, state.session_id)) == 2) and idle(app)
    assert roles(store, state.session_id)[0][1] == f"work in this repo: {folder}"
    assert state.pending_task is None


def test_code_only_commands_say_so(tmp_path):
    app, ctl, state, store, client, buf = setup(tmp_path)
    app.handle_line("/steps 5")
    assert "controls the code agent" in buf.getvalue()


def test_compact_sets_the_flag(tmp_path):
    app, ctl, state, store, client, buf = setup(tmp_path)
    app.handle_line("/compact")
    assert state.compact.pending is True


def test_approval_round_trip_caution_deny_then_allow(tmp_path):
    ran = []
    app, ctl, state, store, client, buf = setup(
        tmp_path, [call("sendit", x="a"), say("Denied so I stopped here."), call("sendit", x="b"), say("Sent it now, done.")], ran=ran)
    app.handle_line("do a")
    assert wait_for(lambda: app.worker.pending_approval is not None)
    assert app.worker.pending_approval.tool == "sendit"
    app.handle_line("n")
    assert idle(app) and ran == []
    app.handle_line("do b")
    assert wait_for(lambda: app.worker.pending_approval is not None)
    app.handle_line("y")
    assert wait_for(lambda: ran == [("sendit", "b")]) and idle(app)


def test_dangerous_needs_the_typed_word_yes(tmp_path):
    ran = []
    app, ctl, state, store, client, buf = setup(tmp_path, [call("nuke", x="all"), say("It is gone now, finished.")], ran=ran)
    app.handle_line("do it")
    assert wait_for(lambda: app.worker.pending_approval is not None)
    assert app.worker.pending_approval.risk == "DANGEROUS"
    app.handle_line("y")
    app.handle_line("a")
    assert app.worker.pending_approval is not None and ran == []      # y / a refused for DANGEROUS
    assert "type  yes" in buf.getvalue()
    app.handle_line("yes")
    assert wait_for(lambda: ran == [("nuke", "all")]) and idle(app)


def test_verify_without_a_reply_says_so(tmp_path):
    app, ctl, state, store, client, buf = setup(tmp_path)
    app.handle_line("/verify")
    assert "nothing to check yet" in buf.getvalue()


def test_interrupt_keeps_partial_and_marks_it(tmp_path):
    gate = threading.Event()
    app, ctl, state, store, client, buf = setup(tmp_path, [say("Partial answer that is kept.")], gate=gate)
    app.handle_line("slow")
    assert wait_for(lambda: app.worker.busy and client.calls == 1)
    app.stop_turn()
    gate.set()
    assert idle(app)
    nap(0.05)
    assert roles(store, state.session_id)[0] == ("user", "slow")
