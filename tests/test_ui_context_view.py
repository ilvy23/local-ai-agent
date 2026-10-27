"""/context inspector and footer facts: offline, read-only, fakes only."""

import io

import pytest
from rich.console import Console

from agent.ui import context_view as CV
from agent.ui import facts as F
from tests._chatfake import FakeClient, make_app, make_registry, make_state, roles, say, wait_for


def text_of(renderable):
    buf = io.StringIO()
    Console(file=buf, width=100, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


@pytest.fixture
def chat(tmp_path):
    client = FakeClient([say("Berlin is the capital of Germany indeed.")])
    state, cfg, store = make_state(tmp_path, client, make_registry([]))
    return state, cfg, store, client


def test_parts_cover_the_prompt_with_tokens(chat):
    state, cfg, store, client = chat
    store.add_fact("User lives in Munich", None, kind="personal")
    store.append_working_note(state.session_id, "ship it friday")
    store.add_message(state.session_id, "user", "what is the capital of Germany?")
    store.add_message(state.session_id, "assistant", "Berlin.")
    store.add_message(state.session_id, "tool", "Web search results: Berlin is the capital")
    parts = {p.name: p for p in CV.collect(state)}
    for name in ("persona", "system rules + tools", "environment", "facts about you", "session notes", "earlier summary",
                 "recalled memories", "plan", "project notes", "conversation", "recent tool results"):
        assert name in parts
    assert parts["facts about you"].chars > 0 and "Munich" in parts["facts about you"].lines[0]
    assert parts["session notes"].tokens > 0
    assert parts["conversation"].note.startswith("2 messages")
    assert parts["recent tool results"].note == "1 kept"
    assert parts["persona"].tokens == CV.est_tokens(len("You are Agent."))


def test_render_shows_percent_budget_and_never_mutates(chat):
    state, cfg, store, client = chat
    store.add_message(state.session_id, "user", "hello")
    before = (store.get_session_compaction(state.session_id), roles(store, state.session_id), client.calls)
    out = text_of(CV.render(CV.collect(state), model="m:1", limit=10000,
                            explain={"computed": 50000, "override": None, "final": 50000, "ceiling": 95000, "source": "ram"},
                            plain=True))
    assert "next prompt ~" in out and "of 10,000 (" in out and "persona" in out and "%" in out
    assert "computed 50,000" in out and "override none" in out and "ceiling 95,000" in out
    after = (store.get_session_compaction(state.session_id), roles(store, state.session_id), client.calls)
    assert before == after


def test_trimmed_history_is_reported_not_compacted(chat):
    state, cfg, store, client = chat
    cfg["memory"]["context_char_budget"] = 400
    for i in range(10):
        store.add_message(state.session_id, "user" if i % 2 == 0 else "assistant", f"message number {i} " + "x" * 120)
    conv = next(p for p in CV.collect(state) if p.name == "conversation")
    assert "older would be summarised" in conv.note
    assert store.get_session_compaction(state.session_id) == (None, None)


def test_recalls_with_similarity(chat):
    state, cfg, store, client = chat

    class Vec:
        available = True

        def search(self, emb, k, kinds):
            return [("User lives in Munich", "fact", 1, 0.5), ("far away thing", "fact", 2, 1.4)]

    client.embed = lambda texts, model=None: [[0.0]]
    state.vectors = Vec()
    cfg["memory"]["recall_min_similarity"] = 0.6
    p = next(p for p in CV.collect(state, "where do I live") if p.name == "recalled memories")
    assert p.lines == ["0.88 fact: User lives in Munich"] and "1 kept" in p.note


def test_inspector_works_through_the_app(tmp_path):
    client = FakeClient([say("Short answer for the test.")])
    state, cfg, store = make_state(tmp_path, client, make_registry([]))
    app, ctl, buf = make_app(state, cfg, tmp_path)
    app.handle_line("hello")
    assert wait_for(lambda: len(roles(store, state.session_id)) == 2 and not app.worker.busy)
    app.handle_line("/context")
    out = buf.getvalue()
    assert "next prompt ~" in out and "conversation" in out and "context budget" in out


def test_inspector_never_raises(chat):
    state, cfg, store, client = chat
    state.store = None
    state.registry = None
    out = text_of(CV.inspector(state, plain=True)())
    assert "failed" in out or "next prompt" in out


def test_footer_facts_are_defensive(monkeypatch):
    f = F.footer_facts({"models": {"chat": "qwen:27b"}}, folder="/tmp")
    assert f["model"] == "qwen:27b" and f["backend"] in ("Ollama", "Vulkan") and f["access"] in ("ask", "medium", "auto")
    assert f["folder"] == "/tmp" and f["ctx_limit"] > 0
    assert F.footer_facts(None)["model"] == ""
    assert F.footer_facts(object())["model"] == ""      # garbage config never raises


def test_backend_follows_vulkan_routing(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm, "wants_vulkan", lambda m: m.startswith("big"))
    assert F.backend_name("big:27b") == "Vulkan" and F.backend_name("small:7b") == "Ollama"
    monkeypatch.setattr(llm, "wants_vulkan", lambda m: 1 / 0)
    assert F.backend_name("x") == "Ollama"
