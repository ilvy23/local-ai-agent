from companion.memory.recall import build_context
from companion.memory.store import Store
from companion.memory.vectors import VectorIndex

CONFIG = {
    "persona": {"style": "You are Companion."},
    "models": {"embed": "nomic-embed-text"},
    "memory": {"recall_k": 6, "context_char_budget": 24000},
}


def _vec(*values):
    return list(values) + [0.0] * (VectorIndex.DIM - len(values))


class FakeLLM:
    """embed() returns a fixed vector per call so recall is deterministic."""

    def __init__(self, vector=None):
        self.vector = vector or _vec(1.0, 0.0)

    def embed(self, texts, model):
        return [self.vector for _ in texts]


def _make(tmp_path):
    store = Store(tmp_path / "companion.db")
    return store, VectorIndex(store)


def test_system_prompt_includes_persona_and_facts(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("dog is named Rex", source_session_id=None)
    store.add_fact("lives in Berlin", source_session_id=None)
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")

    system = messages[0]
    assert system["role"] == "system"
    assert "You are Companion." in system["content"]
    assert "Rex" in system["content"]
    assert "Berlin" in system["content"]
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_semantic_hits_included_when_present(tmp_path):
    store, vectors = _make(tmp_path)
    vectors.add("message", 5, "we talked about hiking in the Alps", _vec(1.0, 0.0))
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(_vec(1.0, 0.0)), CONFIG, session_id, "trip?")

    assert "hiking in the Alps" in messages[0]["content"]
    assert "relevant past memories" in messages[0]["content"].lower()


def test_recall_section_skipped_when_no_hits(tmp_path):
    store, vectors = _make(tmp_path)  # empty index
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")

    assert "relevant past memories" not in messages[0]["content"].lower()


def test_current_session_messages_included(tmp_path):
    store, vectors = _make(tmp_path)
    session_id = store.create_session()
    store.add_message(session_id, "user", "earlier q")
    store.add_message(session_id, "assistant", "earlier a")

    messages = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "new q")

    roles = [(m["role"], m["content"]) for m in messages]
    assert ("user", "earlier q") in roles
    assert ("assistant", "earlier a") in roles
    assert roles[-1] == ("user", "new q")
    assert roles[0][0] == "system"


def test_budget_drops_oldest_messages_first_never_facts(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("important fact about the user", source_session_id=None)
    session_id = store.create_session()
    for i in range(20):
        store.add_message(session_id, "user", f"OLDEST-{i} " + "x" * 500)
    store.add_message(session_id, "user", "NEWEST " + "y" * 500)

    tight = {**CONFIG, "memory": {"recall_k": 6, "context_char_budget": 1500}}
    messages = build_context(store, vectors, FakeLLM(), tight, session_id, "final")

    joined = "".join(m["content"] for m in messages)
    assert "important fact about the user" in joined  # facts never dropped
    assert "NEWEST" in joined  # newest kept
    assert "OLDEST-0" not in joined  # oldest dropped first


def test_budget_trim_never_leaves_orphaned_leading_tool_message(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("important fact about the user", source_session_id=None)
    session_id = store.create_session()
    # Several assistant-tool-call + tool + assistant-answer triples, padded
    # to force the char budget to cut through the middle of one.
    for i in range(10):
        store.add_message(session_id, "assistant", f"CALL-{i} " + "a" * 200)
        store.add_message(session_id, "tool", f"RESULT-{i} " + "b" * 200)
        store.add_message(session_id, "assistant", f"ANSWER-{i} " + "c" * 200)
    store.add_message(session_id, "user", "NEWEST " + "y" * 200)

    tight = {**CONFIG, "memory": {"recall_k": 6, "context_char_budget": 1500}}
    messages = build_context(store, vectors, FakeLLM(), tight, session_id, "final")

    joined = "".join(m["content"] for m in messages)
    assert "important fact about the user" in joined  # facts never dropped

    # Find where history starts (right after the system message).
    history = messages[1:-1]
    assert history, "expected some history to survive the trim"
    assert history[0]["role"] != "tool"


def test_tool_instructions_included_when_tools_are_available(tmp_path):
    store, vectors = _make(tmp_path)
    session_id = store.create_session()

    messages = build_context(
        store, vectors, FakeLLM(), CONFIG, session_id, "hi", tool_names=["current_time"]
    )

    content = messages[0]["content"]
    assert "current_time" in content
    assert '"tool"' in content
    assert '"arguments"' in content


def test_tool_instructions_omitted_when_no_tools(tmp_path):
    store, vectors = _make(tmp_path)
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")

    assert '"tool"' not in messages[0]["content"]


def test_embed_failure_does_not_break_context(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("a fact", source_session_id=None)
    session_id = store.create_session()

    class BoomLLM:
        def embed(self, texts, model):
            raise RuntimeError("embed down")

    messages = build_context(store, vectors, BoomLLM(), CONFIG, session_id, "hi")

    assert "a fact" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "hi"}
