from agent.memory.distill import _extract_json_array, distill_session
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

CONFIG = {
    "models": {"background": "llama3.1:8b", "embed": "nomic-embed-text"},
}


def _vec(*values):
    return list(values) + [0.0] * (VectorIndex.DIM - len(values))


class FakeLLM:
    """chat() yields a scripted reply; embed() returns a fixed or per-text vector."""

    def __init__(self, reply, vectors=None):
        self._reply = reply
        self._vectors = vectors or {}

    def chat(self, messages, model, stream=True, **kwargs):
        yield self._reply

    def embed(self, texts, model):
        return [self._vectors.get(t, _vec(1.0, 0.0)) for t in texts]


# ---- JSON extraction robustness ----

def test_extract_clean_array():
    assert _extract_json_array('["a", "b"]') == ["a", "b"]


def test_extract_junk_wrapped_array():
    reply = 'Sure! Here are the facts:\n```json\n["likes tea", "has a cat"]\n```\nHope that helps.'
    assert _extract_json_array(reply) == ["likes tea", "has a cat"]


def test_extract_garbage_returns_none():
    assert _extract_json_array("I could not find any facts, sorry.") is None


def test_extract_ignores_non_string_items():
    assert _extract_json_array('["ok", 5, null, "good"]') == ["ok", "good"]


# ---- distill_session behaviour ----

def _session_with_user_messages(store, count):
    session_id = store.create_session()
    for i in range(count):
        store.add_message(session_id, "user", f"user line {i}")
        store.add_message(session_id, "assistant", f"assistant line {i}")
    return session_id


def test_skips_when_fewer_than_two_user_messages(tmp_path):
    store = Store(tmp_path / "agent.db")
    vectors = VectorIndex(store)
    session_id = _session_with_user_messages(store, 1)

    added = distill_session(store, vectors, FakeLLM('["a fact"]'), CONFIG, session_id)

    assert added == []
    assert store.get_active_facts() == []


def test_stores_new_facts_and_embeds_them(tmp_path):
    store = Store(tmp_path / "agent.db")
    vectors = VectorIndex(store)
    session_id = _session_with_user_messages(store, 2)
    llm = FakeLLM(
        '["dog named Rex", "lives in Berlin"]',
        vectors={"dog named Rex": _vec(1.0, 0.0), "lives in Berlin": _vec(0.0, 1.0)},
    )

    added = distill_session(store, vectors, llm, CONFIG, session_id)

    assert set(added) == {"dog named Rex", "lives in Berlin"}
    contents = {f["content"] for f in store.get_active_facts()}
    assert contents == {"dog named Rex", "lives in Berlin"}
    # embedded as facts
    hits = vectors.search(_vec(1.0, 0.0), k=8, kinds=["fact"])
    assert any(h[0] == "dog named Rex" for h in hits)


def test_dedupes_exact_existing_fact(tmp_path):
    store = Store(tmp_path / "agent.db")
    vectors = VectorIndex(store)
    store.add_fact("dog named Rex", source_session_id=None)
    session_id = _session_with_user_messages(store, 2)

    added = distill_session(store, vectors, FakeLLM('["dog named Rex"]'), CONFIG, session_id)

    assert added == []
    assert len(store.get_active_facts()) == 1


def test_garbage_reply_skips_without_error(tmp_path):
    store = Store(tmp_path / "agent.db")
    vectors = VectorIndex(store)
    session_id = _session_with_user_messages(store, 2)

    added = distill_session(store, vectors, FakeLLM("no facts here"), CONFIG, session_id)

    assert added == []
    assert store.get_active_facts() == []


def test_dedupes_semantically_similar_fact(tmp_path):
    store = Store(tmp_path / "agent.db")
    vectors = VectorIndex(store)
    # existing fact embeds to nearly the same vector as the candidate
    similar = _vec(1.0, 0.01)
    store.add_fact("has a dog called Rex", source_session_id=None)
    vectors.add("fact", 1, "has a dog called Rex", similar)
    session_id = _session_with_user_messages(store, 2)

    llm = FakeLLM('["dog is named Rex"]', vectors={"dog is named Rex": _vec(1.0, 0.0)})
    added = distill_session(store, vectors, llm, CONFIG, session_id)

    assert added == []
    assert len(store.get_active_facts()) == 1
