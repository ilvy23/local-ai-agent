from agent.memory.recall import build_context
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

CONFIG = {
    "persona": {"style": "You are Agent."},
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
    store = Store(tmp_path / "agent.db")
    return store, VectorIndex(store)


def test_system_prompt_includes_persona_and_facts(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("dog is named Rex", source_session_id=None)
    store.add_fact("lives in Berlin", source_session_id=None)
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")

    system = messages[0]
    assert system["role"] == "system"
    assert "You are Agent." in system["content"]
    assert "Rex" in system["content"]
    assert "Berlin" in system["content"]
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_semantic_hits_included_when_present(tmp_path):
    store, vectors = _make(tmp_path)
    vectors.add("message", 5, "we talked about hiking in the Alps", _vec(1.0, 0.0))
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(_vec(1.0, 0.0)), CONFIG, session_id, "trip?")

    everything = "\n".join(m["content"] for m in messages)
    assert "hiking in the Alps" in everything and "relevant past memories" in everything.lower()


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

    # Facts are never dropped — they live in the system prompt.
    system_content = messages[0]["content"]
    assert "important fact about the user" in system_content

    # Newest message must be in the chat history context window.
    joined = "".join(m["content"] for m in messages)
    assert "NEWEST" in joined

    # Oldest messages should NOT appear in the chat history (non-system) part.
    # OLDEST-0 is the first user message; it IS preserved in the system prompt as
    # the "original task" anchor — that's the intended behaviour. But OLDEST-1
    # through OLDEST-19 (excluding 0) should be gone from chat history.
    history_content = "".join(m["content"] for m in messages if m["role"] != "system")
    assert "OLDEST-5" not in history_content   # a mid-range old message is dropped
    assert "OLDEST-10" not in history_content  # and another


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


# ── Stale "assumed" facts ─────────────────────────────────────────────────────

def _age_fact(store, content: str, timestamp: str) -> None:
    store.conn.execute(
        "UPDATE facts SET updated_at = ? WHERE content = ?", (timestamp, content)
    )
    store.conn.commit()


def test_old_assumed_fact_is_flagged_unverified(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("probably likes jazz", source_session_id=None, confidence="assumed")
    _age_fact(store, "probably likes jazz", "2020-01-01T00:00:00+00:00")
    session_id = store.create_session()

    content = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")[0]["content"]
    assert "probably likes jazz ⚠" in content


def test_verified_fact_is_never_flagged(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("lives in Oslo", source_session_id=None, confidence="verified")
    _age_fact(store, "lives in Oslo", "2020-01-01T00:00:00+00:00")
    session_id = store.create_session()

    content = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")[0]["content"]
    assert "lives in Oslo" in content
    assert "lives in Oslo ⚠" not in content


def test_fresh_assumed_fact_is_not_flagged(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("maybe plays guitar", source_session_id=None, confidence="assumed")
    session_id = store.create_session()

    content = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")[0]["content"]
    assert "maybe plays guitar ⚠" not in content


def test_naive_timestamp_still_flags_a_stale_assumed_fact(tmp_path):
    """A tz-naive updated_at raised TypeError and silently disabled flagging."""
    store, vectors = _make(tmp_path)
    store.add_fact("guesswork fact", source_session_id=None, confidence="assumed")
    _age_fact(store, "guesswork fact", "2020-01-01T00:00:00")  # no UTC offset
    session_id = store.create_session()

    content = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")[0]["content"]
    assert "guesswork fact ⚠" in content


def test_unparseable_timestamp_does_not_break_the_turn(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("junk ts fact", source_session_id=None, confidence="assumed")
    _age_fact(store, "junk ts fact", "not-a-timestamp")
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(), CONFIG, session_id, "hi")
    assert "junk ts fact" in messages[0]["content"]


# ── Compaction summary counts against the char budget ────────────────────────

class _CompactingLLM(FakeLLM):
    """Embeds like FakeLLM; returns a long fixed summary from chat()."""

    SUMMARY = "SUMMARY-CHUNK " * 120  # ~1680 chars

    def chat(self, messages, model, stream=False, **kwargs):
        yield self.SUMMARY


def test_stored_compaction_summary_counts_against_the_char_budget(tmp_path):
    """The summary is part of the system prompt and grows without bound.

    Leaving it out of the trim estimate meant every later turn overshot
    context_char_budget by the full length of the summary.
    """
    store, vectors = _make(tmp_path)
    session_id = store.create_session()
    store.add_message(session_id, "user", "the original task")
    for i in range(30):
        store.add_message(session_id, "assistant", f"a{i} " * 60)
        store.add_message(session_id, "user", f"u{i} " * 60)

    budget = 4000
    cfg = {
        **CONFIG,
        "models": {"embed": "nomic-embed-text", "background": "bg"},
        "memory": {"recall_k": 6, "context_char_budget": budget},
    }
    llm = _CompactingLLM()

    # First turn writes the summary; later turns must budget around it.
    build_context(store, vectors, llm, cfg, session_id, "q0")
    stored, _through = store.get_session_compaction(session_id)
    assert stored and len(stored) > 1000, "expected a substantial stored summary"

    messages = build_context(store, vectors, llm, cfg, session_id, "q1")
    total = sum(len(m["content"]) for m in messages)
    # Allow a small fixed overhead (persona + formatting guidance + the
    # preserved original task); the summary itself must be accounted for.
    assert total < budget + 600, f"context overshot the budget by {total - budget} chars"
    assert stored in messages[0]["content"]


# --- the history budget follows the chosen context ----------------------------------------------


def test_history_budget_is_the_configured_one_without_a_context_choice():
    from agent.memory.recall import history_char_budget

    assert history_char_budget({"memory": {"context_char_budget": 24000}}) == 24000


def test_history_budget_grows_with_a_big_context_but_never_shrinks():
    from agent import gpumode
    from agent.memory.recall import history_char_budget

    cfg = {"memory": {"context_char_budget": 24000}, "models": {"chat_num_ctx": 16384}}
    assert history_char_budget(cfg) == 26214                       # 16 384 tokens * 1.6 (leaves half the window for the running turn)
    gpumode._state["ctx"] = 204800
    assert history_char_budget(cfg) == 327680
    gpumode._state["ctx"] = 2048
    assert history_char_budget({"memory": {"context_char_budget": 24000}}) == 24000


def test_a_big_tool_result_survives_into_the_next_turn_with_a_big_context(tmp_path):
    """The real failure: two long files read in one turn, then "okay, go on" - with a fixed
    24k-char history budget the file contents were already gone."""
    from agent import gpumode

    store, vectors = _make(tmp_path)
    sid = store.create_session()
    store.add_message(sid, "user", "fasse die dateien zusammen")
    store.add_message(sid, "tool", "DATEI-INHALT " * 15000)             # ~195 000 chars
    store.add_message(sid, "assistant", "Ich habe beide gelesen.")
    cfg = {**CONFIG, "models": {**CONFIG["models"], "chat_num_ctx": 16384}}

    small = build_context(store, vectors, FakeLLM(), cfg, sid, "okay, go on")
    assert not any("DATEI-INHALT" in m["content"] for m in small)         # dropped under the old budget
    gpumode._state["ctx"] = 204800
    big = build_context(store, vectors, FakeLLM(), cfg, sid, "okay, go on")
    assert any("DATEI-INHALT" in m["content"] for m in big)


# ----------------------------------------------------------------------------- cache-friendly prompt structure


def test_recalled_memories_sit_right_before_the_new_message_not_in_the_system_prompt(tmp_path):
    store, vectors = _make(tmp_path)
    vectors.add("message", 5, "we talked about hiking in the Alps", _vec(1.0, 0.0))
    session_id = store.create_session()

    messages = build_context(store, vectors, FakeLLM(_vec(1.0, 0.0)), CONFIG, session_id, "trip?")

    assert "hiking" not in messages[0]["content"] and "past memories" not in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "trip?"}
    assert messages[-2]["role"] == "system" and "hiking in the Alps" in messages[-2]["content"]


def test_system_prompt_and_history_are_identical_between_turns_whatever_is_recalled(tmp_path):
    store, vectors = _make(tmp_path)
    vectors.add("message", 5, "we talked about hiking in the Alps", _vec(1.0, 0.0))
    vectors.add("message", 6, "my favourite colour is a deep forest green", _vec(0.0, 1.0))
    session_id = store.create_session()
    store.add_message(session_id, "user", "hello there")
    store.add_message(session_id, "assistant", "hi, how can I help?")

    a = build_context(store, vectors, FakeLLM(_vec(1.0, 0.0)), CONFIG, session_id, "trip?")
    b = build_context(store, vectors, FakeLLM(_vec(0.0, 1.0)), CONFIG, session_id, "colour?")

    shared = [m for m in a[:-2]]
    assert shared == b[:-2] and len(shared) == 3          # system + the two history messages: byte-identical, so the server cache holds
    assert a[-2] != b[-2]


def test_recalled_chat_lines_say_who_said_them_and_when(tmp_path):
    store, vectors = _make(tmp_path)
    session_id = store.create_session()
    mid = store.add_message(session_id, "user", "I am planning a trip to Norway next summer")
    vectors.add("message", mid, "I am planning a trip to Norway next summer", _vec(1.0, 0.0))

    messages = build_context(store, vectors, FakeLLM(_vec(1.0, 0.0)), CONFIG, store.create_session(), "norway?")

    note = messages[-2]["content"]
    assert "[you said on 20" in note and "trip to Norway" in note


def test_between_turn_summary_uses_the_conversation_prompt_not_the_coding_work_log(tmp_path):
    from agent.memory import recall, worklog

    seen = []

    class Llm:
        def chat(self, messages, **kw):
            seen.append(messages[0]["content"])
            return iter(["TOPICS: x"])

    out = recall._compact_history(Llm(), {"models": {"background": "bg"}}, None, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert out == "TOPICS: x" and seen[0] == worklog.CHAT_SUMMARY_SYSTEM and "WORK LOG" not in seen[0]
