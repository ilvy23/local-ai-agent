"""Facts that change replace the old ones; recall only returns what is actually relevant."""

import math

import pytest

from agent import gpumode
from agent.memory import distill
from agent.memory.distill import distill_session, judge_overlap
from agent.memory.recall import build_context
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

CONFIG = {
    "models": {"background": "m:4b", "embed": "e"},
    "persona": {"style": "You are Agent."},
    "memory": {"recall_k": 6, "context_char_budget": 24000, "recall_min_similarity": 0.55},
}
DIM = VectorIndex.DIM


def unit(*values):
    vec = list(values) + [0.0] * (DIM - len(values))
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def angle(cos):
    """Unit vector whose cosine with unit(1, 0) is `cos`."""
    return unit(cos, math.sqrt(1 - cos * cos))


class ScriptedLLM:
    """chat(): the first call returns the extraction JSON, later calls the judge's verdicts."""

    def __init__(self, extraction, verdicts=(), vectors=None):
        self.extraction, self.verdicts, self.vectors = extraction, list(verdicts), vectors or {}
        self.chat_calls = []

    def chat(self, messages, model, stream=True, **kwargs):
        self.chat_calls.append(messages[0]["content"][:20])
        yield self.extraction if len(self.chat_calls) == 1 else (self.verdicts.pop(0) if self.verdicts else "DIFFERENT")

    def embed(self, texts, model, **kwargs):
        return [self.vectors.get(t, unit(1.0, 0.0)) for t in texts]


def _setup(tmp_path):
    store = Store(tmp_path / "a.db")
    return store, VectorIndex(store)


def _session(store):
    sid = store.create_session()
    store.add_message(sid, "user", "irgendwas")
    return sid


def _seed(store, vectors, text, vec):
    fid = store.add_fact(text, source_session_id=None)
    vectors.add("fact", fid, text, vec)
    return fid


def _active(store):
    return [f["content"] for f in store.get_active_facts()]


# ----------------------------------------------------------------------------- the judge


class _Say:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, model, stream=True, **kwargs):
        if isinstance(self.reply, Exception):
            raise self.reply
        yield self.reply


@pytest.mark.parametrize("reply, expected", [
    ("UPDATE", "update"), ("same.", "same"), ("  Different\n", "different"),
    ("<think>hmm, München statt Hamburg</think>UPDATE", "update"),
    ("I am not sure", "different"), ("", "different"),
])
def test_judge_reads_the_verdict_and_defaults_to_keeping_both(reply, expected):
    assert judge_overlap(_Say(reply), CONFIG, "lives in Hamburg", "lives in Munich") == expected


def test_a_failing_judge_keeps_both_facts():
    assert judge_overlap(_Say(RuntimeError("down")), CONFIG, "a", "b") == "different"


# ----------------------------------------------------------------------------- distillation


def test_an_update_replaces_the_old_fact(tmp_path):
    store, vectors = _setup(tmp_path)
    old = _seed(store, vectors, "lives in Hamburg", unit(1.0, 0.0))
    llm = ScriptedLLM('[{"text": "lives in Munich", "confidence": "verified"}]', ["UPDATE"],
                      {"lives in Munich": angle(0.8)})
    added = distill_session(store, vectors, llm, CONFIG, _session(store))
    assert added == ["lives in Munich"] and _active(store) == ["lives in Munich"]
    assert store.conn.execute("SELECT active FROM facts WHERE id = ?", (old,)).fetchone()[0] == 0
    assert vectors.max_cosine_similarity(unit(1.0, 0.0), "fact") < 0.9     # old one left the vector index too


def test_a_paraphrase_is_not_stored_again(tmp_path):
    store, vectors = _setup(tmp_path)
    _seed(store, vectors, "likes Black Metal and Gothic Rock", unit(1.0, 0.0))
    llm = ScriptedLLM('["favourite music is Black Metal"]', ["SAME"], {"favourite music is Black Metal": angle(0.8)})
    assert distill_session(store, vectors, llm, CONFIG, _session(store)) == []
    assert _active(store) == ["likes Black Metal and Gothic Rock"]


def test_a_different_fact_on_the_same_topic_is_added_next_to_the_old_one(tmp_path):
    store, vectors = _setup(tmp_path)
    _seed(store, vectors, "has a cat called Luna", unit(1.0, 0.0))
    llm = ScriptedLLM('["has a dog called Rex"]', ["DIFFERENT"], {"has a dog called Rex": angle(0.75)})
    distill_session(store, vectors, llm, CONFIG, _session(store))
    assert set(_active(store)) == {"has a cat called Luna", "has a dog called Rex"}


def test_a_near_identical_fact_is_skipped_without_asking_the_model(tmp_path):
    store, vectors = _setup(tmp_path)
    _seed(store, vectors, "has a cat called Luna", unit(1.0, 0.0))
    llm = ScriptedLLM('["owns a cat named Luna"]', [], {"owns a cat named Luna": angle(0.97)})
    assert distill_session(store, vectors, llm, CONFIG, _session(store)) == []
    assert len(llm.chat_calls) == 1                                          # only the extraction call


def test_an_unrelated_fact_is_added_without_a_judge_call(tmp_path):
    store, vectors = _setup(tmp_path)
    _seed(store, vectors, "has a cat called Luna", unit(1.0, 0.0))
    llm = ScriptedLLM('["plays chess"]', [], {"plays chess": angle(0.3)})
    assert distill_session(store, vectors, llm, CONFIG, _session(store)) == ["plays chess"]
    assert len(llm.chat_calls) == 1


def test_the_prompt_keeps_setup_and_game_facts_and_stated_changes():
    text = distill._PROMPT
    assert "operating system, hardware, software and games" in text
    assert "plays chess" in text and "moved to Munich" in text
    assert "(what they SAY about their own setup" in text


# ----------------------------------------------------------------------------- vector helper


def test_similar_returns_the_best_matches_first_and_honours_the_floor(tmp_path):
    store, vectors = _setup(tmp_path)
    for text, cos in (("far", 0.2), ("close", 0.85), ("mid", 0.65)):
        _seed(store, vectors, text, angle(cos))
    found = vectors.similar(unit(1.0, 0.0), "fact", k=3, min_similarity=0.6)
    assert [t for _, t, _ in found] == ["close", "mid"]
    assert found[0][2] == pytest.approx(0.85, abs=1e-3)
    assert vectors.similar(unit(1.0, 0.0), "message") == []


# ----------------------------------------------------------------------------- recall cutoff


def _recall(tmp_path, items, config=CONFIG):
    store, vectors = _setup(tmp_path)
    for kind, text, cos in items:
        vectors.add(kind, None, text, angle(cos))
    sid = store.create_session()
    messages = build_context(store, vectors, ScriptedLLM("[]", vectors={"frage": unit(1.0, 0.0)}), config, sid, "frage")
    return "\n".join(m["content"] for m in messages)      # recalled memories sit in a note before the user message, not in the system prompt


def test_unrelated_memories_are_not_shown(tmp_path):
    prompt = _recall(tmp_path, [("fact", "lives in Munich", 0.7), ("fact", "plays chess", 0.3), ("message", "ein ganz anderes Thema hier", 0.4)])
    assert "lives in Munich" in prompt and "plays chess" not in prompt and "anderes Thema" not in prompt


def test_the_cutoff_can_be_switched_off(tmp_path):
    cfg = {**CONFIG, "memory": {**CONFIG["memory"], "recall_min_similarity": 0}}
    prompt = _recall(tmp_path, [("fact", "plays chess", 0.3)], cfg)
    assert "plays chess" in prompt


def test_tiny_chat_lines_like_ja_are_never_recalled(tmp_path):
    prompt = _recall(tmp_path, [("message", "ja", 0.95), ("message", "ich höre am liebsten Black Metal", 0.9)])
    assert "- ja" not in prompt and "Black Metal" in prompt


def test_a_big_context_does_not_change_the_cutoff(tmp_path):
    gpumode._state["ctx"] = 204800
    assert "plays chess" not in _recall(tmp_path, [("fact", "plays chess", 0.3)])
