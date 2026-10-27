"""A fact that fails to embed must not become permanently unrecallable.

Embedding happens right after a chat turn, while the chat model still holds the
GPU — on a smaller card Ollama can answer 500 because the embedding model won't
fit. That's transient, but the old code stored the fact and never indexed it, so
it never reached `memory_items` and semantic recall could never see it again
(not even `reembed`, which reads from `memory_items`).
"""

from __future__ import annotations

import pytest

from agent.memory.distill import embed_with_retry, repair_unembedded_facts
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

CONFIG = {"models": {"embed": "bge-m3"}}


class FlakyLLM:
    """Fails `fail_times` times, then succeeds."""

    def __init__(self, fail_times: int, dim: int = 8):
        self.fail_times = fail_times
        self.calls = 0
        self.dim = dim

    def embed(self, texts, model):  # noqa: ARG002
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("500 Internal Server Error")
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "agent.db")
    yield s
    s.close()


def test_retry_rides_out_a_transient_failure(monkeypatch):
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _s: None)
    llm = FlakyLLM(fail_times=2)
    vec = embed_with_retry(llm, "user has a sister named Lena", "bge-m3")
    assert vec is not None and llm.calls == 3


def test_gives_up_quietly_when_it_never_recovers(monkeypatch):
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _s: None)
    llm = FlakyLLM(fail_times=99)
    assert embed_with_retry(llm, "x", "bge-m3") is None  # returns None, doesn't raise


def test_repair_indexes_a_fact_that_missed_its_embedding(store, monkeypatch):
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _s: None)
    vectors = VectorIndex(store)

    # A fact stored while embedding was down: in `facts`, absent from the index.
    fact_id = store.add_fact("works night shifts as a nurse", source_session_id=None)
    assert store.get_active_facts()  # the user can see it...
    assert vectors.search([1.0] + [0.0] * 7, k=5) == []  # ...but recall cannot

    fixed = repair_unembedded_facts(store, vectors, FlakyLLM(0), CONFIG)
    assert fixed == 1

    hits = vectors.search([1.0] + [0.0] * 7, k=5)
    assert [h[0] for h in hits] == ["works night shifts as a nurse"]
    assert [h[2] for h in hits] == [fact_id]


def test_repair_is_idempotent(store, monkeypatch):
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _s: None)
    vectors = VectorIndex(store)
    store.add_fact("likes trains", source_session_id=None)

    assert repair_unembedded_facts(store, vectors, FlakyLLM(0), CONFIG) == 1
    assert repair_unembedded_facts(store, vectors, FlakyLLM(0), CONFIG) == 0  # nothing left
    assert len(vectors.search([1.0] + [0.0] * 7, k=5)) == 1  # not duplicated


def test_repair_leaves_the_fact_for_next_time_if_still_failing(store, monkeypatch):
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _s: None)
    vectors = VectorIndex(store)
    store.add_fact("still broken", source_session_id=None)

    assert repair_unembedded_facts(store, vectors, FlakyLLM(99), CONFIG) == 0
    assert store.get_active_facts()  # the fact is kept, not dropped
    # and a later run, once the GPU has room, still fixes it
    assert repair_unembedded_facts(store, vectors, FlakyLLM(0), CONFIG) == 1


def test_repair_skips_inactive_facts(store, monkeypatch):
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _s: None)
    vectors = VectorIndex(store)
    fid = store.add_fact("forgotten thing", source_session_id=None)
    store.deactivate_fact(fid)
    assert repair_unembedded_facts(store, vectors, FlakyLLM(0), CONFIG) == 0
