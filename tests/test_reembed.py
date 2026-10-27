from __future__ import annotations

import pytest

from agent.memory.reembed import reembed_all
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex


class FakeLLM:
    """Embeds each text into `dim` dims; deterministic from text length."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, texts, model):  # noqa: ARG002
        return [[float(len(t) % 7)] + [0.0] * (self.dim - 1) for t in texts]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "agent.db")
    yield s
    s.close()


def test_reembed_rebuilds_at_new_dim(store, monkeypatch):


    # seed the index; the first embedding decides the size
    idx = VectorIndex(store)
    assert idx.available and idx.DIM is None  # nothing embedded yet
    idx.add("note", 1, "hallo wie geht es dir", [1.0] + [0.0] * 767)
    assert idx.DIM == 768  # adopted from the vector it was actually given
    idx.add("fact", 2, "der Nutzer mag Zug fahren", [0.0, 1.0] + [0.0] * 766)

    # re-embed with a "1024-dim" model
    result = reembed_all(store, FakeLLM(1024), {"background": {}}, "bge-m3", batch=1)
    assert result["dim"] == 1024
    assert result["reembedded"] == 2 and result["total"] == 2

    # the stored dim is persisted, and a fresh index picks it up
    fresh = VectorIndex(store)
    assert fresh.DIM == 1024
    # search works at the new dim (query must match the new size)
    q = [1.0] + [0.0] * 1023
    hits = fresh.search(q, k=2)
    assert len(hits) == 2
    # add() now accepts 1024-dim and rejects the old 768
    with pytest.raises(ValueError):
        fresh.add("fact", 3, "x", [0.0] * 768)


def test_reembed_empty_index_ok(store, monkeypatch):

    VectorIndex(store)  # create the table
    result = reembed_all(store, FakeLLM(512), {"background": {}}, "m", batch=8)
    assert result == {"reembedded": 0, "failed": 0, "dim": 512, "total": 0}


def test_fresh_db_adopts_the_models_dimension(store):
    """Regression: a fresh install used to hardcode 768 and then blow up on the
    first message, because the default embed model (bge-m3) returns 1024."""
    idx = VectorIndex(store)
    # Searching before anything is embedded must not explode — there's nothing
    # to match against yet, whatever size the query is.
    assert idx.search([0.0] * 1024, k=3) == []

    idx.add("fact", 1, "user likes trains", [1.0] + [0.0] * 1023)
    assert idx.DIM == 1024

    # A new index on the same DB picks the size back up, and search works.
    fresh = VectorIndex(store)
    assert fresh.DIM == 1024
    assert len(fresh.search([1.0] + [0.0] * 1023, k=3)) == 1


def test_stale_empty_table_from_an_old_version_is_healed(store):
    """An older build created the vec table from a hardcoded 768 guess. If it's
    empty, drop it rather than failing every query — nothing is lost."""
    VectorIndex(store)  # loads the sqlite-vec extension onto the connection
    store.conn.execute("CREATE VIRTUAL TABLE memory_vectors USING vec0(embedding float[768])")
    store.conn.commit()

    idx = VectorIndex(store)
    assert idx.DIM is None  # the guessed, empty table was thrown away

    idx.add("fact", 1, "x", [1.0] + [0.0] * 1023)  # a 1024-dim model
    assert idx.DIM == 1024
    assert len(idx.search([1.0] + [0.0] * 1023, k=1)) == 1


def test_populated_legacy_table_is_kept_not_dropped(store):
    """If the old table actually has vectors in it, never destroy them."""
    idx = VectorIndex(store)
    idx.add("fact", 1, "keep me", [1.0] + [0.0] * 767)
    store.conn.execute("DELETE FROM app_state WHERE k = 'embed_dim'")  # unrecorded, as before
    store.conn.commit()

    fresh = VectorIndex(store)
    assert fresh.DIM == 768  # detected from the table, data intact
    assert len(fresh.search([1.0] + [0.0] * 767, k=1)) == 1


def test_query_at_the_wrong_size_explains_itself(store):
    idx = VectorIndex(store)
    idx.add("fact", 1, "x", [1.0] + [0.0] * 1023)
    with pytest.raises(ValueError, match="reembed"):
        idx.search([0.0] * 768, k=1)
