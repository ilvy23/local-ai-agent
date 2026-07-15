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


    # seed the index at the default 768 dims
    idx = VectorIndex(store)
    assert idx.available and idx.DIM == 768
    idx.add("note", 1, "hallo wie geht es dir", [1.0] + [0.0] * 767)
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
