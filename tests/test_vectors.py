import logging

import pytest

from companion.memory.store import Store
from companion.memory.vectors import VectorIndex


def _vec(*values: float) -> list[float]:
    """Pad a short vector out to the index dimension with zeros."""
    return list(values) + [0.0] * (VectorIndex.DIM - len(values))


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "companion.db")
    yield s
    s.close()


def test_add_and_search_returns_nearest_first(store):
    index = VectorIndex(store)
    assert index.available

    index.add("fact", 1, "cats", _vec(1.0, 0.0, 0.0))
    index.add("fact", 2, "dogs", _vec(0.0, 1.0, 0.0))

    results = index.search(_vec(0.9, 0.1, 0.0), k=2)

    assert [r[0] for r in results] == ["cats", "dogs"]
    text, kind, ref_id, distance = results[0]
    assert kind == "fact"
    assert ref_id == 1
    assert distance < results[1][3]


def test_search_kinds_filter(store):
    index = VectorIndex(store)
    index.add("fact", 1, "a fact", _vec(1.0, 0.0))
    index.add("message", 2, "a message", _vec(1.0, 0.0))

    results = index.search(_vec(1.0, 0.0), k=8, kinds=["fact"])

    assert [r[1] for r in results] == ["fact"]
    assert [r[0] for r in results] == ["a fact"]


def test_search_empty_index_returns_empty(store):
    index = VectorIndex(store)
    assert index.search(_vec(1.0), k=8) == []


def test_add_wrong_dimension_raises_value_error(store):
    index = VectorIndex(store)
    assert index.available

    with pytest.raises(ValueError, match="768"):
        index.add("fact", 1, "bad vector", [1.0, 2.0, 3.0])


def test_graceful_degrade_when_extension_unavailable(store, monkeypatch, caplog):
    import companion.memory.vectors as vectors_module

    def _boom(conn):
        raise RuntimeError("no extension here")

    monkeypatch.setattr(vectors_module, "_load_extension", _boom)

    with caplog.at_level(logging.WARNING):
        index = VectorIndex(store)

    assert index.available is False
    # add is a no-op, search returns [] — chat must keep working.
    index.add("fact", 1, "x", _vec(1.0))
    assert index.search(_vec(1.0), k=8) == []
    assert any("semantic" in r.message.lower() or "vector" in r.message.lower()
               for r in caplog.records)
