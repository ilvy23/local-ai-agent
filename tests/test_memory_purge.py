"""Regression tests for Store's memory-purge paths (delete_session,
delete_all_sessions, deactivate_fact) actually cleaning up the semantic
recall index, not just the relational tables.

This exact area produced a real bug two review rounds in a row on the
night these tests were written: deleting only `memory_items` and leaving
`memory_vectors` behind isn't a no-op, it's actively worse than doing
nothing - VectorIndex.search() runs its KNN search inside `memory_vectors`
FIRST and only joins against `memory_items` afterward, so an orphaned
vector still occupies one of the k nearest-neighbour slots and silently
starves out a real, unrelated result. One session delete was enough to
turn a healthy 8-result search into a 0-result search for facts that were
never touched. These tests exist so that regression can't come back
unnoticed a third time.
"""

from pathlib import Path

from agent.memory.store import Store
from agent.memory.vectors import VectorIndex


def _vec(*values: float) -> list[float]:
    """Pad a short vector out to the index dimension with zeros."""
    return list(values) + [0.0] * (VectorIndex.DIM - len(values))


def _orphan_count(store: Store) -> int:
    return store._conn.execute(
        "SELECT COUNT(*) FROM memory_vectors WHERE rowid NOT IN (SELECT id FROM memory_items)"
    ).fetchone()[0]


def test_delete_session_does_not_blind_search_for_unrelated_content(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    index = VectorIndex(store)
    assert index.available

    survivor_session = store.create_session()
    survivor_msg_id = store.add_message(survivor_session, "user", "survivor text")
    index.add("message", survivor_msg_id, "survivor text", _vec(1.0, 0.0, 0.0))

    victim_session = store.create_session()
    victim_msg_id = store.add_message(victim_session, "user", "victim text")
    index.add("message", victim_msg_id, "victim text", _vec(0.0, 1.0, 0.0))

    before = index.search(_vec(1.0, 0.0, 0.0), k=8)
    assert before[0][0] == "survivor text"

    store.delete_session(victim_session)

    after = index.search(_vec(1.0, 0.0, 0.0), k=8)
    assert [r[0] for r in after] == ["survivor text"], (
        "deleting an unrelated session blinded search() for surviving content - "
        "this is the exact orphaned-vector regression"
    )

    # The victim's own text must be gone too, not just correctly ranked.
    victim_search = index.search(_vec(0.0, 1.0, 0.0), k=8)
    assert "victim text" not in [r[0] for r in victim_search]

    assert _orphan_count(store) == 0, "delete_session left orphaned memory_vectors rows behind"
    store.close()


def test_delete_all_sessions_purges_all_message_vectors(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    index = VectorIndex(store)

    for i in range(3):
        session_id = store.create_session()
        msg_id = store.add_message(session_id, "user", f"message {i}")
        index.add("message", msg_id, f"message {i}", _vec(float(i), 0.0))

    assert len(index.search(_vec(0.0, 0.0), k=8)) == 3

    store.delete_all_sessions()

    assert index.search(_vec(0.0, 0.0), k=8) == []
    assert _orphan_count(store) == 0
    store.close()


def test_delete_session_does_not_touch_facts(tmp_path: Path) -> None:
    """Facts are long-term memory, not scoped to the session that produced
    them - deleting a session must not delete or blind recall for facts."""
    store = Store(tmp_path / "agent.db")
    index = VectorIndex(store)

    session_id = store.create_session()
    fact_id = store.add_fact("the user's cat is named Mochi", source_session_id=session_id)
    index.add("fact", fact_id, "the user's cat is named Mochi", _vec(1.0, 1.0))

    store.delete_session(session_id)

    assert store.get_active_facts()[0]["content"] == "the user's cat is named Mochi"
    assert [r[0] for r in index.search(_vec(1.0, 1.0), k=8)] == ["the user's cat is named Mochi"]
    store.close()


def test_deactivate_fact_purges_its_vector_too(tmp_path: Path) -> None:
    """A "forgotten" fact must stop being recalled via semantic search, not
    just get excluded from the plain active-facts listing - VectorIndex's
    search() has no join back to facts.active at all, so without an
    explicit purge a forgotten fact kept getting injected into every
    turn's context regardless of the flag."""
    store = Store(tmp_path / "agent.db")
    index = VectorIndex(store)

    keep_id = store.add_fact("likes tea", source_session_id=None)
    forget_id = store.add_fact("likes coffee", source_session_id=None)
    index.add("fact", keep_id, "likes tea", _vec(1.0, 0.0))
    index.add("fact", forget_id, "likes coffee", _vec(0.0, 1.0))

    store.deactivate_fact(forget_id)

    assert [f["content"] for f in store.get_active_facts()] == ["likes tea"]
    remaining = [r[0] for r in index.search(_vec(0.0, 1.0), k=8)]
    assert "likes coffee" not in remaining, (
        "a deactivated fact is still returned by semantic search - "
        "recall.py would keep injecting it into context"
    )
    assert _orphan_count(store) == 0
    store.close()
