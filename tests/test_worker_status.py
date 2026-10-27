"""Tests for agent.worker_status.AgentStatus and StatusRegistry."""
from __future__ import annotations

import pytest

from agent.worker_status import StatusRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the StatusRegistry singleton before and after each test so tests are isolated."""
    StatusRegistry._instance = None
    yield
    StatusRegistry._instance = None


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


def test_get_returns_singleton():
    """StatusRegistry.get() returns the same object on successive calls."""
    reg1 = StatusRegistry.get()
    reg2 = StatusRegistry.get()
    assert reg1 is reg2


# ---------------------------------------------------------------------------
# register / all_agents
# ---------------------------------------------------------------------------


def test_register_adds_agent():
    """register() makes the agent visible in all_agents()."""
    reg = StatusRegistry.get()
    reg.register("Agt-1", "qwen3.5:9b")
    agents = reg.all_agents()
    assert len(agents) == 1
    assert agents[0].agent_id == "Agt-1"
    assert agents[0].model == "qwen3.5:9b"


def test_all_agents_returns_all():
    """all_agents() returns every registered agent."""
    reg = StatusRegistry.get()
    reg.register("Agt-A", "model-a")
    reg.register("Agt-B", "model-b")
    reg.register("Agt-C", "model-c")
    agents = reg.all_agents()
    assert len(agents) == 3
    ids = {a.agent_id for a in agents}
    assert ids == {"Agt-A", "Agt-B", "Agt-C"}


def test_all_agents_returns_snapshot():
    """all_agents() returns a list copy — mutations to it don't affect the registry."""
    reg = StatusRegistry.get()
    reg.register("Agt-X", "model-x")
    snapshot = reg.all_agents()
    snapshot.clear()  # mutate the returned list
    assert len(reg.all_agents()) == 1  # registry unchanged


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_modifies_milestone_and_last_action():
    """update() applies milestone and last_action changes."""
    reg = StatusRegistry.get()
    reg.register("Agt-2", "llama3")
    reg.update("Agt-2", milestone="M2 Coding", last_action="write auth.py")
    agent = reg.all_agents()[0]
    assert agent.milestone == "M2 Coding"
    assert agent.last_action == "write auth.py"


def test_update_modifies_status():
    """update() changes the status field."""
    reg = StatusRegistry.get()
    reg.register("Agt-3", "llama3")
    reg.update("Agt-3", status="running")
    assert reg.all_agents()[0].status == "running"


def test_update_partial_leaves_other_fields_unchanged():
    """update() with only some kwargs leaves the rest untouched."""
    reg = StatusRegistry.get()
    reg.register("Agt-4", "llama3")
    reg.update("Agt-4", milestone="M1 Planning")
    agent = reg.all_agents()[0]
    assert agent.milestone == "M1 Planning"
    assert agent.last_action == ""   # default, untouched
    assert agent.status == "idle"    # default, untouched


def test_update_unknown_agent_is_noop():
    """update() on an unregistered agent_id silently does nothing."""
    reg = StatusRegistry.get()
    reg.update("ghost-99", status="running")  # should not raise


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------


def test_unregister_removes_agent():
    """unregister() drops the agent from the registry."""
    reg = StatusRegistry.get()
    reg.register("Agt-5", "llama3")
    assert len(reg.all_agents()) == 1
    reg.unregister("Agt-5")
    assert len(reg.all_agents()) == 0


def test_unregister_unknown_agent_is_noop():
    """unregister() on an absent agent_id silently does nothing."""
    reg = StatusRegistry.get()
    reg.unregister("does-not-exist")  # should not raise
    assert len(reg.all_agents()) == 0


# ---------------------------------------------------------------------------
# mark_review_pending / clear_review
# ---------------------------------------------------------------------------


def test_mark_review_pending_sets_flag():
    """mark_review_pending() sets review_pending=True and status='waiting_review'."""
    reg = StatusRegistry.get()
    reg.register("Agt-6", "llama3")
    assert not reg.all_agents()[0].review_pending
    reg.mark_review_pending("Agt-6")
    agent = reg.all_agents()[0]
    assert agent.review_pending is True
    assert agent.status == "waiting_review"


def test_clear_review_clears_flag():
    """clear_review() resets review_pending and restores status to 'running'."""
    reg = StatusRegistry.get()
    reg.register("Agt-7", "llama3")
    reg.mark_review_pending("Agt-7")
    reg.clear_review("Agt-7")
    agent = reg.all_agents()[0]
    assert agent.review_pending is False
    assert agent.status == "running"


# ---------------------------------------------------------------------------
# Snapshot isolation (thread-safety)
# ---------------------------------------------------------------------------


def test_all_agents_returns_detached_copies():
    """The returned rows must not be the live objects worker threads mutate."""
    reg = StatusRegistry.get()
    reg.register("Agt-Copy", "model")
    first = reg.all_agents()[0]
    reg.update("Agt-Copy", status="running", milestone="M9")
    assert first.status == "idle"       # the earlier snapshot is frozen
    assert first.milestone == ""
    fresh = reg.all_agents()[0]
    assert fresh.status == "running"    # a new snapshot sees the update
    assert fresh.milestone == "M9"


def test_mutating_a_snapshot_row_does_not_affect_the_registry():
    reg = StatusRegistry.get()
    reg.register("Agt-Mut", "model")
    row = reg.all_agents()[0]
    row.status = "tampered"
    assert reg.all_agents()[0].status == "idle"


def test_concurrent_updates_do_not_corrupt_snapshots():
    """Hammer update() from threads while snapshotting; rows stay self-consistent."""
    import threading

    reg = StatusRegistry.get()
    for i in range(5):
        reg.register(f"Agt-{i}", "model")

    stop = threading.Event()
    errors: list[BaseException] = []

    def churn() -> None:
        try:
            while not stop.is_set():
                for i in range(5):
                    reg.update(f"Agt-{i}", status="running", last_action="x")
                    reg.update(f"Agt-{i}", status="idle", last_action="y")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writers = [threading.Thread(target=churn, daemon=True) for _ in range(3)]
    for t in writers:
        t.start()
    try:
        for _ in range(200):
            rows = reg.all_agents()
            assert len(rows) == 5
            assert all(r.status in ("running", "idle") for r in rows)
    finally:
        stop.set()
        for t in writers:
            t.join(timeout=2)
    assert not errors
