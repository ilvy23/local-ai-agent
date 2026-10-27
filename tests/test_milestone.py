"""Tests for agent/modes/milestone.py — interactive coding-phase review gates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.modes import (
    AutoSession,
    MilestoneSession,
    auto_mode_unlocked,
    mark_milestone_mode_used,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _console() -> MagicMock:
    c = MagicMock()
    c.print = MagicMock()
    return c


# ── MilestoneSession.complete_milestone ──────────────────────────────────────

def test_complete_milestone_increments_counter() -> None:
    session = MilestoneSession(_console())
    assert session.milestone_count == 0
    session.complete_milestone("did a thing")
    assert session.milestone_count == 1
    session.complete_milestone("did another thing")
    assert session.milestone_count == 2


def test_complete_milestone_prints_header() -> None:
    console = _console()
    session = MilestoneSession(console)
    session.complete_milestone("wrote the code")
    printed = " ".join(str(c) for call in console.print.call_args_list for c in call.args)
    assert "Milestone 1" in printed
    assert "wrote the code" in printed


# ── MilestoneSession.await_review ────────────────────────────────────────────

@pytest.mark.parametrize("response", ["", "ok", "OK", "y", "yes", "continue"])
def test_await_review_returns_true_on_proceed(response: str) -> None:
    console = _console()
    console.input = MagicMock(return_value=response)
    session = MilestoneSession(console)
    session.complete_milestone("step done")
    assert session.await_review() is True


@pytest.mark.parametrize("response", ["stop", "s", "n", "no", "abort", "quit", "q"])
def test_await_review_returns_false_on_stop(response: str) -> None:
    console = _console()
    console.input = MagicMock(return_value=response)
    session = MilestoneSession(console)
    session.complete_milestone("step done")
    assert session.await_review() is False


def test_await_review_returns_false_on_eof() -> None:
    console = _console()
    console.input = MagicMock(side_effect=EOFError)
    session = MilestoneSession(console)
    session.complete_milestone("step done")
    assert session.await_review() is False


def test_await_review_returns_false_on_keyboard_interrupt() -> None:
    console = _console()
    console.input = MagicMock(side_effect=KeyboardInterrupt)
    session = MilestoneSession(console)
    session.complete_milestone("step done")
    assert session.await_review() is False


# ── AutoSession ───────────────────────────────────────────────────────────────

def test_auto_session_await_review_always_returns_true() -> None:
    console = _console()
    session = AutoSession(console)
    session.complete_milestone("step")
    assert session.await_review() is True


def test_auto_session_await_review_never_calls_input() -> None:
    console = _console()
    console.input = MagicMock()
    session = AutoSession(console)
    session.complete_milestone("step")
    session.await_review()
    console.input.assert_not_called()


def test_auto_session_prints_auto_continuing() -> None:
    console = _console()
    session = AutoSession(console)
    session.complete_milestone("step")
    session.await_review()
    printed = " ".join(str(c) for call in console.print.call_args_list for c in call.args)
    assert "auto" in printed.lower()


def test_auto_session_inherits_milestone_count() -> None:
    console = _console()
    session = AutoSession(console)
    session.complete_milestone("a")
    session.complete_milestone("b")
    assert session.milestone_count == 2


# ── Mode-B unlock tracking ────────────────────────────────────────────────────

@pytest.fixture()
def tmp_store(tmp_path):
    """A real MemoryStore backed by a temp DB."""
    from agent.memory.store import Store
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_auto_mode_unlocked_false_initially(tmp_store) -> None:
    assert auto_mode_unlocked(tmp_store) is False


def test_mark_milestone_mode_used_unlocks_auto(tmp_store) -> None:
    mark_milestone_mode_used(tmp_store)
    assert auto_mode_unlocked(tmp_store) is True


def test_mark_milestone_mode_used_idempotent(tmp_store) -> None:
    mark_milestone_mode_used(tmp_store)
    mark_milestone_mode_used(tmp_store)  # second call must not raise
    assert auto_mode_unlocked(tmp_store) is True


# ── Integration: abort propagates as KeyboardInterrupt ───────────────────────

def test_abort_on_stop_raises_keyboard_interrupt() -> None:
    """The coding CLI raises KeyboardInterrupt when await_review returns False."""
    console = _console()
    console.input = MagicMock(return_value="stop")
    session = MilestoneSession(console)
    session.complete_milestone("generated a patch")
    abort = not session.await_review()
    assert abort  # caller must raise KeyboardInterrupt when this is True
