from __future__ import annotations

import time

import pytest

from agent.budget import Budget, BudgetExhausted


def test_actions_limit_raises():
    b = Budget(max_actions=3, max_tokens=99999, max_seconds=9999)
    b.tick()
    b.tick()
    b.tick()
    with pytest.raises(BudgetExhausted, match="action"):
        b.tick()


def test_tokens_limit_raises():
    b = Budget(max_actions=9999, max_tokens=100, max_seconds=9999)
    b.tick(tokens_used=60)
    with pytest.raises(BudgetExhausted, match="token"):
        b.tick(tokens_used=50)


def test_time_limit_raises(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    b = Budget(max_actions=9999, max_tokens=99999, max_seconds=10)
    b.tick()
    clock[0] = 11.0
    with pytest.raises(BudgetExhausted, match="time"):
        b.tick()


def test_is_exhausted_false_before_limit():
    b = Budget(max_actions=10, max_tokens=1000, max_seconds=9999)
    b.tick(tokens_used=50)
    assert b.is_exhausted() is False


def test_is_exhausted_true_after_actions_reached():
    b = Budget(max_actions=2, max_tokens=9999, max_seconds=9999)
    b.tick()
    b.tick()
    assert b.is_exhausted() is True


def test_summary_format():
    b = Budget(max_actions=50, max_tokens=50000, max_seconds=1800)
    b.tick(tokens_used=200)
    s = b.summary()
    assert "1/50 actions" in s
    assert "200/50000 tokens" in s
    assert "/1800s" in s


def test_from_config_reads_values():
    cfg = {"budget": {"max_actions": 7, "max_tokens": 300, "max_seconds": 60}}
    b = Budget.from_config(cfg)
    assert b._max_actions == 7
    assert b._max_tokens == 300
    assert b._max_seconds == 60


def test_from_config_fallback_defaults():
    b = Budget.from_config({})
    assert b._max_actions == 50
    assert b._max_tokens == 50_000
    assert b._max_seconds == 1800


def test_multiple_ticks_accumulate():
    b = Budget(max_actions=100, max_tokens=9999, max_seconds=9999)
    for _ in range(5):
        b.tick(tokens_used=10)
    assert b.actions == 5
    assert b.tokens == 50


def test_from_config_tolerates_null_budget_block():
    """YAML `budget:` with nothing under it parses as None, not {}."""
    b = Budget.from_config({"budget": None})
    assert b._max_actions == 50
    b.tick()  # must not raise a TypeError comparing int to None


def test_from_config_tolerates_null_limit_values():
    """YAML `max_actions:` with no value parses as None."""
    b = Budget.from_config({"budget": {"max_actions": None, "max_seconds": None}})
    assert b._max_actions == 50
    assert b._max_seconds == 1800
    b.tick()


def test_from_config_tolerates_none_config():
    b = Budget.from_config(None)
    assert b._max_tokens == 50_000


def test_from_config_keeps_an_explicit_zero_limit():
    b = Budget.from_config({"budget": {"max_actions": 0}})
    assert b._max_actions == 0
    with pytest.raises(BudgetExhausted):
        b.tick()
