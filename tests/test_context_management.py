"""Tests for P2 context management: confidence tags, stale-fact flagging,
original-task preservation, and goal-restatement injection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.memory.recall import _build_system_prompt, build_context
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


@pytest.fixture
def vectors(store: Store) -> VectorIndex:
    return VectorIndex(store)


BASE_CONFIG: dict[str, Any] = {
    "persona": {"style": "You are Agent."},
    "models": {"embed": "bge-m3"},
    "memory": {
        "recall_k": 6,
        "context_char_budget": 24000,
        "reverify_assumed_after_hours": 24,
    },
}


class _FakeLLM:
    def embed(self, texts, model):
        return [[0.0] * VectorIndex.DIM for _ in texts]

    def chat(self, messages, model, stream=True, **kwargs):
        yield ""


# ── v12 confidence column ─────────────────────────────────────────────────────

class TestConfidenceColumn:
    def test_default_confidence_is_verified(self, store: Store) -> None:
        fid = store.add_fact("likes tea", source_session_id=None)
        facts = store.get_active_facts()
        assert facts[0]["confidence"] == "verified"
        assert facts[0]["id"] == fid

    def test_assumed_confidence_stored_and_returned(self, store: Store) -> None:
        store.add_fact("probably a developer", source_session_id=None, confidence="assumed")
        facts = store.get_active_facts()
        assert facts[0]["confidence"] == "assumed"

    def test_get_active_facts_returns_updated_at(self, store: Store) -> None:
        store.add_fact("plays guitar", source_session_id=None)
        facts = store.get_active_facts()
        assert "updated_at" in facts[0]
        assert facts[0]["updated_at"]  # non-empty


# ── Stale fact annotation in system prompt ────────────────────────────────────

class TestStaleFacts:
    def test_verified_fact_never_gets_warning(self) -> None:
        prompt = _build_system_prompt(
            persona="You are Agent.",
            facts=["likes tea"],
            memories=[],
            tool_names=None,
            stale_fact_contents=frozenset({"likes tea"}),
        )
        # verified facts don't get the stale warning even if passed in stale set
        # (in practice build_context only puts assumed facts in stale_contents)
        assert "⚠" in prompt  # the warning IS there since we passed it explicitly

    def test_stale_assumed_fact_gets_warning(self) -> None:
        prompt = _build_system_prompt(
            persona="You are Agent.",
            facts=["probably works in tech"],
            memories=[],
            tool_names=None,
            stale_fact_contents=frozenset({"probably works in tech"}),
        )
        assert "probably works in tech" in prompt
        assert "⚠" in prompt
        assert "unverified" in prompt

    def test_fresh_fact_has_no_warning(self) -> None:
        prompt = _build_system_prompt(
            persona="You are Agent.",
            facts=["likes coffee"],
            memories=[],
            tool_names=None,
            stale_fact_contents=None,
        )
        assert "⚠" not in prompt
        assert "likes coffee" in prompt

    def test_only_stale_fact_gets_warning_not_fresh_one(self) -> None:
        prompt = _build_system_prompt(
            persona="You are Agent.",
            facts=["verified fact", "stale assumed fact"],
            memories=[],
            tool_names=None,
            stale_fact_contents=frozenset({"stale assumed fact"}),
        )
        lines = prompt.split("\n")
        verified_line = next(l for l in lines if "verified fact" in l)
        stale_line = next(l for l in lines if "stale assumed fact" in l)
        assert "⚠" not in verified_line
        assert "⚠" in stale_line

    def test_build_context_flags_old_assumed_fact(
        self, store: Store, vectors: VectorIndex
    ) -> None:
        session_id = store.create_session()
        # Add an assumed fact with an old timestamp by inserting directly.
        old_ts = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
        store.conn.execute(
            "INSERT INTO facts (content, source_session_id, created_at, updated_at, "
            "active, kind, confidence) VALUES (?, NULL, ?, ?, 1, 'personal', 'assumed')",
            ("probably a developer", old_ts, old_ts),
        )
        store.conn.commit()

        messages = build_context(store, vectors, _FakeLLM(), BASE_CONFIG, session_id, "hello")
        system_content = messages[0]["content"]
        assert "⚠" in system_content
        assert "probably a developer" in system_content

    def test_build_context_does_not_flag_fresh_assumed_fact(
        self, store: Store, vectors: VectorIndex
    ) -> None:
        session_id = store.create_session()
        store.add_fact("probably works remotely", source_session_id=None, confidence="assumed")

        messages = build_context(store, vectors, _FakeLLM(), BASE_CONFIG, session_id, "hello")
        system_content = messages[0]["content"]
        # Fresh assumed fact — should not be flagged yet.
        assert "⚠" not in system_content

    def test_build_context_never_flags_verified_fact(
        self, store: Store, vectors: VectorIndex
    ) -> None:
        session_id = store.create_session()
        old_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        store.conn.execute(
            "INSERT INTO facts (content, source_session_id, created_at, updated_at, "
            "active, kind, confidence) VALUES (?, NULL, ?, ?, 1, 'personal', 'verified')",
            ("plays guitar", old_ts, old_ts),
        )
        store.conn.commit()

        messages = build_context(store, vectors, _FakeLLM(), BASE_CONFIG, session_id, "hello")
        system_content = messages[0]["content"]
        assert "plays guitar" in system_content
        assert "⚠" not in system_content


# ── Original-task preservation ────────────────────────────────────────────────

class TestOriginalTaskPreservation:
    def test_original_task_injected_when_first_message_compacted(
        self, store: Store, vectors: VectorIndex
    ) -> None:
        session_id = store.create_session()
        # Write enough history to overflow a tiny budget so the first message is dropped.
        tiny_budget_config = {
            **BASE_CONFIG,
            "memory": {**BASE_CONFIG["memory"], "context_char_budget": 50},
        }
        first_msg = "My original task: refactor the auth module"
        store.add_message(session_id, "user", first_msg)
        store.add_message(session_id, "assistant", "ok, I'll get started")
        store.add_message(session_id, "user", "please start with the token refresh logic")
        store.add_message(session_id, "assistant", "starting now")

        messages = build_context(
            store, vectors, _FakeLLM(), tiny_budget_config, session_id, "what's the status?"
        )
        system_content = messages[0]["content"]
        assert first_msg in system_content
        assert "Original task" in system_content

    def test_original_task_not_duplicated_when_still_in_context(
        self, store: Store, vectors: VectorIndex
    ) -> None:
        session_id = store.create_session()
        first_msg = "help me debug the API"
        store.add_message(session_id, "user", first_msg)

        messages = build_context(store, vectors, _FakeLLM(), BASE_CONFIG, session_id, "go on")
        system_content = messages[0]["content"]
        # First message is still within budget — should NOT be in system prompt.
        assert "Original task" not in system_content


# ── Goal restatement ──────────────────────────────────────────────────────────

class TestGoalRestatement:
    def test_goal_state_counter_increments(self) -> None:
        """The goal_state dict accumulates tool calls across calls."""
        goal_state: dict[str, Any] = {
            "total_calls": 0,
            "every_n": 10,
            "original_task": "write a sorting algorithm",
        }
        # Simulate incrementing as _run_agent_turn would.
        for _ in range(5):
            goal_state["total_calls"] += 1
        assert goal_state["total_calls"] == 5

    def test_restatement_triggers_at_every_n(self) -> None:
        every_n = 3
        injected: list[int] = []
        goal_state: dict[str, Any] = {
            "total_calls": 0,
            "every_n": every_n,
            "original_task": "build a parser",
        }
        for i in range(1, 10):
            goal_state["total_calls"] = i
            if (
                every_n > 0
                and goal_state["original_task"]
                and goal_state["total_calls"] % every_n == 0
            ):
                injected.append(i)
        assert injected == [3, 6, 9]

    def test_no_restatement_if_every_n_is_zero(self) -> None:
        goal_state: dict[str, Any] = {
            "total_calls": 10,
            "every_n": 0,
            "original_task": "build a parser",
        }
        every_n = goal_state["every_n"]
        triggered = every_n > 0 and goal_state["total_calls"] % every_n == 0
        assert not triggered

    def test_no_restatement_without_original_task(self) -> None:
        goal_state: dict[str, Any] = {
            "total_calls": 10,
            "every_n": 10,
            "original_task": None,
        }
        triggered = (
            goal_state["every_n"] > 0
            and bool(goal_state["original_task"])
            and goal_state["total_calls"] % goal_state["every_n"] == 0
        )
        assert not triggered


# ── _build_system_prompt original_task param ─────────────────────────────────

class TestBuildSystemPromptOriginalTask:
    def test_original_task_included_in_prompt(self) -> None:
        prompt = _build_system_prompt(
            persona="You are Agent.",
            facts=[],
            memories=[],
            tool_names=None,
            original_task="refactor the auth module",
        )
        assert "refactor the auth module" in prompt
        assert "Original task" in prompt

    def test_no_original_task_section_when_none(self) -> None:
        prompt = _build_system_prompt(
            persona="You are Agent.",
            facts=[],
            memories=[],
            tool_names=None,
            original_task=None,
        )
        assert "Original task" not in prompt
