"""Tests for the skill persistence system (agent/memory/skills.py + store.py v11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.memory.skills import (
    delete_skill,
    find_similar_skill,
    increment_failure,
    increment_success,
    list_skills,
    save_skill,
)
from agent.memory.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "skills_test.db")


# ── Store-layer tests ────────────────────────────────────────────────────────

class TestSkillsTable:
    def test_skills_table_created(self, store: Store) -> None:
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "skills" in tables

    def test_add_and_get_skill(self, store: Store) -> None:
        skill_id = store.add_skill(
            title="Parse CSV",
            problem_description="Parse a CSV file into a list of dicts",
            solution_steps=["open file", "use csv.DictReader", "return list"],
            tools_used=["read_file"],
            tags=["csv", "parsing"],
        )
        assert skill_id > 0
        skill = store.get_skill(skill_id)
        assert skill is not None
        assert skill["title"] == "Parse CSV"
        assert skill["solution_steps"] == ["open file", "use csv.DictReader", "return list"]
        assert skill["tools_used"] == ["read_file"]
        assert skill["tags"] == ["csv", "parsing"]
        assert skill["success_count"] == 0
        assert skill["failure_count"] == 0

    def test_get_skill_not_found(self, store: Store) -> None:
        assert store.get_skill(9999) is None

    def test_list_skills_empty(self, store: Store) -> None:
        assert store.list_skills() == []

    def test_list_skills_returns_all(self, store: Store) -> None:
        store.add_skill("A", "problem a", ["step"], [], [])
        store.add_skill("B", "problem b", ["step"], [], [])
        skills = store.list_skills()
        assert len(skills) == 2
        titles = {s["title"] for s in skills}
        assert titles == {"A", "B"}

    def test_list_skills_ordered_by_success(self, store: Store) -> None:
        id1 = store.add_skill("Low", "problem", ["step"], [], [])
        id2 = store.add_skill("High", "problem", ["step"], [], [])
        store.update_skill_success(id2)
        store.update_skill_success(id2)
        skills = store.list_skills()
        assert skills[0]["id"] == id2

    def test_update_skill_success(self, store: Store) -> None:
        skill_id = store.add_skill("S", "p", ["step"], [], [])
        store.update_skill_success(skill_id)
        store.update_skill_success(skill_id)
        skill = store.get_skill(skill_id)
        assert skill["success_count"] == 2
        assert skill["last_used"] is not None

    def test_update_skill_failure(self, store: Store) -> None:
        skill_id = store.add_skill("S", "p", ["step"], [], [])
        store.update_skill_failure(skill_id)
        skill = store.get_skill(skill_id)
        assert skill["failure_count"] == 1

    def test_delete_skill(self, store: Store) -> None:
        skill_id = store.add_skill("Delete me", "p", ["step"], [], [])
        store.delete_skill(skill_id)
        assert store.get_skill(skill_id) is None
        assert store.list_skills() == []

    def test_search_skills_like(self, store: Store) -> None:
        store.add_skill("CSV Parser", "parse csv files", ["step"], [], ["csv"])
        store.add_skill("JSON Writer", "write json output", ["step"], [], ["json"])
        results = store.search_skills_like("csv")
        assert len(results) == 1
        assert results[0]["title"] == "CSV Parser"


# ── skills.py API tests ──────────────────────────────────────────────────────

class TestSkillsAPI:
    def test_save_and_list(self, store: Store) -> None:
        skill_id = save_skill(store, "My Skill", "A problem", ["do x", "do y"])
        assert skill_id > 0
        all_skills = list_skills(store)
        assert len(all_skills) == 1
        assert all_skills[0]["title"] == "My Skill"

    def test_save_with_optional_fields(self, store: Store) -> None:
        skill_id = save_skill(
            store, "T", "P", ["s"], tools=["shell"], tags=["automation"]
        )
        skill = store.get_skill(skill_id)
        assert skill["tools_used"] == ["shell"]
        assert skill["tags"] == ["automation"]

    def test_save_with_none_optional_fields(self, store: Store) -> None:
        skill_id = save_skill(store, "T", "P", ["s"], tools=None, tags=None)
        skill = store.get_skill(skill_id)
        assert skill["tools_used"] == []
        assert skill["tags"] == []

    def test_increment_success(self, store: Store) -> None:
        sid = save_skill(store, "T", "P", ["s"])
        increment_success(store, sid)
        assert store.get_skill(sid)["success_count"] == 1

    def test_increment_failure(self, store: Store) -> None:
        sid = save_skill(store, "T", "P", ["s"])
        increment_failure(store, sid)
        assert store.get_skill(sid)["failure_count"] == 1

    def test_delete_skill(self, store: Store) -> None:
        sid = save_skill(store, "T", "P", ["s"])
        delete_skill(store, sid)
        assert list_skills(store) == []


# ── find_similar_skill tests ─────────────────────────────────────────────────

class TestFindSimilarSkill:
    def test_empty_store_returns_empty(self, store: Store) -> None:
        assert find_similar_skill(store, "parse csv file") == []

    def test_finds_matching_skill(self, store: Store) -> None:
        save_skill(store, "CSV Parser", "parse csv files into dicts", ["use DictReader"])
        results = find_similar_skill(store, "parse csv files")
        assert len(results) == 1
        assert results[0]["title"] == "CSV Parser"

    def test_no_match_returns_empty(self, store: Store) -> None:
        save_skill(store, "CSV Parser", "parse csv files", ["step"])
        results = find_similar_skill(store, "send email notification")
        assert results == []

    def test_ignores_short_words(self, store: Store) -> None:
        save_skill(store, "CSV", "csv", ["step"])
        # "the" and "a" are too short to match
        results = find_similar_skill(store, "the a")
        assert results == []

    def test_ranks_by_overlap_then_success_count(self, store: Store) -> None:
        # "compressed" (10 chars) differentiates them; all filter words are >= 4 chars
        id_weak = save_skill(store, "Weak match", "parse files from disk", ["step"])
        id_strong = save_skill(
            store, "Strong match", "parse compressed files from archive", ["step"]
        )
        increment_success(store, id_weak)
        # strong matches parse+compressed+files (score 3); weak matches parse+files (score 2)
        results = find_similar_skill(store, "parse compressed files")
        assert results[0]["id"] == id_strong

    def test_respects_limit(self, store: Store) -> None:
        for i in range(5):
            save_skill(store, f"Skill {i}", "parse csv data", ["step"])
        results = find_similar_skill(store, "parse csv data", limit=2)
        assert len(results) == 2

    def test_matches_in_tags(self, store: Store) -> None:
        save_skill(store, "Some Skill", "unrelated description", ["step"], tags=["automation"])
        results = find_similar_skill(store, "automation scripts")
        assert len(results) == 1

    def test_matches_in_title(self, store: Store) -> None:
        save_skill(store, "JSON Formatter", "format output", ["step"])
        results = find_similar_skill(store, "JSON output format")
        assert len(results) == 1
