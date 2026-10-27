import json
import pytest
from datetime import datetime, timedelta, timezone


@pytest.fixture
def store_with_skills(tmp_path):
    from agent.memory.store import Store

    s = Store(tmp_path / "curator_test.db")
    yield s
    s.close()


def _insert_skill(
    store,
    title,
    problem,
    success=0,
    failure=0,
    last_used=None,
    created_days_ago=1,
):
    created = (
        datetime.now(timezone.utc) - timedelta(days=created_days_ago)
    ).isoformat()
    lu = (
        (datetime.now(timezone.utc) - timedelta(days=last_used)).isoformat()
        if last_used
        else None
    )
    store.conn.execute(
        "INSERT INTO skills "
        "(title, problem_description, solution_steps, tools_used, "
        "success_count, failure_count, last_used, created_at, tags) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (title, problem, "[]", "[]", success, failure, lu, created, "[]"),
    )
    store.conn.commit()


def test_archives_high_failure_ratio(store_with_skills):
    from agent.memory.curator import curate_skills

    _insert_skill(store_with_skills, "bad", "broken thing", success=1, failure=5)
    result = curate_skills(store_with_skills)
    assert result["archived"] >= 1


def test_dry_run_makes_no_changes(store_with_skills):
    from agent.memory.curator import curate_skills

    _insert_skill(store_with_skills, "bad", "broken thing", success=0, failure=5)
    before = store_with_skills.conn.execute(
        "SELECT tags FROM skills"
    ).fetchone()[0]
    curate_skills(store_with_skills, dry_run=True)
    after = store_with_skills.conn.execute(
        "SELECT tags FROM skills"
    ).fetchone()[0]
    assert before == after


def test_keeps_healthy_skills(store_with_skills):
    from agent.memory.curator import curate_skills

    _insert_skill(store_with_skills, "good", "working thing", success=10, failure=1)
    result = curate_skills(store_with_skills)
    assert result["kept"] >= 1


def test_archives_old_unused_skill(store_with_skills):
    from agent.memory.curator import curate_skills

    _insert_skill(store_with_skills, "stale", "old thing", created_days_ago=100)
    result = curate_skills(store_with_skills)
    assert result["archived"] >= 1


def test_merges_similar_skills(store_with_skills):
    from agent.memory.curator import curate_skills

    # Two near-identical skills
    _insert_skill(
        store_with_skills,
        "skill_a",
        "fix the broken authentication system",
        success=5,
    )
    _insert_skill(
        store_with_skills,
        "skill_b",
        "fix broken authentication system problem",
        success=2,
    )
    result = curate_skills(store_with_skills)
    assert result["merged"] >= 1


def test_should_run_curator_initially_true(store_with_skills):
    from agent.memory.curator import should_run_curator

    assert should_run_curator(store_with_skills) is True


def test_mark_curator_ran_prevents_immediate_rerun(store_with_skills):
    from agent.memory.curator import mark_curator_ran, should_run_curator

    mark_curator_ran(store_with_skills)
    assert should_run_curator(store_with_skills) is False
