"""Skill persistence — save and recall reusable solution patterns.

Each skill captures a class of problem, the solution steps, and usage stats.
`find_similar_skill` uses multi-word keyword scoring today; the design leaves
room for an embedding upgrade once the skills table grows large enough to
warrant it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.memory.store import Store


def save_skill(
    store: Store,
    title: str,
    problem: str,
    steps: list[str],
    tools: list[str] | None = None,
    tags: list[str] | None = None,
) -> int:
    """Persist a new skill and return its id."""
    return store.add_skill(title, problem, steps, tools or [], tags or [])


def find_similar_skill(
    store: Store,
    problem_description: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return up to `limit` skills whose title/description/tags overlap with
    `problem_description`, ranked by keyword overlap then success count.

    Words shorter than four characters are ignored to avoid noise from
    articles, prepositions, and common short tokens.
    """
    words = {w.lower() for w in problem_description.split() if len(w) >= 4}
    if not words:
        return []

    all_skills = store.list_skills()
    scored: list[tuple[int, dict[str, Any]]] = []
    for skill in all_skills:
        haystack = (
            skill["title"]
            + " "
            + skill["problem_description"]
            + " "
            + " ".join(skill["tags"])
        ).lower()
        score = sum(1 for w in words if w in haystack)
        if score > 0:
            scored.append((score, skill))

    scored.sort(key=lambda x: (-x[0], -x[1]["success_count"]))
    return [s for _, s in scored[:limit]]


def increment_success(store: Store, skill_id: int) -> None:
    store.update_skill_success(skill_id)


def increment_failure(store: Store, skill_id: int) -> None:
    store.update_skill_failure(skill_id)


def list_skills(store: Store) -> list[dict[str, Any]]:
    return store.list_skills()


def delete_skill(store: Store, skill_id: int) -> None:
    store.delete_skill(skill_id)
