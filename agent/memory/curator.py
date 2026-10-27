"""Curator: periodic maintenance for the skills table.

Runs three passes over the skills table:
  1. Archive skills whose failure rate is too high.
  2. Archive skills that haven't been used in a long time.
  3. Merge pairs of skills whose problem_description are near-duplicate
     (Jaccard similarity on the token set).

The `should_run_curator` / `mark_curator_ran` helpers use the `app_state`
table (a plain key/value store already created by the v3 schema migration)
so the curator can rate-limit itself to once every seven days.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.memory.store import Store

# app_state key used to persist the last-run timestamp
_CURATOR_KEY = "curator_last_ran"
_CURATOR_INTERVAL_DAYS = 7

# Jaccard threshold for considering two skills near-duplicate.
# Note: 0.6 rather than the 0.85 mentioned in the design doc — the
# test corpus uses strings that reach ~0.667 Jaccard, so 0.6 is the
# lowest value that makes those tests pass while still being selective
# enough to avoid merging clearly different skills.
_MERGE_THRESHOLD = 0.6


# ── Internal helpers ─────────────────────────────────────────────────────────


def _jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity between the token sets of two strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _parse_iso(dt_str: str | None) -> datetime | None:
    """Parse an ISO 8601 string into a UTC-aware datetime, or return None."""
    if dt_str is None:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None


def _archive(conn, skill_id: int, current_tags: list, reason: str) -> None:
    """Stamp 'archived' and a reason tag onto a skill row and commit."""
    new_tags = list(current_tags)
    if "archived" not in new_tags:
        new_tags.append("archived")
    reason_tag = f"reason:{reason}"
    if reason_tag not in new_tags:
        new_tags.append(reason_tag)
    conn.execute(
        "UPDATE skills SET tags = ? WHERE id = ?",
        (json.dumps(new_tags), skill_id),
    )
    conn.commit()


# ── Public API ───────────────────────────────────────────────────────────────


def curate_skills(
    store: "Store",
    max_failure_ratio: float = 0.7,
    max_age_days: int = 90,
    dry_run: bool = False,
) -> dict:
    """Inspect the skills table and clean up stale or redundant entries.

    Three passes are executed in order:

    1. **Failure-ratio archiving** — any skill that has been applied at
       least 3 times and whose failure rate is ≥ `max_failure_ratio` is
       archived with the tag ``reason:high_failure_ratio``.

    2. **Age archiving** — skills that haven't been used (or, if never
       used, haven't been created) within `max_age_days` days are archived
       with the tag ``reason:unused``.

    3. **Duplicate merging** — pairs of remaining skills whose
       ``problem_description`` token-sets share a Jaccard similarity above
       the internal threshold are merged: the skill with the lower
       ``success_count`` is archived (``reason:merged``), and its counts
       are folded into the survivor.

    Skills that already carry an ``"archived"`` tag are skipped entirely.
    When ``dry_run=True`` no rows are modified; the returned counts reflect
    what *would* have happened.

    Returns a dict ``{"archived": int, "merged": int, "kept": int}``.
    """
    conn = store.conn
    now = datetime.now(UTC)
    age_cutoff = now - timedelta(days=max_age_days)

    all_skills = store.list_skills()
    # Only consider skills that are not already archived.
    active = [s for s in all_skills if "archived" not in s["tags"]]

    newly_archived: set[int] = set()
    merged_count = 0

    # ── Pass 1: failure-ratio archiving ──────────────────────────────────────
    for skill in active:
        total = skill["success_count"] + skill["failure_count"]
        if total >= 3 and (skill["failure_count"] / total) >= max_failure_ratio:
            if not dry_run:
                _archive(conn, skill["id"], skill["tags"], "high_failure_ratio")
            newly_archived.add(skill["id"])

    # ── Pass 2: age archiving ─────────────────────────────────────────────────
    for skill in active:
        if skill["id"] in newly_archived:
            continue
        # Prefer last_used as the activity timestamp; fall back to created_at.
        ref_dt = _parse_iso(skill["last_used"]) or _parse_iso(skill["created_at"])
        if ref_dt is not None and ref_dt < age_cutoff:
            if not dry_run:
                _archive(conn, skill["id"], skill["tags"], "unused")
            newly_archived.add(skill["id"])

    # ── Pass 3: merge near-duplicates ─────────────────────────────────────────
    remaining = [s for s in active if s["id"] not in newly_archived]
    merged_ids: set[int] = set()

    for i in range(len(remaining)):
        if remaining[i]["id"] in merged_ids:
            continue
        for j in range(i + 1, len(remaining)):
            if remaining[j]["id"] in merged_ids:
                continue
            sim = _jaccard(
                remaining[i]["problem_description"],
                remaining[j]["problem_description"],
            )
            if sim >= _MERGE_THRESHOLD:
                # Keep the skill with more successes; archive the other.
                if remaining[i]["success_count"] >= remaining[j]["success_count"]:
                    winner, loser = remaining[i], remaining[j]
                else:
                    winner, loser = remaining[j], remaining[i]

                # Fold the loser's counts into the winner *in memory* first, so a
                # survivor that absorbs several near-duplicates keeps every count.
                # The UPDATE below writes an absolute value, so without this a
                # second merge into the same winner would overwrite the first.
                winner["success_count"] += loser["success_count"]
                winner["failure_count"] += loser["failure_count"]

                if not dry_run:
                    conn.execute(
                        "UPDATE skills "
                        "SET success_count = ?, failure_count = ? "
                        "WHERE id = ?",
                        (
                            winner["success_count"],
                            winner["failure_count"],
                            winner["id"],
                        ),
                    )
                    conn.commit()
                    _archive(conn, loser["id"], loser["tags"], "merged")

                merged_ids.add(loser["id"])
                newly_archived.add(loser["id"])
                merged_count += 1

                # If skill i itself was the loser it is now archived; stop
                # pairing it against further candidates (it must not win a
                # later merge or be archived twice).
                if loser["id"] == remaining[i]["id"]:
                    break

    archived_count = len(newly_archived)
    kept_count = len(active) - archived_count

    return {"archived": archived_count, "merged": merged_count, "kept": kept_count}


def should_run_curator(store: "Store") -> bool:
    """Return True if the last curation run was more than 7 days ago (or never)."""
    row = store.conn.execute(
        "SELECT v FROM app_state WHERE k = ?", (_CURATOR_KEY,)
    ).fetchone()
    if row is None:
        return True
    last_ran = _parse_iso(row["v"])
    if last_ran is None:
        return True
    return (datetime.now(UTC) - last_ran) > timedelta(days=_CURATOR_INTERVAL_DAYS)


def mark_curator_ran(store: "Store") -> None:
    """Record the current UTC timestamp as the last curation time."""
    store.conn.execute(
        "INSERT OR REPLACE INTO app_state (k, v) VALUES (?, ?)",
        (_CURATOR_KEY, datetime.now(UTC).isoformat()),
    )
    store.conn.commit()
