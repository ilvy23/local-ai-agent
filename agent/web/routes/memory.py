"""Memory, skills and the audit log for the GUI. Thin wrappers over agent.memory.store.Store."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.memory.vectors import VectorIndex
from agent.web.deps import get_client, get_config, get_store

router = APIRouter(prefix="/api", tags=["memory"])

_STALE_HOURS_KEY = "reverify_assumed_after_hours"


class FactIn(BaseModel):
    content: str
    kind: str = "personal"


def _fact(row: Any, stale_hours: float) -> dict[str, Any]:
    from datetime import UTC, datetime

    stale = False
    if row["confidence"] == "assumed" and row["updated_at"]:
        try:
            updated = datetime.fromisoformat(row["updated_at"])
            updated = updated if updated.tzinfo else updated.replace(tzinfo=UTC)
            stale = (datetime.now(UTC) - updated).total_seconds() / 3600 > stale_hours
        except ValueError:
            pass
    content = row["content"]
    origin = "web" if "(web, " in content else "user" if row["kind"] == "personal" else "distilled"
    return {"id": row["id"], "content": content, "kind": row["kind"], "confidence": row["confidence"], "stale": stale,
            "origin": origin, "created_at": row["created_at"], "updated_at": row["updated_at"]}


@router.get("/memory/facts")
def list_facts(kind: str | None = None, q: str = "", store=Depends(get_store), config=Depends(get_config)) -> list[dict[str, Any]]:
    sql = "SELECT id, content, kind, confidence, created_at, updated_at FROM facts WHERE active = 1"
    args: list[Any] = []
    if kind in ("personal", "incidental"):
        sql += " AND kind = ?"
        args.append(kind)
    if q.strip():
        sql += " AND content LIKE ?"
        args.append(f"%{q.strip()}%")
    rows = store.conn.execute(sql + " ORDER BY id DESC LIMIT 500", args).fetchall()
    hours = float((config.get("memory") or {}).get(_STALE_HOURS_KEY, 24))
    return [_fact(r, hours) for r in rows]


@router.post("/memory/facts")
def add_fact(body: FactIn, store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    content = body.content.strip()
    if not content or len(content) > 1500:
        raise HTTPException(422, "A fact needs 1-1500 characters.")
    if body.kind not in ("personal", "incidental"):
        raise HTTPException(422, "kind must be personal or incidental.")
    fact_id = store.add_fact(content, source_session_id=None, kind=body.kind)
    vectors = VectorIndex(store)
    if vectors.available:
        try:
            vectors.add("fact", fact_id, content, client.embed([content], model=config["models"]["embed"])[0])
        except Exception:  # noqa: BLE001 - an unindexed fact is repaired later; the fact itself is saved
            pass
    row = store.conn.execute("SELECT id, content, kind, confidence, created_at, updated_at FROM facts WHERE id = ?", (fact_id,)).fetchone()
    return _fact(row, 24)


@router.delete("/memory/facts/{fact_id}")
def forget_fact(fact_id: int, store=Depends(get_store)) -> dict[str, bool]:
    if store.conn.execute("SELECT 1 FROM facts WHERE id = ? AND active = 1", (fact_id,)).fetchone() is None:
        raise HTTPException(404, "No such fact.")
    store.deactivate_fact(fact_id)
    return {"ok": True}


@router.get("/skills")
def list_skills(store=Depends(get_store)) -> list[dict[str, Any]]:
    return store.list_skills()


@router.get("/skills/{skill_id}")
def get_skill(skill_id: int, store=Depends(get_store)) -> dict[str, Any]:
    skill = store.get_skill(skill_id)
    if skill is None:
        raise HTTPException(404, "No such skill.")
    return skill


@router.delete("/skills/{skill_id}")
def delete_skill(skill_id: int, store=Depends(get_store)) -> dict[str, bool]:
    if store.get_skill(skill_id) is None:
        raise HTTPException(404, "No such skill.")
    store.delete_skill(skill_id)
    return {"ok": True}


@router.get("/audit")
def audit(kind: str | None = None, q: str = "", limit: int = 100, store=Depends(get_store)) -> list[dict[str, Any]]:
    rows = store.get_audit_log(limit=max(1, min(limit, 500)), kind=kind if kind in ("tool", "shell") else None)
    needle = q.strip().lower()
    return [r for r in rows if not needle or needle in str(r.get("detail", "")).lower() or needle in str(r.get("result", "")).lower()]
