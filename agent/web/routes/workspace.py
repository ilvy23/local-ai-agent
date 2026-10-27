"""Code-agent runs (read-only), project notes and the prompt studio for the GUI.

The coding agent itself is started from the terminal (`agent code`); these routes only let the GUI show what it did:
saved sessions (agent.codeagent.sessionstore) and its event log (agent.codeagent.eventlog), so a run can be inspected
from the phone. Project notes and the prompt studio wrap agent.projectmem and agent.promptlab.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent import projectmem, promptlab
from agent.codeagent import sessionstore, timing
from agent.web.deps import get_client, get_config, get_store

router = APIRouter(prefix="/api", tags=["workspace"])

_LEDGER_MAX_CHARS = 20_000
_EVENTS_MAX = 1000
_EVENT_SLACK_S = 120          # an event log entry is attributed to a saved session if it falls in its time span +- this


def _meta(m: sessionstore.Meta) -> dict[str, Any]:
    return {"id": m.id, "root": m.root, "title": m.title, "created": m.created, "updated": m.updated, "steps": m.steps,
            "model": m.model, "state": m.shown_state(), "files": m.files}


def _find(session_id: str) -> sessionstore.Meta:
    meta, why = sessionstore.find(session_id)
    if meta is None:
        raise HTTPException(404, why)
    return meta


@router.get("/code/sessions")
def code_sessions() -> list[dict[str, Any]]:
    return [_meta(m) for m in sessionstore.list_sessions()]


@router.get("/code/sessions/{session_id}")
def code_session(session_id: str) -> dict[str, Any]:
    meta = _find(session_id)
    try:
        data = sessionstore.load(meta.path)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    ledger = json.dumps(data.get("ledger") or {}, default=str)
    return {**_meta(meta), "requests": data.get("requests") or [], "num_ctx": data.get("num_ctx"),
            "messages": len(data.get("messages") or []), "plan": data.get("plan"), "pending": data.get("pending") or {},
            "ledger": json.loads(ledger) if len(ledger) <= _LEDGER_MAX_CHARS else {"truncated": True, "size": len(ledger)}}


@router.get("/code/sessions/{session_id}/events")
def code_events(session_id: str, kind: str | None = None, limit: int = 200) -> dict[str, Any]:
    """Event-log lines that belong to this saved session: same folder name, inside its time span. The log names its
    sessions `<folder>-<start time>`, not by the saved id, so attribution is by folder and time."""
    meta = _find(session_id)
    folder = Path(meta.root).name
    lo, hi = meta.created - _EVENT_SLACK_S, meta.updated + _EVENT_SLACK_S
    events = [e for e in timing.read_log() if str(e.get("session", "")).startswith(f"{folder}-") and lo <= float(e.get("ts") or 0) <= hi]
    breakdown = None
    try:
        breakdown = dataclasses.asdict(timing.breakdown(events))
    except Exception:  # noqa: BLE001 - the timing summary is a bonus
        pass
    if kind:
        events = [e for e in events if e.get("kind") == kind]
    return {"total": len(events), "events": events[-max(1, min(limit, _EVENTS_MAX)):], "timing": breakdown}


# --------------------------------------------------------------------------------------------------- project notes


@router.get("/projects")
def projects(store=Depends(get_store)) -> list[dict[str, Any]]:
    projectmem.ProjectMemory(store.conn)                       # makes sure the table exists
    rows = store.conn.execute("SELECT project, COUNT(*) FROM project_notes GROUP BY project ORDER BY MAX(created_at) DESC").fetchall()
    return [{"key": r[0], "name": Path(r[0]).name, "notes": r[1]} for r in rows]


@router.get("/projects/notes")
def project_notes(key: str, store=Depends(get_store)) -> list[dict[str, Any]]:
    return [{"id": i, "note": n} for i, n in projectmem.ProjectMemory(store.conn).notes(key, limit=200)]


@router.delete("/projects/notes")
def clear_project_notes(key: str, store=Depends(get_store)) -> dict[str, int]:
    return {"forgotten": projectmem.ProjectMemory(store.conn).clear(key)}


# --------------------------------------------------------------------------------------------------- prompt studio


class PromptIn(BaseModel):
    text: str
    answers: str = ""
    strength: str = "heavy"


def _check(body: PromptIn) -> str:
    text = body.text.strip()
    if not text or len(text) > 20_000:
        raise HTTPException(422, "Give 1-20000 characters of text.")
    return text


@router.post("/prompt/improve")
async def prompt_improve(body: PromptIn, client=Depends(get_client), config=Depends(get_config)) -> dict[str, str]:
    text = _check(body)
    return {"text": await asyncio.to_thread(promptlab.improve, client, config, text, body.answers)}


@router.post("/prompt/anonymize")
async def prompt_anonymize(body: PromptIn, client=Depends(get_client), config=Depends(get_config)) -> dict[str, str]:
    text = _check(body)
    if body.strength not in ("light", "heavy", "poison", "max"):
        raise HTTPException(422, "strength must be light, heavy, poison or max.")
    return {"text": await asyncio.to_thread(promptlab.anonymize, client, config, text, body.strength)}
