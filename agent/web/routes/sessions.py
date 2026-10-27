"""Session lifecycle: create/list/resume, /clear and /notes equivalents, plus the Sessions
screen's own extras (rename, pin, export, fork-from-message, notes editor)."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from agent.memory.vectors import VectorIndex
from agent.web.deps import get_client, get_config, get_store
from agent.web.schemas import MessageOut, NotesOut, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut)
def create_session(store=Depends(get_store)) -> SessionOut:
    session_id = store.create_session()
    return SessionOut(id=session_id, title=None, started_at=None, message_count=0)


@router.get("", response_model=list[SessionOut])
def list_sessions(limit: int = 20, store=Depends(get_store)) -> list[SessionOut]:
    return [SessionOut(**row) for row in store.list_sessions(limit=limit)]


@router.delete("")
def delete_all_sessions(store=Depends(get_store)) -> dict[str, bool]:
    store.delete_all_sessions()
    return {"ok": True}


@router.delete("/{session_id}")
def delete_session(session_id: int, store=Depends(get_store)) -> dict[str, bool]:
    store.delete_session(session_id)
    return {"ok": True}


@router.get("/{session_id}/messages", response_model=list[MessageOut])
def get_messages(session_id: int, store=Depends(get_store)) -> list[MessageOut]:
    return [
        MessageOut(role=m["role"], content=m["content"]) for m in store.get_messages(session_id)
    ]


@router.post("/{session_id}/clear", response_model=SessionOut)
def clear_session(session_id: int, store=Depends(get_store)) -> SessionOut:
    """The `/clear` command's logic: wrap up the finished session (facts,
    auto-compaction), then start a new one. Mirrors tui.py's run_repl
    (search for "/clear") - tool_exchanges is empty here since the web
    route doesn't yet track per-turn tool exchanges across a session the
    way the CLI loop does; the end-of-session personal-fact distillation
    is queued and saved in the background, only the "learned via a tool this session" pass is skipped.
    """
    from agent.tui import _distill_on_exit
    from agent.web.console_shim import WebConsole

    client = get_client()
    config = get_config()
    vectors = VectorIndex(store)
    _distill_on_exit(store, vectors, client, config, session_id, WebConsole(), [])
    from agent.memory.queue import kick_drain

    kick_drain(store.db_path, client, config)         # facts are saved in the background, the request does not wait
    new_id = store.create_session()
    return SessionOut(id=new_id, title=None, started_at=None, message_count=0)


@router.get("/{session_id}/notes", response_model=NotesOut)
def get_notes(session_id: int, store=Depends(get_store)) -> NotesOut:
    return NotesOut(notes=store.get_working_notes(session_id))


@router.put("/{session_id}/notes", response_model=NotesOut)
def set_notes(session_id: int, body: NotesOut, store=Depends(get_store)) -> NotesOut:
    return NotesOut(notes=store.set_working_notes(session_id, body.notes or ""))


@router.delete("/{session_id}/notes")
def clear_notes(session_id: int, store=Depends(get_store)) -> dict[str, bool]:
    store.clear_working_notes(session_id)
    return {"ok": True}


def _get_session(store, session_id: int) -> dict:
    row = next((s for s in store.list_sessions(limit=10_000) if s["id"] == session_id), None)
    if row is None:
        raise HTTPException(404, "No such session.")
    return row


class SessionPatch(BaseModel):
    title: str | None = None
    pinned: bool | None = None


@router.patch("/{session_id}", response_model=SessionOut)
def update_session(session_id: int, body: SessionPatch, store=Depends(get_store)) -> SessionOut:
    """Rename and/or pin - the Sessions screen's "Rename" and "Pin" actions."""
    _get_session(store, session_id)
    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise HTTPException(422, "A title needs at least one character.")
        store.set_session_title(session_id, title[:200])
    if body.pinned is not None:
        store.set_session_pinned(session_id, body.pinned)
    return SessionOut(**_get_session(store, session_id))


class ForkIn(BaseModel):
    message_id: int


@router.post("/{session_id}/fork", response_model=SessionOut)
def fork_session(session_id: int, body: ForkIn, store=Depends(get_store)) -> SessionOut:
    """"Fork from here": a new session carrying a copy of the messages up to (and including)
    `message_id`. The original session, its facts and its vector index are untouched."""
    _get_session(store, session_id)
    if not any(m["id"] == body.message_id for m in store.get_messages(session_id)):
        raise HTTPException(404, "No such message in this session.")
    new_id = store.fork_session(session_id, body.message_id)
    return SessionOut(**_get_session(store, new_id))


@router.get("/{session_id}/export")
def export_session(session_id: int, format: str = "markdown", store=Depends(get_store)) -> Response:
    """Export this session's messages as Markdown or JSON (the Sessions screen's "Export" actions)."""
    if format not in ("markdown", "json"):
        raise HTTPException(422, "format must be markdown or json.")
    session = _get_session(store, session_id)
    messages = store.get_messages(session_id)
    title = session["title"] or f"session-{session_id}"
    if format == "json":
        body = json.dumps({"id": session_id, "title": session["title"], "messages": messages}, indent=2)
        media_type, ext = "application/json", "json"
    else:
        lines = [f"# {title}", ""]
        for m in messages:
            if m["role"] not in ("user", "assistant"):
                continue
            lines += [f"## {m['role']}", "", m["content"], ""]
        body, media_type, ext = "\n".join(lines), "text/markdown", "md"
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80].strip() or f"session-{session_id}"
    stamp = datetime.now().strftime("%Y%m%d")
    return Response(content=body, media_type=media_type,
                     headers={"Content-Disposition": f'attachment; filename="{safe_name}-{stamp}.{ext}"'})
