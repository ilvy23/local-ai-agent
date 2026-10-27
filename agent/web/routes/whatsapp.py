"""Read-only view of the user's own synced WhatsApp messages (tables wa_chats / wa_messages in the local database).

The database is opened read-only (SQLite mode=ro) and every statement is parametrised. There is deliberately no
write or send endpoint: sending is blocked by policy. A missing database or missing tables gives empty lists.
The only writes on this module are contact nicknames (wa_aliases, through the plugin). Sync, jobs, profiles, digest,
persona and auto-reply state live in whatsapp_sync / whatsapp_people / whatsapp_digest / whatsapp_persona.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from agent.web import deps
from agent.web.deps import get_config
from agent.web.routes import whatsapp_core as core
from agent.web.routes.whatsapp_core import ro_rows as _rows

router = APIRouter(prefix="/api/wa", tags=["whatsapp"])


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


@router.get("/chats")
def chats(kind: str = Query("all", pattern="^(all|dm|group)$"), config=Depends(get_config)) -> list[dict[str, Any]]:
    rows = _rows(config, """
        SELECT c.jid AS id, c.name AS name, c.is_group AS is_group, COUNT(m.id) AS message_count,
               MAX(m.timestamp) AS last_message_at,
               (SELECT x.text FROM wa_messages x WHERE x.chat_jid = c.jid ORDER BY x.timestamp DESC LIMIT 1) AS preview,
               (SELECT x.from_me FROM wa_messages x WHERE x.chat_jid = c.jid ORDER BY x.timestamp DESC LIMIT 1) AS last_from_me
        FROM wa_chats c JOIN wa_messages m ON m.chat_jid = c.jid
        WHERE c.jid NOT LIKE '%@broadcast' AND (? = 'all' OR (? = 'group') = (COALESCE(c.is_group, 0) = 1))
        GROUP BY c.jid ORDER BY last_message_at DESC LIMIT 200""", (kind, kind))
    for r in rows:
        r["is_group"] = bool(r["is_group"])
        r["preview"] = (r["preview"] or "").replace("\n", " ")[:120]
        r["last_from_me"] = bool(r["last_from_me"])
    return rows


@router.get("/chats/{chat_id}/messages")
def messages(chat_id: str, limit: int = Query(50, ge=1, le=200), before: float | None = None,
             config=Depends(get_config)) -> list[dict[str, Any]]:
    """The newest `limit` messages (older than `before`, an epoch-seconds timestamp), oldest first."""
    rows = _rows(config, """
        SELECT id, from_me, sender_jid AS sender, timestamp, text, type FROM wa_messages
        WHERE chat_jid = ? AND (? IS NULL OR timestamp < ?) ORDER BY timestamp DESC, rowid DESC LIMIT ?""",
        (chat_id, before, before, limit))
    for r in rows:
        r["from_me"] = bool(r["from_me"])
    return rows[::-1]


@router.get("/search")
def search(q: str = Query(..., min_length=1, max_length=200), contact: str = Query("", max_length=100),
           limit: int = Query(50, ge=1, le=200), since: float | None = None, until: float | None = None,
           who: str = Query("", pattern="^(|me|them)$"), type: str = Query("", max_length=30),
           config=Depends(get_config)) -> list[dict[str, Any]]:
    """Keyword search. Filters: contact (name contains), since/until (epoch seconds), who (me|them), type (text, image, ...)."""
    rows = _rows(config, """
        SELECT m.id AS id, m.chat_jid AS chat_id, COALESCE(c.name, '') AS chat, m.from_me AS from_me,
               m.timestamp AS timestamp, m.text AS text, m.type AS type
        FROM wa_messages m LEFT JOIN wa_chats c ON c.jid = m.chat_jid
        WHERE m.text LIKE ? ESCAPE '\\' AND (? = '' OR COALESCE(c.name, '') LIKE ? ESCAPE '\\')
          AND (? IS NULL OR m.timestamp >= ?) AND (? IS NULL OR m.timestamp < ?)
          AND (? = '' OR (? = 'me') = (m.from_me = 1)) AND (? = '' OR m.type = ?)
        ORDER BY m.timestamp DESC LIMIT ?""", (_like(q), contact, _like(contact), since, since, until, until, who, who, type, type, limit))
    for r in rows:
        r["from_me"] = bool(r["from_me"])
    return rows


@router.get("/search/semantic")
def search_semantic(q: str = Query(..., min_length=1, max_length=200), contact: str = Query("", max_length=100),
                    config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    """The plugin's own search tool (by meaning, falls back to text). It answers with formatted text, not rows."""
    core.require()
    from agent.memory.vectors import VectorIndex

    try:
        with core.open_store(core.wa_config(config)) as store:
            tool = core.mod("tools")._make_search_tool(store, VectorIndex(store), client, config)
            return {"text": str(tool.handler(query=q, contact=contact or None))}
    except core.PluginMissing as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/aliases")
def aliases(config=Depends(get_config)) -> list[dict[str, Any]]:
    return _rows(config, "SELECT alias, jid, created_at FROM wa_aliases ORDER BY alias")


class AliasIn(BaseModel):
    alias: str = Field(min_length=1, max_length=40)
    contact: str = Field(min_length=1, max_length=120)


@router.post("/aliases")
def set_alias(body: AliasIn, config=Depends(get_config)) -> dict[str, Any]:
    """Map a nickname to a contact that has no synced name (`agent wa alias set`)."""
    core.require()
    with core.open_store(core.wa_config(config)) as store:
        res = core.mod("tools").set_alias(store, body.alias.strip(), body.contact.strip())
    if res.ambiguous:
        raise HTTPException(409, "Several contacts match: " + ", ".join((n or j) for j, n in res.matches[:6]) + ". Use the exact id.")
    if res.jid is None:
        raise HTTPException(404, "No contact matches that.")
    return {"alias": body.alias.strip().lower(), "jid": res.jid, "name": res.name}
