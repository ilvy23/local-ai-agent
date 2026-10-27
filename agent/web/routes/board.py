"""The Board: blocks the user pinned from answers, kept with a grid position. Table is created on first use."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.web.deps import get_client, get_config, get_store

router = APIRouter(prefix="/api/board", tags=["board"])

_SCHEMA = """
CREATE TABLE IF NOT EXISTS board_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, block_id TEXT, block TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
    x REAL NOT NULL DEFAULT 0, y REAL NOT NULL DEFAULT 0, w REAL NOT NULL DEFAULT 1, h REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
"""
MAX_BLOCK_CHARS = 512_000
# A block is refreshable when its spec carries {"refresh": "status"}: the tiles are rebuilt from the live server status.
REFRESH_SOURCES = ("status",)


class BlockIn(BaseModel):
    block: dict[str, Any]


class ItemPatch(BaseModel):
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    title: str | None = None


def _conn(store):
    store.conn.executescript(_SCHEMA)
    return store.conn


def _refresh_hint(block: dict[str, Any]) -> str | None:
    spec = block.get("spec")
    hint = spec.get("refresh") if isinstance(spec, dict) else None
    return hint if hint in REFRESH_SOURCES else None


def _item(row: Any) -> dict[str, Any]:
    block = json.loads(row["block"])
    return {"id": row["id"], "block": block, "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"], "title": row["title"],
            "refreshable": _refresh_hint(block) is not None, "created_at": row["created_at"], "updated_at": row["updated_at"]}


def _get(conn, item_id: int):
    row = conn.execute("SELECT * FROM board_items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "No such Board item.")
    return row


@router.get("/items")
def list_items(store=Depends(get_store)) -> list[dict[str, Any]]:
    return [_item(r) for r in _conn(store).execute("SELECT * FROM board_items ORDER BY y, id").fetchall()]


@router.post("/items")
def pin(body: BlockIn, store=Depends(get_store)) -> dict[str, Any]:
    block = body.block
    if not isinstance(block.get("kind"), str) or not isinstance(block.get("spec"), dict):
        raise HTTPException(422, "A block needs a kind and a spec.")
    raw = json.dumps(block)
    if len(raw) > MAX_BLOCK_CHARS:
        raise HTTPException(422, "This block is too large to pin.")
    conn = _conn(store)
    block_id = str(block.get("id") or "")
    if block_id:                                                   # pinning the same block twice keeps one copy
        row = conn.execute("SELECT * FROM board_items WHERE block_id = ?", (block_id,)).fetchone()
        if row is not None:
            return _item(row)
    now = time.time()
    last = conn.execute("SELECT COALESCE(MAX(y), -1) FROM board_items").fetchone()[0]
    cur = conn.execute("INSERT INTO board_items (block_id, block, title, y, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                       (block_id, raw, str(block.get("title") or "")[:120], last + 1, now, now))
    conn.commit()
    return _item(_get(conn, cur.lastrowid))


@router.patch("/items/{item_id}")
def move(item_id: int, body: ItemPatch, store=Depends(get_store)) -> dict[str, Any]:
    conn = _conn(store)
    _get(conn, item_id)
    fields = body.model_dump(exclude_none=True)
    if "w" in fields and not 1 <= fields["w"] <= 4:
        raise HTTPException(422, "w must be between 1 and 4 columns.")
    if "h" in fields and not 0 <= fields["h"] <= 2000:
        raise HTTPException(422, "h must be between 0 (auto) and 2000.")
    if "title" in fields:
        fields["title"] = fields["title"].strip()[:120]
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)                # keys come from the model, never from the request
        conn.execute(f"UPDATE board_items SET {sets}, updated_at = ? WHERE id = ?", (*fields.values(), time.time(), item_id))
        conn.commit()
    return _item(_get(conn, item_id))


@router.delete("/items/{item_id}")
def unpin(item_id: int, store=Depends(get_store)) -> dict[str, bool]:
    conn = _conn(store)
    _get(conn, item_id)
    conn.execute("DELETE FROM board_items WHERE id = ?", (item_id,))
    conn.commit()
    return {"ok": True}


@router.post("/items/{item_id}/refresh")
def refresh(item_id: int, store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    from agent.web.routes.system import status

    conn = _conn(store)
    block = json.loads(_get(conn, item_id)["block"])
    if _refresh_hint(block) is None:
        raise HTTPException(400, "This block has no live data to refresh.")
    s = status(client, config)
    block["spec"]["tiles"] = [
        {"label": "Chat model", "value": s["chat_model"]}, {"label": "GPU mode", "value": s["gpu_mode"]},
        {"label": "Models loaded", "value": len(s["loaded_models"])}, {"label": "Game running", "value": "yes" if s["game_detected"] else "no"}]
    conn.execute("UPDATE board_items SET block = ?, updated_at = ? WHERE id = ?", (json.dumps(block), time.time(), item_id))
    conn.commit()
    return _item(_get(conn, item_id))
