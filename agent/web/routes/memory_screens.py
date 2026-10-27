"""Extras for the Memory screen: edit a fact, the per-answer recall trace, the web research cache, re-embed and prune."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.memory import recall
from agent.memory.reembed import reembed_all
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex
from agent.research.memory import _SCHEMA as WEB_SCHEMA, ttl_for
from agent.web.deps import get_client, get_config, get_store
from agent.web.routes.memory import _fact

router = APIRouter(prefix="/api", tags=["memory"])


# ------------------------------------------------------------------------------------------------ edit a fact


class FactPatch(BaseModel):
    content: str
    kind: str | None = None


@router.patch("/memory/facts/{fact_id}")
def edit_fact(fact_id: int, body: FactPatch, store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    old = store.conn.execute("SELECT kind FROM facts WHERE id = ? AND active = 1", (fact_id,)).fetchone()
    if old is None:
        raise HTTPException(404, "No such fact.")
    content, kind = body.content.strip(), body.kind or old["kind"]
    if not content or len(content) > 1500 or kind not in ("personal", "incidental"):
        raise HTTPException(422, "A fact needs 1-1500 characters and a kind of personal or incidental.")
    store.deactivate_fact(fact_id)                                  # also purges its vector entry
    new_id = store.add_fact(content, source_session_id=None, kind=kind)
    vectors = VectorIndex(store)
    if vectors.available:
        try:
            vectors.add("fact", new_id, content, client.embed([content], model=config["models"]["embed"])[0])
        except Exception:  # noqa: BLE001 - an unindexed fact is repaired later
            pass
    row = store.conn.execute("SELECT id, content, kind, confidence, created_at, updated_at FROM facts WHERE id = ?", (new_id,)).fetchone()
    return _fact(row, 24)


# ------------------------------------------------------------------------------------------------ per-answer trace


@router.get("/sessions/{session_id}/messages/{idx}/recall")
def answer_recall(session_id: int, idx: int, store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    """What recall.build_context would have put in front of the model for the user message before this answer."""
    msgs = store.get_messages(session_id)
    if idx == -1:
        idx = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
    if not 0 <= idx < len(msgs) or msgs[idx]["role"] != "assistant":
        raise HTTPException(404, "No such answer in this session.")
    question = next((m["content"] for m in reversed(msgs[:idx]) if m["role"] == "user"), None)
    if question is None:
        raise HTTPException(404, "This answer has no question before it.")

    conn = store.conn
    out: dict[str, Any] = {"facts": [], "messages": [], "web_notes": []}
    sims: dict[str, float] = {}
    vectors = VectorIndex(store)
    min_sim = float(config["memory"].get("recall_min_similarity", 0.0) or 0.0)
    if vectors.available:
        try:
            emb = client.embed([question], model=config["models"]["embed"])[0]
            hits = vectors.search(emb, k=config["memory"]["recall_k"], kinds=["fact", "message", "web"])
        except Exception:  # noqa: BLE001 - same rule as recall: no embedding means no semantic recall
            hits = []
        web = 0
        for text, kind, ref, dist in hits:
            sim = round(1.0 - dist * dist / 2.0, 2)
            if min_sim and sim < min_sim:
                continue
            if kind == "fact":
                sims[text] = sim
                if text not in {f["content"] for f in out["facts"]}:
                    row = conn.execute("SELECT kind, confidence FROM facts WHERE content = ? AND active = 1", (text,)).fetchone()
                    if row:
                        out["facts"].append({"content": text, "kind": row["kind"], "confidence": row["confidence"], "similarity": sim})
            elif kind == "message":
                if len(text.strip()) < recall.MIN_MESSAGE_RECALL_CHARS:
                    continue
                row = conn.execute("SELECT role, created_at FROM messages WHERE id = ?", (ref,)).fetchone()
                out["messages"].append({"role": row["role"] if row else "", "date": str(row["created_at"] or "")[:10] if row else "", "text": text, "similarity": sim})
            elif kind == "web":
                if sim < max(min_sim, recall.WEB_NOTE_MIN_SIMILARITY) or web >= 2 or recall._web_note_too_old(text):
                    continue
                web += 1
                m = re.search(r"Web research on (\d{4}-\d{2}-\d{2})", text)
                out["web_notes"].append({"text": text, "date": m.group(1) if m else ""})
    seen = {f["content"] for f in out["facts"]}                     # personal facts go into every prompt, matched or not
    for f in store.get_active_facts(kind="personal"):
        if f["content"] not in seen:
            out["facts"].append({"content": f["content"], "kind": "personal", "confidence": f["confidence"], "similarity": sims.get(f["content"])})
    out["facts"].sort(key=lambda f: -(f["similarity"] or 0))
    return out


# ------------------------------------------------------------------------------------------------ web cache


def _web_tables(store) -> None:
    store.conn.executescript(WEB_SCHEMA)


@router.get("/memory/web-cache")
def web_cache(store=Depends(get_store)) -> dict[str, Any]:
    _web_tables(store)
    now = time.time()
    pages = [{"url": r["url"], "title": r["title"] or "", "published": r["published"] or "", "fetched_at": r["fetched_at"], "age": max(0, now - (r["fetched_at"] or now))}
             for r in store.conn.execute("SELECT url, title, published, fetched_at FROM web_pages ORDER BY fetched_at DESC LIMIT 200")]
    queries = []
    for r in store.conn.execute("SELECT key, query, hits, searched_at FROM web_queries ORDER BY searched_at DESC LIMIT 200"):
        age = max(0, now - (r["searched_at"] or now))
        freshness = r["key"].split("|", 1)[0] if "|" in r["key"] else "any"
        try:
            n = len(json.loads(r["hits"] or "[]"))
        except ValueError:
            n = 0
        queries.append({"query": r["query"], "searched_at": r["searched_at"], "age": age, "sources": n, "expires_in": ttl_for(r["query"], freshness) - age})
    return {"pages": pages, "queries": queries}


@router.delete("/memory/web-cache")
def clear_web_cache(store=Depends(get_store)) -> dict[str, int]:
    _web_tables(store)
    pages = store.conn.execute("DELETE FROM web_pages").rowcount
    queries = store.conn.execute("DELETE FROM web_queries").rowcount
    store.conn.commit()
    return {"pages": pages, "queries": queries}


# ------------------------------------------------------------------------------------------------ re-embed


_job: dict[str, Any] = {"state": "idle", "done": 0, "total": 0, "started": 0.0, "error": ""}
_job_lock = threading.Lock()
_PROGRESS = re.compile(r"re-embedded (\d+)/(\d+)")


def _run_reembed(db_path: str, client: Any, config: dict[str, Any]) -> None:
    def log(msg: str) -> None:
        m = _PROGRESS.search(msg)
        if m:
            _job["done"], _job["total"] = int(m[1]), int(m[2])

    try:
        store = Store(db_path)                                       # sqlite connections are per thread
        try:
            result = reembed_all(store, client, config, config["models"]["embed"], log=log)
        finally:
            store.conn.close()
        _job.update(state="done", done=result["reembedded"], total=result["total"])
    except Exception as exc:  # noqa: BLE001 - reported through the status endpoint
        _job.update(state="error", error=str(exc)[:300])


@router.post("/memory/reembed")
def start_reembed(store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    with _job_lock:
        if _job["state"] == "running":
            raise HTTPException(409, "A re-embed is already running.")
        total = store.conn.execute("SELECT COUNT(*) FROM memory_items WHERE text IS NOT NULL AND text != ''").fetchone()[0]
        _job.update(state="running", done=0, total=total, started=time.time(), error="")
    threading.Thread(target=_run_reembed, args=(config["data"]["db_path"], client, config), daemon=True).start()
    return {"job": "reembed", "state": "running", "total": total}


@router.get("/memory/reembed/status")
def reembed_status() -> dict[str, Any]:
    done, total = _job["done"], _job["total"]
    eta = None
    if _job["state"] == "running" and done:
        eta = round((time.time() - _job["started"]) / done * (total - done))
    return {"state": _job["state"], "done": done, "total": total, "eta_seconds": eta, "error": _job["error"]}


# ------------------------------------------------------------------------------------------------ prune


class PruneIn(BaseModel):
    dry_run: bool = False


@router.post("/memory/prune")
def prune(body: PruneIn | None = None, store=Depends(get_store)) -> dict[str, Any]:
    """Forget junk facts (paths, tool output, timestamps) the distiller captured; the `memory prune` CLI logic."""
    from agent.memory.distill import JUNK_FACT_RE

    junk = [f for f in store.get_active_facts() if JUNK_FACT_RE.search(f["content"] or "")]
    dry = bool(body and body.dry_run)
    if not dry:
        for f in junk:
            store.deactivate_fact(f["id"])
    return {"count": len(junk), "dry_run": dry, "facts": [f["content"][:120] for f in junk[:20]]}


# ------------------------------------------------------------------------------------------------ knowledge screen


@router.get("/memory/search")
def search_memory(q: str, store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    """Semantic fact search with similarity scores (like `memory search`); falls back to substring match."""
    q = q.strip()
    if not q:
        raise HTTPException(422, "Give something to search for.")
    vectors = VectorIndex(store)
    hits: list[dict[str, Any]] = []
    mode = "semantic"
    if vectors.available:
        try:
            emb = client.embed([q], model=config["models"]["embed"])[0]
            for text, _kind, ref, dist in vectors.search(emb, k=config["memory"]["recall_k"] * 2, kinds=["fact"]):
                row = store.conn.execute("SELECT kind FROM facts WHERE id = ? AND active = 1", (ref,)).fetchone()
                if row:
                    hits.append({"id": ref, "content": text, "kind": row["kind"], "similarity": round(1.0 - dist * dist / 2.0, 2)})
        except Exception:  # noqa: BLE001 - no embedding means the substring fallback
            hits = []
    if not hits:
        mode = "substring"
        hits = [{"id": f["id"], "content": f["content"], "kind": f.get("kind", "personal"), "similarity": None} for f in store.search_facts_like(q)]
    return {"mode": mode, "results": hits}


@router.get("/memory/status")
def memory_status(store=Depends(get_store), config=Depends(get_config)) -> dict[str, Any]:
    """Embedding model, index health and counts for the Maintenance tab."""
    from agent.memory.distill import _UNEMBEDDED

    vectors = VectorIndex(store)
    missing = {k: len(store.conn.execute(sql).fetchall()) for k, sql in _UNEMBEDDED.items()} if vectors.available else {}
    vec_count = 0
    if vectors.available:
        try:
            vec_count = store.conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
        except Exception:  # noqa: BLE001 - no table yet
            pass
    return {"embed_model": config["models"]["embed"], "available": vectors.available, "dim": vectors.DIM if vectors.available else None,
            "vectors": vec_count, "unindexed": missing, "healthy": vectors.available and not any(missing.values())}


@router.post("/memory/repair")
def repair_index(store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    """Embed facts/messages that never got a vector (the `memory repair` CLI logic)."""
    from agent.memory.distill import repair_unembedded

    vectors = VectorIndex(store)
    if not vectors.available:
        raise HTTPException(409, "Vector index unavailable - nothing to repair.")
    return {"repaired": repair_unembedded(store, vectors, client, config)}


@router.get("/memory/export")
def export_memory(store=Depends(get_store)) -> dict[str, Any]:
    return {"facts": [{"content": f["content"], "kind": f.get("kind", "personal")} for f in store.get_active_facts()]}


def _skill_rows(store) -> list[dict[str, Any]]:
    return store.list_skills()


@router.get("/skills-curator/report")
def curator_report(store=Depends(get_store), config=Depends(get_config)) -> dict[str, Any]:
    """What the curator would do now (dry run) plus the skills it already archived and why."""
    from agent.memory import curator

    cfg = config.get("curator") or {}
    preview = curator.curate_skills(store, max_failure_ratio=float(cfg.get("max_failure_ratio", 0.7)),
                                    max_age_days=int(cfg.get("max_age_days", 90)), dry_run=True)
    archived = []
    for s in _skill_rows(store):
        if "archived" in s["tags"]:
            reason = next((t.split(":", 1)[1] for t in s["tags"] if t.startswith("reason:")), "archived")
            archived.append({"id": s["id"], "title": s["title"], "reason": reason})
    row = store.conn.execute("SELECT v FROM app_state WHERE k = 'curator_last_ran'").fetchone()
    return {"would": preview, "archived": archived, "last_ran": row["v"] if row else None,
            "settings": {"max_failure_ratio": cfg.get("max_failure_ratio", 0.7), "max_age_days": cfg.get("max_age_days", 90)}}


@router.post("/skills-curator/run")
def curator_run(store=Depends(get_store), config=Depends(get_config)) -> dict[str, Any]:
    from agent.memory import curator

    cfg = config.get("curator") or {}
    result = curator.curate_skills(store, max_failure_ratio=float(cfg.get("max_failure_ratio", 0.7)),
                                   max_age_days=int(cfg.get("max_age_days", 90)))
    curator.mark_curator_ran(store)
    return result
