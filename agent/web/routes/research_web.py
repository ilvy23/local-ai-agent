"""Research, trust & web screen: why a source is rated as it is, rule import, feedback history, search engine chain,
connectors, cache size and the history of "Verify this answer" runs. Thin wrappers over agent.research / agent.tools.web."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.research import connectors, trust
from agent.web.deps import get_store
from agent.web.routes.system import _rules_doc

router = APIRouter(prefix="/api/research", tags=["research"])


# ------------------------------------------------------------------------------------------------ trust rules


@router.get("/explain")
def explain(domain: str) -> dict[str, Any]:
    """The rating a bare domain gets, with the reasons (rules only, no network)."""
    host = domain.strip().lower().removeprefix("https://").removeprefix("http://").split("/")[0]
    if "." not in host:
        raise HTTPException(422, "Give a domain such as example.org.")
    result = trust.assess(f"https://{host}/").as_dict()
    result["domain"] = trust.registrable(host)
    return result


class RulesImport(BaseModel):
    allow: list[str] = []
    deny: list[str] = []
    tiers: dict[str, str] = {}


@router.post("/rules/import")
def import_rules(body: RulesImport) -> dict[str, Any]:
    """Merge a rules document (as exported from the rules table) into the current one. Invalid entries are rejected as a whole."""
    bad = [t for t in body.tiers.values() if str(t).upper() not in trust.TIER_LEVEL]
    if bad:
        raise HTTPException(422, "tiers must be T1, T2, T3 or T4.")
    doc = _rules_doc(trust.RULES_PATH)
    for name in body.allow + body.deny + list(body.tiers):
        if "." not in trust.registrable(str(name)):
            raise HTTPException(422, f"{name!r} is not a domain.")
    keys = {trust.registrable(str(d)) for d in body.allow + body.deny + list(body.tiers)}
    doc["allow"] = [d for d in doc["allow"] if trust.registrable(str(d)) not in keys]
    doc["deny"] = [d for d in doc["deny"] if trust.registrable(str(d)) not in keys]
    doc["tiers"] = {k: v for k, v in doc["tiers"].items() if trust.registrable(str(k)) not in keys}
    doc["allow"] += [trust.registrable(str(d)) for d in body.allow]
    doc["deny"] += [trust.registrable(str(d)) for d in body.deny]
    doc["tiers"].update({trust.registrable(str(k)): str(v).upper() for k, v in body.tiers.items()})
    trust.RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    trust.RULES_PATH.write_text(yaml.safe_dump(doc, sort_keys=True))
    return doc


@router.get("/feedback")
def feedback_history() -> dict[str, Any]:
    """Source feedback from the chat ("Good source" = allow, "Poor" = tier T3, "Block domain" = deny) is stored as rules, so the
    history is the rules themselves. They carry no timestamps; only the file's last change is known."""
    doc = _rules_doc(trust.RULES_PATH)
    try:
        changed = trust.RULES_PATH.stat().st_mtime
    except OSError:
        changed = None
    rows = [{"domain": d, "said": "good"} for d in doc["allow"]]
    rows += [{"domain": d, "said": "block"} for d in doc["deny"]]
    rows += [{"domain": d, "said": "poor" if t in ("T3", "T4") else f"tier {t}"} for d, t in doc["tiers"].items()]
    return {"rows": rows, "rules_changed_at": changed}


# ------------------------------------------------------------------------------------------------ engines, connectors, cache


def _engine_table() -> list[dict[str, Any]]:
    searx = os.environ.get("AGENT_SEARXNG_URL", "").strip()
    brave = bool(os.environ.get("BRAVE_API_KEY", "").strip())
    return [
        {"name": "searxng", "label": f"SearXNG ({searx})" if searx else "SearXNG", "configured": bool(searx), "how": "set AGENT_SEARXNG_URL"},
        {"name": "ddg", "label": "DuckDuckGo", "configured": True, "how": "built in"},
        {"name": "brave", "label": "Brave" + (" (key set)" if brave else ""), "configured": brave, "how": "set BRAVE_API_KEY"},
    ]


@router.get("/engines")
def engines() -> list[dict[str, Any]]:
    from agent.tools import web

    active = {n for n, _f in web._engines()}
    return [{**e, "active": e["name"] in active} for e in _engine_table()]


@router.post("/engines/{name}/test")
def test_engine(name: str) -> dict[str, Any]:
    """One tiny search through one engine. This is a real network call, made only when the person presses Test."""
    from agent.tools import web

    fn = dict(web._engines()).get(name)
    if fn is None:
        raise HTTPException(404, "That engine is not configured.")
    started = time.time()
    try:
        hits = fn("test", 1)
    except web.SearchBlocked as exc:
        return {"ok": False, "status": "blocked", "detail": str(exc)[:200]}
    except Exception as exc:  # noqa: BLE001 - an unreachable engine is a result
        return {"ok": False, "status": "unreachable", "detail": type(exc).__name__}
    return {"ok": bool(hits), "status": "reachable" if hits else "no results", "detail": f"{len(hits)} hit(s) in {time.time() - started:.1f} s"}


_CONNECTORS = [("Wikipedia", "definitions ('what is ...')"), ("PyPI", "Python package versions"), ("npm", "npm package versions"),
               ("arXiv", "papers (when the question says arXiv/paper)"), ("GitHub", "repositories (when it says GitHub)"),
               ("Hacker News", "discussions (when it says Hacker News)"), ("Weather", "weather in a city"), ("Exchange rates", "currency conversion")]


@router.get("/connectors")
def list_connectors() -> list[dict[str, str]]:
    return [{"name": n, "when": w} for n, w in _CONNECTORS]


@router.get("/cache")
def cache_info(store=Depends(get_store)) -> dict[str, Any]:
    from agent.research.memory import _SCHEMA, TTL_DEFAULT, TTL_EVERGREEN, TTL_NEWS

    store.conn.executescript(_SCHEMA)
    pages, size = store.conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0) FROM web_pages").fetchone()
    queries = store.conn.execute("SELECT COUNT(*) FROM web_queries").fetchone()[0]
    return {"pages": pages, "queries": queries, "bytes": size, "ttl_seconds": {"news": TTL_NEWS, "default": TTL_DEFAULT, "docs": TTL_EVERGREEN}}


@router.get("/ssrf")
def ssrf_guard() -> dict[str, Any]:
    """The address guard is code, always on; this proves it with a few sample addresses (no network)."""
    from agent.research import netguard

    samples = ["http://127.0.0.1:11434/", "http://192.168.1.1/", "http://100.64.0.5/", "https://example.org/"]
    return {"on": True, "samples": [{"url": u, "blocked": netguard.blocked_reason(u, resolve=False)} for u in samples]}


# ------------------------------------------------------------------------------------------------ verify history


@router.get("/verify-history")
def verify_history(limit: int = 50, store=Depends(get_store)) -> list[dict[str, Any]]:
    """Every "Verify this answer" run stored with its chat, with the verdict the answer had before and the result after."""
    rows = store.conn.execute(
        "SELECT m.id, m.session_id, m.content, m.created_at, s.title FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE m.role = 'evidence' AND m.content LIKE '%verify_result%' ORDER BY m.id DESC LIMIT ?", (max(1, min(limit, 200)),)).fetchall()
    out = []
    for r in rows:
        try:
            ev = json.loads(r["content"])
        except ValueError:
            continue
        if ev.get("type") != "verify_result":
            continue
        before = ""
        for v in store.conn.execute("SELECT content FROM messages WHERE session_id = ? AND id < ? AND role = 'evidence' ORDER BY id DESC LIMIT 20",
                                     (r["session_id"], r["id"])):
            try:
                e = json.loads(v["content"])
            except ValueError:
                continue
            if e.get("type") == "verdict":
                before = e.get("label", "")
                break
        answer = next((m["content"] for m in store.conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND id < ? AND role = 'assistant' ORDER BY id DESC LIMIT 1", (r["session_id"], r["id"]))), "")
        out.append({"id": r["id"], "session_id": r["session_id"], "chat": r["title"] or "New chat", "when": r["created_at"],
                    "answer": " ".join(answer.split())[:100], "before": before, "after": ev.get("overall", ""),
                    "claims": len(ev.get("claims") or []), "searches": ev.get("searches", 0)})
    return out
