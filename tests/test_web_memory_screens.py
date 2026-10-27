"""Memory screen API: edit fact, per-answer trace, web cache, re-embed job, prune."""

from __future__ import annotations

import copy
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.memory.vectors import VectorIndex  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402
from agent.web.routes import memory_screens as ms  # noqa: E402


class FakeClient:
    def embed(self, texts, model, **kw):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "web.db")
    config["memory"]["recall_min_similarity"] = 0.0
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: FakeClient()
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app), config
    app.dependency_overrides.clear()


def test_edit_fact_replaces_it(env):
    tc, config = env
    old = tc.post("/api/memory/facts", json={"content": "likes tea", "kind": "personal"}).json()
    new = tc.patch(f"/api/memory/facts/{old['id']}", json={"content": "likes coffee"}).json()
    assert new["content"] == "likes coffee" and new["kind"] == "personal" and new["id"] != old["id"]
    assert [f["content"] for f in tc.get("/api/memory/facts").json()] == ["likes coffee"]
    assert tc.patch(f"/api/memory/facts/{old['id']}", json={"content": "x"}).status_code == 404
    assert tc.patch(f"/api/memory/facts/{new['id']}", json={"content": " "}).status_code == 422
    assert tc.patch(f"/api/memory/facts/{new['id']}", json={"content": "x", "kind": "odd"}).status_code == 422


def test_recall_trace_for_an_answer(env):
    tc, config = env
    store = Store(config["data"]["db_path"])
    sid = store.create_session()
    store.add_message(sid, "user", "which gpu do I have?")
    store.add_message(sid, "assistant", "Two.")
    store.add_message(sid, "user", "thanks")
    store.add_message(sid, "assistant", "np")
    tc.post("/api/memory/facts", json={"content": "Runs Debian on a desktop PC.", "kind": "personal"})
    tc.post("/api/memory/facts", json={"content": "VPN name is desktop.", "kind": "incidental"})
    r = tc.get(f"/api/sessions/{sid}/messages/1/recall")
    assert r.status_code == 200
    body = r.json()
    assert {"facts", "messages", "web_notes"} <= set(body)
    assert any(f["content"] == "Runs Debian on a desktop PC." and f["kind"] == "personal" for f in body["facts"])
    assert tc.get(f"/api/sessions/{sid}/messages/-1/recall").status_code == 200
    assert tc.get(f"/api/sessions/{sid}/messages/0/recall").status_code == 404     # a user message, not an answer
    assert tc.get(f"/api/sessions/{sid}/messages/99/recall").status_code == 404
    assert tc.get("/api/sessions/4242/messages/-1/recall").status_code == 404


def test_recall_trace_includes_semantic_hits_when_the_index_works(env):
    tc, config = env
    store = Store(config["data"]["db_path"])
    if not VectorIndex(store).available:
        pytest.skip("sqlite-vec not installed")
    sid = store.create_session()
    store.add_message(sid, "user", "what is the release plan?")
    store.add_message(sid, "assistant", "Friday.")
    tc.post("/api/memory/facts", json={"content": "Rust 1.90 shipped in September.", "kind": "incidental"})
    facts = tc.get(f"/api/sessions/{sid}/messages/-1/recall").json()["facts"]
    assert facts and facts[0]["content"] == "Rust 1.90 shipped in September." and facts[0]["similarity"] == 1.0


def test_web_cache_list_and_clear(env):
    tc, config = env
    store = Store(config["data"]["db_path"])
    ms._web_tables(store)
    now = time.time()
    store.conn.execute("INSERT INTO web_pages (url, title, published, text, fetched_at) VALUES ('https://a.org/x', 'A', '2026-10-01', 't', ?)", (now - 60,))
    store.conn.execute("INSERT INTO web_queries (key, query, hits, searched_at) VALUES ('any|rust release', 'rust release', '[{}, {}]', ?)", (now - 100,))
    store.conn.execute("INSERT INTO web_queries (key, query, hits, searched_at) VALUES ('any|old', 'old', '[]', ?)", (now - 10 * 86400,))
    store.conn.commit()
    body = tc.get("/api/memory/web-cache").json()
    assert body["pages"][0]["url"] == "https://a.org/x" and 50 < body["pages"][0]["age"] < 90
    fresh, old = body["queries"]
    assert fresh["sources"] == 2 and fresh["expires_in"] > 0 and old["expires_in"] < 0
    assert tc.delete("/api/memory/web-cache").json() == {"pages": 1, "queries": 2}
    assert tc.get("/api/memory/web-cache").json() == {"pages": [], "queries": []}


def test_reembed_job(env, monkeypatch):
    tc, config = env
    seen = {}

    def fake(store, llm, cfg, model, log=None):
        seen["model"] = model
        log("re-embedded 5/5")
        return {"reembedded": 5, "failed": 0, "dim": 4, "total": 5}

    monkeypatch.setattr(ms, "reembed_all", fake)
    monkeypatch.setitem(ms._job, "state", "idle")
    assert tc.get("/api/memory/reembed/status").json()["state"] == "idle"
    assert tc.post("/api/memory/reembed").json()["state"] == "running"
    for _ in range(100):
        s = tc.get("/api/memory/reembed/status").json()
        if s["state"] != "running":
            break
        time.sleep(0.02)
    assert s["state"] == "done" and (s["done"], s["total"]) == (5, 5) and seen["model"] == config["models"]["embed"]
    monkeypatch.setitem(ms._job, "state", "running")
    assert tc.post("/api/memory/reembed").status_code == 409


def test_prune_dry_run_then_real(env):
    tc, config = env
    store = Store(config["data"]["db_path"])
    store.add_fact("/home/user/code/main.py", source_session_id=None, kind="incidental")
    store.add_fact("Likes black metal", source_session_id=None, kind="personal")
    dry = tc.post("/api/memory/prune", json={"dry_run": True}).json()
    assert dry["dry_run"] and dry["count"] == 1 and len(tc.get("/api/memory/facts").json()) == 2
    real = tc.post("/api/memory/prune").json()
    assert real["count"] == 1 and [f["content"] for f in tc.get("/api/memory/facts").json()] == ["Likes black metal"]
