"""Research screen API: rule explain/import, feedback history, engines, cache, verify history. No network."""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "r.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app), Store(config["data"]["db_path"])
    app.dependency_overrides.clear()


def test_explain_and_rules_roundtrip(env):
    tc, _ = env
    assert tc.get("/api/research/explain?domain=blog.rust-lang.org").json()["tier"] == "T1"
    assert tc.get("/api/research/explain?domain=nodot").status_code == 422
    r = tc.post("/api/research/rules/import", json={"allow": ["a.example"], "deny": ["spam.example"], "tiers": {"lwn.net": "t2"}})
    assert r.status_code == 200 and r.json()["tiers"] == {"lwn.net": "T2"}
    assert tc.post("/api/research/rules/import", json={"tiers": {"x.org": "T9"}}).status_code == 422
    fb = tc.get("/api/research/feedback").json()
    assert {(x["domain"], x["said"]) for x in fb["rows"]} == {("a.example", "good"), ("spam.example", "block"), ("lwn.net", "tier T2")}
    assert tc.get("/api/research/explain?domain=spam.example").json()["level"] == "avoid"


def test_engines_cache_and_guard(env, monkeypatch):
    tc, _ = env
    assert [(e["name"], e["active"]) for e in tc.get("/api/research/engines").json()] == [("searxng", False), ("ddg", True), ("brave", False)]
    assert tc.post("/api/research/engines/brave/test").status_code == 404
    from agent.tools import web
    monkeypatch.setattr(web, "_search_ddg", lambda q, k, df=None: [{"url": "https://x.org", "title": "t", "snippet": "s"}])
    assert tc.post("/api/research/engines/ddg/test").json()["status"] == "reachable"
    assert tc.get("/api/research/cache").json()["pages"] == 0
    guard = tc.get("/api/research/ssrf").json()
    assert guard["on"] and [bool(s["blocked"]) for s in guard["samples"]] == [True, True, True, False]


def test_verify_history(env):
    tc, store = env
    sid = store.create_session("Rust")
    store.add_message(sid, "assistant", "Rust 1.90 came out in 2026.")
    store.add_message(sid, "evidence", json.dumps({"type": "verdict", "label": "single source"}))
    store.add_message(sid, "evidence", json.dumps({"type": "verify_result", "overall": "supported", "claims": [{}], "searches": 2}))
    h = tc.get("/api/research/verify-history").json()
    assert len(h) == 1 and h[0]["before"] == "single source" and h[0]["after"] == "supported" and h[0]["answer"].startswith("Rust 1.90")
