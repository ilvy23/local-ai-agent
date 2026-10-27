"""Knowledge screen API: semantic search, index status, repair, export, curator."""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


class FakeClient:
    def embed(self, texts, model, **kw):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "k.db")
    config["memory"]["recall_min_similarity"] = 0.0
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: FakeClient()
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app), Store(config["data"]["db_path"])
    app.dependency_overrides.clear()


def test_search_scores_and_empty_query(env):
    tc, _ = env
    tc.post("/api/memory/facts", json={"content": "Runs Debian on a desktop PC", "kind": "personal"})
    r = tc.get("/api/memory/search?q=gpus").json()
    assert r["results"] and r["results"][0]["content"].startswith("Runs Debian")
    assert tc.get("/api/memory/search?q=%20").status_code == 422


def test_status_repair_export(env):
    tc, store = env
    tc.post("/api/memory/facts", json={"content": "likes tea", "kind": "personal"})
    store.add_fact("never embedded", None)                    # stored without a vector
    s = tc.get("/api/memory/status").json()
    if not s["available"]:
        pytest.skip("sqlite-vec not available")
    assert s["unindexed"]["fact"] == 1 and not s["healthy"]
    assert tc.post("/api/memory/repair").json()["repaired"]["fact"] == 1
    assert tc.get("/api/memory/status").json()["healthy"]
    assert {f["content"] for f in tc.get("/api/memory/export").json()["facts"]} == {"likes tea", "never embedded"}


def test_curator_report_and_run(env):
    tc, store = env
    sid = store.add_skill("t", "problem one", ["step"], [], ["x"])
    for _ in range(4):
        store.update_skill_failure(sid)
    rep = tc.get("/api/skills-curator/report").json()
    assert rep["would"]["archived"] == 1 and rep["archived"] == [] and rep["last_ran"] is None
    assert tc.post("/api/skills-curator/run").json()["archived"] == 1
    rep = tc.get("/api/skills-curator/report").json()
    assert rep["archived"][0]["reason"] == "high_failure_ratio" and rep["last_ran"]
