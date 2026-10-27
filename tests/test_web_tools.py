"""Tools & audit screen API (read-only). No tool is ever run."""

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


class FakeClient:
    def embed(self, texts, model, **kw):
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "t.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: FakeClient()
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app), Store(config["data"]["db_path"])
    app.dependency_overrides.clear()


def test_tool_list_has_risk_and_last_use(env):
    tc, store = env
    store.add_audit_log("tool", json.dumps({"name": "current_time", "arguments": {}}), 1, "now")
    tools = {t["name"]: t for t in tc.get("/api/tools").json()}
    assert tools["current_time"]["risk"] == "SAFE" and tools["current_time"]["last_use"]
    assert tools["write_file"]["risk"] in ("CAUTION", "DANGEROUS") and tools["run_command"]["risk"] == "DYNAMIC"
    assert tools["write_file"]["last_use"] is None


def test_audit_filters(env):
    tc, store = env
    store.add_audit_log("shell", "pytest -q", 1, "56 passed")
    store.add_audit_log("shell", "rm -r ~/old", 0, "denied")
    store.add_audit_log("tool", json.dumps({"name": "write_file", "arguments": {"path": "~/.ssh/config"}}), 0, "BLOCKED by safety policy: x")
    rows = tc.get("/api/tools/audit").json()
    assert {r["decision"] for r in rows} == {"approved", "denied", "refused"}
    assert [r["tool"] for r in tc.get("/api/tools/audit?decision=refused").json()] == ["write_file"]
    assert len(tc.get("/api/tools/audit?q=pytest").json()) == 1


def test_packs_are_not_startable_from_the_web(env):
    tc, _ = env
    p = tc.get("/api/tools/packs").json()
    assert p["available"] is False and "terminal" in p["reason"]
    assert not [r for r in app.routes if getattr(r, "path", "").startswith("/api/tools/packs") and "POST" in getattr(r, "methods", ())]
