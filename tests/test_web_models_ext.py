"""Models detail and saved prompt chats. Needs the web group."""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import promptlab  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


class FakeClient:
    def installed_sizes(self):
        return {"b:1": 2000, "a:1": 9000}

    def loaded_models(self):
        return ["a:1"]


@pytest.fixture()
def tc(monkeypatch):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["models"]["chat"] = "a:1"
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: FakeClient()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_models_detail(tc):
    rows = tc.get("/api/models/detail").json()
    assert rows == [{"name": "a:1", "size_mb": 9000, "loaded": True, "current": True},
                    {"name": "b:1", "size_mb": 2000, "loaded": False, "current": False}]


def test_models_detail_ollama_down(tc):
    class Down:
        def installed_sizes(self):
            raise RuntimeError("down")

        def loaded_models(self):
            raise RuntimeError("down")

    app.dependency_overrides[deps.get_client] = lambda: Down()
    assert tc.get("/api/models/detail").json() == []


def test_prompt_sessions(tc, monkeypatch):
    monkeypatch.setattr(promptlab, "_load_sessions", lambda: [
        {"id": 1, "title": "old", "mode": "improve", "created": "x", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "A"}]},
        {"id": 2, "title": "new", "mode": "anon", "created": "y", "messages": []}])
    rows = tc.get("/api/prompt/sessions").json()
    assert [r["id"] for r in rows] == [2, 1] and rows[1]["last"] == "A" and rows[0]["last"] == ""
