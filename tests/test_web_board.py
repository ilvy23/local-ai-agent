"""Board API: pin, list, move/resize, refresh, unpin."""

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
    def loaded_models(self):
        return ["m1", "m2"]


@pytest.fixture()
def tc(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "web.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: FakeClient()
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app)
    app.dependency_overrides.clear()


def block(bid="b_1", **spec):
    return {"id": bid, "kind": "stats", "title": "Host", "spec": {"tiles": [{"label": "a", "value": 1}], **spec}, "source_ids": [], "provenance": "computed"}


def test_pin_list_move_unpin(tc):
    assert tc.get("/api/board/items").json() == []
    a = tc.post("/api/board/items", json={"block": block("b_1")}).json()
    b = tc.post("/api/board/items", json={"block": block("b_2")}).json()
    assert a["block"]["id"] == "b_1" and a["title"] == "Host" and a["refreshable"] is False and b["y"] > a["y"]
    assert tc.post("/api/board/items", json={"block": block("b_1")}).json()["id"] == a["id"]        # no duplicate pin
    assert [i["id"] for i in tc.get("/api/board/items").json()] == [a["id"], b["id"]]
    moved = tc.patch(f"/api/board/items/{a['id']}", json={"y": 9, "w": 2, "h": 300, "title": " New "}).json()
    assert (moved["y"], moved["w"], moved["h"], moved["title"]) == (9, 2, 300, "New")
    assert [i["id"] for i in tc.get("/api/board/items").json()] == [b["id"], a["id"]]
    assert tc.patch(f"/api/board/items/{a['id']}", json={"w": 9}).status_code == 422
    assert tc.patch("/api/board/items/999", json={"w": 1}).status_code == 404
    assert tc.delete(f"/api/board/items/{a['id']}").json() == {"ok": True}
    assert tc.delete(f"/api/board/items/{a['id']}").status_code == 404


def test_pin_rejects_bad_blocks(tc):
    assert tc.post("/api/board/items", json={"block": {"kind": "stats"}}).status_code == 422
    assert tc.post("/api/board/items", json={"block": {"kind": "html", "spec": {"html": "x" * 600_000}}}).status_code == 422


def test_refresh_only_for_blocks_with_a_hint(tc):
    plain = tc.post("/api/board/items", json={"block": block("b_1")}).json()
    r = tc.post(f"/api/board/items/{plain['id']}/refresh")
    assert r.status_code == 400 and "no live data" in r.json()["detail"]
    live = tc.post("/api/board/items", json={"block": block("b_2", refresh="status")}).json()
    assert live["refreshable"] is True
    done = tc.post(f"/api/board/items/{live['id']}/refresh").json()
    tiles = {t["label"]: t["value"] for t in done["block"]["spec"]["tiles"]}
    assert tiles["Models loaded"] == 2 and done["block"]["spec"]["refresh"] == "status"
