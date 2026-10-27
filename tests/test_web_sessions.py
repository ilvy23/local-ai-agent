"""Sessions screen API: rename, pin, export, fork, notes editor (and the v14 migration keeping old rows)."""

from __future__ import annotations

import copy
import sqlite3

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
    config["data"]["db_path"] = str(tmp_path / "s.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app), Store(config["data"]["db_path"])
    app.dependency_overrides.clear()


def test_rename_and_pin(env):
    tc, store = env
    a, b = store.create_session("old"), store.create_session("new")
    r = tc.patch(f"/api/sessions/{a}", json={"title": "Renamed", "pinned": True}).json()
    assert r["title"] == "Renamed" and r["pinned"] is True
    assert [s["id"] for s in tc.get("/api/sessions").json()] == [a, b]            # pinned first
    assert tc.patch(f"/api/sessions/{a}", json={"title": "  "}).status_code == 422
    assert tc.patch("/api/sessions/999", json={"pinned": True}).status_code == 404


def test_export_markdown_and_json(env):
    tc, store = env
    sid = store.create_session("Chat A")
    store.add_message(sid, "user", "hi")
    store.add_message(sid, "assistant", "hello")
    md = tc.get(f"/api/sessions/{sid}/export?format=markdown")
    assert "## user" in md.text and "hello" in md.text and ".md" in md.headers["content-disposition"]
    assert tc.get(f"/api/sessions/{sid}/export?format=json").json()["messages"][1]["content"] == "hello"
    assert tc.get(f"/api/sessions/{sid}/export?format=pdf").status_code == 422


def test_fork_copies_up_to_the_message(env):
    tc, store = env
    sid = store.create_session("T")
    m1 = store.add_message(sid, "user", "one")
    store.add_message(sid, "assistant", "two")
    new = tc.post(f"/api/sessions/{sid}/fork", json={"message_id": m1}).json()
    assert [m["content"] for m in store.get_messages(new["id"])] == ["one"] and new["title"] == "T (fork)"
    assert len(store.get_messages(sid)) == 2
    assert tc.post(f"/api/sessions/{sid}/fork", json={"message_id": 12345}).status_code == 404


def test_notes_editor_replaces(env):
    tc, store = env
    sid = store.create_session()
    assert tc.put(f"/api/sessions/{sid}/notes", json={"notes": "pin nixpkgs"}).json()["notes"] == "pin nixpkgs"
    assert tc.get(f"/api/sessions/{sid}/notes").json()["notes"] == "pin nixpkgs"
    tc.put(f"/api/sessions/{sid}/notes", json={"notes": ""})
    assert tc.get(f"/api/sessions/{sid}/notes").json()["notes"] is None


def test_v14_migration_keeps_existing_rows(tmp_path):
    path = tmp_path / "old.db"
    s = Store(path)
    sid = s.create_session("keep")
    s.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 13")
    conn.execute("ALTER TABLE sessions DROP COLUMN pinned")
    conn.commit(); conn.close()
    s2 = Store(path)
    assert s2.list_sessions()[0]["id"] == sid and s2.list_sessions()[0]["pinned"] is False
