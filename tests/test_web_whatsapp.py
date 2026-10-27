"""Read-only WhatsApp routes, against a synthetic database with invented names and messages."""

import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


@pytest.fixture()
def tc(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    db = tmp_path / "synthetic.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE wa_chats (jid TEXT PRIMARY KEY, name TEXT, is_group INTEGER, last_message_at INTEGER);
        CREATE TABLE wa_messages (id TEXT PRIMARY KEY, chat_jid TEXT, sender_jid TEXT, from_me INTEGER, timestamp INTEGER, text TEXT, type TEXT);
        INSERT INTO wa_chats VALUES ('a@x', 'Test Alice', 0, 300), ('b@x', 'Test Group', 1, 200), ('e@x', 'Empty', 0, 1), ('s@broadcast', 'Status', 0, 5);
        INSERT INTO wa_messages VALUES ('1','a@x',NULL,1,100,'hello there','text'), ('2','a@x','a@x',0,200,'100% sure_thing','text'),
            ('3','a@x','a@x',0,300,NULL,'image'), ('4','b@x','b@x',0,150,'group hello','text'), ('5','s@broadcast','s',0,5,'hello','text');
    """)
    c.commit(); c.close()
    cfg = {"data": {"db_path": str(db)}}
    app.dependency_overrides[deps.get_config] = lambda: cfg
    yield TestClient(app), cfg
    app.dependency_overrides.clear()


def test_chats_newest_first_without_empty_or_broadcast(tc):
    c, _ = tc
    rows = c.get("/api/wa/chats").json()
    assert [r["name"] for r in rows] == ["Test Alice", "Test Group"]
    assert rows[0]["message_count"] == 3 and rows[0]["last_message_at"] == 300 and rows[1]["is_group"] is True


def test_messages_paging_oldest_first(tc):
    c, _ = tc
    assert [m["id"] for m in c.get("/api/wa/chats/a@x/messages").json()] == ["1", "2", "3"]
    page = c.get("/api/wa/chats/a@x/messages?limit=1&before=300").json()
    assert [m["id"] for m in page] == ["2"] and page[0]["from_me"] is False
    assert c.get("/api/wa/chats/nobody/messages").json() == []


def test_search_escapes_wildcards_and_filters_contact(tc):
    c, _ = tc
    assert len(c.get("/api/wa/search", params={"q": "hello"}).json()) == 3
    assert [m["id"] for m in c.get("/api/wa/search", params={"q": "hello", "contact": "group"}).json()] == ["4"]
    assert [m["id"] for m in c.get("/api/wa/search", params={"q": "100%"}).json()] == ["2"]
    assert [m["id"] for m in c.get("/api/wa/search", params={"q": "%"}).json()] == ["2"]
    assert c.get("/api/wa/search", params={"q": "' OR 1=1 --"}).json() == []


def test_missing_database_or_tables_give_empty_lists(tc, tmp_path):
    c, cfg = tc
    cfg["data"]["db_path"] = str(tmp_path / "nope.db")
    assert c.get("/api/wa/chats").json() == [] and c.get("/api/wa/search", params={"q": "x"}).json() == []
    sqlite3.connect(tmp_path / "empty.db").close()
    cfg["data"]["db_path"] = str(tmp_path / "empty.db")
    assert c.get("/api/wa/chats").json() == []


def test_read_only_no_write_routes(tc):
    c, _ = tc
    assert c.post("/api/wa/chats/a@x/messages", json={"text": "hi"}).status_code == 405
