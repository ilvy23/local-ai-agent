import sqlite3
import stat
from pathlib import Path

import pytest

from agent.memory.store import Store


def test_creates_db_file_with_secure_permissions(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    assert not db_path.exists()

    Store(db_path)

    assert db_path.exists()
    mode = stat.S_IMODE(db_path.stat().st_mode)
    assert mode == 0o600


def test_sets_busy_timeout_for_concurrent_access(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    store = Store(db_path)

    timeout_ms = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]

    assert timeout_ms == 5000
    store.close()


def test_enables_wal_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    store = Store(db_path)

    conn = sqlite3.connect(db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()

    assert mode.lower() == "wal"
    store.close()


def test_creates_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    Store(db_path)

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()

    assert {"sessions", "messages", "facts", "audit_log"} <= tables


def test_sets_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    store = Store(db_path)

    from agent.memory.store import SCHEMA_VERSION

    assert store.schema_version() == SCHEMA_VERSION


def test_reopening_existing_db_does_not_fail(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    Store(db_path).close()

    from agent.memory.store import SCHEMA_VERSION

    store = Store(db_path)
    assert store.schema_version() == SCHEMA_VERSION
    store.close()


def test_migrate_does_not_rerun_already_applied_steps(tmp_path: Path, monkeypatch) -> None:
    from agent.memory import store as store_module

    calls: list[int] = []
    original_steps = store_module._MIGRATIONS

    def _tracking_step(conn: sqlite3.Connection) -> None:
        calls.append(1)

    monkeypatch.setattr(store_module, "_MIGRATIONS", [(1, _tracking_step)])

    db_path = tmp_path / "agent.db"
    Store(db_path).close()
    assert calls == [1]

    Store(db_path).close()
    assert calls == [1]

    monkeypatch.setattr(store_module, "_MIGRATIONS", original_steps)


def test_context_manager_closes_connection_on_exit(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"

    with Store(db_path) as store:
        store.create_session()

    with pytest.raises(sqlite3.ProgrammingError):
        store.create_session()


def test_migrations_versions_are_strictly_increasing() -> None:
    from agent.memory.store import _MIGRATIONS

    versions = [v for v, _ in _MIGRATIONS]
    assert versions == sorted(set(versions))


def test_create_session_returns_id_and_sets_timestamps(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")

    session_id = store.create_session()

    assert isinstance(session_id, int)
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id
    assert sessions[0]["started_at"]


def test_create_session_with_title(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")

    session_id = store.create_session(title="Hello world")

    sessions = store.list_sessions()
    assert sessions[0]["title"] == "Hello world"
    assert sessions[0]["id"] == session_id


def test_add_message_returns_row_id(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    session_id = store.create_session()

    first = store.add_message(session_id, "user", "hi")
    second = store.add_message(session_id, "assistant", "hello")

    assert isinstance(first, int)
    assert second == first + 1


def test_memory_items_table_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    Store(db_path).close()
    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()
    assert "memory_items" in tables


def test_facts_round_trip_add_list_deactivate(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")

    fact_id = store.add_fact("dog is named Rex", source_session_id=None)
    assert isinstance(fact_id, int)

    facts = store.get_active_facts()
    assert [f["content"] for f in facts] == ["dog is named Rex"]
    assert facts[0]["id"] == fact_id

    store.deactivate_fact(fact_id)
    assert store.get_active_facts() == []


def test_get_active_facts_excludes_inactive(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    keep = store.add_fact("likes tea", source_session_id=None)
    drop = store.add_fact("likes coffee", source_session_id=None)
    store.deactivate_fact(drop)

    contents = [f["content"] for f in store.get_active_facts()]
    assert contents == ["likes tea"]
    assert keep


def test_add_message_and_get_messages_round_trip(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    session_id = store.create_session()

    store.add_message(session_id, "user", "hi there")
    store.add_message(session_id, "assistant", "hello!")

    messages = store.get_messages(session_id)
    assert messages == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"},
    ]


def test_list_sessions_reports_message_count(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    session_id = store.create_session(title="Chat 1")
    store.add_message(session_id, "user", "hi")
    store.add_message(session_id, "assistant", "hello")

    other_id = store.create_session(title="Chat 2")
    store.add_message(other_id, "user", "one message only")

    sessions = store.list_sessions()

    by_id = {s["id"]: s for s in sessions}
    assert by_id[session_id]["message_count"] == 2
    assert by_id[other_id]["message_count"] == 1


def test_list_sessions_orders_most_recent_first(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    first_id = store.create_session(title="First")
    second_id = store.create_session(title="Second")

    sessions = store.list_sessions()

    assert [s["id"] for s in sessions] == [second_id, first_id]


def test_list_sessions_respects_limit(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    for i in range(5):
        store.create_session(title=f"Chat {i}")

    sessions = store.list_sessions(limit=2)

    assert len(sessions) == 2


def test_get_last_session_id_returns_most_recent(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    store.create_session(title="First")
    second_id = store.create_session(title="Second")

    assert store.get_last_session_id() == second_id


def test_get_last_session_id_returns_none_when_no_sessions(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")

    assert store.get_last_session_id() is None


def test_touch_session_updates_last_active_at(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    session_id = store.create_session()

    conn = sqlite3.connect(store.db_path)
    before = conn.execute(
        "SELECT last_active_at FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()[0]
    conn.close()

    store.touch_session(session_id)

    conn = sqlite3.connect(store.db_path)
    after = conn.execute(
        "SELECT last_active_at FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()[0]
    conn.close()

    assert after >= before


def test_set_session_title_updates_title(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    session_id = store.create_session()

    store.set_session_title(session_id, "New title")

    sessions = store.list_sessions()
    assert sessions[0]["title"] == "New title"


def test_add_audit_log_round_trip(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")

    row_id = store.add_audit_log(
        kind="tool", detail='{"name": "current_time", "arguments": {}}', approved=1, result="ok"
    )

    assert isinstance(row_id, int)
    conn = sqlite3.connect(store.db_path)
    row = conn.execute(
        "SELECT kind, detail, approved, result FROM audit_log WHERE id = ?", (row_id,)
    ).fetchone()
    conn.close()
    assert row == ("tool", '{"name": "current_time", "arguments": {}}', 1, "ok")
    store.close()


def test_get_audit_log_returns_recent_first_with_filter(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    store.add_audit_log(kind="shell", detail="ls", approved=1, result="ok")
    store.add_audit_log(kind="tool", detail="read", approved=1, result="read 5 chars")
    store.add_audit_log(kind="shell", detail="pwd", approved=1, result="ok")

    rows = store.get_audit_log(limit=30)
    assert [r["detail"] for r in rows] == ["pwd", "read", "ls"]  # newest first
    assert set(rows[0].keys()) >= {"ts", "kind", "detail", "approved", "result"}

    shell_only = store.get_audit_log(limit=30, kind="shell")
    assert {r["kind"] for r in shell_only} == {"shell"}

    limited = store.get_audit_log(limit=1)
    assert len(limited) == 1
    store.close()


def test_messages_persist_across_reopening_store(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    store = Store(db_path)
    session_id = store.create_session(title="Persisted chat")
    store.add_message(session_id, "user", "hi there")
    store.add_message(session_id, "assistant", "hello!")
    store.close()

    reopened = Store(db_path)
    messages = reopened.get_messages(session_id)
    assert messages == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"},
    ]
    reopened.close()


def test_schema_version_is_current(tmp_path: Path) -> None:
    from agent.memory.store import SCHEMA_VERSION

    store = Store(tmp_path / "agent.db")
    assert store.schema_version() == SCHEMA_VERSION == 3
    store.close()


def test_v3_creates_app_state_table(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.db")
    tables = {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "app_state" in tables
    # A key/value round-trip works (this is where reembed stores embed_dim).
    store.conn.execute("INSERT INTO app_state (k, v) VALUES ('embed_dim', '1024')")
    row = store.conn.execute("SELECT v FROM app_state WHERE k = 'embed_dim'").fetchone()
    assert row[0] == "1024"
    store.close()


def test_migrations_do_not_disturb_core_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    store = Store(db_path)
    session_id = store.create_session(title="still here")
    store.add_message(session_id, "user", "hi")

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()

    assert {"sessions", "messages", "facts", "audit_log", "memory_items", "app_state"} <= tables
    assert store.get_messages(session_id) == [{"role": "user", "content": "hi"}]
    store.close()
