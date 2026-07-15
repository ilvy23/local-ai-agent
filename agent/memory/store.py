"""SQLite persistence for sessions, messages, facts, and audit log.

One database file backs the whole app (`data/agent.db` by default).
WAL mode and 0600 permissions are set on open. A later task adds vector
memory on top of the same DB; this module stays focused on relational
persistence only.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    title TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    source_session_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT,
    approved INTEGER,
    result TEXT
);
"""


def _create_v1_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


# Metadata rows for entries in the sqlite-vec index. The vec0 virtual table
# itself (`memory_vectors`) is created lazily by VectorIndex on the shared
# connection, since it requires the sqlite-vec extension to be loaded.
_V2_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    ref_id INTEGER,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _create_v2_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_V2_SCHEMA)


# App-wide key/value state — e.g. the active embedding dimension written by
# `reembed`. Kept separate from the domain tables so any small setting has a home.
_V3_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_state (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""


def _create_v3_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_V3_SCHEMA)


# Ordered list of (target_version, migration) steps. Each step runs only when
# the database's current `user_version` is below its target version. Append
# new (version, callable) entries here for future schema changes; never
# rewrite existing entries.
_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _create_v1_schema),
    (2, _create_v2_schema),
    (3, _create_v3_schema),
]

# Guard against an out-of-order or duplicated append, which would make _migrate
# silently skip steps.
_versions = [v for v, _ in _MIGRATIONS]
assert _versions == sorted(set(_versions)), "_MIGRATIONS versions must be strictly increasing"

SCHEMA_VERSION = _MIGRATIONS[-1][0]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    """Opens (creating if needed) the agent SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not self.db_path.exists()

        self._conn = sqlite3.connect(self.db_path, timeout=5.0)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row

        if is_new:
            self.db_path.chmod(0o600)

        self._migrate()

    def _migrate(self) -> None:
        current_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        for target_version, step in _MIGRATIONS:
            if current_version < target_version:
                step(self._conn)
                self._conn.execute(f"PRAGMA user_version = {target_version}")
                current_version = target_version
        self._conn.commit()

    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def create_session(self, title: str | None = None) -> int:
        now = _utcnow()
        cursor = self._conn.execute(
            "INSERT INTO sessions (started_at, last_active_at, title) VALUES (?, ?, ?)",
            (now, now, title),
        )
        self._conn.commit()
        return cursor.lastrowid

    @property
    def conn(self) -> sqlite3.Connection:
        """The underlying connection, shared with the vector index."""
        return self._conn

    def add_message(self, session_id: int, role: str, content: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _utcnow()),
        )
        self._conn.commit()
        return cursor.lastrowid

    def add_fact(self, content: str, source_session_id: int | None) -> int:
        now = _utcnow()
        cursor = self._conn.execute(
            "INSERT INTO facts (content, source_session_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (content, source_session_id, now, now),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_active_facts(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, content, created_at FROM facts WHERE active = 1 ORDER BY id"
        ).fetchall()
        return [
            {"id": row["id"], "content": row["content"], "created_at": row["created_at"]}
            for row in rows
        ]

    def search_facts_like(self, query: str, limit: int = 8) -> list[dict]:
        """Substring fact search, used as a fallback when vectors are off."""
        rows = self._conn.execute(
            "SELECT id, content, created_at FROM facts "
            "WHERE active = 1 AND content LIKE ? ORDER BY id LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [
            {"id": row["id"], "content": row["content"], "created_at": row["created_at"]}
            for row in rows
        ]

    def deactivate_fact(self, fact_id: int) -> None:
        self._conn.execute(
            "UPDATE facts SET active = 0, updated_at = ? WHERE id = ?",
            (_utcnow(), fact_id),
        )
        self._conn.commit()

    def get_messages(self, session_id: int) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT s.id AS id, s.title AS title, s.started_at AS started_at,
                   COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "started_at": row["started_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def get_last_session_id(self) -> int | None:
        row = self._conn.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def touch_session(self, session_id: int) -> None:
        self._conn.execute(
            "UPDATE sessions SET last_active_at = ? WHERE id = ?",
            (_utcnow(), session_id),
        )
        self._conn.commit()

    def set_session_title(self, session_id: int, title: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )
        self._conn.commit()

    def get_audit_log(self, limit: int = 30, kind: str | None = None) -> list[dict]:
        """Return the most recent audit rows (newest first), optionally by kind."""
        if kind is None:
            rows = self._conn.execute(
                "SELECT ts, kind, detail, approved, result FROM audit_log "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT ts, kind, detail, approved, result FROM audit_log "
                "WHERE kind = ? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [
            {
                "ts": row["ts"],
                "kind": row["kind"],
                "detail": row["detail"],
                "approved": row["approved"],
                "result": row["result"],
            }
            for row in rows
        ]

    def add_audit_log(self, kind: str, detail: str, approved: int, result: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO audit_log (ts, kind, detail, approved, result) VALUES (?, ?, ?, ?, ?)",
            (_utcnow(), kind, detail, approved, result),
        )
        self._conn.commit()
        return cursor.lastrowid
