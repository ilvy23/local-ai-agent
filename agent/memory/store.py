"""SQLite persistence for sessions, messages, facts, and audit log.

One database file backs the whole app (`data/agent.db` by default).
WAL mode and 0600 permissions are set on open. A later task adds vector
memory on top of the same DB; this module stays focused on relational
persistence only.
"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


def _create_v4_schema(conn: sqlite3.Connection) -> None:
    """Distinguish "personal" facts (explicitly asked to be remembered, or
    distilled from the user's own words about themselves - always relevant,
    safe to show every turn) from "incidental" ones (passively picked up
    while doing something else, e.g. a product spec learned mid-search -
    only worth showing when actually relevant to what's being discussed).

    Without this, every fact ever learned - personal or not - got dumped
    into every single system prompt under "Known facts about the user,"
    which both grew unbounded and mislabeled things like a headphone spec
    as user context, giving a small model plausible-looking but irrelevant
    material to latch onto mid-reply. Existing rows default to "personal"
    since that was the only kind that existed before this column.
    """
    conn.executescript(
        "ALTER TABLE facts ADD COLUMN kind TEXT NOT NULL DEFAULT 'personal';"
    )


def _create_v5_schema(conn: sqlite3.Connection) -> None:
    """Tracks auto-compaction progress for a long-running session: how far
    (by message id) the old messages trimmed for budget have already been
    folded into `sessions.summary`. Without a watermark, every turn would
    either re-summarize the same dropped messages again (wasted background-
    model calls every turn) or never know whether new material needs
    folding in. NULL means nothing has been compacted yet.
    """
    conn.executescript(
        "ALTER TABLE sessions ADD COLUMN summarized_through_id INTEGER;"
    )


def _create_v6_schema(conn: sqlite3.Connection) -> None:
    """A session-scoped scratchpad the model can jot to during a
    conversation (via the `take_note` tool) - its own working state for a
    long/complex session ("step 3 done, next is X"), separate from both the
    long-term `facts` table and the automatic compaction summary. Cleared
    naturally: a new session starts with NULL here, nothing carries over.
    """
    conn.executescript(
        "ALTER TABLE sessions ADD COLUMN working_notes TEXT;"
    )


def _create_v7_schema(conn: sqlite3.Connection) -> None:
    """Durable pending-tool-approval queue for the web frontend: unlike the
    terminal's console.input(), an HTTP request can't block waiting for a
    tap on a phone that may reconnect from a different device/tab minutes
    later. A CAUTION/DANGEROUS tool call is persisted here instead of
    gating inline, and a separate endpoint later resumes the exact same
    agent turn once a decision arrives - see agent/web/agent_runtime.py.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pending_approvals (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            tool_name TEXT NOT NULL,
            arguments TEXT NOT NULL,         -- JSON
            risk_level TEXT NOT NULL,        -- RiskLevel name, e.g. "CAUTION"/"DANGEROUS"
            description TEXT NOT NULL,       -- what _format_tool_call would have shown
            status TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | denied
            created_at TEXT NOT NULL,
            decided_at TEXT,
            -- everything needed to resume _run_agent_turn's loop exactly
            -- where it left off:
            turn_state TEXT NOT NULL         -- JSON: {"messages": [...], "tools_called": [...], "iteration": N}
        );
        """
    )


def _create_v8_schema(conn: sqlite3.Connection) -> None:
    """Download tokens for the send_file tool: a random, single-purpose
    token maps to one absolute path, so the web frontend can hand the phone
    a link (/api/files/{token}) without ever exposing the real filesystem
    path or making any file on disk guessable/enumerable.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS file_shares (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            token TEXT NOT NULL UNIQUE,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def _create_v9_schema(conn: sqlite3.Connection) -> None:
    """Adds the download-vs-stream choice to file_shares: send_file now lets
    the caller force `inline` (play/scrub in the chat, e.g. big videos) or
    `attachment` (save-to-device prompt) instead of only guessing from the
    file's mimetype - a 3GB movie is exactly the case where re-downloading
    the whole thing just to change your mind about the mode is painful.
    """
    conn.execute(
        "ALTER TABLE file_shares ADD COLUMN disposition TEXT NOT NULL DEFAULT 'attachment'"
    )


def _create_v10_schema(conn: sqlite3.Connection) -> None:
    """The name shown to the browser (Content-Disposition) can now differ
    from the served file's own basename - needed once send_file started
    transparently remuxing incompatible video containers: the browser
    should still see "Movie Name.mp4", not the remux cache's hashed
    filename. NULL for rows written before this column existed; callers
    fall back to the served path's own basename in that case.
    """
    conn.execute("ALTER TABLE file_shares ADD COLUMN display_name TEXT")


def _create_v11_schema(conn: sqlite3.Connection) -> None:
    """Skill persistence: reusable solution patterns the agent learns across
    sessions. Each row captures a class of problem, the steps that solved it,
    and usage counters so successful patterns surface first. solution_steps,
    tools_used, and tags are JSON arrays stored as TEXT.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            problem_description TEXT NOT NULL,
            solution_steps TEXT NOT NULL,
            tools_used TEXT,
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            last_used TEXT,
            created_at TEXT NOT NULL,
            tags TEXT
        );
        """
    )


def _create_v12_schema(conn: sqlite3.Connection) -> None:
    """Confidence tag on every fact: 'verified' (user stated it explicitly, or
    it came from a tool result) vs 'assumed' (agent inferred it — not directly
    stated). Used to flag stale assumed facts in the system prompt so the model
    knows to re-verify them rather than cite them as established truth.

    Existing rows default to 'verified': the pre-v12 distiller already required
    only things the user actually said, so all legacy facts qualify.
    """
    conn.execute(
        "ALTER TABLE facts ADD COLUMN confidence TEXT NOT NULL DEFAULT 'verified'"
    )


def _create_v13_schema(conn: sqlite3.Connection) -> None:
    """Security Audit Mode persistence: three tables that back an immutable
    audit session lifecycle.

    audit_sessions — one row per audit engagement, written once at session
    start and never updated by agent code (only a human restart changes scope).
    allowed_hosts and allowed_tools are JSON arrays; status is 'running' |
    'completed' | 'aborted'.

    audit_evidence — evidence items collected during the audit, each linked to
    a session. data is a JSON blob whose shape depends on source_tool.
    severity follows a standard triage vocabulary: 'info', 'low', 'medium',
    'high', 'critical'.

    audit_milestones — one row per completed phase gate, linking a session to
    a phase_id (e.g. 'M1'–'M6'). approved_by distinguishes automatic
    progression ('auto') from explicit human sign-off ('human').
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS audit_sessions (
            id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            allowed_hosts TEXT NOT NULL,
            allowed_tools TEXT NOT NULL,
            notes TEXT DEFAULT '',
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL DEFAULT 'running'
        );

        CREATE TABLE IF NOT EXISTS audit_evidence (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            source_tool TEXT NOT NULL,
            label TEXT NOT NULL,
            severity TEXT NOT NULL,
            data TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES audit_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS audit_milestones (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            phase_id TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            approved_by TEXT,
            summary TEXT,
            FOREIGN KEY (session_id) REFERENCES audit_sessions(id)
        );
        """
    )


def _create_v14_schema(conn: sqlite3.Connection) -> None:
    """Pin flag for the Sessions screen ("pinned first" sort, a pinned tag on the list).
    Default 0 so every existing session stays unpinned."""
    conn.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")


def _create_v15_schema(conn: sqlite3.Connection) -> None:
    """Albums: the items of one send_album share a group_id (and the group's display name), so Apps > Shared files
    can show them as one row. NULL for ordinary single-file shares."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(file_shares)")}      # idempotent: tests re-run it on a rewound version
    for col in ("group_id", "group_name"):
        if col not in have:
            conn.execute(f"ALTER TABLE file_shares ADD COLUMN {col} TEXT")


# Ordered list of (target_version, migration) steps. Each step runs only when
# the database's current `user_version` is below its target version. Append
# new (version, callable) entries here for future schema changes; never
# rewrite existing entries.
_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _create_v1_schema),
    (2, _create_v2_schema),
    (3, _create_v3_schema),
    (4, _create_v4_schema),
    (5, _create_v5_schema),
    (6, _create_v6_schema),
    (7, _create_v7_schema),
    (8, _create_v8_schema),
    (9, _create_v9_schema),
    (10, _create_v10_schema),
    (11, _create_v11_schema),
    (12, _create_v12_schema),
    (13, _create_v13_schema),
    (14, _create_v14_schema),
    (15, _create_v15_schema),
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

        # check_same_thread=False: the CLI never touches this from more than
        # one thread, but the web backend (agent/web/) hands one Store per
        # request to asyncio.to_thread() calls that run on a *different*
        # thread than the one that created it - never concurrently, always
        # sequentially within that one request, which is exactly the safe
        # case this flag exists for (sqlite3's own check is stricter than
        # the actual hazard: same-thread-only, not same-thread-concurrent).
        self._conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
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

    def add_fact(
        self,
        content: str,
        source_session_id: int | None,
        kind: str = "personal",
        confidence: str = "verified",
    ) -> int:
        """`kind="personal"` (the default) for something explicitly asked to
        be remembered, or genuinely about the user - always shown to the
        model. `kind="incidental"` for something passively picked up while
        doing something else (a spec found mid-search, etc.) - only shown
        when it's actually relevant to the current turn. See _create_v4_schema
        for why this distinction exists.

        `confidence="verified"` (the default) for facts the user stated
        directly or that came from a real tool result. `confidence="assumed"`
        for things the model inferred rather than was told - these are
        flagged in the system prompt as unverified after `reverify_assumed_after_hours`
        hours so the model knows to re-check them rather than cite them as fact.
        """
        now = _utcnow()
        cursor = self._conn.execute(
            "INSERT INTO facts (content, source_session_id, created_at, updated_at, kind, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, source_session_id, now, now, kind, confidence),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_active_facts(self, kind: str | None = None) -> list[dict]:
        """All active facts, or only those of one `kind` (see add_fact).

        Each returned dict includes `id`, `content`, `created_at`,
        `updated_at`, and `confidence` — the last two are used by recall.py
        to flag stale assumed facts in the system prompt.
        """
        if kind is None:
            rows = self._conn.execute(
                "SELECT id, content, created_at, updated_at, confidence "
                "FROM facts WHERE active = 1 ORDER BY id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, content, created_at, updated_at, confidence "
                "FROM facts WHERE active = 1 AND kind = ? ORDER BY id",
                (kind,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    def search_facts_like(self, query: str, limit: int = 8) -> list[dict]:
        """Substring fact search, used as a fallback when vectors are off."""
        rows = self._conn.execute(
            "SELECT id, content, created_at, updated_at, confidence FROM facts "
            "WHERE active = 1 AND content LIKE ? ORDER BY id LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    def deactivate_fact(self, fact_id: int) -> None:
        """"Forgetting" a fact (memory forget/prune) needs to also purge
        its semantic-recall entry, not just flip `active` off - the
        non-vector recall path (recall.py's plain query) does check
        `active`, but VectorIndex.search() returns memory_items.text with
        no join back to `facts.active` at all, so a "forgotten" fact kept
        getting pulled into every turn's context via semantic search
        regardless of this flag - the same missing-cleanup bug as
        deleting a session, just on the facts side."""
        self._conn.execute(
            "UPDATE facts SET active = 0, updated_at = ? WHERE id = ?",
            (_utcnow(), fact_id),
        )
        self._purge_memory_items("fact", [fact_id])
        self._conn.commit()

    def get_messages(self, session_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            {"id": row["id"], "role": row["role"], "content": row["content"]} for row in rows
        ]

    def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT s.id AS id, s.title AS title, s.started_at AS started_at, s.pinned AS pinned,
                   COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.pinned DESC, s.id DESC
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
                "pinned": bool(row["pinned"]),
            }
            for row in rows
        ]

    def set_session_pinned(self, session_id: int, pinned: bool) -> None:
        self._conn.execute("UPDATE sessions SET pinned = ? WHERE id = ?", (1 if pinned else 0, session_id))
        self._conn.commit()

    def fork_session(self, session_id: int, through_message_id: int) -> int:
        """New session carrying a copy of `session_id`'s messages up to and including
        `through_message_id` ("Fork from here"). Facts and the vector index are untouched -
        a fork only branches the conversation, it does not duplicate long-term memory."""
        new_id = self.create_session(title=None)
        rows = self._conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? AND id <= ? ORDER BY id",
            (session_id, through_message_id),
        ).fetchall()
        self._conn.executemany(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [(new_id, r["role"], r["content"], r["created_at"]) for r in rows],
        )
        src = self._conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if src and src["title"]:
            self.set_session_title(new_id, f"{src['title']} (fork)"[:200])
        self._conn.commit()
        return new_id

    def _vectors_table_exists(self) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'memory_vectors'"
            ).fetchone()
            is not None
        )

    def _ensure_vec_extension(self) -> bool:
        """Loads sqlite-vec onto this connection if it isn't already -
        needed before touching `memory_vectors` directly. True if the
        table can now be queried, False if the extension genuinely isn't
        available on this platform (VectorIndex degrades the same way).
        Importing `sqlite_vec` directly (not `agent.memory.vectors`) is
        deliberate: `vectors.py` imports `Store`, so importing it back
        from here would be circular - the third-party extension loader
        itself has no such cycle.

        Catches bare `Exception`, matching VectorIndex's own degradation
        contract: `enable_load_extension` is compiled out entirely on a
        CPython built without --enable-loadable-sqlite-extensions, which
        raises AttributeError, not OperationalError - a narrower catch
        here would let that escape and break every session delete on
        such a build instead of just skipping vector cleanup."""
        try:
            self._conn.execute("SELECT 1 FROM memory_vectors LIMIT 0")
            return True  # already loaded (e.g. a VectorIndex shares this conn)
        except sqlite3.OperationalError:
            pass
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.execute("SELECT 1 FROM memory_vectors LIMIT 0")
            return True
        except Exception:
            return False

    def _purge_memory_items(self, kind: str, ref_ids: list[int]) -> None:
        """Removes the memory_items AND memory_vectors rows for a set of
        `kind`/`ref_id` pairs (kind='message' + a messages.id, or
        kind='fact' + a facts.id), so removed content can't still get
        pulled into a future conversation's context via semantic recall -
        and so its orphaned vectors don't crowd out real results either.

        Deleting only memory_items and leaving memory_vectors behind was
        tried first and is actually WORSE than doing nothing:
        VectorIndex.search() runs KNN inside memory_vectors first (`WHERE
        embedding MATCH ? AND k = ?`) and only THEN joins against
        memory_items - so an orphaned vector still consumes one of the k
        nearest-neighbour slots, silently pushing out a real, unrelated
        result. Deleting one session was enough to make search() return
        nothing at all, permanently, for facts that were never touched.
        Both tables have to go together.

        If memory_vectors exists but can't be opened right now (extension
        missing on THIS machine - e.g. this DB gets synced between a PC
        and a laptop that don't necessarily have matching Python builds -
        or a transient lock), this is genuinely unsafe to do half of:
        deleting memory_items alone would orphan whatever vectors are in
        there, recreating the exact bug this method exists to fix. So the
        whole purge is skipped (both tables left untouched) rather than
        only fixing half the problem - a rare skipped cleanup is a far
        smaller cost than actively re-breaking semantic recall."""
        if not ref_ids:
            return
        placeholders = ",".join("?" * len(ref_ids))
        item_ids = [
            row["id"]
            for row in self._conn.execute(
                f"SELECT id FROM memory_items WHERE kind = ? AND ref_id IN ({placeholders})",
                [kind, *ref_ids],
            )
        ]
        if not item_ids:
            return
        vectors_ok = self._ensure_vec_extension()
        if not vectors_ok and self._vectors_table_exists():
            logger.warning(
                "Skipping memory purge for %d %s item(s): memory_vectors exists but "
                "isn't openable on this connection (sqlite-vec unavailable here?) - "
                "deleting memory_items alone would orphan their vectors instead.",
                len(item_ids), kind,
            )
            return
        item_placeholders = ",".join("?" * len(item_ids))
        if vectors_ok:
            self._conn.execute(
                f"DELETE FROM memory_vectors WHERE rowid IN ({item_placeholders})", item_ids
            )
        self._conn.execute(
            f"DELETE FROM memory_items WHERE id IN ({item_placeholders})", item_ids
        )

    def delete_session(self, session_id: int) -> None:
        """Deletes one session and everything scoped to it, including its
        embedded messages' entries in the semantic-recall index - see
        _purge_memory_items. working_notes and the compaction summary
        live as plain columns on the sessions row itself, so deleting that
        row handles those automatically. Distilled `facts` are deliberately
        left alone: they're long-term memory the user asked to be
        remembered, not scoped to the conversation that produced them."""
        message_ids = [
            row["id"]
            for row in self._conn.execute(
                "SELECT id FROM messages WHERE session_id = ?", (session_id,)
            )
        ]
        self._purge_memory_items("message", message_ids)
        for table in ("messages", "pending_approvals", "file_shares"):
            self._conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    def delete_all_sessions(self) -> None:
        message_ids = [row["id"] for row in self._conn.execute("SELECT id FROM messages")]
        self._purge_memory_items("message", message_ids)
        for table in ("messages", "pending_approvals", "file_shares", "sessions"):
            self._conn.execute(f"DELETE FROM {table}")
        self._conn.commit()

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

    def get_session_compaction(self, session_id: int) -> tuple[str | None, int | None]:
        """(summary, summarized_through_id) - both None if nothing compacted yet."""
        row = self._conn.execute(
            "SELECT summary, summarized_through_id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None, None
        return row["summary"], row["summarized_through_id"]

    def set_session_compaction(self, session_id: int, summary: str, through_id: int) -> None:
        self._conn.execute(
            "UPDATE sessions SET summary = ?, summarized_through_id = ? WHERE id = ?",
            (summary, through_id, session_id),
        )
        self._conn.commit()

    def get_working_notes(self, session_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT working_notes FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row["working_notes"] if row else None

    def append_working_note(
        self, session_id: int, note: str, max_chars: int = 2000
    ) -> str:
        """Append `note` to the session's scratchpad, trimming the oldest
        content if the total exceeds `max_chars` - a running note, not an
        ever-growing log. Returns the new full text."""
        existing = self.get_working_notes(session_id) or ""
        combined = f"{existing}\n- {note}".strip() if existing else f"- {note}"
        if len(combined) > max_chars:
            combined = combined[-max_chars:]
        self._conn.execute(
            "UPDATE sessions SET working_notes = ? WHERE id = ?", (combined, session_id)
        )
        self._conn.commit()
        return combined

    def clear_working_notes(self, session_id: int) -> None:
        self._conn.execute(
            "UPDATE sessions SET working_notes = NULL WHERE id = ?", (session_id,)
        )
        self._conn.commit()

    def set_working_notes(self, session_id: int, notes: str, max_chars: int = 2000) -> str:
        """Replace (not append) the session's scratchpad - the Sessions screen's notes editor;
        `append_working_note` (the take_note tool) still appends during a chat turn."""
        notes = notes[-max_chars:]
        self._conn.execute("UPDATE sessions SET working_notes = ? WHERE id = ?", (notes or None, session_id))
        self._conn.commit()
        return notes

    def add_pending_approval(
        self,
        session_id: int,
        tool_name: str,
        arguments: dict,
        risk_level: str,
        description: str,
        turn_state: dict,
    ) -> int:
        """Persist a CAUTION/DANGEROUS tool call awaiting a yes/no decision -
        the web frontend's alternative to the terminal's blocking
        console.input() prompt. `turn_state` carries everything needed to
        resume the agent loop once a decision arrives (see
        agent/web/agent_runtime.py)."""
        cursor = self._conn.execute(
            "INSERT INTO pending_approvals "
            "(session_id, tool_name, arguments, risk_level, description, "
            "status, created_at, turn_state) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                session_id, tool_name, json.dumps(arguments), risk_level,
                description, _utcnow(), json.dumps(turn_state),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_pending_approval(self, approval_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, session_id, tool_name, arguments, risk_level, description, "
            "status, created_at, decided_at, turn_state FROM pending_approvals WHERE id = ?",
            (approval_id,),
        ).fetchone()
        return self._approval_row_to_dict(row) if row else None

    def list_pending_approvals(
        self, session_id: int | None = None, status: str = "pending"
    ) -> list[dict]:
        if session_id is None:
            rows = self._conn.execute(
                "SELECT id, session_id, tool_name, arguments, risk_level, description, "
                "status, created_at, decided_at, turn_state FROM pending_approvals "
                "WHERE status = ? ORDER BY id",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, session_id, tool_name, arguments, risk_level, description, "
                "status, created_at, decided_at, turn_state FROM pending_approvals "
                "WHERE status = ? AND session_id = ? ORDER BY id",
                (status, session_id),
            ).fetchall()
        return [self._approval_row_to_dict(row) for row in rows]

    def decide_pending_approval(self, approval_id: int, status: str) -> None:
        """`status` is "approved" or "denied"."""
        self._conn.execute(
            "UPDATE pending_approvals SET status = ?, decided_at = ? WHERE id = ?",
            (status, _utcnow(), approval_id),
        )
        self._conn.commit()

    @staticmethod
    def _approval_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "tool_name": row["tool_name"],
            "arguments": json.loads(row["arguments"]),
            "risk_level": row["risk_level"],
            "description": row["description"],
            "status": row["status"],
            "created_at": row["created_at"],
            "decided_at": row["decided_at"],
            "turn_state": json.loads(row["turn_state"]),
        }

    def add_file_share(
        self, session_id: int, path: str, disposition: str = "attachment",
        display_name: str | None = None, group_id: str | None = None, group_name: str | None = None,
    ) -> str:
        """Mint a random download token for `path` and return it. The token,
        not the path, is what ever reaches the browser/model reply.
        `disposition` is "inline" (stream/play in the browser) or
        "attachment" (save-to-device prompt) - fixed at creation time so the
        link's behavior can't be changed after the fact. `display_name`
        overrides what the browser sees the file named as (Content-
        Disposition) when it differs from `path`'s own basename - e.g. a
        remuxed video cached under a hashed filename should still present
        as the real movie name."""
        token = secrets.token_urlsafe(32)
        self._conn.execute(
            "INSERT INTO file_shares (session_id, token, path, created_at, disposition, display_name, group_id, group_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, token, path, _utcnow(), disposition, display_name, group_id, group_name),
        )
        self._conn.commit()
        return token

    def get_file_share(self, token: str) -> dict | None:
        row = self._conn.execute(
            "SELECT path, disposition, display_name FROM file_shares WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        return {
            "path": row["path"],
            "disposition": row["disposition"],
            "display_name": row["display_name"],
        }

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

    # ── Skill persistence ────────────────────────────────────────────────────

    def add_skill(
        self,
        title: str,
        problem_description: str,
        solution_steps: list[str],
        tools_used: list[str],
        tags: list[str],
    ) -> int:
        now = _utcnow()
        cursor = self._conn.execute(
            "INSERT INTO skills "
            "(title, problem_description, solution_steps, tools_used, created_at, tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                title,
                problem_description,
                json.dumps(solution_steps),
                json.dumps(tools_used),
                now,
                json.dumps(tags),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_skill(self, skill_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, title, problem_description, solution_steps, tools_used, "
            "success_count, failure_count, last_used, created_at, tags "
            "FROM skills WHERE id = ?",
            (skill_id,),
        ).fetchone()
        return self._skill_row_to_dict(row) if row else None

    def list_skills(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, title, problem_description, solution_steps, tools_used, "
            "success_count, failure_count, last_used, created_at, tags "
            "FROM skills ORDER BY success_count DESC, id DESC"
        ).fetchall()
        return [self._skill_row_to_dict(row) for row in rows]

    def search_skills_like(self, query: str, limit: int = 8) -> list[dict]:
        """Substring search over title + problem_description + tags."""
        rows = self._conn.execute(
            "SELECT id, title, problem_description, solution_steps, tools_used, "
            "success_count, failure_count, last_used, created_at, tags "
            "FROM skills WHERE title LIKE ? OR problem_description LIKE ? OR tags LIKE ? "
            "ORDER BY success_count DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [self._skill_row_to_dict(row) for row in rows]

    def update_skill_success(self, skill_id: int) -> None:
        self._conn.execute(
            "UPDATE skills SET success_count = success_count + 1, last_used = ? WHERE id = ?",
            (_utcnow(), skill_id),
        )
        self._conn.commit()

    def update_skill_failure(self, skill_id: int) -> None:
        self._conn.execute(
            "UPDATE skills SET failure_count = failure_count + 1, last_used = ? WHERE id = ?",
            (_utcnow(), skill_id),
        )
        self._conn.commit()

    def delete_skill(self, skill_id: int) -> None:
        self._conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        self._conn.commit()

    @staticmethod
    def _skill_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "problem_description": row["problem_description"],
            "solution_steps": json.loads(row["solution_steps"]),
            "tools_used": json.loads(row["tools_used"] or "[]"),
            "success_count": row["success_count"],
            "failure_count": row["failure_count"],
            "last_used": row["last_used"],
            "created_at": row["created_at"],
            "tags": json.loads(row["tags"] or "[]"),
        }
