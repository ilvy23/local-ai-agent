"""Semantic vector memory backed by sqlite-vec on the shared DB connection.

The vec0 virtual table lives in the same SQLite database as the relational
store; each vector's rowid matches a `memory_items` metadata row. If the
sqlite-vec extension can't be loaded on this platform, the index degrades
gracefully: writes are dropped and searches return nothing, so the chat loop
keeps working without semantic recall.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from datetime import UTC, datetime

from agent.memory.store import Store

logger = logging.getLogger(__name__)


def _load_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec onto `conn`. Raises if unavailable (caught by caller)."""
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _pack(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


class VectorIndex:
    """KNN vector store over `memory_items`, keyed by rowid."""

    # Default output size (nomic-embed-text). The active size can differ if the
    # embedding model was swapped (e.g. bge-m3 = 1024); it's stored in the DB by
    # `reembed` and read back here, so the vec0 table and add() always agree.
    DIM = 768

    def __init__(self, store: Store) -> None:
        self._store = store
        self._conn = store.conn
        self.available = False
        self.DIM = self._read_dim()

        try:
            _load_extension(self._conn)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors "
                f"USING vec0(embedding float[{self.DIM}])"
            )
            self.available = True
        except Exception as exc:  # noqa: BLE001 - any load failure must degrade
            logger.warning(
                "Vector memory disabled (sqlite-vec unavailable): %s. "
                "Chat still works; semantic recall is off.",
                exc,
            )

    def _read_dim(self) -> int:
        """Active embedding size, stored by `reembed`; falls back to the default."""
        try:
            row = self._conn.execute(
                "SELECT v FROM app_state WHERE k = 'embed_dim'"
            ).fetchone()
            return int(row[0]) if row and row[0] else type(self).DIM
        except Exception:  # noqa: BLE001 - missing table on a very fresh DB
            return type(self).DIM

    def add(self, kind: str, ref_id: int | None, text: str, embedding: list[float]) -> None:
        """Store a metadata row plus its embedding. No-op if unavailable."""
        if not self.available:
            return

        if len(embedding) != self.DIM:
            raise ValueError(
                f"Embedding has {len(embedding)} dimensions, expected {self.DIM}. "
                "models.embed in config.yaml changed without re-embedding — run "
                "`agent reembed <model>` to rebuild the index at the new size."
            )

        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO memory_items (kind, ref_id, text, created_at) VALUES (?, ?, ?, ?)",
            (kind, ref_id, text, now),
        )
        item_id = cursor.lastrowid
        self._conn.execute(
            "INSERT INTO memory_vectors (rowid, embedding) VALUES (?, ?)",
            (item_id, _pack(embedding)),
        )
        self._conn.commit()

    def search(
        self,
        embedding: list[float],
        k: int = 8,
        kinds: list[str] | None = None,
    ) -> list[tuple[str, str, int | None, float]]:
        """Return up to k nearest items as (text, kind, ref_id, distance)."""
        if not self.available:
            return []

        # Over-fetch when filtering by kind so the post-filter can still return
        # k results (KNN can't be combined with a WHERE on the joined table).
        fetch_k = k * 4 if kinds else k
        rows = self._conn.execute(
            """
            SELECT i.text, i.kind, i.ref_id, v.distance
            FROM memory_vectors v
            JOIN memory_items i ON i.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_pack(embedding), fetch_k),
        ).fetchall()

        results = [(r["text"], r["kind"], r["ref_id"], r["distance"]) for r in rows]
        if kinds is not None:
            allowed = set(kinds)
            results = [r for r in results if r[1] in allowed]
        return results[:k]

    def max_cosine_similarity(self, embedding: list[float], kind: str) -> float:
        """Highest cosine similarity between `embedding` and any stored item of
        `kind`. Returns 0.0 if the index is empty or unavailable."""
        if not self.available:
            return 0.0
        row = self._conn.execute(
            """
            SELECT MIN(vec_distance_cosine(v.embedding, ?)) AS d
            FROM memory_vectors v
            JOIN memory_items i ON i.id = v.rowid
            WHERE i.kind = ?
            """,
            (_pack(embedding), kind),
        ).fetchone()
        if row is None or row["d"] is None:
            return 0.0
        return 1.0 - row["d"]
