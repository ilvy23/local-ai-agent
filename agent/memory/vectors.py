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
import re
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

    # Fallback size only, for callers that need *a* number before anything has
    # been embedded. The real size is never guessed: it's whatever the embedding
    # model actually returns, adopted on the first insert and recorded in
    # app_state, so the vec0 table always matches the model in config.
    DIM = 768

    def __init__(self, store: Store) -> None:
        self._store = store
        self._conn = store.conn
        self.available = False

        try:
            _load_extension(self._conn)
            self.available = True
        except Exception as exc:  # noqa: BLE001 - any load failure must degrade
            logger.warning(
                "Vector memory disabled (sqlite-vec unavailable): %s. "
                "Chat still works; semantic recall is off.",
                exc,
            )
            self.DIM = self._read_dim() or type(self).DIM
            return

        # Known size = whatever's recorded, else whatever an existing table was
        # built at. None means "nothing embedded yet" — the first add() decides.
        dim = self._read_dim()
        if dim is None:
            dim = self._table_dim()
            if dim is not None and self._is_empty():
                # An older version created this table from a hardcoded guess and
                # never embedded anything into it. An empty index has nothing to
                # lose, so drop it and let the real model size win instead of
                # failing every query with a dimension mismatch.
                self._conn.execute("DROP TABLE memory_vectors")
                self._conn.commit()
                dim = None
            elif dim is not None:
                self._remember_dim(dim)  # populated but unrecorded: trust it
        if dim:
            self._create_table(dim)
        self.DIM = dim

    def _is_empty(self) -> bool:
        try:
            return self._conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0] == 0
        except Exception:  # noqa: BLE001 - no table yet == empty
            return True

    def _read_dim(self) -> int | None:
        """The recorded embedding size, or None if nothing has been embedded."""
        try:
            row = self._conn.execute(
                "SELECT v FROM app_state WHERE k = 'embed_dim'"
            ).fetchone()
            return int(row[0]) if row and row[0] else None
        except Exception:  # noqa: BLE001 - missing table on a very fresh DB
            return None

    def _table_dim(self) -> int | None:
        """Size an existing vec0 table was built at, read back from its DDL."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_vectors'"
        ).fetchone()
        if not row or not row["sql"]:
            return None
        m = re.search(r"float\[(\d+)\]", row["sql"])
        return int(m.group(1)) if m else None

    def _remember_dim(self, dim: int) -> None:
        self._conn.execute(
            "INSERT INTO app_state (k, v) VALUES ('embed_dim', ?) "
            "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (str(dim),),
        )
        self._conn.commit()

    def _create_table(self, dim: int) -> None:
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors "
            f"USING vec0(embedding float[{dim}])"
        )

    def add(self, kind: str, ref_id: int | None, text: str, embedding: list[float]) -> None:
        """Store a metadata row plus its embedding. No-op if unavailable."""
        if not self.available:
            return

        if self.DIM is None:
            # First thing ever embedded: the model's output defines the index size.
            self.DIM = len(embedding)
            self._create_table(self.DIM)
            self._remember_dim(self.DIM)
        elif len(embedding) != self.DIM:
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
        if not self.available or self.DIM is None:
            return []  # nothing embedded yet — nothing to match against
        if len(embedding) != self.DIM:
            raise ValueError(
                f"Query embedding has {len(embedding)} dimensions but the index is "
                f"{self.DIM}. models.embed changed after the index was built — run "
                "`agent reembed <model>` to rebuild it at the new size."
            )

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
        if not self.available or self.DIM is None or len(embedding) != self.DIM:
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
