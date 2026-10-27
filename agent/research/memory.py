"""Cache of what the agent read on the web, tied into the long-term memory.

Two things are stored in the main database:
  * web_pages / web_queries: the cleaned text of every page read and the result list of every search,
    so a repeated or related question is answered from disk (instantly, no network) until the entry is
    too old for that kind of question (news: 30 min, ordinary: 12 h, docs and definitions: 7 days).
  * a short research note per search, embedded into the vector index as kind "web", so a later chat
    turn can recall "I looked this up on 2026-10-04: ..." together with its date, instead of searching
    again or answering from stale training data. Notes are never mixed into personal facts.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.parse
from datetime import date
from typing import Any, Callable

NEWS_RE = re.compile(
    r"(?i)\b(news|latest|today|now|current|currently|price|prices|weather|score|live|release|version|"
    r"aktuell\w*|heute|jetzt|preis\w*|wetter|neueste\w*|nachrichten)\b"
)
EVERGREEN_RE = re.compile(
    r"(?i)\b(docs?|documentation|tutorial|how to|what is|definition|meaning|history of|wiki\w*|syntax|"
    r"anleitung|bedeutung|erkl\w+|geschichte)\b"
)
TTL_NEWS = 30 * 60
TTL_DEFAULT = 12 * 3600
TTL_EVERGREEN = 7 * 86400
MAX_PAGES = 400
NOTE_MAX_CHARS = 700
WEB_NOTE_KIND = "web"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_pages (
    url TEXT PRIMARY KEY, title TEXT, published TEXT, text TEXT, fetched_at REAL
);
CREATE TABLE IF NOT EXISTS web_queries (
    key TEXT PRIMARY KEY, query TEXT, hits TEXT, searched_at REAL, noted INTEGER DEFAULT 0
);
"""


def ttl_for(query: str, freshness: str = "any") -> int:
    if freshness in ("day", "week") or NEWS_RE.search(query):
        return TTL_NEWS if freshness != "week" else 6 * 3600
    if EVERGREEN_RE.search(query):
        return TTL_EVERGREEN
    return TTL_DEFAULT


def query_key(query: str, freshness: str = "any") -> str:
    return f"{freshness}|{' '.join(query.lower().split())}"


class WebMemory:
    def __init__(
        self, conn: sqlite3.Connection, vectors: Any = None,
        embed: Callable[[list[str]], list[list[float]]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._conn, self._vectors, self._embed, self._clock = conn, vectors, embed, clock
        conn.executescript(_SCHEMA)
        try:                                                   # older databases: trust signals + injection count per page
            conn.execute("ALTER TABLE web_pages ADD COLUMN signals TEXT")
        except sqlite3.OperationalError:
            pass

    # pages ---------------------------------------------------------------------------------
    def page(self, url: str, max_age: float) -> dict[str, str] | None:
        row = self._conn.execute(
            "SELECT title, published, text, fetched_at, signals FROM web_pages WHERE url = ?", (url,)).fetchone()
        if not row or self._clock() - row[3] > max_age:
            return None
        try:
            signals = json.loads(row[4]) if row[4] else {}
        except ValueError:
            signals = {}
        return {"title": row[0], "published": row[1], "text": row[2], "signals": signals}

    def put_page(self, url: str, title: str, published: str, text: str, signals: dict | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO web_pages (url, title, published, text, fetched_at, signals) VALUES (?,?,?,?,?,?)",
            (url, title, published, text, self._clock(), json.dumps(signals or {})),
        )
        self._conn.execute(
            "DELETE FROM web_pages WHERE url IN (SELECT url FROM web_pages ORDER BY fetched_at DESC LIMIT -1 OFFSET ?)",
            (MAX_PAGES,),
        )
        self._conn.commit()

    # searches ------------------------------------------------------------------------------
    def hits(self, key: str, max_age: float) -> list[dict[str, str]] | None:
        row = self._conn.execute("SELECT hits, searched_at FROM web_queries WHERE key = ?", (key,)).fetchone()
        if not row or self._clock() - row[1] > max_age:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def put_hits(self, key: str, query: str, hits: list[dict[str, str]]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO web_queries (key, query, hits, searched_at, noted) VALUES (?,?,?,?,0)",
            (key, query, json.dumps(hits), self._clock()),
        )
        self._conn.commit()

    # long-term note ------------------------------------------------------------------------
    def remember(self, key: str, query: str, sources: list[dict[str, str]]) -> bool:
        """Embed one short research note for this search. Never raises; False if nothing was stored."""
        if not (self._vectors and getattr(self._vectors, "available", False) and self._embed and sources):
            return False
        row = self._conn.execute("SELECT rowid, noted FROM web_queries WHERE key = ?", (key,)).fetchone()
        if not row or row[1]:
            return False
        parts = []
        for s in sources[:3]:
            host = urllib.parse.urlparse(s["url"]).netloc.removeprefix("www.")
            when = s.get("published") or "undated"
            parts.append(f"{s['title'][:80]} ({host}, {when}): {' '.join(s.get('best', '').split())[:170]}")
        note = f"Web research on {date.fromtimestamp(self._clock()).isoformat()} for '{query[:80]}': " + " | ".join(parts)
        note = note[:NOTE_MAX_CHARS]
        try:
            embedding = self._embed([note])[0]
            self._vectors.add(WEB_NOTE_KIND, row[0], note, embedding)
            self._conn.execute("UPDATE web_queries SET noted = 1 WHERE key = ?", (key,))
            self._conn.commit()
            return True
        except Exception:  # noqa: BLE001 - memory is a bonus, the search result still goes out
            return False
