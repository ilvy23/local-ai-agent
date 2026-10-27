"""Saving facts while you chat, never while you quit.

The old design distilled a whole session at exit: one model call over every message, then an embedding and up to three
overlap-judge calls per fact, all behind a spinner. Now:

  * the queue is durable and lives in the database (schema v16): `distill_progress` marks, per session, the last user
    message already handed to the extractor; "queue" = user messages above the mark. `learn_queue` holds tool exchanges
    that wait for the learned-fact check.
  * a background thread inside the running process takes small batches whenever the chat model is idle (and the
    governor allows it: no game, GPU not busy), extracts facts with `models.background`, and writes the facts and the
    new mark in ONE transaction. A crash before the commit loses nothing (the messages are still above the mark and are
    redone); a crash after it saves nothing twice (the mark moved with the facts).
  * quitting only stops the thread (waiting at most `memory.exit_wait_seconds` for a step that is already running),
    counts what is left and says so in one line. The leftovers are taken at the next start (`BackgroundDistiller`
    starts with a quiet catch-up) or by `nightage memory process`.

Evidence gating is unchanged: the extractor sees only the user's own words, `JUNK_FACT_RE`, the duplicate checks and the
overlap judge (agent.memory.distill.plan_facts) decide what is kept, and learned facts still need verified web evidence.
Embedding degrades gracefully: with no embedder the fact is stored and found by keyword; `repair_unembedded` indexes it
once an embedder exists."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from agent.memory import distill
from agent.memory.store import Store, _utcnow
from agent.memory.vectors import VectorIndex

logger = logging.getLogger(__name__)

BATCH_MESSAGES = 8          # user messages per extraction call
MESSAGE_CHARS = 1500        # a pasted article is not a statement about the user: only its start is read
BATCH_CHARS = 6000
MAX_LEARN_ATTEMPTS = 5
IDLE_DELAY = 1.5            # seconds of quiet after a turn before background work starts
POLL_SECONDS = 20.0         # how often a paused or failed worker looks again


def settings(config: dict[str, Any]) -> dict[str, Any]:
    m = config.get("memory", {}) or {}
    return {
        "every_turns": max(1, int(m.get("distill_every_turns", 1) or 1)),
        "batch": max(1, int(m.get("distill_batch_messages", BATCH_MESSAGES) or BATCH_MESSAGES)),
        "exit_wait": max(0.0, float(m.get("exit_wait_seconds", 3.0) if m.get("exit_wait_seconds") is not None else 3.0)),
    }


# ------------------------------------------------------------------ queue state (plain SQL, cheap)

def watermark(conn: Any, session_id: int) -> int:
    row = conn.execute("SELECT last_message_id FROM distill_progress WHERE session_id = ?", (session_id,)).fetchone()
    return int(row[0]) if row else 0


def pending_by_session(conn: Any) -> dict[int, int]:
    """{session id: number of user messages not yet handed to the extractor}, oldest session first."""
    rows = conn.execute(
        "SELECT m.session_id, COUNT(*) FROM messages m LEFT JOIN distill_progress p ON p.session_id = m.session_id "
        "WHERE m.role = 'user' AND m.id > COALESCE(p.last_message_id, 0) GROUP BY m.session_id ORDER BY MIN(m.id)"
    ).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def pending_count(conn: Any) -> int:
    """Messages waiting for fact extraction plus tool exchanges waiting for the learned-fact check."""
    jobs = conn.execute("SELECT COUNT(*) FROM learn_queue").fetchone()[0]
    return sum(pending_by_session(conn).values()) + int(jobs)


def enqueue_exchanges(store: Store, session_id: int | None, exchanges: list[tuple]) -> int:
    """Make tool exchanges `(question, reply[, evidence])` durable; returns how many were queued."""
    n = 0
    for ex in exchanges:
        evidence = ex[2] if len(ex) > 2 else None
        store.conn.execute(
            "INSERT INTO learn_queue (session_id, question, reply, evidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, ex[0], ex[1], json.dumps(evidence, default=str) if evidence else None, _utcnow()),
        )
        n += 1
    if n:
        store.conn.commit()
    return n


# ------------------------------------------------------------------ one unit of work

def distill_batch(store: Store, vectors: VectorIndex, llm: Any, config: dict[str, Any], session_id: int) -> int | None:
    """Extract facts from the next batch of one session's unprocessed user messages and commit them together with the
    new mark. Returns the number of facts saved, or None when nothing was pending. Model/connection errors propagate
    and leave the queue untouched."""
    conn = store.conn
    start = watermark(conn, session_id)
    rows = conn.execute(
        "SELECT id, content FROM messages WHERE session_id = ? AND role = 'user' AND id > ? ORDER BY id LIMIT ?",
        (session_id, start, settings(config)["batch"]),
    ).fetchall()
    if not rows:
        return None
    texts: list[str] = []
    used = 0
    end_id = start
    for r in rows:
        text = (r["content"] or "").strip()[:MESSAGE_CHARS]
        if texts and used + len(text) > BATCH_CHARS:
            break
        if text:
            texts.append(text)
            used += len(text)
        end_id = r["id"]
    saved: list[tuple[int, distill.PlannedFact]] = []
    if texts:
        candidates = distill.extract_candidates(llm, config, texts)      # may raise: queue untouched
        if candidates is None:
            logger.warning("Fact extraction returned no parseable JSON array; skipping these messages.")
        planned = distill.plan_facts(store, vectors, llm, config, candidates or []) if candidates else []
        saved = _commit(store, session_id, start, end_id, planned)
    else:
        _commit(store, session_id, start, end_id, [])
    for fact_id, p in saved:
        for old in p.supersedes:
            store.deactivate_fact(old)
        distill.index_fact(vectors, fact_id, p)
    return len(saved)


def _commit(store: Store, session_id: int, start: int, end_id: int, planned: list[distill.PlannedFact]
            ) -> list[tuple[int, distill.PlannedFact]]:
    """Facts + the new mark in one transaction; a concurrent worker that got there first wins (we save nothing)."""
    conn = store.conn
    conn.execute("BEGIN IMMEDIATE")
    try:
        if watermark(conn, session_id) != start:
            conn.rollback()
            return []
        have = {r[0] for r in conn.execute("SELECT content FROM facts WHERE active = 1")}
        saved = []
        for p in planned:
            if p.text in have:
                continue
            saved.append((store.add_fact(p.text, source_session_id=session_id, confidence=p.confidence, commit=False), p))
            have.add(p.text)
        conn.execute(
            "INSERT INTO distill_progress (session_id, last_message_id, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET last_message_id = excluded.last_message_id, updated_at = excluded.updated_at",
            (session_id, end_id, _utcnow()),
        )
        conn.commit()
        return saved
    except BaseException:
        conn.rollback()
        raise


def learn_one(store: Store, vectors: VectorIndex, llm: Any, config: dict[str, Any]) -> bool:
    """Run the learned-fact check for the oldest queued tool exchange. False when none is queued."""
    row = store.conn.execute("SELECT id, question, reply, evidence, attempts FROM learn_queue ORDER BY id LIMIT 1").fetchone()
    if row is None:
        return False
    from agent.tui import _save_learned_facts

    evidence = json.loads(row["evidence"]) if row["evidence"] else None
    try:
        _save_learned_facts(llm, config, store, vectors, [(row["question"], row["reply"], evidence)], strict=True)
    except Exception:
        if row["attempts"] + 1 >= MAX_LEARN_ATTEMPTS:
            logger.warning("Dropping a tool exchange after %d failed attempts", MAX_LEARN_ATTEMPTS, exc_info=True)
            store.conn.execute("DELETE FROM learn_queue WHERE id = ?", (row["id"],))
        else:
            store.conn.execute("UPDATE learn_queue SET attempts = attempts + 1 WHERE id = ?", (row["id"],))
        store.conn.commit()
        raise
    store.conn.execute("DELETE FROM learn_queue WHERE id = ?", (row["id"],))
    store.conn.commit()
    return True


def _background_model_missing(llm: Any, config: dict[str, Any]) -> bool:
    """True when a fresh model list says the background (notes) model is not installed. Never raises, never blocks."""
    try:
        from agent import modelpick

        client = getattr(llm, "_client", None)
        base = str(getattr(client, "base_url", "") or "")
        return bool(base) and modelpick.known_missing(base, str((config.get("models") or {}).get("background") or ""))
    except Exception:  # noqa: BLE001
        return False


def step(db_path: Path | str, llm: Any, config: dict[str, Any], *, active_session: int | None = None) -> bool:
    """One unit of work on a fresh connection (a Store belongs to the thread that made it). True = did something.

    The chat's own session waits until `memory.distill_every_turns` user messages are pending; older sessions
    (the catch-up) go at once."""
    every = settings(config)["every_turns"]
    if _background_model_missing(llm, config):
        return False                          # the notes model is not installed: leave the work queued instead of failing at every try
    with Store(Path(db_path)) as store:
        pend = pending_by_session(store.conn)
        for sid, n in pend.items():
            if sid == active_session and n < every:
                continue
            vectors = VectorIndex(store)
            distill_batch(store, vectors, llm, config, sid)
            return True
        if store.conn.execute("SELECT 1 FROM learn_queue LIMIT 1").fetchone():
            return learn_one(store, VectorIndex(store), llm, config)
    return False


def drain(db_path: Path | str, llm: Any, config: dict[str, Any], *, max_seconds: float | None = None,
          pause_fn: Callable[[dict], str | None] | None = None, on_step: Callable[[], None] | None = None) -> int:
    """Work through the whole queue (CLI `memory process`, the web server). Returns steps done."""
    pause_fn = pause_fn or governor_pause
    t0, n = time.monotonic(), 0
    while max_seconds is None or time.monotonic() - t0 < max_seconds:
        if pause_fn(config):
            break
        if not step(db_path, llm, config):
            break
        n += 1
        if on_step:
            on_step()
    return n


def governor_pause(config: dict[str, Any]) -> str | None:
    """Why background work should wait (a game, a busy GPU); the chat itself being open does not count."""
    try:
        from agent.governor import pause_reason

        return pause_reason(config, ignore_foreground=True)
    except Exception:  # noqa: BLE001 - a broken probe must not stop remembering
        return None


_drain_lock = threading.Lock()


def kick_drain(db_path: Path | str, llm: Any, config: dict[str, Any]) -> threading.Thread | None:
    """For a long-running process without a chat loop (the web server): drain the queue on a daemon thread, one
    drain at a time. None when one is already running."""
    if not _drain_lock.acquire(blocking=False):
        return None

    def work() -> None:
        try:
            drain(db_path, llm, config)
        except Exception as exc:  # noqa: BLE001 - the queue keeps the work
            logger.warning("background fact extraction failed (will retry): %s", exc)
        finally:
            _drain_lock.release()

    t = threading.Thread(target=work, name="memory-drain", daemon=True)
    t.start()
    return t


# ------------------------------------------------------------------ the thread inside a running chat

class BackgroundDistiller:
    """Takes queued work whenever the chat is idle. `busy(True/False)` brackets every turn; `stop()` is the exit."""

    def __init__(self, config: dict[str, Any], llm: Any, db_path: Path | str, *, session_id: int | None = None,
                 pause_fn: Callable[[dict], str | None] | None = None, idle_delay: float = IDLE_DELAY,
                 poll: float = POLL_SECONDS) -> None:
        self.config, self.llm, self.db_path = config, llm, Path(db_path)
        self.session_id = session_id
        self.pause_fn = pause_fn or governor_pause
        self.idle_delay, self.poll = idle_delay, poll
        self._busy = False
        self._idle_since = time.monotonic()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.in_flight = False
        self.errors = 0
        self.steps = 0
        self._problem = ""

    def take_problem(self) -> str:
        """The one-line note about a failed background save, ONCE per session (then ''): the chat shows it between turns."""
        line, self._problem = self._problem, ""
        return line

    def _failed(self, exc: Exception) -> None:
        logger.warning("background fact extraction failed (will retry): %s", exc)
        from agent import friendly

        if friendly.once("memory-extract-failed"):
            line, _hint = friendly.explain(exc, str((self.config.get("models") or {}).get("background") or ""))
            self._problem = f"Saving notes in the background failed: {line} It will retry quietly. Details: nightage doctor"

    def start(self) -> "BackgroundDistiller":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="memory-distill", daemon=True)
            self._thread.start()
            self._wake.set()                       # quiet catch-up of whatever an earlier run left behind
        return self

    def busy(self, value: bool) -> None:
        """True while the chat model works on a turn (nothing starts then); False (and a wake-up) when it is idle."""
        self._busy = value
        if not value:
            self._idle_since = time.monotonic()
            self._wake.set()

    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float = 0.0) -> bool:
        """Ask the thread to end; wait at most `timeout` seconds for a step that is already running. True = ended."""
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is None:
            return True
        t.join(timeout if self.in_flight else min(timeout, 0.2))
        return not t.is_alive()

    def _run(self) -> None:
        backoff = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=max(self.poll, backoff))
            self._wake.clear()
            backoff = 0.0
            while not self._stop.is_set():
                if self._busy:
                    break                           # busy(False) wakes us again
                wait = self.idle_delay - (time.monotonic() - self._idle_since)
                if wait > 0:
                    self._stop.wait(min(wait, 0.25))
                    continue
                if self.pause_fn(self.config):
                    break
                try:
                    self.in_flight = True
                    did = step(self.db_path, self.llm, self.config, active_session=self.session_id)
                except Exception as exc:  # noqa: BLE001 - remembering is a bonus; the queue keeps the work
                    self.errors += 1
                    self._failed(exc)
                    backoff = min(300.0, 30.0 * self.errors)
                    break
                finally:
                    self.in_flight = False
                if not did:
                    break
                self.errors = 0
                self.steps += 1


def finish(distiller: BackgroundDistiller | None, store: Store, session_id: int | None, tool_exchanges: list[tuple],
           config: dict[str, Any], console: Any = None) -> int:
    """The exit path. Durably queue the tool exchanges, stop the thread (a step already running gets at most
    `memory.exit_wait_seconds`), say in one dim line what is left. Never calls a model; returns the number left."""
    n = 0
    try:
        enqueue_exchanges(store, session_id, tool_exchanges)
        if distiller is not None:
            distiller.stop(settings(config)["exit_wait"])
        n = pending_count(store.conn)
        if n and console is not None:
            console.print(f"[dim]{n} message{'s' if n != 1 else ''} will be remembered in the background[/dim]")
    except Exception as exc:  # noqa: BLE001 - remembering must never crash on exit
        logger.warning("Wrapping up the memory queue failed: %s", exc)
    return n
