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
IDLE_DELAY = 1.5            # seconds of quiet after a turn before background work starts (when `memory.idle_seconds` is 0)
IDLE_SECONDS = 30.0         # default `memory.idle_seconds`: background work waits until the user has been quiet this long
POLL_SECONDS = 20.0         # how often a paused or failed worker looks again
MAX_BATCH_FAILS = 3         # an unparseable / cut-off extraction is retried with half the batch, at most this often


def settings(config: dict[str, Any]) -> dict[str, Any]:
    m = config.get("memory", {}) or {}
    return {
        "every_turns": max(1, int(m.get("distill_every_turns", 1) or 1)),
        "batch": max(1, int(m.get("distill_batch_messages", BATCH_MESSAGES) or BATCH_MESSAGES)),
        "exit_wait": max(0.0, float(m.get("exit_wait_seconds", 3.0) if m.get("exit_wait_seconds") is not None else 3.0)),
        # Background work replaces the conversation Ollama keeps cached (one slot): it runs only after this much quiet, never
        # while you are in the middle of a conversation, and never before the first reply of a session.
        "idle_seconds": max(0.0, float(m.get("idle_seconds", IDLE_SECONDS) if m.get("idle_seconds") is not None else IDLE_SECONDS)),
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
    fails = _BATCH_FAILS.get((str(store.db_path), session_id, start), 0)
    rows = conn.execute(
        "SELECT id, content FROM messages WHERE session_id = ? AND role = 'user' AND id > ? ORDER BY id LIMIT ?",
        (session_id, start, max(1, settings(config)["batch"] >> fails)),       # a failed batch is retried at half the size
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
        candidates, why = distill.extract_candidates_checked(llm, config, texts)      # may raise: queue untouched
        if candidates is None or why == "length":
            # cut off by the output cap or no parseable JSON: the facts are missing or incomplete, so the mark must NOT move -
            # the same messages are tried again in a smaller batch, and only skipped after MAX_BATCH_FAILS attempts
            key = (str(store.db_path), session_id, start)
            fails = _BATCH_FAILS.get(key, 0) + 1
            if fails < MAX_BATCH_FAILS:
                _BATCH_FAILS[key] = fails
                logger.info("Fact extraction %s; retrying with fewer messages", "was cut off" if why == "length" else "was unreadable")
                return 0
            _BATCH_FAILS.pop(key, None)
            logger.warning("Fact extraction returned no usable JSON array %d times; skipping these messages.", fails)
            candidates = None
        _BATCH_FAILS.pop((str(store.db_path), session_id, start), None)
        planned = distill.plan_facts(store, vectors, llm, config, candidates or []) if candidates else []
        saved = _commit(store, session_id, start, end_id, planned)
    else:
        _commit(store, session_id, start, end_id, [])
    for fact_id, p in saved:
        for old in p.supersedes:
            store.deactivate_fact(old)
        distill.index_fact(vectors, fact_id, p)
    return len(saved)


_BATCH_FAILS: dict[tuple[str, int, int], int] = {}      # (db, session, mark) -> failed extraction attempts


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
        if active_session is not None:
            from agent.memory import recall

            if recall.slide_if_needed(store, llm, config, active_session):
                return True                   # the chat's history window was cut back and summarised: off the reply path
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
                 pause_fn: Callable[[dict], str | None] | None = None, idle_delay: float | None = None,
                 poll: float = POLL_SECONDS, after_first_reply: bool = True) -> None:
        self.config, self.llm, self.db_path = config, llm, Path(db_path)
        self.session_id = session_id
        self.pause_fn = pause_fn or governor_pause
        if idle_delay is None:
            idle_delay = settings(config)["idle_seconds"] or IDLE_DELAY
        self.idle_delay, self.poll = idle_delay, poll
        self.after_first_reply = after_first_reply          # nothing at all before the first reply of the session
        self._turns = 0
        self._cancel = threading.Event()                    # set by busy(True): a running helper call stops at once
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
        """True while the chat model works on a turn (nothing starts then, and a step already running is cancelled: its helper
        call closes the connection so the reply does not wait behind it); False (and a wake-up) when it is idle again."""
        self._busy = value
        if value:
            self._turns += 1
            self._cancel.set()
        else:
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
            worked = False
            while not self._stop.is_set():
                if self._busy:
                    break                           # busy(False) wakes us again
                if self.after_first_reply and self._turns == 0:
                    break                           # the session has not had its first reply yet: busy(False) after it wakes us
                wait = self.idle_delay - (time.monotonic() - self._idle_since)
                if wait > 0:
                    self._stop.wait(min(wait, 0.25))
                    continue
                if self.pause_fn(self.config):
                    break
                cancel = threading.Event()
                self._cancel = cancel
                try:
                    self.in_flight = True
                    from agent.llm import GenerationAborted, cancel_scope

                    with cancel_scope(cancel):
                        did = step(self.db_path, self.llm, self.config, active_session=self.session_id)
                except GenerationAborted:
                    break                           # the user started a turn: the work stays queued for the next quiet moment
                except Exception as exc:  # noqa: BLE001 - remembering is a bonus; the queue keeps the work
                    self.errors += 1
                    self._failed(exc)
                    backoff = min(300.0, 30.0 * self.errors)
                    break
                finally:
                    self.in_flight = False
                if not did:
                    break
                worked = True
                self.errors = 0
                self.steps += 1
            if worked and not self._busy and not self._stop.is_set():
                self._reprime()

    def _reprime(self) -> None:
        """After background work on the chat model: put the conversation back into the server's cache (the next turn's prefix,
        or the last request again), so the next reply starts without re-reading the whole chat. Runs while the user is idle."""
        models = self.config.get("models") or {}
        chat = str(models.get("chat") or "")
        if not chat or not hasattr(self.llm, "reprime") or str(models.get("background") or "") != chat:
            return                                  # a separate helper model never touched the chat model's cache
        try:
            messages = None
            if self.session_id is not None:
                from agent.memory import recall

                with Store(self.db_path) as store:
                    messages = recall.prompt_prefix(store, self.config, self.session_id)
            self.llm.reprime(chat, messages)
        except Exception:  # noqa: BLE001 - only an optimisation
            pass


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
