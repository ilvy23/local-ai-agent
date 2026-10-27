"""Facts are saved while chatting; quitting never waits for a model; nothing is lost or doubled (agent/memory/queue.py)."""

from __future__ import annotations

import io
import math
import re
import threading
import time

import pytest
from rich.console import Console

from agent.llm import EmbeddingUnavailable
from agent.memory import queue
from agent.memory.recall import _semantic_hits
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

CONFIG = {
    "models": {"background": "bg", "embed": "emb", "chat": "chat"},
    "memory": {"recall_k": 5, "recall_min_similarity": 0.3, "distill_every_turns": 1, "exit_wait_seconds": 0.2},
    "persona": {"name": "Agent", "style": "x"},
}
_STOP = {"a", "is", "my", "what", "has", "in", "i", "am", "and", "live", "the", "called", "name"}


def _embed_text(text: str) -> list[float]:
    """Bag of (non-stop) words hashed into 64 slots, unit length: related sentences share direction."""
    v = [0.0] * 64
    for w in re.findall(r"[a-z]+", text.lower()):
        if w not in _STOP:
            v[sum(map(ord, w)) % 64] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeLLM:
    def __init__(self, delay: float = 0.0, fail: int = 0, embeds: bool = True):
        self.delay, self.fail, self.embeds = delay, fail, embeds
        self.chat_calls: list[str] = []
        self.lock = threading.Lock()

    def chat(self, messages, model, stream=True, **kw):
        text = messages[-1]["content"]
        with self.lock:
            self.chat_calls.append(text)
            if self.fail:
                self.fail -= 1
                raise ConnectionError("model went away")
        if self.delay:
            threading.Event().wait(self.delay)      # conftest makes time.sleep a no-op
        facts = []
        if "Mira" in text:
            facts.append({"text": "has a cat called Mira", "confidence": "verified"})
        if "Lisbon" in text:
            facts.append({"text": "lives in Lisbon", "confidence": "verified"})
        if "junk" in text:
            facts.append({"text": "/home/user/notes.txt", "confidence": "verified"})
        import json
        yield json.dumps(facts)

    def embed(self, texts, model, num_gpu=0):
        if not self.embeds:
            raise EmbeddingUnavailable(model)
        return [_embed_text(t) for t in texts]


def _chat(store, *lines):
    sid = store.create_session()
    for line in lines:
        store.add_message(sid, "user", line)
        store.add_message(sid, "assistant", "ok")
    return sid


def _facts(db):
    with Store(db) as s:
        return [f["content"] for f in s.get_active_facts()]


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _dist(db, llm, sid=None, **kw):
    kw.setdefault("pause_fn", lambda c: None)
    kw.setdefault("idle_delay", 0.0)
    return queue.BackgroundDistiller(CONFIG, llm, db, session_id=sid, **kw)


# ---------------------------------------------------------------- saved during the chat

def test_facts_are_persisted_while_the_chat_is_still_open(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira and I live in Lisbon")
    d = _dist(db, FakeLLM(), sid).start()
    try:
        assert _wait(lambda: len(_facts(db)) == 2)          # before any exit
    finally:
        d.stop(1)
    assert sorted(_facts(db)) == ["has a cat called Mira", "lives in Lisbon"]
    assert queue.pending_count(store.conn) == 0


def test_nothing_starts_while_the_chat_model_is_busy(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    llm = FakeLLM()
    d = _dist(db, llm, sid)
    d.busy(True)
    d.start()
    time.sleep(0.4)
    assert llm.chat_calls == []                              # busy: the model is left alone
    d.busy(False)
    assert _wait(lambda: _facts(db) == ["has a cat called Mira"])
    d.stop(1)


def test_governor_pause_holds_work_back(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    llm = FakeLLM()
    d = _dist(db, llm, sid, pause_fn=lambda c: "game running")
    d.start()
    time.sleep(0.4)
    d.stop(0.5)
    assert llm.chat_calls == [] and queue.pending_count(store.conn) == 1


def test_every_n_turns_for_the_active_session(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    cfg = {**CONFIG, "memory": {**CONFIG["memory"], "distill_every_turns": 2}}
    sid = _chat(store, "My cat is called Mira")
    llm = FakeLLM()
    assert queue.step(db, llm, cfg, active_session=sid) is False      # one message: below the threshold
    store.add_message(sid, "user", "I live in Lisbon")
    assert queue.step(db, llm, cfg, active_session=sid) is True
    assert sorted(_facts(db)) == ["has a cat called Mira", "lives in Lisbon"]


# ---------------------------------------------------------------- exit

def test_exit_with_pending_messages_returns_immediately_and_next_start_drains(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    slow = FakeLLM(delay=5.0)
    d = _dist(db, slow, sid, idle_delay=60.0).start()        # never gets going: the chat just ended
    out = io.StringIO()
    t0 = time.monotonic()
    left = queue.finish(d, store, sid, [], CONFIG, Console(file=out, force_terminal=False, width=100))
    assert time.monotonic() - t0 < 0.5
    assert left == 1 and "1 message will be remembered in the background" in out.getvalue()
    assert _facts(db) == []

    nxt = _dist(db, FakeLLM()).start()                        # the NEXT start: quiet catch-up
    try:
        assert _wait(lambda: _facts(db) == ["has a cat called Mira"])
    finally:
        nxt.stop(1)
    assert queue.pending_count(store.conn) == 0


def test_exit_while_a_step_is_running_waits_at_most_the_cap(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    slow = FakeLLM(delay=3.0)
    d = _dist(db, slow, sid).start()
    assert _wait(lambda: d.in_flight)
    t0 = time.monotonic()
    queue.finish(d, store, sid, [], CONFIG, None)             # exit_wait_seconds = 0.2 in CONFIG
    assert time.monotonic() - t0 < 0.5
    assert queue.pending_count(store.conn) == 1               # still queued: the step never committed


def test_exit_queues_tool_exchanges_durably(tmp_path):
    store = Store(tmp_path / "a.db")
    sid = _chat(store)
    queue.finish(None, store, sid, [("q", "a", {"label": "verified", "sources": []})], CONFIG, None)
    assert queue.pending_count(store.conn) == 1
    assert store.conn.execute("SELECT question FROM learn_queue").fetchone()[0] == "q"


# ---------------------------------------------------------------- crash safety, idempotence

def test_model_failure_loses_nothing_and_retry_saves_once(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    _chat(store, "My cat is called Mira")
    llm = FakeLLM(fail=1)
    with pytest.raises(ConnectionError):
        queue.step(db, llm, CONFIG)
    assert queue.pending_count(store.conn) == 1 and _facts(db) == []
    assert queue.step(db, llm, CONFIG) is True
    assert queue.step(db, llm, CONFIG) is False               # nothing left; running again adds nothing
    assert _facts(db) == ["has a cat called Mira"]


def test_a_kill_before_the_commit_redoes_the_batch_without_duplicates(tmp_path, monkeypatch):
    db = tmp_path / "a.db"
    store = Store(db)
    _chat(store, "My cat is called Mira")
    real = queue._commit

    def boom(*a, **k):
        raise KeyboardInterrupt()                              # the process dies between extraction and commit
    monkeypatch.setattr(queue, "_commit", boom)
    with pytest.raises(KeyboardInterrupt):
        queue.step(db, FakeLLM(), CONFIG)
    monkeypatch.setattr(queue, "_commit", real)
    assert _facts(db) == [] and queue.pending_count(store.conn) == 1
    queue.step(db, FakeLLM(), CONFIG)
    queue.step(db, FakeLLM(), CONFIG)
    assert _facts(db) == ["has a cat called Mira"]


def test_two_workers_on_the_same_batch_save_it_once(tmp_path, monkeypatch):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    other = Store(db)
    llm = FakeLLM()
    real = queue.distill.extract_candidates

    inner = []

    def racing(*a, **k):
        res = real(*a, **k)
        if not inner:
            inner.append(1)
            queue.distill_batch(other, VectorIndex(other), llm, CONFIG, sid)   # the other process finishes first
        return res
    monkeypatch.setattr(queue.distill, "extract_candidates", racing)
    queue.step(db, llm, CONFIG)
    assert _facts(db) == ["has a cat called Mira"]


def test_junk_is_still_rejected_but_the_batch_is_consumed(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    _chat(store, "junk Mira")
    queue.step(db, FakeLLM(), CONFIG)
    assert _facts(db) == ["has a cat called Mira"]            # the path-like "fact" never gets in
    assert queue.pending_count(store.conn) == 0


def test_unparseable_reply_does_not_loop_forever(tmp_path):
    class Chatty(FakeLLM):
        def chat(self, messages, model, stream=True, **kw):
            yield "I cannot help with that."
    db = tmp_path / "a.db"
    store = Store(db)
    _chat(store, "hello")
    queue.step(db, Chatty(), CONFIG)
    assert queue.pending_count(store.conn) == 0 and _facts(db) == []


def test_existing_sessions_are_not_reprocessed_after_the_migration(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    store.conn.execute("DROP TABLE distill_progress")
    store.conn.execute("DROP TABLE learn_queue")
    store.conn.execute("PRAGMA user_version = 15")
    store.conn.commit()
    store.close()
    again = Store(db)                                          # v16 migration marks old sessions as done
    assert queue.pending_count(again.conn) == 0
    again.add_message(sid, "user", "I live in Lisbon")
    assert queue.pending_count(again.conn) == 1


# ---------------------------------------------------------------- learned facts queue

def test_learned_fact_queue_is_processed_and_cleared(tmp_path, monkeypatch):
    db = tmp_path / "a.db"
    store = Store(db)
    queue.enqueue_exchanges(store, None, [("height of the Eiffel Tower?", "330 m", None)])

    class L(FakeLLM):
        def chat(self, messages, model, stream=True, **kw):
            yield "The Eiffel Tower is 330 m tall."
    assert queue.step(db, L(), CONFIG) is True
    assert queue.pending_count(store.conn) == 0
    assert _facts(db) == ["The Eiffel Tower is 330 m tall."]
    queue.enqueue_exchanges(store, None, [("again", "330 m", None)])
    queue.step(db, L(), CONFIG)
    assert _facts(db) == ["The Eiffel Tower is 330 m tall."]  # a replayed exchange does not double it


def test_failing_learned_exchange_is_retried_then_dropped(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    queue.enqueue_exchanges(store, None, [("q", "a", None)])
    for _ in range(queue.MAX_LEARN_ATTEMPTS):
        with pytest.raises(ConnectionError):
            queue.step(db, FakeLLM(fail=1), CONFIG)
    assert queue.pending_count(store.conn) == 0


# ---------------------------------------------------------------- recall in a NEW session

def _new_session_recall(db, llm, question):
    with Store(db) as store:
        return _semantic_hits(VectorIndex(store), llm, CONFIG, question, store)


def test_recall_by_meaning_in_a_new_session(tmp_path):
    db = tmp_path / "a.db"
    with Store(db) as store:
        _chat(store, "My cat is called Mira and I live in Lisbon")
    queue.drain(db, FakeLLM(), CONFIG, pause_fn=lambda c: None)
    hits = _new_session_recall(db, FakeLLM(), "What is my cat called?")
    assert "has a cat called Mira" in hits
    assert "lives in Lisbon" not in hits                      # unrelated facts are not dragged in


def test_recall_falls_back_to_keywords_without_an_embedder(tmp_path):
    db = tmp_path / "a.db"
    with Store(db) as store:
        _chat(store, "My cat is called Mira and I live in Lisbon")
    noembed = FakeLLM(embeds=False)
    queue.drain(db, noembed, CONFIG, pause_fn=lambda c: None)
    assert sorted(_facts(db)) == ["has a cat called Mira", "lives in Lisbon"]    # saved even with no embedder
    assert "lives in Lisbon" in _new_session_recall(db, noembed, "Where do I live?")
    # an embedder arrives later: the repair pass indexes the facts and meaning-recall works
    from agent.memory.distill import repair_unembedded

    with Store(db) as store:
        repair_unembedded(store, VectorIndex(store), FakeLLM(), CONFIG)
    assert "has a cat called Mira" in _new_session_recall(db, FakeLLM(), "What is my cat called?")


def test_a_dimension_change_never_mixes_vectors(tmp_path):
    db = tmp_path / "a.db"
    with Store(db) as store:
        _chat(store, "My cat is called Mira")
    queue.drain(db, FakeLLM(), CONFIG, pause_fn=lambda c: None)

    class Wide(FakeLLM):
        def embed(self, texts, model, num_gpu=0):
            return [[0.1] * 10 for _ in texts]
    with Store(db) as store:
        sid = _chat(store, "I live in Lisbon")
    queue.drain(db, Wide(), CONFIG, pause_fn=lambda c: None)
    assert "lives in Lisbon" in _facts(db)                    # saved ...
    with Store(db) as store:
        VectorIndex(store)
        dims = {len(r[0]) // 4 for r in store.conn.execute("SELECT embedding FROM memory_vectors")}
    assert dims == {64}                                        # ... but its 10-dim vector never entered the 64-dim index


# ---------------------------------------------------------------- CLI

def test_memory_process_and_check_commands(tmp_path, monkeypatch):
    import copy

    from typer.testing import CliRunner

    import agent.config as cfgmod
    import agent.main as main
    from agent.config import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data"]["db_path"] = str(tmp_path / "a.db")
    llm = FakeLLM()
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(main, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(main, "_open_llm", lambda: llm)
    monkeypatch.setattr(queue, "governor_pause", lambda c: None)
    with Store(tmp_path / "a.db") as store:
        _chat(store, "My cat is called Mira")
    r = CliRunner().invoke(main.app, ["memory", "process"])
    assert r.exit_code == 0 and "Everything is remembered" in r.output, r.output
    assert _facts(tmp_path / "a.db") == ["has a cat called Mira"]
    assert "Nothing is waiting" in CliRunner().invoke(main.app, ["memory", "process"]).output
    chk = CliRunner().invoke(main.app, ["memory", "check"])
    assert chk.exit_code == 0 and "recalled by meaning" in chk.output, chk.output
    assert "Pixel" not in _facts(tmp_path / "a.db").__str__()          # the check never touches your own data
