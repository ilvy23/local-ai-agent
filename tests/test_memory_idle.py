"""Phase 7: the prompt prefix stays identical between turns (facts learned later and working notes go into the note at the end),
the history window only moves in chunks, and the memory worker works only when the user is idle - never before the first reply,
stopped the moment a turn starts - and afterwards puts the conversation back into the server's cache."""

from __future__ import annotations

import json
import threading
import time

from agent.memory import queue, recall
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex
from tests.test_memory_queue import CONFIG, FakeLLM, _chat, _facts, _wait

RCONFIG = {
    "persona": {"style": "You are Agent."},
    "models": {"embed": "nomic-embed-text", "background": "bg", "chat": "bg"},
    "memory": {"recall_k": 6, "context_char_budget": 100_000},
}


class NoRecallLLM:
    def __init__(self):
        self.calls = 0

    def embed(self, texts, model, num_gpu=0):
        raise RuntimeError("no embedder in this test")

    def chat(self, messages, model, stream=True, **kw):
        self.calls += 1
        yield "summary of the earlier part"


def _make(tmp_path):
    store = Store(tmp_path / "r.db")
    return store, VectorIndex(store)


def test_a_fact_saved_in_the_background_leaves_the_system_prompt_byte_identical(tmp_path):
    store, vectors = _make(tmp_path)
    store.add_fact("likes green tea", source_session_id=None)
    sid = store.create_session()
    store.add_message(sid, "user", "hello")
    first = recall.build_context(store, vectors, NoRecallLLM(), RCONFIG, sid, "one")
    store.add_fact("lives in Porto", source_session_id=None)            # the memory worker saved a new fact meanwhile
    store.set_working_notes(sid, "remember the deadline") if hasattr(store, "set_working_notes") else None
    second = recall.build_context(store, vectors, NoRecallLLM(), RCONFIG, sid, "two")
    assert second[0] == first[0]                                         # same system prompt: no re-read of the conversation
    assert "lives in Porto" not in second[0]["content"]
    assert any("lives in Porto" in m["content"] for m in second[1:-1])   # ... the new fact is in the note at the end
    assert second[-1] == {"role": "user", "content": "two"}


def test_the_history_window_does_not_move_every_turn(tmp_path):
    store, vectors = _make(tmp_path)
    sid = store.create_session()
    for i in range(8):
        store.add_message(sid, "user", f"msg-{i} " + "x" * 200)
    cfg = {**RCONFIG, "memory": {**RCONFIG["memory"], "context_char_budget": 1900}}   # a bit over budget, under 1.25x
    llm = NoRecallLLM()
    a = recall.build_context(store, vectors, llm, cfg, sid, "q1")
    store.add_message(sid, "user", "msg-8 " + "x" * 10)
    b = recall.build_context(store, vectors, llm, cfg, sid, "q2")
    assert llm.calls == 0                                                # no summary call on the reply path below 1.25x
    assert a[1]["content"] == b[1]["content"]                            # the window starts at the same message


def test_the_worker_slides_the_window_while_idle(tmp_path):
    store, vectors = _make(tmp_path)
    sid = store.create_session()
    for i in range(8):
        store.add_message(sid, "user", f"msg-{i} " + "x" * 200)
    cfg = {**RCONFIG, "memory": {**RCONFIG["memory"], "context_char_budget": 1900}}
    llm = NoRecallLLM()
    recall.build_context(store, vectors, llm, cfg, sid, "q1")
    assert recall.slide_if_needed(store, llm, cfg, sid) is True and llm.calls == 1
    summary, through = store.get_session_compaction(sid)
    assert summary == "summary of the earlier part" and through is not None
    after = recall.build_context(store, vectors, llm, cfg, sid, "q2")
    assert "summary of the earlier part" in after[0]["content"] and recall.slide_if_needed(store, llm, cfg, sid) is False


def test_nothing_runs_before_the_first_reply_of_a_session(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    llm = FakeLLM()
    d = queue.BackgroundDistiller(CONFIG, llm, db, session_id=sid, pause_fn=lambda c: None, idle_delay=0.0).start()
    try:
        time.sleep(0.4)
        assert llm.chat_calls == []                                      # the catch-up waits for the first reply
        d.busy(True)
        d.busy(False)                                                    # the first turn has been answered
        assert _wait(lambda: _facts(db) == ["has a cat called Mira"])
    finally:
        d.stop(1)


def test_the_idle_time_comes_from_the_config(tmp_path):
    d = queue.BackgroundDistiller({**CONFIG, "memory": {"idle_seconds": 45}}, FakeLLM(), tmp_path / "a.db")
    assert d.idle_delay == 45
    assert queue.BackgroundDistiller(CONFIG, FakeLLM(), tmp_path / "a.db").idle_delay == queue.IDLE_SECONDS


def test_a_turn_stops_a_running_step_and_the_work_stays_queued(tmp_path):
    from agent.llm import GenerationAborted

    started = threading.Event()

    class Slow(FakeLLM):
        def chat(self, messages, model, stream=True, cancel=None, **kw):
            started.set()
            assert cancel is not None and cancel.wait(5)                 # waits until the turn starts
            raise GenerationAborted("stopped")
            yield ""                                                     # pragma: no cover

    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    d = queue.BackgroundDistiller(CONFIG, Slow(), db, session_id=sid, pause_fn=lambda c: None, idle_delay=0.0,
                                  after_first_reply=False).start()
    try:
        assert started.wait(5)
        d.busy(True)                                                     # the user sends a message
        assert _wait(lambda: not d.in_flight)
        assert d.errors == 0 and queue.pending_count(store.conn) == 1    # not an error, nothing lost
    finally:
        d.stop(1)


def test_after_work_on_the_chat_model_the_cache_is_reprimed(tmp_path):
    class Primed(FakeLLM):
        def __init__(self):
            super().__init__()
            self.primed = []

        def reprime(self, model, messages=None):
            self.primed.append(model)
            return True

    db = tmp_path / "a.db"
    store = Store(db)
    sid = _chat(store, "My cat is called Mira")
    llm = Primed()
    cfg = {**CONFIG, "models": {**CONFIG["models"], "chat": CONFIG["models"]["background"]}}
    d = queue.BackgroundDistiller(cfg, llm, db, session_id=sid, pause_fn=lambda c: None, idle_delay=0.0,
                                  after_first_reply=False).start()
    try:
        assert _wait(lambda: llm.primed == [cfg["models"]["chat"]])
    finally:
        d.stop(1)
