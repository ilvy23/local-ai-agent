"""TurnWorker with fake turn functions: lifecycle, stop, pause, approvals, queue hand-over, activity registry."""

import threading
import time

import pytest

from agent.ui.activity import ActivityRegistry
from agent.ui.demo import FakeAgent
from agent.ui.events import EventBus
from agent.ui.queue import PromptQueue
from agent.ui.worker import ApprovalDecision, ApprovalRequest, TurnWorker


def nap(seconds):
    threading.Event().wait(seconds)     # conftest patches time.sleep globally to a no-op


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        nap(0.01)
    return False


class Gate:
    """A turn function that blocks until released (or stopped) and records what it ran."""

    def __init__(self):
        self.release = threading.Event()
        self.ran = []

    def __call__(self, prompt, emit, stop, approve):
        self.ran.append(prompt)
        emit({"type": "status", "state": "thinking", "label": "waiting for the model"})
        while not (self.release.is_set() or stop.is_set()):
            nap(0.005)
        if not stop.is_set():
            emit({"type": "done", "reply": prompt})


def test_runs_a_turn_publishes_events_and_returns_to_idle():
    bus, seen = EventBus(), []
    bus.subscribe(seen.append)
    w = TurnWorker(FakeAgent(0), bus=bus)
    assert w.state == "idle" and w.submit("hello") is True
    assert w.wait_idle(3) and w.state == "idle" and w.last_outcome == "done"
    kinds = [e.kind for e in seen]
    assert "token" in kinds and kinds[-1] == "done"
    assert all(e.turn == 1 for e in seen)


def test_submit_while_busy_is_refused_and_state_is_running():
    g = Gate()
    w = TurnWorker(g)
    w.submit("a")
    assert wait_for(lambda: w.state == "running")
    assert w.submit("b") is False and w.busy
    snap = w.snapshot()
    assert snap["busy"] and snap["now"] == "waiting for the model" and snap["prompt"] == "a"
    g.release.set()
    assert w.wait_idle(3)


def test_stop_interrupts_and_pauses_the_queue_with_a_notice():
    g, q, bus, seen = Gate(), PromptQueue(), EventBus(), []
    bus.subscribe(seen.append)
    w = TurnWorker(g, bus=bus, queue=q)
    w.submit("a")
    q.enqueue("b")
    assert wait_for(lambda: g.ran == ["a"]) and w.stop() is True
    assert w.wait_idle(3) and w.last_outcome == "interrupted"
    assert q.paused and len(q) == 1 and g.ran == ["a"]            # b did NOT start
    notices = [e.data["text"] for e in seen if e.data.get("notice")]
    assert any("queue paused (1 waiting)" in n for n in notices)
    assert w.stop() is False                                       # nothing left to stop


def test_queue_advances_automatically_after_a_clean_turn():
    g, q = Gate(), PromptQueue()
    g.release.set()
    w = TurnWorker(g, queue=q)
    q.enqueue("second"), q.enqueue("third")
    w.submit("first")
    assert wait_for(lambda: g.ran == ["first", "second", "third"] and not w.busy)
    assert len(q) == 0 and not q.paused


def test_paused_queue_is_not_advanced_and_resume_plus_advance_continues():
    g, q = Gate(), PromptQueue()
    g.release.set()
    w = TurnWorker(g, queue=q)
    q.enqueue("later")
    q.pause()
    w.submit("now")
    assert w.wait_idle(3) and g.ran == ["now"] and len(q) == 1
    q.resume()
    w.advance()
    assert wait_for(lambda: g.ran == ["now", "later"])


def test_queued_commands_go_to_dispatch_in_order():
    ran = []
    g, q = Gate(), PromptQueue()
    g.release.set()
    w = TurnWorker(g, queue=q, dispatch=lambda it: ran.append(it.text))
    q.enqueue("/clear", "command"), q.enqueue("after")
    w.submit("first")
    assert wait_for(lambda: g.ran == ["first", "after"])
    assert ran == ["/clear"]


def test_a_crashing_turn_reports_an_error_and_pauses_the_queue():
    def boom(prompt, emit, stop, approve):
        raise RuntimeError("model exploded")
    bus, seen, q = EventBus(), [], PromptQueue()
    bus.subscribe(seen.append)
    w = TurnWorker(boom, bus=bus, queue=q)
    q.enqueue("next")
    w.submit("x")
    assert w.wait_idle(3) and w.last_outcome == "error" and q.paused
    assert any(e.kind == "error" and "model exploded" in e.data["message"] for e in seen)


def test_pause_and_resume_are_cooperative():
    progress = []

    def turn(prompt, emit, stop, approve):
        for i in range(200):
            if not approve.wait_if_paused():
                return
            progress.append(i)
            nap(0.005)
    w = TurnWorker(turn)
    w.submit("x")
    assert wait_for(lambda: len(progress) > 3)
    assert w.pause() and w.state == "paused"
    nap(0.05)
    n = len(progress)
    nap(0.1)
    assert len(progress) <= n + 1                                   # frozen
    assert w.resume() and w.state == "running"
    assert wait_for(lambda: len(progress) > n + 5)
    w.stop()
    assert w.wait_idle(3)


def test_stop_wakes_a_paused_turn():
    def turn(prompt, emit, stop, approve):
        while approve.wait_if_paused():
            nap(0.005)
    w = TurnWorker(turn)
    w.submit("x")
    assert wait_for(lambda: w.state == "running")
    w.pause()
    w.stop()
    assert w.wait_idle(3) and w.last_outcome == "interrupted"


@pytest.mark.parametrize("answer,expected", [("y", True), ("yes", True), ("n", False), ("no", False)])
def test_approval_round_trip(answer, expected):
    got = []

    def turn(prompt, emit, stop, approve):
        got.append(approve({"tool_name": "run_command", "command": "ls", "risk_level": "CAUTION"}))
    bus, seen = EventBus(), []
    bus.subscribe(seen.append)
    w = TurnWorker(turn, bus=bus)
    w.submit("x")
    assert wait_for(lambda: w.state == "waiting_approval")
    assert w.pending_approval.tool == "run_command" and w.snapshot()["approval"] == "run_command"
    assert any(e.kind == "awaiting_approval" for e in seen)
    assert w.answer_approval(answer) is True
    assert w.wait_idle(3) and bool(got[0]) is expected


def test_approval_always_and_dangerous_requires_typed_yes():
    got = []

    def turn(prompt, emit, stop, approve):
        got.append(approve(ApprovalRequest(tool="run_command", command="rm -r build", risk="DANGEROUS")))
        got.append(approve({"tool_name": "edit", "risk": "CAUTION"}))
    w = TurnWorker(turn)
    w.submit("x")
    assert wait_for(lambda: w.pending_approval is not None)
    assert w.answer_approval("y") is False and w.answer_approval("a") is False and w.answer_approval("always") is False
    assert w.pending_approval is not None                           # still waiting - a stray y never runs a DANGEROUS call
    assert w.answer_approval("maybe") is False
    assert w.answer_approval("yes") is True
    assert wait_for(lambda: len(got) == 1 and w.pending_approval is not None)
    assert w.answer_approval("a") is True
    assert w.wait_idle(3)
    assert got[0].approve and not got[0].always and got[1].approve and got[1].always


def test_dangerous_can_always_be_denied_with_n():
    got = []
    w = TurnWorker(lambda p, e, s, a: got.append(a({"risk_level": "DANGEROUS", "command": "rm -rf /"})))
    w.submit("x")
    assert wait_for(lambda: w.pending_approval is not None)
    assert w.answer_approval("n") and w.wait_idle(3) and not got[0]


def test_approval_times_out_as_denied():
    got = []
    w = TurnWorker(lambda p, e, s, a: got.append(a({"command": "x"})), approval_timeout=0.15)
    w.submit("x")
    assert w.wait_idle(3) and got[0].approve is False and got[0].reason == "timed out"


def test_stop_while_waiting_for_approval_denies_it():
    got = []
    w = TurnWorker(lambda p, e, s, a: got.append(a({"command": "x"})))
    w.submit("x")
    assert wait_for(lambda: w.state == "waiting_approval")
    w.stop()
    assert w.wait_idle(3) and got[0].approve is False and got[0].reason == "stopped"


def test_chat_approve_adapter_returns_exact_yes_or_false():
    class Req:  # chat_adapter.ApprovalRequest lookalike
        name, risk, description, arguments, diff, from_web = "run_command", "DANGEROUS", "rm -r x", {}, "", True
    got = []

    def turn(prompt, emit, stop, approve):
        fn = approve.as_chat_approve()
        got.append(fn(Req()))
    w = TurnWorker(turn)
    w.submit("x")
    assert wait_for(lambda: w.pending_approval is not None)
    p = w.pending_approval
    assert p.tool == "run_command" and p.risk == "DANGEROUS" and p.command == "rm -r x" and "web page" in p.reason
    w.answer_approval("yes")
    assert w.wait_idle(3) and got == ["yes"]


def test_registers_in_the_activity_registry_and_updates_now_line():
    reg = ActivityRegistry()
    g = Gate()
    w = TurnWorker(g, activities=reg)
    w.submit("write the thing")
    assert wait_for(lambda: reg.list(active_only=True))
    a = reg.list()[0]
    assert a.kind == "chat" and a.stop is not None and "write the thing" in a.label
    assert wait_for(lambda: a.now == "waiting for the model")
    w.emit({"type": "tool_call", "name": "read_file", "arguments": {"path": "x.py"}})
    assert a.now == "reading x.py"
    g.release.set()
    assert w.wait_idle(3) and wait_for(lambda: reg.get(a.id).state == "done")


def test_stopped_turn_is_marked_stopped_in_the_registry():
    reg = ActivityRegistry()
    w = TurnWorker(Gate(), activities=reg)
    w.submit("x")
    wait_for(lambda: reg.list(True))
    w.stop()
    w.wait_idle(3)
    assert wait_for(lambda: reg.list()[0].state == "stopped")


def test_snapshot_tracks_context_and_speed():
    g = Gate()
    w = TurnWorker(g)
    w.submit("x")
    wait_for(lambda: w.busy)
    w.emit({"type": "context", "used": 100, "limit": 1000, "percent": 10, "tokens_per_s": 40.0})
    w.emit({"type": "token", "text": "abcdefgh"})
    s = w.snapshot()
    assert (s["ctx_used"], s["ctx_limit"], s["tok_s"]) == (100, 1000, 40.0) and s["tokens"] >= 2
    g.release.set()
    w.wait_idle(3)
    assert w.snapshot()["busy"] is False and w.snapshot()["now"] == ""


def test_fake_agent_honours_stop_mid_stream():
    bus, seen = EventBus(), []
    bus.subscribe(seen.append)
    w = TurnWorker(FakeAgent(1.0), bus=bus)
    w.submit("short")
    assert wait_for(lambda: any(e.kind == "token" for e in seen))
    w.stop()
    assert w.wait_idle(3) and w.last_outcome == "interrupted"
    assert not any(e.kind == "done" for e in seen)


def test_slow_model_hint_after_ten_seconds_and_live_rate(monkeypatch):
    from agent.ui.dock import activity_row

    g = Gate()
    w = TurnWorker(g)
    w.submit("hi")
    assert wait_for(lambda: w.state == "running")
    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "time", lambda: clock["t"])
    w._started = clock["t"]
    w.emit({"type": "status", "state": "thinking", "label": "waiting for model", "detail": "qwen3.5:4b",
            "since": clock["t"], "prompt_tokens": 2100})
    clock["t"] += 4
    snap = w.snapshot()
    assert snap["waiting_s"] == 4 and "reading the prompt" not in activity_row(snap, 120, plain=True)
    clock["t"] += 96                                              # 1m 40s without a token
    snap = w.snapshot()
    row = activity_row(snap, 120, plain=True)
    assert "reading the prompt (~2.1k tokens)... on CPU this can take a few minutes" in row and "\n" not in row
    assert "1m 40s" in activity_row(snap, 60, plain=True)         # the clock survives narrow widths
    for _ in range(10):                                           # streaming starts: hint gone, rate shown
        w.emit({"type": "token", "text": "word " * 4})
    clock["t"] += 5
    snap = w.snapshot()
    assert snap["waiting_s"] == 0 and snap["tok_s"] and "reading the prompt" not in activity_row(snap, 120, plain=True)
    assert "tok/s" in activity_row(snap, 120, plain=True)
    g.release.set()
    w.wait_idle(3)
