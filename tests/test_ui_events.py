import threading

from agent.ui.events import CODE_KINDS, KINDS, WEB_KINDS, EventBus, UiEvent, from_record


def test_kinds():
    for k in ("token", "tool_call", "awaiting_approval", "file_ready", "mood", "stage_start", "autopilot_decision", "step"):
        assert k in KINDS
    assert KINDS == WEB_KINDS | CODE_KINDS and len(WEB_KINDS) == 20 and len(CODE_KINDS) == 17


def test_subscribe_unsubscribe_and_recent():
    bus, got = EventBus(maxlen=3), []
    off = bus.subscribe(got.append)
    for i in range(5):
        bus.emit(UiEvent("token" if i % 2 else "status", {"i": i}))
    assert [e.data["i"] for e in got] == [0, 1, 2, 3, 4]
    assert [e.data["i"] for e in bus.recent()] == [2, 3, 4]
    assert [e.data["i"] for e in bus.recent("token")] == [3]
    assert len(bus.recent(n=1)) == 1
    off(); off()
    bus.emit(UiEvent("done"))
    assert len(got) == 5


def test_failing_subscriber_isolated():
    bus, got = EventBus(), []

    def boom(e):
        raise RuntimeError("x")

    bus.subscribe(boom)
    bus.subscribe(got.append)
    bus.emit(UiEvent("error"))
    assert len(got) == 1


def test_thread_safety():
    bus, count = EventBus(maxlen=10000), []
    bus.subscribe(lambda e: count.append(1))

    def work():
        for _ in range(500):
            bus.emit(UiEvent("token"))

    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(count) == 4000 and len(bus.recent(n=10000)) == 4000


def test_from_record():
    e = from_record({"ts": 12.5, "session": "s1", "kind": "compact", "saved": 100, "turn": 3}, "code")
    assert (e.kind, e.ts, e.source, e.turn) == ("compact", 12.5, "code", 3)
    assert e.data["saved"] == 100 and e.data["session"] == "s1" and "kind" not in e.data
    e2 = from_record({"kind": "step"}, "code")
    assert e2.turn is None and e2.ts > 0
