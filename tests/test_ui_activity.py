import threading

from agent.ui.activity import ActivityRegistry
from agent.ui.events import EventBus


def test_lifecycle_and_snapshot():
    reg = ActivityRegistry()
    a = reg.register("code", "fix bug", now="reading foo.py", stop=lambda: None)
    assert a.state == "running" and reg.get(a.id) is a
    reg.update(a.id, now="editing", detail={"step": 3})
    reg.update(a.id, detail={"max": 20})
    assert reg.get(a.id).now == "editing" and reg.get(a.id).detail == {"step": 3, "max": 20}
    assert reg.list(active_only=True) == [a]
    reg.finish(a.id, "failed")
    assert a.state == "failed" and a.ended is not None and reg.list(active_only=True) == []
    snap = reg.snapshot()
    assert snap[0]["state"] == "failed" and snap[0]["can_stop"] and not snap[0]["can_pause"]
    assert reg.update("nope", now="x") is None


def test_elapsed_and_ids():
    reg = ActivityRegistry()
    a = reg.register("chat", "t")
    b = reg.register("chat", "u")
    assert a.id != b.id
    assert reg.elapsed(a, now=a.started + 5) == 5
    reg.finish(a.id)
    assert reg.elapsed(a, now=a.started + 999) < 5


def test_pause_resume_state_and_bus():
    bus = EventBus()
    reg = ActivityRegistry(bus)
    a = reg.register("worker", "gov")
    reg.update(a.id, state="paused")
    reg.update(a.id, now="still paused")   # no state change -> no event
    reg.update(a.id, state="running")
    evs = bus.recent("log")
    assert len(evs) == 3 and evs[1].data["state"] == "paused" and a.ended is None


def test_threads():
    reg = ActivityRegistry()

    def w():
        for _ in range(100):
            x = reg.register("other", "x")
            reg.update(x.id, now="n")
            reg.finish(x.id)

    ts = [threading.Thread(target=w) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(reg.list()) == 800 and len({a.id for a in reg.list()}) == 800
