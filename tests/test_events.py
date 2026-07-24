from agent.events import ActivityEvent, EventBus


def test_emit_delivers_to_subscriber():
    bus = EventBus()
    seen = []
    bus.subscribe(seen.append)
    bus.emit(ActivityEvent(kind="attempt", message="hi"))
    assert seen[0].kind == "attempt"
    assert seen[0].message == "hi"


def test_emit_kind_builds_event():
    bus = EventBus()
    seen = []
    bus.subscribe(seen.append)
    bus.emit_kind("check", "ast", name="ast", ok=True)
    assert seen[0].kind == "check"
    assert seen[0].data == {"name": "ast", "ok": True}


def test_no_subscribers_is_noop():
    EventBus().emit_kind("x")


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    seen = []
    off = bus.subscribe(seen.append)
    off()
    bus.emit_kind("x")
    assert seen == []


def test_unsubscribe_twice_is_safe():
    bus = EventBus()
    off = bus.subscribe(lambda e: None)
    off()
    off()


def test_subscriber_exception_is_isolated():
    bus = EventBus()
    seen = []

    def boom(_event):
        raise RuntimeError("nope")

    bus.subscribe(boom)
    bus.subscribe(seen.append)
    bus.emit_kind("x")
    assert len(seen) == 1
