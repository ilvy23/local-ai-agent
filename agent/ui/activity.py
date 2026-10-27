"""Registry of everything that is running (chat turn, code run, research, workers ...), for /tasks and the dock."""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.ui.events import EventBus, UiEvent

KINDS = ("chat", "code", "research", "worker", "whatsapp", "other")
STATES = ("running", "paused", "waiting", "done", "failed", "stopped")
ACTIVE_STATES = frozenset({"running", "paused", "waiting"})


@dataclass
class Activity:
    id: str
    kind: str = "other"
    label: str = ""
    state: str = "running"
    now: str = ""
    started: float = field(default_factory=time.time)
    ended: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    stop: Callable[[], Any] | None = None
    pause: Callable[[], Any] | None = None
    resume: Callable[[], Any] | None = None

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES


class ActivityRegistry:
    def __init__(self, bus: EventBus | None = None) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, Activity] = {}
        self._seq = itertools.count(1)
        self.bus = bus

    def _publish(self, a: Activity, msg: str) -> None:
        if self.bus is not None:
            self.bus.emit(UiEvent("log", {"activity": a.id, "state": a.state, "message": msg}, source=a.kind))

    def register(self, kind: str = "other", label: str = "", *, id: str | None = None, state: str = "running",
                 now: str = "", detail: dict[str, Any] | None = None, stop: Callable[[], Any] | None = None,
                 pause: Callable[[], Any] | None = None, resume: Callable[[], Any] | None = None) -> Activity:
        with self._lock:
            aid = id or f"{kind}-{next(self._seq)}"
            a = Activity(id=aid, kind=kind, label=label, state=state, now=now, detail=dict(detail or {}),
                         stop=stop, pause=pause, resume=resume)
            if state not in ACTIVE_STATES:
                a.ended = a.started
            self._items[aid] = a
        self._publish(a, f"{label or aid}: {state}")
        return a

    def update(self, id: str, **fields: Any) -> Activity | None:
        with self._lock:
            a = self._items.get(id)
            if a is None:
                return None
            old = a.state
            for k, v in fields.items():
                if k == "detail" and isinstance(v, dict):
                    a.detail.update(v)
                elif hasattr(a, k) and k not in ("id", "active"):
                    setattr(a, k, v)
            if a.state not in ACTIVE_STATES and a.ended is None:
                a.ended = time.time()
            elif a.state in ACTIVE_STATES:
                a.ended = None
            changed = a.state != old
        if changed:
            self._publish(a, f"{a.label or a.id}: {a.state}")
        return a

    def finish(self, id: str, state: str = "done") -> Activity | None:
        return self.update(id, state=state, now="")

    def get(self, id: str) -> Activity | None:
        with self._lock:
            return self._items.get(id)

    def list(self, active_only: bool = False) -> list[Activity]:
        with self._lock:
            items = list(self._items.values())
        return [a for a in items if a.active] if active_only else items

    def elapsed(self, a: Activity, now: float | None = None) -> float:
        end = a.ended if a.ended is not None else (now if now is not None else time.time())
        return max(0.0, end - a.started)

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            {"id": a.id, "kind": a.kind, "label": a.label, "state": a.state, "now": a.now,
             "started": a.started, "ended": a.ended, "elapsed": self.elapsed(a, now),
             "detail": dict(a.detail), "can_stop": a.stop is not None,
             "can_pause": a.pause is not None, "can_resume": a.resume is not None}
            for a in self.list()
        ]
