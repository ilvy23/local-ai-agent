"""UI event vocabulary and a small thread-safe bus.

The vocabulary is the web contract (docs/design/02) plus the coding engine's own event kinds, so the terminal and
the web UI can share one stream."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

WEB_KINDS = frozenset(
    {
        "token", "status", "thinking", "tool_call", "tool_result", "sources", "verdict", "block", "plan",
        "verify", "stuck", "context", "changes", "verify_result", "awaiting_approval", "error", "done",
        "log", "file_ready", "mood",
    }
)
CODE_KINDS = frozenset(
    {
        "stage_start", "stage_done", "stage_blocked", "check", "budget_set", "budget_extend", "compact",
        "mask", "loop", "loop_abort", "edit", "tool_start", "steer", "autopilot_start", "autopilot_end",
        "autopilot_decision", "step",
    }
)
KINDS = WEB_KINDS | CODE_KINDS


@dataclass
class UiEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    source: str = "chat"
    turn: int | None = None


Subscriber = Callable[[UiEvent], None]


class EventBus:
    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.RLock()
        self._subs: list[Subscriber] = []
        self._ring: deque[UiEvent] = deque(maxlen=maxlen)

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subs.remove(fn)
                except ValueError:
                    pass

        return unsubscribe

    def emit(self, event: UiEvent) -> None:
        with self._lock:
            self._ring.append(event)
            subs = list(self._subs)
        for fn in subs:  # outside the lock: a slow/reentrant subscriber must not block others
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a broken subscriber must never break the emitter
                pass

    def recent(self, kind: str | None = None, n: int = 50) -> list[UiEvent]:
        with self._lock:
            items = [e for e in self._ring if kind is None or e.kind == kind]
        return items[-n:] if n > 0 else []


def from_record(rec: dict[str, Any], source: str = "code") -> UiEvent:
    """Convert a codeagent EventLog record ({ts, session, kind, ...}) into a UiEvent."""
    data = {k: v for k, v in rec.items() if k not in ("ts", "kind")}
    turn = data.get("turn")
    ts = rec.get("ts")
    return UiEvent(
        kind=str(rec.get("kind", "log")),
        data=data,
        ts=float(ts) if isinstance(ts, (int, float)) else time.time(),
        source=source,
        turn=turn if isinstance(turn, int) else None,
    )
