"""Tiny event bus so `coding/` never prints and `cli/` never calls Ollama.

Core code emits ActivityEvents; the UI (or a test) subscribes. This keeps the
display swappable and preserves the "tests run without Ollama" property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class Events:
    """A synchronous fan-out. A broken subscriber can't break the coding loop."""

    def __init__(self) -> None:
        self._subs: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[Event], None]:
        self._subs.append(fn)
        return fn

    def emit(self, kind: str, **data: Any) -> Event:
        ev = Event(kind, data)
        for fn in self._subs:
            try:
                fn(ev)
            except Exception:  # noqa: BLE001 - the UI must never crash the core
                pass
        return ev


NULL = Events()  # default no-op sink for headless/test use
