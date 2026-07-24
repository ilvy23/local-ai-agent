from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActivityEvent:
    kind: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


Subscriber = Callable[[ActivityEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

        return unsubscribe

    def emit(self, event: ActivityEvent) -> None:
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:  # noqa: BLE001, S112
                continue

    def emit_kind(self, kind: str, message: str = "", **data: Any) -> None:
        self.emit(ActivityEvent(kind=kind, message=message, data=data))
