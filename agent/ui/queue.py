"""Thread-safe FIFO of prompts typed while the agent is busy."""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field


@dataclass
class QueueItem:
    id: int
    text: str
    kind: str = "prompt"  # prompt | command
    added: float = field(default_factory=time.time)
    note: str = ""


class PromptQueue:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: list[QueueItem] = []
        self._ids = itertools.count(1)
        self.paused = False

    def enqueue(self, text: str, kind: str = "prompt", note: str = "") -> QueueItem:
        with self._lock:
            item = QueueItem(next(self._ids), text, kind, note=note)
            self._items.append(item)
            return item

    def pop_next(self) -> QueueItem | None:
        with self._lock:
            if self.paused or not self._items:
                return None
            return self._items.pop(0)

    def items(self) -> list[QueueItem]:
        with self._lock:
            return list(self._items)

    def remove(self, n: int) -> QueueItem | None:
        """Remove the n-th item (1-based, as displayed)."""
        with self._lock:
            if not 1 <= n <= len(self._items):
                return None
            return self._items.pop(n - 1)

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items.clear()
            return n

    def move(self, n: int, to: int) -> bool:
        """Move item n to position `to` (both 1-based; `to` is clamped)."""
        with self._lock:
            if not 1 <= n <= len(self._items):
                return False
            item = self._items.pop(n - 1)
            to = min(max(to, 1), len(self._items) + 1)
            self._items.insert(to - 1, item)
            return True

    def pull_back(self) -> QueueItem | None:
        """Remove and return the most recently queued item (to put it back into the editor)."""
        with self._lock:
            return self._items.pop() if self._items else None

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def lines(self, width: int = 80) -> list[str]:
        items = self.items()
        if not items:
            return []
        head = f"queued ({len(items)})" + (" [paused]" if self.paused else "") + ":"
        out = [head]
        for i, it in enumerate(items, 1):
            text = " ".join(it.text.split())
            prefix = f"  {i}. " + ("(cmd) " if it.kind == "command" else "")
            room = max(4, width - len(prefix))
            if len(text) > room:
                text = text[: room - 1] + "…"
            out.append(prefix + text)
        return out
