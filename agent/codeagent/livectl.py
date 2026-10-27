"""Thread-safe control state for a live UI (terminal dock, web): steering messages, forced compaction, pause / resume / stop, extra step
hooks and the "what is happening now" line. The engine owns one (`engine.live`) and checks it at every step boundary only, so nothing here
interrupts a tool or a model call half way (stop additionally aborts a streaming reply between two pieces)."""

from __future__ import annotations

import threading
from typing import Any, Callable

PAUSE_POLL_S = 0.25


class LiveControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steer: list[str] = []
        self._compact = False
        self._stop = False
        self._run = threading.Event()              # set = may run; cleared = paused
        self._run.set()
        self.hooks: list[Callable[[], None]] = []
        self.now = "idle"
        self.waiting = ""                          # non-empty: blocked on something outside the engine (approval, model server)

    # steering ------------------------------------------------------------------------------------------------------
    def queue_message(self, text: str) -> None:
        text = str(text or "").strip()
        if text:
            with self._lock:
                self._steer.append(text)

    def drain_messages(self) -> list[str]:
        with self._lock:
            out, self._steer = self._steer, []
        return out

    def queued(self) -> int:
        with self._lock:
            return len(self._steer)

    # compaction ----------------------------------------------------------------------------------------------------
    def request_compact(self) -> None:
        with self._lock:
            self._compact = True

    def take_compact(self) -> bool:
        with self._lock:
            want, self._compact = self._compact, False
        return want

    # pause / stop --------------------------------------------------------------------------------------------------
    def pause(self) -> None:
        self._run.clear()

    def resume(self) -> None:
        self._run.set()

    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
        self._run.set()                            # a paused run must wake up to end

    @property
    def stopping(self) -> bool:
        return self._stop

    def clear_stop(self) -> None:
        with self._lock:
            self._stop = False

    def wait_while_paused(self, on_tick: Callable[[], None] | None = None) -> bool:
        """Block while paused (waking up every PAUSE_POLL_S). True when it actually paused."""
        did = False
        while not self._run.is_set() and not self._stop:
            did = True
            if on_tick:
                on_tick()
            self._run.wait(PAUSE_POLL_S)
        return did

    # hooks ---------------------------------------------------------------------------------------------------------
    def add_hook(self, fn: Callable[[], None]) -> None:
        with self._lock:
            if fn not in self.hooks:
                self.hooks = [*self.hooks, fn]

    def remove_hook(self, fn: Callable[[], None]) -> None:
        with self._lock:
            self.hooks = [h for h in self.hooks if h is not fn and h != fn]

    def fire_hooks(self) -> None:
        for fn in self.hooks:                      # the list is replaced, never mutated: iterating a snapshot is safe
            try:
                fn()
            except Exception:  # noqa: BLE001 - a UI hook must never break the run
                pass


def clip(text: Any, limit: int = 20000) -> str:
    s = str(text)
    return s if len(s) <= limit else s[:limit] + f"\n… [{len(s) - limit} more characters not shown]"
