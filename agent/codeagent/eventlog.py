"""Append-only JSONL log of what the coding engine did to its own context (masks, compactions, per-step timing).

The earlier engine kept this only in memory, so after a slow or confusing run nobody could say when the context was
trimmed or where the seconds went. One line per event: {"ts", "session", "kind", ...}. Never raises."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from agent import osport
from typing import Any, Callable


def default_path() -> Path:
    base = str(osport.state_home())
    return Path(os.environ.get("AGENT_CODE_EVENTS") or Path(base) / "agent" / "code-events.jsonl")


class EventLog:
    def __init__(self, session: str, path: Path | None = None) -> None:
        self.session = session
        self.path = path or default_path()
        self.recent: list[dict[str, Any]] = []
        self._subs: list[Callable[[dict[str, Any]], None]] = []
        self._sub_lock = threading.Lock()

    def subscribe(self, fn: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Call `fn(record)` for every event (persisted and live). Returns the unsubscribe function. `fn` runs on the emitting thread
        (usually the engine's): keep it quick, hand work to a queue. Its exceptions are swallowed."""
        with self._sub_lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._sub_lock:
                if fn in self._subs:
                    self._subs.remove(fn)
        return unsubscribe

    def _notify(self, rec: dict[str, Any]) -> None:
        with self._sub_lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(rec)
            except Exception:  # noqa: BLE001 - a broken UI must never break the engine
                pass

    def emit_live(self, kind: str, /, **data: Any) -> dict[str, Any]:
        """Like emit, but only for subscribers: not written to the JSONL file and not kept in `recent` (full diffs, full tool arguments and
        results, thinking text - large or sensitive payloads that must not persist)."""
        rec = {"ts": round(time.time(), 2), "session": self.session, "kind": kind, "live": True, **data}
        self._notify(rec)
        return rec

    def emit(self, kind: str, /, **data: Any) -> dict[str, Any]:
        rec = {"ts": round(time.time(), 2), "session": self.session, "kind": kind, **data}
        self.recent.append(rec)
        del self.recent[:-200]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass
        self._notify(rec)
        return rec

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.recent if r["kind"] == kind]
