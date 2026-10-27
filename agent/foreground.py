"""Foreground-activity lock: pause background work while you use the tool.

When you're chatting, running a deep dive, or any interactive model command, we
touch a small lock file (refreshed by a daemon thread for the command's whole
duration). The governor treats a fresh lock as "user is active" and pauses the
background watcher + overnight queue, so your interactive work gets the GPU to
itself. The lock is cleared on exit and self-expires (TTL) if a process dies.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

# A lock older than this is ignored (covers a crashed process that never
# cleared it). The refresh thread re-touches well within this window.
TTL_SECONDS = 90
_REFRESH_SECONDS = 30


def _path(config: dict) -> Path:
    from agent.config import PROJECT_ROOT

    db = Path(config["data"]["db_path"])
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    return db.parent / "foreground.lock"


def mark(config: dict) -> None:
    try:
        _path(config).write_text(str(int(time.time())))
    except Exception:  # noqa: BLE001 - best-effort; never break a command
        pass


def clear(config: dict) -> None:
    try:
        _path(config).unlink()
    except Exception:  # noqa: BLE001 - includes FileNotFoundError / bad config
        pass


def active(config: dict, ttl: int = TTL_SECONDS) -> bool:
    """True if an interactive session touched the lock within `ttl` seconds.

    Any problem (missing lock, malformed config without a data path) reads as
    'not active' so background work is never wedged by a bad foreground check."""
    try:
        ts = int(_path(config).read_text().strip())
    except Exception:  # noqa: BLE001
        return False
    return (time.time() - ts) < ttl


_process_started = False


def begin(config: dict) -> None:
    """Mark this whole process as a foreground session (idempotent).

    Called by the interactive LLM entrypoint so any model command (chat, deep
    dive, profile…) counts as active for its lifetime. Starts one daemon
    refresher and clears the lock at process exit."""
    global _process_started
    if _process_started:
        return
    _process_started = True
    mark(config)
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(_REFRESH_SECONDS):
            mark(config)

    threading.Thread(target=_loop, daemon=True).start()
    import atexit

    atexit.register(lambda: (stop.set(), clear(config)))


class session:
    """Context manager marking foreground activity for its whole duration.

    Use around interactive/heavy model commands (chat, deep dives). A daemon
    thread refreshes the lock so long-running or idle-but-open sessions still
    count as active; the lock is cleared on exit."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._stop: threading.Event | None = None

    def __enter__(self) -> "session":
        mark(self._config)
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self) -> None:
        assert self._stop is not None
        while not self._stop.wait(_REFRESH_SECONDS):
            mark(self._config)

    def __exit__(self, *exc: object) -> None:
        if self._stop is not None:
            self._stop.set()
        clear(self._config)
