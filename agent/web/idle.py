"""Tracks "is anyone actually using this right now" for the idle-auto-suspend
script (see scripts/agent-idle-suspend.sh) - a plain file with an epoch
timestamp, touched on every request. Deliberately outside the SQLite store:
this needs to be readable by a root-run shell script with zero Python/DB
dependency, and a single mtime-bearing file is the simplest thing that works
for that.
"""

from __future__ import annotations

import time

from agent.config import DEFAULT_DATA_DIR

LAST_ACTIVE_FILE = DEFAULT_DATA_DIR / "last_active"

# The idle-suspend script only checks in 20-MINUTE granularity, so there is
# no reason to hit disk on every single request. Throttling matters for more
# than just efficiency: touch_active() runs inline in an ASGI middleware, on
# the event loop, on every request - an unthrottled blocking write here once
# stalled EVERY concurrent request (including trivial health checks the
# WoL relay depends on to know the PC is awake) the moment a real video
# transcode was hammering the same disk with a multi-GB write at the same
# time. A rare, throttled write no longer meaningfully contends with that.
_WRITE_THROTTLE_S = 10.0
_last_written_at = 0.0


def touch_active() -> None:
    global _last_written_at
    now = time.monotonic()
    if now - _last_written_at < _WRITE_THROTTLE_S:
        return
    _last_written_at = now
    LAST_ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_ACTIVE_FILE.write_text(str(int(time.time())))
