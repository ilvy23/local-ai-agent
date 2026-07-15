"""Live 'what the worker is doing right now' marker.

The watcher writes a one-line activity (e.g. "summarizing a chat") to a small
JSON file in the data dir as it moves item to item; `companion worker status`
reads it back so you can see the current task, not just the last pass. Best-
effort — a failed write never disrupts the actual work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _activity_path(config: dict) -> Path:
    from companion.config import PROJECT_ROOT

    db = Path(config["data"]["db_path"])
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    return db.parent / "worker_activity.json"


def set_activity(config: dict, text: str) -> None:
    """Record what the worker is doing right now (best-effort)."""
    try:
        _activity_path(config).write_text(json.dumps({"text": text, "at": int(time.time())}))
    except OSError:
        pass


def read_activity(config: dict) -> tuple[str, int] | None:
    """(text, seconds_ago) of the current activity, or None if unknown."""
    try:
        d = json.loads(_activity_path(config).read_text())
        return d.get("text", ""), max(0, int(time.time()) - int(d.get("at", 0)))
    except (OSError, ValueError):
        return None
