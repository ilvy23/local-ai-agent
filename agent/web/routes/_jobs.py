"""Tiny in-memory job registry for slow model calls: start a function in a thread, poll it, ask it to stop.

The function gets the job dict and updates `text` / `log` / `pct` as it goes and should check `job["stop"]` between steps.
Jobs live in memory only (a restart forgets them) and the oldest finished ones are dropped.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
_KEEP = 40


def start(kind: str, fn: Callable[[dict[str, Any]], Any], **meta: Any) -> str:
    jid = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {"id": jid, "kind": kind, "status": "running", "text": "", "log": [], "error": None, "result": None,
                           "stop": False, "t0": time.time(), "t1": None, **meta}
    with _LOCK:
        _JOBS[jid] = job
        for old in [k for k, v in _JOBS.items() if v["status"] != "running"][:-_KEEP]:
            _JOBS.pop(old, None)

    def run() -> None:
        try:
            job["result"] = fn(job)
            job["status"] = "stopped" if job["stop"] else "done"
        except Exception as exc:  # noqa: BLE001 - any failure becomes the job's error text
            job["error"] = str(exc)[:600] or exc.__class__.__name__
            job["status"] = "failed"
        job["t1"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return jid


def get(jid: str) -> dict[str, Any] | None:
    with _LOCK:
        j = _JOBS.get(jid)
    if j is None:
        return None
    end = j["t1"] or time.time()
    return {k: v for k, v in j.items() if k != "stop"} | {"elapsed_s": round(end - j["t0"], 1)}


def stop(jid: str) -> bool:
    with _LOCK:
        j = _JOBS.get(jid)
    if j is None:
        return False
    j["stop"] = True
    return True


def running(kind: str, **match: Any) -> dict[str, Any] | None:
    with _LOCK:
        for j in _JOBS.values():
            if j["kind"] == kind and j["status"] == "running" and all(j.get(k) == v for k, v in match.items()):
                return j
    return None
