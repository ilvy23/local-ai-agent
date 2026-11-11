"""Stray processes of an attempt. An attempt that starts a dev server, a watcher or a `python -m http.server` and then ends leaves it running all night: it holds a port,
RAM and sometimes the GPU. Every attempt is started as its own session (runner.py: start_new_session=True), so the session id is the attempt's pid and every process it
left behind carries it. `reap_session()` ends exactly those - never anything of the user's, whatever its cwd. (A process that detaches into a new session escapes: the
shortcut of this design; it is rare for the commands a coding agent runs.)"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Callable


def _sid(stat_text: str) -> int | None:
    """The session id of a /proc/<pid>/stat line ('pid (comm) state ppid pgrp session ...'); comm may contain spaces and parentheses."""
    try:
        rest = stat_text[stat_text.rindex(")") + 2:].split()
        return int(rest[3])
    except (ValueError, IndexError):
        return None


def members(sid: int, proc: Path = Path("/proc")) -> list[tuple[int, str]]:
    """(pid, command line) of every live process in session `sid`, this process excluded."""
    out = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if _sid((entry / "stat").read_text()) != sid:
                continue
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip() or "(no command line)"
        except OSError:
            continue
        out.append((int(entry.name), cmd[:100]))
    return out


def reap_session(sid: int, *, proc: Path = Path("/proc"), kill: Callable[[int, int], None] = os.kill, sleep: Callable[[float], None] = time.sleep, grace_s: float = 3.0) -> list[str]:
    """Stop everything left in session `sid`: SIGTERM, a short grace period, then SIGKILL. Returns the command lines that were stopped."""
    left = members(sid, proc)
    if not left:
        return []
    for pid, _ in left:
        try:
            kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    sleep(grace_s)
    for pid, _ in members(sid, proc):
        try:
            kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return [cmd for _, cmd in left]
