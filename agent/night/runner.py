"""One attempt = one `nightage code --autopilot` process, watched from outside.

A process (not a thread) so that a hung or crashed attempt can be killed and the next one starts with a completely fresh model context.
The watchdog ends an attempt when it ran past its time slice ("timeout") or when the event log of the coding agent has not grown for
`stall_s` ("stalled"); the whole process group is stopped."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.night import reap


@dataclass
class Attempt:
    task: str                      # the full task text; empty with new=False means "continue the saved session"
    new: bool
    timeout_s: float
    stall_s: float = 20 * 60
    idle_s: float = 30 * 60        # no change of the work tree for this long (events may still flow: a model that only reads and thinks) ends the attempt
    log: str = ""                  # file the output goes to
    on_poll: Callable[..., None] | None = None         # called at every poll while the attempt runs (proof of life for the watchdog)
    on_start: Callable[[int], None] | None = None      # called with the pid of the attempt's process (session leader) right after it started


@dataclass
class Outcome:
    code: int                      # the autopilot's exit code: 0 done, 2 needs a human, 3 partial, 130 stopped; -1 killed
    why: str = ""                  # "" | timeout | stalled | crash
    reaped: list[str] | None = None    # command lines of processes the attempt left behind (a dev server ...) that were stopped


def _stop(proc: subprocess.Popen, sleep: Callable[[float], None]) -> None:
    for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        for _ in range(wait):
            if proc.poll() is not None:
                return
            sleep(1)


def subprocess_runner(root: Path, events: Path, *, mode: str = "auto", clock: Callable[[], float] = time.monotonic,
                      sleep: Callable[[float], None] = time.sleep, poll_s: float = 15.0, env: dict[str, str] | None = None,
                      progress: Callable[[], str] | None = None, progress_every_s: float = 60.0,
                      popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> Callable[[Attempt], Outcome]:
    def run(a: Attempt) -> Outcome:
        argv = [sys.executable, "-c", "from agent.night.overlay import boot; boot()", "code"]
        argv += [a.task, "--new"] if a.new else ["--resume"]
        argv += ["--folder", str(root), "--autopilot", "--mode", mode, "--plain", "--old-ui"]
        out = open(a.log or os.devnull, "ab")
        try:
            proc = popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, **(env or {})})
        finally:
            out.close()
        if a.on_start:
            try:
                a.on_start(proc.pid)
            except OSError:
                pass
        start = clock()
        seen = _mtime(events)
        seen_at = start
        token, token_at, probed = progress() if progress else "", start, start
        try:
            return _watch(proc, a, start, seen, seen_at, token, token_at, probed)
        except BaseException:                                     # Ctrl+C / SIGTERM of the governor: the attempt must not go on working alone in the folder
            _stop(proc, sleep)
            _reap(proc.pid, sleep)
            raise

    def _watch(proc, a: Attempt, start: float, seen: float, seen_at: float, token: str, token_at: float, probed: float) -> Outcome:
        while proc.poll() is None:
            sleep(poll_s)
            if a.on_poll:
                a.on_poll()
            now = clock()
            m = _mtime(events)
            if m != seen:
                seen, seen_at = m, now
            if now - start > a.timeout_s:
                _stop(proc, sleep)
                return Outcome(-1, "timeout", _reap(proc.pid, sleep))
            if progress and now - probed >= progress_every_s:
                probed = now
                fresh = progress()
                if fresh != token:
                    token, token_at = fresh, now
                elif now - token_at > a.idle_s and now - start > a.idle_s:
                    _stop(proc, sleep)
                    return Outcome(-1, "idle", _reap(proc.pid, sleep))
            if now - seen_at > a.stall_s:
                _stop(proc, sleep)
                return Outcome(-1, "stalled", _reap(proc.pid, sleep))
        code = proc.returncode
        return Outcome(code, "" if code in (0, 2, 3, 130) else "crash", _reap(proc.pid, sleep))
    return run


def _reap(pid: int, sleep: Callable[[float], None]) -> list[str]:
    try:
        return reap.reap_session(pid, sleep=sleep) if real_session(pid) else []
    except OSError:
        return []


def real_session(pid: int) -> bool:
    """Only a real session is reaped: a fake pid in a test, or a pid that is not its own session leader, is left alone."""
    try:
        return os.getsid(pid) == pid
    except OSError:
        return False


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0
