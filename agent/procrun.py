"""Run one command for a tool so that it can neither hang the agent nor lose the part of its output that matters.

  * output goes to a temp file, not a pipe: a background child (`server &`) that keeps a pipe open used to make the call wait
    until the timeout although the command itself had finished;
  * the command runs in its own process group, and a timeout kills the WHOLE group (`make` and the compiler it started, a test
    runner and its workers), not only the shell;
  * stdin is closed, so a command that waits for input fails at once instead of hanging until the timeout;
  * long output keeps its head AND its tail (`cap`): test runners and builds print the summary last.

Commands that must talk to the terminal (`sudo` asking for a password) are the caller's to run differently."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass

from agent import osport

__all__ = ["Result", "run", "cap"]

HEAD_CHARS = 2500
TAIL_CHARS = 7000


@dataclass
class Result:
    code: int | None            # None: stopped at the timeout
    output: str
    seconds: float
    timed_out: bool = False


def cap(text: str, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> str:
    """`text` if it fits, else its first `head` and last `tail` characters with a marker saying how much was cut."""
    if len(text) <= head + tail:
        return text
    cut = len(text) - head - tail
    return f"{text[:head]}\n[... {cut} characters cut from the middle ...]\n{text[-tail:]}"


def _kill_group(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        osport.kill_group(proc.pid, getattr(signal, "SIGKILL", None))
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def run(argv: list[str], *, timeout: float, cwd: str | None = None, env: dict[str, str] | None = None) -> Result:
    """Run `argv` (see the module docstring). Never raises for the command's own failures; a missing program is exit code 127."""
    t0 = time.monotonic()
    kwargs: dict = {"start_new_session": True} if os.name != "nt" else \
        {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    with tempfile.TemporaryFile() as out:
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **kwargs)
        except OSError as exc:
            return Result(127, f"could not start the command: {exc}", 0.0)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
        except BaseException:                     # Ctrl-C / shutdown: never leave the command running behind us
            _kill_group(proc)
            raise
        out.seek(0)
        data = out.read()
    return Result(None if timed_out else proc.returncode, data.decode("utf-8", errors="replace"), time.monotonic() - t0, timed_out)
