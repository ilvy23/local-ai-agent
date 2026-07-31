"""Run a subprocess with a wall-clock timeout and a memory cap.

ponytail: no true network isolation yet — that needs `unshare`/namespaces (root
or CAP_NET_ADMIN) or firejail. Timeout + memory cap cover the common failure
modes (infinite loops, fork bombs, runaway allocation). Add net isolation when a
task actually needs to run untrusted network code.
"""

from __future__ import annotations

import os
import resource
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    rc: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out

    @property
    def output(self) -> str:
        return (self.stdout + ("\n" + self.stderr if self.stderr else "")).strip()


def _limits(mem_mb: int):
    def _apply() -> None:
        limit = mem_mb * 1024 * 1024
        for res in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(res, (limit, limit))
            except (ValueError, OSError):
                pass
    return _apply


def run(cmd: list[str], cwd: str | Path, timeout: int = 30, mem_mb: int = 2048,
        env: dict | None = None) -> RunResult:
    """Execute `cmd` in `cwd`, capturing output. Never raises on child failure.

    stdin is closed (subprocess.DEVNULL) so that a command which tries to prompt
    interactively — e.g. sudo asking for your password, apt-get asking Y/n —
    cannot hijack the terminal; it will just get EOF and exit."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
            preexec_fn=_limits(mem_mb) if os.name == "posix" else None,
            env=env,
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            124, exc.stdout or "", (exc.stderr or "") + f"\n[killed: exceeded {timeout}s]", True
        )
    except MemoryError:
        return RunResult(137, "", f"[killed: exceeded {mem_mb} MB]", False)
