from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _limit_memory(memory_mb: int | None):
    if resource is None or not memory_mb:
        return None

    limit_bytes = memory_mb * 1024 * 1024

    def _apply() -> None:  # pragma: no cover
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        os.setsid()

    return _apply


def _offline_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        env.pop(key, None)
    env["NO_NETWORK"] = "1"
    if extra:
        env.update(extra)
    return env


def run(
    args: list[str],
    cwd: Path,
    timeout_seconds: float = 30.0,
    memory_mb: int | None = 2048,
    env: dict[str, str] | None = None,
) -> RunResult:
    preexec = _limit_memory(memory_mb) if os.name == "posix" else None
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_offline_env(env),
            preexec_fn=preexec,  # noqa: PLW1509
        )
    except (OSError, ValueError) as exc:
        return RunResult(returncode=127, stdout="", stderr=f"failed to launch {args[0]}: {exc}")

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return RunResult(returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
        return RunResult(
            returncode=-signal.SIGKILL,
            stdout=stdout or "",
            stderr=(stderr or "") + f"\n[killed: exceeded {timeout_seconds}s timeout]",
            timed_out=True,
        )


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run_python(module_args: list[str], cwd: Path, **kwargs) -> RunResult:
    return run([sys.executable, "-m", *module_args], cwd=cwd, **kwargs)
