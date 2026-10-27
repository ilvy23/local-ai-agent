"""Small portability helpers: where per-user folders live on each OS, advisory file locks and process-group kills.

Linux and macOS behave exactly as before (XDG-style `~/.config`, `~/.local/state`, `~/.local/share`; `fcntl.flock`; `os.killpg`).
On Windows (UNVERIFIED on real hardware - there is no Windows machine in the project's test setup, see docs/PORTABILITY.md) the folders
follow the platform convention: settings under %APPDATA%, state and data under %LOCALAPPDATA%; locks use `msvcrt.locking`.
Every helper decides at CALL time from `sys.platform`, so tests can simulate Windows.
"""

from __future__ import annotations

import contextlib
import ntpath
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import IO, Any, Mapping


def is_windows() -> bool:
    return sys.platform == "win32"


# ── per-user folders ────────────────────────────────────────────────────────

def windows_dirs(env: Mapping[str, str], home: str) -> dict[str, str]:
    """The Windows layout as plain strings (ntpath): {'config': %APPDATA%, 'state': %LOCALAPPDATA%, 'data': %LOCALAPPDATA%}.
    Pure, so it is testable on any OS."""
    roaming = env.get("APPDATA") or ntpath.join(home, "AppData", "Roaming")
    local = env.get("LOCALAPPDATA") or ntpath.join(home, "AppData", "Local")
    return {"config": roaming, "state": local, "data": local}


def _home(home: str | Path | None) -> Path:
    return Path(home) if home is not None else Path.home()


def config_home(env: Mapping[str, str] | None = None, home: str | Path | None = None) -> Path:
    """The folder that CONTAINS the `agent` config dir: $XDG_CONFIG_HOME or ~/.config (Windows: %APPDATA%)."""
    e = os.environ if env is None else env
    if is_windows():
        return Path(windows_dirs(e, str(_home(home)))["config"])
    return Path(e.get("XDG_CONFIG_HOME") or _home(home) / ".config")


def state_home(env: Mapping[str, str] | None = None, home: str | Path | None = None) -> Path:
    """$XDG_STATE_HOME or ~/.local/state (Windows: %LOCALAPPDATA%)."""
    e = os.environ if env is None else env
    if is_windows():
        return Path(windows_dirs(e, str(_home(home)))["state"])
    return Path(e.get("XDG_STATE_HOME") or _home(home) / ".local" / "state")


def data_home(env: Mapping[str, str] | None = None, home: str | Path | None = None) -> Path:
    """~/.local/share (Windows: %LOCALAPPDATA%). Not XDG_DATA_HOME on purpose: that is how installs have always behaved."""
    e = os.environ if env is None else env
    if is_windows():
        return Path(windows_dirs(e, str(_home(home)))["data"])
    return _home(home) / ".local" / "share"


# ── restrict a secret file to its owner ─────────────────────────────────────

def private_mode(path: str | Path, mode: int = 0o600) -> None:
    """chmod where the OS has POSIX modes; on Windows files in %APPDATA% are already private to the user's profile (ACLs), so no-op."""
    if is_windows():
        return
    with contextlib.suppress(OSError):
        os.chmod(path, mode)


# ── advisory file locks ─────────────────────────────────────────────────────

def try_lock(fh: IO[Any]) -> bool:
    """Take an exclusive non-blocking lock on the open file; False when another process holds it. Released when the file closes
    or the process dies (flock / msvcrt.locking both behave that way)."""
    if is_windows():
        import msvcrt

        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)       # type: ignore[attr-defined]
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def unlock(fh: IO[Any]) -> None:
    if is_windows():
        import msvcrt

        with contextlib.suppress(OSError):
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)       # type: ignore[attr-defined]
        return
    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(fh, fcntl.LOCK_UN)


# ── processes ───────────────────────────────────────────────────────────────

def kill_group(pid: int, sig: int | None = None) -> None:
    """Signal a whole process group (POSIX: os.killpg, default SIGKILL). Windows: `taskkill /T /F` for the process tree.
    Raises ProcessLookupError / PermissionError like os.killpg so callers' existing handlers keep working."""
    if is_windows():
        res = subprocess.run(["taskkill", "/T", "/F", "/PID", str(int(pid))], capture_output=True, check=False, timeout=15)
        if res.returncode not in (0, 128):
            raise ProcessLookupError(pid)
        return
    os.killpg(pid, sig if sig is not None else signal.SIGKILL)


def term_signal() -> int:
    """SIGTERM exists on Windows too (os.kill with it terminates the process)."""
    return int(signal.SIGTERM)


# ── shell commands for the agents' run_command tools ────────────────────────

NO_SHELL_MESSAGE = ("this tool runs commands with bash, and bash was not found on this Windows PC. Install Git for Windows (it brings Git Bash) "
                    "or, better, use WSL (Windows Subsystem for Linux) and run nightage inside it - see docs/INSTALL.md. Chat, memory and "
                    "the demo work without bash; the shell-based code agent needs it.")


def shell_argv(command: str) -> list[str] | None:
    """argv that runs `command` through bash (`bash -lc`, exactly as before on POSIX). Windows: only when a bash exists on PATH
    (Git Bash / WSL); otherwise None and the caller reports NO_SHELL_MESSAGE."""
    if is_windows():
        import shutil

        exe = shutil.which("bash")
        return [exe, "-lc", command] if exe else None
    return ["bash", "-lc", command]


# ── RAM where there is no /proc/meminfo ─────────────────────────────────────

def native_ram_mb() -> tuple[int, int]:
    """(total_mb, available_mb) from the OS API where /proc/meminfo does not exist: Windows (GlobalMemoryStatusEx via ctypes) and
    macOS (sysctl hw.memsize + vm_stat is overkill, so only the total). (0, 0) on any other system or on any error. Never raises."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _Mem()
            m.dwLength = ctypes.sizeof(_Mem)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):      # type: ignore[attr-defined]
                return 0, 0
            return int(m.ullTotalPhys // (1024 * 1024)), int(m.ullAvailPhys // (1024 * 1024))
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
            total = int(out) // (1024 * 1024)
            return total, total // 2
    except Exception:  # noqa: BLE001 - a probe must never break its caller
        pass
    return 0, 0


# ── Windows console / text encoding ─────────────────────────────────────────

ENTRY_NAMES = frozenset({"nightage", "agent", "lantern"})


def windows_utf8_setup() -> None:
    """Windows only (UNVERIFIED on real hardware); a no-op everywhere else and when already in UTF-8 mode.

    Python on Windows defaults to the legacy ANSI code page (cp1252) for files and pipes, which crashes on the unicode glyphs and
    emoji this program prints and mis-reads UTF-8 config files. Fix: when started as the `nightage` command, restart ONCE with
    `-X utf8` (set NIGHTAGE_NO_REEXEC=1 to skip), and make stdout/stderr tolerant either way (`errors=replace`)."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")      # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - a stream without reconfigure (a test capture) is fine
            pass
    if sys.flags.utf8_mode or os.environ.get("NIGHTAGE_NO_REEXEC"):
        return
    stem = ntpath.splitext(ntpath.basename(sys.argv[0] if sys.argv else ""))[0].lower()
    if stem.replace("-script", "") not in ENTRY_NAMES or not getattr(sys, "orig_argv", None):
        return
    try:
        env = {**os.environ, "PYTHONUTF8": "1", "NIGHTAGE_NO_REEXEC": "1"}
        raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.orig_argv[1:]], env=env))
    except OSError:
        return                                                          # could not restart: carry on in the legacy encoding
