"""Run the WhatsApp Node bridge detached in the background for the lifetime of the interactive
home screen, instead of making you keep a second terminal open on `nightage wa bridge`.

    ensure_running(config)   start it (detached, own process group) if it isn't already
                             answering and it has been paired before; self-installs the
                             cleanup hooks, so the caller never has to stop it explicitly
    stop()                   stop the bridge THIS process started (never one that was
                             already running - e.g. a manually-started `wa bridge` elsewhere)

Rules that keep this safe:
  * never starts anything when the bridge already answers on `whatsapp.bridge_url`;
  * never starts anything before the first pairing (no `auth/creds.json` yet) - that needs a
    QR code on screen, which a silent background process can't show; run `nightage wa bridge`
    once yourself for that, same as always;
  * off with `whatsapp.bridge_autostart: false` or the plugin not installed;
  * output goes to a log file, never the terminal (the whole point is a quiet home screen);
  * cleanup is the same idempotent-on-any-exit shape as agent/ollama_local.py (atexit, SIGINT,
    SIGTERM, SIGHUP), scaled down: one local process, no model-unload step, no multi-client
    bookkeeping (only the process that actually spawned it ever registers a stop hook).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_PLUGIN_DIR = os.path.expanduser("~/.config/agent/plugins")
START_WAIT_S = 8.0
STOP_WAIT_S = 5.0


def _bridge_dir() -> Path:
    return Path(_PLUGIN_DIR) / "whatsapp" / "bridge"


def is_paired() -> bool:
    return (_bridge_dir() / "auth" / "creds.json").exists()


def _health(bridge_url: str) -> dict | None:
    try:
        r = httpx.get(f"{bridge_url}/health", timeout=1.5)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


@dataclass
class Result:
    ok: bool
    action: str   # "already-running" | "started" | "needs-pairing" | "disabled" | "not-installed" | "failed"
    detail: str = ""


_lock = threading.RLock()
_proc: subprocess.Popen | None = None
_hooks_installed = False
_prev_handlers: dict[int, Any] = {}


def _kill_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            os.kill(pid, sig)


def stop(wait: float = STOP_WAIT_S) -> None:
    """Stop the bridge process THIS module started, if any. Safe to call any number of times."""
    global _proc
    with _lock:
        p = _proc
        _proc = None
    if p is None or p.poll() is not None:
        return
    _kill_group(p.pid, signal.SIGTERM)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and p.poll() is None:
        time.sleep(0.05)
    if p.poll() is None:
        _kill_group(p.pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def _on_signal(signum: int, frame: Any) -> None:
    stop()
    prev = _prev_handlers.get(signum)
    with contextlib.suppress(Exception):
        signal.signal(signum, signal.SIG_DFL)
    if signum == signal.SIGINT and callable(prev):
        prev(signum, frame)   # Python's default raises KeyboardInterrupt: the entry point's own finally still runs
        return
    os.kill(os.getpid(), signum)


def _install_hooks() -> None:
    global _hooks_installed
    with _lock:
        if _hooks_installed:
            return
        _hooks_installed = True
    atexit.register(stop)
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)
            if prev is signal.SIG_IGN:
                continue
            _prev_handlers[int(sig)] = prev
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass


def ensure_running(config: dict) -> Result:
    """Make sure the bridge is reachable, starting it detached if it makes sense to. Never
    raises - a WhatsApp plumbing hiccup must never block the home screen from opening."""
    try:
        wa = config.get("whatsapp") or {}
        if wa.get("bridge_autostart", True) is False:
            return Result(True, "disabled")
        bridge_dir = _bridge_dir()
        if not (bridge_dir / "index.js").exists():
            return Result(True, "not-installed")
        bridge_url = wa.get("bridge_url") or "http://127.0.0.1:8765"
        if _health(bridge_url) is not None:
            return Result(True, "already-running")
        if not is_paired():
            return Result(False, "needs-pairing", f"run `nightage wa bridge` once to pair (scan the QR), "
                          "then it starts itself from here on")
        log_path = bridge_dir / "bridge.log"
        with _lock:
            global _proc
            if _proc is not None and _proc.poll() is None:
                return Result(True, "already-running")   # this process already started one
            log = open(log_path, "a")  # noqa: SIM115 - handed to Popen, which owns it; never read back here
            kwargs: dict[str, Any] = {}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True   # own process group: Ctrl+C in the parent's shell doesn't kill it directly
            _proc = subprocess.Popen(["node", "index.js"], cwd=bridge_dir, stdout=log, stderr=log, stdin=subprocess.DEVNULL, **kwargs)
        _install_hooks()
        deadline = time.monotonic() + START_WAIT_S
        while time.monotonic() < deadline:
            if _health(bridge_url) is not None:
                return Result(True, "started")
            if _proc.poll() is not None:
                return Result(False, "failed", f"bridge exited early (code {_proc.returncode}); see {log_path}")
            time.sleep(0.25)
        return Result(False, "failed", f"started but did not answer within {START_WAIT_S:.0f}s; see {log_path}")
    except Exception as exc:  # noqa: BLE001 - never break the home screen over bridge plumbing
        return Result(False, "failed", str(exc))
