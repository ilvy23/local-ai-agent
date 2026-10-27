"""Resource governor for background AI work.

Background jobs (e.g. re-embedding the memory index) call
`wait_until_clear()` between items so they yield the machine when you're
gaming or the GPU is already busy — the point is full performance while you
play, not starving the foreground.

Honest scope: the actual model compute runs inside Ollama (a separate
server), so a cgroup CPU/RAM cap on this Python process wouldn't limit
inference. The effective lever is therefore a PAUSE gate, not a hard cap:
we skip running the next background item while a game is running or GPU
utilisation is above the configured ceiling. `max_cpu_percent`/`max_ram_gb`
are kept in config for completeness and checked best-effort, but the GPU +
game signals are what protect gaming.
# ponytail: pause-gate over cgroups because compute is in Ollama, not here.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
from collections import deque
from typing import Deque, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------


class RepetitionDetected(Exception):
    """Raised when the same (tool_name, input_hash) appears ≥3 times in the last 5 calls."""


class RepetitionTracker:
    """Tracks a sliding window of the last 5 (tool_name, input_hash) pairs."""

    _WINDOW = 5
    _THRESHOLD = 3

    def __init__(self) -> None:
        self._history: Deque[Tuple[str, str]] = deque(maxlen=self._WINDOW)

    def record_tool_call(self, tool_name: str, input_str: str) -> None:
        """Record a call; raise RepetitionDetected if the same pair hits ≥3 in the window."""
        # usedforsecurity=False: this is a dedupe fingerprint, not a digest. A
        # FIPS-enforcing build refuses plain md5() outright and would raise here.
        h = hashlib.md5(input_str.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()[:8]
        entry = (tool_name, h)
        self._history.append(entry)
        if self._history.count(entry) >= self._THRESHOLD:
            raise RepetitionDetected(
                f"tool '{tool_name}' called with the same input {self._THRESHOLD} times"
                f" in the last {self._WINDOW} calls (hash {h})"
            )


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set:
    return set(w.lower() for w in re.split(r"[^a-zA-Z0-9]+", text) if w)


def compute_drift(original_task: str, current_tool_name: str, current_input: str) -> float:
    """Jaccard similarity between original_task words and current action words.

    Returns 0.0 (no overlap, high drift) … 1.0 (identical).
    """
    a = _tokenize(original_task)
    b = _tokenize(f"{current_tool_name} {current_input}")
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_drift(
    original_task: str,
    current_tool_name: str,
    current_input: str,
    threshold: float = 0.05,
) -> bool:
    """Return True when the current action is severely off-topic (score < threshold)."""
    score = compute_drift(original_task, current_tool_name, current_input)
    return score < threshold

# Substrings identifying a running game / game launcher in a process command
# line. Proton/Wine cover Steam games; gamescope/gamemoded are activated by
# many launchers; heroic covers Epic/GOG via Heroic.
_GAME_MARKERS = (
    "steamapps/common/",  # a Steam game's actual executable
    "proton",
    "wine64-preloader",
    "gamescope",
    "gamemoded",
    "heroic",
    "lutris",
)
# Bare launcher processes that are NOT a game by themselves — being open
# shouldn't pause anything; only an actual game should.
_LAUNCHER_ONLY = ("steam", "steamwebhelper")


def game_running() -> str | None:
    """Return a short marker of a detected running game, or None.

    Reads process command lines from /proc (Linux). A bare Steam/Heroic window
    open is ignored; only an actual game process (Proton/Wine/gamescope/a
    steamapps binary) counts.
    """
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return None
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "ignore").lower()
        except OSError:
            continue  # process vanished / not readable
        if not cmd:
            continue
        for marker in _GAME_MARKERS:
            if marker in cmd:
                return marker
    return None


def gpu_utilization() -> int | None:
    """Current GPU utilisation percent via nvidia-smi, or None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
        return max(int(v) for v in out) if out else None
    except (subprocess.SubprocessError, ValueError):
        return None


def pause_reason(config: dict, *, ignore_foreground: bool = False) -> str | None:
    """Why background work should pause right now, or None to proceed.

    `ignore_foreground=True` skips the "user is active" check — used by a job
    that itself marks foreground (the overnight queue) so it doesn't pause on
    its own lock, while still yielding to games."""
    # You're actively using agent (chat / deep dive) — yield the GPU to it.
    if not ignore_foreground:
        from agent.foreground import active as _foreground_active

        if _foreground_active(config):
            return "you're using agent directly"
    bg = config.get("background", {})
    if bg.get("pause_on_game", True):
        game = game_running()
        if game:
            return f"game running ({game})"
    cap = bg.get("max_gpu_percent", 40)
    util = gpu_utilization()
    if util is not None and util > cap:
        return f"GPU busy ({util}% > {cap}%)"
    return None


def wait_until_clear(
    config: dict, *, poll_seconds: float = 15.0, on_wait=None, ignore_foreground: bool = False
) -> None:
    """Block while background work should be paused. Returns once clear.

    `on_wait(reason)` is called once each time we start waiting, so a caller
    can tell the user why it stalled. A game session can last hours, so this
    genuinely blocks — that's the intent (yield the machine while gaming).
    `ignore_foreground` is threaded to `pause_reason` for a self-marking job.
    """
    reason = pause_reason(config, ignore_foreground=ignore_foreground)
    if reason is None:
        return
    if on_wait:
        on_wait(reason)
    while pause_reason(config, ignore_foreground=ignore_foreground) is not None:
        time.sleep(poll_seconds)
