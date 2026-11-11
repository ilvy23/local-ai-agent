"""Is the machine fit to work right now? One gate before every attempt: model server, disk, a game, the GPU's temperature.

`Health.gate()` returns None (go) or a reason to wait for. Waiting is the governor's job (it counts against the night's clock, so a long
pause shortens the work instead of extending the night). A dead model server is revived (`restart_cmd`, default the user service
that runs it) with backoff; everything is injectable so it can be tested without a server, a disk or a game."""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

MIN_FREE_GB = 2.0
HOT_C = 92.0


def ping(url: str, timeout: float = 3.0) -> bool:
    try:
        return httpx.get(f"{url.rstrip('/')}/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def gpu_temps() -> list[float]:
    """Edge temperatures of the amdgpu cards in degrees C (sysfs); [] elsewhere."""
    out = []
    for f in Path("/sys/class/drm").glob("card[0-9]/device/hwmon/hwmon*/temp1_input"):
        try:
            out.append(int(f.read_text()) / 1000.0)
        except (OSError, ValueError):
            pass
    return out


@dataclass
class Health:
    root: Path
    urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:11436"])
    restart_cmds: list[list[str]] = field(default_factory=lambda: [["systemctl", "--user", "restart", "ollama-vk"]])
    pause_on_game: bool = True
    clock: Callable[[], float] = time.monotonic
    probe: Callable[[str], bool] = ping
    run: Callable[..., object] = subprocess.run
    game: Callable[[], str | None] | None = None
    temps: Callable[[], list[float]] = gpu_temps
    free_gb: Callable[[Path], float] = lambda p: shutil.disk_usage(p).free / 2**30
    sleep: Callable[[float], None] = time.sleep
    last_restart: float = -1e9
    restarts: int = 0

    def server_down(self) -> str | None:
        down = [u for u in self.urls if not self.probe(u)]
        return f"model server down ({', '.join(down)})" if down else None

    def revive(self, min_gap_s: float = 120.0) -> bool:
        """Try to bring the model server back (at most every `min_gap_s`); True when it answers afterwards."""
        if self.clock() - self.last_restart < min_gap_s:
            return False
        self.last_restart = self.clock()
        self.restarts += 1
        for cmd in self.restart_cmds:
            try:
                self.run(cmd, capture_output=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                continue
        return self.server_down() is None

    def restart(self, min_gap_s: float = 1800.0, wait_s: float = 90.0) -> bool:
        """Restart the model server even though it answers (a server that got slow: fragmented graphics memory, a leaked runner). At most every `min_gap_s`;
        waits up to `wait_s` for it to answer again. True when it did restart and answers."""
        if self.clock() - self.last_restart < min_gap_s or not self.restart_cmds:
            return False
        self.last_restart = self.clock()
        self.restarts += 1
        for cmd in self.restart_cmds:
            try:
                self.run(cmd, capture_output=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                continue
        waited = 0.0
        while self.server_down() and waited < wait_s:
            self.sleep(2.0)
            waited += 2.0
        return self.server_down() is None

    def gate(self) -> str | None:
        down = self.server_down()
        if down and not self.revive():
            return down
        if self.free_gb(self.root) < MIN_FREE_GB:
            return f"disk almost full (< {MIN_FREE_GB:.0f} GB free)"
        if self.pause_on_game:
            game = self.game() if self.game else _game()
            if game:
                return f"a game is running ({game})"
        hot = [t for t in self.temps() if t >= HOT_C]
        if hot:
            return f"GPU is hot ({max(hot):.0f} C)"
        return None


def _game() -> str | None:
    from agent.governor import game_running

    return game_running()
