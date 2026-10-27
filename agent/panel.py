"""Live status panel — one screen: machine, Ollama, governor, data.

No psutil dependency: system stats come from /proc the same way the governor
reads game/GPU state. Refreshes in place until Ctrl-C.
"""

from __future__ import annotations

import contextlib
import os
import platform
import select
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_CYAN, _PINK, _GREEN, _PURPLE, _DIM = "#05d9e8", "#ff2a6d", "#39ff14", "#b967ff", "#5a5a8a"

# The panel refreshes every ~2s to keep CPU/RAM/temp live, but the DB counts and
# the Ollama query are heavier and barely change — cache them so a live panel
# doesn't reopen the store and poll Ollama every frame.
_SLOW_TTL_S = 10.0
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, fn):
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < _SLOW_TTL_S:
        return hit[1]
    val = fn()
    _cache[key] = (time.monotonic(), val)
    return val


def _cpu_times() -> tuple[int, int]:
    """(idle, total) jiffies from /proc/stat's first line; (0, 0) where /proc/stat is unreadable (Android blocks it for apps)."""
    try:
        with open("/proc/stat") as fh:
            parts = [int(x) for x in fh.readline().split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle + iowait
        return idle, sum(parts)
    except (OSError, ValueError, IndexError):
        return 0, 0


def _cpu_percent(prev: tuple[int, int]) -> tuple[float, tuple[int, int]]:
    """CPU busy % since `prev` sample. Returns (percent, new_sample)."""
    cur = _cpu_times()
    d_idle, d_total = cur[0] - prev[0], cur[1] - prev[1]
    pct = 0.0 if d_total <= 0 else 100.0 * (1 - d_idle / d_total)
    return pct, cur


def _mem() -> tuple[float, float]:
    """(used_gb, total_gb) from /proc/meminfo (MemAvailable-based); (0.0, 0.0) when it cannot be read."""
    from agent import sysinfo

    total_mb, avail_mb = sysinfo.ram_mb()
    if not total_mb:
        return 0.0, 0.0
    return (total_mb - avail_mb) / 1024, total_mb / 1024


def _loadavg() -> str:
    try:
        return "  ".join(Path("/proc/loadavg").read_text().split()[:3])
    except OSError:
        try:
            return "  ".join(f"{x:.2f}" for x in os.getloadavg())
        except (OSError, AttributeError):
            return "n/a"


def _ollama_models() -> list[str]:
    """Resident models via Ollama's HTTP /api/ps, best-effort.

    Uses the HTTP API (honoring OLLAMA_HOST) rather than the `ollama` binary, so
    it also works when pointed at a remote Ollama server."""
    import httpx

    from agent.llm import current_base_url, vulkan_base_url

    rows = []
    # The default server plus the dual-GPU Vulkan server (if it is running).
    for base in dict.fromkeys([current_base_url(), vulkan_base_url()]):
        try:
            data = httpx.get(f"{base}/api/ps", timeout=3).json()
        except (httpx.HTTPError, ValueError):
            continue
        for m in data.get("models", []):
            name = m.get("name") or m.get("model", "?")
            size = m.get("size_vram", 0) or m.get("size", 0)
            total = m.get("size", 1) or 1
            where = "GPU" if size >= total else ("CPU" if not size else "part")
            rows.append(f"{name}  {where}")
    return rows


def _disk(config) -> tuple[float, float]:
    """(used_gb, total_gb) for the filesystem holding the DB."""
    from agent.config import PROJECT_ROOT

    db = Path(config["data"]["db_path"])
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    target = db.parent if db.parent.exists() else PROJECT_ROOT
    total, used, _free = shutil.disk_usage(target)
    return used / 1e9, total / 1e9


def _cpu_temp() -> float | None:
    """Hottest thermal zone in °C, or None."""
    best = None
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            c = int(zone.read_text().strip()) / 1000
        except (OSError, ValueError):
            continue
        best = c if best is None else max(best, c)
    return best


def _uptime() -> str:
    try:
        secs = int(float(Path("/proc/uptime").read_text().split()[0]))
    except (OSError, ValueError):
        return "?"
    d, rem = divmod(secs, 86400)
    h, m = divmod(rem // 60, 60)
    return (f"{d}d " if d else "") + f"{h}h{m:02d}m"


def _bar(pct: float, width: int = 20) -> Text:
    filled = int(round(pct / 100 * width))
    colour = _GREEN if pct < 60 else (_PINK if pct > 85 else _PURPLE)
    t = Text()
    t.append("█" * filled, style=colour)
    t.append("░" * (width - filled), style=_DIM)
    t.append(f" {pct:4.0f}%", style=colour)
    return t


def _data_stats(config) -> dict:
    """Counts straight from the DB. Empty dict if it can't be opened."""
    from agent.main import _open_store

    try:
        with _open_store(config) as store:
            c = store.conn
            one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
            return {
                "sessions": one("SELECT COUNT(*) FROM sessions"),
                "messages": one("SELECT COUNT(*) FROM messages"),
                "facts": one("SELECT COUNT(*) FROM facts WHERE active = 1"),
            }
    except Exception:  # noqa: BLE001 - a locked/missing DB shouldn't kill the panel
        return {}


def _frame(config, cpu_pct: float) -> Panel:
    from agent.governor import game_running, gpu_utilization, pause_reason

    used, total = _mem()
    gpu = gpu_utilization()
    game = game_running()
    reason = pause_reason(config)
    stats = _cached("stats", lambda: _data_stats(config))
    disk_used, disk_total = _disk(config)
    temp = _cpu_temp()

    sysgrid = Table.grid(padding=(0, 2))
    sysgrid.add_column(style=_DIM, justify="right")
    sysgrid.add_column()
    sysgrid.add_row("CPU", _bar(cpu_pct))
    sysgrid.add_row("RAM", _bar(100 * used / total if total else 0))
    sysgrid.add_row("", Text(f"{used:.1f} / {total:.1f} GB", style=_DIM))
    if gpu is not None:
        sysgrid.add_row("GPU", _bar(float(gpu)))
    sysgrid.add_row("disk", _bar(100 * disk_used / disk_total if disk_total else 0))
    sysgrid.add_row("", Text(f"{disk_used:.0f} / {disk_total:.0f} GB", style=_DIM))
    if temp is not None:
        sysgrid.add_row("temp", Text(f"{temp:.0f}°C", style=_PINK if temp >= 80 else _CYAN))
    sysgrid.add_row("load", Text(_loadavg(), style=_CYAN))

    models = _cached("models", _ollama_models)
    ol = Text()
    if models:
        for m in models:
            ol.append(f"● {m}\n", style=_GREEN)
    else:
        ol.append("○ no model resident\n", style=_DIM)

    gov = Text()
    if reason:
        gov.append(f"⏸ PAUSED — {reason}\n", style=_PINK)
    else:
        gov.append("▶ clear — background work runs\n", style=_GREEN)
    gov.append(f"game: {game or 'none'}\n", style=_DIM)
    from agent.worker_status import read_activity

    act = read_activity(config)
    if act and act[1] < 120:  # only if recent enough to be "current"
        gov.append(f"▸ {act[0]}\n", style=_GREEN)

    dat = Table.grid(padding=(0, 2))
    dat.add_column(style=_DIM, justify="right")
    dat.add_column(style=_CYAN)
    if stats:
        dat.add_row("sessions", str(stats["sessions"]))
        dat.add_row("messages", f"{stats['messages']:,}")
        dat.add_row("facts", str(stats["facts"]))
    else:
        dat.add_row("data", "unavailable")

    body = Table.grid(padding=(0, 3))
    body.add_column()
    body.add_column()
    body.add_row(
        Panel(sysgrid, title=f"[{_CYAN}]MACHINE[/]", border_style=_DIM),
        Panel(dat, title=f"[{_CYAN}]DATA[/]", border_style=_DIM),
    )
    body.add_row(
        Panel(ol, title=f"[{_CYAN}]OLLAMA[/]", border_style=_DIM),
        Panel(gov, title=f"[{_CYAN}]GOVERNOR[/]", border_style=_DIM),
    )

    clock = time.strftime("%H:%M:%S")
    host = platform.node() or "node"
    return Panel(
        Group(body),
        border_style=_PINK,
        title=f"[{_GREEN}]AGENT · {host}[/]",
        subtitle=f"[{_DIM}]{clock} · up {_uptime()} · Ctrl-C to exit[/]",
    )


def snapshot(config, cpu_pct: float) -> dict:
    """Machine-readable state — the same numbers the panel draws. For --json,
    cron alerts, or a remote e-ink/OLED status display."""
    import platform

    from agent.governor import game_running, gpu_utilization, pause_reason

    used, total = _mem()
    disk_used, disk_total = _disk(config)
    return {
        "host": platform.node(),
        "at": int(time.time()),
        "cpu_pct": round(cpu_pct, 1),
        "ram_used_gb": round(used, 1),
        "ram_total_gb": round(total, 1),
        "disk_used_gb": round(disk_used, 1),
        "disk_total_gb": round(disk_total, 1),
        "gpu_pct": gpu_utilization(),
        "cpu_temp_c": _cpu_temp(),
        "game": game_running(),
        "paused": pause_reason(config),
        "ollama_models": _ollama_models(),
        "data": _data_stats(config),
    }


# ---------------------------------------------------------------------------
# Live agent-status dashboard
# ---------------------------------------------------------------------------

_STATUS_COLORS: dict[str, str] = {
    "running": "green",
    "idle": "dim",
    "waiting_review": "yellow bold",
    "done": "dim",
}


class StatusPanel:
    """Live Rich dashboard: agent-status table + log tail.

    Usage::

        panel = StatusPanel()
        with panel.live():
            # your code updates StatusRegistry and trajectory log
            pass

    The dashboard refreshes itself at *refresh_rate* seconds in the background
    while the ``with`` body runs.  After the body exits the live display stops.

    Key bindings (stdin, best-effort — requires a real tty):
      ``j``  approve the first agent waiting for review
      ``n``  reject  the first agent waiting for review
      ``q``  kill the whole process (SIGTERM)

    The ``review_approved`` / ``review_rejected`` :class:`threading.Event` s
    are set when the corresponding key is pressed.
    """

    def __init__(self, log_lines: int = 10, refresh_rate: float = 0.5) -> None:
        self._log_lines = log_lines
        self._refresh_rate = refresh_rate
        self.review_approved: threading.Event = threading.Event()
        self.review_rejected: threading.Event = threading.Event()
        self._stop_event: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """Build the current Rich renderable (Table + log tail Panel + hint)."""
        from agent.trajectory import read_recent
        from agent.worker_status import StatusRegistry

        registry = StatusRegistry.get()
        agents = registry.all_agents()

        # --- Agent status table ---
        table = Table(title="Agent Status", border_style=_DIM, expand=True)
        table.add_column("Agent", style=_CYAN, no_wrap=True)
        table.add_column("Milestone")
        table.add_column("Last Action")
        table.add_column("Status", no_wrap=True)
        table.add_column("Review?", no_wrap=True, justify="center")

        for agent in agents:
            color = _STATUS_COLORS.get(agent.status, "white")
            row_style = "yellow" if agent.status == "waiting_review" else ""
            table.add_row(
                agent.agent_id,
                agent.milestone,
                agent.last_action,
                Text(agent.status, style=color),
                Text("⏳", style="yellow") if agent.review_pending else Text(""),
                style=row_style,
            )

        # --- Log tail ---
        records = read_recent(self._log_lines)
        log_text = Text()
        for rec in records:
            ts = (rec.get("ts") or "")[:19]
            tool = rec.get("tool") or rec.get("action") or "?"
            inp = rec.get("input") or ""
            out = rec.get("output") or ""
            log_text.append(f"[{ts}] {tool}: {inp} → {out}\n", style=_DIM)
        if not records:
            log_text.append("(no log entries yet)", style=_DIM)

        log_panel = Panel(
            log_text,
            title=f"[{_CYAN}]Live Log[/]",
            border_style=_DIM,
        )

        # --- Hint line ---
        hint = Text("j = approve  n = reject  q = kill all", style=_DIM)

        return Group(table, log_panel, hint)

    # ------------------------------------------------------------------
    # Key-handling helpers
    # ------------------------------------------------------------------

    def _handle_approve(self) -> None:
        from agent.worker_status import StatusRegistry

        registry = StatusRegistry.get()
        for agent in registry.all_agents():
            if agent.review_pending:
                registry.clear_review(agent.agent_id)
                self.review_approved.set()
                break

    def _handle_reject(self) -> None:
        from agent.worker_status import StatusRegistry

        registry = StatusRegistry.get()
        for agent in registry.all_agents():
            if agent.review_pending:
                registry.clear_review(agent.agent_id)
                self.review_rejected.set()
                break

    def _read_keys(self) -> None:
        """Background daemon thread: read stdin and dispatch j / n / q.

        Uses ``select`` so it never blocks the refresh loop.  Silently exits
        if stdin is not readable (e.g. a pipe or in tests).
        """
        try:
            while not self._stop_event.is_set():
                try:
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable:
                        ch = sys.stdin.read(1)
                        if not ch:
                            break  # EOF / stdin closed
                        if ch == "q":
                            os.kill(os.getpid(), signal.SIGTERM)
                        elif ch == "j":
                            self._handle_approve()
                        elif ch == "n":
                            self._handle_reject()
                except (OSError, ValueError, AttributeError):
                    break  # stdin not available or closed — give up quietly
        except Exception:  # noqa: BLE001 — key reader must never crash the dashboard
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def live(self):
        """Run a live Rich dashboard that auto-refreshes while the body executes."""
        self._stop_event.clear()
        self.review_approved.clear()
        self.review_rejected.clear()

        refresh_per_sec = max(1, int(1.0 / max(self._refresh_rate, 0.05)))

        with Live(
            self.render(),
            refresh_per_second=refresh_per_sec,
            auto_refresh=False,
        ) as live_ctx:
            # Background thread: push fresh renders to the live display.
            def _refresh_loop() -> None:
                while not self._stop_event.is_set():
                    try:
                        live_ctx.update(self.render())
                        live_ctx.refresh()
                    except Exception:  # noqa: BLE001
                        pass
                    self._stop_event.wait(self._refresh_rate)

            refresh_thread = threading.Thread(target=_refresh_loop, daemon=True)
            refresh_thread.start()

            # Background thread: listen for keyboard input.
            key_thread = threading.Thread(target=self._read_keys, daemon=True)
            key_thread.start()

            try:
                yield self
            finally:
                self._stop_event.set()



def run_panel(refresh_s: float = 2.0, once: bool = False, as_json: bool = False) -> None:
    from agent.config import load_config

    config = load_config()
    prev = _cpu_times()
    time.sleep(0.2)
    pct, prev = _cpu_percent(prev)
    if as_json:
        import json

        print(json.dumps(snapshot(config, pct)))
        return
    if once:
        Console().print(_frame(config, pct))
        return
    with Live(_frame(config, pct), refresh_per_second=4, screen=False) as live:
        try:
            while True:
                time.sleep(refresh_s)
                pct, prev = _cpu_percent(prev)
                live.update(_frame(config, pct))
        except KeyboardInterrupt:
            pass
