"""Live status panel — one screen: machine, Ollama, governor, data.

No psutil dependency: system stats come from /proc the same way the governor
reads game/GPU state. Refreshes in place until Ctrl-C.
"""

from __future__ import annotations

import platform
import shutil
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
    """(idle, total) jiffies from /proc/stat's first line."""
    with open("/proc/stat") as fh:
        parts = [int(x) for x in fh.readline().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle + iowait
    return idle, sum(parts)


def _cpu_percent(prev: tuple[int, int]) -> tuple[float, tuple[int, int]]:
    """CPU busy % since `prev` sample. Returns (percent, new_sample)."""
    cur = _cpu_times()
    d_idle, d_total = cur[0] - prev[0], cur[1] - prev[1]
    pct = 0.0 if d_total <= 0 else 100.0 * (1 - d_idle / d_total)
    return pct, cur


def _mem() -> tuple[float, float]:
    """(used_gb, total_gb) from /proc/meminfo (MemAvailable-based)."""
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, rest = line.partition(":")
            info[k] = int(rest.split()[0])  # kB
    total = info["MemTotal"] / 1024 / 1024
    avail = info.get("MemAvailable", info["MemFree"]) / 1024 / 1024
    return total - avail, total


def _loadavg() -> str:
    return "  ".join(Path("/proc/loadavg").read_text().split()[:3])


def _ollama_models() -> list[str]:
    """Resident models via Ollama's HTTP /api/ps, best-effort.

    Uses the HTTP API (honoring OLLAMA_HOST) rather than the `ollama` binary, so
    it also works when pointed at a remote Ollama server."""
    import httpx

    from agent.llm import DEFAULT_BASE_URL

    try:
        data = httpx.get(f"{DEFAULT_BASE_URL}/api/ps", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        return []
    rows = []
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
