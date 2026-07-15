"""The system_stats tool: a read-only snapshot of machine resource usage.

Stdlib only (no psutil): /proc/meminfo for RAM, shutil.disk_usage for disks,
os.getloadavg for load, and `ps` for the top memory consumers. Everything is
best-effort — any unavailable metric degrades to a note rather than raising, so
the handler always returns a readable string.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from companion.tools.registry import Tool

_GB = 1024**3


def _gb(n: int) -> str:
    return f"{n / _GB:.1f} GB"


def _ram_line() -> str:
    try:
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.strip().split()[0]) * 1024  # kB -> bytes
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - available
        pct = (used / total * 100) if total else 0
        return f"RAM: {_gb(used)} used / {_gb(total)} total ({pct:.0f}%)"
    except (OSError, ValueError, IndexError):
        return "RAM: (unavailable)"


def _disk_line(label: str, path: str) -> str:
    try:
        usage = shutil.disk_usage(path)
        pct = usage.used / usage.total * 100 if usage.total else 0
        return f"Disk {label} ({path}): {_gb(usage.used)} used / {_gb(usage.total)} total ({pct:.0f}%)"
    except OSError:
        return f"Disk {label} ({path}): (unavailable)"


def _load_line() -> str:
    try:
        one, five, fifteen = os.getloadavg()
        return f"Load average: {one:.2f} {five:.2f} {fifteen:.2f}"
    except (OSError, AttributeError):
        return "Load average: (unavailable)"


def _top_processes() -> str:
    try:
        out = subprocess.run(
            ["ps", "axo", "pid,pmem,comm", "--sort=-%mem"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        lines = out.strip().splitlines()
        top = lines[: 5 + 1]  # header + 5 rows
        return "Top processes by memory:\n" + "\n".join("  " + line.strip() for line in top)
    except (OSError, subprocess.SubprocessError):
        return "Top processes: (unavailable)"


def system_stats(**_kwargs: Any) -> str:
    """Return a readable multi-line snapshot of RAM, disk, load, and top procs."""
    home = os.path.expanduser("~")
    parts = [
        _ram_line(),
        _disk_line("/", "/"),
        _disk_line("home", home),
        _load_line(),
        _top_processes(),
    ]
    return "\n".join(parts)


SYSTEM_STATS_TOOL = Tool(
    name="system_stats",
    description="Report current RAM, disk usage, load average, and top processes.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=system_stats,
    risk="safe",
)
