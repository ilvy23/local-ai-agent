"""Real hardware stats for the "Host status" page - GPU via nvidia-smi, CPU
temp via the kernel's thermal zones, RAM/CPU load/network/uptime via the
standard library (no extra dependency needed for any of it). Best-effort
throughout: any piece that isn't available is simply omitted from the
response rather than failing the whole request.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/host", tags=["host"])

_NVIDIA_SMI_TIMEOUT_S = 5
_START_TIME = time.time()


def _gpu_stats() -> dict | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                (
                    "--query-gpu=name,memory.used,memory.total,temperature.gpu,"
                    "fan.speed,utilization.gpu,power.draw,power.limit"
                ),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S, check=True,
        )
        name, mem_used, mem_total, temp, fan, util, power, power_limit = (
            p.strip() for p in proc.stdout.strip().split(",")
        )
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return {
        "name": name,
        "vram_used_mb": round(float(mem_used)),
        "vram_total_mb": round(float(mem_total)),
        "temp_c": round(float(temp)),
        "fan_percent": round(float(fan)) if fan not in ("", "N/A") else None,
        "utilization_percent": round(float(util)),
        "power_watts": round(float(power), 1),
        "power_limit_watts": round(float(power_limit)) if power_limit not in ("", "N/A") else None,
    }


def _cpu_temp_c() -> float | None:
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            millidegrees = int((zone / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        return round(millidegrees / 1000, 1)
    return None


def _cpu_load_percent() -> float | None:
    try:
        load_1min = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
    except (OSError, AttributeError):
        return None
    return round(min(100.0, load_1min / cpu_count * 100), 1)


def _ram_stats() -> dict | None:
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = int(rest.strip().split()[0])  # kB
        total_kb = fields["MemTotal"]
        available_kb = fields["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        return None
    used_kb = total_kb - available_kb
    return {
        "used_gb": round(used_kb / 1024 / 1024, 1),
        "total_gb": round(total_kb / 1024 / 1024, 1),
    }


def _local_ip() -> str | None:
    """The IP this machine would use to reach the outside world on its real
    LAN interface - deliberately not the Tailscale interface. Opens no
    actual connection (UDP, connect() just picks a route/source address)."""
    try:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _uptime_str() -> str:
    seconds = int(time.time() - _START_TIME)
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


@router.get("/stats")
def host_stats() -> dict:
    return {
        "gpu": _gpu_stats(),
        "cpu_temp_c": _cpu_temp_c(),
        "cpu_load_percent": _cpu_load_percent(),
        "ram": _ram_stats(),
        "local_ip": _local_ip(),
        "uptime": _uptime_str(),
    }


@router.post("/suspend")
def suspend_host() -> dict:
    """Suspends this machine on request (the manual counterpart to the
    idle-auto-suspend timer). Fires a DETACHED, delayed shell command rather
    than calling `systemctl suspend` inline: suspending pauses this very
    process along with everything else, so a synchronous call here would
    leave the HTTP response (and the relay proxying it back to the phone)
    hanging unanswered until the NEXT wake - the 1s delay is just long
    enough for this response to actually leave the machine first. No sudo
    needed: systemd-logind grants the active local user permission to
    suspend without elevation, which is also why the CLI equivalent
    (`systemctl suspend`) never needed it earlier in this same setup."""
    if shutil.which("systemctl") is None:
        raise HTTPException(501, "Suspend needs systemd (systemctl), which is not available on this machine.")
    subprocess.Popen(["sh", "-c", "sleep 1 && systemctl suspend"])
    return {"suspending": True}
