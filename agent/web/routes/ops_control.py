"""Operations screen: governor, worker control, panel snapshot, log tails, backup and DB-sync status.

Everything here reuses the product's own modules (governor, panel, worker_status) and the same `systemctl --user` calls as
`agent worker start|stop`. Backup and sync are report-only: the product has no backup command, and sync is a pair of shell
scripts the user runs by hand; nothing in this module starts either.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from agent import governor, panel, worker_status
from agent.config import PROJECT_ROOT
from agent.web.deps import get_config
from agent.web.routes import host, ops

router = APIRouter(prefix="/api/ops", tags=["ops"])

# journal sources the Logs tab may read; the caller picks an id, never a unit name
LOG_SOURCES: dict[str, dict[str, Any]] = {
    "worker": {"label": "Background worker", "unit": ops.WATCH_UNIT, "user": True},
    "model-server": {"label": "Dual-GPU model server", "unit": "ollama-vk.service", "user": True},
    "ollama": {"label": "Ollama (system)", "unit": "ollama.service", "user": False},
}
_LOG_MAX = 500


# --------------------------------------------------------------------------------------------------- governor


@router.get("/governor")
def governor_state(config=Depends(get_config)) -> dict[str, Any]:
    bg = config.get("background", {})
    used, total = panel._mem()
    try:
        paused = governor.pause_reason(config)
    except Exception:  # noqa: BLE001
        paused = None
    return {"gpu_percent": governor.gpu_utilization(), "cpu_percent": host._cpu_load_percent(), "ram_used_gb": round(used, 1),
            "ram_total_gb": round(total, 1), "game": governor.game_running(), "paused_reason": paused,
            "caps": {"gpu": bg.get("max_gpu_percent"), "cpu": bg.get("max_cpu_percent"), "ram_free_gb": bg.get("max_ram_gb"),
                     "pause_on_game": bool(bg.get("pause_on_game", True))}}


# --------------------------------------------------------------------------------------------------- worker


def _journal(unit: str, user: bool, lines: int = 40) -> list[dict[str, Any]]:
    """Last journal lines of one unit as {ts, lv, t}; [] when journalctl or the unit is not there."""
    cmd = ["journalctl", *(["--user"] if user else []), "-u", unit, "-n", str(max(1, min(lines, _LOG_MAX))), "--no-pager", "-o", "json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for raw in out.splitlines():
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        msg = e.get("MESSAGE")
        if not isinstance(msg, str):
            continue
        try:
            prio = int(e.get("PRIORITY", 6))
        except (TypeError, ValueError):
            prio = 6
        ts = int(e.get("__REALTIME_TIMESTAMP", 0) or 0) / 1e6
        rows.append({"ts": ts or None, "lv": "E" if prio <= 3 else "W" if prio == 4 else "I", "t": msg})
    return rows


def _worker_state(config: dict[str, Any]) -> dict[str, Any]:
    try:
        paused = governor.pause_reason(config)
    except Exception:  # noqa: BLE001
        paused = None
    w = ops._worker(config, paused)
    w["enabled"] = ops._systemctl("is-enabled", ops.WATCH_UNIT) or None
    return w


@router.get("/worker")
def worker(config=Depends(get_config)) -> dict[str, Any]:
    return {**_worker_state(config), "journal": _journal(ops.WATCH_UNIT, True, 12)}


@router.post("/worker/{action}")
def worker_control(action: str, config=Depends(get_config)) -> dict[str, Any]:
    if action not in ("start", "stop"):
        raise HTTPException(404, "action must be start or stop.")
    try:
        r = subprocess.run(["systemctl", "--user", action, ops.WATCH_UNIT], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        raise HTTPException(503, "systemctl is not available on this machine.") from None
    if r.returncode != 0:
        raise HTTPException(502, (r.stderr or r.stdout or f"systemctl {action} failed").strip())
    return _worker_state(config)


# --------------------------------------------------------------------------------------------------- panel


@router.get("/panel")
def panel_snapshot(config=Depends(get_config)) -> dict[str, Any]:
    snap = panel.snapshot(config, host._cpu_load_percent() or 0.0)
    snap["uptime"] = panel._uptime()
    snap["gpu"] = host._gpu_stats()
    return snap


# --------------------------------------------------------------------------------------------------- logs


@router.get("/logs")
def logs(source: str = "worker", level: str = "all", q: str = "", lines: int = 200) -> dict[str, Any]:
    src = LOG_SOURCES.get(source)
    if src is None:
        raise HTTPException(404, "Unknown log source.")
    if level not in ("all", "I", "W", "E"):
        raise HTTPException(422, "level must be all, I, W or E.")
    rows = _journal(src["unit"], src["user"], _LOG_MAX if (q or level != "all") else lines)
    if level == "W":
        rows = [r for r in rows if r["lv"] in ("W", "E")]
    elif level == "E":
        rows = [r for r in rows if r["lv"] == "E"]
    if q:
        rows = [r for r in rows if q.lower() in r["t"].lower()]
    return {"source": source, "label": src["label"], "sources": [{"id": k, "label": v["label"]} for k, v in LOG_SOURCES.items()],
            "lines": rows[-max(1, min(lines, _LOG_MAX)):]}


# --------------------------------------------------------------------------------------------------- backup / sync (report only)


def _stat(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        return {"path": str(path), "bytes": st.st_size, "modified": st.st_mtime, "exists": True}
    except OSError:
        return {"path": str(path), "bytes": 0, "modified": None, "exists": False}


def _data_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    db = Path(config["data"]["db_path"])
    db = db if db.is_absolute() else PROJECT_ROOT / db
    names = [db, db.with_name(db.name + "-wal"), db.parent / "vectors", db.parent / "worker_activity.json"]
    return [s for s in (_stat(p) for p in names) if s["exists"]]


@router.get("/backup")
def backup(config=Depends(get_config)) -> dict[str, Any]:
    return {"available": False, "reason": "The product has no backup command yet. Copy the data folder yourself while the worker is stopped.",
            "files": _data_files(config), "now": time.time()}


@router.get("/sync")
def sync(config=Depends(get_config)) -> dict[str, Any]:
    bindir = Path(os.environ.get("XDG_BIN_HOME") or Path.home() / ".local" / "bin")
    try:
        scripts = sorted([*bindir.glob("*-pull-db.sh"), *bindir.glob("*-push-db.sh")])
    except OSError:
        scripts = []
    return {"available": bool(scripts), "reason": None if scripts else "No sync scripts found in ~/.local/bin.",
            "scripts": [{"name": p.name, **_stat(p)} for p in scripts], "database": _stat(Path(config["data"]["db_path"]) if Path(config["data"]["db_path"]).is_absolute() else PROJECT_ROOT / config["data"]["db_path"]),
            "now": time.time()}
