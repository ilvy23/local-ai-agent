"""WhatsApp GUI: overview, bridge control, sync / ingest / embedding jobs, the job list, and the plugin's numeric settings.

Mirrors `agent wa status|bridge|sync|merge-lids`. Sending is not reachable from here: the bridge client is used for health,
backfill and reading only (and it refuses to send unless AGENT_WA_ALLOW_SEND=1, which no route here ever calls anyway).
"""

from __future__ import annotations

from agent import brand

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.web import deps
from agent.web.deps import get_config
from agent.web.routes import whatsapp_core as core

router = APIRouter(prefix="/api/wa", tags=["whatsapp"])

_PLACEHOLDERS = "('[image]','[video]','[audio]','[document]','[sticker]','[contact]','[location]')"


def _pid_file(config: dict[str, Any]) -> Path:
    return core.db_path(config).parent / "wa-bridge.pid"


def spawn_bridge(bridge_dir: Path) -> int:
    """Start `node index.js` detached, output discarded (it may print chat events). Tests replace this."""
    p = subprocess.Popen(["node", "index.js"], cwd=bridge_dir, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    return p.pid


def _bridge_status(config: dict[str, Any]) -> dict[str, Any]:
    try:
        h = core.bridge_client(config).health()
        return {"reachable": True, "connected": bool(h.get("connected")), "paired": bool(h.get("paired")), "status": h.get("status", "")}
    except core.PluginMissing:
        return {"reachable": False, "error": "plugin missing"}
    except Exception as exc:  # noqa: BLE001 - BridgeError / HTTP errors: "not running" is a normal state
        return {"reachable": False, "error": str(exc)[:160]}


def _owned_pid(config: dict[str, Any]) -> int | None:
    """The pid of a bridge this app started, if it is still that process."""
    try:
        pid = int(_pid_file(config).read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, ValueError):
        return None
    return pid if "index.js" in cmd else None


def _heartbeat(config: dict[str, Any]) -> dict[str, Any] | None:
    try:
        d = json.loads((core.db_path(config).parent / "last_refresh.json").read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def auto_state(config: dict[str, Any]) -> dict[str, Any]:
    """Auto-reply as STATE only (config + the wa_autoreply log). Nothing here changes or triggers sending."""
    wa = config.get("whatsapp") or {}
    ac = wa.get("auto_reply") or {}
    loop = ac.get("loop") or {}
    day0 = int(time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1)))
    sent = core.ro_one(config, "SELECT COUNT(*) AS n FROM wa_autoreply WHERE status='sent' AND send_at >= ?", (day0,)) or {"n": 0}
    held = core.ro_one(config, "SELECT COUNT(*) AS n FROM wa_autoreply WHERE status IN ('held','scheduled')") or {"n": 0}
    return {"mode": ac.get("mode", "off"), "speed": ac.get("speed", "normal"), "daily_cap": ac.get("daily_cap", 0),
            "loop": bool(loop.get("enabled")), "contacts": len(wa.get("auto_reply_contacts") or []),
            "sent_today": sent["n"], "pending": held["n"]}


@router.get("/overview")
def overview(config=Depends(get_config)) -> dict[str, Any]:
    core.wa_config(config)
    inst = core.installed()
    counts = {k: (core.ro_one(config, f"SELECT COUNT(*) AS n FROM {t}") or {"n": 0})["n"] for k, t in (("messages", "wa_messages"), ("chats", "wa_chats"), ("contacts", "wa_contacts"))}
    last = core.ro_one(config, "SELECT v FROM wa_sync_state WHERE k='last_sync_at'")
    emb = core.ro_one(config, f"SELECT COUNT(*) AS total, COALESCE(SUM(i.id IS NULL), 0) AS missing FROM wa_messages m "
                              f"LEFT JOIN memory_items i ON i.kind='wa_message' AND i.ref_id = m.rowid "
                              f"WHERE m.text IS NOT NULL AND trim(m.text) != '' AND trim(m.text) NOT IN {_PLACEHOLDERS}") or {"total": 0, "missing": 0}
    running = [j for j in core.list_jobs() if j["state"] == "running"]
    return {"installed": inst, "hint": core.CLI_HINT, "send": core.send_state(),
            "bridge": _bridge_status(config) if inst else None, "bridge_owned": bool(inst and _owned_pid(config)),
            "counts": counts, "last_sync_at": (last or {}).get("v"),
            "embed": {"embeddable": emb["total"], "embedded": emb["total"] - emb["missing"], "missing": emb["missing"]},
            "refresh": _heartbeat(config), "auto": auto_state(config), "running": running[0] if running else None,
            "settings": {k: config["whatsapp"].get(k) for k in ("min_messages", "profile_max_messages", "summarize_batch")}}


# ------------------------------------------------------------------------------------------------ bridge


@router.post("/bridge/start")
def bridge_start(config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    core.wa_config(config)
    if _bridge_status(config)["reachable"]:
        raise HTTPException(409, "The bridge is already running.")
    bridge_dir = core.plugin_root() / "bridge"
    if not (bridge_dir / "index.js").is_file():
        raise HTTPException(404, f"No bridge found at {bridge_dir}.")
    try:
        pid = spawn_bridge(bridge_dir)
    except OSError as exc:
        raise HTTPException(500, f"Could not start node: {exc}") from exc
    _pid_file(config).write_text(str(pid))
    return {"started": True, "pid": pid, "note": f"If it is not paired yet, pairing needs the QR code: run `{brand.CMD} wa bridge` in a terminal."}


@router.post("/bridge/stop")
def bridge_stop(config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    pid = _owned_pid(config)
    if not pid:
        raise HTTPException(409, "This app did not start the running bridge. Stop it where it was started (terminal or service).")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        raise HTTPException(500, f"Could not stop it: {exc}") from exc
    _pid_file(config).unlink(missing_ok=True)
    return {"stopped": True}


# ------------------------------------------------------------------------------------------------ jobs


@router.get("/jobs")
def jobs() -> list[dict[str, Any]]:
    return core.list_jobs()


@router.get("/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    return core.get_job(job_id).view(full=True)


@router.post("/jobs/{job_id}/stop")
def job_stop(job_id: str) -> dict[str, Any]:
    j = core.get_job(job_id)
    if j.state != "running":
        raise HTTPException(409, "That job already finished.")
    core.stop_job(j)
    return j.view()


@router.delete("/jobs/{job_id}")
def job_dismiss(job_id: str) -> dict[str, bool]:
    j = core.get_job(job_id)
    if j.state == "running":
        raise HTTPException(409, "Stop the job first.")
    core._jobs.pop(job_id, None)
    return {"ok": True}


# ------------------------------------------------------------------------------------------------ sync / ingest / embed


class SyncIn(BaseModel):
    days: int | None = Field(None, ge=1, le=3650, description="Ask the bridge to backfill this many days first.")


class ConfirmIn(BaseModel):
    confirm: bool = False


def _sync_job(config, client, kind: str, label: str, *, days: int | None = None, embed: bool = True, reset_cursor: bool = False):
    core.require()
    core.wa_config(config)

    def work(job: core.Job) -> dict[str, Any]:
        from agent.memory.vectors import VectorIndex

        ing = core.mod("ingest")
        with core.open_store(config) as store:
            bridge = core.bridge_client(config, timeout=10.0)
            if days:
                job.say(f"asking the bridge for a {days}-day backfill")
                bridge.backfill(days)
            if reset_cursor:
                ing._set_state(store, "cursor", "0")
            job.say("reading from the bridge")
            r = ing.sync(store, VectorIndex(store), core.CountingLLM(client, job) if embed else client, config, bridge, embed=embed)
        return {"new_messages": r["new_messages"], "chats": r["chats"], "contacts": r["contacts"]}

    return core.start_job(kind, label, work).view()


@router.post("/sync")
def sync(body: SyncIn | None = None, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    """Pull new messages, contacts and chats from the bridge and embed them (`agent wa sync [--days N]`)."""
    return _sync_job(config, client, "sync", "Sync and embed", days=body.days if body else None)


@router.post("/sync/pull")
def pull(config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    """Pull without embedding (what the menu does at start); embed later."""
    return _sync_job(config, client, "pull", "Pull without embedding", embed=False)


@router.post("/sync/reingest")
def reingest(body: ConfirmIn, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    """Read the bridge from the very beginning again. Safe to repeat (messages are keyed by id), but it takes a while."""
    if not body.confirm:
        raise HTTPException(400, "Confirm the re-ingest first.")
    return _sync_job(config, client, "reingest", "Re-ingest everything", reset_cursor=True)


@router.post("/sync/embed")
def embed_backlog(config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    """Embed messages that have no vector yet (semantic search)."""
    core.require()
    core.wa_config(config)
    missing = overview(config)["embed"]["missing"]

    def work(job: core.Job) -> dict[str, Any]:
        from agent.memory.vectors import VectorIndex

        with core.open_store(config) as store:
            n = core.mod("ingest")._embed_backlog(store, VectorIndex(store), core.CountingLLM(client, job, missing or None), config)
        return {"embedded": n}

    return core.start_job("embed", "Embed messages", work).view()


@router.post("/sync/merge-lids")
def merge_lids(config=Depends(get_config)) -> dict[str, Any]:
    """Fold duplicate @lid chats into their phone-number contact (`agent wa merge-lids`)."""
    core.require()
    core.wa_config(config)
    lm = core.mod("lidmerge")
    refreshed = None
    with core.open_store(config) as store:
        try:
            refreshed = lm.store_lid_map(store, core.bridge_client(config).lid_map().get("lid_map", []))
        except Exception:  # noqa: BLE001 - bridge offline: merge with the pairs already stored
            refreshed = None
        res = lm.merge_lids(store)
    return {"merged": res["merged"], "pairs": res["pairs"], "refreshed_from_bridge": refreshed}


# ------------------------------------------------------------------------------------------------ settings


_INT_KEYS = {"min_messages": (1, 100), "profile_max_messages": (10, 5000), "summarize_batch": (10, 2000)}


class SettingIn(BaseModel):
    key: str
    value: int


@router.put("/config")
def set_config(body: SettingIn, config=Depends(get_config)) -> dict[str, Any]:
    """Change one of the plugin's numeric settings in config.yaml (`whatsapp.<key>`)."""
    if body.key not in _INT_KEYS:
        raise HTTPException(400, "Unknown setting.")
    lo, hi = _INT_KEYS[body.key]
    if not lo <= body.value <= hi:
        raise HTTPException(400, f"Must be between {lo} and {hi}.")
    core.update_config_yaml(config, lambda d: d.setdefault("whatsapp", {}).__setitem__(body.key, body.value))
    return {body.key: body.value}
