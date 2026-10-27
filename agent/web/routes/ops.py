"""Read-only operations views for the Status and Code-agent screens.

Nothing here starts, stops or loads anything: the worker is only asked `systemctl --user is-active`, the model servers
only `GET /api/version` and `/api/ps`, and a code run is rebuilt from what it already saved (session file + event log).
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException

from agent import gpumode, governor, llm, worker_status
from agent.codeagent import sessionstore
from agent.web.deps import get_config
from agent.web.routes import workspace

router = APIRouter(prefix="/api", tags=["ops"])

WATCH_UNIT = "agent-watch.service"        # the unit `agent worker start|stop` controls
_PROBE_TIMEOUT_S = 1.5
_TIMELINE_MAX = 300


# --------------------------------------------------------------------------------------------------- status detail


def _systemctl(*args: str) -> str:
    """Output of a read-only `systemctl --user ...` ("" when systemd is not there)."""
    try:
        r = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=5, check=False)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _probe(url: str) -> dict[str, Any]:
    """Is an Ollama server answering, and what does it hold in memory? Never starts anything."""
    try:
        with httpx.Client(base_url=url, timeout=_PROBE_TIMEOUT_S) as c:
            if c.get("/api/version").status_code != 200:
                return {"reachable": False, "models": []}
            try:
                ps = c.get("/api/ps").json().get("models", [])
            except (httpx.HTTPError, ValueError, AttributeError):
                ps = []
    except httpx.HTTPError:
        return {"reachable": False, "models": []}
    return {"reachable": True, "models": [{"name": str(m.get("name")), "vram_mb": int(m.get("size_vram") or 0) // (1024 * 1024)}
                                          for m in ps if isinstance(m, dict) and m.get("name")]}


def _worker(config: dict[str, Any], paused: str | None) -> dict[str, Any]:
    active = _systemctl("is-active", WATCH_UNIT)
    if not active:
        state = "unknown"
    elif active == "active":
        state = "paused" if paused else "running"
    else:
        state = "stopped"
    act = worker_status.read_activity(config)
    return {"name": "Background worker", "unit": WATCH_UNIT, "state": state, "systemd": active or None,
            "reason": paused if state == "paused" else None,
            "activity": act[0] if act else None, "activity_ago_s": act[1] if act else None}


@router.get("/status/detail")
def status_detail(config=Depends(get_config)) -> dict[str, Any]:
    try:
        paused = governor.pause_reason(config)
    except Exception:  # noqa: BLE001 - a failed probe is "no reason", not a 500
        paused = None
    try:
        game = governor.game_running()
    except Exception:  # noqa: BLE001
        game = None
    servers = [{"id": "default", "name": "default", "url": llm.current_base_url(), **_probe(llm.current_base_url())},
               {"id": "dual-gpu", "name": "dual-GPU (Vulkan)", "url": llm.vulkan_base_url(), **_probe(llm.vulkan_base_url())}]
    return {"paused_reason": paused, "game": game, "gpu_mode": gpumode.current_mode(),
            "worker": _worker(config, paused), "servers": servers}


# --------------------------------------------------------------------------------------------------- code run timeline

_CHECK_TONE = {"pass": ("ok", "Check passed"), "fail": ("danger", "Check failed")}
_STOP_WORDS = {
    "budget": "It used its step budget and is waiting to be told to go on.",
    "loop": "It stopped because it kept repeating itself.",
    "verify_budget": "It stopped after too many verification calls.",
    "stage_cap": "A stage hit its size cap.",
    "fresh_context": "It stopped to restart with a fresh context.",
    "announce": "It stopped after announcing work without doing it.",
    "gate": "It stopped with acceptance items still open.",
    "error": "It stopped on an error.",
    "interrupted": "It was interrupted.",
    "killed": "The process was killed while working.",
    "unknown": "It stopped early (reason not recorded).",
}


def _item(ts: float | None, created: float, kind: str, tone: str, title: str, detail: str = "") -> dict[str, Any]:
    return {"ts": ts, "t": None if ts is None else max(0.0, round(ts - created, 1)), "kind": kind, "tone": tone, "title": title, "detail": detail}


def build_timeline(data: dict[str, Any], events: list[dict[str, Any]], running: bool = False) -> dict[str, Any]:
    """Steps, edits, checks and stage changes of a saved run, oldest first, with summary numbers."""
    created = float(data.get("created") or 0)
    steps = sorted((int(e["n"]), float(e["ts"])) for e in events if e.get("kind") == "step" and isinstance(e.get("n"), int) and e.get("ts"))

    def ts_of(step: Any) -> float | None:           # the ledger knows steps, not times: borrow the time of that step's log line
        best = None
        for n, ts in steps:
            if isinstance(step, int) and n <= step:
                best = ts
        return best or (created or None)

    items: list[dict[str, Any]] = []
    ledger = data.get("ledger") or {}
    files = ledger.get("files") or {}
    for path, e in files.items():
        e = e if isinstance(e, dict) else {}
        items.append(_item(ts_of(e.get("step")), created, "edit", "info", str(e.get("action") or "edited").capitalize(),
                           f"{path}" + (f" · {e['lines']} lines" if e.get("lines") else "")))
    failed_runs = 0
    for run in ledger.get("runs") or []:
        if not isinstance(run, (list, tuple)) or len(run) != 4:
            continue
        step, cmd, code, tail = run
        if code not in (0, None):
            failed_runs += 1
        tone, title = ("ok", "Command passed") if code == 0 else ("neutral", "Command ran") if code is None else ("danger", "Command failed")
        items.append(_item(ts_of(step), created, "run", tone, title, f"{cmd} · exit {code if code is not None else '?'}" + (f" · {tail}" if code else "")))
    failed_checks = 0
    for e in events:
        kind = e.get("kind")
        ts = e.get("ts")
        if kind == "check":
            tone, title = _CHECK_TONE.get(str(e.get("status")), ("neutral", f"Check {e.get('status')}"))
            failed_checks += e.get("status") == "fail"
            items.append(_item(ts, created, "check", tone, title, f"{e.get('check_kind', '')} · {e.get('id', '')}".strip(" ·")))
        elif kind == "compact":
            items.append(_item(ts, created, "context", "neutral", "Context compacted", f"{e.get('before')} → {e.get('after')} tokens"))
        elif kind == "budget_extend":
            items.append(_item(ts, created, "budget", "neutral", "Step budget extended", f"+{e.get('steps')} steps while making progress"))
        elif kind == "model_error":
            items.append(_item(ts, created, "model", "warn", "Model error", str(e.get("error") or "")[:200]))
        elif kind == "autopilot_end":
            items.append(_item(ts, created, "end", "ok" if e.get("verdict") == "done" else "warn", "Autopilot ended", str(e.get("reason") or e.get("verdict") or "")))
    stages = (data.get("stages") or {}).get("stages") or []
    for st in stages if isinstance(stages, list) else []:
        if not isinstance(st, dict):
            continue
        label = f"Stage {st.get('id')}: {st.get('title')}"
        if st.get("started_at"):
            items.append(_item(float(st["started_at"]), created, "stage", "info", "Stage started", label))
        if st.get("status") == "done" and st.get("finished_at"):
            items.append(_item(float(st["finished_at"]), created, "stage", "ok", "Stage done", label))
        elif st.get("status") == "blocked":
            items.append(_item(float(st.get("finished_at") or 0) or None, created, "stage", "warn", "Stage blocked", f"{label} — {st.get('reason') or 'no reason recorded'}"))
    items.sort(key=lambda i: (i["ts"] is None, i["ts"] or 0))

    ctx = data.get("ctx_state") or {}
    plan = data.get("plan") or {}
    plan_items = plan.get("items") or []
    stopped = str((data.get("pending") or {}).get("stopped") or "")
    progress = ({"label": "Stages done", "done": sum(1 for s in stages if isinstance(s, dict) and s.get("status") == "done"), "total": len(stages)} if stages
                else {"label": "Acceptance items", "done": sum(1 for i in plan_items if i.get("done")), "total": len(plan_items)} if plan_items else None)
    return {
        "items": items[-_TIMELINE_MAX:],
        "summary": {"edits": len(files), "failed_checks": failed_checks, "failed_commands": failed_runs, "steps": int(data.get("step") or 0),
                    "elapsed_s": max(0.0, float(data.get("updated") or 0) - created) if created else None, "running": running,
                    "context": {"used": int(ctx.get("last_prompt") or 0), "limit": int(data.get("num_ctx") or 0)}, "progress": progress},
        "decision": {"stopped": stopped, "text": _STOP_WORDS.get(stopped, "It stopped early." if stopped else ""), "task": (data.get("pending") or {}).get("task", "")}
        if stopped and data.get("state") != "finished" and not running else None,
    }


@router.get("/code/sessions/{session_id}/timeline")
def code_timeline(session_id: str) -> dict[str, Any]:
    meta = workspace._find(session_id)
    detail = workspace.code_session(session_id)
    try:
        data = sessionstore.load(meta.path)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    events = workspace.code_events(session_id, limit=workspace._EVENTS_MAX)["events"]
    return {**{k: detail[k] for k in ("id", "root", "title", "model", "state", "created", "updated")}, "now": time.time(),
            **build_timeline(data, events, running=detail["state"] == "in progress")}
