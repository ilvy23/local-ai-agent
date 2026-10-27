"""Machine-level surfaces for the GUI: status, GPU mode, folder access and source-trust rules."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent import gpumode, governor, repoaccess
from agent.research import trust
from agent.web.deps import get_client, get_config

router = APIRouter(prefix="/api", tags=["system"])


# --------------------------------------------------------------------------------------------------- status


@router.get("/status")
def status(client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    try:
        loaded = client.loaded_models()
    except Exception:  # noqa: BLE001 - an unreachable server is a status, not an error
        loaded = []
    try:
        paused = governor.pause_reason(config)
    except Exception:  # noqa: BLE001
        paused = None
    return {"loaded_models": loaded, "paused_reason": paused, "game_detected": bool(governor.game_running()),
            "gpu_mode": gpumode.current_mode(), "chat_model": config["models"]["chat"]}


# --------------------------------------------------------------------------------------------------- GPU mode


class GpuIn(BaseModel):
    mode: str
    ctx: int | None = None


def _estimate(e: gpumode.Estimate | None) -> dict[str, Any] | None:
    return dataclasses.asdict(e) if e else None


@router.get("/gpu")
def gpu(client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    gpus = gpumode.detect_gpus()
    resident = client.vram_in_use_mb() if hasattr(client, "vram_in_use_mb") else 0
    options = gpumode.build_options(gpus, resident)
    rows = gpumode.table_rows(client, config, options)
    return {
        "gpus": [dataclasses.asdict(g) for g in gpus],
        "options": [{"mode": o.mode, "label": o.label, "usable_gb": round(o.usable_mb / 1024, 1), "kv_type": o.kv_type} for o in options],
        "table": [{"model": m, "estimates": [_estimate(e) for e in ests]} for m, ests in rows],
        "chosen": {"mode": gpumode.current_mode(), "ctx": gpumode.chat_ctx_override() or config["models"].get("chat_num_ctx")},
        "last_mode": gpumode.last_mode(),
    }


@router.put("/gpu")
def set_gpu(body: GpuIn) -> dict[str, Any]:
    if body.mode not in (gpumode.SINGLE, gpumode.DUAL):
        raise HTTPException(422, "mode must be single or dual.")
    ctx = max(2048, min(int(body.ctx), 262144)) if body.ctx else None
    gpumode._apply(body.mode, ctx)
    gpumode._remember(body.mode)
    return {"mode": body.mode, "ctx": ctx}


# --------------------------------------------------------------------------------------------------- folder access


class AccessIn(BaseModel):
    path: str
    level: str          # ask | medium | auto | greenlist | ungreenlist


@router.get("/access")
def access() -> dict[str, Any]:
    return {"greenlist": [str(p) for p in repoaccess.greenlist()],
            "session": [{"path": str(f), "level": m} for f, m in sorted(repoaccess._session.items(), key=lambda kv: str(kv[0]))]}


@router.put("/access")
def set_access(body: AccessIn) -> dict[str, Any]:
    if body.level not in ("ask", "medium", "auto", "greenlist", "ungreenlist"):
        raise HTTPException(422, "level must be ask, medium, auto, greenlist or ungreenlist.")
    handled = repoaccess.handle_command(f"/{body.level}-{body.path}")
    if handled is None:
        raise HTTPException(422, "Not a valid access request.")
    _kind, message = handled
    if message.startswith(f"{body.level}:"):                      # handle_command's way of saying the folder was refused
        raise HTTPException(422, message.split(":", 1)[1].strip())
    return {"ok": True, "message": message, **access()}


# --------------------------------------------------------------------------------------------------- source trust


class RuleIn(BaseModel):
    domain: str
    action: str         # allow | deny | tier | clear
    tier: str | None = None


def _rules_doc(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    return {"allow": list(data.get("allow") or []), "deny": list(data.get("deny") or []), "tiers": dict(data.get("tiers") or {})}


@router.get("/trust/rules")
def get_rules() -> dict[str, Any]:
    return _rules_doc(trust.RULES_PATH)


@router.put("/trust/rules")
def put_rule(body: RuleIn) -> dict[str, Any]:
    domain = trust.registrable(body.domain.strip().lower().removeprefix("https://").removeprefix("http://").split("/")[0])
    if "." not in domain:
        raise HTTPException(422, "Give a domain such as example.org.")
    if body.action not in ("allow", "deny", "tier", "clear"):
        raise HTTPException(422, "action must be allow, deny, tier or clear.")
    if body.action == "tier" and body.tier not in trust.TIER_LEVEL:
        raise HTTPException(422, "tier must be T1, T2, T3 or T4.")
    doc = _rules_doc(trust.RULES_PATH)
    doc["allow"] = [d for d in doc["allow"] if trust.registrable(str(d)) != domain]
    doc["deny"] = [d for d in doc["deny"] if trust.registrable(str(d)) != domain]
    doc["tiers"] = {k: v for k, v in doc["tiers"].items() if trust.registrable(str(k)) != domain}
    if body.action == "allow":
        doc["allow"].append(domain)
    elif body.action == "deny":
        doc["deny"].append(domain)
    elif body.action == "tier":
        doc["tiers"][domain] = body.tier
    trust.RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    trust.RULES_PATH.write_text(yaml.safe_dump(doc, sort_keys=True))
    return doc
