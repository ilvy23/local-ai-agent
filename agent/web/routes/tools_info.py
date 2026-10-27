"""Tools & audit screen: every tool the agent can call (with its risk tier and last use) and the audit log with filters.

Read-only. The optional audit tool packs are deliberately not exposed here: they need a signed-off scope and are started
from the terminal menu, where the scope guard applies."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends

from agent.memory.vectors import VectorIndex
from agent.web.deps import build_registry, get_client, get_config, get_store

router = APIRouter(prefix="/api/tools", tags=["tools"])

_RESULT_FLAGS = ("BLOCKED", "Error")


def _tool_of(row: dict[str, Any]) -> str:
    if row["kind"] == "shell":
        return "run_command"
    try:
        return str(json.loads(row["detail"]).get("name", ""))
    except (ValueError, AttributeError):
        return ""


def _decision(row: dict[str, Any]) -> str:
    result = str(row.get("result") or "")
    if row["approved"]:
        return "approved"
    return "refused" if result.startswith("BLOCKED") else "error" if result.startswith("Error") else "denied"


@router.get("")
def list_tools(store=Depends(get_store), client=Depends(get_client), config=Depends(get_config)) -> list[dict[str, Any]]:
    from agent.tui import _static_risk_level

    registry = build_registry(store, VectorIndex(store), client, config, 0)
    last: dict[str, str] = {}
    for row in store.get_audit_log(limit=500):                      # newest first: the first hit per tool is its last use
        last.setdefault(_tool_of(row), row["ts"])
    return [{"name": t.name, "description": t.description, "risk": "DYNAMIC" if t.risk == "dynamic" else _static_risk_level(t.risk).name,
             "last_use": last.get(t.name)} for t in sorted(registry.list(), key=lambda t: t.name)]


@router.get("/audit")
def audit_rows(q: str = "", risk: str = "", decision: str = "", limit: int = 200, store=Depends(get_store)) -> list[dict[str, Any]]:
    """Audit log rows with the tool and the decision spelled out; filter by text, decision (approved/denied/refused/error)."""
    out = []
    needle = q.strip().lower()
    for row in store.get_audit_log(limit=max(1, min(limit, 500))):
        item = {**row, "tool": _tool_of(row), "decision": _decision(row)}
        if needle and needle not in f"{item['tool']} {row['detail']} {row['result']}".lower():
            continue
        if decision and item["decision"] != decision:
            continue
        out.append(item)
    return out


@router.get("/packs")
def packs() -> dict[str, Any]:
    return {"available": False, "reason": "Audit and diagnostic tool packs need a signed-off scope (goal, hosts, tools). They are started from the terminal menu, "
                                          "where the scope guard applies; the web GUI does not start them.",
            "packs": [{"name": "Diagnostics", "summary": "Read system state"}, {"name": "Network audit", "summary": "Needs authorisation: scope and confirmation"}]}
