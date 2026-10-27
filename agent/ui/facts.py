"""Footer / status facts of the chat UI: model, backend, GPU mode, context limit, access mode, folder.

Every helper is defensive: it returns a harmless default instead of raising (hardware probes, missing config keys,
a broken environment must never take the footer down)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def model_name(config: dict[str, Any] | None) -> str:
    try:
        return str(((config or {}).get("models") or {}).get("chat") or "")
    except Exception:  # noqa: BLE001
        return ""


def backend_name(model: str) -> str:
    """'Vulkan' when the model is routed to the dual-GPU Vulkan Ollama server, else 'Ollama'."""
    try:
        from agent import llm

        return "Vulkan" if model and llm.wants_vulkan(model) else "Ollama"
    except Exception:  # noqa: BLE001
        return "Ollama"


def gpu_mode() -> str:
    try:
        from agent import gpumode

        return str(gpumode.current_mode() or "")
    except Exception:  # noqa: BLE001
        return ""


def ctx_limit(config: dict[str, Any] | None, model: str) -> int:
    """The num_ctx a turn is sent with (same rule as agent.tui._num_ctx), 0 when unknown."""
    try:
        from agent import tui

        return int(tui._num_ctx(config or {}, model))
    except Exception:  # noqa: BLE001
        return 0


def access_mode(folder: str | os.PathLike | None = None) -> str:
    """ask | medium | auto for `folder` (default: the working directory)."""
    try:
        from agent import repoaccess

        return str(repoaccess.mode_for(Path(folder or os.getcwd()).resolve()))
    except Exception:  # noqa: BLE001
        return "ask"


def footer_facts(config: dict[str, Any] | None, model: str | None = None, folder: str | None = None) -> dict[str, Any]:
    """Keyword arguments for `agent.ui.dock.FooterState` (static-ish facts; the app adds ctx use, queue, tok/s)."""
    m = model if model is not None else model_name(config)
    f = folder or os.getcwd()
    limit = ctx_limit(config, m)
    out: dict[str, Any] = {
        "model": m, "backend": backend_name(m), "gpu": gpu_mode(), "access": access_mode(f), "folder": f,
    }
    if limit:
        out["ctx_limit"] = limit
        out["ctx_used"] = 0
    return out


def budget_explain(config: dict[str, Any] | None, model: str) -> dict[str, Any]:
    """`ctxbudget.explain` for `model`, or {} when it cannot be computed."""
    try:
        from agent import ctxbudget

        return ctxbudget.explain(model, config=config or {})
    except Exception:  # noqa: BLE001
        return {}
