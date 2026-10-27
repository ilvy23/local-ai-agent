"""Extra read-only data for the Models and Prompt-studio screens (kept out of models.py / workspace.py)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from agent import promptlab
from agent.web.deps import get_client, get_config

router = APIRouter(prefix="/api", tags=["models-ext"])


@router.get("/models/detail")
def models_detail(client=Depends(get_client), config=Depends(get_config)) -> list[dict[str, Any]]:
    """Every installed model with its size on disk (MiB), whether it is resident right now and whether it is the chat model."""
    try:
        sizes = client.installed_sizes()
    except Exception:  # noqa: BLE001 - an unreachable Ollama is an empty list, the screen shows its own state
        sizes = {}
    try:
        loaded = set(client.loaded_models())
    except Exception:  # noqa: BLE001
        loaded = set()
    current = config["models"]["chat"]
    return [{"name": n, "size_mb": s, "loaded": n in loaded, "current": n == current} for n, s in sorted(sizes.items())]


@router.get("/prompt/sessions")
def prompt_sessions() -> list[dict[str, Any]]:
    """Saved prompt-studio chats (newest first): id, title, mode, created and the last assistant message."""
    out = []
    for s in promptlab._load_sessions():
        if not isinstance(s, dict):
            continue
        last = next((m.get("content", "") for m in reversed(s.get("messages") or []) if isinstance(m, dict) and m.get("role") == "assistant"), "")
        out.append({"id": s.get("id"), "title": s.get("title", ""), "mode": s.get("mode", ""), "created": s.get("created", ""), "last": last})
    return out[::-1]
