"""View and edit a curated set of config values from config.yaml.

Only exposes the knobs worth touching, by a short name mapped to a dotted path.
Writes back to the project config.yaml, coercing obvious int/bool values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent.config import DEFAULT_CONFIG_PATH, load_config

# short name -> (dotted path, one-line help)
EDITABLE: dict[str, tuple[str, str]] = {
    "chat_model": ("models.chat", "everyday chat + tools model"),
    "background_model": ("models.background", "fact distillation model"),
    "embed_model": ("models.embed", "embedding model (use `reembed` to change + rebuild)"),
    "recall_k": ("memory.recall_k", "memories recalled per chat turn"),
    "unload_on_exit": ("ollama.unload_on_exit", "free the GPU when you leave the chat"),
    "pause_on_game": ("background.pause_on_game", "pause bg work while gaming"),
    "max_gpu_percent": ("background.max_gpu_percent", "GPU cap for bg work"),
}


def _get(config: dict, dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def current() -> list[tuple[str, str, Any]]:
    """(short_name, help, current_value) for every editable setting."""
    config = load_config()
    return [(name, help_, _get(config, path)) for name, (path, help_) in EDITABLE.items()]


def _coerce(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def set_value(name: str, value: str) -> tuple[str, Any]:
    """Set an editable setting in config.yaml. Returns (dotted_path, coerced)."""
    if name not in EDITABLE:
        raise KeyError(f"unknown setting '{name}' (see: agent settings)")
    dotted, _ = EDITABLE[name]
    coerced = _coerce(value)
    path = Path(DEFAULT_CONFIG_PATH)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    node = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = coerced
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    return dotted, coerced
