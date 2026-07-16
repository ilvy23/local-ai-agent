"""Load and persist agent configuration.

On first run, a config.yaml with sane defaults is created next to the
project root (or wherever the caller points it). Later tasks (SQLite
persistence, vector memory) read models/persona from the same file.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        # Chat MUST be a tool-capable model (`ollama show <m>` lists "tools").
        # Ollama silently drops the tools for models without template support,
        # and the model then invents results instead of running anything.
        "chat": "qwen2.5:7b",
        "background": "qwen2.5:7b",  # fact distillation (no tools needed)
        "embed": "bge-m3",           # embeddings (multilingual, 1024-dim)
    },
    "persona": {
        "name": "Agent",
        "style": (
            "You are Agent, a warm and friendly agent who chats "
            "like a supportive friend: casual, honest, and curious. "
            "Give thorough, detailed answers — explain your reasoning, add "
            "useful context and examples, and cover the relevant angles rather "
            "than one-liners. Structure longer answers with short paragraphs or "
            "bullet points when it helps. Stay natural, not padded: be detailed "
            "because the content earns it, and match a quick reply to a quick "
            "question."
        ),
    },
    "data": {
        "db_path": "data/agent.db",
    },
    "memory": {
        "recall_k": 6,
        "context_char_budget": 24000,
    },
    "tools": {
        "max_iterations": 8,
    },
    "safety": {
        # These lists EXTEND the built-in allow/block lists; they never shrink
        # them. Built-in safe commands and blocked patterns are hard-coded in
        # agent.safety and cannot be removed via config.
        "safe_commands": [],
        "blocked_patterns": [],
        "max_timeout_s": 300,
    },
    "background": {
        # Background bulk jobs pause while a game runs or the GPU is busier
        # than this, so gaming keeps full performance. CPU/RAM caps are
        # best-effort (real compute is in Ollama); the GPU + game signals bite.
        "pause_on_game": True,
        "max_gpu_percent": 40,
        "max_cpu_percent": 50,
        "max_ram_gb": 16,
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge `overrides` onto `base`, recursing into nested dicts."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Load config from `config_path`, creating it with defaults if missing.

    Also ensures `data_dir` exists. Values from `config_path` are deep-merged
    over `DEFAULT_CONFIG` so newly-added default keys are always present, even
    in an older config file. Always returns a fresh dict.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        config_path.write_text(yaml.dump(DEFAULT_CONFIG, sort_keys=False))
        return copy.deepcopy(DEFAULT_CONFIG)

    with config_path.open() as f:
        loaded = yaml.safe_load(f) or {}

    return _deep_merge(copy.deepcopy(DEFAULT_CONFIG), loaded)
