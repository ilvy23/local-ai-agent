"""Coding-mode defaults. Merged over the user's config.yaml `coding:` block.

The sampling here is the researched code config: temperature 0.1 (deterministic),
repeat_penalty 1.05 (NEVER higher — high repetition penalty corrupts code, worst
on small models). num_ctx 16384 fits the coder-7B on 8 GB with an idle GPU.
"""

from __future__ import annotations

import copy
from typing import Any

DEFAULTS: dict[str, Any] = {
    "executor": {
        "model": "huihui_ai/qwen2.5-coder-abliterate:7b",  # uncensored coder; swap in the menu
        "temperature": 0.1,
        "repeat_penalty": 1.05,   # never raise
        "top_p": 0.8,
        "top_k": 20,
        # 16K is rock-solid on 8GB. 32K OOM-crashes the runner intermittently
        # without flash-attention (the loop backs off gracefully if it does).
        # Enable `OLLAMA_FLASH_ATTENTION` + KV-quant, then raise this to 32768.
        "num_ctx": 16384,
        # None = let Ollama auto-place layers, which already puts as MANY on the GPU
        # as fit in VRAM (rest on CPU). The 7B fits entirely → runs fully on GPU.
        # Do NOT force a high number here: for a model too big for VRAM (the 30B on
        # 8GB) that over-commits the GPU and Ollama 500s with a cudaMalloc OOM.
        # Only set a *low* integer to deliberately cap GPU layers for stability.
        "num_gpu": None,
        "max_repair_attempts": 4,
        "whole_file_threshold": 200,   # >N lines → ask for search/replace, not a full rewrite
    },
    # How much the agent writes. Set from the menu. min = smallest single-file
    # solution; mid = balanced with a test; max = thorough, may split files.
    "build": {"effort": "mid"},
    "escalation": {
        "tier3_model": "qwen3-coder:30b",
        "tier3_enabled": False,   # unpulled + unproven >5 tok/s; opt in later
        "total_budget_seconds": 300,
    },
    "guards": {
        "test_files_readonly": True,   # ⚠ do not disable
        "coverage_gate": True,
        "llm_judge": True,
    },
    "sandbox": {
        "kind": "git-worktree",
        "timeout_seconds": 30,
        "memory_mb": 2048,
    },
    "auto_commit": True,          # accepted fixes commit to a scratch branch
}


def load(config: dict | None) -> dict:
    """Deep-merge DEFAULTS with config['coding'] (user values win)."""
    merged = copy.deepcopy(DEFAULTS)
    user = (config or {}).get("coding", {}) if isinstance(config, dict) else {}

    def _merge(base: dict, over: dict) -> None:
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _merge(base[k], v)
            else:
                base[k] = v

    _merge(merged, user)
    return merged
