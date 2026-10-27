"""Coding-mode defaults. Merged over the user's config.yaml `coding:` block.

The sampling here is the researched code config: temperature 0.1 (deterministic),
repeat_penalty 1.05 (NEVER higher — high repetition penalty corrupts code, worst
on small models). `num_ctx` is None here (= "work it out automatically" via
agent.ctxbudget, replacing the old fixed 16384) unless the user's config sets one.
"""

from __future__ import annotations

import copy
from typing import Any

DEFAULTS: dict[str, Any] = {
    "executor": {
        "model": "qwen2.5-coder:7b",  # swap in the menu (`nightage code-model`)
        "temperature": 0.1,
        "repeat_penalty": 1.05,   # never raise
        "top_p": 0.8,
        "top_k": 20,
        # None = computed by agent.ctxbudget.budget_for() at load() time, from detected
        # VRAM/RAM and the ceiling (models.max_ctx_ceiling). 16K used to be hardcoded here
        # for an 8GB card; set an explicit coding.executor.num_ctx in config.yaml to pin it
        # again (e.g. if 32K OOMs your runner without flash-attention/KV-quant).
        "num_ctx": None,
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
    "budget": {
        "max_actions": 100,
        "max_tokens": 100_000,
        "max_seconds": 3600,
    },
}


def load(config: dict | None) -> dict:
    """Deep-merge DEFAULTS with config['coding'] (user values win).

    If ``config['models']['coding']`` is set it is applied as
    ``executor.model`` after the merge, so the top-level routing key and the
    coding agent's actual executor model always stay in sync.  ``models.coding``
    takes final precedence — it is applied after the ``coding`` block merge so
    the routing table and the coding loop always point at the same model.
    """
    merged = copy.deepcopy(DEFAULTS)
    user = (config or {}).get("coding", {}) if isinstance(config, dict) else {}

    def _merge(base: dict, over: dict) -> None:
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _merge(base[k], v)
            else:
                base[k] = v

    _merge(merged, user)

    # models.coding (if set) provides the executor model so both the routing
    # table and the coding loop refer to the same model name.
    coding_model = ((config or {}).get("models") or {}).get("coding")
    if coding_model:
        merged["executor"]["model"] = coding_model

    # num_ctx stays None unless DEFAULTS/user set it explicitly -> work it out from detected
    # hardware instead of a fixed guess. An explicit coding.executor.num_ctx (or a
    # models.overrides.<model>.num_ctx) in config.yaml still always wins.
    if merged["executor"].get("num_ctx") is None:
        from agent import ctxbudget

        merged["executor"]["num_ctx"] = ctxbudget.budget_for(
            merged["executor"].get("model"), None, config=config,
        )

    return merged
