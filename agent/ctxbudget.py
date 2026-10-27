"""Per-model context budget: how many tokens of context a model can run with, here and now.

Generalises what used to be a name-based special case (`llm.cap_ctx` only capped models with
"qwen" in their name at a fixed `MAX_QWEN_CTX`). Every model now gets the same treatment:

  1. An explicit override always wins: ``models.overrides.<model>.num_ctx`` in config.
  2. Otherwise, estimate what fits: free VRAM (from `gpumode.detect_gpus()`) or, with no
     discrete GPU, a conservative slice of system RAM; a KV-cache-bytes-per-token heuristic
     (exact when `gpumode`'s per-model facts are known, else a conservative guess from the
     model's size); clamped to a config/env ceiling (`models.max_ctx_ceiling`, default 95000 -
     the old `MAX_QWEN_CTX` value, kept as the default so behaviour for existing qwen setups
     does not silently change).
  3. The caller's own `requested` value, if any, is never exceeded either.

  AGENT_CTX_CEILING   overrides the ceiling for every model (replaces AGENT_MAX_QWEN_CTX)
  AGENT_MAX_QWEN_CTX  back-compat alias, still read if AGENT_CTX_CEILING is unset

`explain(model_name, ...)` returns the same decision as a dict, for settings/ops screens.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gpumode

DEFAULT_CEILING = 95000
_CTX_FLOOR = 2048
# with no discrete GPU, don't let context eat more than half of RAM minus a safety margin
_RAM_SHARE = 0.5
_RAM_RESERVE_MB = 1024
# conservative KV bytes/token when neither exact model facts nor a usable size are known
_UNKNOWN_KV_BYTES_PER_TOKEN = 128 * 1024.0


@dataclass
class HardwareSummary:
    gpus: list[Any] = field(default_factory=list)
    total_ram_mb: int = 0

    @property
    def cpu_only(self) -> bool:
        return not self.gpus


def read_total_ram_mb() -> int:
    """Total system RAM in MB, or 0 if it can't be read (never raises)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform in ("win32", "darwin"):
        from agent import osport

        return osport.native_ram_mb()[0]
    return 0


def _remote_server() -> bool:
    """True when Ollama runs on another machine (`ollama.host` / OLLAMA_HOST): this machine's RAM/VRAM say nothing about it."""
    try:
        from agent import llm

        return llm.is_remote()
    except Exception:  # noqa: BLE001
        return False


def hardware_summary(gpus: list[Any] | None = None) -> HardwareSummary:
    """Snapshot of what's available right now. Pass `gpus` in tests to avoid real detection."""
    return HardwareSummary(gpus=gpumode.detect_gpus() if gpus is None else list(gpus),
                            total_ram_mb=read_total_ram_mb())


def _usable_mb(hw: HardwareSummary) -> float:
    """Memory available for weights + KV cache: free VRAM if any GPU was detected, else a
    conservative share of RAM (CPU-only / no-GPU laptops included)."""
    if hw.gpus:
        total_free = sum(max(g.total_mb - g.used_mb, 0) for g in hw.gpus)
        return max(total_free - gpumode._RESERVE_MB, 0)
    if hw.total_ram_mb:
        return max(hw.total_ram_mb * _RAM_SHARE - _RAM_RESERVE_MB, 0)
    return 0.0


def _ceiling(config: dict | None) -> int:
    raw = os.environ.get("AGENT_CTX_CEILING", "").strip() or os.environ.get("AGENT_MAX_QWEN_CTX", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    value = ((config or {}).get("models") or {}).get("max_ctx_ceiling")
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_CEILING


def _override(model_name: str | None, config: dict | None) -> int | None:
    overrides = ((config or {}).get("models") or {}).get("overrides") or {}
    if not isinstance(overrides, dict) or not model_name:
        return None
    entry = overrides.get(model_name)
    value = entry.get("num_ctx") if isinstance(entry, dict) else None
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _weights_and_kv(model_size_params_or_blob_size: Any, facts: dict | None) -> tuple[float, float]:
    """(weights_mb, kv_bytes_per_token). Prefers exact `gpumode` model facts; otherwise derives
    a rough weights size from the given param count / blob byte size, and a conservative
    KV-per-token guess that grows with model size."""
    if facts and facts.get("size_mb") and facts.get("kv_elems_per_token"):
        kv_type = facts.get("kv_type", "f16")
        return float(facts["size_mb"]), facts["kv_elems_per_token"] * gpumode.KV_BYTES.get(kv_type, 2.0)

    size = float(model_size_params_or_blob_size or 0)
    if size <= 0:
        return 0.0, _UNKNOWN_KV_BYTES_PER_TOKEN
    # Heuristic: a bare integer under ~1e6 is a parameter count (e.g. 7_000_000_000 for "7B"),
    # anything bigger is already a byte size (ollama blob size, GGUF file size, ...).
    weights_mb = size / (1024 * 1024) if size > 1e6 else (size * 2.0) / 1024  # fp16 bytes/param -> MB
    # more weight usually means more layers/heads -> more KV per token; conservative linear guess
    weights_gb = weights_mb / 1024
    kv_bytes = max(32 * 1024.0, weights_gb * 16 * 1024.0)
    return weights_mb, kv_bytes


def budget_for(
    model_name: str | None,
    model_size_params_or_blob_size: Any = None,
    hw_summary: HardwareSummary | None = None,
    requested: int | None = None,
    *,
    config: dict | None = None,
    facts: dict | None = None,
) -> int:
    """How many context tokens `model_name` should be allowed to use.

    `model_size_params_or_blob_size` is a parameter count (e.g. 7e9) or a byte size (blob/GGUF);
    pass whichever is cheaply at hand, or None to fall back on a conservative constant.
    `hw_summary` lets callers (and tests) supply fake hardware instead of real detection.
    `requested` is clamped like everything else, it is never exceeded.
    An explicit `models.overrides.<model>.num_ctx` in config always wins over all of this.
    """
    override = _override(model_name, config)
    if override:
        return override

    ceiling = _ceiling(config)
    # With no idea at all how big the model is (no facts, no size/blob hint) there is nothing
    # concrete to estimate from - guessing against free RAM/VRAM in that case would punish a
    # small model as if it were huge, which is worse than just trusting the ceiling (a caller
    # that already knows the model's size or facts - e.g. gpumode's own table, or anyone who
    # looked the model up - gets the real VRAM/RAM-aware estimate below).
    has_model_info = bool(model_size_params_or_blob_size) or bool(facts and facts.get("size_mb"))
    if not has_model_info or _remote_server():
        budget = ceiling
    else:
        hw = hw_summary if hw_summary is not None else hardware_summary()
        usable_mb = _usable_mb(hw)
        weights_mb, kv_bytes_per_token = _weights_and_kv(model_size_params_or_blob_size, facts)
        left_mb = max(usable_mb - weights_mb, 0.0)
        estimated = int((left_mb * 1024 * 1024) / kv_bytes_per_token) if kv_bytes_per_token > 0 else ceiling
        estimated = estimated if estimated > 0 else _CTX_FLOOR
        budget = min(estimated, ceiling)

    if requested:
        budget = min(int(requested), budget)
    return max(budget, _CTX_FLOOR)


def explain(
    model_name: str | None,
    model_size_params_or_blob_size: Any = None,
    hw_summary: HardwareSummary | None = None,
    requested: int | None = None,
    *,
    config: dict | None = None,
    facts: dict | None = None,
) -> dict[str, Any]:
    """The same decision as `budget_for`, broken down for a settings/ops screen."""
    override = _override(model_name, config)
    ceiling = _ceiling(config)
    hw = hw_summary if hw_summary is not None else hardware_summary()
    usable_mb = _usable_mb(hw)
    weights_mb, kv_bytes_per_token = _weights_and_kv(model_size_params_or_blob_size, facts)
    computed = budget_for(model_name, model_size_params_or_blob_size, hw, None, config=config, facts=facts)
    final = budget_for(model_name, model_size_params_or_blob_size, hw, requested, config=config, facts=facts)
    return {
        "model": model_name,
        "override": override,
        "computed": computed,
        "requested": requested,
        "final": final,
        "ceiling": ceiling,
        "usable_mb": usable_mb,
        "weights_mb": weights_mb,
        "source": "vram" if hw.gpus else ("ram" if hw.total_ram_mb else "unknown"),
        "cpu_only": hw.cpu_only,
        "gpu_count": len(hw.gpus),
    }
