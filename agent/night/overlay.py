"""Settings for the attempts of a night that must not touch the user's config.yaml: a JSON overlay in the environment (NIGHT_OVERLAY) that is
merged over the loaded config INSIDE each attempt process. `boot()` is what the runner starts instead of the plain CLI."""
from __future__ import annotations

import json
import os
from typing import Any

DEFAULT = {"coding": {"stage_prompt_cap": 64000, "stages": "on"}}      # measured: the best run of the context sweep (95k window, 64k stage cap, 6.6 steps/min, tests added)


# The normal defaults of the coding agent were loosened for everybody (see git log: "coding agent: ..."); a night is more patient still, because nobody is there to restart it.
# Only the attempts of a night get these: the user's config and plain `nightage code` stay as they are.
RELAXED = {"coding": {
    "verify_phase_budget": 0,                                          # never end a tool use because a verification looked clean
    "range_split_lines": 0,                                            # a prompt that names a big file is NOT a "rewrite this file in passes" job (that mode cut an add-a-character prompt into 9 parts of characters.py)
    "stage_split_with_model": False,                                   # a night prompt is already one unit: the model-made split ("read / implement / verify") created a stage with nothing to do that looped for 20 minutes
    "autopilot_max_idle": 12, "autopilot_max_segments": 1000, "stage_attempts": 8,       # a blocked stage is lost for the whole night: try longer
    "stage_max_minutes": 60, "stage_max_steps": 400, "stage_min_edits_per_100_steps": 1,
}}


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` merged into `base` (dicts deeply, everything else replaced); returns `base`."""
    for key, value in over.items():
        if isinstance(value, dict):
            cur = base.get(key)
            base[key] = merge(cur if isinstance(cur, dict) else {}, value)
        else:
            base[key] = value
    return base


def env(stage_cap: int = 64000, ctx: int = 0, extra: dict[str, Any] | None = None, relaxed: bool = True) -> dict[str, str]:
    """The environment variables of an attempt: the overlay, and the context window when `ctx` is set (0 = whatever the config says)."""
    over: dict[str, Any] = json.loads(json.dumps(RELAXED)) if relaxed else {}
    if stage_cap > 0:
        merge(over, {"coding": {"stage_prompt_cap": int(stage_cap), "stages": "on"}})
    if extra:
        merge(over, extra)
    out = {"NIGHT_OVERLAY": json.dumps(over)}
    if ctx > 0:
        out["AGENT_CHAT_CTX"] = str(int(ctx))
    return out


def boot() -> None:
    over = json.loads(os.environ.get("NIGHT_OVERLAY") or "{}")
    from agent import config as config_mod

    orig = config_mod.load_config
    config_mod.load_config = lambda *a, **k: merge(orig(*a, **k), json.loads(json.dumps(over)))      # a fresh copy per load: nobody shares the overlay's objects
    from agent.main import app

    app()
