"""Defaults for `nightage night run` from an optional `night:` section of config.yaml (command-line options always win). Private edition only, so it is
documented in docs/NIGHT.md, not in the shipped config.example.yaml. Every value is validated and clamped: a typo in the config never starts a 40-hour night.

    night:
      hours: 8
      tournament: 3          # candidates per round after two equal failures (0/1 = off)
      review: false          # a review pass after each finished prompt
      map: auto              # off | auto | full
      stage_cap: 64000       # prompt cap per stage for the attempts (0 = leave coding.stage_prompt_cap alone)
      ctx: 0                 # context window for the attempts (0 = the normal one)
      pause_on_game: true"""
from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {"hours": 8.0, "tournament": 3, "review": False, "map": "auto", "stage_cap": 64000, "ctx": 0, "pause_on_game": True}


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        return min(max(float(value), lo), hi)
    except (TypeError, ValueError):
        return default


def _flag(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return True if text in ("true", "yes", "on", "1") else False if text in ("false", "no", "off", "0") else default


def from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = (config or {}).get("night")
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULTS)
    out["hours"] = _num(raw.get("hours", DEFAULTS["hours"]), DEFAULTS["hours"], 0.25, 24.0)
    out["tournament"] = int(_num(raw.get("tournament", DEFAULTS["tournament"]), DEFAULTS["tournament"], 0, 6))
    out["review"] = _flag(raw.get("review", DEFAULTS["review"]), DEFAULTS["review"])
    out["map"] = raw.get("map") if raw.get("map") in ("off", "auto", "full") else DEFAULTS["map"]
    out["stage_cap"] = int(_num(raw.get("stage_cap", DEFAULTS["stage_cap"]), DEFAULTS["stage_cap"], 0, 250000))
    out["ctx"] = int(_num(raw.get("ctx", DEFAULTS["ctx"]), DEFAULTS["ctx"], 0, 262144))
    out["pause_on_game"] = _flag(raw.get("pause_on_game", DEFAULTS["pause_on_game"]), DEFAULTS["pause_on_game"])
    return out
