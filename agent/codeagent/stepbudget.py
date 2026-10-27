"""Adaptive step budget: the agent sizes its own tool-step budget per goal; a user value is a hard cap, the ceiling only an upper bound.

estimate() is a cheap heuristic (goal wording and length, focus files named in the goal, repo size, greenfield vs. edit, effort,
diagnose-first) - no model call. The engine starts the prompt with clamp(estimate, LO, ceiling) and may grow it ONCE (grow()) when
real progress is shown, never past the ceiling."""
from __future__ import annotations

import os
import re
from pathlib import Path

LO = 8
BASE = 14
_BIG = re.compile(r"\b(refactor|migrat\w*|rewrite|overhaul|redesign|framework|engine|server|api|database|backend|frontend|web ?app|"
                  r"gui|multi\w*|several|all (?:the )?(?:files|modules|tests)|end[- ]to[- ]end|complete|full|entire|system|plugin)\b", re.I)
_SMALL = re.compile(r"\b(typo|rename|tweak|tiny|small|simple|minor|quick|one[- ]liner|one line|comment|docstring|bump)\b", re.I)
_FILE = re.compile(r"[\w./-]+\.(?:py|js|ts|tsx|jsx|rs|go|c|h|cpp|java|rb|sh|html|css|toml|yaml|yml|json|md)\b")
_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", ".scratch", "dist", "build", "target"}
EFFORT = {"min": 0.7, "mid": 1.0, "max": 1.5}


def repo_files(root: Path, cap: int = 600) -> int:
    n = 0
    for _d, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP and not d.startswith(".")]
        n += len(files)
        if n >= cap:
            return cap
    return n


def estimate(goal: str, *, files: int = 0, effort: str = "mid", diagnose: bool = False) -> int:
    """Rough number of tool steps the goal needs, including some headroom (unclamped)."""
    words = len(goal.split())
    est = BASE + min(words // 10, 14)
    est += min(6 * len(_BIG.findall(goal)), 24)
    est -= 4 * min(len(_SMALL.findall(goal)), 2)
    est += 3 * min(len(set(_FILE.findall(goal))), 6)             # focus files named in the goal
    if files > 3:                                                # an edit in an existing repo: more reading first
        est += min(files // 60, 10)
    if diagnose:
        est += 4
    return round(est * EFFORT.get(effort, 1.0))


def clamp(est: int, ceiling: int) -> int:
    return max(min(LO, ceiling), min(est, ceiling))


def grow(budget: int, ceiling: int) -> int:
    """The one bounded extension: half again (at least 4 steps), never past the ceiling."""
    return min(budget + max(4, budget // 2), ceiling)
