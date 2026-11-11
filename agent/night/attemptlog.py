"""A measured record of every attempt (.night/attempts.jsonl) and what it teaches: which strategy worked after which kind of failure, in THIS night.

`rows()` reads it, `rates()` turns it into (samples, score) per strategy for one trigger kind (score = greens + half a point for progress, per sample), and
`prefer()` lets the analyzer overrule its default plan - but only with evidence (at least MIN_SAMPLES attempts and a clearly better score), so a few lucky
runs never flip the plan."""
from __future__ import annotations

import json
import time
from pathlib import Path

MIN_SAMPLES, MARGIN = 3, 0.3


def record(path: Path, **row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"ts": int(time.time()), **row}) + "\n")


def rows(path: Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def rates(data: list[dict], trigger: str) -> dict[str, tuple[int, float]]:
    """{strategy: (samples, score)} for the attempts that followed a failure of kind `trigger`."""
    agg: dict[str, list[float]] = {}
    for r in data:
        if r.get("trigger") == trigger and r.get("strategy"):
            agg.setdefault(r["strategy"], []).append(1.0 if r.get("green") else 0.5 if r.get("progress") else 0.0)
    return {s: (len(v), sum(v) / len(v)) for s, v in agg.items()}


def prefer(default: str, candidates: list[str], stats: dict[str, tuple[int, float]]) -> str:
    """`default` unless a candidate has enough samples and a score at least MARGIN better than the default's (an unmeasured default counts as 0.25)."""
    base = stats.get(default, (0, 0.25))[1] if stats.get(default, (0, 0))[0] >= MIN_SAMPLES else 0.25
    best, best_score = default, base
    for c in candidates:
        n, score = stats.get(c, (0, 0.0))
        if n >= MIN_SAMPLES and score >= base + MARGIN and score > best_score:
            best, best_score = c, score
    return best
