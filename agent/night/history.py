"""What nights taught, kept across nights: one line per finished night in ~/.local/state/agent/night-history.jsonl, and the strategy outcomes of all earlier nights as PRIORS
for the evidence-based strategy choice (attemptlog.prefer), so a new night does not start from zero. Priors count for at most PRIOR_WEIGHT samples per strategy: this night's own
evidence always outvotes old habits."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

PRIOR_WEIGHT = 4


def path() -> Path:
    return Path(os.environ.get("NIGHT_HISTORY") or Path.home() / ".local" / "state" / "agent" / "night-history.jsonl")


def record(report_rows: list[dict], *, project: str, done: list[str], partial: list[str], hours: float, tests_ok: bool, where: Path | None = None) -> None:
    tally: dict[str, dict[str, list[int]]] = {}
    for r in report_rows:
        t = tally.setdefault(r.get("trigger") or "", {}).setdefault(r.get("strategy", ""), [0, 0, 0])
        t[0] += 1
        t[1] += 1 if r.get("green") else 0
        t[2] += 1 if r.get("progress") else 0
    f = where or path()
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as out:
        out.write(json.dumps({"ts": int(time.time()), "project": project, "done": len(done), "partial": len(partial), "hours": round(hours, 2), "tests_ok": tests_ok, "tally": tally}) + "\n")


def rows(where: Path | None = None) -> list[dict]:
    try:
        lines = (where or path()).read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def priors(trigger: str, where: Path | None = None) -> dict[str, tuple[int, float]]:
    """{strategy: (samples, score)} from earlier nights for the attempts that followed a failure of kind `trigger`, capped at PRIOR_WEIGHT samples."""
    n: dict[str, int] = {}
    pts: dict[str, float] = {}
    for night in rows(where):
        for strategy, (count, greens, progress) in (night.get("tally", {}).get(trigger) or {}).items():
            n[strategy] = n.get(strategy, 0) + count
            pts[strategy] = pts.get(strategy, 0.0) + greens + 0.5 * max(progress - greens, 0)
    return {s: (min(c, PRIOR_WEIGHT), pts[s] / c) for s, c in n.items() if c}


def merge(own: dict[str, tuple[int, float]], prior: dict[str, tuple[int, float]]) -> dict[str, tuple[int, float]]:
    """This night's rates plus the priors, sample-weighted."""
    out = dict(own)
    for s, (pn, pscore) in prior.items():
        on, oscore = out.get(s, (0, 0.0))
        out[s] = (on + pn, (on * oscore + pn * pscore) / (on + pn))
    return out
