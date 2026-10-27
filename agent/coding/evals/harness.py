"""Run the eval tasks and report honest metrics.

pass@k uses the unbiased estimator (Chen et al. 2021), not the naive formula.
A "pass" means the fix passed the HELD-OUT tests too — passing only the visible
tests but failing held-out is counted as a *false success*, reported separately.

Usage:
    uv run python -m agent.coding.evals.harness [--seeds N] [--task NAME]

Small models vary 5-15 points across seeds, so ≥3 seeds is the honest minimum;
a single seed is a smoke test, not a measurement.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from math import comb
from pathlib import Path

from agent.coding import repair
from agent.coding.evals.tasks import TASKS, Task
from agent.coding.verify import runner


@dataclass
class TrialResult:
    task: str
    seed: int
    solved: bool           # visible checks passed AND held-out passed
    visible_ok: bool       # the loop reported success (visible + guards)
    attempts: int
    malformed: int
    seconds: float
    false_success: bool    # visible_ok but held-out failed


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: prob at least one of k samples is correct, given c/n correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _run_heldout(repo: Path, task: Task, branch: str) -> bool:
    """Materialise the fixed files + held-out tests in a clean dir and run pytest."""
    with tempfile.TemporaryDirectory() as cd:
        cdir = Path(cd)
        for p, c in task.files.items():
            (cdir / p).write_text(c)
        for f in task.edit_files:  # overwrite with the model's fixed version
            fixed = runner.run(["git", "show", f"{branch}:{f}"], cwd=repo).stdout
            (cdir / f).write_text(fixed)
        for p, c in task.held_out.items():
            (cdir / p).write_text(c)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        r = runner.run([sys.executable, "-m", "pytest", "-q", "--no-header",
                        "-p", "no:cacheprovider"], cwd=cdir, env=env, timeout=120)
        return r.ok


def run_trial(task: Task, seed: int, llm=None, config=None) -> TrialResult:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "repo"
        repo.mkdir()
        for p, c in {**task.files, **task.visible_tests}.items():
            (repo / p).write_text(c)
        _git(["init", "-q"], repo)
        _git(["config", "user.email", "e@e"], repo)
        _git(["config", "user.name", "e"], repo)
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "init"], repo)

        t0 = time.time()
        out = repair.run(repo, task.instruction, task.edit_files,
                         config=config, llm=llm, seed=seed)
        dt = time.time() - t0

        held = _run_heldout(repo, task, out.branch) if (out.ok and out.branch) else False
        return TrialResult(
            task=task.name, seed=seed, solved=bool(out.ok and held),
            visible_ok=out.ok, attempts=out.attempts, malformed=out.malformed_count,
            seconds=dt, false_success=bool(out.ok and not held),
        )


def run_harness(seeds: int = 3, task: str | None = None) -> None:
    tasks = [t for t in TASKS if not task or t.name == task]
    seeds = list(range(seeds))
    results: list[TrialResult] = []

    print(f"running {len(tasks)} task(s) × {len(seeds)} seed(s)…\n")
    for task in tasks:
        for seed in seeds:
            r = run_trial(task, seed)
            results.append(r)
            mark = "✓" if r.solved else ("⚠ false" if r.false_success else "✗")
            print(f"  {mark} {task.name:22} seed {seed}  "
                  f"{r.attempts} att · {r.malformed} malformed · {r.seconds:4.1f}s")

    print("\n=== per task ===")
    print(f"{'task':22} {'pass@1':>7} {'false-succ':>11} {'mean-att':>9} {'mean-s':>7}")
    for task in tasks:
        rs = [r for r in results if r.task == task.name]
        n, c = len(rs), sum(r.solved for r in rs)
        fs = sum(r.false_success for r in rs)
        print(f"{task.name:22} {pass_at_k(n, c, 1):7.2f} {fs}/{n:<9} "
              f"{sum(r.attempts for r in rs) / n:9.1f} {sum(r.seconds for r in rs) / n:7.1f}")

    n, c = len(results), sum(r.solved for r in results)
    fs = sum(r.false_success for r in results)
    mal = sum(r.malformed for r in results)
    print("\n=== overall ===")
    print(f"trials            {n}")
    print(f"pass@1            {pass_at_k(n, c, 1):.2f}  ({c}/{n} solved incl. held-out)")
    if len(seeds) >= 2:
        print(f"pass@{len(seeds)}            {pass_at_k(n, c, len(seeds)):.2f}")
    print(f"false-success     {fs}/{n}  (passed visible, failed held-out — reward hacking)")
    print(f"malformed edits   {mal} total")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="seeds per task (≥3 for real numbers)")
    ap.add_argument("--task", default=None, help="run only this task name")
    args = ap.parse_args()
    run_harness(args.seeds, args.task)


if __name__ == "__main__":
    main()
