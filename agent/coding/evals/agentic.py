"""Agentic eval: measure the open-ended loop, not the focused fix loop.

Each task gives the autonomous agent a goal starting from an empty repo. After it
finishes, we materialise its committed branch, drop in HELD-OUT tests it never
saw, and run them. This is how we tell whether it actually built the right thing
(and, for multi-file tasks, the right STRUCTURE) versus something that merely
passed its own tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from agent.coding.agent_loop import run_goal
from agent.coding.verify import runner


@dataclass
class AgenticTask:
    name: str
    goal: str
    held_out: dict[str, str]   # path -> content, run after the agent finishes


TASKS: list[AgenticTask] = [
    AgenticTask(
        name="shapes-package",
        goal=("Create a Python package `shapes` with two files: shapes/area.py defining "
              "circle_area(r), rectangle_area(w, h), and triangle_area(b, h); and "
              "shapes/__init__.py that exports all three. Add tests and make them pass."),
        held_out={"test_held.py": (
            "import math\n"
            "from shapes.area import circle_area, rectangle_area, triangle_area\n\n\n"
            "def test_circle():\n    assert abs(circle_area(2) - math.pi * 4) < 1e-6\n\n\n"
            "def test_rect():\n    assert rectangle_area(3, 4) == 12\n\n\n"
            "def test_tri():\n    assert triangle_area(6, 4) == 12.0\n"
        )},
    ),
    AgenticTask(
        name="cli-wordcount",
        goal=("Create wc.py with a function count_words(text) that returns the number of "
              "whitespace-separated words, handling empty and multi-space strings. Add tests."),
        held_out={"test_held.py": (
            "from wc import count_words\n\n\n"
            "def test_basic():\n    assert count_words('a b  c') == 3\n\n\n"
            "def test_empty():\n    assert count_words('   ') == 0\n"
        )},
    ),
]


def _heldout_passes(repo: Path, task: AgenticTask, branch: str) -> bool:
    with tempfile.TemporaryDirectory() as cd:
        wt = Path(cd) / "wt"
        add = subprocess.run(["git", "worktree", "add", "--detach", str(wt), branch],
                             cwd=repo, capture_output=True)
        if add.returncode != 0:
            return False
        try:
            for p, c in task.held_out.items():
                (wt / p).parent.mkdir(parents=True, exist_ok=True)
                (wt / p).write_text(c)
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            res = runner.run([sys.executable, "-m", "pytest", "-q", "--no-header",
                              "-p", "no:cacheprovider"], cwd=wt, env=env, timeout=120)
            return res.ok
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=repo, capture_output=True)


def run_trial(task: AgenticTask, llm=None, config=None, max_steps: int = 40) -> tuple:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "repo"
        repo.mkdir()
        for a in (["init", "-q"], ["config", "user.email", "e@e"], ["config", "user.name", "e"]):
            subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
        (repo / "README.md").write_text("# proj\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, capture_output=True)

        t0 = time.time()
        out = run_goal(repo, task.goal, config=config, llm=llm, max_steps=max_steps)
        dt = time.time() - t0
        solved = _heldout_passes(repo, task, out.branch) if (out.ok and out.branch) else False
        return task.name, out.ok, solved, out.attempts, dt


def run_agentic_harness(trials: int = 2, task: str | None = None) -> None:
    tasks = [t for t in TASKS if not task or t.name == task]
    print(f"agentic eval · {len(tasks)} task(s) × {trials} trial(s) — this is slow (real autonomous runs)\n")
    rows = []
    for t in tasks:
        for i in range(trials):
            name, done, solved, steps, secs = run_trial(t)
            rows.append((name, done, solved, steps, secs))
            mark = "✓ solved" if solved else ("~ done, held-out failed" if done else "✗ gave up")
            print(f"  {mark:24} {name:16} trial {i}  {steps} steps · {secs:4.0f}s")

    print("\n=== per task ===")
    print(f"{'task':16} {'solved':>8} {'finished':>9} {'mean-steps':>11} {'mean-s':>7}")
    for t in tasks:
        rs = [r for r in rows if r[0] == t.name]
        n = len(rs)
        print(f"{t.name:16} {sum(r[2] for r in rs)}/{n:<7} {sum(r[1] for r in rs)}/{n:<8} "
              f"{sum(r[3] for r in rs) / n:11.1f} {sum(r[4] for r in rs) / n:7.0f}")
    n = len(rows)
    print(f"\noverall: {sum(r[2] for r in rows)}/{n} solved (held-out), "
          f"{sum(r[1] for r in rows)}/{n} finished")
