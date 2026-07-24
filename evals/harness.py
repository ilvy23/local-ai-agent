from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from evals.stats import Interval, pass_at_k, wilson_interval

DEFAULT_SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    prompt: str
    files: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    task_id: str
    seed: int
    passed: bool
    malformed: int = 0
    attempts: int = 0
    wall_seconds: float = 0.0
    crashed: bool = False


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    n: int
    c: int
    pass_at_1: float


@dataclass(frozen=True)
class EvalSummary:
    tasks: list[TaskSummary]
    pass_rate: float
    pass_rate_ci: Interval
    malformed_rate: float
    mean_attempts: float
    crashes: int
    total_trials: int


RunTrial = Callable[[EvalTask, int], TrialResult]


def evaluate(
    tasks: list[EvalTask],
    run_trial: RunTrial,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> EvalSummary:
    results: list[TrialResult] = []
    for task in tasks:
        for seed in seeds:
            results.append(run_trial(task, seed))
    return summarize(results)


def summarize(results: list[TrialResult]) -> EvalSummary:
    by_task: dict[str, list[TrialResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    task_summaries = []
    for task_id, trials in by_task.items():
        n = len(trials)
        c = sum(1 for t in trials if t.passed and not t.crashed)
        task_summaries.append(
            TaskSummary(task_id=task_id, n=n, c=c, pass_at_1=pass_at_k(n, c, 1))
        )

    total = len(results)
    passed = sum(1 for r in results if r.passed and not r.crashed)
    malformed = sum(r.malformed for r in results)
    attempts = sum(r.attempts for r in results)
    crashes = sum(1 for r in results if r.crashed)

    return EvalSummary(
        tasks=task_summaries,
        pass_rate=passed / total if total else 0.0,
        pass_rate_ci=wilson_interval(passed, total),
        malformed_rate=malformed / total if total else 0.0,
        mean_attempts=attempts / total if total else 0.0,
        crashes=crashes,
        total_trials=total,
    )
