import pytest

from evals.harness import EvalTask, TrialResult, evaluate, summarize
from evals.stats import intervals_overlap, mcnemar, pass_at_k, wilson_interval


def test_pass_at_k_all_pass():
    assert pass_at_k(5, 5, 1) == 1.0


def test_pass_at_k_none_pass():
    assert pass_at_k(5, 0, 1) == 0.0


def test_pass_at_k_partial():
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)


def test_pass_at_k_k_gt_available():
    assert pass_at_k(5, 1, 3) == pytest.approx(1 - (4 / 5) * (3 / 4) * (2 / 3))


def test_pass_at_k_validates():
    with pytest.raises(ValueError):
        pass_at_k(5, 5, 0)


def test_wilson_interval_brackets_point():
    ci = wilson_interval(5, 10)
    assert ci.low < ci.point < ci.high
    assert 0.0 <= ci.low and ci.high <= 1.0


def test_wilson_zero_total():
    ci = wilson_interval(0, 0)
    assert ci.point == 0.0


def test_mcnemar_symmetric_is_insignificant():
    result = mcnemar(10, 10)
    assert result.p_value > 0.5


def test_mcnemar_lopsided_is_significant():
    result = mcnemar(20, 2)
    assert result.p_value < 0.05


def test_mcnemar_no_discordant():
    assert mcnemar(0, 0).p_value == 1.0


def test_intervals_overlap():
    a = wilson_interval(5, 10)
    b = wilson_interval(6, 10)
    assert intervals_overlap(a, b)


def test_evaluate_aggregates():
    tasks = [EvalTask("t1", "p1"), EvalTask("t2", "p2")]

    def run_trial(task, seed):
        passed = task.task_id == "t1"
        return TrialResult(task.task_id, seed, passed=passed, attempts=2, malformed=1)

    summary = evaluate(tasks, run_trial, seeds=(1, 2, 3))
    assert summary.total_trials == 6
    assert summary.pass_rate == pytest.approx(0.5)
    assert summary.mean_attempts == pytest.approx(2.0)
    assert summary.malformed_rate == pytest.approx(1.0)


def test_summarize_counts_crashes():
    results = [
        TrialResult("t", 1, passed=False, crashed=True),
        TrialResult("t", 2, passed=True),
    ]
    summary = summarize(results)
    assert summary.crashes == 1
    assert summary.tasks[0].c == 1
