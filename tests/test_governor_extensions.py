from __future__ import annotations

import pytest

from agent.governor import (
    RepetitionDetected,
    RepetitionTracker,
    check_drift,
    compute_drift,
)


# ---------------------------------------------------------------------------
# RepetitionTracker
# ---------------------------------------------------------------------------


def test_repetition_raised_on_third_identical_call():
    t = RepetitionTracker()
    t.record_tool_call("shell", "ls /tmp")
    t.record_tool_call("shell", "ls /tmp")
    with pytest.raises(RepetitionDetected):
        t.record_tool_call("shell", "ls /tmp")


def test_repetition_not_raised_on_two_identical_calls():
    t = RepetitionTracker()
    t.record_tool_call("shell", "ls /tmp")
    t.record_tool_call("shell", "ls /tmp")  # second — must not raise


def test_repetition_not_raised_when_inputs_differ():
    t = RepetitionTracker()
    t.record_tool_call("shell", "ls /tmp")
    t.record_tool_call("shell", "ls /home")
    t.record_tool_call("shell", "ls /var")  # same tool, different input — fine


def test_repetition_window_evicts_old_entries():
    """After the window slides, old duplicates no longer count."""
    t = RepetitionTracker()
    t.record_tool_call("shell", "ls /tmp")
    t.record_tool_call("shell", "ls /tmp")
    # Push two unrelated calls so the first duplicate falls off the 5-entry window
    t.record_tool_call("read", "file1")
    t.record_tool_call("read", "file2")
    t.record_tool_call("read", "file3")
    # Now add another "ls /tmp" — only 1 in the window, no raise
    t.record_tool_call("shell", "ls /tmp")


# ---------------------------------------------------------------------------
# compute_drift
# ---------------------------------------------------------------------------


def test_compute_drift_identical_text():
    score = compute_drift("search files for foo", "search", "files for foo")
    assert score == pytest.approx(1.0)


def test_compute_drift_completely_different():
    score = compute_drift("banana mango papaya", "xyzzy", "qwerty quux")
    assert score == pytest.approx(0.0)


def test_compute_drift_partial_overlap():
    score = compute_drift("search for files", "search", "database records")
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# check_drift
# ---------------------------------------------------------------------------


def test_check_drift_returns_true_when_severe():
    # completely unrelated words → score 0.0 < 0.05
    assert check_drift("read source code", "banana", "papaya mango") is True


def test_check_drift_returns_false_for_related_action():
    # "read" and "source" appear in both → score above default 0.05
    assert check_drift("read source code file", "read", "source code file path") is False


def test_repetition_tracker_handles_non_ascii_input():
    """input_str is hashed as bytes — a surrogate/odd char must not blow up."""
    t = RepetitionTracker()
    t.record_tool_call("shell", "echo ünïcödé ✓")
    t.record_tool_call("shell", "echo \ud800")  # lone surrogate
    t.record_tool_call("shell", "echo ünïcödé ✓")  # only 2 of this one — no raise


def test_repetition_counts_within_the_window_not_consecutively():
    """a,b,a,b,a is three 'a' calls inside the 5-call window — that's a rut."""
    t = RepetitionTracker()
    with pytest.raises(RepetitionDetected):
        for name in ["a", "b", "a", "b", "a"]:
            t.record_tool_call(name, "same-input")


def test_compute_drift_is_symmetric_on_empty_inputs():
    assert compute_drift("", "", "") == 1.0     # nothing vs nothing
    assert compute_drift("", "read", "file") == 0.0
    assert compute_drift("read a file", "", "") == 0.0


def test_check_drift_threshold_is_configurable():
    # "read" overlaps, so the score is small but non-zero.
    score = compute_drift("read the source code of the parser", "read", "unrelated words")
    assert 0.0 < score < 0.5
    assert check_drift("read the source code of the parser", "read",
                       "unrelated words", threshold=score + 0.01) is True
    assert check_drift("read the source code of the parser", "read",
                       "unrelated words", threshold=score - 0.01) is False
