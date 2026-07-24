from agent.coding.repair import (
    Attempt,
    Decision,
    RepairState,
    Tier,
    decide_next,
    detect_stall,
    is_specific_blocker,
    normalized_ast,
)


def test_normalized_ast_ignores_comments_and_whitespace():
    a = normalized_ast("x = 1  # comment")
    b = normalized_ast("x = 1")
    assert a == b


def test_normalized_ast_none_on_syntax_error():
    assert normalized_ast("def broken(:") is None


def test_detect_stall_first_attempt_never_stalls():
    a = Attempt.make("x = 1")
    assert not detect_stall(a, None).stalled


def test_detect_stall_identical_ast():
    prev = Attempt.make("x = 1  # v1")
    cur = Attempt.make("x = 1  # v2")
    assert detect_stall(cur, prev).stalled


def test_detect_stall_identical_error():
    prev = Attempt.make("x = 1", error="TypeError: nope")
    cur = Attempt.make("y = 2", error="TypeError: nope")
    assert detect_stall(cur, prev).stalled


def test_detect_stall_progress():
    prev = Attempt.make("x = 1", error="E1")
    cur = Attempt.make("x = 2", error="E2")
    assert not detect_stall(cur, prev).stalled


def test_is_specific_blocker():
    assert is_specific_blocker("missing helpers/util.py")
    assert is_specific_blocker("cannot find ConfigLoader")


def test_is_vague_blocker():
    assert not is_specific_blocker("may need refactoring")
    assert not is_specific_blocker(None)
    assert not is_specific_blocker("")


def test_decide_next_retry_on_progress():
    state = RepairState(max_attempts=4)
    decide_next(state, Attempt.make("x = 1", error="E1"))
    decision = decide_next(state, Attempt.make("x = 2", error="E2"))
    assert decision.action is Decision.RETRY


def test_decide_next_escalates_on_stall():
    state = RepairState(max_attempts=4)
    decide_next(state, Attempt.make("x = 1", error="E"))
    decision = decide_next(state, Attempt.make("x = 1", error="E"))
    assert decision.action is Decision.ESCALATE
    assert decision.tier is Tier.FRESH_CONTEXT


def test_decide_next_satisfies_specific_blocker():
    state = RepairState(max_attempts=4)
    decision = decide_next(state, Attempt.make("x = 1", blocker="missing lib/foo.py"))
    assert decision.action is Decision.SATISFY_BLOCKER


def test_decide_next_escalates_on_attempt_cap():
    state = RepairState(max_attempts=2)
    decide_next(state, Attempt.make("a = 1", error="E1"))
    decision = decide_next(state, Attempt.make("b = 2", error="E2"))
    assert decision.action is Decision.ESCALATE


def test_specific_blocker_and_stall_escalates():
    state = RepairState(max_attempts=4)
    decide_next(state, Attempt.make("x = 1", blocker="missing lib/foo.py"))
    decision = decide_next(state, Attempt.make("x = 1", blocker="missing lib/foo.py"))
    assert decision.action is Decision.ESCALATE
