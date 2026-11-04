"""Phase 11: flip-flopping edits are not progress, a stopped stream keeps the context estimate, and read_files stays bounded."""

from __future__ import annotations

from tests.test_codeagent_engine import FakeLLM, engine


def test_a_to_b_to_a_is_not_progress(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    eng._seen_states = set()
    a, b = ("hashA", 0), ("hashB", 0)
    assert eng._advanced(a, b) is True            # a new state
    assert eng._advanced(b, a) is False           # back to a state already seen
    assert eng._advanced(a, b) is False           # and forth again
    assert eng._advanced(a, ("hashA", 1)) is True  # one more green check is progress


def test_a_stopped_stream_keeps_the_context_estimate(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.note_usage({"_prompt_eval_count": 30000, "_eval_count": 10})
    before = eng.ctx.estimate()
    eng.ctx.note_usage({"_aborted": "stopped", "_prompt_eval_count": 0})
    assert eng.ctx.last_prompt == 30000 and eng.ctx.estimate() >= before


def test_read_files_is_bounded_and_never_marks_unread_files_as_read(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    cap = eng.tools.read_max_chars
    for i in range(4):
        (repo / f"f{i}.txt").write_text(("x" * 100 + "\n") * (cap // 101))
    out = eng.execute("read_files", {"paths": [f"f{i}.txt" for i in range(4)]})
    assert "not read here" in out
    skipped = [f"f{i}.txt" for i in range(4) if f"f{i}.txt: not read here" in out]
    again = eng.execute("read_file", {"path": skipped[0]})
    assert "unchanged since step" not in again            # it was not shown, so it is not "already in your context"
