"""mini-swe-agent idea: end the segment after N consecutive unusable replies (coding.malformed_abort, default 5) instead of burning steps.
Unusable = no readable tool call and no answer: a garbled call, a reply cut off by the output cap, an empty reply, a reply that is only thinking."""

from __future__ import annotations

from tests.test_codeagent_engine import FakeLLM, _isolated, call, engine as _engine, no_summary, say  # noqa: F401 - _isolated is an autouse fixture


def engine(*args, **kw):
    """The engine of these tests pins the bare stop: no summary turn after it (that one is covered in test_codeagent_finalize.py)."""
    eng, repo, out = _engine(*args, **kw)
    return no_summary(eng), repo, out

GARBLED = "<function=read_file>\n<parameter=path"
CUT = {"role": "assistant", "content": '{"tool": "write_file", "arguments": {"path": "a.py", "content": "def f():', "_done_reason": "length", "_eval_count": 999}
THINK_ONLY = {"role": "assistant", "content": "", "thinking": "let me think about what to do next " * 20, "_prompt_eval_count": 1300, "_eval_count": 40}
THINK_TAGS = say("<think>hmm, what now</think>")


def _reminders(eng):
    return [m["content"] for m in eng.ctx.messages if m["role"] == "system"
            and any(k in m["content"] for k in ("could not be parsed", "reply was empty", "cut off after"))]


def test_n_garbled_replies_in_a_row_end_the_segment_as_malformed(tmp_path):
    llm = FakeLLM([say(GARBLED)] * 8)
    eng, _repo, out = engine(tmp_path, llm)
    res = eng.turn("anything")
    assert res.stopped == "error" and res.reason == "malformed" and res.text is None
    assert len(llm.calls) == 5                                   # default malformed_abort = 5, no step burnt after it
    assert len(_reminders(eng)) == 1                             # ONE reminder for the streak
    assert eng.events.of("malformed_abort")[0]["n"] == 5
    assert "unusable replies" in out.getvalue()
    assert "model produced unusable replies" in eng.dead_ends.text().replace(":", "")      # the restart sees the dead end


def test_below_n_the_turn_recovers_and_gets_one_reminder(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([say(GARBLED)] * 4 + [say("Final.")]))
    res = eng.turn("anything")
    assert res.text == "Final." and res.stopped == "" and len(_reminders(eng)) == 1


def test_a_readable_call_starts_a_new_streak_with_a_new_reminder(tmp_path):
    llm = FakeLLM([say(GARBLED)] * 4 + [call("list_dir", path=".")] + [say(GARBLED)] * 4 + [say("Final.")])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("anything")
    assert res.text == "Final." and res.stopped == "" and len(_reminders(eng)) == 2


def test_thinking_only_replies_count(tmp_path):
    llm = FakeLLM([dict(THINK_ONLY), THINK_TAGS, dict(THINK_ONLY), THINK_TAGS, dict(THINK_ONLY), say("never reached")])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("anything")
    assert res.stopped == "error" and res.reason == "malformed" and len(llm.calls) == 5


def test_empty_replies_count_and_the_first_blank_is_a_silent_reroll(tmp_path):
    llm = FakeLLM([say("")] * 9)
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("anything")
    assert res.reason == "malformed" and len(llm.calls) == 5
    assert len(_reminders(eng)) == 1 and len(llm.calls[1]["snapshot"]) == len(llm.calls[0]["snapshot"])      # blank 1: same context again


def test_mixed_unusable_kinds_share_one_streak_and_one_reminder(tmp_path):
    llm = FakeLLM([say(""), say(GARBLED), dict(CUT), say(""), say(GARBLED), say("never reached")])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("anything")
    assert res.reason == "malformed" and len(llm.calls) == 5 and len(_reminders(eng)) == 1


def test_a_cut_off_reply_is_no_longer_taken_as_the_answer_after_three(tmp_path):
    llm = FakeLLM([dict(CUT)] * 4 + [say("Final.")])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("write a.py")
    assert res.text == "Final." and len(_reminders(eng)) == 1 and "cut off after 999 tokens" in _reminders(eng)[0]


def test_malformed_abort_is_configurable(tmp_path):
    llm = FakeLLM([say(GARBLED)] * 8)
    eng, _repo, _ = engine(tmp_path, llm)
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "malformed_abort": 3}}
    res = eng.turn("anything")
    assert res.reason == "malformed" and len(llm.calls) == 3


def test_plain_prose_is_an_answer_not_a_malformed_reply(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([say("Nothing to change here.")]))
    res = eng.turn("look")
    assert res.text == "Nothing to change here." and res.stopped == "" and not eng.events.of("malformed_abort")


def test_stage_runner_reports_the_reason(tmp_path):
    import types

    from agent.codeagent import stages as stages_mod

    runner = types.SimpleNamespace(engine=types.SimpleNamespace(tools=types.SimpleNamespace(syntax_bad={}, static_bad={}, evidence=[])))
    why = stages_mod._Runner.not_green(runner, types.SimpleNamespace(stopped="error", reason="malformed", text=None, not_verified=[], verified=None))
    assert why == "stopped: error (malformed)"
