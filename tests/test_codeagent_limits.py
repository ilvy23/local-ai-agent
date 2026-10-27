"""Lifetime of the engine's limits (B4: nudges, malformed calls, announcements, budget extensions are scoped per finish attempt /
stage segment and reset by progress; open acceptance items keep blocking the end) and the tunable coding.* keys (B5)."""
from __future__ import annotations

import types

import pytest

from agent import plan as plan_mod
from agent.codeagent import engine as E
from tests.test_codeagent_engine import CFG, FakeLLM, _isolated, call, engine, say  # noqa: F401 - _isolated is an autouse fixture


def plan_of(*texts):
    return plan_mod.Plan(task="t", items=[plan_mod.Item(i, t) for i, t in enumerate(texts, 1)])


def with_coding(eng, **kw):
    eng.config = {**CFG, "coding": {**CFG["coding"], **kw}}


def sys_notes(eng, needle):
    return [m["content"] for m in eng.ctx.messages if m["role"] == "system" and needle in m["content"]]


# ── B4.1 gate nudges: per finish attempt, reset by progress, open items keep blocking ──────────────

def test_gate_nudges_are_reset_by_progress_so_a_later_real_problem_is_still_nudged(tmp_path):
    claim = "It works and is tested."
    llm = FakeLLM([
        call("write_file", path="app.py", content="VALUE = 1\n"),
        say(claim), say(claim), say(claim), say(claim), say(claim),           # one bounce (open items), then verify x2, plan x1, claims x1: the budget of 4
        call("write_file", path="other.py", content="print(undefined_name)\n"),   # progress, and a new real problem (static error)
        *[say(claim)] * 12,
    ])
    eng, repo, _ = engine(tmp_path, llm, test_cmd="python3 check.py")
    (repo / "check.py").write_text("import app\nassert app.VALUE == 2\n")
    eng.plan.plan = plan_of("a", "b")
    eng.config = {**CFG, "coding": {"claims_check": True}}
    res = eng.turn("make check.py pass")
    assert sys_notes(eng, "the static check finds errors")                    # old: 4 nudges used up early -> accepted without a word
    assert res.stopped == "gate"


def test_open_acceptance_items_end_the_run_as_gate_not_as_silently_accepted(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([say("Everything is finished.")] * 6))
    eng.plan.plan = plan_of("first thing", "second thing")
    res = eng.turn("do both")
    assert res.stopped == "gate" and res.text.startswith("NOT VERIFIED: ") and "acceptance item 1" in res.text
    assert res.text.endswith("Everything is finished.") and res.verified is False
    assert eng.last_stopped == "gate"
    assert not any("NOT VERIFIED" in m["content"] for m in eng.ctx.messages if m["role"] == "assistant")     # stored message stays unprefixed


def test_confirmed_items_do_not_trigger_the_gate_stop(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="X = 1\n"), call("mark_done", ids=[1, 2], evidence="python3 a.py exit 0 prints nothing"),
                   say("Done.")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.plan.plan = plan_of("a", "b")
    res = eng.turn("x")
    assert res.stopped == "" and res.text == "Done."


def test_gate_is_an_unproductive_stop_for_the_stage_runner(tmp_path):
    from agent.codeagent import stages as stages_mod

    runner = types.SimpleNamespace(engine=types.SimpleNamespace(tools=types.SimpleNamespace(syntax_bad={}, static_bad={}, evidence=[])))
    why = stages_mod._Runner.not_green(runner, types.SimpleNamespace(stopped="gate", text="NOT VERIFIED: x", not_verified=["x"], verified=False))
    assert why == "stopped: gate (x)"                       # the unmet items ride along in the next segment's note


# ── B4.2 malformed-call counter ─────────────────────────────────────────────────────────────────────

def test_malformed_counter_recovers_after_a_successful_tool_call(tmp_path):
    garbled = "<function=read_file>\n<parameter=path"
    llm = FakeLLM([say(garbled)] * 3 + [call("list_dir", path=".")] + [say(garbled)] * 3 + [say("Final.")])
    eng, repo, _ = engine(tmp_path, llm)
    res = eng.turn("anything")
    assert res.text == "Final." and len(eng.events.of("malformed_call")) == 6      # old: the 4th garbled reply was accepted as the answer
    assert eng.events.of("malformed_call")[3]["n"] == 1                              # the counter started again


def test_malformed_counter_is_not_reset_by_nothing(tmp_path):
    garbled = "<function=read_file>\n<parameter=path"
    eng, repo, _ = engine(tmp_path, FakeLLM([say(garbled)] * 5))
    eng.config = {**eng.config, "coding": {**(eng.config.get("coding") or {}), "forced_summary": False}}      # the bare stop (the summary: test_codeagent_finalize.py)
    res = eng.turn("anything")
    assert len(eng.events.of("malformed_call")) == 5 and res.stopped == "error" and res.reason == "malformed" and res.text is None   # bounded without a successful call in between


# ── B4.3 announcements: structural rule ─────────────────────────────────────────────────────────────

def test_a_pure_text_answer_with_open_plan_items_is_an_announcement_whatever_the_wording(tmp_path):
    llm = FakeLLM([say("The module is complete and tidy."), call("write_file", path="a.py", content="X = 1\n"),
                   call("mark_done", ids=[1], evidence="python3 a.py exit 0 prints nothing"), say("Done.")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.plan.plan = plan_of("a.py exists")
    res = eng.turn("x")
    assert res.text == "Done." and len(eng.events.of("announce_continue")) == 1 and (repo / "a.py").exists()


def test_a_staged_run_with_errors_counts_a_pure_text_answer_as_an_announcement(tmp_path):
    from agent.codeagent import stages as stages_mod

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    assert eng._open_work() is None                                   # no plan, no stage plan: the wording decides
    eng.stage_plan = stages_mod.StagePlan([stages_mod.Stage(1, "one", "goal", status="active")])
    assert eng._open_work() == []                                     # a stage without errors: a text answer is a final answer
    eng.tools.syntax_bad["a.py"] = "line 1: invalid"
    assert "a.py has errors" in eng._open_work()
    msg = eng._announced_next("All good here.", 5)
    assert msg and "answered without a tool call" in msg and "a.py has errors" in msg


def test_without_plan_or_stage_info_the_wording_heuristic_still_applies(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([say("Now update main.py ..."), say("Done.")]))
    assert eng._open_work() is None
    res = eng.turn("x")
    assert len(eng.events.of("announce_continue")) == 1 and not eng.events.of("announce_continue")[0]["structural"] and res.text == "Done."


def test_neutral_text_with_no_plan_is_never_an_announcement(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([say("Everything is finished.")]))
    assert eng.turn("x").text == "Everything is finished." and not eng.events.of("announce_continue")


def test_announce_limit_is_per_attempt_and_reset_by_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "ANNOUNCE_MAX", 1)
    llm = FakeLLM([say("Not yet."), call("write_file", path="a.py", content="X = 1\n"), say("Not yet."),
                   call("write_file", path="b.py", content="Y = 1\n"), say("Not yet."), say("Not yet."), say("Not yet."), say("Not yet.")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.plan.plan = plan_of("a", "b", "c")
    eng.turn("x")
    assert len(eng.events.of("announce_continue")) == 3          # old: one bounce, then the budget of the whole prompt was gone


def test_announce_limit_without_progress_stays_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "ANNOUNCE_MAX", 2)
    steps = [x for i in range(6) for x in (say("Not yet."), call("list_dir", path=f"d{i}"))]
    eng, repo, _ = engine(tmp_path, FakeLLM([*steps, say("end")]))
    eng.plan.plan = plan_of("a", "b")
    eng.turn("x")
    assert len(eng.events.of("announce_continue")) == 2


# ── B4.4 auto extension per segment ─────────────────────────────────────────────────────────────────

def test_auto_extension_count_is_reset_by_real_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "AUTO_EXTEND_MAX", 1)
    writes = [call("write_file", path=f"m{i}.py", content=f"X{i} = {i}\n") for i in range(12)]
    eng, repo, out = engine(tmp_path, FakeLLM([*writes, say("twelve modules written")]), max_steps=5)      # 5 is the policy minimum
    res = eng.turn("write twelve modules")
    assert res.stopped == "" and res.steps == 13                  # old: stopped at step 10 ("the automatic extensions are used up")


def test_auto_extension_still_stops_when_the_writes_change_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "AUTO_EXTEND_MAX", 1)
    same = [call("write_file", path="m.py", content="X = 1\n") for _ in range(10)]
    eng, repo, out = engine(tmp_path, FakeLLM(same), max_steps=5)
    res = eng.turn("rewrite the same file")
    assert res.stopped == "budget" and "automatic extensions are used up" in " ".join(out.getvalue().split())


# ── B5.2 coding.* keys ──────────────────────────────────────────────────────────────────────────────

def test_num_predict_and_think_predict_come_from_the_config(tmp_path):
    llm = FakeLLM([say("hello"), say("hello")])
    eng, repo, _ = engine(tmp_path, llm)
    with_coding(eng, num_predict=999, think_predict=77)
    eng.turn("x")
    assert llm.calls[0]["opts"]["num_predict"] == 999
    eng.think_steps = 1
    eng.turn("y")
    assert llm.calls[1]["think"] is True and llm.calls[1]["opts"]["num_predict"] == 77


def test_defaults_are_unchanged_without_overrides(tmp_path):
    llm = FakeLLM([say("hello")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.turn("x")
    assert llm.calls[0]["opts"]["num_predict"] == E.NUM_PREDICT == 8192 and E.THINK_PREDICT == 4000
    assert (E.MAX_GATE_NUDGES, E.GATE_CONSECUTIVE, E.ANNOUNCE_MAX, E.AUTO_EXTEND_MAX, E.LOOP_HINT, E.LOOK_BLOCK_AT) == (12, 4, 8, 15, 3, 16)   # MAX_GATE_NUDGES 4 -> 12: now the hard ceiling, GATE_CONSECUTIVE is the cap
    assert eng.lookact.block_at == 16


def test_look_block_at_comes_from_the_config(tmp_path):
    reads = [call("list_dir", path=f"d{i}") for i in range(6)]
    eng, repo, _ = engine(tmp_path, FakeLLM([*reads, say("end")]))
    with_coding(eng, look_block_at=3)
    for i in range(6):
        (repo / f"d{i}").mkdir()
    eng.turn("look around")
    tools = [m["content"] for m in eng.ctx.messages if m["role"] == "tool"]
    assert sum("reading is paused" in t for t in tools) >= 2
    (tmp_path / "p").mkdir()
    plain, repo2, _o = engine(tmp_path / "p", FakeLLM([*reads, say("end")]))
    for i in range(6):
        (repo2 / f"d{i}").mkdir()
    plain.turn("look around")
    assert not any("reading is paused" in m["content"] for m in plain.ctx.messages if m["role"] == "tool")


def test_loop_repeat_comes_from_the_config(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    for _ in range(2):
        eng._loop_guard("read_file", {"path": "a.py"}, "x = 1")
    assert not sys_notes(eng, "identical")                         # default: the hint comes at 3
    with_coding(eng, loop_repeat=2)
    eng.ctx.step += 10
    eng._loop_guard("read_file", {"path": "a.py"}, "x = 1")
    assert sys_notes(eng, "identical")


def test_gate_nudges_announce_max_and_auto_extend_max_come_from_the_config(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    with_coding(eng, gate_nudges=1, announce_max=2, auto_extend_max=3, junk="x")
    assert (eng._limit("gate_nudges", E.MAX_GATE_NUDGES), eng._limit("announce_max", E.ANNOUNCE_MAX),
            eng._limit("auto_extend_max", E.AUTO_EXTEND_MAX)) == (1, 2, 3)
    with_coding(eng, gate_nudges="nonsense", announce_max=None, gate_consecutive="x")
    assert eng._limit("gate_consecutive", 4) == 4
    assert eng._limit("gate_nudges", 4) == 4 and eng._limit("announce_max", 8) == 8       # a broken value falls back to the default
    with_coding(eng, auto_extend_max=0)
    assert eng._limit("auto_extend_max", 15) == 0


def test_auto_extend_max_zero_in_config_disables_the_extension(tmp_path):
    writes = [call("write_file", path=f"m{i}.py", content=f"X{i} = {i}\n") for i in range(8)]
    eng, repo, _ = engine(tmp_path, FakeLLM([*writes, say("x")]), max_steps=2)
    with_coding(eng, auto_extend_max=0)
    assert eng.turn("write").stopped == "budget"


# ── B5.1 --resume ID parsing ────────────────────────────────────────────────────────────────────────

def _cli(monkeypatch, ids):
    from agent import main
    from agent.codeagent import session as S, sessionstore

    seen = []
    monkeypatch.setattr(S, "run_session", lambda *a, **kw: seen.append(a) or 0)
    monkeypatch.setattr(sessionstore, "list_sessions", lambda *a, **k: [types.SimpleNamespace(id=i) for i in ids])
    return main, seen


def test_resume_word_that_looks_like_hex_is_task_text_when_no_such_session_exists(monkeypatch):
    from typer.testing import CliRunner

    main, seen = _cli(monkeypatch, ["3fa91c2e"])
    CliRunner().invoke(main.app, ["code", "--old-ui", "--resume", "fade"])
    assert seen[0][1] == "fade" and seen[0][5] == "latest"
    CliRunner().invoke(main.app, ["code", "--old-ui", "--resume", "fade the lights"])
    assert seen[1][1] == "fade the lights" and seen[1][5] == "latest"
    CliRunner().invoke(main.app, ["code", "--old-ui", "--resume", "dead"])
    assert seen[2][1] == "dead" and seen[2][5] == "latest"


def test_resume_with_a_real_session_id_or_prefix_still_resolves(monkeypatch):
    from typer.testing import CliRunner

    main, seen = _cli(monkeypatch, ["3fa91c2e", "cafe0123"])
    CliRunner().invoke(main.app, ["code", "--old-ui", "--resume", "3fa9"])
    assert seen[0][1] is None and seen[0][5] == "3fa9"
    CliRunner().invoke(main.app, ["code", "--old-ui", "--resume", "cafe"])               # a word that IS a saved id prefix is an id
    assert seen[1][1] is None and seen[1][5] == "cafe"
    CliRunner().invoke(main.app, ["code", "--old-ui", "--resume", "continue"])
    assert seen[2][1] == "continue" and seen[2][5] == "latest"


# ── A: the gate cap counts CONSECUTIVE nudges without a tool call (Claude Code Stop-hook rule), gate_nudges is the hard ceiling ──────

CLAIM = "It works and is tested."


NUDGE_STARTS = ("You are not done", "VERIFICATION FAILED", "ACCEPTANCE LIST", "Your answer claims")


def gate_nudge_count(eng):
    return len([m for m in eng.ctx.messages if m["role"] == "system" and m["content"].startswith(NUDGE_STARTS)])


def gate_engine(tmp_path, steps, **coding):
    eng, repo, _ = engine(tmp_path, FakeLLM(steps), test_cmd="python3 check.py")
    (repo / "check.py").write_text("import app\nassert app.VALUE == 2\n")
    eng.plan.plan = plan_of("a", "b")
    eng.config = {**CFG, "coding": {"claims_check": True, **coding}}
    return eng, repo


def test_a_tool_call_after_a_nudge_resets_the_consecutive_count_so_more_than_four_nudges_are_possible(tmp_path):
    # one bounce, static x2, verify x1 (3 nudges in a row), a read_file, then verify, plan, claims: 6 nudges in total, never 4 in a row
    steps = [call("write_file", path="app.py", content="VALUE = 1\n"), call("write_file", path="other.py", content="print(undefined_name)\n"),
             say(CLAIM), say(CLAIM), say(CLAIM), say(CLAIM), call("read_file", path="app.py"), *[say(CLAIM)] * 8]
    eng, repo = gate_engine(tmp_path, steps)
    res = eng.turn("make check.py pass")
    assert len(sys_notes(eng, "the static check finds errors")) == 2
    assert sys_notes(eng, "Your answer claims things")                              # the 6th nudge: the old cap of 4 per attempt would have ended before it
    assert gate_nudge_count(eng) >= 6 and res.stopped == "gate" and eng._gate_streak <= 4


def test_text_only_replies_hit_the_consecutive_cap(tmp_path):
    eng, repo = gate_engine(tmp_path, [call("write_file", path="app.py", content="VALUE = 1\n"), *[say(CLAIM)] * 20])
    res = eng.turn("make check.py pass")
    assert res.stopped == "gate" and res.text.startswith("NOT VERIFIED: ")
    assert eng._gate_streak == E.GATE_CONSECUTIVE == 4
    assert len(eng.llm.calls) == 1 + 1 + E.GATE_CONSECUTIVE + 1       # write, the announce bounce, 4 nudged replies, the reply that is let through


def test_the_consecutive_cap_is_configurable(tmp_path):
    eng, repo = gate_engine(tmp_path, [call("write_file", path="app.py", content="VALUE = 1\n"), *[say(CLAIM)] * 20], gate_consecutive=2)
    res = eng.turn("make check.py pass")
    assert res.stopped == "gate" and eng._gate_streak == 2


def test_gate_nudges_is_the_hard_ceiling_even_when_the_model_keeps_calling_tools(tmp_path):
    steps = [call("write_file", path="app.py", content="VALUE = 1\n"), *[call("write_file", path=f"f{i}.py", content=f"X{i} = {i}\n") for i in range(8)]]
    for i in range(8):
        steps += [say(CLAIM), call("read_file", path=f"f{i}.py")]                # every nudge is answered with a real call: the streak never grows
    eng, repo = gate_engine(tmp_path, steps + [say(CLAIM)] * 5, gate_nudges=3, announce_max=0)
    res = eng.turn("make check.py pass")
    assert res.stopped == "gate"
    assert eng._gate_streak <= 1 and gate_nudge_count(eng) <= 3


def test_a_neutral_call_does_not_reset_the_consecutive_count(tmp_path):
    steps = [call("write_file", path="app.py", content="VALUE = 1\n"), say(CLAIM), say(CLAIM), call("mark_done", ids=[99], evidence="nothing"), *[say(CLAIM)] * 10]
    eng, repo = gate_engine(tmp_path, steps)
    eng.turn("make check.py pass")
    assert eng._gate_streak == E.GATE_CONSECUTIVE


def test_the_checks_gate_honours_the_consecutive_cap(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    from agent.codeagent import checks as checks_mod
    eng.checks.add(checks_mod.Check("c1", "always fails", "command", {"command": "false"}, "model", []))
    res = types.SimpleNamespace(verified=True, not_verified=[], stopped="", reason="")
    eng._gate_streak = 0
    assert eng._checks_gate(res, spent=False) is not None
    res2 = types.SimpleNamespace(verified=True, not_verified=[], stopped="", reason="")
    eng._check_returns = {}
    assert eng._checks_gate(res2, spent=True) is None and res2.not_verified and res2.verified is False
