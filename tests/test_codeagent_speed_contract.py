"""The speed contract: numbers and behaviours that were each MEASURED on a real run and that keep a long run fast. A change to any of them
makes a test here fail ON PURPOSE - change the constant and its test together, in one commit, and say in the message what you measured.

  MASK .72 / COMPACT .80 / EMERGENCY .92 / FOLD .72   context fill levels (context.py). A mask or compaction makes the server re-read the
        whole prompt (measured 50-77 s for ~50k tokens), so they must come late and rarely; FOLD was 0.45 once and a resumed session
        folded at once and re-read the whole code base.
  STAGE_MASK_AT .70 / STAGE_MASK_GAP .40 / STAGE_MASK_TARGET .50 / MASK_KEEP_TOKENS .20 / MASK_KEEP_TURNS 2 / STAGE_MASK_MIN_GAIN .25 /
  STAGE_MASK_WINDOW 8   inside a stage (cap set) the budget is min(window, stage cap) and old tool results are stubbed in ONE batch (oldest
        first, down to 50 % of the cap, keeping the newest result and a working set of 20 % of the cap; 2 turns is only a floor) at 70 % of it, only
        if that frees >= 25 % of the prompt, not again before the prompt grew by 40 % of the cap; two fruitless batches within 8 steps, or one
        needed again within 6 steps, end the segment (coding.stage_mask_at, stage_mask_gap, stage_mask_target, mask_keep_tokens, mask_keep_turns,
        stage_mask_window). Measured (second run): keep-6-turns batches freed only ~3k of 29-34k (29198->26008, 34315->26725, 33758->27223),
        the prompt stayed at the cap, every mask re-read ~26k tokens: 11 big prefills = 135 s in 8 min, prefill 46 % of model time.
        Earlier measurement: Measured: the old line (.72 x 96k = 69k) was never reached below the 28k cap, 6 caps in 11 min, 82
        reads vs 14 edits, every restart a full prompt re-read (9 in 13 min) while normal steps hit the server's prompt cache (LCP 0.9+).
        Every mutation of the prefix costs that cache, so it happens rarely and in bulk.
  NUM_PREDICT 8192      output cap of one reply (~27 KB of code; bigger files go in chunks). Runaway generations of 16k and 32k tokens (14 min each)
        set the first cap; one 16k generation in a real run (a crashing curses game, step 2) still cost ~5 min at 54 tok/s, so it is 8192 now.
  SPRAWL 6000 / 2000    a generation past 6000 tokens with no tool call or code fence open or closed in the last 2000 is stopped (sprawl.py).
  THINK_PREDICT 4000    hard cap of a reasoning pass. One forced pass ran 8.5k tokens = 4.7 min and changed nothing.
  stage cap 28000       absolute prompt cap of a stage segment (stages.py). One long session grew to a 78k prompt and prefill ate 35-84 %
        of the wall time (2.5 steps/min instead of 7-9).
  LOOK_BLOCK_AT 16      look-steps in a row after which reading is paused (lookact.py). Measured decision: it was 25 and read/search
        streaks up to 54 were observed in a real run (109 look_blocked events), so 25 came too late. The first 6 steps of a stage are free
        (initial orientation reads do not count).
  No look/wander guard may force a reasoning pass (measured live: one took 4.7 min); only the edit_gap guard (and the loop / paging /
        failing-test guards of the engine, which fire after a measured repetition) may.
  The command-family and read-only-run-spam guards (FAMILY 5/8 of the last 12, RUN_SPAM 10/14 of the last 20, loopguard.py) append one
        line and end the segment; they never set think_steps.
  A stage starts small: system + brief + goal + verified state + notes stay under 8000 tokens even for a repo with 300 files."""

from __future__ import annotations

import pytest

from agent.codeagent import context as C, engine as E, lookact as L, loopguard as LG, stages as ST, verified
from agent.codeagent.context import est_tokens

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import new_engine, TASK3


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("AGENT_CODE_STAGE_PROMPT_CAP", raising=False)
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def test_the_measured_defaults_are_unchanged():
    assert (C.MASK_AT, C.COMPACT_AT, C.EMERGENCY_AT, C.FOLD_AT) == (0.72, 0.80, 0.92, 0.72)
    assert (E.NUM_PREDICT, E.THINK_PREDICT) == (8192, 4000)
    assert ST.DEFAULT_CAP == 28000
    assert (C.STAGE_MASK_AT, C.STAGE_MASK_GAP, C.MASK_KEEP_TURNS) == (0.70, 0.40, 2)
    assert (C.STAGE_MASK_TARGET, C.MASK_KEEP_TOKENS, C.STAGE_MASK_MIN_GAIN, C.STAGE_MASK_WINDOW, C.STAGE_MASK_RETRY_STEPS) == (0.50, 0.20, 0.25, 8, 6)
    assert C.STAGE_MASK_AT * ST.DEFAULT_CAP < ST.DEFAULT_CAP and C.MASK_AT * 96 * 1024 > ST.DEFAULT_CAP     # why: the old line lay above the cap
    assert L.LOOK_BLOCK_AT == 16 and L.FREE_STAGE_STEPS == 6


def test_the_stage_runner_uses_the_default_cap_unless_configured(tmp_path):
    eng, _repo, _o = new_engine(tmp_path, FakeLLM([]))
    plan = ST.StagePlan([ST.Stage(1, "t", "x")], 0, "single", "")
    assert ST.make_runner(eng, plan).cap == 28000


def test_look_and_wander_guards_never_ask_for_a_reasoning_pass():
    lines = {"n": 0}
    la = L.LookAct(read_lines=lambda: lines["n"], reset_lines=lambda: lines.update(n=0))
    seen = []
    for i in range(L.LOOK_BLOCK_AT + 15):                              # past the step nudge (10, 20 ...), the pause (16) and beyond
        lines["n"] = i * 40 if i >= 30 else 0                         # and past the line-volume nudge (700 lines)
        seen += la.observe("read_file", {"path": "a.py"}, "text")
    assert la.look_streak >= L.LOOK_BLOCK_AT and [r.event for r in seen].count("wander") >= 3
    assert not any(r.think for r in seen if r.event != "edit_gap")
    assert la.blocked("read_file", {"path": "a.py"})                  # the pause itself is not a reaction and reasons nothing either
    other = L.LookAct()
    other.pause()
    assert not any(r.think for r in other.observe("search", {"pattern": "x"}, "hits"))


def test_engine_run_past_every_look_threshold_enables_no_thinking(tmp_path):
    n = L.LOOK_BLOCK_AT + 8
    llm = FakeLLM([call("read_file", path=f"f{i}.py") for i in range(n)] + [say("I only read.")])      # distinct files: no repeat-loop guard
    eng, repo, _o = new_engine(tmp_path, llm)
    for i in range(n):
        (repo / f"f{i}.py").write_text("x = 1\n" * 50 + f"# {i}\n")
    eng.pol.max_steps, eng.pol.steps_set = n + 5, True
    eng.turn("read a lot")
    assert len(eng.events.of("wander")) >= 2 and eng.events.of("look_blocked")          # the guards did fire ...
    assert all(c["think"] is False for c in llm.calls)                               # ... and not one request enabled thinking
    assert all(c["opts"]["num_predict"] == 8192 for c in llm.calls)


def test_the_only_lookact_reaction_that_thinks_is_edit_gap():
    la = L.LookAct(read_lines=lambda: 5000, reset_lines=lambda: None)
    reactions = []
    for i in range(40):
        reactions += la.observe("read_file", {"path": "a.py"}, "text")                         # wander, far past every threshold
    la.observe("write_file", {"path": "a.py"}, "wrote a.py (3 lines)")
    for i in range(L.TEST_STREAK_NUDGE * 3):
        reactions += la.observe("run", {"command": f"python3 a.py {i}"}, "exit code 0")        # edit_gap, twice at most
    assert {r.event for r in reactions} == {"wander", "edit_gap"}
    assert {r.event for r in reactions if r.think} == {"edit_gap"}
    assert sum(r.event == "edit_gap" for r in reactions) == 2                                  # bounded: at 15 and 30, never again


def test_a_stage_starts_small_even_in_a_repo_with_300_files(tmp_path):
    eng, repo, _o = new_engine(tmp_path, FakeLLM([]))
    for d in range(15):
        sub = repo / "pkg" / f"mod{d:02d}"
        sub.mkdir(parents=True)
        for i in range(20):
            (sub / f"file_{i:02d}.py").write_text(f"def f{i}():\n    return {i}\n")
    (repo / "README.md").write_text("# demo\n")
    assert sum(1 for p in repo.rglob("*") if p.is_file()) >= 300
    plan = ST.StagePlan([ST.Stage(1, "t", "Overall task: x\n\ndo the first part of the work")], 0, "single", TASK3)
    runner = ST.make_runner(eng, plan)                                                # pins the brief card of the whole task
    st = plan.stages[0]
    state = runner.measure()
    eng.begin_stage(st.goal, state.text, runner.notes_for(st), reset_plan=True)
    tokens = sum(est_tokens(str(m.get("content", ""))) for m in eng.ctx.messages)
    assert 0 < tokens < 8000, tokens
    roles = [m["role"] for m in eng.ctx.messages]
    assert roles.count("user") == 1 and any("STAGE GOAL" in str(m["content"]) for m in eng.ctx.messages)


def test_family_and_run_spam_guards_never_set_think_steps(tmp_path):
    assert (LG.FAMILY_NOTE, LG.FAMILY_ABORT, LG.RUN_SPAM_NOTE, LG.RUN_SPAM_ABORT) == (5, 8, 10, 14)
    eng = new_engine(tmp_path, FakeLLM([]))[0]
    eng.family.reset_stage()
    for i in range(30):
        args = {"command": f"python3 -c \"import re; open('tests/x.py'); print({i})\""}
        eng._loop_guard("run", args, f"exit code 0\nout {i}" + "z" * i)
        assert eng.think_steps == 0
    assert eng._abort_loop
