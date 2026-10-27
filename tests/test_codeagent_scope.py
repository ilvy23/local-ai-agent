"""Measured on real runs of the stage runner: stages like 'rewrite ALL texts' never fit one context (broad split at
stage start), a stage was marked done after changing one file of five (scope gate), a stage ended with a red test (stage tests), and a
deleted symbol that other files still used went unnoticed (removed-symbol note). FakeLLM only."""

from __future__ import annotations

import json

import pytest

from agent.codeagent import profile as PR, stages as ST, verified
from agent.codeagent.tools import CodeTools

from tests.test_codeagent_context import cm
from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import cfg, new_engine, request_text, subs_json, write_lines


def sub(scope, goal="Clear up the texts of these files and keep them connected", **kw):
    return ST.Stage(1, "Part", f"Overall task: x\n\n{goal}", scope=list(scope), parent=1, depth=1, **kw)


def sib(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def run_one(eng, st):
    return ST.run_stages(eng, ST.StagePlan([st], 0, "headings", "the task"))


# ── 3. broad-scope stages are split before the first segment ────────────────

BROAD = "Rewrite all game texts to be clear"
FIVE = subs_json(("Part A", "Clear the texts of the first two files", ["a.md", "b.md"]),
                 ("Part B", "Clear the texts of the last three files", ["c.md", "d.md", "e.md"]))


def broad_engine(tmp_path, steps, background, config=None, goal=BROAD, files=("a.md", "b.md", "c.md", "d.md", "e.md"), n=300):
    llm = FakeLLM(steps, background=background)
    eng, repo, out = new_engine(tmp_path, llm, config)
    for name in files:
        write_lines(repo, name, n)
    plan = ST.StagePlan([ST.Stage(1, "texts", goal)], 0, "single", goal)
    return llm, eng, repo, out, plan


def test_a_broad_goal_on_a_five_file_repo_is_split_before_any_model_step(tmp_path):
    steps = [call("write_file", path="a.md", content="a\n"), call("write_file", path="b.md", content="b\n"), say("A done."),
             call("write_file", path="c.md", content="c\n"), call("write_file", path="d.md", content="d\n"),
             call("write_file", path="e.md", content="e\n"), say("B done.")]
    llm, eng, repo, out, plan = broad_engine(tmp_path, steps, [FIVE])
    report = ST.run_stages(eng, plan)
    assert report.verdict == "done" and [(s.scope, s.depth, s.parent) for s in plan.stages] == [(["a.md", "b.md"], 1, 1), (["c.md", "d.md", "e.md"], 1, 1)]
    ev = eng.events.of("stage_resplit")
    assert len(ev) == 1 and ev[0]["source"] == "broad" and ev[0]["to"] == [1, 2]
    assert len(llm.calls) == 7 and llm.background == []                       # the first model step already ran in a sub-stage
    assert "Files of this part: a.md, b.md" in request_text(llm.calls[0]["snapshot"])


def test_a_broad_goal_falls_back_to_a_deterministic_split_naming_files(tmp_path):
    llm, eng, repo, out, plan = broad_engine(tmp_path, [], ["not json"])
    runner = ST.make_runner(eng, plan)
    seg = runner.segment(plan.stages[0])
    assert seg.why == "resplit" and llm.calls == []
    ev = eng.events.of("stage_resplit")
    assert len(ev) == 1 and ev[0]["source"] == "broad" and len(plan.stages) >= 2
    assert all(s.scope and s.depth == 1 for s in plan.stages) and sorted(f for s in plan.stages for f in s.scope) == ["a.md", "b.md", "c.md", "d.md", "e.md"]


@pytest.mark.parametrize("goal", ["Fix the typo in a.md", "Overall task: rewrite everything\n\nFix the typo in the intro of a.md"])
def test_a_narrow_goal_is_not_split(tmp_path, goal):
    llm, eng, repo, out, plan = broad_engine(tmp_path, [say("Fixed.")], [FIVE], goal=goal)
    assert ST.run_stages(eng, plan).verdict == "done"
    assert len(plan.stages) == 1 and eng.events.of("stage_resplit") == [] and llm.background == [FIVE]


@pytest.mark.parametrize("word", ["every", "EACH", "whole", "entire", "throughout", "alle", "jede", "jeden", "jedes", "sämtliche", "überall", "all"])
def test_the_broad_words(tmp_path, word):
    _llm, eng, _repo, _o, plan = broad_engine(tmp_path, [], [FIVE], goal=f"Rewrite {word} texts of the game")
    assert ST.make_runner(eng, plan).broad_applies(plan.stages[0])


def test_small_repos_and_the_switch_do_not_split(tmp_path):
    _l, eng, _r, _o, plan = broad_engine(tmp_path, [], [FIVE], files=("a.md", "b.md"), n=900)          # 2 files
    assert not ST.make_runner(eng, plan).broad_applies(plan.stages[0])
    _l, eng, _r, _o, plan = broad_engine(sib(tmp_path, "s"), [], [FIVE], n=100)                             # 5 files, 500 lines
    assert not ST.make_runner(eng, plan).broad_applies(plan.stages[0])
    _l, eng, _r, _o, plan = broad_engine(sib(tmp_path, "o"), [], [FIVE], config=cfg(broad_split=False))
    assert not ST.make_runner(eng, plan).broad_applies(plan.stages[0])


def test_an_already_split_stage_is_never_split_a_second_time(tmp_path):
    steps = [call("write_file", path="a.md", content="a\n"), call("write_file", path="b.md", content="b\n"), say("A done."),
             call("write_file", path="c.md", content="c\n"), call("write_file", path="d.md", content="d\n"),
             call("write_file", path="e.md", content="e\n"), say("B done.")]
    llm, eng, repo, out, plan = broad_engine(tmp_path, steps, [FIVE, FIVE])           # the sub-stage goals contain "all"/"Overall"
    runner = ST.make_runner(eng, plan)
    st = plan.stages[0]
    assert runner.broad_applies(st) and runner.resplit(st, "broad")
    assert not any(runner.broad_applies(s) for s in plan.stages) and not runner.should_resplit(plan.stages[0])
    ST.run_stages(eng, plan)
    assert len(eng.events.of("stage_resplit")) == 1 and len(llm.background) == 1


def test_the_broad_split_is_tried_once_per_stage(tmp_path):
    llm, eng, repo, out, plan = broad_engine(tmp_path, [say("done.")], [])
    runner = ST.make_runner(eng, plan)
    tried = []
    runner.resplit = lambda st, source=None: tried.append(source) or False               # the split is impossible
    seg = runner.segment(plan.stages[0])
    assert tried == ["broad"] and seg.kind == "green" and not runner.broad_applies(plan.stages[0])


# ── 4. the scope gate ───────────────────────────────────────────────────────

def test_a_sub_stage_is_not_green_until_every_named_file_was_changed(tmp_path):
    llm = FakeLLM([call("write_file", path="a.md", content="a\n"), say("All clear."),
                   call("write_file", path="b.md", content="b\n"), say("Now really all clear.")])
    eng, repo, out = new_engine(tmp_path, llm)
    for name in ("a.md", "b.md"):
        (repo / name).write_text("old\n")
    st = sub(["a.md", "b.md"])
    assert run_one(eng, st).verdict == "done" and st.files == ["a.md", "b.md"] and st.attempts == 1
    nudge = [m["content"] for m in llm.calls[2]["snapshot"] if m["role"] == "system" and "not handled" in str(m["content"])]
    assert nudge and "b.md" in nudge[0] and "a.md" not in nudge[0].split("not handled")[1].split("\n")[0]


def test_a_file_that_needs_no_change_can_be_waived_with_evidence_naming_it(tmp_path):
    llm = FakeLLM([call("write_file", path="a.md", content="a\n"), say("All clear."),
                   call("mark_done", id=1, evidence="b.md was read and needs no change, it is already clear"), say("All clear, b.md needs no change.")])
    eng, repo, out = new_engine(tmp_path, llm)
    for name in ("a.md", "b.md"):
        (repo / name).write_text("old\n")
    st = sub(["a.md", "b.md"])
    assert run_one(eng, st).verdict == "done" and st.files == ["a.md"] and st.waived == ["b.md"]
    assert (repo / "b.md").read_text() == "old\n"


def test_evidence_that_does_not_name_the_file_waives_nothing(tmp_path):
    llm = FakeLLM([call("write_file", path="a.md", content="a\n"), say("Done."),
                   call("mark_done", id=1, evidence="everything is clear already, nothing else to do"), say("Done."), say("Done."), say("Done.")])
    eng, repo, out = new_engine(tmp_path, llm, cfg(stage_attempts=1))
    st = sub(["a.md", "b.md"])
    report = run_one(eng, st)
    assert report.verdict == "failed" and st.status == "blocked" and "files not handled: b.md" in st.reason and st.waived == []


def test_unhandled_files_end_the_segment_as_gate_not_verified(tmp_path):
    llm = FakeLLM([call("write_file", path="a.md", content="a\n")] + [say("Done.")] * 4)
    eng, repo, out = new_engine(tmp_path, llm)
    eng.stage_scope, eng.stage_goal = ["a.md", "b.md", "c.md"], "goal"
    res = eng.turn("goal")
    assert res.stopped == "gate" and res.verified is False and res.not_verified == ["files not handled: b.md, c.md"]
    assert res.text.startswith("NOT VERIFIED: files not handled: b.md, c.md")
    assert eng.events.of("gate_stop")


def test_the_waiver_survives_into_a_later_segment_and_stages_without_files_are_unaffected(tmp_path):
    eng, repo, out = new_engine(tmp_path, FakeLLM([say("Nothing to do here.")]))
    plain = ST.Stage(1, "t", "Explain the code")
    assert run_one(eng, plain).verdict == "done" and eng.events.of("gate_stop") == []
    assert eng.scope_missing() == []                                                         # no scope outside a stage segment
    st = sub(["a.md"], waived=["a.md"])
    llm2 = FakeLLM([say("a.md needs no change.")])
    eng2, _r, _o = new_engine(sib(tmp_path, "two"), llm2)
    assert run_one(eng2, st).verdict == "done"


def test_stage_json_keeps_scope_and_waived():
    plan = ST.StagePlan([sub(["a.md"], waived=["a.md"])], 0, "headings", "t")
    back = ST.StagePlan.from_json(json.loads(json.dumps(plan.to_json())))
    assert back.stages[0].scope == ["a.md"] and back.stages[0].waived == ["a.md"]


# ── 5. the stage runner runs the project's tests before it accepts a stage ──

# the stage-end rule is "a stage must not make things worse": the fake suite is red for every file red_<name> in the repo
# (FAILED tests/t.py::<name>), green otherwise; the scenario "the stage's edit of a.md breaks test_one" is the file red_test_one
SUITE = r'''echo run >> runs.log
n=0
for f in red_*; do
  [ -e "$f" ] || continue
  t=${f#red_}; n=$((n+1))
  echo "FAILED tests/t.py::$t - AssertionError"; echo "________ $t ________"; echo "E   assert $t == 1"
done
if [ $n -gt 0 ]; then echo "$n failed, 3 passed in 0.1s"; exit 1; fi
echo "5 passed in 0.1s"
'''


def engine_with_tests(tmp_path, steps, config=None, red=()):
    llm = FakeLLM(steps)
    eng, repo, out = new_engine(tmp_path, llm, config)
    (repo / "t.sh").write_text(SUITE)
    for name in red:
        (repo / f"red_{name}").write_text("x\n")
    eng.tools.profile.test_cmd = "sh t.sh"
    return llm, eng, repo, out


def runs(repo):
    f = repo / "runs.log"
    return len(f.read_text().split()) if f.exists() else 0


def two(a="Change a.md", b="Change b.md"):
    return ST.StagePlan([ST.Stage(1, "one", a), ST.Stage(2, "two", b)], 0, "headings", "the task")


def test_a_red_test_at_stage_end_is_not_green_and_the_next_note_names_the_failures(tmp_path):
    llm, eng, repo, out = engine_with_tests(tmp_path, [call("write_file", path="red_test_one", content="x\n"), say("Done."),
                                                    call("write_file", path="a.md", content="a\n"), say("Fixed.")])
    plan = two()                                            # stage 1 of 2: the non-strict rule applies
    st = plan.stages[0]
    real = eng.tools._exec
    eng.tools._exec = lambda cmd, timeout: ((repo / "red_test_one").unlink(missing_ok=True) if (repo / "a.md").exists() else None) or real(cmd, timeout)
    ST.run_stages(eng, plan, max_stages=1)
    assert st.status == "done" and st.attempts == 2
    note = request_text(llm.calls[2]["snapshot"])
    assert "tests/t.py::test_one" in note and "assert test_one == 1" in note and "passed before this stage" in note


def test_red_tests_count_as_an_attempt_and_block_at_the_limit(tmp_path):
    llm, eng, repo, out = engine_with_tests(tmp_path, [call("write_file", path="red_test_one", content="x\n"), say("Done.")], cfg(stage_attempts=1))
    plan = two()
    ST.run_stages(eng, plan, max_stages=1)
    st = plan.stages[0]
    assert st.status == "blocked" and st.attempts == 1 and "tests/t.py::test_one" in st.reason


def test_stage_tests_can_be_switched_off_and_no_test_command_is_fine(tmp_path):
    _l, eng, repo, out = engine_with_tests(tmp_path, [call("write_file", path="red_test_one", content="x\n"), say("Done.")], cfg(stage_tests=False))
    st = ST.Stage(1, "t", "Change a.md")
    assert run_one(eng, st).verdict == "done" and st.attempts == 1 and runs(repo) == 1       # only VERIFIED STATE's own run at the segment start; no baseline, no stage-end run
    eng2, repo2, _o = new_engine(sib(tmp_path, "n"), FakeLLM([say("Done.")]))
    st2 = ST.Stage(1, "t", "Explain")
    assert eng2.profile.test_cmd is None and run_one(eng2, st2).verdict == "done"


def test_green_tests_accept_the_stage_and_honour_the_timeout(tmp_path):
    _l, eng, repo, out = engine_with_tests(tmp_path, [call("write_file", path="a.md", content="x\n"), say("Done.")])
    assert run_one(eng, ST.Stage(1, "t", "Change a.md")).verdict == "done"
    seen = {}
    real = eng.tools._exec
    eng.tools._exec = lambda cmd, timeout: seen.update(timeout=timeout) or real(cmd, timeout)
    eng.config["coding"]["verified_timeout_s"] = 7
    verified.run_tests(eng)
    assert seen["timeout"] == 7


# ── 5b. baseline of known failures: a stage must not make things worse ────

def test_the_baseline_is_measured_once_at_the_first_stage_start_and_old_reds_do_not_block(tmp_path):
    _l, eng, repo, out = engine_with_tests(tmp_path, [call("write_file", path="a.md", content="a\n"), say("Done."),
                                                   call("write_file", path="b.md", content="b\n"), say("Done.")], red=["old_one", "old_two"])
    plan = two()
    ST.run_stages(eng, plan, max_stages=1)
    s1 = plan.stages[0]
    assert s1.status == "done" and s1.attempts == 1 and "2 tests were already red before this stage: tests/t.py::old_one, tests/t.py::old_two" in s1.summary
    assert sorted(plan.baseline_red) == ["tests/t.py::old_one", "tests/t.py::old_two"]


def test_a_new_red_blocks_with_only_the_new_ids_and_the_old_ones_stay_out_of_the_list(tmp_path):
    llm, eng, repo, out = engine_with_tests(tmp_path, [call("write_file", path="red_new_one", content="x\n"), say("Done."),
                                                    call("write_file", path="a.md", content="a\n"), say("Fixed.")], red=["old_one"])
    real = eng.tools._exec
    eng.tools._exec = lambda cmd, timeout: ((repo / "red_new_one").unlink(missing_ok=True) if (repo / "a.md").exists() else None) or real(cmd, timeout)
    plan = two()
    ST.run_stages(eng, plan, max_stages=1)
    s1 = plan.stages[0]
    assert s1.status == "done" and s1.attempts == 2
    note = request_text(llm.calls[2]["snapshot"])
    assert "tests/t.py::new_one" in note and "assert new_one == 1" in note and "passed before this stage" in note
    why = note[note.index("continuing:"):].split("see VERIFIED STATE")[0]          # the stage runner's own sentence (VERIFIED STATE lists the whole suite)
    assert "tests/t.py::old_one" not in why.split("1 other tests")[0] and "1 other tests are red from earlier work; they are not part of this stage" in note


def test_fixed_tests_leave_the_baseline_after_acceptance_and_a_later_regression_is_new(tmp_path):
    _l, eng, repo, out = engine_with_tests(tmp_path, [call("write_file", path="a.md", content="a\n"), say("Done."),
                                                   call("write_file", path="red_old_one", content="x\n"), say("Done.")], cfg(stage_attempts=1), red=["old_one"])
    real = eng.tools._exec
    eng.tools._exec = lambda cmd, timeout: ((repo / "red_old_one").unlink(missing_ok=True) if (repo / "a.md").exists() and not (repo / "b.md").exists()
                                            and plan.stages[0].status == "active" else None) or real(cmd, timeout)
    plan = two()
    ST.run_stages(eng, plan, max_stages=1)
    assert plan.stages[0].status == "done" and plan.baseline_red == []
    ST.run_stages(eng, plan, max_stages=1)               # stage 2 (last, strict) brings old_one back: it counts as new now
    assert plan.stages[1].status == "blocked" and "tests/t.py::old_one" in plan.stages[1].reason


def test_resume_restores_the_baseline_and_an_old_session_loads_without_it(tmp_path):
    _l, eng, repo, out = engine_with_tests(tmp_path, [say("Done.")], red=["old_one"])
    plan = two()
    runner = ST.make_runner(eng, plan)
    runner.ensure_baseline()
    assert plan.baseline_red == ["tests/t.py::old_one"] and runs(repo) == 1
    back = ST.StagePlan.from_json(json.loads(json.dumps(plan.to_json())))
    assert back.baseline_red == ["tests/t.py::old_one"] and back.baseline_ok
    ST.make_runner(eng, back).ensure_baseline()
    assert runs(repo) == 1                                  # restored, not measured again
    old = plan.to_json()
    del old["baseline_red"], old["baseline_ok"]
    legacy = ST.StagePlan.from_json(old)
    assert legacy.baseline_red is None and legacy.baseline_ok
    ST.make_runner(eng, legacy).ensure_baseline()
    assert legacy.baseline_red == ["tests/t.py::old_one"] and runs(repo) == 2


def test_a_baseline_run_without_result_falls_back_to_scope_names(tmp_path):
    llm = FakeLLM([call("write_file", path="engine.py", content="x = 1\n"), say("Done.")])
    eng, repo, out = new_engine(tmp_path, llm)
    (repo / "t.sh").write_text('echo run >> runs.log\nif [ -e engine.py ]; then echo "FAILED tests/test_engine.py::test_a"; echo "FAILED tests/test_other.py::test_b";'
                              ' echo "2 failed in 0.1s"; exit 1; fi\nsleep 5\n')
    eng.tools.profile.test_cmd = "sh t.sh"
    eng.config.setdefault("coding", {})["verified_timeout_s"] = 1
    plan = ST.StagePlan([ST.Stage(1, "one", "Change engine.py", scope=["engine.py"]), ST.Stage(2, "two", "later")], 0, "headings", "t")
    runner = ST.make_runner(eng, plan)
    runner.ensure_baseline()                                # engine.py does not exist yet: the run hits the 1s timeout
    assert plan.baseline_red == [] and plan.baseline_ok is False
    (repo / "engine.py").write_text("x = 1\n")
    why = runner.stage_tests(plan.stages[0])
    assert "tests/test_engine.py::test_a" in why and "test_other" not in why
    assert runner.stage_tests(ST.Stage(1, "x", "no scope")) == ""          # no scope names, no baseline: nothing can block


def test_the_last_stage_is_strict_unless_switched_off(tmp_path):
    for strict, expect in ((True, "blocked"), (False, "done")):
        d = sib(tmp_path, f"s{strict}")
        _l, eng, repo, out = engine_with_tests(d, [call("write_file", path="a.md", content="a\n"), say("Done.")], cfg(stage_attempts=1, stage_tests_strict_last=strict), red=["old_one"])
        st = ST.Stage(1, "only", "Change a.md")
        run_one(eng, st)
        assert st.status == expect, strict
        assert (strict is False) or "all must be green" in st.reason


def test_stage_tests_false_runs_nothing_at_all(tmp_path, monkeypatch):
    calls = []
    real = verified.run_tests
    monkeypatch.setattr(verified, "run_tests", lambda e: calls.append(1) or real(e))
    _l, eng, repo, out = engine_with_tests(tmp_path, [say("Done."), say("Done.")], cfg(stage_tests=False), red=["old_one"])
    plan = two()
    ST.run_stages(eng, plan)
    assert calls == [] and plan.baseline_red is None


# ── 6. a write that removes a symbol other files still use ──────────────────

def tools_for(tmp_path):
    return CodeTools(tmp_path, PR.detect(tmp_path), cm(tmp_path))


STATE = "class State:\n    def keep(self):\n        return 1\n\n    def mood_change(self, n):\n        return n\n"


def test_removing_a_symbol_that_another_file_uses_adds_one_factual_line(tmp_path):
    (tmp_path / "state.py").write_text(STATE)
    (tmp_path / "dialogue.py").write_text("from state import State\n\ndef talk(s):\n    return s.mood_change(2)\n")
    (tmp_path / "other.py").write_text("def f(s):\n    s.mood_change(1)\n")
    t = tools_for(tmp_path)
    t.read_file("state.py")
    res = t.replace_lines("state.py", 4, 6, "")
    lines = [ln for ln in res.splitlines() if ln.startswith("Removed ")]
    assert len(lines) == 1 and lines[0].startswith("Removed mood_change but it is still used in: ")
    assert "dialogue.py:4" in lines[0] and "other.py:2" in lines[0] and "state.py" not in lines[0]


def test_no_note_when_nothing_else_uses_it_or_it_is_still_defined_or_the_file_is_broken(tmp_path):
    (tmp_path / "state.py").write_text(STATE)
    t = tools_for(tmp_path)
    t.read_file("state.py")
    assert "Removed " not in t.replace_lines("state.py", 4, 6, "")                       # unused elsewhere
    (tmp_path / "state.py").write_text(STATE)
    (tmp_path / "dialogue.py").write_text("def talk(s):\n    return s.mood_change(2)\n")
    t = tools_for(tmp_path)
    t.read_file("state.py")
    assert "Removed " not in t.edit_file("state.py", [{"search": "def mood_change(self, n):", "replace": "def mood_change(self, n, m=0):"}])
    assert "Removed " not in t.write_file("state.py", "class State:\n    def keep(self:\n")      # a syntax error is not a removal
    assert "Removed " not in t.write_file("fresh.py", "def brand_new():\n    pass\n")        # a new file removes nothing


def test_a_moved_symbol_is_not_reported_when_the_edit_keeps_its_name(tmp_path):
    (tmp_path / "state.py").write_text(STATE)
    (tmp_path / "dialogue.py").write_text("def talk(s):\n    return s.mood_change(2)\n")
    t = tools_for(tmp_path)
    out = t.write_file("state.py", "class State:\n    def mood_change(self, n):\n        return n\n\n    def keep(self):\n        return 1\n")
    assert "Removed " not in out
    out = t.write_file("state.py", "def mood_change(n):\n    return n\n")                  # the method became a function: still defined
    assert "Removed " not in out


def test_a_feature_goal_with_a_broad_word_is_not_a_sweep(tmp_path):
    """Seen live: 'Add a level and achievement system ... each ending and each death' was split into one part per FILE (PLAN-v2.md ...)."""
    goal = "Add a simple, visible progression: achievements covering exploring places and reaching each ending and each death."
    _l, eng, _r, _o, plan = broad_engine(tmp_path, [], [FIVE], goal=goal)
    assert not ST.make_runner(eng, plan).broad_applies(plan.stages[0])
    goal2 = "Stage 3: Implement a level system for every ending"
    _l, eng, _r, _o, plan = broad_engine(sib(tmp_path, "g2"), [], [FIVE], goal=goal2)
    assert not ST.make_runner(eng, plan).broad_applies(plan.stages[0])


def test_sweeps_that_remove_or_rewrite_everywhere_still_split(tmp_path):
    for n, goal in enumerate(["Find every on-screen message about X and remove them.", "Rewrite all texts", "Delete all debug prints throughout the code",
                              "Entferne überall die alten Meldungen"]):
        _l, eng, _r, _o, plan = broad_engine(sib(tmp_path, f"s{n}"), [], [FIVE], goal=goal)
        assert ST.make_runner(eng, plan).broad_applies(plan.stages[0]), goal


def test_a_feature_stage_never_gets_the_file_grouping_fallback(tmp_path):
    goal = "Add a level and achievement system with a status screen and an achievements command."
    _l, eng, _r, _o, plan = broad_engine(tmp_path, [], ["not json"], goal=goal)
    runner = ST.make_runner(eng, plan)
    assert runner.resplit(plan.stages[0]) is False and len(plan.stages) == 1


def test_the_utility_split_calls_ask_for_json_mode_and_survive_a_backend_without_it(tmp_path):
    seen = []

    class WithJson:
        def chat(self, messages, model, stream=False, background=False, json_mode=False, extra_options=None, **kw):
            seen.append(json_mode)
            return iter(['{"stages": []}'])

    class Without:
        def chat(self, messages, model, stream=False, background=False, extra_options=None, **kw):
            seen.append("legacy")
            return iter(['{"stages": []}'])

    assert ST._ask(WithJson(), {}, "s", "u", "m", 100) == '{"stages": []}'
    assert ST._ask(Without(), {}, "s", "u", "m", 100) == '{"stages": []}'
    assert seen == [True, "legacy"]


def test_the_resplit_prompt_forbids_verify_only_parts():
    """Seen live: a feature stage was split into 'Verify ...' / 'Audit ...' parts that investigated for 30 minutes and delivered nothing."""
    low = ST._RESPLIT_SYSTEM.lower()
    assert "never create" in low and "verify" in low and "audit" in low and "deliver" in low


def test_save_games_and_data_files_never_become_stage_scope(tmp_path):
    """Seen live: a sub-stage got 'save.json' (the player's real save) in its scope."""
    (tmp_path / "a.py").write_text("x = 1\n" * 50)
    (tmp_path / "save.json").write_text("{}\n")
    (tmp_path / "data.db").write_text("x")
    (tmp_path / "song.flac").write_text("x")
    names = [r[0] for r in ST._repo_files(tmp_path)]
    assert "a.py" in names and not any(n in names for n in ("save.json", "data.db", "song.flac"))
    assert ST._safe_path("save.json") is None and ST._safe_path("tests/fixtures/save_v1.json") is None and ST._safe_path("main.py") == "main.py"
