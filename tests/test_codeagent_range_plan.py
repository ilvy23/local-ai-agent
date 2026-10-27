"""Measured live (run B, same prompt as run A after the range split was fixed): the model-based split returned ONE stage, so split() gave source
'single', run_task_staged returned None, the prompt ran as an ordinary one-shot turn on the 687-line dialogue.py (9 searches, 7 reads, 12 runs, zero
edits, 4m48s) and the range split never had a chance. Now a task that names 1-3 files, one of them above coding.range_split_lines, with
editing-by-passes intent, gets the deterministic range plan (source 'ranges') straight from split(): no model call, whatever a model would answer.
FakeLLM only."""

from __future__ import annotations

import json

from agent.codeagent import intake, session, stages as ST, supervisor as SV

from tests.test_codeagent_ranges import NOT_USED, TASK, Recorder, game, stage_ready
from tests.test_codeagent_stages import cfg

ONE_STAGE = json.dumps({"stages": [{"title": "Rewrite dialogue.py", "goal": "Rewrite the spoken lines of dialogue.py so they are plain and easy to understand"}]})
TWO_STAGES = json.dumps({"stages": [
    {"title": "Rewrite dialogue.py", "goal": "Rewrite the spoken lines of dialogue.py so they are plain and easy to understand, section by section"},
    {"title": "Update tests", "goal": "Update the tests so they match the rewritten lines of dialogue.py"}]})
TESTS_TITLE = "Run the tests and fix what YOUR rewrites broke"


def planned(tmp_path, answer=NOT_USED, task=TASK, config=None, **kw):
    llm = Recorder([], background=[answer])
    _l, eng, repo, _plan = game(tmp_path, llm=llm, task=task, **kw)
    return llm, eng, repo, ST.split(task, llm=llm, cfg=config or eng.config, model="m", root=repo)


# ── 1. the measured failure: a single-stage model answer no longer decides ──

def test_a_model_answer_of_one_stage_cannot_hide_the_range_plan(tmp_path):
    llm, eng, repo, plan = planned(tmp_path, ONE_STAGE)
    assert plan.source == "ranges" and llm.asked == [] and llm.calls == []                    # not even asked
    parts = [s for s in plan.stages if s.lines]
    assert len(parts) >= 4 and all(s.scope == ["dialogue.py"] and s.depth == 1 for s in parts)


def test_a_model_answer_of_two_stages_cannot_either(tmp_path):
    llm, _eng, _repo, plan = planned(tmp_path, TWO_STAGES)
    assert plan.source == "ranges" and llm.asked == []
    assert not any(s.title == "Update tests" or s.title == "Rewrite dialogue.py" for s in plan.stages)


def test_the_parts_are_contiguous_over_the_whole_file_and_never_touch_another_file(tmp_path):
    _llm, _eng, _repo, plan = planned(tmp_path)
    parts = [s for s in plan.stages if s.lines]
    ranges = [s.lines for s in parts]
    assert ranges[0][0] == 1 and ranges[-1][1] == 687 and all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:]))
    n = len(parts)
    assert [s.title for s in parts] == [f"Part {i}/{n}: dialogue.py lines {a}-{b}" for i, (a, b) in enumerate(ranges, 1)]
    assert all(s.lines_total == 687 and s.goal.startswith("Overall task:") for s in parts)
    everything = " ".join(s.title + s.goal + " ".join(s.scope) for s in plan.stages)
    assert "main.py" not in everything and "PLAN" not in everything
    assert [s.id for s in plan.stages] == list(range(1, len(plan.stages) + 1)) and plan.task == TASK


def test_the_last_stage_runs_the_tests_when_the_repo_has_tests(tmp_path):
    _llm, _eng, _repo, plan = planned(tmp_path)                                               # game() has tests/test_game.py
    last = plan.stages[-1]
    assert last.title == TESTS_TITLE and last.scope == [] and last.lines is None and last.depth == 0
    assert sum(1 for s in plan.stages if s.title == TESTS_TITLE) == 1
    assert not ST._BROAD.search(ST._body(last)) and not ST.named_files(last, ST._repo_files(_repo))


def test_no_tests_stage_without_tests_or_a_mention_of_them(tmp_path):
    llm = Recorder([], background=[NOT_USED])
    _l, eng, repo, _p = game(tmp_path, llm=llm)
    (repo / "tests" / "test_game.py").unlink()
    (repo / "tests").rmdir()
    plan = ST.split(TASK, llm=llm, cfg=eng.config, model="m", root=repo)
    assert plan.source == "ranges" and all(s.lines for s in plan.stages)                      # parts only


def test_a_task_that_mentions_tests_gets_the_tests_stage_even_without_test_files(tmp_path):
    task = TASK + " Run the tests at the end."
    llm = Recorder([], background=[NOT_USED])
    _l, eng, repo, _p = game(tmp_path, llm=llm, task=task)
    (repo / "tests" / "test_game.py").unlink()
    (repo / "tests").rmdir()
    plan = ST.split(task, llm=llm, cfg=eng.config, model="m", root=repo)
    assert plan.source == "ranges" and plan.stages[-1].title == TESTS_TITLE


# ── 2. what stays as it was ─────────────────────────────────────────────────

def test_a_small_named_file_keeps_the_old_behaviour(tmp_path):
    llm, _eng, _repo, plan = planned(tmp_path, ONE_STAGE, dialogue=120)
    assert plan.source == "single" and len(plan.stages) == 1 and len(llm.asked) == 1           # the model split still runs and says one stage


def test_a_small_named_file_still_takes_the_model_stages(tmp_path):
    _llm, _eng, _repo, plan = planned(tmp_path, TWO_STAGES, dialogue=120)
    assert plan.source == "model" and [s.title for s in plan.stages] == ["Rewrite dialogue.py", "Update tests"]


def test_a_creation_prompt_is_not_a_range_job(tmp_path):
    task = ("Add a new dialogue file next to dialogue.py and rewrite the spoken lines of dialogue.py into it so they are plain and easy to "
            "understand. Work through the whole file section by section.")
    llm, _eng, _repo, plan = planned(tmp_path, ONE_STAGE, task=task)
    assert plan.source == "single" and len(llm.asked) == 1


def test_a_prompt_that_names_no_file_is_not_a_range_job(tmp_path):
    task = ("Rewrite the spoken lines of the game so they are plain and easy to understand, keep every speaker and every choice. "
            "Work through the whole game section by section.")
    llm, _eng, _repo, plan = planned(tmp_path, ONE_STAGE, task=task)
    assert plan.source == "single" and len(llm.asked) == 1


def test_a_task_without_editing_by_passes_intent_is_not_a_range_job(tmp_path):
    task = "In dialogue.py, find out which speaker has the most lines and tell me which section of the file has the most choices in it."
    _llm, _eng, _repo, plan = planned(tmp_path, ONE_STAGE, task=task)
    assert plan.source == "single"


def test_without_a_root_or_with_the_threshold_off_nothing_changes(tmp_path):
    llm = Recorder([], background=[ONE_STAGE])
    _l, eng, repo, _p = game(tmp_path, llm=llm)
    assert ST.split(TASK, llm=llm, cfg=eng.config, model="m").source == "single"              # old call shape: no root
    off = cfg(range_split_lines=0)
    assert ST.split(TASK, llm=Recorder([], background=[ONE_STAGE]), cfg=off, model="m", root=repo).source == "single"


def test_headings_that_name_other_files_keep_the_structure_plan(tmp_path):
    task = ("Stage 1: In dialogue.py, rewrite the spoken lines so they are plain and easy to understand and keep every speaker.\n"
            "Stage 2: In main.py, rewrite the comments so they are plain and easy to understand and keep every function.")
    llm = Recorder([], background=[NOT_USED])
    _l, eng, repo, _p = game(tmp_path, llm=llm, task=task)
    plan = ST.split(task, llm=llm, cfg=eng.config, model="m", root=repo)
    assert plan.source == "headings" and len(plan.stages) == 2


def test_headings_that_all_name_the_big_file_are_replaced_by_the_finer_range_plan(tmp_path):
    task = ("Stage 1: In dialogue.py, rewrite the spoken lines so they are plain and easy to understand and keep every speaker.\n"
            "Stage 2: In dialogue.py, rewrite the narration lines in the same plain style and keep every choice of the player.")
    llm = Recorder([], background=[NOT_USED])
    _l, eng, repo, _p = game(tmp_path, llm=llm, task=task)
    plan = ST.split(task, llm=llm, cfg=eng.config, model="m", root=repo)
    assert plan.source == "ranges" and len([s for s in plan.stages if s.lines]) >= 4


# ── 3. the staged path is taken ─────────────────────────────────────────────

def test_stage_mode_auto_runs_a_ranges_plan_as_stages(tmp_path, monkeypatch):
    llm = Recorder([], background=[ONE_STAGE])
    _l, eng, repo, _p = game(tmp_path, llm=llm)
    assert session.stage_mode(eng.config) == "auto"
    seen = []
    monkeypatch.setattr(ST, "run_stages", lambda engine, plan, **kw: seen.append(plan) or "RAN")
    assert session.run_task_staged(eng, TASK) == "RAN"                                         # not None: the caller does NOT run engine.turn
    assert seen[0].source == "ranges" and len(seen[0].stages) >= 2 and eng.stage_plan is seen[0]
    assert llm.asked == [] and eng.ctx.user_requests[-1] == TASK


def test_stage_mode_off_still_means_no_stages(tmp_path):
    _l, eng, repo, _p = game(tmp_path, config=cfg(stages="off"))
    assert session.run_task_staged(eng, TASK) is None


def test_the_autopilot_predicate_accepts_a_short_ranged_task(tmp_path):
    _l, eng, repo, _p = game(tmp_path)
    auto = cfg()
    assert len(TASK) < SV.AUTO_MIN_CHARS
    assert SV.should_autopilot(TASK, auto, repo)
    assert not SV.should_autopilot(TASK, auto)                                                 # no root: the old rule
    assert not SV.should_autopilot(TASK, cfg(autopilot="off"), repo)
    assert not SV.should_autopilot(TASK, cfg(range_split_lines=0), repo)


def test_the_autopilot_predicate_leaves_other_short_tasks_alone(tmp_path):
    _l, eng, repo, _p = game(tmp_path)
    auto = cfg()
    assert not SV.should_autopilot("Add a new dialogue file next to dialogue.py and rewrite the lines of dialogue.py into it.", auto, repo)
    assert not SV.should_autopilot("Rewrite the spoken lines of the game so they are plain.", auto, repo)
    assert not SV.should_autopilot("fix the typo in main.py", auto, repo)
    (tmp_path / "small").mkdir()
    _l, _eng, repo2, _p = game(tmp_path / "small", dialogue=120)
    assert not SV.should_autopilot(TASK, auto, repo2)


def test_the_predicate_makes_no_model_call_and_agrees_with_split(tmp_path):
    llm = Recorder([], background=[NOT_USED])
    _l, eng, repo, _p = game(tmp_path, llm=llm)
    assert SV.should_autopilot(TASK, cfg(), repo) and llm.asked == []
    assert ST.split(TASK, cfg=cfg(), root=repo).source == "ranges"


def test_the_supervisor_plans_with_the_ranges(tmp_path):
    _l, eng, repo, _p = game(tmp_path)
    sup = SV.Supervisor.__new__(SV.Supervisor)
    sup.session, sup.cfg = eng, eng.config
    plan = sup._plan(TASK)
    assert plan.source == "ranges" and eng.stage_plan is plan


# ── 4. no intake questions for these tasks ──────────────────────────────────

def test_the_intake_skips_a_task_with_a_named_file_and_a_clear_verb(tmp_path):
    _l, eng, repo, _p = game(tmp_path)
    assert intake.looks_specific(TASK, eng)


# ── 5. direct-to-edit pressure of a ranged part ─────────────────────────────

def test_a_ranged_part_note_says_read_the_range_and_rewrite_it_at_once(tmp_path):
    _llm, eng, repo, plan = planned(tmp_path)
    runner = ST.make_runner(eng, plan)
    st = plan.stages[1]
    a, b = st.lines
    note = runner.notes_for(st)
    assert (f"Start by reading lines {a}-{b} of dialogue.py (read_file path start={a} end={b}) and rewrite them immediately in place with "
            "replace_lines/edit_file; do not search the rest of the repository first; tests are run once at the end.") in note
    assert "Start by reading lines" not in runner.notes_for(plan.stages[-1])                    # the tests stage has no range


def test_stray_searches_and_runs_of_a_ranged_part_count_double_for_the_first_eight(tmp_path):
    _l, eng, repo, plan = planned_engine(tmp_path)
    stage_ready(eng, plan.stages[0])
    eng._run_call({"name": "search", "arguments": {"pattern": "weather"}})
    assert eng.lookact.look_streak == 2
    eng._run_call({"name": "list_dir", "arguments": {"path": "."}})
    assert eng.lookact.look_streak == 4
    eng._run_call({"name": "run", "arguments": {"command": "grep -n weather dialogue.py"}})
    assert eng.lookact.look_streak == 6
    for _ in range(5):
        eng._run_call({"name": "search", "arguments": {"pattern": "weather"}})
    assert eng.lookact.look_streak == 16 and eng._stray_looks == 8
    eng._run_call({"name": "search", "arguments": {"pattern": "weather"}})
    assert eng.lookact.look_streak == 17                                                       # the ninth counts once again


def test_reading_the_range_itself_stays_cheap_and_an_edit_still_resets(tmp_path):
    _l, eng, repo, plan = planned_engine(tmp_path)
    p1 = plan.stages[0]
    stage_ready(eng, p1)
    eng.lookact.free_steps = 6
    eng._run_call({"name": "read_file", "arguments": {"path": "dialogue.py", "start": 1, "end": 40}})
    assert eng.lookact.look_streak == 0                                                        # the orientation read of the range is free
    eng._run_call({"name": "search", "arguments": {"pattern": "weather"}})
    assert eng.lookact.look_streak == 2                                                        # a search is never free in a ranged part
    eng._run_call({"name": "read_file", "arguments": {"path": "dialogue.py", "start": 1, "end": 40}})
    assert eng.lookact.look_streak == 2                                                        # free step, not counted
    eng.lookact.free_steps = 0
    eng._run_call({"name": "read_file", "arguments": {"path": "dialogue.py", "start": 1, "end": 40}})
    assert eng.lookact.look_streak == 3
    eng._run_call({"name": "replace_lines", "arguments": {"path": "dialogue.py", "start": 2, "end": 2, "text": '    "plain line",'}})
    assert eng.lookact.look_streak == 0


def test_other_stages_and_other_tools_are_not_weighted(tmp_path):
    _l, eng, repo, plan = planned_engine(tmp_path)
    plain = ST.Stage(1, "t", "Look at things", scope=["dialogue.py"])                          # no range
    stage_ready(eng, plain)
    eng._run_call({"name": "search", "arguments": {"pattern": "weather"}})
    assert eng.lookact.look_streak == 1
    stage_ready(eng, plan.stages[0])
    eng.lookact.look_streak = 0
    eng._run_call({"name": "read_file", "arguments": {"path": "dialogue.py", "start": 1, "end": 5}})
    eng._run_call({"name": "outline", "arguments": {"path": "dialogue.py"}})
    assert eng.lookact.look_streak == 2 and eng._stray_looks == 0


def test_a_new_stage_starts_the_stray_count_again(tmp_path):
    _l, eng, repo, plan = planned_engine(tmp_path)
    stage_ready(eng, plan.stages[0])
    eng._run_call({"name": "search", "arguments": {"pattern": "weather"}})
    assert eng._stray_looks == 1
    eng.begin_stage("goal", "state")
    assert eng._stray_looks == 0


def test_the_stray_weight_is_configurable(tmp_path):
    _l, eng, repo, plan = planned_engine(tmp_path, config=cfg(stray_look_weight=3))
    stage_ready(eng, plan.stages[0])
    eng._run_call({"name": "search", "arguments": {"pattern": "weather"}})
    assert eng.lookact.look_streak == 3


def planned_engine(tmp_path, config=None):
    llm = Recorder([], background=[NOT_USED])
    _l, eng, repo, _p = game(tmp_path, llm=llm, config=config)
    plan = ST.split(TASK, llm=llm, cfg=eng.config, model="m", root=repo)
    eng.stage_plan = plan
    return llm, eng, repo, plan
