"""Stages: a big task in fresh small contexts. split() is deterministic first; run_stages() reuses CodeEngine unchanged as the
worker (FakeLLM only - no model, no GPU)."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from agent.codeagent import brief as brief_mod, checks as checks_mod, policy as P, session as S, sessionstore as SS, stages as ST, verified
from agent.codeagent.engine import CodeEngine
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_engine import CFG, FakeLLM, call, say


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def cfg(**coding):
    return {**CFG, "coding": {"claims_check": False, **coding}}


def new_engine(tmp_path, llm, config=None):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    P.save_repo(repo, mode="auto")
    out = io.StringIO()
    eng = CodeEngine(repo, config or cfg(), llm, Console(file=out, width=140, force_terminal=False), interactive=False,
                     events=EventLog("t", tmp_path / "ev.jsonl"))
    return eng, repo, out


def big(step, tokens):
    """The same scripted reply, but the server reports a big prompt for it."""
    return {**step, "_prompt_eval_count": tokens}


def block(n, what="create a file"):
    return f"{what} number {n} and make sure that it works and that it is verified by running it once."


TASK3 = ("Build a tiny tool in three parts. Keep every file small and readable.\n\n"
         "STAGE 1: create a.py with a function one() that returns 1, and make sure that it imports cleanly.\n"
         "STAGE 2: create b.py with a function two() that returns 2, and make sure that it imports cleanly.\n"
         "STAGE 3: create c.py with a function three() that returns 3, and make sure that it imports cleanly.\n")


def one_stage_plan(goal="do the thing"):
    return ST.StagePlan([ST.Stage(1, "t", goal)], 0, "single", "")


# ── split ───────────────────────────────────────────────────────────────────

def test_split_explicit_stage_headers():
    plan = ST.split(TASK3)
    assert plan.source == "headings" and [s.id for s in plan.stages] == [1, 2, 3]
    assert plan.stages[1].title.startswith("STAGE 2") and "b.py" in plan.stages[1].goal and "a.py" not in plan.stages[1].goal.split("\n\n", 1)[1]
    assert plan.stages[1].goal.startswith("Overall task: Build a tiny tool in three parts.")
    for word in ("Stufe", "Phase", "Part", "Teil"):                                 # the other header words
        assert len(ST.split(f"{word} 1: {block(1)}\n{word} 2: {block(2)}").stages) == 2


def long_block(n):
    return block(n) + " more detail" * 150


def test_split_markdown_headings_and_no_split_inside_code_fences():
    text = f"# Plan\nintro\n## Build the parser\n{long_block(1)}\n```\n# not a heading\n```\n## Build the writer\n{long_block(2)}\n"
    plan = ST.split(text)
    assert [s.title for s in plan.stages] == ["Build the parser", "Build the writer"] and plan.source == "headings"
    assert "# not a heading" in plan.stages[0].goal


def test_split_bullet_list_of_four():
    text = "Do these things in this order:\n" + "".join(f"- {long_block(n)}\n" for n in range(1, 5))
    plan = ST.split(text)
    assert len(plan.stages) == 4 and plan.stages[2].title.startswith("create a file number 3")
    assert len(ST.split("Do:\n" + "".join(f"{n}. {long_block(n)}\n" for n in range(1, 4))).stages) == 3          # numbered, 3 items
    assert len(ST.split("Do:\n" + "".join(f"- {long_block(n)}\n" for n in range(1, 3))).stages) == 1             # 2 items are not a structure


def test_a_list_of_small_requirements_is_one_job_not_stages():
    text = "Add music:\n" + "".join(f"- {block(n)}\n" for n in range(1, 8))
    assert len(ST.split(text).stages) == 1 and ST.split(text).source == "single"
    assert len(ST.split("Stage 1: a\n" + block(1) + "\nStage 2: b\n" + block(2)).stages) == 2     # explicit headers still split


def test_short_task_is_a_single_stage_with_the_task_as_goal():
    for task in ("add a --version flag", "- a\n- b\n- c\n- d", ""):                  # tiny list items merge into one block -> single
        plan = ST.split(task)
        assert plan.source == "single" and len(plan.stages) == 1 and plan.stages[0].goal == task.strip()


def test_more_than_12_blocks_are_merged_to_12_without_losing_text():
    text = "\n".join(f"STAGE {n}: {block(n)}" for n in range(1, 16))
    plan = ST.split(text)
    assert len(plan.stages) == 12 and [s.id for s in plan.stages] == list(range(1, 13))
    joined = "\n".join(s.goal for s in plan.stages)
    assert all(f"STAGE {n}:" in joined for n in range(1, 16))


def test_tiny_blocks_are_merged_with_their_neighbour():
    text = f"STAGE 1: fix typo\nSTAGE 2: {block(2)}\nSTAGE 3: {block(3)}\nSTAGE 4: ok\n"
    plan = ST.split(text)
    assert len(plan.stages) == 2
    assert "STAGE 1: fix typo" in plan.stages[0].goal and "STAGE 2" in plan.stages[0].goal
    assert "STAGE 4: ok" in plan.stages[1].goal                                      # a tiny last block joins the previous one


LONG = " ".join(f"Sentence {n} says something about the work without any structure at all." for n in range(40))


def test_model_split_valid_invalid_and_off_when_configured():
    # changed with the intake work: stage_split_with_model now defaults to TRUE, goals need 20..1200 chars, the minimum task length is 160
    g1, g2 = "write the core module core.py with the data model", "write the cli in cli.py that uses the core module"
    ok = json.dumps({"stages": [{"title": "Core", "goal": g1}, {"title": "CLI", "goal": g2}]})
    config = {"models": {"background": "m"}, "coding": {"stage_split_with_model": True}}
    llm = FakeLLM([], background=[ok])
    plan = ST.split(LONG, llm=llm, cfg=config)
    assert plan.source == "model" and [s.title for s in plan.stages] == ["Core", "CLI"] and g2 in plan.stages[1].goal
    for bad in ("not json at all", json.dumps({"stages": [{"title": "only", "goal": g1}]}),
                json.dumps({"stages": [{"title": "a", "goal": g1}, {"title": 3, "goal": g2}]}), json.dumps({"stages": "x"})):
        assert ST.split(LONG, llm=FakeLLM([], background=[bad]), cfg=config).source == "single"
    off = FakeLLM([], background=[ok])
    assert ST.split(LONG, llm=off, cfg={"models": {"background": "m"}, "coding": {"stage_split_with_model": False}}).source == "single"
    assert off.background == [ok]                                                                                           # not asked
    assert ST.split("short task", llm=off, cfg=config).source == "single" and off.background == [ok]                       # too short


# ── running ─────────────────────────────────────────────────────────────────

def request_text(snapshot):
    return "\n".join(str(m.get("content", "")) for m in snapshot)


def three_stage_script():
    return [call("write_file", path="a.py", content="def one():\n    return 1\n"),
            call("run", command="""python3 -c "print('S1-' + 'TOOLOUT')" """.strip()),
            say("Stage one finished: a.py has one()."),
            call("write_file", path="b.py", content="def two():\n    return 2\n"),
            say("Stage two finished: b.py has two()."),
            say("Stage three: nothing more to do, c.py is not needed.")]


def test_three_stages_run_in_three_fresh_contexts(tmp_path):
    llm = FakeLLM(three_stage_script())
    eng, repo, _ = new_engine(tmp_path, llm)
    plan = ST.split(TASK3)
    report = ST.run_stages(eng, plan)
    assert report.verdict == "done" and [s.status for s in plan.stages] == ["done"] * 3 and [s.attempts for s in plan.stages] == [1, 1, 1]
    assert plan.stages[0].files == ["a.py"] and plan.stages[1].files == ["b.py"] and plan.stages[2].files == []
    first2 = llm.calls[3]["snapshot"]                                       # stage 2's first model request
    text = request_text(first2)
    goals = [m["content"] for m in first2 if str(m.get("content", "")).startswith("STAGE GOAL")]
    assert len(goals) == 1 and plan.stages[1].goal in goals[0] and plan.stages[0].goal not in text.replace(brief_mod.make(TASK3), "")
    assert "S1-TOOLOUT" not in text and not [m for m in first2 if m["role"] in ("assistant", "tool")]    # nothing of stage 1's conversation
    assert plan.stages[0].summary in text and "a.py" in plan.stages[0].summary and "Stage one finished" in plan.stages[0].summary
    assert any(brief_mod.is_card(m) for m in first2)
    assert text.count("VERIFIED STATE") == 1 and sum(m["role"] == "user" for m in first2) == 2     # the state block + the stage prompt
    assert request_text(llm.calls[0]["snapshot"]).count("VERIFIED STATE") == 1
    # the ledger lives on: stage 1's file is in stage 2's VERIFIED STATE (FILES), and stage 3 knows both
    assert "a.py" in text.split("FILES", 1)[1]
    third = request_text(llm.calls[5]["snapshot"])
    assert "a.py" in third.split("FILES", 1)[1] and "b.py" in third.split("FILES", 1)[1]
    assert set(eng.ctx.ledger.files) == {"a.py", "b.py"}
    kinds = [e["kind"] for e in eng.events.recent if e["kind"].startswith("stage_")]
    assert kinds == ["stage_start", "stage_done"] * 3
    done = eng.events.of("stage_done")[0]
    assert done["id"] == 1 and done["title"].startswith("STAGE 1") and done["prompt_peak"] == 1300 and "seconds" in done


def test_cap_ends_a_segment_not_the_stage(tmp_path):
    plan = one_stage_plan()
    seen = []

    def second(messages):
        seen.append((plan.stages[0].attempts, request_text(messages)))
        return say("All done: x.py exists.")

    llm = FakeLLM([big(call("write_file", path="x.py", content="x = 1\n"), 40000), second])
    eng, repo, _ = new_engine(tmp_path, llm)
    report = ST.run_stages(eng, plan)
    attempts, text = seen[0]
    assert attempts == 0                                                      # the cap segment was no failure
    assert ST.CAP_NOTE in text and text.count(verified.HEADER) == 1 and "x.py" in text.split("FILES", 1)[1]    # the note itself names the block
    assert "now on disk" not in text and "x = 1" not in text                  # a fresh context: the first segment's calls are gone
    assert report.verdict == "done" and plan.stages[0].attempts == 1 and plan.stages[0].prompt_peak == 40000
    assert [e["kind"] for e in eng.events.recent if e["kind"].startswith("stage_")] == ["stage_start", "stage_cap", "stage_done"]
    assert eng.stage_cap is None                                              # disabled again after the run


def test_two_cap_segments_without_a_changed_file_block_the_stage(tmp_path):
    plan = one_stage_plan()
    llm = FakeLLM([big(call("run", command="echo hi"), 40000)] * 2 + [say("never reached")])
    eng, _repo, _ = new_engine(tmp_path, llm)
    report = ST.run_stages(eng, plan)
    st = plan.stages[0]
    assert st.status == "blocked" and "no progress" in st.reason and st.attempts == 0 and report.verdict == "failed"
    assert len(llm.calls) == 2 and eng.events.of("stage_blocked")[0]["reason"] == st.reason
    # a cap segment that DID change a file resets the count
    plan2 = one_stage_plan()
    llm2 = FakeLLM([big(call("write_file", path="y.py", content="y = 1\n"), 40000), big(call("run", command="echo hi"), 40000),
                    big(call("write_file", path="y.py", content="y = 2\n"), 40000), say("fine")])
    eng2, _r, _o = new_engine(tmp_path, llm2)
    assert ST.run_stages(eng2, plan2).verdict == "done"


def test_red_check_keeps_the_stage_active_then_blocks_and_the_run_goes_on(tmp_path):
    llm = FakeLLM([say("Stage one: I think it is fine.")] * 8
                  + [call("write_file", path="ok.flag", content="1\n"), say("Stage two: wrote ok.flag.")])
    eng, repo, _ = new_engine(tmp_path, llm, cfg(stage_attempts=2))
    eng.checks.add(checks_mod.Check("flag", "ok.flag exists", "command", {"command": "test -f ok.flag"}))
    plan = ST.split(f"STAGE 1: {block(1)}\nSTAGE 2: {block(2)}\n")
    report = ST.run_stages(eng, plan)
    s1, s2 = plan.stages
    assert (s1.status, s1.attempts) == ("blocked", 2) and "flag" in s1.reason and "attempts" in s1.reason
    assert (s2.status, s2.attempts) == ("done", 1) and report.verdict == "partial"
    # the second segment of stage 1 was told why it continues, in a fresh context with the failing check in VERIFIED STATE
    seg2 = request_text(llm.calls[4]["snapshot"])
    assert "continuing: the previous segment ended without a green result" in seg2 and "flag: FAIL" in seg2
    assert "flag" in s2.summary or s2.summary                                  # blocked stages are carried as notes too
    assert "[blocked]" in request_text(llm.calls[8]["snapshot"])
    assert [e["kind"] for e in eng.events.recent if e["kind"] in ("stage_blocked", "stage_done")] == ["stage_blocked", "stage_done"]


def test_stage_plan_is_saved_and_resume_continues_at_the_next_stage(tmp_path):
    llm1 = FakeLLM(three_stage_script()[:3])
    eng1, repo, _ = new_engine(tmp_path, llm1)
    plan = ST.split(TASK3)
    eng1.session_id = SS.new_id()
    ST.run_stages(eng1, plan, save=lambda: S.save_session(eng1), max_stages=1)
    assert [s.status for s in plan.stages] == ["done", "pending", "pending"]
    saved = json.loads(SS.latest(repo).path.read_text())
    assert [s["status"] for s in saved["stages"]["stages"]] == ["done", "pending", "pending"] and saved["stages"]["source"] == "headings"

    llm2 = FakeLLM(three_stage_script()[3:])
    eng2, _r, _o = new_engine(tmp_path, llm2)
    info = S.resume_into(eng2, SS.latest(repo))
    assert info and eng2.stage_plan is not None and [s.status for s in eng2.stage_plan.stages] == ["done", "pending", "pending"]
    assert not eng2.resume_note or "VERIFIED STATE" not in eng2.resume_note                      # not computed twice
    report = ST.run_stages(eng2, eng2.stage_plan, save=lambda: S.save_session(eng2))
    assert report.verdict == "done" and len(llm2.calls) == 3                                      # stage 1 did not run again
    first = request_text(llm2.calls[0]["snapshot"])
    goals = [m["content"] for m in llm2.calls[0]["snapshot"] if str(m.get("content", "")).startswith("STAGE GOAL")]
    assert len(goals) == 1 and "b.py" in goals[0] and first.count("VERIFIED STATE") == 1
    assert "a.py" in first.split("FILES", 1)[1] and eng2.stage_plan.stages[0].summary in first
    assert [s["status"] for s in json.loads(SS.latest(repo).path.read_text())["stages"]["stages"]] == ["done"] * 3


def test_resume_of_an_active_stage_keeps_it_active_and_does_not_double_the_state(tmp_path):
    plan = ST.split(TASK3)
    plan.stages[0].status, plan.stages[0].attempts = "active", 1                # a killed run left the stage active
    eng1, repo, _ = new_engine(tmp_path, FakeLLM([]))
    eng1.session_id = SS.new_id()
    eng1.ctx.user_requests.append(TASK3)
    eng1.stage_plan = plan
    S.save_session(eng1)
    llm = FakeLLM([say("a.py already fine."), say("b"), say("c")])
    eng2, _r, _o = new_engine(tmp_path, llm)
    S.resume_into(eng2, SS.latest(repo))
    assert "VERIFIED STATE" not in (eng2.resume_note or "")
    ST.run_stages(eng2, eng2.stage_plan)
    assert eng2.stage_plan.stages[0].attempts == 2 and "STAGE 1" in request_text(llm.calls[0]["snapshot"])


def test_stages_off_never_touches_stages(tmp_path):
    llm = FakeLLM([say("Just an answer.")])
    eng, _repo, _ = new_engine(tmp_path, llm, cfg(stages="off"))
    assert S.stage_mode(eng.config) == "off" and S.run_task_staged(eng, TASK3) is None
    res = eng.turn("what is this?")
    assert res.text == "Just an answer." and eng.stage_plan is None and eng.stage_cap is None and eng.stage_goal is None
    assert not any(e["kind"].startswith("stage") for e in eng.events.recent)
    assert "stages" in SS.snapshot(eng) and SS.snapshot(eng)["stages"] is None
    assert "VERIFIED STATE" not in request_text(llm.calls[0]["snapshot"])


def test_default_is_auto():
    assert S.stage_mode({}) == "auto"


def test_auto_mode_stages_only_a_task_that_splits(tmp_path):
    llm = FakeLLM([say("x")])
    eng, _repo, _ = new_engine(tmp_path, llm, cfg(stages="auto"))
    assert S.run_task_staged(eng, "add a --version flag") is None and eng.stage_plan is None and not llm.calls
    eng2, _r, _o = new_engine(tmp_path, FakeLLM(three_stage_script()), cfg(stages="auto"))
    report = S.run_task_staged(eng2, TASK3)
    assert report.verdict == "done" and eng2.stage_plan.source == "headings" and eng2.session_id
    assert S.with_stages({"coding": {"x": 1}}, True)["coding"] == {"x": 1, "stages": "on"} and S.with_stages({}, False) == {}


def test_old_session_file_without_stages_key_loads(tmp_path):
    eng, repo, _ = new_engine(tmp_path, FakeLLM([say("hello")]))
    eng.session_id = SS.new_id()
    eng.turn("say hello")
    S.save_session(eng)
    path = SS.latest(repo).path
    data = json.loads(path.read_text())
    del data["stages"]
    path.write_text(json.dumps(data))
    eng2, _r, _o = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng2, SS.latest(repo)) is not None and eng2.stage_plan is None
    assert ST.StagePlan.from_json({"stages": "junk"}) is None and ST.StagePlan.from_json(None) is None
    round_trip = ST.StagePlan.from_json(ST.split(TASK3).to_json())
    assert round_trip is not None and len(round_trip.stages) == 3 and round_trip.stages[0].status == "pending"


def test_stages_slash_command_prints_the_table(tmp_path):
    eng, _repo, out = new_engine(tmp_path, FakeLLM([]))
    console = Console(file=io.StringIO(), width=140, force_terminal=False)
    assert S.handle_command(console, eng, "/stages") is True
    assert "no staged task" in console.file.getvalue()
    eng.stage_plan = ST.split(TASK3)
    eng.stage_plan.stages[0].status = "done"
    S.handle_command(console, eng, "/stages")
    shown = console.file.getvalue()
    assert "done" in shown and "pending" in shown and "STAGE 3" in shown


# ── re-split of a stage that keeps ending at the prompt cap ─────────────────

def write_lines(repo, name, n):
    (repo / name).write_text("".join(f"line number {i:04d} of the text\n" for i in range(n)))


def subs_json(*parts):
    return json.dumps({"stages": [{"title": t, "goal": g, "files": f} for t, g, f in parts]})


GOAL_A, GOAL_B = "Clear up the texts in a.md and keep them connected", "Clear up the texts in b.md and c.md and connect them"
TWO = subs_json(("Texts A", GOAL_A, ["a.md"]), ("Texts B", GOAL_B, ["b.md", "c.md"]))


def touch(*names):
    """Scripted writes of the files a sub-stage covers: its scope gate wants every named file changed (or waived) before the stage is green."""
    return [call("write_file", path=n, content=f"{n} is clear\n") for n in names]


def capped_engine(tmp_path, steps, background, config=None, goal="Make all game texts clear in a.md, b.md and c.md"):
    config = config or cfg()
    config = {**config, "coding": {**config["coding"], "broad_split": False, "range_split_lines": 0}}      # these tests are about the cap-based MODEL re-split (a named 700-line file would be range-split without a model call: test_codeagent_ranges.py)
    llm = FakeLLM(steps, background=background)
    eng, repo, out = new_engine(tmp_path, llm, config)
    for name in ("a.md", "b.md", "c.md"):
        write_lines(repo, name, 700)
    plan = one_stage_plan(goal)
    plan.task = "Make the whole game readable"
    return llm, eng, repo, out, plan


def test_second_cap_resplits_the_stage_and_all_sub_stages_finish(tmp_path):
    steps = [big(call("write_file", path="x1.md", content="one\n"), 40000), big(call("write_file", path="x1.md", content="two\n"), 40000),
             *touch("a.md"), say("Part A is clear now."), *touch("b.md", "c.md"), say("Part B is clear now.")]
    llm, eng, repo, out, plan = capped_engine(tmp_path, steps, [TWO])
    eng.session_id = SS.new_id()
    report = ST.run_stages(eng, plan, save=lambda: S.save_session(eng))
    assert report.verdict == "done" and [(s.id, s.title, s.parent, s.depth, s.status) for s in plan.stages] == [
        (1, "Texts A", 1, 1, "done"), (2, "Texts B", 1, 1, "done")]
    assert plan.stages[1].scope == ["b.md", "c.md"] and "b.md and c.md" in plan.stages[1].goal and plan.stages[0].goal.startswith("Overall task:")
    ev = eng.events.of("stage_resplit")
    assert len(ev) == 1 and ev[0]["from"] == 1 and ev[0]["to"] == [1, 2] and ev[0]["source"] == "model"
    shown = out.getvalue()
    assert "re-split stage 1 into 2 parts:" in shown and "Texts A" in shown and "Texts B" in shown
    assert len(llm.calls) == 7 and llm.background == []                       # exactly ONE re-split call
    # the sub-stage's fresh context pins ITS goal and carries the LARGE FILES hint (b.md and c.md have ~700 lines, > 12000 chars)
    second = request_text(llm.calls[4]["snapshot"])
    assert "LARGE FILES" in second and "b.md (701 lines)" in second and "c.md (701 lines)" in second and "read_file with a line range" in second
    assert "a.md (" not in second.split("LARGE FILES", 1)[1].splitlines()[0]
    saved = json.loads(SS.latest(repo).path.read_text())["stages"]["stages"]
    back = ST.StagePlan.from_json(json.loads(SS.latest(repo).path.read_text())["stages"])
    assert [s["parent"] for s in saved] == [1, 1] and [s["depth"] for s in saved] == [1, 1] and back.stages[1].scope == ["b.md", "c.md"]
    assert back.stages[0].parent == 1 and back.stages[0].depth == 1


def test_resplit_keeps_earlier_status_and_renumbers_the_whole_plan(tmp_path):
    llm, eng, repo, out, _p = capped_engine(tmp_path, [], [TWO])
    plan = ST.StagePlan([ST.Stage(1, "one", "first", status="done", attempts=2), ST.Stage(2, "two", "Clear a.md, b.md and c.md", status="active"),
                         ST.Stage(3, "three", "last")], 1, "headings", "the task")
    plan.stages[1].cap_segments, plan.stages[1].files = 2, ["a.md"]
    runner = ST.make_runner(eng, plan)
    assert runner.should_resplit(plan.stages[1]) and runner.resplit(plan.stages[1])
    assert [(s.id, s.title, s.status, s.attempts) for s in plan.stages] == [
        (1, "one", "done", 2), (2, "Texts A", "pending", 0), (3, "Texts B", "pending", 0), (4, "three", "pending", 0)]
    assert [s.parent for s in plan.stages] == [0, 2, 2, 0] and plan.current == 1
    assert not runner.should_resplit(plan.stages[1])                          # a sub-stage is never split again (depth 1)
    prompt = llm.calls  # no model steps ran
    assert prompt == []


def test_the_resplit_prompt_has_goal_task_files_changed_files_and_open_items(tmp_path):
    seen = []

    class Spy(FakeLLM):
        def chat(self, messages, **kw):
            seen.append(messages)
            return super().chat(messages, **kw)

    llm = Spy([], background=[TWO])
    eng, repo, _o = new_engine(tmp_path, llm)
    write_lines(repo, "a.md", 700)
    write_lines(repo, "b.md", 30)
    st = ST.Stage(1, "texts", "Make all game texts clear", cap_segments=2, files=["a.md"])
    plan = ST.StagePlan([st], 0, "single", "T" * 700)
    runner = ST.make_runner(eng, plan)
    runner.last_state = verified.VerifiedState("x", {"OPEN ITEMS": {"text": "OPEN: item seven is open"}, "DEAD ENDS": {"text": "DEAD: grep loop"}}, 0.0)
    assert runner.resplit(st)
    system, user = seen[0][0]["content"], seen[0][1]["content"]
    assert "DISJOINT" in system and "NOT yet done" in system
    assert "Make all game texts clear" in user and "T" * 600 in user and "T" * 601 not in user
    assert "a.md (701)" in user and "b.md (31)" in user and "CHANGED SO FAR IN THIS STAGE:\na.md" in user
    assert "item seven is open" in user and "grep loop" in user


def test_invalid_model_answers_fall_back_to_the_files_named_in_the_goal(tmp_path):
    bads = ["not json at all", json.dumps({"stages": [{"title": "only", "goal": GOAL_A, "files": ["a.md"]}]}),
            subs_json(("A", GOAL_A, ["a.md"]), ("A", GOAL_B, ["b.md"])),                     # duplicate titles
            subs_json(("A", GOAL_A, ["../etc/passwd"]), ("B", GOAL_B, ["b.md"])),             # unsafe path
            subs_json(("A", "too short", ["a.md"]), ("B", GOAL_B, ["b.md"])),                 # goal < 20
            subs_json(("A" * 81, GOAL_A, ["a.md"]), ("B", GOAL_B, ["b.md"]))]                # title > 80
    for bad in bads:
        _llm, eng, repo, out, plan = capped_engine(tmp_path, [], [bad])
        st = plan.stages[0]
        st.cap_segments, st.files = 2, ["a.md"]
        runner = ST.make_runner(eng, plan)
        assert runner.resplit(st), bad
        # files the stage has not changed come first; every part is <= ~600 lines, so each 700-line file is a part of its own
        assert [s.scope for s in plan.stages] == [["b.md"], ["c.md"], ["a.md"]] and eng.events.of("stage_resplit")[-1]["source"] == "fallback"
        assert [s.id for s in plan.stages] == [1, 2, 3] and all(s.parent == 1 and s.depth == 1 for s in plan.stages)


def test_fallback_groups_small_files_and_uses_the_most_read_files_when_the_goal_names_none(tmp_path):
    _llm, eng, repo, _o, plan = capped_engine(tmp_path, [], ["nope"], goal="Make all game texts clear")
    for name, n in (("s1.md", 200), ("s2.md", 300), ("s3.md", 250)):
        write_lines(repo, name, n)
    st = plan.stages[0]
    st.cap_segments = 2
    runner = ST.make_runner(eng, plan)
    runner.reads[1] = {"s1.md": 5, "s2.md": 4, "s3.md": 3}
    assert runner.resplit(st)
    assert [s.scope for s in plan.stages] == [["s1.md", "s2.md"], ["s3.md"]]            # 201 + 301 <= 600, 252 would make 753


def test_fewer_than_two_groups_changes_nothing(tmp_path):
    for bad in ("nope", ""):
        _llm, eng, repo, out, plan = capped_engine(tmp_path, [], [bad], goal="Make the texts in a.md clear")
        for other in ("b.md", "c.md"):
            (repo / other).unlink()                                                              # one file only: nothing to group
        st = plan.stages[0]
        st.cap_segments = 2
        runner = ST.make_runner(eng, plan)
        assert not runner.resplit(st) and plan.stages == [st] and st.depth == 0 and eng.events.of("stage_resplit") == []
        failed = eng.events.of("stage_resplit_failed")                                           # never silent
        assert len(failed) == 1 and failed[0]["stage"] == 1 and "fewer than 2" in failed[0]["reason"]
        assert "re-split of stage 1 not possible" in out.getvalue()
    # an exception in the model call is the same as an unusable answer
    class Boom(FakeLLM):
        def chat(self, *a, **kw):
            raise RuntimeError("down")

    eng2, repo2, _o = new_engine(tmp_path, Boom([]), cfg(range_split_lines=0))        # the file-grouping fallback (named 700-line files would be range-split otherwise)
    for name in ("a.md", "b.md"):
        write_lines(repo2, name, 700)
    st2 = ST.Stage(1, "t", "Fix a.md and b.md", cap_segments=2)
    plan2 = ST.StagePlan([st2], 0, "single", "")
    assert ST.make_runner(eng2, plan2).resplit(st2) and [s.scope for s in plan2.stages] == [["a.md"], ["b.md"]]


def test_first_cap_does_not_split_and_cap_restarts_do_not_use_attempts(tmp_path):
    steps = [big(call("write_file", path="x1.md", content="one\n"), 40000), say("All good.")]
    llm, eng, repo, _o, plan = capped_engine(tmp_path, steps, [TWO])
    assert ST.run_stages(eng, plan).verdict == "done"
    assert len(plan.stages) == 1 and plan.stages[0].cap_segments == 1 and llm.background == [TWO] and plan.stages[0].attempts == 1
    # with resplit_after_caps 3 the second cap does not split either; the third does
    steps = [big(call("write_file", path="x1.md", content=str(i)), 40000) for i in range(3)] + [*touch("a.md"), say("A done."), *touch("b.md", "c.md"), say("B done.")]
    llm, eng, repo, _o, plan = capped_engine(tmp_path, steps, [TWO], cfg(resplit_after_caps=3))
    assert ST.run_stages(eng, plan).verdict == "done" and [s.title for s in plan.stages] == ["Texts A", "Texts B"]
    assert plan.stages[0].attempts == 1 and plan.stages[1].attempts == 1       # one accepted answer each; the 3 cap segments cost nothing


def test_resplit_off_never_splits(tmp_path):
    steps = [big(call("write_file", path="x1.md", content=str(i)), 40000) for i in range(3)] + [say("All done.")]
    llm, eng, repo, _o, plan = capped_engine(tmp_path, steps, [TWO], cfg(resplit="off"))
    assert ST.run_stages(eng, plan).verdict == "done"
    assert len(plan.stages) == 1 and plan.stages[0].cap_segments == 3 and llm.background == [TWO] and eng.events.of("stage_resplit") == []


def test_a_sub_stage_that_hits_the_cap_twice_is_not_split_again(tmp_path):
    steps = [big(call("write_file", path="x1.md", content=str(i)), 40000) for i in range(2)]            # caps of the parent
    steps += [big(call("write_file", path="y1.md", content=str(i)), 40000) for i in range(3)] + [*touch("a.md"), say("A done."), *touch("b.md", "c.md"), say("B done.")]
    llm, eng, repo, _o, plan = capped_engine(tmp_path, steps, [TWO])
    assert ST.run_stages(eng, plan).verdict == "done"
    assert [s.title for s in plan.stages] == ["Texts A", "Texts B"] and plan.stages[0].cap_segments == 3 and plan.stages[0].depth == 1
    assert len(eng.events.of("stage_resplit")) == 1 and len(eng.events.of("stage_cap")) == 5            # 3 caps in the sub-stage, no 2nd split


def test_old_session_stage_without_the_new_fields_loads_with_defaults():
    old = {"source": "single", "current": 0, "task": "t", "stages": [{"id": 1, "title": "a", "goal": "do it", "status": "active", "attempts": 1}]}
    st = ST.StagePlan.from_json(old).stages[0]
    assert (st.cap_segments, st.parent, st.depth, st.scope, st.status, st.attempts) == (0, 0, 0, [], "active", 1)
