"""planview.py: a pure aggregator over stages.py / plan.py / loopguard.py's own state - no new engine state, no model call of its
own. Same FakeLLM / new_engine fixtures as test_codeagent_stages.py (no real model, no GPU)."""

from __future__ import annotations

from agent import plan as plan_mod
from agent.codeagent import checks as checks_mod, planview as PV, session as S, stages as ST

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import TASK3, block, cfg, new_engine, three_stage_script


# ── build() reflects the stage plan as it runs ───────────────────────────────

def test_planview_is_empty_before_anything_is_staged(tmp_path):
    eng, _repo, _out = new_engine(tmp_path, FakeLLM([]))
    view = PV.build(eng)
    assert view.stages == [] and view.checklist == [] and view.dead_ends == 0
    assert view.now == "no stage plan yet"


def test_planview_tracks_pending_active_done_as_stages_run(tmp_path):
    llm = FakeLLM(three_stage_script())
    eng, _repo, _out = new_engine(tmp_path, llm)
    plan = ST.split(TASK3)
    view0 = PV.from_parts(plan, eng.plan.plan, 0, eng.config)
    assert [s.status for s in view0.stages] == ["pending", "pending", "pending"]
    assert view0.now.startswith("waiting to start stage 1")
    report = ST.run_stages(eng, plan)
    assert report.verdict == "done"
    view = PV.build(eng)
    assert [s.status for s in view.stages] == ["done", "done", "done"]
    assert view.now == "finished stage 3: " + plan.stages[2].title     # the event log's last stage_done is sharper than the status fallback
    # the file each stage touched rides along (stages.py's own Stage.files)
    assert view.stages[0].files == ["a.py"] and view.stages[1].files == ["b.py"]


def test_planview_reflects_a_blocked_stage_and_its_budget_numbers(tmp_path):
    llm = FakeLLM([say("Stage one: I think it is fine.")] * 8
                  + [call("write_file", path="ok.flag", content="1\n"), say("Stage two: wrote ok.flag.")])
    eng, _repo, _out = new_engine(tmp_path, llm, cfg(stage_attempts=2))
    eng.checks.add(checks_mod.Check("flag", "ok.flag exists", "command", {"command": "test -f ok.flag"}))
    plan = ST.split(f"STAGE 1: {block(1)}\nSTAGE 2: {block(2)}\n")
    report = ST.run_stages(eng, plan)
    assert report.verdict == "partial"
    view = PV.build(eng)
    s1, s2 = view.stages
    assert s1.status == "blocked" and "attempts" in s1.reason and s2.status == "done"
    assert s1.attempts == plan.stages[0].attempts == 2 and s2.attempts == plan.stages[1].attempts == 1
    # the exact numbers the stage runner itself tracked (stages.py budget_limits/budget_share), not a re-derivation
    max_s, max_steps, _min_edits = ST.budget_limits(eng.config, plan.stages[0])
    assert s1.budget_limit_s == round(max_s, 1) and s1.budget_limit_steps == int(max_steps)
    assert s1.budget_used_steps == plan.stages[0].spent_steps
    assert s1.budget_share == round(ST.budget_share(plan.stages[0], eng.config), 3)
    assert view.now == "finished stage 2: " + plan.stages[1].title     # the event log's last stage_done, sharper than the status fallback
    # the status-only fallback (no event log) still reports the block plainly
    fallback = PV.from_parts(plan, eng.plan.plan, 0, eng.config)
    assert fallback.now == f"blocked: {plan.stages[0].reason}"
    assert view.dead_ends == len(eng.dead_ends.items)


# ── the acceptance checklist (plan.py) rides along unchanged ────────────────

def test_planview_checklist_mirrors_mark_done(tmp_path):
    eng, _repo, _out = new_engine(tmp_path, FakeLLM([]))
    eng.plan.plan = plan_mod.Plan(task="t", items=[plan_mod.Item(1, "tests pass"), plan_mod.Item(2, "docs updated")])
    eng.plan.plan.mark_done(1, "ran pytest, 3 passed, 0 failed")
    view = PV.build(eng)
    assert view.checklist_confirmed == 1 and view.checklist_total == 2
    byid = {i.id: i for i in view.checklist}
    assert byid[1].done and byid[1].evidence == "ran pytest, 3 passed, 0 failed"
    assert not byid[2].done and byid[2].evidence == ""


def test_planview_dead_ends_count_comes_from_the_loop_guard(tmp_path):
    eng, _repo, _out = new_engine(tmp_path, FakeLLM([]))
    eng.dead_ends.record("run", "echo hi", "hi", 3, error=False, files=[], rsha="same-hash")
    view = PV.build(eng)
    assert view.dead_ends == 1


# ── rendering: table()/lines() never crash and carry the stage titles ───────

def test_render_table_and_lines_carry_the_stage_titles(tmp_path):
    eng, _repo, _out = new_engine(tmp_path, FakeLLM([]))
    plan = ST.split(TASK3)
    view = PV.from_parts(plan, eng.plan.plan, 0, eng.config)
    rendered = PV.lines(view)
    joined = "\n".join(rendered)
    assert "STAGE 1" in joined.upper() or "a.py" in joined        # at least the first stage's title shows up
    assert str(len(plan.stages)) in rendered[0]
    group = PV.table(view)        # a rich renderable: must build without raising, and actually print something
    from rich.console import Console
    import io

    out = io.StringIO()
    Console(file=out, width=140, force_terminal=False).print(group)
    assert out.getvalue().strip()


# ── --plan-first: the CLI gate (session.plan_first_gate) ────────────────────

def test_plan_first_gate_shows_the_plan_and_a_decline_runs_nothing(tmp_path):
    llm = FakeLLM([])                                  # a decline must not cause a single engine step
    eng, _repo, out = new_engine(tmp_path, llm)
    proceed = S.plan_first_gate(eng.console, eng, TASK3, eng.config, confirm=lambda: False)
    assert proceed is False
    assert eng.stage_plan is None                       # nothing was pinned for the (never-run) stage runner
    assert llm.calls == []                               # no tool-call step happened: nothing executed
    printed = out.getvalue()
    assert "STAGE 1" in printed.upper() or "a.py" in printed      # the plan was actually shown
    assert "declined" in printed.lower()


def test_plan_first_gate_confirmed_pins_the_plan_for_the_existing_stage_runner(tmp_path):
    llm = FakeLLM(three_stage_script())
    eng, _repo, _out = new_engine(tmp_path, llm)
    proceed = S.plan_first_gate(eng.console, eng, TASK3, eng.config, confirm=lambda: True)
    assert proceed is True
    plan = eng.stage_plan
    assert plan is not None and len(plan.stages) == 3 and plan.unfinished()
    assert eng.ctx.user_requests[-1] == TASK3
    # the SAME plan object runs to completion through the ordinary stage runner: no second execution path
    report = ST.run_stages(eng, plan)
    assert report.verdict == "done" and [s.status for s in plan.stages] == ["done"] * 3
