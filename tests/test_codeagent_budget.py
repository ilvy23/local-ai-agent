"""The hard per-stage budget: a stage that spends its minutes / steps without editing at a healthy rate is blocked at a segment boundary, the
supervisor goes on and retries it ONCE with half a budget; a satisfied stage that has gone quiet is closed by the harness. Scripted fake
segments (seconds / steps / edits per segment), a fake clock - no model."""

from __future__ import annotations

import types

import pytest

from agent.codeagent import stages as ST, supervisor as SV

from tests.test_codeagent_stages import new_engine
from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_supervisor import _isolated, make, plan_of  # noqa: F401 - _isolated is an autouse fixture


class BudgetRunner:
    """Per stage a list of segments: (seconds, steps, edits) or 'ok'. A segment adds them to the stage the way the real runner does."""

    def __init__(self, script, counter, tests=""):
        self.script, self.n, self.calls, self.last_state = {k: list(v) for k, v in script.items()}, counter, [], None
        self.tests, self.closed = tests, []

    def activate(self, st):
        fresh = st.status == "pending"
        st.status = "active"
        return fresh

    def segment(self, st, note="", *, fresh_plan=False):
        self.calls.append((st.id, note, fresh_plan, st.budget_scale))
        out = self.script[st.id].pop(0)
        st.attempts += 1
        if out == "ok":
            self.n["n"] += 1
            return ST.Segment(types.SimpleNamespace(stopped="", text="done"), "green")
        secs, steps, edits = out
        st.spent_s += secs
        st.spent_steps += steps
        st.real_edits += edits
        st.no_edit_segments = 0 if edits else st.no_edit_segments + 1
        if edits:
            self.n["n"] += 1                                                   # an edit is progress for the supervisor's own idle rule
        return ST.Segment(types.SimpleNamespace(stopped="announce", text=None), "red", "stopped: announce")

    def stage_tests(self, st):
        return self.tests

    def finish(self, st, res):
        st.status = "done"
        return "done"

    def close(self, st, note):
        st.status, st.summary = "done", note
        self.closed.append(st.id)
        return "done"

    def block(self, st, reason):
        st.status, st.reason = "blocked", reason
        return "blocked"


def run(tmp_path, script, *, stages=2, coding=None, tests="", plan=None, task="t"):
    sup, sess, _r, clock = make(tmp_path, {}, stages=stages, coding={"autopilot_max_idle": 50, **(coding or {})}, plan=plan)
    runner = BudgetRunner(script, {"n": 0}, tests)
    sup.runner = runner
    sup.progress = lambda: (str(runner.n["n"]), 0)
    return sup, sess, runner, sup.run(task)


def stage_budget_events(sess):
    return [(e["id"], e["action"]) for e in sess.events.of("stage_budget")]


# ── the pure rule ───────────────────────────────────────────────────────────

def test_rule_defaults_and_config_overrides():
    st = ST.Stage(1, "a", "g")
    assert ST.budget_limits({}, st) == (25 * 60, 180, 3)
    assert ST.budget_limits({"coding": {"stage_max_minutes": 10, "stage_max_steps": "50", "stage_min_edits_per_100_steps": 5}}, st) == (600, 50, 5)
    assert ST.budget_limits({"coding": {"stage_max_minutes": "junk"}}, st)[0] == 25 * 60               # a broken value falls back
    st.spent_steps = 179
    assert ST.budget_blocked(st, {}) == ""
    st.spent_steps = 180
    assert ST.budget_blocked(st, {}).startswith("budget: ")                                             # steps alone, no edits
    assert ST.budget_blocked(st, {"coding": {"stage_max_steps": 400}}) == ""                           # the override moves the line
    st.spent_steps, st.spent_s = 0, 25 * 60
    assert ST.budget_blocked(st, {}) == "budget: 25 min / 0 steps / 0 edits"
    st.budget_scale = 0.5
    st.spent_s = 12 * 60
    assert ST.budget_blocked(st, {}) == "" and ST.budget_blocked(ST.Stage(1, "a", "g", spent_s=750, budget_scale=0.5), {})


def test_rule_edits_keep_a_stage_alive_up_to_3x():
    cfg = {"coding": {"stage_max_minutes": 10, "stage_max_steps": 100}}
    st = ST.Stage(1, "a", "g", spent_s=700, spent_steps=120, real_edits=4)                              # needs ceil(1.2 * 3) = 4
    assert ST.budget_blocked(st, cfg) == ""
    st.real_edits = 3
    assert ST.budget_blocked(st, cfg) == "budget: 12 min / 120 steps / 3 edits"
    st.real_edits, st.spent_steps = 999, 299
    assert ST.budget_blocked(st, cfg) == ""
    st.spent_steps = 300                                                                                # the hard step ceiling
    assert ST.budget_blocked(st, cfg).startswith("budget:")
    st.spent_steps, st.spent_s = 10, 1800
    assert ST.budget_blocked(st, cfg).startswith("budget:")                                             # the hard time ceiling


# ── the supervisor ──────────────────────────────────────────────────────────

def test_budget_without_edits_blocks_and_the_next_stage_runs(tmp_path):
    sup, sess, runner, res = run(tmp_path, {1: [(700, 50, 0), "ok"], 2: ["ok"]}, coding={"stage_max_minutes": 10})
    s1 = sess.stage_plan.stages[0]
    assert [c[0] for c in runner.calls][:2] == [1, 2]                                                   # stage 2 right after the block
    assert stage_budget_events(sess)[0] == (1, "block")
    assert any(d["action"] == "budget_block" and d["stage"] == 1 for d in res.decisions)
    assert s1.status == "done"                                                                          # ... and the retry round finished it


def test_healthy_editing_may_exceed_the_budget_until_3x(tmp_path):
    seg = (300, 40, 3)                                                                                  # 3 edits per 40 steps: well above 3 / 100
    sup, sess, runner, res = run(tmp_path, {1: [seg] * 9, 2: ["ok"]}, coding={"stage_max_minutes": 10, "stage_max_steps": 10000})
    assert [c[0] for c in runner.calls[:6]] == [1] * 6 and runner.calls[6][0] == 2
    assert sum(1 for c in runner.calls if c[0] == 1 and c[3] == 1.0) == 6                               # 6 x 300 s = 1800 s = 3 x 10 min
    first = sess.events.of("stage_budget")[0]
    assert (first["id"], first["action"], first["spent_s"], first["steps"], first["edits"]) == (1, "block", 1800.0, 240, 18)


def test_ceiling_by_steps(tmp_path):
    sup, sess, runner, res = run(tmp_path, {1: [(1, 100, 50)] * 9, 2: ["ok"]}, stages=2, coding={"stage_max_steps": 100, "stage_max_minutes": 0})
    assert sum(1 for c in runner.calls if c[0] == 1 and c[3] == 1.0) == 3                               # 300 steps = 3 x 100


def test_retry_round_half_budget_sharp_note_and_second_block_is_final(tmp_path):
    sup, sess, runner, res = run(tmp_path, {1: [(700, 50, 0), (200, 20, 0), (200, 20, 0)], 2: ["ok"]}, coding={"stage_max_minutes": 10})
    s1, s2 = sess.stage_plan.stages
    assert [c[0] for c in runner.calls] == [1, 2, 1, 1] and s2.status == "done"
    first, retry = runner.calls[2], runner.calls[3]
    assert first[2] is True and first[3] == 0.5                                                          # a fresh acceptance list, half a budget
    assert "Previous attempt used the whole budget without finishing" in first[1]
    assert "Decide quickly: finish what is left of THIS part, or state with mark_done evidence that nothing is left." in first[1]
    assert s1.status == "blocked" and s1.reason == "budget: 7 min / 40 steps / 0 edits"        # 200 + 200 s >= 300 s (half of 600)
    assert stage_budget_events(sess) == [(1, "block"), (1, "retry"), (1, "final")]
    assert (res.verdict, res.exit_code) == ("partial", 3)
    todo = res.report.split("## NOT VERIFIED", 1)[1].split("##", 1)[0]
    assert "stage 1 stage 1: budget: 7 min / 40 steps / 0 edits (retried once with half a budget)" in todo
    assert [d["action"] for d in res.decisions if d["stage"] == 1 and d["action"] != "resume"] == ["budget_block", "budget_retry", "needs_human"]


def test_retry_that_finishes_is_done_and_there_is_one_retry_only(tmp_path):
    sup, sess, runner, res = run(tmp_path, {1: [(700, 50, 0), (100, 10, 1), "ok"]}, stages=1, coding={"stage_max_minutes": 10})
    assert sess.stage_plan.stages[0].status == "done" and (res.verdict, res.exit_code) == ("done", 0)
    assert stage_budget_events(sess) == [(1, "block"), (1, "retry")]
    assert sess.stage_plan.stages[0].spent_s == 100                                                      # fresh counters for the retry


def test_single_stage_blocked_by_budget_twice_is_partial_3(tmp_path):
    sup, sess, runner, res = run(tmp_path, {1: [(700, 50, 0), (700, 50, 0)]}, stages=1, coding={"stage_max_minutes": 10})
    assert (res.verdict, res.exit_code) == ("partial", 3) and "budget:" in res.report


def test_default_budget_without_config(tmp_path):
    sup, sess, runner, res = run(tmp_path, {1: [(1499, 179, 0), (2, 1, 0), "ok"], 2: ["ok"]})
    first = sess.events.of("stage_budget")[0]
    assert (first["action"], first["spent_s"], first["steps"]) == ("block", 1501.0, 180)           # the defaults: 25 min or 180 steps


def test_sub_stages_get_the_same_budget_each(tmp_path):
    plan = plan_of(2)
    plan.stages[0].parent, plan.stages[0].depth = 1, 1
    plan.stages[1].parent, plan.stages[1].depth = 1, 1
    sup, sess, runner, res = run(tmp_path, {1: [(700, 5, 0)], 2: [(500, 5, 0), "ok", "ok"]}, plan=plan, coding={"stage_max_minutes": 10})
    assert stage_budget_events(sess)[0] == (1, "block") and (2, "block") not in stage_budget_events(sess)[:1]


# ── the stage-end interplay: closed by the harness ──────────────────────────

def close_plan(scope=("a.py",), files=("a.py",), waived=()):
    plan = plan_of(2)
    st = plan.stages[0]
    st.scope, st.files, st.waived = list(scope), list(files), list(waived)
    return plan


QUIET = [(400, 20, 0), (5, 2, 0)]                       # 60 % of 10 min = 360 s used after the first, no edit in two segments


def test_harness_closes_a_satisfied_quiet_stage(tmp_path):
    sup, sess, runner, res = run(tmp_path, {1: QUIET, 2: ["ok"]}, plan=close_plan(), coding={"stage_max_minutes": 10})
    s1 = sess.stage_plan.stages[0]
    assert s1.status == "done" and s1.summary == "closed by the harness: scope satisfied, no new test failures, no edits in the last 2 segments"
    assert runner.closed == [1] and len(runner.calls) == 3 and (res.verdict, res.exit_code) == ("done", 0)
    assert any(d["action"] == "closed_by_harness" for d in res.decisions) and not sess.events.of("stage_budget")


def test_harness_closes_with_a_waived_scope_file(tmp_path):
    sup, sess, runner, _res = run(tmp_path, {1: QUIET, 2: ["ok"]}, plan=close_plan(("a.py", "b.py"), ("a.py",), ("b.py",)), coding={"stage_max_minutes": 10})
    assert runner.closed == [1]


@pytest.mark.parametrize("why,script,plan,tests", [
    ("an edit in the last 2 segments", [(400, 20, 0), (5, 2, 1), (5, 2, 0)], close_plan(), ""),
    ("new test failures", QUIET, close_plan(), "1 tests are newly red"),
    ("scope not satisfied", QUIET, close_plan(("a.py", "b.py"), ("a.py",)), ""),
    ("no scope at all", QUIET, close_plan((), ()), ""),
    ("less than 60 % of the budget", [(100, 20, 0), (5, 2, 0)], close_plan(), ""),
])
def test_harness_does_not_close(tmp_path, why, script, plan, tests):
    sup, sess, runner, _res = run(tmp_path, {1: script + ["ok"] * 3, 2: ["ok"]}, plan=plan, tests=tests, coding={"stage_max_minutes": 10})
    assert runner.closed == [], why


# ── the real runner: counters, resume, idle gaps, old sessions ──────────────

def test_persisted_counters_and_old_session_defaults():
    st = ST.Stage(3, "t", "g", spent_s=61.5, spent_steps=40, real_edits=7, no_edit_segments=2, budget_scale=0.5, budget_round=1)
    plan = ST.StagePlan([st], 0, "single", "t")
    back = ST.StagePlan.from_json(plan.to_json()).stages[0]
    assert (back.spent_s, back.spent_steps, back.real_edits, back.no_edit_segments, back.budget_scale, back.budget_round) == (61.5, 40, 7, 2, 0.5, 1)
    old = {"source": "single", "current": 0, "task": "t", "stages": [{"id": 1, "title": "a", "goal": "do it", "status": "active", "attempts": 1}]}
    o = ST.StagePlan.from_json(old).stages[0]
    assert (o.spent_s, o.spent_steps, o.real_edits, o.no_edit_segments, o.budget_scale, o.budget_round) == (0, 0, 0, 0, 1.0, 0)


def test_real_runner_counts_segment_seconds_steps_and_real_edits_not_idle_gaps(tmp_path, monkeypatch):
    now = {"t": 5000.0}
    real_time = ST.time
    monkeypatch.setattr(ST, "time", types.SimpleNamespace(monotonic=lambda: now["t"], time=real_time.time))

    def tick(reply):
        def step(messages):
            now["t"] += 100                                              # every model step takes 100 s
            return reply
        return step

    llm = FakeLLM([tick(call("write_file", path="x.py", content="x = 1\n")), tick(call("write_file", path=".scratch/p.py", content="p = 1\n")),
                   tick(say("All done: x.py exists."))])
    eng, repo, _ = new_engine(tmp_path, llm)
    plan = ST.StagePlan([ST.Stage(1, "t", "do the thing")], 0, "single", "")
    ST.run_stages(eng, plan)
    st = plan.stages[0]
    assert st.real_edits == 1                                            # the scratch write is no edit
    assert st.spent_steps == 3 and st.spent_s == 300
    now["t"] += 7200                                                     # the machine sat idle for 2 h between two processes
    runner = ST.make_runner(eng, plan)
    st.status = "pending"
    eng.llm = FakeLLM([tick(say("All done: nothing left."))])
    runner.segment(st)
    assert st.spent_s == 400 and st.spent_steps == 4 and st.real_edits == 1 and st.no_edit_segments == 1     # the 2 h gap is not in it
