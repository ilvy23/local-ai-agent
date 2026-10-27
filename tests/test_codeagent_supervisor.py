"""The autopilot supervisor: graded recovery after every non-final stop, from measurements only. A fake stage runner / FakeLLM, an injectable
clock and sleep, a mocked server probe and a mocked systemd-inhibit - no model, no GPU, no real systemd."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import types
from pathlib import Path

import pytest
from rich.console import Console

from agent.codeagent import lookact, session as S, stages as ST, supervisor as SV
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_stages import TASK3, new_engine, three_stage_script
from tests.test_codeagent_engine import FakeLLM, say


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(SV.shutil, "which", lambda name: None)             # never a real systemd-inhibit
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


# ── fakes ───────────────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.t, self.jump, self.sleeps = 1000.0, 0.0, []

    def __call__(self):
        return self.t

    def wall(self):
        return self.t + self.jump

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeSession:
    """Just what the supervisor touches of a CodeEngine."""

    def __init__(self, tmp_path, llm=None):
        self.root = tmp_path / "proj"
        self.root.mkdir(exist_ok=True)
        self.config = {"models": {"chat": "m", "background": "m"}, "coding": {}}
        self.events = EventLog("t", tmp_path / "ev.jsonl")
        self.out = io.StringIO()
        self.console = Console(file=self.out, width=140, force_terminal=False)
        self.stage_plan = None
        self.ctx = types.SimpleNamespace(user_requests=[], ledger=types.SimpleNamespace(files={}, runs=[]))
        self.lookact = lookact.LookAct()
        self._sigs, self._declined = [], set()
        self.llm, self.model, self.briefs = llm, "m", []
        self.saves = 0
        self.segment_end_check = None

    def save(self):
        self.saves += 1


class Runner:
    """A scripted stage runner: outcomes per stage id. 'ok' green + progress, 'ok0' green without progress, 'loop' / 'budget' / 'error' /
    'announce' stops without progress (the '+' suffix: with progress), 'cap' / 'cap+' the context cap."""

    def __init__(self, script, counter):
        self.script, self.n, self.calls, self.last_state = {k: list(v) for k, v in script.items()}, counter, [], None
        self.on_segment = None

    def activate(self, st):
        fresh = st.status == "pending"
        st.status = "active"
        return fresh

    def segment(self, st, note="", *, fresh_plan=False):
        self.calls.append((st.id, note, fresh_plan))
        if self.on_segment:
            self.on_segment()
        out = self.script[st.id].pop(0)
        if callable(out):
            out = out()
        st.attempts += 1
        if out.endswith("+") or out.startswith("ok"):
            if out != "ok0":
                self.n["n"] += 1
        base = out.rstrip("+")
        if base in ("ok", "ok0"):
            return ST.Segment(types.SimpleNamespace(stopped="", text="done"), "green")
        if base == "cap":
            return ST.Segment(types.SimpleNamespace(stopped="stage_cap", text=None), "cap")
        if base == "fresh_context":                                        # the measured slow-prefill rule ended the segment (engine.segment_end_check)
            return ST.Segment(types.SimpleNamespace(stopped="fresh_context", text=None), "cap")
        if base == "interrupted":
            return ST.Segment(types.SimpleNamespace(stopped="interrupted", text=None), "interrupted")
        return ST.Segment(types.SimpleNamespace(stopped=base, text=None), "red", f"stopped: {base}")

    def finish(self, st, res):
        st.status = "done"
        return "done"

    def block(self, st, reason):
        st.status, st.reason = "blocked", reason
        return "blocked"


def plan_of(n):
    return ST.StagePlan([ST.Stage(i, f"stage {i}", f"Overall task: x\n\ndo part {i}") for i in range(1, n + 1)], 0, "headings", "the task")


def green_state(tests="TESTS: 3 passed, 0 failed (1s)", t_status="pass", checks=None):
    return types.SimpleNamespace(text="", sections={"TESTS": {"text": tests, "status": t_status, "failed": 0, "errors": 0, "failing": []},
                                                     "CHECKS": {"text": "", "results": checks or {}}})


def make(tmp_path, script, *, stages=3, coding=None, verify=None, probe=None, llm=None, plan=None, **kw):
    clock, n = Clock(), {"n": 0}
    sess = FakeSession(tmp_path, llm)
    sess.config["coding"].update({"autopilot_inhibit": False, **(coding or {})})
    sess.stage_plan = plan if plan is not None else plan_of(stages)
    runner = Runner(script, n)
    sup = SV.Supervisor(sess, cfg=sess.config, clock=clock, wall=clock.wall, sleep=clock.sleep, save=sess.save, runner=runner,
                        progress=lambda: (str(n["n"]), 0), verify=verify or (lambda: green_state()), probe=probe, **kw)
    return sup, sess, runner, clock


def actions(res):
    return [d["action"] for d in res.decisions]


# ── the happy path and the report ───────────────────────────────────────────

def test_happy_path_three_stages_exit_0_and_report_from_measurements(tmp_path):
    sup, sess, runner, clock = make(tmp_path, {1: ["ok"], 2: ["ok"], 3: ["ok"]})
    sess.ctx.ledger.files = {"a.py": 1, "b.py": 1}
    (sess.root / ".scratch").mkdir()
    (sess.root / ".scratch" / "probe.py").write_text("x = 1\n")
    (sess.root / "tmp").mkdir()
    (sess.root / "tmp" / "t.txt").write_text("t\n")
    res = sup.run("the task")
    assert (res.verdict, res.exit_code) == ("done", 0) and res.decisions == []
    assert [s.status for s in sess.stage_plan.stages] == ["done"] * 3 and [c[0] for c in runner.calls] == [1, 2, 3]
    rep = res.report
    for needle in ("done", "stage 1", "stage 3", "TESTS: 3 passed", "a.py", "b.py", ".scratch/probe.py", "tmp/t.txt", "segments: 3"):
        assert needle in rep, needle
    assert "## NOT VERIFIED\n- none" in rep and (sess.root / ".scratch" / "autopilot-report.md").read_text() == rep
    assert "autopilot-report.md" not in rep.split("## scratch", 1)[1]
    kinds = [e["kind"] for e in sess.events.recent]
    assert kinds[0] == "autopilot_start" and kinds[-1] == "autopilot_end" and sess.saves >= 3


# ── graded recovery ─────────────────────────────────────────────────────────

def test_loop_twice_then_success_resume_then_resume_pause(tmp_path):
    seen = {}

    def third():
        seen["streak"], seen["sigs"] = sess.lookact.look_streak, list(sess._sigs)
        return "ok"

    sup, sess, runner, _c = make(tmp_path, {1: ["loop", "loop", third]}, stages=1)
    sess._sigs.append(("run", "x"))
    sess._declined.add(("run", "abc"))
    res = sup.run("t")
    assert res.verdict == "done" and actions(res) == ["resume", "resume_pause"]
    assert seen["streak"] >= lookact.LOOK_BLOCK_AT and seen["sigs"] == [] and sess._declined == set()
    assert [d["reason"] for d in res.decisions] == ["loop", "loop"] and all(d["progress"] is False and d["stage"] == 1 for d in res.decisions)
    assert "continuing" in runner.calls[1][1] and runner.calls[2][2] is False


def test_pause_is_called_on_lookact(tmp_path, monkeypatch):
    sup, sess, _r, _c = make(tmp_path, {1: ["loop", "loop", "ok"]}, stages=1)
    called = []
    monkeypatch.setattr(sess.lookact, "pause", lambda: called.append(1))
    sup.run("t")
    assert called == [1]


def test_three_unproductive_segments_fresh_stage_then_needs_human_and_stage_3_still_runs(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["ok"], 2: ["loop", "budget", "error", "loop"], 3: ["ok"]})
    sess.stage_plan.stages[1].files = ["bad.py"]
    res = sup.run("t")
    assert actions(res) == ["resume", "resume_pause", "fresh_stage", "needs_human"]
    assert [d["reason"] for d in res.decisions] == ["loop", "budget", "error", "loop"]
    assert [c[0] for c in runner.calls] == [1, 2, 2, 2, 2, 3]
    fresh = runner.calls[4]                                               # the segment after 'fresh_stage': plan reset + a note
    assert fresh[0] == 2 and fresh[2] is True and "no progress" in fresh[1]
    s2 = sess.stage_plan.stages[1]
    assert s2.status == "blocked" and "needs human" in s2.reason and sess.stage_plan.stages[2].status == "done"
    assert (res.verdict, res.exit_code) == ("partial", 3) and "stage 2" in res.report and "NOT VERIFIED" in res.report
    assert "needs human" in res.report


def test_fresh_stage_note_names_the_failing_check_from_verified_state(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["loop", "loop", "loop", "ok"]}, stages=1)
    runner.last_state = types.SimpleNamespace(text="", sections={"TESTS": {"status": "fail", "failing": ["tests/test_a.py::test_x"]},
                                                                 "CHECKS": {"results": {"c1": "fail"}}})
    sup.run("t")
    assert "tests/test_a.py::test_x" in runner.calls[3][1]


def test_dependent_stage_is_blocked_with_its_dependency(tmp_path):
    plan = plan_of(3)
    plan.stages[2].goal = "Overall task: x after stage 1 and 2 \n\nAfter stage 2: wire it up"
    sup, sess, runner, _c = make(tmp_path, {1: ["ok"], 2: ["loop"] * 4, 3: []}, plan=plan)
    res = sup.run("t")
    s3 = sess.stage_plan.stages[2]
    assert s3.status == "blocked" and "stage 2" in s3.reason and [c[0] for c in runner.calls] == [1, 2, 2, 2, 2]
    assert res.verdict == "partial"


def test_productive_stop_resets_the_count(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["loop", "loop+", "loop", "ok"]}, stages=1)
    res = sup.run("t")
    assert actions(res) == ["resume", "resume", "resume"] and [d["progress"] for d in res.decisions] == [False, True, False]


def test_cap_segment_counts_as_productive_when_progress_changed(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["cap+", "cap+", "cap+", "cap+", "cap+", "ok"]}, stages=1, coding={"autopilot_max_idle": 2})
    res = sup.run("t")
    assert res.verdict == "done" and set(actions(res)) == {"resume"} and "context cap" in runner.calls[1][1]
    sup2, _s, _r, _c2 = make(tmp_path, {1: ["cap", "cap"]}, stages=1, coding={"autopilot_max_idle": 2})
    assert sup2.run("t").exit_code == 2


def test_announce_limit_is_a_graded_stop(tmp_path):
    sup, sess, _r, _c = make(tmp_path, {1: ["announce", "ok"]}, stages=1)
    res = sup.run("t")
    assert [d["reason"] for d in res.decisions] == ["announce"] and res.verdict == "done"


# ── global stops ────────────────────────────────────────────────────────────

def test_max_idle_global_stop_exit_2(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["loop"] * 9, 2: [], 3: []}, coding={"autopilot_max_idle": 2})
    res = sup.run("t")
    assert (res.verdict, res.exit_code) == ("needs_human", 2) and len(runner.calls) == 2 and "idle" in res.report
    assert sess.stage_plan.stages[0].status == "active"                    # nothing is marked done or blocked that was not decided


def test_hours_budget_stop(tmp_path):
    sup, sess, runner, clock = make(tmp_path, {1: [lambda: clock_adv(clock) or "ok"], 2: [lambda: clock_adv(clock) or "ok"], 3: ["ok"]},
                                    coding={"autopilot_max_hours": 1})
    res = sup.run("t")
    assert [s.status for s in sess.stage_plan.stages] == ["done", "done", "pending"]
    assert (res.verdict, res.exit_code) == ("partial", 3) and "time budget" in res.report


def clock_adv(clock):
    clock.t += 2400


def test_segment_cap(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["ok"], 2: ["ok"], 3: ["ok"]}, coding={"autopilot_max_segments": 2})
    res = sup.run("t")
    assert len(runner.calls) == 2 and (res.verdict, res.exit_code) == ("partial", 3) and "segment cap" in res.report


# ── server health ───────────────────────────────────────────────────────────

class ServerLLM:
    def server_url(self, model):
        return "http://127.0.0.1:1"


def test_server_down_for_a_while_waits_then_continues(tmp_path):
    answers = iter([False, False, False, True] + [True] * 20)
    sup, sess, runner, clock = make(tmp_path, {1: ["ok"]}, stages=1, llm=ServerLLM(), probe=lambda url: next(answers))
    res = sup.run("t")
    assert res.verdict == "done" and clock.sleeps == [1, 2] and len(runner.calls) == 1


def test_server_never_back_is_needs_human_server_down(tmp_path):
    sup, sess, runner, clock = make(tmp_path, {1: ["ok"]}, stages=1, llm=ServerLLM(), probe=lambda url: False,
                                    coding={"autopilot_server_wait_s": 60})
    res = sup.run("t")
    assert (res.verdict, res.exit_code) == ("needs_human", 2) and "server down" in res.report and runner.calls == []
    assert sum(clock.sleeps) <= 60 and any(e.get("reason") == "server down" for e in sess.events.of("autopilot_end"))


# ── crashes, interrupts, suspend, kill ──────────────────────────────────────

def test_exception_in_a_segment_is_logged_and_treated_as_unproductive(tmp_path):
    def boom():
        raise RuntimeError("kaboom")

    sup, sess, runner, _c = make(tmp_path, {1: [boom, "ok"]}, stages=1)
    res = sup.run("t")
    assert res.verdict == "done" and [(d["reason"], d["action"]) for d in res.decisions] == [("error", "resume")]
    crash = sess.events.of("autopilot_crash")
    assert crash and "kaboom" in crash[0]["error"] and "Traceback" in crash[0]["traceback"] and "RuntimeError" in crash[0]["traceback"]


def test_keyboard_interrupt_saves_and_exits_130(tmp_path):
    def stop():
        raise KeyboardInterrupt

    sup, sess, runner, _c = make(tmp_path, {1: ["ok"], 2: [stop], 3: ["ok"]})
    before = sess.saves
    res = sup.run("t")
    assert res.exit_code == 130 and res.verdict == "partial" and sess.saves > before and len(runner.calls) == 2
    assert sess.stage_plan.stages[1].status == "active" and sess.stage_plan.stages[2].status == "pending"
    sup2, sess2, _r, _c2 = make(tmp_path, {1: ["interrupted"]}, stages=1)                  # the engine's own Ctrl-C result
    assert sup2.run("t").exit_code == 130


def test_clock_jump_during_a_segment_is_not_an_idle_segment(tmp_path):
    def suspended():
        clock.t += 5                                                     # 5 s of monotonic time ...
        clock.jump += 7200                                               # ... but the wall clock jumped 2 h (suspend)
        return "loop"

    sup, sess, runner, clock = make(tmp_path, {1: [suspended, "ok"]}, stages=1, coding={"autopilot_max_idle": 1})
    res = sup.run("t")
    assert res.verdict == "done" and [d["reason"] for d in res.decisions] == ["suspend"] and res.decisions[0]["action"] == "resume"
    sup2, _s, _r, _c2 = make(tmp_path, {1: ["loop", "ok"]}, stages=1, coding={"autopilot_max_idle": 1})
    assert sup2.run("t").exit_code == 2                                  # the same stop without a jump counts


def test_fake_done_with_red_tests_is_not_done(tmp_path):
    states = iter([green_state("TESTS: 1 passed, 2 failed (1s)", "fail"), green_state()])
    sup, sess, runner, _c = make(tmp_path, {1: ["ok", "ok"]}, stages=1, verify=lambda: next(states))
    res = sup.run("t")
    assert len(runner.calls) == 2 and res.verdict == "done" and [d["reason"] for d in res.decisions] == ["red"]
    assert "red" in runner.calls[1][1] or "failed" in runner.calls[1][1]
    # and when it never turns green the stage is not claimed
    sup2, sess2, _r2, _c2 = make(tmp_path, {1: ["ok0"] * 6}, stages=1, verify=lambda: green_state("TESTS: 0 passed, 1 failed", "fail"),
                                 coding={"autopilot_max_idle": 3})
    res2 = sup2.run("t")
    assert res2.exit_code == 2 and sess2.stage_plan.stages[0].status != "done" and "failed" in res2.report


def test_red_check_blocks_done_too(tmp_path):
    states = iter([green_state(checks={"c1": "fail"}), green_state(checks={"c1": "pass"})])
    sup, sess, runner, _c = make(tmp_path, {1: ["ok", "ok"]}, stages=1, verify=lambda: next(states))
    assert sup.run("t").verdict == "done" and len(runner.calls) == 2


def test_kill_9_then_a_new_supervisor_continues_the_active_stage(tmp_path):
    def killed():
        raise SystemExit(137)                                            # the process ended abruptly after the session was saved

    sup, sess, runner, _c = make(tmp_path, {1: ["ok"], 2: [killed]})
    with pytest.raises(SystemExit):
        sup.run("t")
    saved = sess.stage_plan.to_json()
    assert [s["status"] for s in saved["stages"]] == ["done", "active", "pending"]
    sup2, sess2, runner2, _c2 = make(tmp_path, {2: ["ok"], 3: ["ok"]}, plan=ST.StagePlan.from_json(saved))      # a new process, the saved plan
    res = sup2.run("t")
    assert res.verdict == "done" and [c[0] for c in runner2.calls] == [2, 3]
    assert sess2.ctx.user_requests == []                                 # a resumed plan is not a new request


# ── systemd-inhibit ─────────────────────────────────────────────────────────

class Proc:
    def __init__(self, argv):
        self.argv, self.pid, self.terminated, self.waited = argv, 4242, False, False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True

    def kill(self):
        pass

    def poll(self):
        return None


def test_inhibit_child_is_started_and_terminated(tmp_path):
    procs, kills = [], []

    def popen(argv, **kw):
        procs.append(Proc(argv))
        return procs[-1]

    sup, sess, _r, _c = make(tmp_path, {1: ["ok"]}, stages=1, coding={"autopilot_inhibit": True},
                             which=lambda n: "/usr/bin/" + n, popen=popen, killpg=lambda pid, sig: kills.append((pid, sig)))
    assert sup.run("t").verdict == "done"
    argv = procs[0].argv
    assert argv[0].endswith("systemd-inhibit") and "--what=sleep:idle" in argv and "--who=agent" in argv and argv[-4:] == ["tail", f"--pid={os.getpid()}", "-f", "/dev/null"]
    assert kills == [(4242, signal.SIGTERM)] and procs[0].waited


def test_inhibit_is_terminated_even_when_the_run_dies(tmp_path):
    procs, kills = [], []

    def popen(argv, **kw):
        procs.append(Proc(argv))
        return procs[-1]

    def killed():
        raise SystemExit(1)

    sup, *_ = make(tmp_path, {1: [killed]}, stages=1, coding={"autopilot_inhibit": True}, which=lambda n: "/x/" + n, popen=popen,
                   killpg=lambda pid, sig: kills.append(pid))
    with pytest.raises(SystemExit):
        sup.run("t")
    assert kills == [4242]


def test_inhibit_skipped_silently_when_missing_failing_or_off(tmp_path):
    calls = []
    sup, *_ = make(tmp_path, {1: ["ok"]}, stages=1, coding={"autopilot_inhibit": True}, which=lambda n: None, popen=lambda *a, **k: calls.append(1))
    assert sup.run("t").verdict == "done" and calls == []

    def bad(argv, **kw):
        raise OSError("no")

    sup2, *_ = make(tmp_path, {1: ["ok"]}, stages=1, coding={"autopilot_inhibit": True}, which=lambda n: "/x/" + n, popen=bad)
    assert sup2.run("t").verdict == "done"
    sup3, *_ = make(tmp_path, {1: ["ok"]}, stages=1, coding={"autopilot_inhibit": False}, which=lambda n: "/x/" + n, popen=lambda *a, **k: calls.append(1))
    assert sup3.run("t").verdict == "done" and calls == []


# ── default-on predicate and the CLI ────────────────────────────────────────

BULLETS = "Do this:\n- one thing\n- two things\n- three things\n"


def test_should_autopilot_predicate():
    auto = {"coding": {"autopilot": "auto"}}
    assert SV.should_autopilot(BULLETS, auto) and SV.should_autopilot("x" * 600, auto)
    assert not SV.should_autopilot("fix the typo", auto) and not SV.should_autopilot("Do:\n- a\n- b\n", auto)
    assert not SV.should_autopilot("Do:\n  - a\n  - b\n  - c\n  - d\n", auto)                  # nested bullets are not top level
    assert not SV.should_autopilot("```\n- a\n- b\n- c\n```", auto)                             # inside a fence
    assert SV.should_autopilot(BULLETS, {})                                                   # default is auto
    for off in ({"coding": {"autopilot": "off"}}, {"coding": {"autopilot": "weird"}}):
        assert not SV.should_autopilot(BULLETS + "x" * 700, off)
    assert not SV.should_autopilot(None, auto) and not SV.should_autopilot("", auto)


def test_cli_flags_are_parsed_and_the_exit_code_is_propagated(monkeypatch):
    from typer.testing import CliRunner

    from agent import main

    seen = []
    monkeypatch.setattr(S, "run_session", lambda *a, **kw: seen.append((a, kw)) or 3)
    r = CliRunner().invoke(main.app, ["code", "--old-ui", "--autopilot", "--mode", "auto", "build it"])
    assert r.exit_code == 3 and seen[0][1] == {"autopilot": True, "mode": "auto"} and seen[0][0][1] == "build it"
    r = CliRunner().invoke(main.app, ["code", "--old-ui", "--autopilot", "--resume", "build it"])
    assert seen[1][1] == {"autopilot": True} and seen[1][0][5] == "latest"
    CliRunner().invoke(main.app, ["code", "--old-ui", "plain"])
    assert seen[2][1] == {}
    assert CliRunner().invoke(main.app, ["code", "--old-ui", "--mode", "nonsense", "x"]).exit_code != 0
    assert "--autopilot" in CliRunner().invoke(main.app, ["code", "--old-ui", "--help"]).output


def test_run_session_runs_the_supervisor_with_a_fake_engine_factory(tmp_path, monkeypatch):
    llm = FakeLLM(three_stage_script())
    eng, repo, _out = new_engine(tmp_path, llm)
    monkeypatch.setattr(S, "free_gpu", lambda *a, **k: None)
    monkeypatch.setattr(S, "find_root", lambda p: repo)
    monkeypatch.setattr(S, "acquire_lock", lambda root: (True, ""))
    released = []
    monkeypatch.setattr(S, "release_gpu_on_exit", lambda console, llm_, config: released.append(config))
    monkeypatch.setattr("agent.config.load_config", lambda: {**eng.config, "ollama": {"unload_on_exit": True}})
    monkeypatch.setattr("agent.foreground.session", lambda config: contextlib.nullcontext())
    got = {}
    monkeypatch.setattr(S, "open_engine", lambda console, config, root, interactive, test=None: got.update(config=config) or (eng, llm, None))
    code = S.run_session(str(repo), TASK3, autopilot=True, mode="medium")
    assert code == 0 and eng.pol.mode == "medium"
    assert [s.status for s in eng.stage_plan.stages] == ["done"] * 3
    assert (repo / ".scratch" / "autopilot-report.md").is_file()
    assert got["config"]["coding"]["stages"] == "on" and got["config"]["ollama"]["unload_on_exit"] is False   # while it runs
    assert released and released[0]["ollama"]["unload_on_exit"] is True                                         # the normal unload at the very end


def test_autopilot_mode_rules(tmp_path):
    eng, repo, _o = new_engine(tmp_path, FakeLLM([]))
    eng.pol.mode = "ask"
    assert S.autopilot_mode(eng, None) == "medium" and eng.pol.mode == "medium"
    eng.pol.mode = "ask"
    assert S.autopilot_mode(eng, "ask") == "ask" and eng.pol.mode == "ask"               # the user's explicit --mode wins
    eng.pol.mode = "auto"
    assert S.autopilot_mode(eng, None) == "auto"


# ── the real engine under the supervisor ────────────────────────────────────

def test_real_engine_three_stages_with_fake_llm(tmp_path):
    llm = FakeLLM(three_stage_script())
    eng, repo, _o = new_engine(tmp_path, llm)
    eng.session_id = "sess"
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "autopilot_inhibit": False}}
    sup = SV.Supervisor(eng, clock=Clock(), save=lambda: S.save_session(eng))
    res = sup.run(TASK3)
    assert (res.verdict, res.exit_code) == ("done", 0) and [s.status for s in eng.stage_plan.stages] == ["done"] * 3
    assert "a.py" in res.report and (repo / ".scratch" / "autopilot-report.md").is_file()
    assert eng.ctx.user_requests[0] == TASK3


def test_real_engine_budget_stop_is_graded_and_the_announce_limit_is_a_stop_reason(tmp_path):
    from tests.test_codeagent_engine import call

    script = [x for i in range(9) for x in (say("Now update main.py:"), call("list_dir", path=f"d{i}"))][:-1]       # 9 announcements, a tool step between
    eng, repo, _o = new_engine(tmp_path, FakeLLM(script))
    eng.begin_stage("goal", "VERIFIED STATE")                                    # a staged run: the limit is a stop reason
    res = eng.turn("goal")
    assert res.stopped == "announce" and res.text is None and eng.events.of("announce_limit")
    plain, _r, _o2 = new_engine(tmp_path, FakeLLM(script))
    plain.pol.steps_set = True                                                   # a fixed budget: this test is about the announce limit, not the step budget
    assert plain.turn("goal").stopped == ""                                       # outside stages the old behaviour: the answer is accepted

    eng2, repo2, _o3 = new_engine(tmp_path, FakeLLM([call("list_dir", path=".")] * 12))
    eng2.config = {**eng2.config, "coding": {**eng2.config["coding"], "autopilot_inhibit": False, "autopilot_max_idle": 1}}
    eng2.pol.max_steps, eng2.pol.steps_set = 5, True
    sup = SV.Supervisor(eng2, clock=Clock(), save=lambda: S.save_session(eng2))
    res2 = sup.run("one small task")
    assert res2.exit_code == 2 and res2.reason == "idle" and sup.decisions[0]["reason"] in ("budget", "loop")


# ── a re-split under the supervisor ─────────────────────────────────────────

class Splitter(Runner):
    """The 2nd segment of stage 1 re-splits it into 2 sub-stages (ids 1, 2), as the real runner does at the 2nd cap."""

    def segment(self, st, note="", *, fresh_plan=False):
        if st.id == 1 and st.depth == 0 and len(self.calls) == 1:
            self.calls.append((st.id, note, fresh_plan))
            st.cap_segments = 2
            plan = self.plan
            plan.stages[:] = [ST.Stage(1, "part a", "Overall task: x\n\ndo a", parent=1, depth=1), ST.Stage(2, "part b", "Overall task: x\n\ndo b", parent=1, depth=1)]
            return ST.Segment(types.SimpleNamespace(stopped="stage_cap", text=None), "cap", "resplit")
        return super().segment(st, note, fresh_plan=fresh_plan)


def test_resplit_resets_the_counters_and_is_progress_not_idle(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["loop", "loop", "ok"], 2: ["ok"]}, stages=1, coding={"autopilot_max_idle": 2})
    split = Splitter(runner.script, runner.n)
    split.plan = sess.stage_plan
    sup.runner = split
    res = sup.run("t")
    # without the reset the stale u=1 / idle=1 of the old stage 1 would make the new stage 1's first loop 'resume_pause' and hit max_idle
    assert res.verdict == "done" and [s.status for s in sess.stage_plan.stages] == ["done", "done"] and [s.title for s in sess.stage_plan.stages] == ["part a", "part b"]
    assert actions(res) == ["resume", "resplit", "resume"] and [d["progress"] for d in res.decisions] == [False, True, False]
    assert res.decisions[1]["reason"] == "stage_cap" and res.decisions[2]["u"] == 0


def test_real_engine_resplit_under_the_supervisor_finishes_all_sub_stages(tmp_path):
    from tests.test_codeagent_stages import TWO, big, capped_engine, touch
    from tests.test_codeagent_engine import call

    steps = [big(call("write_file", path="x1.md", content="one\n"), 40000), big(call("write_file", path="x1.md", content="two\n"), 40000),
             *touch("a.md"), say("Part A is clear now."), *touch("b.md", "c.md"), say("Part B is clear now.")]
    llm, eng, repo, _o, plan = capped_engine(tmp_path, steps, [TWO])
    eng.stage_plan, eng.session_id = plan, "sess"
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "autopilot_inhibit": False}}
    res = SV.Supervisor(eng, clock=Clock(), save=lambda: S.save_session(eng)).run("Make the whole game readable")
    assert (res.verdict, res.exit_code) == ("done", 0) and [s.status for s in eng.stage_plan.stages] == ["done", "done"]
    assert actions(res) == ["resume", "resplit"]
    assert [e["source"] for e in eng.events.of("stage_resplit")] == ["model"]


# ── the real scenario: 4 stages, stage 2 names no files, it ends at the cap again and again ──────────────

def scenario(tmp_path, reply):
    """A 4-stage plan like a real long run (stage 1 done, stage 2 'Remove the morale messages.' names no file); two cap-ending segments."""
    from tests.test_codeagent_stages import big, write_lines
    from tests.test_codeagent_engine import call

    steps = [big(call("write_file", path="x1.md", content="one\n"), 40000), big(call("write_file", path="x1.md", content="two\n"), 40000)] + \
            [x for i in range(6) for x in (call("mark_done", id=1, evidence="engine.py, main.py and lena.py need no change"), say(f"Part {i} is done."))]
    llm = FakeLLM(steps, background=[reply])
    eng, repo, out = ST_new_engine(tmp_path, llm)
    for name, text in (("engine.py", "morale = 3\n"), ("main.py", "print(morale)\n"), ("story.py", "# nothing here\n"), ("lena.py", "def f(morale): pass\n")):
        (repo / name).write_text(text * 40)
    write_lines(repo, "locations.py", 650)
    mk = lambda i, t, status: ST.Stage(i, t, f"do {t}", status=status)
    plan = ST.StagePlan([mk(1, "Stage 1: Make the texts clear.", "done"),
                         ST.Stage(2, "Stage 2: Remove the morale messages.", "Find every on-screen message that comments on the level and remove it.", status="active"),
                         mk(3, "Stage 3: Levels.", "pending"), mk(4, "Stage 4: Features.", "pending")], 1, "headings", "Extend the game")
    eng.stage_plan, eng.session_id = plan, "sess"
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "autopilot_inhibit": False}}
    return eng, plan, out


def ST_new_engine(tmp_path, llm):
    from tests.test_codeagent_stages import new_engine

    return new_engine(tmp_path, llm)


def run_scenario(tmp_path, reply):
    eng, plan, out = scenario(tmp_path, reply)
    # the counter survives a process restart: persist, reload the plan from JSON, a fresh runner
    plan2 = ST.StagePlan.from_json(json.loads(json.dumps(plan.to_json())))
    eng.stage_plan = plan2
    res = SV.Supervisor(eng, clock=Clock(), save=lambda: S.save_session(eng)).run("Extend the game")
    return eng, eng.stage_plan, res, out


def test_real_scenario_stage_without_files_is_resplit_by_the_model_after_the_second_cap(tmp_path):
    reply = json.dumps({"stages": [
        {"title": "Morale in engine", "goal": "Remove the morale messages from engine.py and its tests", "files": ["engine.py"]},
        {"title": "Morale in main", "goal": "Remove the morale messages from main.py and its tests", "files": ["main.py"]}]})
    eng, plan, res, _o = run_scenario(tmp_path, reply)
    ev = eng.events.of("stage_resplit")
    assert len(ev) == 1 and ev[0]["source"] == "model" and ev[0]["from"] == 2 and eng.events.of("stage_resplit_failed") == []
    assert [s.title for s in plan.stages][:4] == ["Stage 1: Make the texts clear.", "Morale in engine", "Morale in main", "Stage 3: Levels."] and len(plan.stages) == 5
    assert plan.stages[1].parent == 2 and plan.stages[1].depth == 1 and res.exit_code == 0


def test_real_scenario_garbage_model_answer_falls_back_to_the_files_that_mention_the_goal(tmp_path):
    eng, plan, res, _o = run_scenario(tmp_path, "sorry, I cannot do that")
    ev = eng.events.of("stage_resplit")
    assert len(ev) == 1 and ev[0]["source"] == "fallback" and eng.events.of("stage_resplit_failed") == []
    subs = [s for s in plan.stages if s.depth == 1]
    assert {f for s in subs for f in s.scope} == {"engine.py", "main.py", "lena.py"} and all(len(s.scope) == 1 for s in subs)
    assert [s.status for s in plan.stages] == ["done"] * len(plan.stages) and res.exit_code == 0


def test_fallback_without_keyword_hits_groups_the_repo_by_size_and_skips_tests(tmp_path):
    (tmp_path / "tests").mkdir()
    for name, n in (("a.py", 400), ("b.py", 400), ("c.py", 100), ("tests/test_a.py", 300), (".scratch/x.py", 300)):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("x = 1\n" * n)
    os.utime(tmp_path / "a.py", (2_000_000_000, 2_000_000_000))                             # a.py changed more recently than b.py
    os.utime(tmp_path / "b.py", (1_900_000_000, 1_900_000_000))
    repo = ST._repo_files(tmp_path)
    st = ST.Stage(1, "Stage 1: Zzzzzz.", "Zzzzzz everything")
    subs = ST._fallback_substages(st, repo, {}, tmp_path, ["c.py"])
    assert [g for _t, _g, g in subs] == [["c.py", "a.py"], ["b.py"]]                       # ledger file first, every part <= ~600 lines
    assert ST._fallback_substages(st, [("only.py", 10, 10)], {}, tmp_path) is None


def test_cap_counter_survives_json_reload_and_fresh_stage(tmp_path):
    st = ST.Stage(2, "t", "g", cap_segments=1)
    back = ST.StagePlan.from_json(ST.StagePlan([st], 0, "single", "").to_json())
    assert back.stages[0].cap_segments == 1
