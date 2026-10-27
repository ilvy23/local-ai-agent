"""Speed diagnostics (timing.py): where the wall time of a run went, measured from the event log only - and the rules that turn it into
"why is it slow". Synthetic event logs; no model, no GPU."""

from __future__ import annotations

import importlib.util
import io
import json
import types
from pathlib import Path

import pytest
from rich.console import Console

from agent.codeagent import supervisor as SV, timing
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import new_engine
from tests.test_codeagent_supervisor import make, actions


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(SV.shutil, "which", lambda name: None)
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def log(spec, session="s", t0=1000.0):
    """spec: a list of (kind, fields) in order; every event's ts is the running clock AFTER it (like the real writer: emitted at its end).
    Durations come from the fields: step -> prefill_s + decode_s + load_s, tool/check/server_wait -> seconds, ('gap', s) -> idle time."""
    t, out = t0, []
    for kind, f in spec:
        if kind == "gap":
            t += f
            continue
        dur = (f.get("prefill_s", 0) + f.get("decode_s", 0) + (f.get("load_s") or 0)) if kind == "step" else f.get("seconds", 0)
        t += dur
        out.append({"ts": t, "session": session, "kind": kind, **f})
    return out


def step(prompt=10000, gen=100, prefill=2.0, decode=3.0, **kw):
    return ("step", {"n": 1, "prompt": prompt, "gen": gen, "prefill_s": prefill, "decode_s": decode, **kw})


def tool(name="edit_file", seconds=1.0, **kw):
    return ("tool", {"name": name, "seconds": seconds, "ok": True, **kw})


# ── D1: breakdown ───────────────────────────────────────────────────────────

def test_breakdown_shares_add_up_and_use_only_measured_values():
    ev = log([step(prefill=4, decode=2), tool(seconds=2), step(prefill=4, decode=2), tool(seconds=2), ("server_wait", {"seconds": 6, "ok": True}),
              ("gap", 4), step(prefill=4, decode=2)])
    b = timing.breakdown(ev)
    assert b.steps == 3 and b.prefill_s == 12 and b.gen_s == 6 and b.tool_s == 4 and b.wait_s == 6
    assert b.wall_s == pytest.approx(12 + 6 + 4 + 6 + 4)                     # the first step's own duration counts: the clock starts before it
    assert b.other_s == pytest.approx(4) and b.prefill_share == pytest.approx(12 / 32)
    parts = b.prefill_share + b.gen_share + b.tool_share + b.wait_share + b.other_share
    assert parts == pytest.approx(1.0)
    assert b.steps_per_min == pytest.approx(3 / (32 / 60))


def test_breakdown_prompt_stats_segments_and_stages():
    ev = log([("stage_start", {"id": 1}), ("segment_start", {"id": 1}), step(prompt=10000), step(prompt=30000),
              ("segment_start", {"id": 1}), step(prompt=20000), ("stage_start", {"id": 2}), ("segment_start", {"id": 2}), step(prompt=50000)])
    b = timing.breakdown(ev)
    assert b.mean_prompt == 27500 and b.peak_prompt == 50000 and b.segments == 3 and b.stages == 2


def test_breakdown_filters_by_session_prefix_and_since():
    a = log([step()], session="proj-100", t0=1000)
    c = log([step(), step()], session="proj-200", t0=5000)
    assert timing.breakdown(a + c, session="proj-200").steps == 2
    assert timing.breakdown(a + c, session="proj-1").steps == 1                # an unambiguous prefix is enough
    assert timing.breakdown(a + c, since=4000).steps == 2


def test_breakdown_load_time_counts_as_waiting_and_none_is_tolerated():
    ev = log([step(prefill=1, decode=1, load_s=8.0), step(prefill=1, decode=1, load_s=None), step(prefill=1, decode=1)])
    b = timing.breakdown(ev)
    assert b.wait_s == 8.0 and b.steps == 3
    old = [{"ts": 10.0, "session": "s", "kind": "step", "n": 1, "prompt": 100, "gen": 5, "prefill_s": 0.5, "decode_s": 0.5}]   # a log from before load_s
    assert timing.breakdown(old).steps == 1


def test_breakdown_of_nothing_is_empty_not_a_crash():
    b = timing.breakdown([])
    assert b.steps == 0 and b.wall_s == 0 and b.prefill_share == 0 and b.steps_per_min == 0
    assert timing.why_slow(b) == ["no model steps in the selected events - nothing to measure"]
    assert "0 steps" in b.table()


def test_breakdown_ignores_junk_events():
    ev = log([step()]) + [{"kind": "step", "ts": "x"}, {"kind": "tool"}, "oops", {"ts": 5, "kind": "tool", "seconds": "slow"}]
    assert timing.breakdown(ev).steps == 1


def test_look_share_and_test_time_come_from_the_tool_events():
    ev = log([step(), tool("read_file"), tool("search"), tool("run", cmd="sed -n 1,20p a.py"), tool("edit_file"),
              tool("run", cmd="pytest -q", seconds=30), tool("run_tests", seconds=20), tool("run", cmd="python3 x.py", seconds=2)])
    b = timing.breakdown(ev)
    assert b.tool_calls == 7 and b.look_calls == 3 and b.look_share == pytest.approx(3 / 7)
    assert b.test_s == 50 and b.tool_s == 1 + 1 + 1 + 1 + 30 + 20 + 2


def test_thinking_share_is_by_generated_tokens():
    ev = log([step(gen=100), step(gen=300, think=True)])
    assert timing.breakdown(ev).think_share == pytest.approx(0.75)


def test_table_lists_every_measured_part():
    ev = log([step(prefill=6, decode=2), tool(seconds=2)])
    text = timing.breakdown(ev).table()
    for word in ("wall", "prefill", "generation", "tools", "waiting", "other", "steps/min", "mean prompt", "peak prompt", "segments", "stages"):
        assert word in text


# ── D1: why_slow ────────────────────────────────────────────────────────────

def test_why_slow_prefill_with_big_prompts_names_the_cap():
    b = timing.breakdown(log([step(prompt=45000, prefill=20, decode=5)] * 6))
    first = timing.why_slow(b)[0]
    assert "prefill" in first and "80%" in first and "45k" in first and "coding.stage_prompt_cap" in first and "--stages" in first


def test_why_slow_slow_steps_per_minute():
    b = timing.breakdown(log([step(prompt=8000, prefill=1, decode=20)] * 6))
    assert any("steps/min" in c and "2.9" in c for c in timing.why_slow(b))


def test_why_slow_read_only_streaks():
    ev = log([step(prefill=1, decode=1)] * 12 + [tool("read_file", 0.1)] * 12 + [tool("edit_file", 0.1)] * 2)
    assert any("read-only" in c and "86%" in c for c in timing.why_slow(timing.breakdown(ev)))


def test_why_slow_tool_time_dominated_by_test_runs():
    ev = log([step(prefill=1, decode=1)] * 6 + [tool("run", cmd="pytest -q tests", seconds=90), tool("edit_file", seconds=2)])
    assert any("test" in c and "98%" in c for c in timing.why_slow(timing.breakdown(ev)))


def test_why_slow_thinking_and_waiting():
    ev = log([step(prefill=1, decode=1, gen=100), step(prefill=1, decode=1, gen=900, think=True)] * 3 + [("server_wait", {"seconds": 60, "ok": True})])
    causes = " | ".join(timing.why_slow(timing.breakdown(ev)))
    assert "thinking" in causes and "90%" in causes and "waiting" in causes.lower()


def test_why_slow_at_most_three_and_quiet_run_says_so():
    bad = log([step(prompt=45000, prefill=20, decode=20, gen=900, think=True)] * 6 + [tool("read_file")] * 20 + [("server_wait", {"seconds": 300, "ok": True})])
    assert len(timing.why_slow(timing.breakdown(bad))) == 3
    good = log([step(prompt=9000, prefill=1, decode=4, gen=200)] * 8 + [tool("edit_file", 0.5)] * 8)
    assert timing.why_slow(timing.breakdown(good)) == ["no obvious bottleneck"]


# ── log reading, since, CLI ─────────────────────────────────────────────────

def test_parse_since():
    assert timing.parse_since("2h") == 7200 and timing.parse_since("90m") == 5400 and timing.parse_since("30s") == 30 and timing.parse_since("1d") == 86400
    assert timing.parse_since("45") == 45 * 60                                   # a bare number is minutes
    for bad in ("", "abc", "2x", "-1h"):
        with pytest.raises(ValueError):
            timing.parse_since(bad)


def test_read_log_skips_damaged_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"ts": 1, "session": "a", "kind": "step"}) + "\nnot json\n\n" + json.dumps([1]) + "\n" + json.dumps({"ts": 2, "session": "b", "kind": "tool"}) + "\n")
    assert [e["session"] for e in timing.read_log(p)] == ["a", "b"]
    assert timing.read_log(tmp_path / "missing.jsonl") == []


def write(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_time_cli_defaults_to_the_latest_session_and_prints_table_and_causes(tmp_path):
    p = tmp_path / "e.jsonl"
    write(p, log([step()], session="old", t0=100) + log([step(prompt=45000, prefill=20, decode=5)] * 6, session="new", t0=5000))
    out = io.StringIO()
    assert timing.time_cli(None, None, path=p, console=Console(file=out, width=120, force_terminal=False)) == 0
    text = out.getvalue()
    assert "session new" in text and "6 steps" in text and "prefill" in text and "coding.stage_prompt_cap" in text


def test_time_cli_errors_are_plain(tmp_path):
    p = tmp_path / "e.jsonl"
    write(p, log([step()]))
    for args in ((None, "banana"), ("nope", None)):
        out = io.StringIO()
        code = timing.time_cli(*args, path=p, console=Console(file=out, width=120, force_terminal=False))
        assert code == 1 and out.getvalue().strip()
    out = io.StringIO()
    assert timing.time_cli(None, None, path=tmp_path / "none.jsonl", console=Console(file=out, width=120, force_terminal=False)) == 1


def test_why_slow_slash_command_prints_this_sessions_numbers(tmp_path):
    from agent.codeagent import session as S

    eng, _repo, out = new_engine(tmp_path, FakeLLM([]))
    for _ in range(6):
        eng.events.emit("step", n=1, prompt=45000, gen=50, prefill_s=20.0, decode_s=5.0)
    assert S.handle_command(Console(file=out, width=120, force_terminal=False), eng, "/why-slow") is True
    text = out.getvalue()
    assert "6 steps" in text and "coding.stage_prompt_cap" in text and "/why-slow" in S.HELP


def test_agent_code_time_runs_without_a_model(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from agent.main import app

    p = tmp_path / "e.jsonl"
    write(p, log([step(prompt=9000, prefill=1, decode=4, gen=200)] * 8 + [tool("edit_file", 0.5)] * 8, session="proj-1", t0=__import__("time").time() - 600))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(p))
    monkeypatch.setattr("agent.main._ask_gpu_mode", lambda: (_ for _ in ()).throw(AssertionError("a read-only command must not ask for the GPU mode")))
    monkeypatch.setattr("sys.argv", ["agent", "code", "time"])
    res = CliRunner().invoke(app, ["code", "time", "--since", "2h"])
    assert res.exit_code == 0 and "8 steps" in res.output and "no obvious bottleneck" in res.output
    res = CliRunner().invoke(app, ["code", "time", "--session", "proj-1"])
    assert res.exit_code == 0 and "session proj-1" in res.output
    assert CliRunner().invoke(app, ["code", "time", "--since", "banana"]).exit_code == 1


def test_step_event_records_load_time_think_flag_and_old_fields(tmp_path):
    eng, _repo, _o = new_engine(tmp_path, FakeLLM([
        {**call("list_dir", path="."), "_prompt_eval_ns": 2_000_000_000, "_eval_ns": 1_000_000_000, "_load_ns": 5_000_000_000}, say("done")]))
    eng.turn("look")
    steps = eng.events.of("step")
    assert steps[0]["load_s"] == 5.0 and steps[0]["prefill_s"] == 2.0 and steps[0]["decode_s"] == 1.0 and steps[0]["think"] is False
    assert steps[1]["load_s"] is None                                           # the server sent no load_duration: None, not 0


def test_tool_event_records_the_command_of_a_run(tmp_path):
    eng, repo, _o = new_engine(tmp_path, FakeLLM([]))
    eng.execute("run", {"command": "echo hi"})
    eng.execute("list_dir", {"path": "."})
    tools = eng.events.of("tool")
    assert tools[0]["cmd"] == "echo hi" and "cmd" not in tools[1]


# ── D2: supervisor hook ─────────────────────────────────────────────────────

def bd(prefill_share=0.7, steps=30, wall=600.0):
    return timing.Breakdown(wall_s=wall, prefill_s=wall * prefill_share, gen_s=wall * (1 - prefill_share) / 2, steps=steps)


def test_supervisor_slow_prefill_rule_uses_the_injected_breakdown(tmp_path):
    sup, _s, _r, _c = make(tmp_path, {1: ["ok"]}, stages=1, breakdown_fn=lambda: bd(0.7))
    assert sup.slow_prefill() is not None and "70%" in sup.slow_prefill()
    sup2, *_ = make(tmp_path, {1: ["ok"]}, stages=1, breakdown_fn=lambda: bd(0.4))
    assert sup2.slow_prefill() is None
    sup3, *_ = make(tmp_path, {1: ["ok"]}, stages=1, breakdown_fn=lambda: bd(0.9, steps=3))        # too few steps to judge
    assert sup3.slow_prefill() is None
    sup5, *_ = make(tmp_path, {1: ["ok"]}, stages=1, breakdown_fn=lambda: bd(0.9, steps=30, wall=90.0))   # a short window with cache-miss spikes
    assert sup5.slow_prefill() is None
    sup4, *_ = make(tmp_path, {1: ["ok"]}, stages=1, coding={"prefill_end_share": 0}, breakdown_fn=lambda: bd(0.9))      # switched off
    assert sup4.slow_prefill() is None


def test_supervisor_fresh_context_is_not_counted_as_unproductive(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["fresh_context", "fresh_context", "fresh_context", "fresh_context", "fresh_context", "ok"]},
                                 stages=1, breakdown_fn=lambda: bd(0.7))
    res = sup.run("task")
    assert res.verdict == "done" and actions(res).count("fresh_context") == 5          # five in a row would be idle (max 4) if they counted
    assert "resume_pause" not in actions(res) and "fresh_stage" not in actions(res)


def test_supervisor_arms_the_engine_hook_for_each_segment(tmp_path):
    sup, sess, runner, _c = make(tmp_path, {1: ["ok"]}, stages=1, breakdown_fn=lambda: bd(0.7))
    seen = []
    runner.on_segment = lambda: seen.append(sess.segment_end_check())
    sup.run("task")
    assert seen and "prefill" in seen[0]
    assert sess.segment_end_check is None                                         # taken off again at the end


def test_engine_ends_the_segment_at_a_step_boundary_when_the_hook_says_so(tmp_path):
    eng, _repo, _o = new_engine(tmp_path, FakeLLM([call("list_dir", path="."), call("list_dir", path="src"), say("done")]))
    asked = []
    eng.segment_end_check = lambda: asked.append(1) or ("prefill is 70% of the last 10 min" if len(asked) >= 2 else None)
    res = eng.turn("look around")
    assert res.stopped == "fresh_context" and res.steps == 2 and len(eng.events.of("fresh_context")) == 1


def test_stage_runner_reports_fresh_context_as_a_cap_style_segment(tmp_path):
    from agent.codeagent import stages as ST

    eng, _repo, _o = new_engine(tmp_path, FakeLLM([call("list_dir", path="."), say("x")]))
    eng.segment_end_check = lambda: "slow"
    plan = ST.StagePlan([ST.Stage(1, "t", "do the thing")], 0, "single", "")
    runner = ST.make_runner(eng, plan)
    seg = runner.segment(plan.stages[0])
    assert seg.kind == "cap" and seg.res.stopped == "fresh_context" and plan.stages[0].attempts == 0
    assert eng.events.of("segment_start")


def test_autopilot_report_has_the_measured_breakdown_and_causes_of_this_run(tmp_path):
    sup, sess, _r, _c = make(tmp_path, {1: ["ok"]}, stages=1)
    for _ in range(6):
        sess.events.emit("step", n=1, prompt=45000, gen=50, prefill_s=20.0, decode_s=5.0)
    res = sup.run("task")
    assert "## speed (measured)" in res.report and "6 steps" in res.report and "prefill" in res.report
    assert "coding.stage_prompt_cap" in res.report                                       # the cause, with the measured numbers
    assert (sess.root / ".scratch" / "autopilot-report.md").read_text() == res.report


def test_autopilot_report_without_any_model_step_says_so(tmp_path):
    sup, _s, _r, _c = make(tmp_path, {1: ["ok"]}, stages=1)
    assert "no model steps" in sup.run("task").report


def test_stage_prompt_cap_can_be_overridden_by_the_environment(tmp_path, monkeypatch):
    from agent.codeagent import stages as ST

    eng, _repo, _o = new_engine(tmp_path, FakeLLM([]))
    plan = ST.StagePlan([ST.Stage(1, "t", "x")], 0, "single", "")
    assert ST.make_runner(eng, plan).cap == 28000
    monkeypatch.setenv("AGENT_CODE_STAGE_PROMPT_CAP", "16000")
    assert ST.make_runner(eng, plan).cap == 16000
    monkeypatch.setenv("AGENT_CODE_STAGE_PROMPT_CAP", "junk")
    assert ST.make_runner(eng, plan).cap == 28000


# ── D4: scripts/speed_trial.py ──────────────────────────────────────────────

def load_trial():
    path = Path(__file__).resolve().parent.parent / "scripts" / "speed_trial.py"
    spec = importlib.util.spec_from_file_location("speed_trial", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_speed_trial_verdict_passes_a_fast_small_prompt_run():
    st = load_trial()
    b = timing.breakdown(log([step(prompt=12000, prefill=1, decode=4)] * 8))
    ok, lines = st.verdict(b, max_prefill=0.25, min_steps_per_min=6)
    assert ok and any("PASS" in x for x in lines) and not any("FAIL" in x for x in lines)


def test_speed_trial_fails_on_prefill_share_and_on_steps_per_min():
    st = load_trial()
    slow_prefill = timing.breakdown(log([step(prompt=40000, prefill=20, decode=5)] * 8))
    ok, lines = st.verdict(slow_prefill, 0.25, 6)
    assert not ok and any("FAIL" in x and "prefill" in x for x in lines)
    slow_steps = timing.breakdown(log([step(prompt=12000, prefill=1, decode=30)] * 6))
    ok, lines = st.verdict(slow_steps, 0.25, 6)
    assert not ok and any("FAIL" in x and "steps/min" in x for x in lines)


def test_speed_trial_steps_per_min_is_only_judged_when_the_prompt_peak_is_small():
    st = load_trial()
    big_prompt = timing.breakdown(log([step(prompt=45000, prefill=1, decode=30)] * 6))
    ok, lines = st.verdict(big_prompt, 0.25, 6, peak_limit=30000)
    assert ok and any("not judged" in x and "steps/min" in x for x in lines)


def test_speed_trial_no_steps_is_a_fail_not_a_pass():
    st = load_trial()
    ok, lines = st.verdict(timing.breakdown([]), 0.25, 6)
    assert not ok and any("FAIL" in x for x in lines)


def test_speed_trial_dry_run_parses_a_file_and_sets_the_exit_code(tmp_path, capsys):
    st = load_trial()
    p = tmp_path / "e.jsonl"
    write(p, log([step(prompt=12000, prefill=1, decode=4)] * 8, session="run1"))
    assert st.main(["--dry-run", str(p)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "prefill" in out
    write(p, log([step(prompt=40000, prefill=20, decode=5)] * 8, session="run1"))
    assert st.main(["--dry-run", str(p), "--max-prefill", "0.9"]) == 0                     # thresholds are flags
    assert st.main(["--dry-run", str(p)]) == 1
    assert st.main(["--dry-run", str(tmp_path / "missing.jsonl")]) == 2


def test_speed_trial_builds_the_subprocess_command_with_the_cap_override(tmp_path):
    st = load_trial()
    cmd, env = st.build_command(tmp_path, "add a thing", cap=16000, events=tmp_path / "e.jsonl", base_env={"PATH": "/bin"})
    assert cmd[:3] == ["agent", "code", "--autopilot"] and "--stages" in cmd and "--folder" in cmd and cmd[-1] == "add a thing"
    assert env["AGENT_CODE_EVENTS"] == str(tmp_path / "e.jsonl") and env["AGENT_CODE_STAGE_PROMPT_CAP"] == "16000" and env["PATH"] == "/bin"
