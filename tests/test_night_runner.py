"""The watched subprocess of a night attempt (agent/night/runner.py) and the settings overlay it starts with. A fake Popen and a fake clock: nothing is started."""
from __future__ import annotations

import json
import os
import signal

import pytest

from agent.night import overlay
from agent.night.runner import Attempt, subprocess_runner


class FakeProc:
    def __init__(self, run_for: float, clock, code=0):
        self.pid, self.returncode, self._until, self._clock, self._code = 4242, None, clock.t + run_for, clock, code

    def poll(self):
        if self.returncode is None and self._clock.t >= self._until:
            self.returncode = self._code
        return self.returncode


class Clock:
    t = 0.0

    def sleep(self, s):
        self.t += s


def _runner(tmp_path, run_for, code=0, touch_events_every=None, **kw):
    clock = Clock()
    made, killed = {}, []
    events = tmp_path / "events.jsonl"
    events.write_text("x")

    def popen(argv, **k):
        made["argv"], made["kwargs"] = argv, k
        return FakeProc(run_for, clock, code)

    def sleep(s):
        clock.sleep(s)
        if touch_events_every and int(clock.t) % touch_events_every < s:
            os.utime(events, (clock.t + 1000, clock.t + 1000))               # the coding agent logged something

    run = subprocess_runner(tmp_path, events, clock=lambda: clock.t, sleep=sleep, poll_s=10, popen=popen, **kw)
    return run, made, killed, clock


def _go(run, killed_target, **attempt_kw):
    orig = os.killpg
    os.killpg = lambda pid, sig: killed_target.append(sig)
    try:
        return run(Attempt(**{"task": "do it", "new": True, "timeout_s": 600, "stall_s": 300, **attempt_kw}))
    finally:
        os.killpg = orig


def test_a_normal_exit_returns_the_autopilot_code(tmp_path):
    run, made, killed, _ = _runner(tmp_path, run_for=50, code=0, touch_events_every=10)
    out = _go(run, killed)
    assert (out.code, out.why) == (0, "") and killed == []
    argv = made["argv"]
    assert argv[:3][1:] == ["-c", "from agent.night.overlay import boot; boot()"] and argv[3:6] == ["code", "do it", "--new"]
    assert "--autopilot" in argv and "--mode" in argv and "--folder" in argv and made["kwargs"]["start_new_session"] is True


def test_resume_has_no_task_text(tmp_path):
    run, made, killed, _ = _runner(tmp_path, run_for=10, code=3, touch_events_every=10)
    out = _go(run, killed, task="", new=False)
    assert "--resume" in made["argv"] and "--new" not in made["argv"] and "do it" not in made["argv"] and out.code == 3 and out.why == ""


def test_an_unknown_exit_code_is_a_crash(tmp_path):
    run, _, killed, _ = _runner(tmp_path, run_for=10, code=1, touch_events_every=10)
    assert _go(run, killed).why == "crash"


def test_the_time_slice_ends_an_attempt_with_timeout_and_stops_the_process_group(tmp_path):
    run, _, killed, _ = _runner(tmp_path, run_for=10_000, touch_events_every=10)
    out = _go(run, killed, timeout_s=120)
    assert (out.code, out.why) == (-1, "timeout") and killed[0] == signal.SIGTERM


def test_silence_in_the_event_log_ends_an_attempt_as_stalled(tmp_path):
    run, _, killed, _ = _runner(tmp_path, run_for=10_000, touch_events_every=None)
    out = _go(run, killed, timeout_s=5000, stall_s=200)
    assert (out.code, out.why) == (-1, "stalled") and killed


def test_env_is_passed_to_the_attempt(tmp_path):
    run, made, killed, _ = _runner(tmp_path, run_for=10, touch_events_every=10, env={"NIGHT_OVERLAY": "{}", "AGENT_CHAT_CTX": "98304"})
    _go(run, killed)
    assert made["kwargs"]["env"]["AGENT_CHAT_CTX"] == "98304" and made["kwargs"]["env"]["NIGHT_OVERLAY"] == "{}"


def test_overlay_merge_is_deep_and_env_carries_stage_cap_and_context():
    assert overlay.merge({"coding": {"a": 1, "stages": "off"}, "x": 1}, {"coding": {"stages": "on", "cap": 5}}) == {"coding": {"a": 1, "stages": "on", "cap": 5}, "x": 1}
    e = overlay.env(stage_cap=48000, ctx=131072, relaxed=False)
    assert json.loads(e["NIGHT_OVERLAY"]) == {"coding": {"stage_prompt_cap": 48000, "stages": "on"}} and e["AGENT_CHAT_CTX"] == "131072"
    assert json.loads(overlay.env(stage_cap=0, relaxed=False)["NIGHT_OVERLAY"]) == {} and "AGENT_CHAT_CTX" not in overlay.env(stage_cap=0)
    relaxed = json.loads(overlay.env(stage_cap=48000)["NIGHT_OVERLAY"])["coding"]
    assert relaxed["stage_prompt_cap"] == 48000 and relaxed["autopilot_max_idle"] == 12


def test_boot_overlays_the_loaded_config_in_memory(monkeypatch):
    from agent import config

    seen = {}
    monkeypatch.setenv("NIGHT_OVERLAY", json.dumps({"coding": {"stage_prompt_cap": 64000}}))
    monkeypatch.setattr(config, "load_config", lambda *a, **k: {"coding": {"stage_prompt_cap": 28000, "keep": 1}})
    import agent.main as main

    monkeypatch.setattr(main, "app", lambda: seen.update(cfg=config.load_config()))
    overlay.boot()
    assert seen["cfg"] == {"coding": {"stage_prompt_cap": 64000, "keep": 1}}


def test_night_attempts_do_not_let_a_model_split_the_prompt_into_stages():
    over = json.loads(overlay.env(64000)["NIGHT_OVERLAY"])["coding"]
    assert over["stage_split_with_model"] is False                   # headed "Stage N:" prompts still split deterministically (stages.split)


def test_night_attempts_never_use_the_rewrite_in_passes_plan():
    assert json.loads(overlay.env(64000)["NIGHT_OVERLAY"])["coding"]["range_split_lines"] == 0


def test_an_interrupted_governor_stops_the_attempt_it_started(tmp_path):
    from agent.night import runner as R

    stopped = []

    class Proc(FakeProc):
        pass

    clock = Clock()
    events = tmp_path / "events.jsonl"
    events.write_text("x")
    proc = {}

    def popen(argv, **k):
        proc["p"] = Proc(100_000, clock)
        return proc["p"]

    def sleep(s):
        raise KeyboardInterrupt                                                  # Ctrl+C while the governor waits for the attempt

    orig_stop = R._stop
    R._stop = lambda p, sl: stopped.append(p.pid)
    try:
        run = R.subprocess_runner(tmp_path, events, clock=lambda: clock.t, sleep=sleep, poll_s=10, popen=popen)
        with pytest.raises(KeyboardInterrupt):
            run(Attempt(task="t", new=True, timeout_s=1000))
    finally:
        R._stop = orig_stop
    assert stopped == [4242]                                                      # the attempt's process group was stopped, then the interrupt went on
