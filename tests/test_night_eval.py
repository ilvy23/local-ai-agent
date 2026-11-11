"""The evidence of a night, the systemd watchdog, prepare/score hooks of evaluation prompts, the --then evaluation phase and the unit's new directives."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import pytest

from agent.night import evidence, gitbox, queue, sdnotify
from agent.night.runner import Outcome
from agent.night.testwatch import TestState
from tests.test_night_governor import _night, green, red
from tests.test_night_more import cli  # noqa: F401 - the fixture


LOG = """\
  › read  main.py
  › edit  wren.py
    edited wren.py: 1 edit(s) applied (exact).
  note  the bakery opens only after chapter 2
  › mark_done  {"ids": [1, 2], "evidence": "wren.py exists and tests pass"}
╭──────────────────────────────────────────────────────────────────────────────╮
│ All acceptance items confirmed.                                              │
│ Added Wren and two tests.                                                    │
╰───────────────────────── 12.000 / 95.000 tokens · 36 tok/s ──────────────────╯
"""


def test_parse_log_extracts_what_the_model_said(tmp_path):
    f = tmp_path / "a.log"
    f.write_text(LOG + "NOT RUN: reading is paused\nthe model repeated the same call 8 times\n")
    d = evidence.parse_log(f)
    assert "All acceptance items confirmed." in d["summary"] and "Added Wren and two tests." in d["summary"]
    assert d["mark_done"] == ['{"ids": [1, 2], "evidence": "wren.py exists and tests pass"}'] and d["notes"] == ["the bakery opens only after chapter 2"]
    assert d["not_run"] == 1 and d["loop_msgs"] == 1 and evidence.parse_log(tmp_path / "missing.log") == {}


def test_a_night_writes_per_attempt_evidence_a_summary_and_an_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHT_EVIDENCE_DIR", str(tmp_path / "archive"))
    root = tmp_path / "game"
    root.mkdir()

    def script(a, i):
        (root / "wren.py").write_text("x = 1\n" * 30)
        (root / ".night").mkdir(exist_ok=True)
        (root / ".night" / "wren-attempt1.log").write_text(LOG)
        return Outcome(0), 600

    n, _ = _night(root, [queue.Item("wren", "Add Wren, it must work.")], script, green, hours=2, env_lines=["coding model: qwen"], prompts_src=[])
    rep = n.run()
    rec = json.loads((root / ".night" / "evidence" / "wren-attempt1.json").read_text())
    assert rec["prompt"] == "wren" and rec["strategy"] == "start" and rec["verdict"]["kind"] == "green" and rec["tests_after"]["ok"] and rec["diff"]["added"] >= 30
    assert rec["model_said"]["mark_done"] and "Added Wren" in rec["model_said"]["summary"] and rec["events"]["steps"] == 0 and rec["head_after"]
    summary = json.loads((root / ".night" / "EVAL.json").read_text())
    assert summary["totals"]["done"] == 1 and summary["prompts"][0]["name"] == "wren" and summary["prompts"][0]["lines_added"] >= 30 and summary["machine"] == ["coding model: qwen"]
    md = (root / ".night" / "EVAL.md").read_text()
    assert "1/1 prompts done" in md and "what the model said at the end" in md and "Added Wren and two tests." in md
    archives = list((tmp_path / "archive").glob("game-*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as tar:
        names = tar.getnames()
    assert "night/EVAL.json" in names and "night/final.patch" in names and "night/git-log.txt" in names and "night/evidence/wren-attempt1.json" in names and "night/LOCK" not in names
    assert any("evidence archived" in e for e in n.events) and rep.done == ["wren"]


def test_evidence_never_stops_a_night(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "summarize", lambda night: 1 / 0)
    monkeypatch.setattr(evidence, "parse_log", lambda p, limit=0: 1 / 0)
    n, _ = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 60))[1], green, hours=1)
    assert n.run().done == ["p1"]


# ----------------------------------------------------------------------------- systemd watchdog
def test_sdnotify_sends_to_the_notify_socket(tmp_path):
    path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(path)
    srv.settimeout(2)
    try:
        assert sdnotify.send("READY=1", path) is True and srv.recv(100) == b"READY=1"
        assert sdnotify.send("WATCHDOG=1", "") is False and sdnotify.send("x", str(tmp_path / "nope.sock")) is False
    finally:
        srv.close()


def test_the_heartbeat_only_beats_while_the_main_loop_shows_life():
    sent, now = [], {"t": 1000.0}
    hb = sdnotify.Heartbeat(every_s=60, patience_s=3000, clock=lambda: now["t"], sender=lambda m: sent.append(m) or True)
    assert hb.beat() and sent == ["WATCHDOG=1"]
    now["t"] += 2900
    assert hb.beat()
    now["t"] += 200                                                                   # 3100 s without a sign of life: the loop is stuck
    assert hb.beat() is False and sent == ["WATCHDOG=1", "WATCHDOG=1"]
    hb.alive("tests run")                                                             # a sign of life
    assert hb.beat() and sent[-2:] == ["STATUS=tests run", "WATCHDOG=1"]


def test_the_heartbeat_announces_ready_and_stopping():
    sent = []
    with sdnotify.Heartbeat(every_s=3600, sender=lambda m: sent.append(m) or True):
        pass
    assert sent[0] == "READY=1" and sent[-1] == "STOPPING=1"


def test_the_governor_proves_life_during_attempts_waits_and_tests(tmp_path):
    beats = []
    n, _ = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 60))[1], green, hours=1, heartbeat=lambda *a, **k: beats.append(a))
    n.run()
    assert beats and any("p1 attempt 1" in str(b) for b in beats) and any("tests run" in str(b) for b in beats)


def test_the_unit_is_a_notify_service_with_a_watchdog_and_an_exit_log(tmp_path):
    from agent.night import unit

    text = unit.render(tmp_path, ["p"], "nightage", extra_args=["--then", "/p/eval"])
    for want in ("Type=notify", "NotifyAccess=main", "WatchdogSec=3900", "TimeoutStartSec=900", "ExecStopPost=", "unit-exits.log", "--then /p/eval", "SuccessExitStatus=3"):
        assert want in text, want


def test_git_never_hangs(tmp_path, monkeypatch):
    monkeypatch.setattr(gitbox.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 1)))
    r = gitbox.git(tmp_path, "status")
    assert r.returncode == 124 and "timed out" in r.stderr


# ----------------------------------------------------------------------------- prepare / score
def test_front_matter_carries_prepare_and_score(tmp_path):
    (tmp_path / "p.md").write_text("---\nname: hunt\nprepare: python3 inject.py\nscore: python3 score.py\n---\nFind the bugs, the tests must pass.\n")
    items, _ = queue.load([str(tmp_path)])
    assert items[0].prepare == "python3 inject.py" and items[0].score == "python3 score.py"


def test_prepare_sets_the_floor_and_score_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    root = tmp_path / "game"
    root.mkdir()
    (root / "core.py").write_text("def f():\n    return 1\n")
    seen = {}

    def script(a, i):
        seen["task"] = a.task
        (root / "core.py").write_text("def f():\n    return 1\n")                       # the model fixes the injected bug
        return Outcome(0), 300

    def tests():
        return green() if (root / "core.py").read_text().endswith("return 1\n") else red("tests/test_core.py::test_f")

    prepare = "printf 'def f():\\n    return 2\\n' > core.py && echo '{\"injected\": 1}'"
    score = "echo noise; echo '{\"injected\": 1, \"fixed\": 1}'"
    n, _ = _night(root, [queue.Item("hunt", "Find the bug, the tests must pass.", prepare=prepare, score=score)], script, tests, hours=2)
    rep = n.run()
    it = n.items[0]
    assert it.prepared and it.scored == {"injected": 1, "fixed": 1} and rep.done == ["hunt"]
    assert any("prepared" in e for e in n.events) and any("score:" in e for e in n.events) and (root / ".night" / "hunt-prepare.log").exists()
    assert it.repro_sha                                                                   # the injected state is the floor of this prompt
    summary = json.loads((root / ".night" / "EVAL.json").read_text())
    assert summary["prompts"][0]["score"] == {"injected": 1, "fixed": 1}


def test_a_broken_score_command_is_recorded_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    n, _ = _night(tmp_path / "g", [queue.Item("p", "T", score="exit 3")], lambda a, i: (Outcome(0), 60), green, hours=1) if (tmp_path / "g").mkdir() is None else (None, None)
    n.run()
    assert "no JSON result" in n.items[0].scored["error"]


# ----------------------------------------------------------------------------- the --then evaluation phase
def test_then_phase_runs_the_eval_prompts_on_a_copy_without_night_state(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    (proj / ".night").mkdir(exist_ok=True)
    (proj / ".night" / "state.json").write_text("{}")
    gitbox.ensure_repo(proj)
    gitbox.git(proj, "tag", "night/green-1")
    calls = []
    real_run = subprocess.run

    def fake_run(cmd, **k):
        if isinstance(cmd, list) and "night" in cmd and "run" in cmd and "-c" in cmd:      # only the nested `night run` is faked: git must really run
            calls.append(cmd)
            return type("R", (), {"returncode": 0})()
        return real_run(cmd, **k)

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    code = C._then_phase(proj, str(tmp_path / "eval-prompts"), time.time() + 3 * 3600)
    dest = proj.parent / f"{proj.name}-eval"
    assert code == 0 and dest.exists() and (dest / "app.py").exists() and not (dest / ".night").exists()
    assert gitbox.git(dest, "tag", "-l", "night/*").stdout.strip() == ""                    # the copy starts a night of its own
    cmd = calls[0]
    assert "--folder" in cmd and str(dest) in cmd and "--resume" in cmd and "--until" in cmd and str(tmp_path / "eval-prompts") in cmd
    assert (proj / "app.py").exists()                                                       # the main result is untouched


def test_the_done_marker_waits_for_the_evaluation_phase(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    order = []
    monkeypatch.setattr(C, "_then_phase", lambda root, then, end: order.append(("then", (proj / ".night" / "DONE").exists())) or 0)
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary", "--hours", "3", "--then", str(tmp_path / "e"), "--then-min-hours", "1"])
    assert res.exit_code == 0 and order == [("then", False)]                                   # no DONE marker while the evaluation phase is still to come
    assert (proj / ".night" / "DONE").exists() and "evaluation phase" in (proj / ".night" / "DONE").read_text()      # ... and one after it


def test_no_evaluation_phase_when_too_little_time_is_left(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    called = []
    monkeypatch.setattr(C, "_then_phase", lambda *a: called.append(a) or 0)
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary", "--hours", "1", "--then", str(tmp_path / "e"), "--then-min-hours", "2"])
    assert res.exit_code == 0 and called == [] and (proj / ".night" / "DONE").exists()


def test_resume_of_a_finished_main_night_goes_straight_to_the_evaluation_phase(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    nd = proj / ".night"
    nd.mkdir(exist_ok=True)
    state = {"left_s": 20000, "end_wall": time.time() + 5 * 3600, "items": [{"name": "inline-1", "status": "done", "attempts": 1, "seconds": 5.0, "note": "", "kinds": ["green"]}], "events": []}
    (nd / "state.json").write_text(json.dumps(state))
    ran = []
    monkeypatch.setattr(C, "_then_phase", lambda root, then, end: ran.append(then) or 0)
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary", "--resume", "--then", str(tmp_path / "e")])
    assert res.exit_code == 0 and ran == [str(tmp_path / "e")] and "main prompts are done" in res.output


def test_eval_command_prints_the_evaluation(cli):
    runner, C, proj = cli
    assert runner.invoke(C.night_app, ["eval", "--folder", str(proj)]).exit_code == 1
    (proj / ".night").mkdir(exist_ok=True)
    (proj / ".night" / "EVAL.md").write_text("# Evaluation of the night: x\n")
    res = runner.invoke(C.night_app, ["eval", "--folder", str(proj)])
    assert res.exit_code == 0 and "Evaluation of the night" in res.output


def test_a_stopped_night_still_leaves_a_report_and_an_evaluation(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHT_EVIDENCE_DIR", str(tmp_path / "archive"))
    runner, C, proj = cli
    real = C.Night

    class Stopped(real):
        def run(self):
            self.items[0].attempts = 1
            raise KeyboardInterrupt                                                       # Ctrl+C / SIGTERM (systemctl stop) during the night

    monkeypatch.setattr(C, "Night", Stopped)
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary"])
    assert res.exit_code == 130
    nd = proj / ".night"
    assert "NOT an end state" in (nd / "NIGHT_REPORT.md").read_text() and (nd / "EVAL.md").exists() and (nd / "EVAL.json").exists()
    assert list((tmp_path / "archive").glob("*.tar.gz")) and not (nd / "LOCK").exists() and not (nd / "DONE").exists()
