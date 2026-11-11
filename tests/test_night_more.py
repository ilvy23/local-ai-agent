"""Robustness parts of `nightage night`: the whole-night inhibitor, the folder lock, the attempt log and what it teaches, machine info, doctor, error isolation in
the governor, and the CLI end to end on fakes (no model, no server)."""
from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from agent.night import analyzer, attemptlog, doctor, envinfo, gitbox, handoff, inhibit, lock, queue
from agent.night.probe import Facts
from agent.night.runner import Outcome
from agent.night.testwatch import TestState
from tests.test_night_governor import _night, green
from tests.test_night_parts import _events


# ----------------------------------------------------------------------------- inhibit / lock
class FakeProc:
    def __init__(self, *a, **k):
        self.argv, self.terminated, self.alive = a[0], False, True

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated, self.alive = True, False

    def wait(self, timeout=None):
        return 0


def test_inhibitor_holds_systemd_inhibit_for_the_whole_run_and_ends_it():
    made = []
    with inhibit.Inhibitor(popen=lambda *a, **k: made.append(FakeProc(*a)) or made[-1], which=lambda n: "/bin/" + n) as inh:
        assert inh.active and "--what=sleep:idle" in made[0].argv and made[0].argv[-2:] == ["sleep", str(14 * 3600)]
    assert made[0].terminated and not inh.active
    with inhibit.Inhibitor(popen=lambda *a, **k: 1 / 0, which=lambda n: None) as none:
        assert not none.active                                                     # no systemd-inhibit: the run still works


def test_lock_blocks_a_second_live_night_but_takes_over_a_stale_one(tmp_path, monkeypatch):
    assert lock.acquire(tmp_path) and lock.owner(tmp_path) is None                 # our own pid is not "another night"
    (tmp_path / ".night" / "LOCK").write_text("1")                                 # pid 1 is alive and is not us
    assert lock.owner(tmp_path) == 1 and not lock.acquire(tmp_path)
    (tmp_path / ".night" / "LOCK").write_text("999999999")                         # a pid that does not exist: a crash left it
    assert lock.owner(tmp_path) is None and lock.acquire(tmp_path)
    lock.release(tmp_path)
    assert not (tmp_path / ".night" / "LOCK").exists()
    (tmp_path / ".night" / "LOCK").write_text("1")
    lock.release(tmp_path)                                                         # never removes somebody else's lock
    assert (tmp_path / ".night" / "LOCK").exists()


# ----------------------------------------------------------------------------- attempt log / learning
def test_attempt_log_round_trip_and_rates(tmp_path):
    f = tmp_path / "a.jsonl"
    for strategy, green_, progress in [("small", True, True), ("small", True, True), ("small", False, True), ("fresh", False, False)]:
        attemptlog.record(f, strategy=strategy, trigger="no_edits", green=green_, progress=progress)
    attemptlog.record(f, strategy="small", trigger="loop", green=True)
    f.write_text(f.read_text() + "not json\n")
    data = attemptlog.rows(f)
    assert len(data) == 5 and attemptlog.rows(tmp_path / "none.jsonl") == []
    assert attemptlog.rates(data, "no_edits") == {"small": (3, pytest.approx((1 + 1 + 0.5) / 3)), "fresh": (1, 0.0)}


def test_prefer_needs_evidence_to_overrule_the_default():
    few = {"small": (2, 1.0)}
    assert attemptlog.prefer("fresh", ["fresh", "small"], few) == "fresh"                       # two lucky runs change nothing
    enough = {"small": (3, 1.0), "fresh": (3, 0.0)}
    assert attemptlog.prefer("fresh", ["fresh", "small"], enough) == "small"
    close = {"small": (5, 0.6), "fresh": (5, 0.5)}
    assert attemptlog.prefer("fresh", ["fresh", "small"], close) == "fresh"                      # not clearly better
    assert analyzer.next_strategy("no_edits", [], {"fresh": (3, 1.0), "small": (3, 0.0)}) == "fresh"   # the night's own evidence beat the default plan


def test_the_governor_records_every_attempt_and_the_report_shows_them(tmp_path):
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 300))[1], green, hours=1, env_lines=["coding model: qwen"])
    rep = n.run()
    rows = attemptlog.rows(tmp_path / ".night" / "attempts.jsonl")
    assert len(rows) == 1 and rows[0]["prompt"] == "p1" and rows[0]["green"] and rows[0]["strategy"] == "start" and rows[0]["minutes"] == 5.0
    assert "## Attempts (measured)" in rep.text and "| p1 | 1 | start |" in rep.text and "## Machine" in rep.text and "coding model: qwen" in rep.text and "Strategies: start 1/1 green" in rep.text


# ----------------------------------------------------------------------------- machine info / doctor
def test_envinfo_reports_models_cap_and_disk(tmp_path):
    lines = envinfo.collect(tmp_path, {"models": {"chat": "chat-m", "coding": "code-m"}}, {"coding": {"stage_prompt_cap": 64000}})
    text = "\n".join(lines)
    assert "coding model: code-m" in text and "stage prompt cap: 64000" in text and "disk free:" in text and "host:" in text


def _facts(ok=True):
    return Facts("pytest -q", 3.0, TestState(ok=True, passed=12)) if ok else Facts("pytest", 0.1, TestState(ok=False, code=127, tail="command not found"), "the test command cannot run: x")


def test_doctor_marks_fatal_problems_and_warnings_apart(tmp_path):
    good = doctor.run(tmp_path, coding_model="qwen3.8:27b-text", installed=["qwen3.8:27b-text"], urls=["http://x"], test_facts=_facts(), probe=lambda u: True,
                      game=lambda: None, temps=lambda: [55.0], which=lambda n: "/bin/" + n, free_gb=100)
    assert doctor.text(good).endswith("ready") and "not ready" not in doctor.text(good)
    bad = doctor.run(tmp_path, coding_model="qwen3.8:27b-text", installed=["other:1b"], urls=["http://x"], test_facts=_facts(False), probe=lambda u: False,
                     game=lambda: "proton", temps=lambda: [88.0], which=lambda n: None, free_gb=1)
    text = doctor.text(bad)
    by = {c.name: c for c in bad}
    assert not by["coding model"].ok and not by["test command"].ok and not by["disk"].ok and not by["git"].ok
    assert not by["model server http://x"].fatal and not by["no game running"].fatal and not by["gpu temperature"].fatal and not by["keeps the pc awake"].fatal
    assert "not ready: git, coding model, test command, disk" in text and "FAIL  coding model" in text and "warn  no game running" in text


# ----------------------------------------------------------------------------- governor error isolation
def test_a_governor_bug_costs_one_prompt_not_the_night(tmp_path):
    calls = []

    def script(a, i):
        calls.append(a.task)
        if "BOOM" in a.task:
            raise RuntimeError("runner exploded")
        (tmp_path / f"f{i}").write_text("x")
        return Outcome(0), 300

    n, seen = _night(tmp_path, [queue.Item("p1", "BOOM please"), queue.Item("p2", "fine task")], script, green, hours=4)
    rep = n.run()
    assert "p2" in rep.done and "p1" in rep.partial and "governor error" in n.items[0].note
    assert "runner exploded" in (tmp_path / ".night" / "errors.log").read_text() and any("governor error" in e for e in n.events)


# ----------------------------------------------------------------------------- the CLI end to end
@pytest.fixture
def cli(tmp_path, monkeypatch):
    from agent.night import cli as C

    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    proj = tmp_path / "work" / "deep" / "project"
    proj.mkdir(parents=True)
    (proj / "app.py").write_text("def main():\n    return 1\n")

    def fake_runner(root, events, env=None, **kw):
        def run(attempt):
            if "night_canary.txt" in attempt.task:
                (root / "night_canary.txt").write_text("ok\n")
                return Outcome(0)
            (root / "feature.py").write_text("def feature():\n    return 2\n")
            return Outcome(0)
        return run

    class Quiet:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(C, "subprocess_runner", fake_runner)
    monkeypatch.setattr(C, "_tests", lambda root, timeout_s=0: (lambda: TestState(ok=True, passed=1, digest="ok", tail="1 passed")))
    class Fit:
        def gate(self):
            return None

    monkeypatch.setattr(C, "_health", lambda root, pause: Fit())
    monkeypatch.setattr(C.inhibit, "Inhibitor", Quiet)
    monkeypatch.setattr("agent.night.notify.print", lambda *a, **k: None, raising=False)
    return CliRunner(), C, proj


def test_cli_run_end_to_end_restores_the_policy_releases_the_lock_and_leaves_a_report(cli):
    runner, C, proj = cli
    from agent.codeagent import policy

    before = policy.for_repo(proj).mode
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--hours", "2", "--ignore-games"])
    assert res.exit_code == 0, res.output
    nd = proj / ".night"
    assert (nd / "NIGHT_REPORT.md").exists() and (nd / "DONE").exists() and (proj / "feature.py").exists() and not (nd / "LOCK").exists()
    assert json.loads((nd / "state.json").read_text())["items"][0]["status"] == "done" and "## Machine" in (nd / "NIGHT_REPORT.md").read_text()
    assert policy.for_repo(proj).mode == before and not policy.for_repo(proj).dangerous_allow               # the policy is back to what it was
    assert gitbox.git(proj, "tag", "-l", "night/green-*").stdout.strip()
    status = runner.invoke(C.night_app, ["status", "--folder", str(proj)])
    assert status.exit_code == 0 and "done" in status.output
    rep = runner.invoke(C.night_app, ["report", "--folder", str(proj)])
    assert rep.exit_code == 0 and "Night report" in rep.output


def test_cli_run_refuses_a_second_night_a_bad_folder_and_empty_prompts(cli, tmp_path):
    runner, C, proj = cli
    (proj / ".night").mkdir()
    (proj / ".night" / "LOCK").write_text("1")                                     # a live pid that is not us
    assert runner.invoke(C.night_app, ["run", "Do it, it must work and the tests pass " * 3, "--folder", str(proj), "--yes", "--no-model"]).exit_code == 1
    assert runner.invoke(C.night_app, ["run", "x", "--folder", str(os.path.expanduser("~")), "--yes"]).exit_code == 1
    assert runner.invoke(C.night_app, ["run", str(tmp_path / "missing-and-not-text"), "--folder", str(proj), "--yes", "--no-model", "--hours", "1"]).exit_code in (0, 1, 3)


def test_cli_stop_pause_go_write_and_clear_the_control_file(cli):
    runner, C, proj = cli
    (proj / ".night").mkdir()
    assert runner.invoke(C.night_app, ["stop", "--folder", str(proj)]).exit_code == 0 and (proj / ".night" / "CONTROL").read_text().strip() == "stop"
    runner.invoke(C.night_app, ["pause", "--folder", str(proj)])
    assert (proj / ".night" / "CONTROL").read_text().strip() == "pause"
    runner.invoke(C.night_app, ["go", "--folder", str(proj)])
    assert not (proj / ".night" / "CONTROL").exists()
    assert runner.invoke(C.night_app, ["stop", "--folder", str(proj / "nope")]).exit_code == 1


def test_cli_init_split_and_plan(cli, tmp_path):
    runner, C, proj = cli
    out = tmp_path / "prompts"
    assert runner.invoke(C.night_app, ["init", str(out)]).exit_code == 0 and (out / "rules.md").exists() and (out / "02-bug.md").exists()
    big = tmp_path / "big.md"
    big.write_text("Stage 1: one\nDo the first thing, it must work.\n\nStage 2: two\nDo the second thing, tests should pass.\n")
    assert runner.invoke(C.night_app, ["split", str(big), str(tmp_path / "parts")]).exit_code == 0 and len(list((tmp_path / "parts").glob("*.md"))) == 2
    one = tmp_path / "one.md"
    one.write_text("no stages here")
    assert runner.invoke(C.night_app, ["split", str(one), str(tmp_path / "x")]).exit_code == 1
    res = runner.invoke(C.night_app, ["plan", str(tmp_path / "parts"), "--folder", str(proj), "--hours", "6"])
    assert res.exit_code == 0 and "Night plan for" in res.output and "01-stage-1-one" in res.output and "Machine:" in res.output


# ----------------------------------------------------------------------------- reaper / vram hogs
def test_reap_session_ends_exactly_what_an_attempt_left_behind(tmp_path):
    import subprocess
    import sys
    import time

    from agent.night import reap

    keep = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])                         # another session (ours): must survive
    stray = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)  # what an attempt left running
    try:
        time.sleep(0.2)
        assert [pid for pid, _ in reap.members(stray.pid)] == [stray.pid]
        killed = reap.reap_session(stray.pid, grace_s=0.2)
        assert len(killed) == 1 and "time.sleep" in killed[0]
        stray.wait(timeout=5)
        assert stray.returncode is not None and keep.poll() is None
        assert reap.reap_session(stray.pid, grace_s=0) == []
    finally:
        keep.kill()
        stray.kill()


def test_reap_stat_parser_survives_spaces_and_parentheses_in_the_command_name():
    from agent.night import reap

    assert reap._sid("123 (my (weird) cmd) S 1 123 777 0 -1 4194560") == 777
    assert reap._sid("garbage") is None


def test_the_runner_only_reaps_real_session_leaders(tmp_path):
    from agent.night import runner

    assert runner.real_session(999999999) is False and runner.real_session(os.getpid()) in (True, False)
    assert runner._reap(999999999, lambda s: None) == []


def _fdinfo(tmp_path, pid, comm, files):
    d = tmp_path / str(pid)
    (d / "fdinfo").mkdir(parents=True)
    (d / "comm").write_text(comm + "\n")
    for i, text in enumerate(files):
        (d / "fdinfo" / str(i + 3)).write_text(text)


def test_vram_hogs_sums_per_client_once_and_adds_nvidia(tmp_path):
    from agent.night import vramhogs

    _fdinfo(tmp_path, 100, "firefox", ["drm-client-id: 7\ndrm-memory-vram: 524288 KiB\n", "drm-client-id: 7\ndrm-memory-vram: 524288 KiB\n", "drm-client-id: 8\ndrm-memory-vram:\t262144 KiB\n"])
    _fdinfo(tmp_path, 200, "steam", ["drm-total-vram0: 2 GiB\n"])
    _fdinfo(tmp_path, 300, "tiny", ["drm-memory-vram: 1024 KiB\n"])
    _fdinfo(tmp_path, 400, "notgpu", ["pos: 0\nflags: 02\n"])
    (tmp_path / "self").mkdir()
    rows = vramhogs.hogs(tmp_path, nvidia=lambda: [(100, 300), (999, 50)])
    assert rows == [(200, "steam", 2048), (100, "firefox", 1068)]
    assert vramhogs.summary(rows) == "3116 MB held by other programs: steam 2048 MB, firefox 1068 MB" and "no other program" in vramhogs.summary([])


def test_doctor_warns_about_graphics_memory_hogs(tmp_path):
    kw = dict(coding_model=None, installed=None, urls=[], test_facts=_facts(), probe=lambda u: True, game=lambda: None, temps=lambda: [], which=lambda n: "/bin/" + n, free_gb=100)
    busy = {c.name: c for c in doctor.run(tmp_path, hogs=[(1, "game", 3000)], **kw)}
    assert not busy["graphics memory is free"].ok and not busy["graphics memory is free"].fatal and "game 3000 MB" in busy["graphics memory is free"].detail
    assert {c.name: c for c in doctor.run(tmp_path, hogs=[], **kw)}["graphics memory is free"].ok


# ----------------------------------------------------------------------------- tune
def _cpu(tmp_path, spec):
    """spec: {cpu_number: (max_khz, siblings)}"""
    for n, (khz, sib) in spec.items():
        d = tmp_path / f"cpu{n}"
        (d / "cpufreq").mkdir(parents=True)
        (d / "topology").mkdir()
        (d / "cpufreq" / "cpuinfo_max_freq").write_text(str(khz))
        (d / "topology" / "thread_siblings_list").write_text(sib)
    (tmp_path / "cpufreq").mkdir()                                                   # noise next to the cpuN folders


def test_p_core_threads_picks_one_thread_per_fast_core_on_a_hybrid_cpu(tmp_path):
    from agent.night import tune

    _cpu(tmp_path, {0: (5100000, "0-1"), 1: (5100000, "0-1"), 2: (5100000, "2-3"), 3: (5100000, "2-3"), 4: (3900000, "4"), 5: (3900000, "5")})
    assert tune.p_core_threads(tmp_path) == [0, 2]
    flat = tmp_path / "flat"
    flat.mkdir()
    _cpu(flat, {0: (4000000, "0"), 1: (4000000, "1")})
    assert tune.p_core_threads(flat) == [] and tune.p_core_threads(tmp_path / "missing") == []                 # not hybrid / unknown: no pinning


def test_tune_writes_only_its_own_dropin_and_reverts_it(tmp_path):
    from agent.night import tune

    calls = []
    run = lambda cmd, **k: calls.append(cmd)
    mine = tmp_path / "systemd" / "user" / "ollama-vk.service.d"
    mine.mkdir(parents=True)
    (mine / "tune.conf").write_text("[Service]\nEnvironment=LLAMA_ARG_FIT_TARGET=256\n")                       # the user's own file
    plan = tune.Plan(64, [0, 2, 4])
    assert plan.text().count("[Service]") == 2 and "LLAMA_ARG_FIT_TARGET=64" in plan.text() and "CPUAffinity=0 2 4" in plan.text()
    assert tune.Plan(None, []).text().count("[Service]") == 0
    p = tune.apply(plan, tmp_path, run)
    assert p.name == "zz-night.conf" and p.parent == mine and tune.current(tmp_path) == plan.text() and calls == [["systemctl", "--user", "daemon-reload"]]
    assert (mine / "tune.conf").read_text().endswith("=256\n")                                                  # untouched
    tune.apply(plan, tmp_path, run, restart=True)
    assert calls[-1] == ["systemctl", "--user", "restart", "ollama-vk"]
    assert tune.revert(tmp_path, run) is True and tune.current(tmp_path) is None and (mine / "tune.conf").exists() and tune.revert(tmp_path, run) is False


def test_cli_tune_shows_by_default_and_writes_only_with_apply(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    from agent.night import tune

    monkeypatch.setattr(tune, "dropin_path", lambda ch=None: tmp_path / "zz-night.conf")
    monkeypatch.setattr(tune.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr("agent.night.vramhogs.hogs", lambda *a, **k: [])
    shown = runner.invoke(C.night_app, ["tune"])
    assert shown.exit_code == 0 and "not written" in shown.output and not (tmp_path / "zz-night.conf").exists() and "FIT_TARGET=64" in shown.output
    done = runner.invoke(C.night_app, ["tune", "--fit", "128", "--apply"])
    assert done.exit_code == 0 and "FIT_TARGET=128" in (tmp_path / "zz-night.conf").read_text() and "written" in done.output
    assert "removed" in runner.invoke(C.night_app, ["tune", "--revert"]).output and not (tmp_path / "zz-night.conf").exists()


# ----------------------------------------------------------------------------- morning tools: review / undo / squash
def _night_repo(tmp_path):
    (tmp_path / "a.txt").write_text("base")
    gitbox.ensure_repo(tmp_path)
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "b.txt").write_text("new")
    gitbox.checkpoint(tmp_path, "first")
    gitbox.mark_done(tmp_path, "Prompt One!")
    (tmp_path / "a.txt").write_text("two")
    gitbox.checkpoint(tmp_path, "second")
    return tmp_path


def test_done_tags_numstat_and_review_text(tmp_path):
    root = _night_repo(tmp_path)
    assert gitbox.tags(root, "night/done-") == ["night/done-prompt-one"]
    assert gitbox.numstat(root, "night/base", "night/done-prompt-one") == (2, 2, 1)
    text = gitbox.review_text(root)
    assert "Finished prompts: prompt-one" in text and "nightage night squash" in text and "nightage night undo" in text and "a.txt" in text


def test_undo_goes_back_to_the_baseline_and_keeps_the_night_under_a_tag(tmp_path):
    root = _night_repo(tmp_path)
    (root / "late.txt").write_text("after the last checkpoint")
    keep = gitbox.undo(root)
    assert keep == "night/undone-1" and (root / "a.txt").read_text() == "base" and not (root / "b.txt").exists() and not (root / "late.txt").exists()
    show = gitbox.git(root, "show", f"{keep}:late.txt")
    assert show.stdout == "after the last checkpoint"                                      # nothing was lost
    assert gitbox.undo(root) == "night/undone-2"


def test_squash_makes_one_commit_and_keeps_the_old_tip(tmp_path):
    root = _night_repo(tmp_path)
    sha = gitbox.squash(root, "overnight work")
    assert sha and gitbox.git(root, "rev-list", "--count", "night/base..HEAD").stdout.strip() == "1"
    assert (root / "a.txt").read_text() == "two" and (root / "b.txt").exists()
    assert gitbox.git(root, "log", "-1", "--format=%s").stdout.strip() == "overnight work" and gitbox.tags(root, "night/before-squash-") == ["night/before-squash-1"]
    assert gitbox.squash(root, "again") is not None
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "x").write_text("1")
    gitbox.ensure_repo(fresh)
    assert gitbox.squash(fresh, "nothing") is None


def test_cli_review_undo_squash(cli):
    runner, C, proj = cli
    root = _night_repo(proj)
    assert "What the night changed" in runner.invoke(C.night_app, ["review", "--folder", str(root)]).output
    assert runner.invoke(C.night_app, ["squash", "--folder", str(root), "-m", "all in one"]).exit_code == 0
    assert gitbox.git(root, "log", "-1", "--format=%s").stdout.strip() == "all in one"
    (root / ".night").mkdir(exist_ok=True)
    (root / ".night" / "LOCK").write_text("1")
    assert runner.invoke(C.night_app, ["undo", "--folder", str(root), "--yes"]).exit_code == 1                    # a night is still running
    (root / ".night" / "LOCK").unlink()
    assert runner.invoke(C.night_app, ["undo", "--folder", str(root)], input="n\n").exit_code == 1               # asked, answered no
    assert runner.invoke(C.night_app, ["undo", "--folder", str(root), "--yes"]).exit_code == 0 and (root / "a.txt").read_text() == "base"
    bare = proj.parent / "bare"
    bare.mkdir()
    assert runner.invoke(C.night_app, ["review", "--folder", str(bare)]).exit_code == 1


# ----------------------------------------------------------------------------- config defaults / gates
def test_night_config_is_validated_and_clamped():
    from agent.night import nightconf

    assert nightconf.from_config(None) == nightconf.DEFAULTS and nightconf.from_config({"night": "garbage"}) == nightconf.DEFAULTS
    got = nightconf.from_config({"night": {"hours": 500, "tournament": "2", "review": "yes", "map": "everything", "stage_cap": -5, "ctx": "98304", "pause_on_game": "off"}})
    assert got == {"hours": 24.0, "tournament": 2, "review": True, "map": "auto", "stage_cap": 0, "ctx": 98304, "pause_on_game": False}
    assert nightconf.from_config({"night": {"hours": "abc", "tournament": 99}})["hours"] == 8.0 and nightconf.from_config({"night": {"tournament": 99}})["tournament"] == 6


def test_cli_uses_config_defaults_and_options_win(cli, monkeypatch):
    runner, C, proj = cli
    seen = {}
    real_night = C.Night

    class Spy(real_night):
        def __init__(self, root, items, rules, hours, **kw):
            seen.update(hours=hours, tournament=kw["tournament"], review=kw["review"], map_mode=kw["map_mode"])
            super().__init__(root, items, rules, hours, **kw)

    monkeypatch.setattr(C, "Night", Spy)
    monkeypatch.setattr("agent.config.load_config", lambda *a, **k: {"night": {"hours": 3, "tournament": 0, "review": True, "map": "off"}})
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    assert runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games"]).exit_code == 0
    assert seen == {"hours": 3, "tournament": 0, "review": True, "map_mode": "off"}
    (proj / "feature.py").unlink()
    assert runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--hours", "1.5", "--no-review", "--map", "auto"]).exit_code == 0
    assert seen == {"hours": 1.5, "tournament": 0, "review": False, "map_mode": "auto"}


def test_gates_build_must_pass_lint_is_advice(tmp_path):
    from agent.night import gates

    class Prof:
        build_cmd, lint_cmd = "make build", "ruff check"

    gs = gates.from_profile(Prof())
    assert [(g.name, g.required) for g in gs] == [("build", True), ("lint", False)] and gates.from_profile(object()) == []
    base = TestState(ok=True, passed=5, digest="5 passed")
    results = gates.run(tmp_path, gs, runner=lambda cmd, cwd, t: (0, "ok") if "build" in cmd else (1, "E501 line too long\n"))
    folded = gates.apply(base, results)
    assert folded.ok and "lint reports problems (advice only)" in folded.digest and "E501" in folded.digest
    broken = gates.apply(base, gates.run(tmp_path, gs, runner=lambda cmd, cwd, t: (2, "error: cannot find module x") if "build" in cmd else (0, "")))
    assert not broken.ok and broken.digest.startswith("build FAILED (`make build`): error: cannot find module x")
    assert gates.apply(base, []) is base
    assert gates.run(tmp_path, [gates.Gate("x", "echo hi", True)])[0].ok and not gates.run(tmp_path, [gates.Gate("x", "exit 3", True)])[0].ok


# ----------------------------------------------------------------------------- plan file
PLAN = """# Plan

- [ ] 1. Add the Wren class | check: file wren.py
- [ ] 2. Her test passes | check: test tests/test_wren.py::test_line
- [ ] 3. The README mentions her | check: grep -q Wren README.md
- [ ] 4. Polish the dialogue
"""


def test_planfile_parse_render_and_usable():
    from agent.night import planfile

    steps = planfile.parse(PLAN)
    assert [s.text for s in steps] == ["1. Add the Wren class", "2. Her test passes", "3. The README mentions her", "4. Polish the dialogue"]
    assert steps[0].check == "file wren.py" and steps[3].check == "" and planfile.usable(steps)
    assert not planfile.usable(steps[:2]) and not planfile.usable([planfile.Step("a", False), planfile.Step("b", False), planfile.Step("c", False)])
    again = planfile.parse(planfile.render(steps, "x"))
    assert [(s.text.split(". ", 1)[-1], s.done, s.check) for s in again] == [(s.text.split(". ", 1)[-1], s.done, s.check) for s in steps]


def test_planfile_evaluate_ticks_by_measurement_and_unticks_when_a_check_fails(tmp_path):
    from agent.night import planfile

    p = planfile.path(tmp_path, "Chars One")
    p.parent.mkdir(parents=True)
    p.write_text(PLAN.replace("[ ] 4.", "[x] 4."))
    (tmp_path / "wren.py").write_text("")
    state = TestState(ok=False, passed=3, failed=["tests/test_other.py::test_x"])
    done, total = planfile.evaluate(tmp_path, "Chars One", state, sh=lambda cmd, cwd: False)
    assert (done, total) == (3, 4)                                                 # file ok, test not failing, README check fails, step 4 (no check) keeps the model's tick
    text = p.read_text()
    assert "[x] 1. Add the Wren class" in text and "[x] 2." in text and "[ ] 3." in text and "[x] 4." in text
    state = TestState(ok=False, failed=["tests/test_wren.py::test_line"])
    assert planfile.evaluate(tmp_path, "Chars One", state, sh=lambda cmd, cwd: True)[0] == 3        # step 2 now fails -> unticked; step 3 now passes
    assert "[ ] 2." in p.read_text() and planfile.remaining(tmp_path, "Chars One").startswith("2. Her test passes")
    assert planfile.evaluate(tmp_path, "nope", state) == (0, 0) and planfile.remaining(tmp_path, "nope") == ""
    assert planfile.check(planfile.Step("x", False, "file ../secret"), tmp_path, None) is False and planfile.check(planfile.Step("x", False, "file /etc/passwd"), tmp_path, None) is False


def test_governor_plans_first_and_follows_the_plan_in_later_attempts(tmp_path):
    seen_tasks = []

    def script(a, i):
        seen_tasks.append(a.task)
        if "checklist in" in a.task:
            plan = tmp_path / ".night" / "plans" / "p1.md"
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text("- [ ] 1. make feature | check: file feature.py\n- [ ] 2. make docs | check: file docs.md\n- [ ] 3. tests | check: file test_feature.py\n")
        else:
            (tmp_path / "feature.py").write_text("x = 1\n")
        return Outcome(0), 600

    n, seen = _night(tmp_path, [queue.Item("p1", "Build the feature, it must work.", plan=True)], script, green, hours=4)
    rep = n.run()
    assert rep.done == ["p1"] and "checklist in .night/plans/p1.md" in seen_tasks[0] and ".night/plans/p1.md: follow it" in seen_tasks[1] and "Still open" in seen_tasks[1]
    assert any("plan with 3 steps" in e for e in n.events) and any("plan: 1/3 steps verified" in e for e in n.events)
    assert "[x] 1." in (tmp_path / ".night" / "plans" / "p1.md").read_text()


def test_an_unusable_plan_is_undone_and_the_prompt_goes_on_without_one(tmp_path):
    def script(a, i):
        if "checklist in" in a.task:
            (tmp_path / "sneaky.py").write_text("changed code instead of planning\n")
            plan = tmp_path / ".night" / "plans" / "p1.md"
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text("- [ ] 1. only one step\n")
        else:
            (tmp_path / "feature.py").write_text("x = 1\n")
        return Outcome(0), 600

    n, seen = _night(tmp_path, [queue.Item("p1", "Build the feature, it must work.", plan=True)], script, green, hours=4)
    rep = n.run()
    assert rep.done == ["p1"] and sum("plan attempt" in e and "unusable" in e for e in n.events) == 2 and not (tmp_path / "sneaky.py").exists()
    assert not (tmp_path / ".night" / "plans" / "p1.md").exists()


# ----------------------------------------------------------------------------- relaxed limits / guard meter / wall clock / idle watchdog / test timeout
def test_relaxed_night_limits_only_loosen_beyond_the_new_defaults_and_use_real_config_keys(tmp_path):
    from agent.night import overlay

    over = json.loads(overlay.env(64000, 0)["NIGHT_OVERLAY"])["coding"]
    from agent.codeagent import stages, supervisor

    assert over["verify_phase_budget"] == 0 and over["stage_prompt_cap"] == 64000 and over["stages"] == "on"
    assert over["autopilot_max_idle"] > supervisor.MAX_IDLE and over["autopilot_max_segments"] > supervisor.MAX_SEGMENTS and over["stage_attempts"] > stages.DEFAULT_ATTEMPTS
    assert over["stage_max_minutes"] > stages.STAGE_MAX_MINUTES and over["stage_max_steps"] > stages.STAGE_MAX_STEPS and over["stage_min_edits_per_100_steps"] < stages.STAGE_MIN_EDITS
    assert "autopilot_max_idle" not in json.loads(overlay.env(64000, 0, relaxed=False)["NIGHT_OVERLAY"])["coding"]
    assert json.loads(overlay.env(0, 0, extra={"coding": {"look_block_at": 7}})["NIGHT_OVERLAY"])["coding"]["look_block_at"] == 7        # an explicit extra wins


def test_guard_meter_counts_interventions_per_step(tmp_path):
    rows = [{"ts": 10, "session": "p-1", "kind": "step", "think": i < 2, "gen": 10, "decode_s": 1, "prefill_s": 1, "prompt": 100} for i in range(10)]
    rows += [{"ts": 11, "session": "p-1", "kind": k} for k in ("look_blocked", "look_blocked", "loop_abort", "verify_budget", "stage_cap", "stage_mask", "edit_gap", "wander")]
    es = handoff.summarize_events(_events(tmp_path, rows), 0, "p")
    assert (es.think_steps, es.loop_aborts, es.verify_budgets, es.masks, es.edit_gaps, es.wanders) == (2, 1, 1, 1, 1, 1)
    assert es.limits_hit == 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 and es.guard_share == pytest.approx(10 / 10) and handoff.EventSummary().guard_share == 0.0


def test_report_tells_how_much_the_guards_cost(tmp_path):
    ev = tmp_path / "events.jsonl"
    rows = [{"ts": 1000 + i, "session": tmp_path.name + "-1", "kind": "step", "gen": 40, "decode_s": 1.0, "prefill_s": 1.0, "prompt": 9000} for i in range(10)]
    rows += [{"ts": 1001, "session": tmp_path.name + "-1", "kind": "look_blocked"}] * 3
    ev.write_text("\n".join(json.dumps(r) for r in rows))
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 100))[1], green, hours=1, events=ev, wall=lambda: 900.0)
    rep = n.run()
    assert "Guards of the coding agent interfered 3 times in 10 model steps (30%): 3 blocked reads" in rep.text


def test_the_night_ends_at_the_wall_clock_deadline_even_if_the_monotonic_clock_stood_still(tmp_path):
    wall = {"t": 1_000_000.0}
    n, seen = _night(tmp_path, [queue.Item("p1", "T"), queue.Item("p2", "T2")], lambda a, i: ((tmp_path / f"f{i}").write_text("x"), wall.__setitem__("t", wall["t"] + 20_000), (Outcome(0), 10))[2],
                     green, hours=4, wall=lambda: wall["t"])
    n.run()
    assert len(seen) == 1 and n.remaining() <= 0 and n.items[1].status != "done"                 # 20 000 s of wall time passed in an attempt the monotonic clock saw as 10 s


def test_runner_ends_an_attempt_when_the_work_tree_stops_moving_though_events_flow(tmp_path):
    run, made, killed, clock = _runner_with_progress(tmp_path)
    out = _go(run, killed, timeout_s=10_000, stall_s=10_000, idle_s=600)
    assert (out.code, out.why) == (-1, "idle") and killed


def test_runner_lets_an_attempt_live_while_the_work_tree_keeps_changing(tmp_path):
    run, made, killed, clock = _runner_with_progress(tmp_path, moving=True, run_for=2000)
    out = _go(run, killed, timeout_s=10_000, stall_s=10_000, idle_s=600)
    assert out.why == "" and killed == []


def _runner_with_progress(tmp_path, moving=False, run_for=100_000):
    from tests.test_night_runner import _runner

    box = {"n": 0}

    def progress():
        box["n"] += 1 if moving else 0
        return str(box["n"])

    return _runner(tmp_path, run_for=run_for, touch_events_every=10, progress=progress, progress_every_s=60)


def _go(run, killed, **kw):
    from tests.test_night_runner import _go as go

    return go(run, killed, **kw)


def test_work_token_moves_only_when_the_work_moves(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    gitbox.ensure_repo(tmp_path)
    t0 = gitbox.work_token(tmp_path)
    assert gitbox.work_token(tmp_path) == t0
    (tmp_path / "a.txt").write_text("2")
    t1 = gitbox.work_token(tmp_path)
    (tmp_path / "new.txt").write_text("x")
    assert t1 != t0 and gitbox.work_token(tmp_path) != t1
    (tmp_path / ".night").mkdir()
    (tmp_path / ".night" / "state.json").write_text("{}")
    assert gitbox.work_token(tmp_path) == gitbox.work_token(tmp_path)


def test_idle_and_tests_hang_are_classified_and_have_their_own_lessons(tmp_path):
    from agent.night.lessons import TEMPLATES

    d = handoff.DiffStat([("a.py", 1, 0)])
    assert analyzer.classify(Outcome(-1, "idle"), None, TestState(ok=True), d, handoff.EventSummary()).kind == "idle"
    assert analyzer.classify(Outcome(0), TestState(ok=True), TestState(ok=False, timed_out=True), d, handoff.EventSummary()).kind == "tests_hang"
    assert analyzer.next_strategy("idle", []) == "small" and analyzer.next_strategy("tests_hang", []) == "resume" and {"idle", "tests_hang"} <= set(TEMPLATES)
    h = handoff.build(tmp_path, "HEAD", None, TestState(ok=False, timed_out=True, digest="x"), handoff.EventSummary(), 0) if (tmp_path / ".git").exists() else None
    assert h is None or "DID NOT FINISH IN TIME" in h.text


def test_test_timeout_comes_from_the_probe(tmp_path):
    from agent.night import probe

    assert probe.test_timeout(Facts("t", 54.0, TestState(ok=True))) == 216.0
    assert probe.test_timeout(Facts("t", 2.0, TestState(ok=True))) == 120.0 and probe.test_timeout(Facts("t", 900.0, TestState(ok=True))) == 1800.0
    assert probe.test_timeout(Facts("", 0.0, None)) == 1800.0


def test_cli_test_command_timeout_marks_the_state_as_hanging(monkeypatch, tmp_path):
    import subprocess as sp

    from agent.codeagent import profile as profile_mod
    from agent.night import cli as C

    class Prof:
        test_cmd, build_cmd, lint_cmd, kinds = "sleep 9", None, None, []

    monkeypatch.setattr(profile_mod, "detect", lambda root, *a, **k: Prof())
    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(sp.TimeoutExpired("sleep 9", 1)))
    state = C._tests(tmp_path, 42)()
    assert state.timed_out and not state.ok and "42 s" in state.digest


# ----------------------------------------------------------------------------- mass deletion / bloat / vacuous tests
def _src_repo(tmp_path, lines=60):
    (tmp_path / "core.py").write_text("".join(f"x{i} = {i}\n" for i in range(lines)))
    (tmp_path / "other.py").write_text("y = 1\n")
    gitbox.ensure_repo(tmp_path)
    return gitbox.last_green(tmp_path)


def test_guard_reverts_deleted_and_gutted_source_unless_the_prompt_asks_for_it(tmp_path):
    from agent.night import guard

    tag = _src_repo(tmp_path)
    (tmp_path / "other.py").unlink()
    (tmp_path / "core.py").write_text("x0 = 0\n")                                           # 60 -> 1 line
    rep = guard.check(tmp_path, tag)
    assert {(f.kind, f.path) for f in rep.fatal} == {("deleted_src", "other.py"), ("deleted_src", "core.py")}
    assert sorted(guard.revert(tmp_path, tag, rep)) == ["core.py", "other.py"] and (tmp_path / "other.py").exists() and (tmp_path / "core.py").read_text().count("\n") == 60
    (tmp_path / "other.py").unlink()
    assert not guard.check(tmp_path, tag, allow_removal=True).fatal                           # "remove the old module" is a legitimate prompt
    assert guard.REMOVAL_WORDS.search("Refactor the dialogue code and remove the dead branch") and not guard.REMOVAL_WORDS.search("Add a character and a test")


def test_guard_small_edits_and_small_files_are_never_flagged(tmp_path):
    from agent.night import guard

    tag = _src_repo(tmp_path, lines=60)
    (tmp_path / "core.py").write_text("".join(f"x{i} = {i}\n" for i in range(45)))           # 60 -> 45: a normal cleanup
    (tmp_path / "other.py").write_text("")                                                   # a 1-line file emptied: below the shrink minimum
    assert not guard.check(tmp_path, tag).fatal


def test_guard_removes_new_huge_files_and_flags_vacuous_tests(tmp_path):
    from agent.night import guard

    tag = _src_repo(tmp_path)
    (tmp_path / "dump.bin").write_bytes(b"0" * (guard.BLOAT_BYTES + 10))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_nothing():\n    x = 1\n\ndef test_real():\n    assert 1\n\nclass T:\n    def test_m(self):\n        self.assertEqual(1, 1)\n\ndef test_raises():\n    import pytest\n    with pytest.raises(ValueError):\n        int('x')\n")
    rep = guard.check(tmp_path, tag)
    assert {(f.kind, f.path) for f in rep.findings} >= {("bloat", "dump.bin"), ("vacuous", "tests/test_x.py")}
    assert [f.detail for f in rep.findings if f.kind == "vacuous"] == ["1 new test(s) without any assertion"]
    assert guard.revert(tmp_path, tag, rep) == ["dump.bin"] and not (tmp_path / "dump.bin").exists() and (tmp_path / "tests" / "test_x.py").exists()


# ----------------------------------------------------------------------------- canary / unit / history
def test_canary_passes_only_when_the_whole_chain_wrote_the_file_and_leaves_nothing(tmp_path):
    from agent.night import canary

    (tmp_path / "a.txt").write_text("base")
    ok = lambda a: ((tmp_path / canary.FILE).write_text("ok\n"), Outcome(0))[1]
    assert canary.run(tmp_path, ok) == (True, "the whole chain works (0 s)") and not (tmp_path / canary.FILE).exists()
    never = lambda a: Outcome(0)
    good, detail = canary.run(tmp_path, never)
    assert not good and "did not write night_canary.txt" in detail
    timeout = lambda a: Outcome(-1, "timeout")
    assert "ended with timeout" in canary.run(tmp_path, timeout)[1]
    wrong = lambda a: ((tmp_path / canary.FILE).write_text("nope"), Outcome(0))[1]
    assert "wrong content" in canary.run(tmp_path, wrong)[1] and not (tmp_path / canary.FILE).exists()


def test_unit_file_is_rendered_not_enabled(tmp_path):
    from agent.night import unit

    root = tmp_path / "My Game!"
    root.mkdir()
    text = unit.render(root, ["/p/prompts dir", "inline text"], "/usr/bin/nightage", ["--until", "07:30"])
    assert unit.name_for(root) == "nightage-night-my-game" and "ConditionPathExists=!" + str(root) + "/.night/DONE" in text
    assert "ExecStart=/usr/bin/nightage night run '/p/prompts dir' 'inline text' --folder" in text and "--resume --yes --no-model --until 07:30" in text and "Restart=on-failure" in text
    assert unit.unit_path(root, tmp_path).parent == tmp_path / "systemd" / "user" and "enable-linger" in unit.instructions(root)


def test_cli_install_unit_prints_by_default_and_writes_with_flag(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    from agent.night import unit

    monkeypatch.setattr(unit, "unit_path", lambda root, ch=None: tmp_path / "unit.service")
    shown = runner.invoke(C.night_app, ["install-unit", "do the thing", "--folder", str(proj)])
    assert shown.exit_code == 0 and "[Service]" in shown.output and "not written" in shown.output and not (tmp_path / "unit.service").exists()
    wrote = runner.invoke(C.night_app, ["install-unit", "do the thing", "--folder", str(proj), "--write"])
    assert (tmp_path / "unit.service").exists() and "systemctl --user enable --now" in wrote.output


def test_history_records_nights_and_turns_them_into_capped_priors(tmp_path):
    from agent.night import history

    f = tmp_path / "h.jsonl"
    attempts = [{"trigger": "no_edits", "strategy": "small", "green": True, "progress": True}] * 10 + [{"trigger": "no_edits", "strategy": "fresh", "green": False, "progress": False}] * 2
    history.record(attempts, project="game", done=["a"], partial=[], hours=7.5, tests_ok=True, where=f)
    history.record(attempts[:3], project="game", done=[], partial=["b"], hours=1.0, tests_ok=False, where=f)
    f.write_text(f.read_text() + "garbage\n")
    assert len(history.rows(f)) == 2 and history.rows(tmp_path / "none") == []
    pri = history.priors("no_edits", f)
    assert pri["small"] == (history.PRIOR_WEIGHT, 1.0) and pri["fresh"][0] == 2 and pri["fresh"][1] == 0.0 and history.priors("loop", f) == {}
    merged = history.merge({"small": (1, 0.0)}, {"small": (4, 1.0), "fresh": (2, 0.5)})
    assert merged["small"] == (5, pytest.approx(0.8)) and merged["fresh"] == (2, 0.5)


def test_cli_history_lists_nights(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    from agent.night import history

    monkeypatch.setenv("NIGHT_HISTORY", str(tmp_path / "h.jsonl"))
    assert "no nights recorded" in runner.invoke(C.night_app, ["nights"]).output
    history.record([], project="game", done=["a", "b"], partial=["c"], hours=6.25, tests_ok=True)
    out = runner.invoke(C.night_app, ["nights"]).output
    assert "game" in out and "done 2" in out and "open 1" in out and "6.25 h" in out and "green" in out


def test_the_governor_never_reads_or_writes_the_real_history_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHT_HISTORY", str(tmp_path / "must-not-exist.jsonl"))
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 100))[1], green, hours=1)
    n.run()
    assert not (tmp_path / "must-not-exist.jsonl").exists()
    n2, _ = _night(tmp_path / "second", [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "second" / "f").write_text("x"), (Outcome(0), 100))[1], green, hours=1, use_history=True) if (tmp_path / "second").mkdir() is None else (None, None)
    n2.run()
    assert (tmp_path / "must-not-exist.jsonl").exists()


# ----------------------------------------------------------------------------- speed drift / health restart / triage
def test_health_restart_restarts_a_server_that_answers_and_is_rate_limited(tmp_path):
    from agent.night import health

    calls, clock = [], {"t": 10_000.0}
    h = health.Health(tmp_path, urls=["http://x"], restart_cmds=[["restart-it"]], clock=lambda: clock["t"], probe=lambda u: True, run=lambda cmd, **k: calls.append(cmd), sleep=lambda s: None)
    assert h.restart() is True and calls == [["restart-it"]] and h.restarts == 1
    assert h.restart() is False and len(calls) == 1                                           # at most every 30 minutes
    clock["t"] += 3600
    assert h.restart() is True and len(calls) == 2
    none = health.Health(tmp_path, restart_cmds=[], probe=lambda u: True)
    assert none.restart() is False


def test_the_governor_notices_a_model_that_got_slow_and_restarts_the_server_once(tmp_path):
    class Gate:
        restarted = 0

        def gate(self):
            return None

        def restart(self):
            Gate.restarted += 1
            return True

    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: (Outcome(0), 10), green, hours=1, health=Gate())
    n.hogs = lambda: [(1, "firefox", 2500)]
    ev = lambda tps: handoff.EventSummary(steps=20, gen_tokens=int(tps * 100), decode_s=100.0)
    item = n.items[0]
    for tps in (35, 34, 33):
        n._check_drift(item, ev(tps))
    assert Gate.restarted == 0 and not any("drifted" in e for e in n.events)
    n._check_drift(item, ev(12))
    n._check_drift(item, ev(11))
    assert Gate.restarted == 2 or Gate.restarted >= 1
    drift = [e for e in n.events if "speed drifted" in e]
    assert drift and "firefox 2500 MB" in drift[0] and "the model server was restarted" in drift[0] and "35 at the start" in drift[0] or "34 at the start" in drift[0]
    short = handoff.EventSummary(steps=3, gen_tokens=10, decode_s=1.0)
    before = len(n.tps_hist)
    n._check_drift(item, short)
    assert len(n.tps_hist) == before                                                           # short attempts say nothing about speed


def test_the_morning_report_lists_what_to_review_first_with_the_measured_reasons(tmp_path):
    def script(a, i):
        (tmp_path / f"f{i}.txt").write_text("x")
        return Outcome(0 if "HARD" not in a.task else 3), 1200

    items = [queue.Item("easy", "EASY thing, it must work."), queue.Item("hard", "HARD thing, it must work.")]
    n, seen = _night(tmp_path, items, script, lambda: red("t") if (tmp_path / "f1.txt").exists() and not (tmp_path / "f9.txt").exists() else green(), hours=3)
    rep = n.run()
    assert "## Look at these first" in rep.text
    block = rep.text.split("## Look at these first")[1].split("##")[0]
    assert "- hard:" in block and "NOT finished" in block and "git diff" in block and "- easy:" not in block
    quiet, _ = _night(tmp_path / "q", [queue.Item("only", "ONE thing, it must work.")], lambda a, i: ((tmp_path / "q" / "f").write_text("x"), (Outcome(0), 60))[1], green, hours=1) if (tmp_path / "q").mkdir() is None else (None, None)
    assert "Look at these first" not in quiet.run().text


def red(*names):
    return TestState(ok=False, passed=1, failed=list(names), digest="red", tail="red")


# ----------------------------------------------------------------------------- event slices / selftest / polish
def test_slice_events_saves_one_attempt_for_the_morning_analysis(tmp_path):
    rows = [{"ts": 5, "session": "proj-1", "kind": "step"}, {"ts": 20, "session": "proj-1", "kind": "step"}, {"ts": 21, "session": "proj-1", "kind": "tool", "name": "edit_file"},
            {"ts": 22, "session": "other-1", "kind": "step"}, {"ts": 99, "session": "proj-1", "kind": "step"}]
    src = _events(tmp_path, rows)
    dest = tmp_path / ".night" / "events" / "p1-attempt1.jsonl"
    assert handoff.slice_events(src, 10, "proj", dest, until=50) == 2
    assert [json.loads(ln)["kind"] for ln in dest.read_text().splitlines()] == ["step", "tool"]
    assert handoff.slice_events(src, 1000, "proj", tmp_path / "x.jsonl") == 0 and not (tmp_path / "x.jsonl").exists()
    assert handoff.slice_events(tmp_path / "missing", 0, "p", tmp_path / "y.jsonl") == 0


def test_the_governor_writes_the_event_slice_of_every_attempt(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text(json.dumps({"ts": 1000, "session": tmp_path.name + "-1", "kind": "step", "gen": 5, "decode_s": 1.0, "prefill_s": 0.1, "prompt": 100}) + "\n")
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 100))[1], green, hours=1, events=ev, wall=lambda: 900.0)
    n.run()
    assert (tmp_path / ".night" / "events" / "p1-attempt1.jsonl").exists()


def test_selftest_builds_a_toy_project_and_verifies_a_finished_night(tmp_path):
    from agent.night import selftest

    root = tmp_path / "toy"
    root.mkdir()
    prompts = selftest.build(root)
    assert sorted(p.name for p in prompts.iterdir()) == ["01-subtract.md", "02-divide.md"] and (root / "tests" / "test_calc.py").exists()
    items, _ = queue.load([str(prompts)])
    assert [i.name for i in items] == ["subtract", "divide"] and items[1].depends == ["subtract"]
    cmd = selftest.command(root, prompts, 0.5)
    assert cmd[3:5] == ["night", "run"] and "--no-model" in cmd and cmd[cmd.index("--hours") + 1] == "0.5"
    before = {c.name: c.ok for c in selftest.verify(root)}
    assert not before["sub exists"] and not before["both prompts done"] and "FAILED" in selftest.text(selftest.verify(root))
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\n\ndef div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n")
    gitbox.ensure_repo(root)
    gitbox.checkpoint(root, "done")
    (root / ".night").mkdir(exist_ok=True)
    (root / ".night" / "NIGHT_REPORT.md").write_text("| subtract | done | 1 |\n| divide | done | 1 |\n")
    checks = selftest.verify(root)
    assert all(c.ok for c in checks), selftest.text(checks)
    assert selftest.text(checks).endswith("selftest PASSED")


def test_cli_selftest_reports_by_exit_code(cli, monkeypatch):
    runner, C, proj = cli
    from agent.night import selftest

    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(selftest, "verify", lambda root: [selftest.Check("x", True)])
    assert runner.invoke(C.night_app, ["selftest"]).exit_code == 0
    monkeypatch.setattr(selftest, "verify", lambda root: [selftest.Check("x", False, "nope")])
    res = runner.invoke(C.night_app, ["selftest"])
    assert res.exit_code == 1 and "selftest FAILED" in res.output


GOOD = "# Task\n\nStage 1: Add `feature.py` with a function that returns 2.\nCheck: a test in `tests/test_feature.py` must pass.\n\nStage 2: Document it.\nCheck: the README mentions the function; the whole suite must pass.\n"


def test_polish_offers_a_version_only_when_the_prompt_check_likes_it_better(tmp_path):
    from agent.night import polish

    vague = "make the game nicer and let me know what you think"
    better, before, after = polish.polish(vague, tmp_path, lambda q: GOOD)
    assert better == GOOD.strip() and len(after) < len(before) and "TASK:\nmake the game nicer" in polish.META.format(task=vague)
    assert polish.polish(vague, tmp_path, lambda q: "")[0] is None                                  # empty reply
    assert polish.polish(vague, tmp_path, lambda q: "ok")[0] is None                                # far shorter than the original
    assert polish.polish(vague, tmp_path, lambda q: (_ for _ in ()).throw(RuntimeError("no model")))[0] is None
    same = "make it nicer, let me know " * 4
    assert polish.polish(same, tmp_path, lambda q: same + " more words " * 3)[0] is None            # not better by the check
    fenced = polish.polish(vague, tmp_path, lambda q: "```markdown\n" + GOOD + "```")[0]
    assert fenced == GOOD.strip()
    assert polish.target(tmp_path / "prompt1.md").name == "prompt1.polished.md"


def test_cli_polish_never_touches_the_original(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    src = tmp_path / "vague.md"
    src.write_text("make the game nicer and let me know what you think")
    monkeypatch.setattr(C, "_ask_model", lambda: (lambda q: GOOD))
    res = runner.invoke(C.night_app, ["polish", str(src), "--folder", str(proj)])
    assert res.exit_code == 0 and src.read_text().startswith("make the game nicer") and (tmp_path / "vague.polished.md").read_text().strip() == GOOD.strip()
    assert "the original is untouched" in res.output
    monkeypatch.setattr(C, "_ask_model", lambda: (lambda q: ""))
    assert runner.invoke(C.night_app, ["polish", str(src), "--folder", str(proj)]).exit_code == 1


def test_selftest_borrows_pytest_and_log_lines_keep_their_brackets(cli, monkeypatch):
    from agent.night import selftest

    monkeypatch.setattr(selftest.shutil, "which", lambda name: "/bin/uv")
    assert selftest.test_command()[:5] == ["uv", "run", "--no-project", "--with", "pytest"]
    monkeypatch.setattr(selftest.shutil, "which", lambda name: None)
    assert selftest.test_command()[1:3] == ["-m", "pytest"]
    runner, C, proj = cli
    import io

    from rich.console import Console

    buf = io.StringIO()
    monkeypatch.setattr(C, "console", Console(file=buf, force_terminal=False, width=200))
    C._say("10:53:52  [subtract] attempt 1 (start) green")
    assert "[subtract] attempt 1" in buf.getvalue()


def test_a_big_change_that_broke_a_few_tests_is_repaired_not_rolled_back():
    big, small = handoff.DiffStat([("wren.py", 600, 0)]), handoff.DiffStat([("a.py", 3, 0)])
    base = TestState(ok=True, passed=300)
    two = TestState(ok=False, passed=298, failed=["t1", "t2"])
    many = TestState(ok=False, passed=290, failed=[f"t{i}" for i in range(9)])
    assert analyzer.classify(Outcome(3), base, two, big, handoff.EventSummary()).kind == "almost"
    assert analyzer.classify(Outcome(3), base, two, small, handoff.EventSummary()).kind == "broke_tests"        # a tiny change that breaks tests is not "nearly done"
    assert analyzer.classify(Outcome(3), base, many, big, handoff.EventSummary()).kind == "broke_tests"
    assert analyzer.next_strategy("almost", []) == "repair" and analyzer.next_strategy("almost", ["almost"]) == "fresh" and analyzer.next_strategy("almost", ["almost"] * 3) == "rollback"


def test_cli_run_indexes_the_night_and_a_crash_leaves_a_saved_state_and_a_message(cli, monkeypatch, tmp_path):
    runner, C, proj = cli
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    from agent import resume

    def boom(self):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(C.Night, "run", boom)
    res = runner.invoke(C.night_app, ["run", "Add feature() returning 2 with a test; it must pass " * 2, "--folder", str(proj), "--yes", "--no-model", "--hours", "1", "--ignore-games", "--no-canary"])
    assert res.exit_code == 70 and "disk exploded" in res.output and "nightage continue" in res.output and "Traceback" not in res.output
    assert (proj / ".night" / "state.json").exists() and not (proj / ".night" / "LOCK").exists()
    idx = resume._index()
    assert list(idx) == [str(proj)] and idx[str(proj)]["options"]["review"] in (True, False)
    from typer.testing import CliRunner
    from agent.main import app

    assert "continue" in CliRunner().invoke(app, ["--help"]).output


def test_canary_allowance_grows_with_the_test_suite():
    from agent.night import canary

    assert canary.timeout_for(0.0) == 480.0 and canary.timeout_for(54.0) == 480.0 and canary.timeout_for(120.0) == 960.0 and canary.timeout_for(900.0) == 1800.0


def test_a_killed_night_leaves_a_visible_charter_that_restore_policy_removes(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    from agent.codeagent import policy
    from agent.night import charter

    root = proj
    (root / ".night").mkdir(exist_ok=True)
    policy.save_repo(root, mode="medium")
    charter.apply(root, charter.Charter(allow=["make *"], dangerous=["rm -r build*"]))                   # as a night start does, then the process dies hard
    (root / ".night" / "state.json").write_text(json.dumps({"left_s": 3600, "items": [], "events": []}))
    assert policy.for_repo(root).mode == "auto"
    assert "restore-policy" in runner.invoke(C.night_app, ["status", "--folder", str(root)]).output
    assert runner.invoke(C.night_app, ["restore-policy", "--folder", str(root)]).exit_code == 0 and policy.for_repo(root).mode == "medium"
    assert "restore-policy" not in runner.invoke(C.night_app, ["status", "--folder", str(root)]).output
    assert "nothing to restore" in runner.invoke(C.night_app, ["restore-policy", "--folder", str(root)]).output
    (root / ".night" / "LOCK").write_text("1")
    assert runner.invoke(C.night_app, ["restore-policy", "--folder", str(root)]).exit_code == 1


def test_sigterm_ends_a_night_like_ctrl_c(cli, monkeypatch):
    import signal

    runner, C, proj = cli
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda sig, h: handlers.__setitem__(sig, h))

    real = C.Night

    class Dies(real):
        def run(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)                                                # the process gets a SIGTERM while it works

    monkeypatch.setattr(C, "Night", Dies)
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary"])
    assert res.exit_code == 130 and "continue with" in res.output.lower() and signal.SIGHUP in handlers
    assert not (proj / ".night" / "LOCK").exists() and not (proj / ".night" / "policy-before.json").exists()       # lock released, policy restored


# ----------------------------------------------------------------------------- unattended nights: unit, wall-clock resume, orphan attempts
def test_unit_does_not_restart_a_finished_or_refused_night_but_restarts_a_crash(tmp_path):
    from agent.night import unit

    text = unit.render(tmp_path, ["p"], "/home/user/.local/bin/nightage", path_env="/usr/bin:/home/user/.nix-profile/bin")
    assert "SuccessExitStatus=3" in text and "RestartPreventExitStatus=1 130" in text and "Restart=on-failure" in text
    assert "Environment=PATH=/usr/bin:/home/user/.nix-profile/bin" in text and "ExecStart=/home/user/.local/bin/nightage night run p" in text
    assert "Environment=" not in unit.render(tmp_path, ["p"], "nightage")


def test_a_resumed_night_counts_its_time_from_the_wall_clock_and_ends_when_it_is_over(cli, monkeypatch):
    import time as _time

    runner, C, proj = cli
    nd = proj / ".night"
    nd.mkdir(exist_ok=True)
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    state = {"left_s": 7200, "end_wall": _time.time() - 60, "items": [{"name": "inline-1", "status": "partial", "attempts": 1, "seconds": 5.0, "note": "", "kinds": []}], "events": []}
    (nd / "state.json").write_text(json.dumps(state))
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary", "--resume"])
    assert res.exit_code == 3 and "time of this night is over" in res.output
    seen = {}
    real = C.Night

    class Spy(real):
        def __init__(self, root, items, rules, hours, **kw):
            seen["hours"] = hours
            super().__init__(root, items, rules, hours, **kw)

    monkeypatch.setattr(C, "Night", Spy)
    state["end_wall"] = _time.time() + 3600 * 2.5                                    # the saved left_s (2 h) is stale: the wall clock says 2.5 h
    (nd / "state.json").write_text(json.dumps(state))
    runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary", "--resume"])
    assert 2.4 < seen["hours"] < 2.6


def test_a_stale_done_marker_does_not_block_a_new_night(cli):
    runner, C, proj = cli
    (proj / ".night").mkdir(exist_ok=True)
    (proj / ".night" / "DONE").write_text("an earlier night\n")
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary"])
    assert res.exit_code == 0 and "earlier night" not in (proj / ".night" / "DONE").read_text()


def test_the_runner_reports_the_attempt_pid_and_the_governor_keeps_it_in_a_file(tmp_path):
    seen = {}

    def runner(a):
        a.on_start(4242)
        seen["file"] = (tmp_path / ".night" / "attempt.pid").read_text()
        return Outcome(0)

    n, _ = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: (Outcome(0), 60), green, hours=1)
    n.runner = runner
    n.run()
    assert seen["file"] == "4242" and not (tmp_path / ".night" / "attempt.pid").exists()                # written while it runs, removed after


def test_an_attempt_left_over_from_a_killed_governor_is_stopped_when_the_night_is_resumed(tmp_path):
    import subprocess
    import sys
    import time

    stray = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (tmp_path / ".night").mkdir()
        (tmp_path / ".night" / "attempt.pid").write_text(str(stray.pid))
        n, _ = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 60))[1], green, hours=1)
        n.run()
        stray.wait(timeout=10)
        assert stray.returncode is not None and any("left over from an earlier run" in e for e in n.events) and not (tmp_path / ".night" / "attempt.pid").exists()
        (tmp_path / ".night" / "attempt.pid").write_text("999999999")                    # a pid that is gone: nothing to stop, no crash
        n2, _ = _night(tmp_path / "x", [queue.Item("p1", "T")], lambda a, i: (Outcome(0), 60), green, hours=1) if (tmp_path / "x").mkdir() is None else (None, None)
        n2._kill_orphan()
    finally:
        stray.kill()


def test_a_crash_of_night_run_gets_its_own_exit_code_and_leaves_the_folder_in_order(cli, monkeypatch):
    runner, C, proj = cli
    real = C.Night

    class Broken(real):
        def run(self):
            raise RuntimeError("a bug in the governor")

    monkeypatch.setattr(C, "Night", Broken)
    prompt = "Add a function `feature()` in `feature.py` that returns 2, and a test; it must pass."
    res = runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--ignore-games", "--no-canary"])
    assert res.exit_code == C.CRASH_EXIT == 70
    assert "a bug in the governor" in (proj / ".night" / "errors.log").read_text() and not (proj / ".night" / "LOCK").exists()
    assert "is saved" in res.output and "nightage continue" in res.output
    # a refusal stays a plain 1 (the unit must not restart it)
    (proj / ".night" / "LOCK").write_text("1")
    assert runner.invoke(C.night_app, ["run", prompt, "--folder", str(proj), "--yes", "--no-model", "--no-canary"]).exit_code == 1


def test_unit_limits_crash_loops_and_is_unattended_by_default(tmp_path):
    from agent.night import unit

    text = unit.render(tmp_path, ["p"], "nightage", extra_args=["--until", "08:00"])
    assert "StartLimitIntervalSec=3600" in text and "StartLimitBurst=10" in text and "--resume --yes --no-model --until 08:00" in text


def test_install_unit_passes_extra_run_options(cli, tmp_path, monkeypatch):
    runner, C, proj = cli
    from agent.night import unit

    monkeypatch.setattr(unit, "unit_path", lambda root, ch=None: tmp_path / "u.service")
    res = runner.invoke(C.night_app, ["install-unit", "do the thing", "--folder", str(proj), "--extra", "--start-at 01:00 --until 08:00 --tournament 2"])
    assert res.exit_code == 0 and "--start-at 01:00 --until 08:00 --tournament 2" in res.output and "--no-model" in res.output
