"""agent/ollama_local.py: start/stop/ensure decisions, exit cleanup with signals, pull progress, size/fit. Fake `ollama` program and fake server only."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agent import ollama_local as O

ROOT = Path(__file__).resolve().parent.parent
FAKE = Path(__file__).resolve().parent / "_fake_ollama.py"
posix = pytest.mark.posix_only


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def requests_of(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake machine: `ollama` on PATH is the fake, state under tmp, the server listens on a free port."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "ollama"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    exe.chmod(0o755)
    port = free_port()
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("FAKE_OLLAMA_LOG", str(log))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("NIGHTAGE_NO_AUTOSTART", raising=False)
    monkeypatch.delenv("NIGHTAGE_OLLAMA_SESSION", raising=False)
    cfg = {"ollama": {"host": f"127.0.0.1:{port}", "autostart": True, "stop_on_exit": True, "unload_on_exit": True}}
    yield type("Env", (), {"cfg": cfg, "port": port, "log": log, "tmp": tmp_path, "url": f"http://127.0.0.1:{port}"})()
    rec = O.read_record()
    if rec:
        O._kill_group(rec["pid"], signal.SIGKILL)


@posix
def test_start_records_pid_and_stop_only_when_ours(env):
    assert O.status(env.cfg).reachable is False
    res = O.start(env.cfg)
    assert res.ok and res.action == "started" and res.pid > 0
    st = O.status(env.cfg)
    assert st.reachable and st.ours and st.pid == res.pid and st.local
    assert json.loads(O.pid_path().read_text())["started_by"] == "nightage"
    assert O.start(env.cfg).action == "running"
    out = O.stop(env.cfg)
    assert out.ok and out.action == "stopped"
    assert not O.pid_path().exists() and O.probe(env.url) is None


@posix
def test_stop_refuses_a_server_it_did_not_start(env):
    proc = subprocess.Popen([sys.executable, str(FAKE), "serve"], env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{env.port}"})
    try:
        for _ in range(50):
            if O.probe(env.url):
                break
            time.sleep(0.1)
        st = O.status(env.cfg)
        assert st.reachable and not st.ours
        res = O.stop(env.cfg)
        assert not res.ok and res.action == "not-ours" and O.probe(env.url)
    finally:
        proc.kill()
        proc.wait()


@posix
def test_ensure_decisions(env, monkeypatch):
    remote = {"ollama": {"host": "100.64.0.5:11434", "autostart": True}}
    assert O.ensure(remote).action == "remote"                       # never starts for another machine
    off = {"ollama": {**env.cfg["ollama"], "autostart": False}}
    assert O.ensure(off).action == "disabled" and O.probe(env.url) is None
    monkeypatch.setenv("NIGHTAGE_NO_AUTOSTART", "1")
    assert O.ensure(env.cfg).action == "disabled"
    monkeypatch.delenv("NIGHTAGE_NO_AUTOSTART")
    monkeypatch.setenv("PATH", "/nonexistent")
    assert O.ensure(env.cfg).action == "no-binary"
    monkeypatch.undo()


@posix
def test_ensure_starts_then_reports_running(env):
    assert O.ensure(env.cfg).action == "started"
    assert O.ensure(env.cfg).action == "running"
    O.stop(env.cfg)


@posix
def test_stale_pid_file_is_cleaned(env):
    O.state_dir().mkdir(parents=True)
    O.pid_path().write_text(json.dumps({"pid": 999999, "started_by": "nightage"}))
    assert O.read_record() is None and not O.pid_path().exists()


@posix
def test_remote_url_is_never_started(env):
    res = O.start({"ollama": {"host": "192.0.2.10"}})
    assert res.action == "remote" and not O.pid_path().exists()


def test_cleanup_unloads_every_loaded_model(env, monkeypatch):
    monkeypatch.setenv("FAKE_OLLAMA_LOADED", "alpha:1b,beta:2b")
    res = O.start(env.cfg)
    assert res.ok
    O.begin(env.cfg)
    out = O.cleanup("test")
    assert sorted(out.unloaded) == ["alpha:1b", "beta:2b"] and out.stopped
    gens = [r for r in requests_of(env.log) if r.get("path") == "/api/generate"]
    assert {g["body"]["model"] for g in gens} == {"alpha:1b", "beta:2b"} and all(g["body"]["keep_alive"] == 0 for g in gens)
    assert O.cleanup("again").skipped == "already done"             # idempotent
    assert O.probe(env.url) is None


def test_cleanup_leaves_a_foreign_server_running_but_still_unloads(env, monkeypatch):
    monkeypatch.setenv("FAKE_OLLAMA_LOADED", "alpha:1b")
    proc = subprocess.Popen([sys.executable, str(FAKE), "serve"], env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{env.port}"})
    try:
        for _ in range(50):
            if O.probe(env.url):
                break
            time.sleep(0.1)
        O.begin(env.cfg)
        out = O.cleanup("test")
        assert out.unloaded == ["alpha:1b"] and not out.stopped and O.probe(env.url)
    finally:
        proc.kill()
        proc.wait()


def test_cleanup_keeps_server_while_another_nightage_uses_it(env):
    assert O.start(env.cfg).ok
    O.begin(env.cfg)
    other = subprocess.Popen(["sleep", "30"])
    try:
        (O.clients_dir() / str(other.pid)).write_text("")
        out = O.cleanup("test")
        assert not out.stopped and "another" in out.skipped and O.probe(env.url)
    finally:
        other.kill()
        other.wait()
        O.stop(env.cfg)


def test_cleanup_without_session_is_a_noop():
    O._session = None
    assert O.cleanup().skipped == "already done"


def test_unload_off_and_stop_off_are_respected(env, monkeypatch):
    monkeypatch.setenv("FAKE_OLLAMA_LOADED", "alpha:1b")
    cfg = {"ollama": {**env.cfg["ollama"], "unload_on_exit": False, "stop_on_exit": False}}
    assert O.start(cfg).ok
    O.begin(cfg)
    out = O.cleanup("t")
    assert out.unloaded == [] and not out.stopped and O.probe(env.url)
    O.stop(cfg)


def test_keep_model_loaded_skips_unload_and_stop(env, monkeypatch):
    monkeypatch.setenv("FAKE_OLLAMA_LOADED", "alpha:1b")
    cfg = {**env.cfg, "coding": {"keep_model_loaded": True}}
    assert O.start(cfg).ok
    O.begin(cfg)
    out = O.cleanup("t")
    assert out.unloaded == [] and not out.stopped
    O.stop(cfg)


# ── signals: a real subprocess runs the same session() the entry points use ─────────────────

RUNNER = textwrap.dedent("""
    import sys, time, json
    sys.path.insert(0, {root!r})
    from agent import ollama_local as O
    cfg = json.loads(sys.argv[1])
    with O.session(cfg):
        print("READY", flush=True)
        time.sleep(60)
""")


def _run_runner(env, tmp_path, monkeypatch, loaded="alpha:1b,beta:2b"):
    script = tmp_path / "runner.py"
    script.write_text(RUNNER.format(root=str(ROOT)))
    penv = {**os.environ, "FAKE_OLLAMA_LOADED": loaded}
    proc = subprocess.Popen([sys.executable, str(script), json.dumps(env.cfg)], stdout=subprocess.PIPE, text=True, env=penv)
    assert proc.stdout.readline().strip() == "READY"
    assert O.probe(env.url) is not None                              # autostart started the fake server
    return proc


@posix
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_unloads_models_and_stops_the_server(env, tmp_path, monkeypatch, sig):
    proc = _run_runner(env, tmp_path, monkeypatch)
    proc.send_signal(sig)
    proc.wait(timeout=15)
    gens = [r["body"] for r in requests_of(env.log) if r.get("path") == "/api/generate"]
    assert {g["model"] for g in gens} == {"alpha:1b", "beta:2b"} and all(g["keep_alive"] == 0 for g in gens)
    assert O.probe(env.url) is None and not O.pid_path().exists()
    if sig != signal.SIGINT:
        assert proc.returncode == -sig                              # it still dies the way the signal wanted
    else:
        assert proc.returncode != 0                                  # KeyboardInterrupt traceback exit


@posix
def test_normal_exit_runs_cleanup_through_atexit(env, tmp_path):
    script = tmp_path / "runner2.py"
    script.write_text(textwrap.dedent(f"""
        import sys, json
        sys.path.insert(0, {str(ROOT)!r})
        from agent import ollama_local as O
        O.begin(json.loads(sys.argv[1]))         # no try/finally on purpose: only atexit is left
    """))
    subprocess.run([sys.executable, str(script), json.dumps(env.cfg)], check=True, env={**os.environ, "FAKE_OLLAMA_LOADED": "alpha:1b"})
    assert any(r.get("path") == "/api/generate" for r in requests_of(env.log))
    assert O.probe(env.url) is None


@posix
def test_leftover_server_after_sigkill_is_found_and_reused_then_stopped(env, tmp_path):
    proc = _run_runner(env, tmp_path, None, loaded="")
    proc.kill()                                                      # SIGKILL: nothing could clean up
    proc.wait()
    assert O.probe(env.url) is not None and O.status(env.cfg).ours   # the pid file still names the leftover
    with O.session(env.cfg) as res:
        assert res.action == "running"
    assert O.probe(env.url) is None                                  # this run adopted it and stopped it on exit


@posix
def test_second_signal_during_cleanup_forces_a_kill(env, tmp_path):
    """A server that ignores SIGTERM and a slow unload: the second SIGTERM kills it hard and exits at once."""
    script = tmp_path / "runner3.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time, json, signal
        sys.path.insert(0, {str(ROOT)!r})
        from agent import ollama_local as O
        cfg = json.loads(sys.argv[1])
        with O.session(cfg):
            print("READY", flush=True)
            time.sleep(60)
    """))
    penv = {**os.environ, "FAKE_OLLAMA_LOADED": ""}
    proc = subprocess.Popen([sys.executable, str(script), json.dumps(env.cfg)], stdout=subprocess.PIPE, text=True, env=penv)
    assert proc.stdout.readline().strip() == "READY"
    rec = O.read_record()
    os.kill(rec["pid"], 0)
    # make the server unkillable by TERM: SIGSTOP freezes it, so cleanup waits for it to answer
    O._kill_group(rec["pid"], signal.SIGSTOP)
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)                                 # second signal while the cleanup is running
    proc.wait(timeout=10)
    time.sleep(0.3)
    assert not O._alive(rec["pid"]) or open(f"/proc/{rec['pid']}/stat").read().rsplit(")", 1)[-1].split()[0] == "Z"


# ── pull ────────────────────────────────────────────────────────────────────────────────────

def test_tracker_monotonic_percent_speed_and_eta():
    t = [0.0]
    tr = O.PullTracker(clock=lambda: t[0])
    tr.feed({"status": "pulling manifest"})
    assert tr.p.pct == 0 and tr.p.total == 0
    tr.feed({"status": "pulling a", "digest": "a", "total": 1000, "completed": 0})
    t[0] = 1.0
    p = tr.feed({"status": "pulling a", "digest": "a", "total": 1000, "completed": 500})
    assert p.pct == 50 and p.speed == 500 and p.eta == pytest.approx(1.0)
    t[0] = 2.0
    p = tr.feed({"status": "pulling b", "digest": "b", "total": 500, "completed": 0})   # a new layer must not move the bar backwards
    assert p.pct == 50 and p.total == 1500
    p = tr.feed({"status": "success"})
    assert p.done and p.pct == 100


def test_pull_against_fake_server(env, monkeypatch):
    monkeypatch.setenv("FAKE_OLLAMA_PULL_BYTES", "5000")
    assert O.start(env.cfg).ok
    seen = []
    try:
        final = O.pull("fake:1b", seen.append, config=env.cfg)
    finally:
        O.stop(env.cfg)
    assert final.done and final.pct == 100 and seen[0].pct == 0
    pcts = [p.pct for p in seen]
    assert pcts == sorted(pcts) and 20 in pcts


def test_pull_error_line(env):
    assert O.start(env.cfg).ok
    try:
        with pytest.raises(O.PullError, match="does not exist"):
            O.pull("missing:1b", None, config=env.cfg)
    finally:
        O.stop(env.cfg)


def test_pull_connection_refused_message():
    with pytest.raises(O.PullError, match="Is it running"):
        O.pull("x", None, url=f"http://127.0.0.1:{free_port()}")


# ── size / fit ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tag,params", [("qwen3.5:4b", 4e9), ("qwen3.5:0.8b", 0.8e9), ("llama3:70b", 70e9), ("x:270m", 270e6),
                                         ("qwen3.5:latest", None), ("bge-m3", None)])
def test_parse_params(tag, params):
    assert O.parse_params(tag) == params


def test_estimate_sources():
    assert O.estimate_size("all-minilm", fetch=None) == (46 * 1024 * 1024, "catalogue")
    assert O.estimate_size("foo:9b", fetch=lambda m: 5_000_000_000) == (5_000_000_000, "registry")
    size, src = O.estimate_size("foo:9b", fetch=lambda m: None)
    assert src == "tag" and size == int(9e9 * O.BYTES_PER_PARAM_Q4)
    assert O.estimate_size("foo", fetch=lambda m: None) == (None, "")


def test_fit_refuses_obviously_too_big_and_accepts_small():
    big = O.check_fit("llama3:70b", size=int(42e9), ram_free_mb=6000, ram_total_mb=11000, disk_free=200 * 1024 ** 3)
    assert big.verdict == "fail"
    small = O.check_fit("qwen3.5:4b", size=int(3.4e9), ram_free_mb=6000, ram_total_mb=11000, disk_free=50 * 1024 ** 3)
    assert small.verdict == "ok"
    disk = O.check_fit("qwen3.5:4b", size=int(3.4e9), ram_free_mb=6000, ram_total_mb=11000, disk_free=1 * 1024 ** 3)
    assert disk.verdict == "fail" and "disk" in disk.text
    assert O.check_fit("mystery", size=None, ram_total_mb=1000, ram_free_mb=500).verdict in ("unknown", "ok", "warn", "fail")


@posix
def test_leftover_and_doctor_lifecycle_checks(env, tmp_path):
    from agent.doctor import core
    from tests._doctorfake import ctx_for
    from agent.doctor.probes import OllamaInfo

    core.discover()
    assert O.leftover() is None
    assert O.start(env.cfg).ok
    O.clients_dir().mkdir(parents=True, exist_ok=True)
    marker = O.clients_dir() / str(os.getppid())                     # another live Nightage window uses it: not a leftover
    marker.write_text("")
    assert O.leftover() is None
    marker.unlink()
    assert O.leftover() is not None                                  # nobody uses it any more, e.g. after a SIGKILL
    ctx = ctx_for(tmp_path, None, ollama=OllamaInfo(reachable=True, url=env.url, version="0.9"))
    res = core.run_checks(ctx, [core.REGISTRY["ollama.leftover"]])[0]
    assert res.status == "warn" and "nightage ollama stop" in res.hint
    org = core.run_checks(ctx, [core.REGISTRY["ollama.origin"], core.REGISTRY["ollama"]])
    assert any("started by Nightage" in r.detail for r in org if r.id == "ollama.origin")
    O.stop(env.cfg)
    assert O.leftover() is None


def test_home_status_names_who_started_ollama():
    from agent.doctor.probes import OllamaInfo
    from agent.ui import home

    st = home.StatusInfo(ollama=OllamaInfo(reachable=True, version="0.9", models={"a": 1}), ollama_ours=True)
    assert "(started by nightage)" in dict((a, b) for a, b, _ in home.status_lines(st, 100))["Ollama"]
    st.ollama_ours = False
    assert "(external)" in dict((a, b) for a, b, _ in home.status_lines(st, 100))["Ollama"]
