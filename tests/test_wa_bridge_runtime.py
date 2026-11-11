"""agent/wa_bridge_runtime.py: starting/stopping the WhatsApp bridge detached for the
home screen's lifetime. No real Node process or network call - subprocess.Popen and
httpx.get are both faked."""

from __future__ import annotations

import signal

import pytest

from agent import wa_bridge_runtime as W


@pytest.fixture(autouse=True)
def _reset():
    """Every test gets a clean module state and its own bridge dir (set per-test below)."""
    W._proc = None
    W._hooks_installed = False
    W._prev_handlers.clear()
    yield
    W._proc = None
    W._hooks_installed = False


@pytest.fixture
def bridge_dir(tmp_path, monkeypatch):
    d = tmp_path / "bridge"
    (d / "auth").mkdir(parents=True)
    (d / "index.js").write_text("// fake bridge")
    monkeypatch.setattr(W, "_bridge_dir", lambda: d)
    return d


class FakeProc:
    def __init__(self, exits_immediately=False, returncode=1):
        self.pid = 4242
        self._exited = exits_immediately
        self.returncode = returncode if exits_immediately else None
        self.killed_with = []

    def poll(self):
        return self.returncode if self._exited else None


def test_disabled_by_config_does_nothing(bridge_dir, monkeypatch):
    called = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: called.append(1))
    r = W.ensure_running({"whatsapp": {"bridge_autostart": False}})
    assert r.ok and r.action == "disabled"
    assert not called


def test_not_installed_when_index_js_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_bridge_dir", lambda: tmp_path / "nowhere")
    r = W.ensure_running({"whatsapp": {}})
    assert r.ok and r.action == "not-installed"


def test_already_running_is_left_alone(bridge_dir, monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp(200, {"status": "ok"}))
    called = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: called.append(1))
    r = W.ensure_running({"whatsapp": {}})
    assert r.ok and r.action == "already-running"
    assert not called


def test_needs_pairing_when_never_linked(bridge_dir, monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: (_ for _ in ()).throw(__import__("httpx").ConnectError("refused")))
    called = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: called.append(1))
    r = W.ensure_running({"whatsapp": {}})
    assert not r.ok and r.action == "needs-pairing"
    assert not called


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def test_starts_detached_when_paired_and_unreachable(bridge_dir, monkeypatch):
    (bridge_dir / "auth" / "creds.json").write_text("{}")
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise __import__("httpx").ConnectError("refused")  # pre-start probe: not running yet
        return _Resp(200, {"status": "ok"})  # after "starting": it answers

    popen_calls = []

    def fake_popen(argv, **kwargs):
        popen_calls.append((argv, kwargs))
        return FakeProc()

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(W, "_install_hooks", lambda: None)  # don't touch real process signal handlers in tests

    r = W.ensure_running({"whatsapp": {}})
    assert r.ok and r.action == "started"
    assert popen_calls and popen_calls[0][0] == ["node", "index.js"]
    assert popen_calls[0][1]["cwd"] == bridge_dir


def test_failed_start_is_reported(bridge_dir, monkeypatch):
    (bridge_dir / "auth" / "creds.json").write_text("{}")
    monkeypatch.setattr("httpx.get", lambda *a, **k: (_ for _ in ()).throw(__import__("httpx").ConnectError("refused")))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: FakeProc(exits_immediately=True, returncode=7))
    monkeypatch.setattr(W, "_install_hooks", lambda: None)

    r = W.ensure_running({"whatsapp": {}})
    assert not r.ok and r.action == "failed" and "7" in r.detail


def test_stop_kills_the_process_group_we_started(monkeypatch):
    killed = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    W._proc = FakeProc()
    W.stop(wait=0)
    assert killed and killed[0] == (4242, signal.SIGTERM)
    assert W._proc is None


def test_stop_is_a_noop_when_nothing_was_started(monkeypatch):
    killed = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    W.stop()
    assert not killed


def test_a_process_that_already_exited_is_not_killed_again(monkeypatch):
    killed = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    W._proc = FakeProc(exits_immediately=True, returncode=0)
    W.stop()
    assert not killed
