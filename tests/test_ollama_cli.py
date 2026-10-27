"""`nightage ollama ...` / `nightage pull`: progress line math at several widths, plain output, CLI wiring against the fake server."""
from __future__ import annotations

import io
import json
import os
import signal
import sys

import pytest
from rich.console import Console
from typer.testing import CliRunner

from agent import ollama_cli as C
from agent import ollama_local as O
from agent.ui import style
from tests.test_ollama_local import FAKE, env, free_port, posix, requests_of  # noqa: F401  (env is a fixture)


def prog(pct, total=4 * 1024 ** 3, speed=12.5 * 1024 ** 2, eta=200.0, done=False):
    return O.Progress(status="pulling", total=total, completed=int(total * pct / 100), pct=pct, speed=speed, eta=eta, done=done)


@pytest.mark.parametrize("width", [30, 40, 60, 80, 100, 200])
@pytest.mark.parametrize("pct", [0, 37, 100])
@pytest.mark.parametrize("g", [style.UNICODE, style.ASCII])
def test_line_fits_every_width(width, pct, g):
    line = C.render_line(prog(pct), width, g)
    assert len(line) <= width - 1, (width, line)
    assert f"{pct:3.0f}%" in line
    assert line.startswith("[") and "]" in line


def test_bar_math_and_min_width():
    line = C.render_line(prog(37), 100, style.UNICODE)
    inner = line[1:line.index("]")]
    assert inner.count("█") == round(0.37 * len(inner)) and len(inner) <= 40
    for w in (40, 60, 100):
        inner = C.render_line(prog(50), w, style.UNICODE)[1:].split("]")[0]
        assert len(inner) >= 10
    assert set(C.render_line(prog(100), 40, style.UNICODE)[1:].split("]")[0]) == {"█"}
    assert set(C.render_line(prog(0), 40, style.UNICODE)[1:].split("]")[0]) == {"░"}


def test_ascii_fallback_has_no_unicode():
    assert C.render_line(prog(37), 40, style.ASCII).isascii()


def test_narrow_drops_details_first():
    assert "ETA" in C.render_line(prog(50), 100)
    narrow = C.render_line(prog(50), 40)
    assert "GB" not in narrow and len(narrow) <= 39


def test_eta_and_bytes_formatting():
    assert C.fmt_eta(None) == "ETA ?" and C.fmt_eta(45) == "ETA 45s" and C.fmt_eta(192) == "ETA 3m 12s" and C.fmt_eta(7300) == "ETA 2h 01m"
    assert C.fmt_bytes(1536) == "2 kB" and C.fmt_bytes(46 * 1024 ** 2) == "46.0 MB" and C.fmt_bytes(3.4 * 1024 ** 3) == "3.4 GB"


def test_non_terminal_view_prints_a_line_per_ten_percent():
    buf = io.StringIO()
    view = C.PullView(Console(file=buf, width=40, force_terminal=False), plain=True)
    for pct in (0, 3, 9, 10, 15, 20, 55, 100):
        view.update(prog(pct))
    lines = [x for x in buf.getvalue().splitlines() if x]
    assert len(lines) == 5 and all(len(x) <= 39 and x.isascii() for x in lines)


def test_terminal_view_rewrites_one_line():
    buf = io.StringIO()
    con = Console(file=buf, width=40, force_terminal=True)
    view = C.PullView(con, plain=False)
    view.update(O.Progress(status="pulling manifest"))
    view.update(prog(50))
    view.finish()
    out = buf.getvalue()
    assert "\r" in out and "pulling manifest" in out and "50%" in out


# ── commands against the fake server ───────────────────────────────────────────────────────

@pytest.fixture
def cli(env, monkeypatch):  # noqa: F811
    cfg = {**env.cfg, "models": {"chat": "fake:1b"}}
    monkeypatch.setattr(C, "_cfg", lambda: cfg)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FAKE_OLLAMA_PULL_BYTES", "4000")
    import typer

    app = typer.Typer()
    C.register(app)
    return lambda *a: CliRunner().invoke(app, list(a))


@posix
def test_status_start_list_stop(cli, env):
    r = cli("ollama", "status")
    assert r.exit_code == 1 and "not running" in r.output and "nightage ollama start" in r.output
    r = cli("ollama", "start")
    assert r.exit_code == 0 and "started" in r.output
    r = cli("ollama", "status")
    assert r.exit_code == 0 and "running" in r.output and "started by Nightage" in r.output
    r = cli("ollama", "list")
    assert "fake:1b" in r.output
    assert cli("ollama", "rm", "fake:1b").exit_code == 0
    assert cli("ollama", "rm", "nope:1b").exit_code == 1
    r = cli("ollama", "stop")
    assert r.exit_code == 0 and "stopped" in r.output
    assert cli("ollama", "stop").output.startswith("no Nightage-started server")


@posix
def test_pull_shortcut_autostarts_downloads_and_stops_again(cli, env):
    r = cli("pull", "fake:1b")
    assert r.exit_code == 0, r.output
    assert "Started Ollama in the background" in r.output and "fake:1b is ready." in r.output and "100%" in r.output
    assert O.probe(env.url) is None                      # started for this download only -> stopped afterwards


@posix
def test_pull_leaves_a_running_server_alone(cli, env):
    assert O.start(env.cfg).ok
    try:
        assert cli("pull", "fake:1b").exit_code == 0
        assert O.probe(env.url) is not None
    finally:
        O.stop(env.cfg)


@posix
def test_pull_unknown_model_reports_error(cli, env):
    r = cli("ollama", "pull", "missing:1b")
    assert r.exit_code == 1 and "Could not download" in r.output and "does not exist" in r.output


@posix
def test_pull_refuses_oversized_model_unless_forced(cli, env, monkeypatch):
    monkeypatch.setattr(O, "estimate_size", lambda m: (int(80e9), "registry"))
    monkeypatch.setattr(C.ollama_local, "disk_free_bytes", lambda: 500 * 1024 ** 3)
    from agent import sysinfo

    monkeypatch.setattr(sysinfo, "current", lambda: sysinfo.SysInfo(ram_total_mb=11000, ram_free_mb=6000))
    r = cli("pull", "huge:70b")
    assert r.exit_code == 2 and "Too big for this machine" in r.output and "--force" in r.output
    assert not any(x.get("path") == "/api/pull" for x in requests_of(env.log))
    r = cli("pull", "huge:70b", "--force")
    assert r.exit_code == 0 and "ready" in r.output


def test_pull_with_server_off_and_autostart_off_says_how_to_start(cli, env):
    env.cfg["ollama"]["autostart"] = False
    r = cli("pull", "fake:1b")
    assert r.exit_code == 1 and "nightage ollama start" in " ".join(r.output.split())


def test_remote_host_is_not_started_nor_fit_checked(monkeypatch, tmp_path):
    cfg = {"ollama": {"host": "192.0.2.10:11434", "autostart": True}}
    monkeypatch.setattr(C, "_cfg", lambda: cfg)
    monkeypatch.setattr(O, "pull", lambda *a, **k: O.Progress(done=True, pct=100))
    monkeypatch.setattr(O, "start", lambda *a, **k: pytest.fail("a remote host must never be started"))
    out = io.StringIO()
    code = C.do_pull("big:70b", console=Console(file=out, width=60, force_terminal=False), config=cfg)
    assert code == 0 and "Too big" not in out.getvalue()
