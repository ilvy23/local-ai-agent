"""Windows readiness that can be proven without Windows (the real thing stays UNVERIFIED, see docs/PORTABILITY.md).

A child interpreter blocks the POSIX-only modules (fcntl, termios, pty, pwd, grp, resource, curses, tty), flips `sys.platform` to
"win32" AFTER the third-party libraries are loaded (they would otherwise reach for Win32 DLLs), then imports every module of the
package and runs the cheap commands in-process. Plus unit tests of the per-user folder helpers for the Windows layout."""
from __future__ import annotations

import ntpath
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent import osport

ROOT = Path(__file__).resolve().parent.parent

CHILD = textwrap.dedent('''
    import importlib, importlib.abc, os, pkgutil, sys, tempfile
    # third-party first, under the real platform
    import typer, typer.testing, rich.console, rich.live, rich.markdown, rich.table, prompt_toolkit, prompt_toolkit.application, httpx, yaml
    import sqlite3, asyncio, subprocess, shutil, logging, json
    BLOCK = {"fcntl", "termios", "pty", "pwd", "grp", "resource", "curses", "_curses", "tty"}
    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCK:
                raise ModuleNotFoundError(f"No module named {name!r} (blocked: simulated Windows)")
    sys.meta_path.insert(0, Blocker())
    for m in list(sys.modules):
        if m.split(".")[0] in BLOCK:
            del sys.modules[m]
    home = tempfile.mkdtemp()
    for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"):
        os.environ.pop(var, None)
    os.environ.update(HOME=home, APPDATA=home + "/Roaming", LOCALAPPDATA=home + "/Local", NIGHTAGE_HOME=home + "/nh",
                      NIGHTAGE_NO_UPDATE_CHECK="1", NO_COLOR="1", NIGHTAGE_PLAIN="1")
    sys.platform = "win32"
    import agent
    bad = []
    for mi in pkgutil.walk_packages(agent.__path__, "agent."):
        if mi.name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(mi.name)
        except BaseException as exc:           # noqa: BLE001
            bad.append(f"{mi.name}: {type(exc).__name__}: {exc}")
    assert not bad, "import failures on simulated Windows:\\n" + "\\n".join(bad)
    from agent.main import app
    runner = typer.testing.CliRunner()
    for args in (["--help"], ["version"], ["doctor", "--list"], ["demo", "--help"]):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, (args, res.exit_code, res.output[-800:], res.exception)
        assert res.output.strip(), args
    print("WINDOWS-SIM-OK")
''')


def test_import_and_core_commands_on_simulated_windows():
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]), "PYTHONDONTWRITEBYTECODE": "1"}
    res = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=240, env=env, cwd=str(ROOT))
    assert "WINDOWS-SIM-OK" in res.stdout, res.stdout[-2000:] + res.stderr[-3000:]


# ── path helpers ─────────────────────────────────────────────────────────────

def test_windows_dirs_use_appdata_and_localappdata():
    d = osport.windows_dirs({"APPDATA": r"C:\Users\user\AppData\Roaming", "LOCALAPPDATA": r"D:\Local"}, r"C:\Users\user")
    assert d == {"config": r"C:\Users\user\AppData\Roaming", "state": r"D:\Local", "data": r"D:\Local"}


def test_windows_dirs_fall_back_to_the_profile():
    d = osport.windows_dirs({}, r"C:\Users\user")
    assert d["config"] == ntpath.join(r"C:\Users\user", "AppData", "Roaming")
    assert d["state"] == ntpath.join(r"C:\Users\user", "AppData", "Local")


def test_posix_layout_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    assert osport.config_home({}, tmp_path) == tmp_path / ".config"
    assert osport.state_home({}, tmp_path) == tmp_path / ".local" / "state"
    assert osport.data_home({"XDG_DATA_HOME": "/ignored"}, tmp_path) == tmp_path / ".local" / "share"
    assert osport.config_home({"XDG_CONFIG_HOME": "/x"}, tmp_path) == Path("/x")
    assert osport.state_home({"XDG_STATE_HOME": "/s"}, tmp_path) == Path("/s")


def test_helpers_pick_windows_dirs_when_platform_is_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    env = {"APPDATA": "/fake/Roaming", "LOCALAPPDATA": "/fake/Local", "XDG_STATE_HOME": "/never"}
    assert osport.config_home(env, "/h") == Path("/fake/Roaming")
    assert osport.state_home(env, "/h") == Path("/fake/Local")
    assert osport.data_home(env, "/h") == Path("/fake/Local")


def test_shell_argv_without_bash_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    assert osport.shell_argv("echo hi") is None
    assert "WSL" in osport.NO_SHELL_MESSAGE
    monkeypatch.setattr(shutil, "which", lambda _n: r"C:\Program Files\Git\bin\bash.exe")
    assert osport.shell_argv("echo hi")[1:] == ["-lc", "echo hi"]


def test_shell_argv_is_plain_bash_on_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert osport.shell_argv("ls") == ["bash", "-lc", "ls"]


def test_run_command_tool_degrades_without_bash(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    from agent.tools.shell import run_command
    out = run_command("echo hi")
    assert "not run" in out and "WSL" in out


def test_windows_utf8_setup_is_a_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    osport.windows_utf8_setup()           # must not raise or re-exec


def test_windows_utf8_setup_skips_non_entry_processes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["pytest"])
    called = []
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: called.append(a) or 0)
    osport.windows_utf8_setup()
    assert not called


def test_windows_utf8_setup_restarts_once_for_the_command(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", [r"C:\x\nightage.exe"])
    monkeypatch.setattr(sys, "orig_argv", ["python", r"C:\x\nightage-script.py", "doctor"], raising=False)
    monkeypatch.delenv("NIGHTAGE_NO_REEXEC", raising=False)
    seen = {}

    def fake_call(argv, env=None):
        seen["argv"], seen["env"] = argv, env
        return 3
    monkeypatch.setattr(subprocess, "call", fake_call)
    if sys.flags.utf8_mode:
        pytest.skip("this interpreter is already in UTF-8 mode")
    with pytest.raises(SystemExit) as ei:
        osport.windows_utf8_setup()
    assert ei.value.code == 3
    assert seen["argv"][1:3] == ["-X", "utf8"] and seen["argv"][-1] == "doctor"
    assert seen["env"]["NIGHTAGE_NO_REEXEC"] == "1"
