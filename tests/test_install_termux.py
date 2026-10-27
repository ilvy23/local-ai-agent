import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "install-termux.sh"
SH = shutil.which("sh")


@pytest.mark.skipif(SH is None, reason="no sh")
def test_syntax_and_help():
    assert subprocess.run([SH, "-n", str(SCRIPT)]).returncode == 0
    out = subprocess.run([SH, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "Termux" in out.stdout


@pytest.mark.skipif(SH is None, reason="no sh")
def test_refuses_outside_termux_and_dry_run_changes_nothing(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PREFIX": str(tmp_path / "prefix")}
    nope = subprocess.run([SH, str(SCRIPT), "--yes"], capture_output=True, text=True, env=env)
    assert nope.returncode == 1 and "Termux" in nope.stderr
    env["NIGHTAGE_REMOTE_HOST"] = "100.64.0.5"
    dry = subprocess.run([SH, str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env)
    assert dry.returncode == 0 and "[dry-run]" in dry.stdout and "ollama.host" in dry.stdout
    assert not (tmp_path / ".nightage-venv").exists() and not (tmp_path / "prefix").exists()
