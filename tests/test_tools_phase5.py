"""Chat file + shell tools: bounded reads of regular files only, long lines shown around the match, the end of long command
output kept, and a timeout that stops everything the command started."""

from __future__ import annotations

import os
import sys
import time

import pytest

from agent import procrun
from agent.tools import files, shell


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_read_file_refuses_a_fifo_instead_of_hanging(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    t0 = time.monotonic()
    out = files.read_file(str(fifo))
    assert "not a regular file" in out and time.monotonic() - t0 < 2


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="needs /dev/zero")
def test_read_file_refuses_dev_zero():
    assert "not a regular file" in files.read_file("/dev/zero")


def test_read_file_reads_a_big_file_only_up_to_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "WHOLE_READ_BYTES", 50_000)          # "big" = over the whole-read size
    big = tmp_path / "big.log"
    big.write_text("x" * 200_000 + "ü" * 10)
    out = files.read_file(str(big), max_chars=1000)
    assert out.count("x") == 1000 and "the file has" in out and "[truncated]" in out


def test_a_cut_read_never_turns_utf8_into_mojibake(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("ä" * 5000, encoding="utf-8")
    out = files.read_file(str(f), max_chars=10)
    assert "ä" * 10 in out and "Ã" not in out


def test_search_shows_a_window_of_a_very_long_line(tmp_path):
    (tmp_path / "app.min.js").write_text("a" * 5000 + "needle" + "b" * 5000)
    out = files.search_files("needle", str(tmp_path))
    line = [x for x in out.splitlines() if "needle" in x and "app.min.js" in x][0]
    assert len(line) < 600


def test_procrun_cap_keeps_head_and_tail():
    text = "HEAD" + "m" * 50_000 + "1 failed, 20 passed"
    capped = procrun.cap(text, 100, 200)
    assert capped.startswith("HEAD") and capped.endswith("1 failed, 20 passed") and "cut from the middle" in capped


def test_run_command_keeps_the_end_of_long_output(tmp_path):
    cmd = f"{sys.executable} -c \"print('x' * 30000); print('== 3 failed, 10 passed ==')\""
    out = shell.run_command(cmd, timeout_s=30, cwd=str(tmp_path))
    assert "3 failed, 10 passed" in out and "exit code 0" in out


@pytest.mark.posix_only
def test_a_timeout_kills_the_children_too(tmp_path):
    marker = tmp_path / "child.pid"
    cmd = f"sleep 30 & echo $! > {marker}; wait"
    t0 = time.monotonic()
    out = shell.run_command(cmd, timeout_s=1, cwd=str(tmp_path))
    assert "Timed out" in out and time.monotonic() - t0 < 10
    pid = int(marker.read_text())
    time.sleep(0.3)
    assert not _running(pid)                             # the grandchild is gone, not left running


def _running(pid: int) -> bool:
    """Alive and not a zombie (a killed child of a dead parent stays a zombie until init reaps it - in a container that can take a while)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


@pytest.mark.posix_only
def test_a_background_child_no_longer_makes_the_call_wait(tmp_path):
    t0 = time.monotonic()
    out = shell.run_command("sleep 20 >/dev/null 2>&1 & echo started", timeout_s=15, cwd=str(tmp_path))
    assert "started" in out and time.monotonic() - t0 < 5
