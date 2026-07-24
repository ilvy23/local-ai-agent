import sys

from agent.coding.verify.runner import run, run_python


def test_run_captures_stdout(tmp_path):
    result = run([sys.executable, "-c", "print('hello')"], cwd=tmp_path)
    assert result.ok
    assert "hello" in result.stdout


def test_run_reports_returncode(tmp_path):
    result = run([sys.executable, "-c", "import sys; sys.exit(3)"], cwd=tmp_path)
    assert result.returncode == 3
    assert not result.ok


def test_run_times_out(tmp_path):
    result = run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        timeout_seconds=0.5,
    )
    assert result.timed_out
    assert not result.ok


def test_run_missing_binary(tmp_path):
    result = run(["definitely-not-a-real-binary-xyz"], cwd=tmp_path)
    assert not result.ok


def test_run_python_module(tmp_path):
    result = run_python(["json.tool", "--help"], cwd=tmp_path)
    assert result.returncode == 0
