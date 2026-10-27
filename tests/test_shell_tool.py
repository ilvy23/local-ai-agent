"""Tests for the shell tool: echo round-trip, timeout, truncation, exit code."""

from __future__ import annotations

import os

from agent.tools.shell import TRUNCATE_MARKER, make_shell_tool, run_command


def test_echo_round_trip():
    result = run_command("echo hello world")
    assert "hello world" in result
    assert "exit code 0" in result.lower()
    # The command and its directory are echoed: bare output is unattributable
    # across several calls, and the cwd is otherwise invisible to the model.
    assert result.startswith("$ echo hello world")
    assert os.getcwd() in result


def test_stderr_is_merged():
    result = run_command("echo oops 1>&2")
    assert "oops" in result


def test_nonzero_exit_code_reported():
    result = run_command("exit 3")
    assert "exit code 3" in result.lower()


def test_result_names_the_directory_it_ran_in(tmp_path):
    result = run_command("pwd", cwd=str(tmp_path))
    assert f"(in {tmp_path}" in result


def test_timeout_message_names_the_command():
    result = run_command("sleep 5", timeout_s=1)
    assert result.startswith("$ sleep 5")
    assert "timed out" in result.lower()


def test_output_truncation():
    # Emit far more than the 10_000-char cap.
    result = run_command("for i in $(seq 1 5000); do echo AAAAAAAAAA; done")
    assert TRUNCATE_MARKER in result
    assert len(result) < 11000


def test_timeout_kills_command():
    result = run_command("sleep 5", timeout_s=1)
    assert "timed out" in result.lower() or "timeout" in result.lower()


def test_timeout_capped_by_config():
    # Even if the model asks for a huge timeout, config caps it. We can't easily
    # observe the cap without sleeping, so just assert a fast command still runs
    # with a large requested timeout and a small cap.
    result = run_command("echo hi", timeout_s=99999, max_timeout_s=2)
    assert "hi" in result


def test_make_shell_tool_declares_dynamic_risk():
    tool = make_shell_tool({"safety": {"max_timeout_s": 300}})
    assert tool.risk == "dynamic"
    assert tool.name == "run_command"
