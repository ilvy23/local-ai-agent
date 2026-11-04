"""Phase 10: verification runs the project's tests in the project's own environment, does not fail on a lint tool that is not
installed, runs the tests of the change first, and a condensed pytest report keeps its summary (which tests failed, how many)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agent.codeagent import outputs, profile


def _proj(tmp_path: Path, pyproject: str = "[tool.pytest.ini_options]\n") -> Path:
    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text(pyproject)
    (root / "app.py").write_text("x = 1\n")
    (root / "tests" / "test_app.py").write_text("def test_x():\n    assert True\n")
    return root


def test_a_uv_project_runs_its_tests_through_uv(tmp_path, monkeypatch):
    root = _proj(tmp_path)
    (root / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(profile.shutil, "which", lambda name: "/usr/bin/" + name if name == "uv" else None)
    assert profile.project_pytest_command(root) == "uv run --frozen --with pytest python -m pytest -q -x"


@pytest.mark.posix_only
def test_a_venv_with_pytest_is_used(tmp_path, monkeypatch):
    root = _proj(tmp_path)
    py = root / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(profile.shutil, "which", lambda name: None)
    profile._python_has.cache_clear()
    assert profile.project_pytest_command(root) == ".venv/bin/python -m pytest -q -x"


def test_a_poetry_project_uses_poetry(tmp_path, monkeypatch):
    root = _proj(tmp_path)
    (root / "poetry.lock").write_text("")
    monkeypatch.setattr(profile.shutil, "which", lambda name: "/usr/bin/poetry" if name == "poetry" else None)
    assert profile.project_pytest_command(root) == "poetry run python -m pytest -q -x"


def test_ruff_configured_but_missing_gives_no_lint_command(tmp_path, monkeypatch):
    root = _proj(tmp_path, "[tool.ruff]\nline-length = 100\n[tool.pytest.ini_options]\n")
    monkeypatch.setattr(profile.shutil, "which", lambda name: None)
    prof = profile.detect(root)
    assert prof.lint_cmd is None and any("ruff is configured" in n for n in prof.notes)
    assert not any("ruff" in c for c in profile.check_commands(prof, ["app.py"]))


def test_the_tests_of_the_change_run_before_the_full_suite(tmp_path, monkeypatch):
    root = _proj(tmp_path)
    monkeypatch.setattr(profile.shutil, "which", lambda name: None)
    prof = profile.detect(root)
    cmds = profile.check_commands(prof, ["app.py"])
    assert cmds[-1] == prof.test_cmd and cmds[-2] == f"{prof.test_cmd} tests/test_app.py"


def test_a_condensed_report_keeps_the_summary_at_the_end():
    body = "\n".join(f"________ test_{i} ________\nE   assert {i} == 0\ntests/test_a.py:{i}: AssertionError" for i in range(200))
    report = ("collected 300 items\n" + body + "\n=========== short test summary info ===========\n"
              + "\n".join(f"FAILED tests/test_a.py::test_{i} - assert" for i in range(200)) + "\n==== 200 failed, 100 passed in 3.2s ====")
    out = outputs.summarize_pytest(report, max_chars=2200)
    assert out.rstrip().endswith("200 failed, 100 passed in 3.2s ====") and "FAILED tests/test_a.py::test_199" in out
