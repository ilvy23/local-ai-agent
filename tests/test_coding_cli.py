import subprocess

from agent.coding_cli import format_event, read_files, run_code, write_back
from agent.coding.workspace import create_workspace
from agent.events import ActivityEvent
from tests.test_session import FIX, FakeClient, _config, _init_repo


def test_format_attempt():
    line = format_event(ActivityEvent("attempt", data={"tier": 1, "model": "m"}))
    assert "tier 1" in line
    assert "m" in line


def test_format_success():
    assert "passing" in format_event(ActivityEvent("success"))


def test_format_check_failure():
    line = format_event(ActivityEvent("check", "boom", data={"ok": False}))
    assert "✗" in line


def test_format_unknown_returns_none():
    assert format_event(ActivityEvent("mystery")) is None


def test_read_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    files = read_files(["a.py"], tmp_path)
    assert files == {"a.py": "x = 1\n"}


def test_write_back(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        (ws.path / "mod.py").write_text("changed = 1\n")
        written = write_back(ws, ("mod.py",), tmp_path)
        assert written == ["mod.py"]
        assert (tmp_path / "mod.py").read_text() == "changed = 1\n"
    finally:
        ws.close()


def test_run_code_applies_on_success(tmp_path):
    _init_repo(tmp_path)
    seen = []
    result = run_code(
        FakeClient([FIX]), _config(), tmp_path, "fix add()", ["mod.py"],
        apply=True, emit=seen.append,
    )
    assert result.success
    assert (tmp_path / "mod.py").read_text() == "def add(a, b):\n    return a + b"
    assert seen


def test_run_code_no_apply_leaves_repo(tmp_path):
    _init_repo(tmp_path)
    result = run_code(
        FakeClient([FIX]), _config(), tmp_path, "fix add()", ["mod.py"],
        apply=False, emit=lambda e: None,
    )
    assert result.success
    assert (tmp_path / "mod.py").read_text() == "def add(a, b):\n    return a - b\n"
