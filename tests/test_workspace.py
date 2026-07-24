import subprocess

import pytest

from agent.coding.workspace import WorkspaceError, create_workspace, is_git_repo


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=path, check=True)


def test_is_git_repo(tmp_path):
    assert not is_git_repo(tmp_path)
    _init_repo(tmp_path)
    assert is_git_repo(tmp_path)


def test_create_workspace_rejects_non_repo(tmp_path):
    with pytest.raises(WorkspaceError):
        create_workspace(tmp_path)


def test_workspace_isolates_changes(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        (ws.path / "seed.py").write_text("x = 2\n")
        assert "x = 2" in ws.read("seed.py")
        assert (tmp_path / "seed.py").read_text() == "x = 1\n"
    finally:
        ws.close()


def test_workspace_diff_and_changed_files(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        (ws.path / "seed.py").write_text("x = 99\n")
        assert "seed.py" in ws.changed_files()
        assert "x = 99" in ws.diff()
    finally:
        ws.close()


def test_workspace_reset(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        (ws.path / "seed.py").write_text("x = 99\n")
        ws.reset()
        assert ws.read("seed.py") == "x = 1\n"
    finally:
        ws.close()


def test_workspace_commit(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        (ws.path / "seed.py").write_text("x = 5\n")
        sha = ws.commit("change")
        assert len(sha) == 40
    finally:
        ws.close()


def test_context_manager_cleans_up(tmp_path):
    _init_repo(tmp_path)
    with create_workspace(tmp_path) as ws:
        root = ws.path
        assert root.exists()
    assert not root.exists()
