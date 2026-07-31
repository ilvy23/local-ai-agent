"""Step 1: the sandbox + check ladder work without a model."""

from __future__ import annotations

import subprocess

import pytest

from agent.coding.verify.checks import run_ladder
from agent.coding.workspace import Workspace


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "t@t"], tmp_path)
    _git(["config", "user.name", "t"], tmp_path)
    (tmp_path / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-q", "-m", "init"], tmp_path)
    return tmp_path


def test_ladder_passes_on_good_code(repo):
    r = run_ladder(repo, [repo / "mod.py"])
    assert r.ok and r.stage == "all", r.error


def test_ladder_catches_test_failure(repo):
    (repo / "mod.py").write_text("def add(a, b):\n    return a - b\n")  # bug
    r = run_ladder(repo, [repo / "mod.py"])
    assert not r.ok and r.stage == "tests"


def test_ladder_catches_syntax_before_tests(repo):
    (repo / "mod.py").write_text("def add(a, b)\n    return a + b\n")  # missing colon
    r = run_ladder(repo, [repo / "mod.py"])
    assert not r.ok and r.stage == "syntax"


def test_ladder_catches_undefined_name(repo):
    (repo / "mod.py").write_text("def add(a, b):\n    return a + c\n")  # c undefined
    r = run_ladder(repo, [repo / "mod.py"])
    assert not r.ok and r.stage in ("lint", "tests")  # F821 or NameError at test time


def test_workspace_isolates_from_working_tree(repo):
    with Workspace(repo) as ws:
        assert "add" in ws.read("mod.py")
        ws.write("mod.py", "def add(a, b):\n    return a + b  # EDITED\n")
        assert "EDITED" in ws.diff()
    # the real working tree is untouched
    assert "EDITED" not in (repo / "mod.py").read_text()


def test_lint_auto_imports_missing_stdlib(tmp_path):
    """The observed snake failure: model uses random/time without `import`. Lint
    catches it (F821), we auto-add the import, re-run, pass."""
    from agent.coding.verify.checks import check_lint
    (tmp_path / "app.py").write_text(
        "def pick():\n    return random.randint(0, 5)\n\ndef wait():\n    time.sleep(0)\n"
    )
    r = check_lint(tmp_path)
    assert r.ok, f"expected auto-fix to pass, got: {r}"
    assert "random" in r.note and "time" in r.note
    # verify the imports were actually written
    src = (tmp_path / "app.py").read_text()
    assert src.startswith("import random\nimport time\n")


def test_lint_hints_at_non_stdlib_undefined_names(tmp_path):
    """For custom undefined names (colours, dimensions) we can't auto-fix — the
    model must, and the message should tell it what to define."""
    from agent.coding.verify.checks import check_lint
    (tmp_path / "game.py").write_text(
        "def draw():\n    screen.fill(black)\n    return screen_width\n"
    )
    r = check_lint(tmp_path)
    assert not r.ok
    assert "screen_width" in r.error and "black" in r.error
    assert "FIX:" in r.error


def test_smoke_run_passes_for_runnable_script(tmp_path):
    """A runnable script with no tests is validated by running it, not by pytest."""
    from agent.coding.verify.checks import check_tests
    (tmp_path / "app.py").write_text(
        "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n"
    )
    r = check_tests(tmp_path)
    assert r.ok and "smoke" in r.note.lower()


def test_smoke_run_catches_crash(tmp_path):
    """If the script crashes on start, smoke fails and says why."""
    from agent.coding.verify.checks import check_tests
    (tmp_path / "app.py").write_text(
        "import nonexistent_pkg_xyz_999\n\nif __name__ == '__main__':\n    pass\n"
    )
    r = check_tests(tmp_path)
    # missing module gets auto-install attempt; if that fails, we should see
    # either the install-failed message OR the smoke crash. Either is not-ok.
    assert not r.ok


def test_missing_module_gets_auto_installed_and_surfaced(tmp_path, monkeypatch):
    """The whole point: a missing package should be installed automatically or,
    if the auto-install budget is spent, surfaced with a clear next step so the
    model never edits code trying to 'fix' an ImportError."""
    from agent.coding.verify import checks as C

    (tmp_path / "game.py").write_text("import totally_fake_pkg_xyz\n\n\ndef s(): pass\n")
    (tmp_path / "test_game.py").write_text(
        "from game import s\n\n\ndef test_it(): s()\n"
    )
    # Budget exhausted → we exercise the surfacing path (no network needed).
    monkeypatch.setattr(C, "_installs_used", C._INSTALL_BUDGET)
    r = C.check_tests(tmp_path)
    assert not r.ok
    assert "totally_fake_pkg_xyz" in r.error   # named the missing thing


def test_installable_refuses_stdlib_and_bad_names():
    """Safety: never pip install stdlib or a name that could reach a bad package."""
    from agent.coding.verify.checks import _installable
    assert _installable("ModuleNotFoundError: No module named 'os'") == []
    assert _installable("ModuleNotFoundError: No module named 'json'") == []
    assert _installable("ModuleNotFoundError: No module named 'pygame'") == ["pygame"]
    assert _installable("ModuleNotFoundError: No module named 'numpy.linalg'") == ["numpy"]
    # a garbage / suspicious name is refused
    assert _installable("ModuleNotFoundError: No module named '../evil'") == []


def test_no_tests_message_adapts_to_repo_state(tmp_path):
    """The 'no tests' message must reflect what actually exists so the model
    isn't told to add a test when there's no code yet (the requirements.txt bug)."""
    from agent.coding.verify.checks import check_tests

    # No .py at all (the actual bug from the snake run) — must be told to write
    # implementation first, not to add a test.
    (tmp_path / "requirements.txt").write_text("pygame\n")
    r = check_tests(tmp_path)
    assert not r.ok and "implementation FIRST" in r.error
    assert "requirements.txt" in r.error  # names the exact wrong path the model took

    # Code exists but test file name is off → tell it how to name a test
    (tmp_path / "game.py").write_text("def play(): pass\n")
    r = check_tests(tmp_path)
    assert not r.ok and "test_<name>.py" in r.error


def test_workspace_commit_keeps_branch(repo):
    with Workspace(repo, branch="agent-code/test") as ws:
        ws.write("mod.py", "def add(a, b):\n    return a + b  # v2\n")
        ws.commit("test change")
        ws.keep = True
    branches = subprocess.run(
        ["git", "branch"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "agent-code/test" in branches
