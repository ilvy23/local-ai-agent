"""'Tests run the standard way' (agent/codeagent/stdpytest.py): the harness runs `python -m pytest` (cwd on sys.path), a user types plain
`pytest`. Measured live: a 3-module CLI whose tests did `from models import Book` passed the harness' way and died with collection errors for
the user. The check is deterministic, no model call; the runner is injected (a fake that behaves like the two pytest forms), plus one real
pytest process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent.codeagent import stdpytest as SP, verified

from tests.test_codeagent_engine import CFG, FakeLLM, call, engine, say
from tests.test_codeagent_sessions import _isolated  # noqa: F401  (the autouse isolation fixture)

PY = sys.executable
CMD = f"{PY} -m pytest -q -x"
MODELS = "class Book:\n    title = 'x'\n"
TEST = "from models import Book\n\n\ndef test_book():\n    assert Book.title == 'x'\n"
NOT_FOUND = "ERROR collecting tests/test_models.py\nE   ModuleNotFoundError: No module named 'models'\n"


def repo_with_tests(tmp_path, name="proj"):
    root = tmp_path / name
    (root / "tests").mkdir(parents=True)
    (root / "models.py").write_text(MODELS)
    (root / "tests" / "test_models.py").write_text(TEST)
    return root


def fake_pytest(root: Path, module_ok=True, bare_ok=None, log=None):
    """Behaves like pytest: the bare form finds `models` only when the root is configured onto sys.path, the -m form always does."""
    def runner(cmd, timeout):
        if log is not None:
            log.append(cmd)
        if "-m pytest" in cmd:
            return (0, "1 test collected", 0.1) if module_ok else (2, NOT_FOUND, 0.1)
        fixed = (root / "conftest.py").exists() or (root / "pytest.ini").exists() if bare_ok is None else bare_ok
        return (0, "1 test collected", 0.1) if fixed else (2, NOT_FOUND, 0.1)
    return runner


# ── the check ───────────────────────────────────────────────────────────────

def test_bare_pytest_that_cannot_import_is_a_finding(tmp_path):
    root = repo_with_tests(tmp_path)
    f = SP.check(root, CMD, fake_pytest(root))
    assert f == SP.Finding("models")
    assert f.not_verified == "tests only pass with python -m pytest: plain pytest fails to import models"
    assert f.nudge.startswith("Plain `pytest` fails with ModuleNotFoundError: models because the project root is not on sys.path.")
    assert "conftest.py in the project root" in f.nudge and "pythonpath = ." in f.nudge and "both `pytest` and `python -m pytest` must pass" in f.nudge


def test_a_root_conftest_or_pytest_ini_pythonpath_means_no_finding(tmp_path):
    root = repo_with_tests(tmp_path)
    log: list[str] = []
    (root / "conftest.py").write_text("")
    assert SP.check(root, CMD, fake_pytest(root, log=log)) is None and log == []                    # config exists: skipped, nothing ran
    (root / "conftest.py").unlink()
    (root / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    assert SP.check(root, CMD, fake_pytest(root, log=log)) is None and log == []


def test_configuration_that_fixes_the_import_path_skips_the_check(tmp_path):
    for i, (rel, text) in enumerate([("pyproject.toml", "[tool.pytest.ini_options]\npythonpath = ['.']\n"),
                                      ("tox.ini", "[pytest]\npythonpath = .\n"), ("setup.cfg", "[tool:pytest]\npythonpath = .\n"),
                                      ("tests/__init__.py", ""), ("tests/conftest.py", "import sys\nsys.path.insert(0, '.')\n")]):
        root = repo_with_tests(tmp_path, f"p{i}")
        (root / rel).write_text(text)
        assert not SP.applies(root, CMD), rel
    root = repo_with_tests(tmp_path, "plain_cfg")                                                       # config without an import path fixes nothing
    (root / "pytest.ini").write_text("[pytest]\naddopts = -q\n")
    (root / "tests" / "conftest.py").write_text("import pytest\n")
    assert SP.applies(root, CMD)


def test_a_root_pyproject_without_pytest_section_still_checks(tmp_path):
    root = repo_with_tests(tmp_path)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert SP.check(root, CMD, fake_pytest(root)) == SP.Finding("models")


def test_both_forms_failing_the_same_way_is_no_finding(tmp_path):
    root = repo_with_tests(tmp_path)
    assert SP.check(root, CMD, fake_pytest(root, module_ok=False)) is None


def test_a_module_that_is_not_in_the_repo_is_no_finding(tmp_path):
    root = repo_with_tests(tmp_path)
    (root / "models.py").unlink()
    assert SP.check(root, CMD, fake_pytest(root)) is None


def test_a_package_directory_counts_as_a_repo_module(tmp_path):
    root = repo_with_tests(tmp_path)
    (root / "models.py").unlink()
    (root / "models").mkdir()
    (root / "models" / "__init__.py").write_text("")
    assert SP.check(root, CMD, fake_pytest(root)) == SP.Finding("models")


def test_not_applicable_repos_are_skipped(tmp_path):
    root = repo_with_tests(tmp_path)
    assert not SP.applies(root, "python3 -m unittest discover -q") and not SP.applies(root, None) and not SP.applies(root, "npm test")
    node = tmp_path / "node"
    node.mkdir()
    (node / "package.json").write_text("{}")
    assert not SP.applies(node, CMD)
    flat = tmp_path / "flat"                                                                            # test files in the root: pytest puts the root on sys.path itself
    flat.mkdir()
    (flat / "models.py").write_text(MODELS)
    (flat / "test_models.py").write_text(TEST)
    assert not SP.applies(flat, CMD)


def test_the_stamp_follows_sources_tests_and_config(tmp_path):
    root = repo_with_tests(tmp_path)
    a = SP.stamp(root)
    assert SP.stamp(root) == a
    (root / "models.py").write_text(MODELS + "y = 1\n")
    b = SP.stamp(root)
    (root / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    assert len({a, b, SP.stamp(root)}) == 3


def test_the_commands_are_collection_only_and_never_put_the_cwd_on_the_path():
    assert SP.module_command("python3 -m pytest -q -x") == "python3 -m pytest -q --collect-only"
    cmd = SP.standard_command("uv run --no-project --with pytest python3 -m pytest -q -x")
    assert cmd == "uv run --no-project --with pytest pytest -q --collect-only -p no:cacheprovider"
    assert SP.standard_command("npm test") is None


# ── one real pytest process ─────────────────────────────────────────────────

def real_runner(root: Path):
    def run(cmd, timeout):
        p = subprocess.run(["bash", "-c", cmd], cwd=root, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr, 0.0
    return run


def test_real_pytest_finds_the_import_path_problem_and_a_conftest_fixes_it(tmp_path, monkeypatch):
    probe = subprocess.run([PY, "-c", "import pytest"], capture_output=True)
    if probe.returncode != 0:
        import pytest

        pytest.skip("pytest cannot be started with this python")
    # plain pytest of THIS interpreter: the console-script form (no cwd entry in sys.path)
    bare = (f"{PY} -c \"import sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')]; from pytest import console_main; "
            "sys.exit(console_main())\" -q --collect-only -p no:cacheprovider")
    monkeypatch.setattr(SP, "standard_command", lambda cmd: bare)
    root = repo_with_tests(tmp_path)
    cmd = f"{PY} -m pytest -q -x -p no:cacheprovider"
    assert SP.check(root, cmd, real_runner(root)) == SP.Finding("models")
    (root / "conftest.py").write_text("")
    assert SP.check(root, cmd, real_runner(root)) is None
    (root / "conftest.py").unlink()
    (root / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    assert SP.applies(root, cmd) is False


# ── in the engine ───────────────────────────────────────────────────────────

def std_engine(tmp_path, script, runner_root=None, **coding):
    tmp_path.mkdir(parents=True, exist_ok=True)
    eng, repo, out = engine(tmp_path, FakeLLM(script))
    eng.config = {**CFG, "coding": {"claims_check": False, **coding}}
    eng.profile.test_cmd = CMD
    eng.log: list[str] = []
    eng.std_runner = fake_pytest(repo, log=eng.log)
    return eng, repo, out


def write_project():
    return [call("write_file", path="models.py", content=MODELS), call("write_file", path="tests/test_models.py", content=TEST)]


def nudges_of(eng):
    return [m["content"] for m in eng.ctx.messages if m["role"] == "system" and m["content"].startswith("Plain `pytest` fails")]


def test_the_gate_nudges_once_then_not_verified_and_runs_once_per_edit_state(tmp_path):
    eng, repo, _ = std_engine(tmp_path, write_project() + [say("Done, all tests pass.")] * 5)
    # Progress (a green project test run between two answers) legitimately resets the nudge cap, and whether the harness' own
    # test run is green depends on the machine (is pytest installed for this python?). Pin "nothing measurable = no progress".
    eng._measure = lambda: None
    res = eng.turn("build a models module with tests")
    assert len(nudges_of(eng)) == 1                                                                      # cap 1
    assert res.text.startswith("NOT VERIFIED: tests only pass with python -m pytest: plain pytest fails to import models")
    assert res.verified is False and "tests only pass with python -m pytest: plain pytest fails to import models" in res.not_verified
    assert len([c for c in eng.log if "-m pytest" not in c]) == 1 and len(eng.events.of("std_pytest")) == 1   # cached: not re-run without an edit


def test_a_fix_clears_the_finding(tmp_path):
    eng, repo, _ = std_engine(tmp_path, write_project() + [say("Done."), call("write_file", path="conftest.py", content="# puts the project root on sys.path\n"), say("Fixed it.")])
    res = eng.turn("build a models module with tests")
    assert len(nudges_of(eng)) == 1 and res.text == "Fixed it." and res.verified is True and eng.std_last is None
    assert [e["fails"] for e in eng.events.of("std_pytest")] == [True]                                  # the fixed state is not applicable any more: nothing to run


def test_no_python_edit_no_run(tmp_path):
    eng, repo, _ = std_engine(tmp_path, [call("write_file", path="NOTES.md", content="hi\n"), say("Noted.")])
    (repo / "models.py").write_text(MODELS)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_models.py").write_text(TEST)
    assert eng.turn("write notes").text == "Noted." and eng.log == [] and eng.events.of("std_pytest") == []


def test_switched_off_by_config_and_skipped_for_other_test_commands(tmp_path):
    eng, repo, _ = std_engine(tmp_path / "off", write_project() + [say("Done.")], standard_pytest=False)
    assert eng.turn("go").text == "Done." and eng.log == []
    eng, repo, _ = std_engine(tmp_path / "unit", [])
    eng.profile.test_cmd = "python3 -m unittest discover -q"
    assert eng._std_check() is None and eng.log == []


def test_verified_build_shows_one_line_and_clears_it_when_fixed(tmp_path):
    eng, repo, _ = std_engine(tmp_path, [])
    (repo / "models.py").write_text(MODELS)
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_models.py").write_text(TEST)
    eng.profile.test_cmd = CMD
    eng.std_runner = fake_pytest(repo, log=eng.log)
    eng.tools._exec = lambda cmd, timeout: (0, "1 passed in 0.01s\n", 0.1)                                 # the harness' own run is green
    st = verified.build(eng, run_tests=True, run_checks=False)
    assert "plain pytest: FAILS (import path)" in st.sections["TESTS"]["text"] and st.sections["TESTS"]["plain_pytest"] == "fails"
    again = verified.build(eng, run_tests=True, run_checks=False)
    assert len([c for c in eng.log if "-m pytest" not in c]) == 1 and "plain pytest" in again.text                    # cached
    (repo / "conftest.py").write_text("")
    fixed = verified.build(eng, run_tests=True, run_checks=False)
    assert "plain pytest" not in fixed.text and "plain_pytest" not in fixed.sections["TESTS"]
