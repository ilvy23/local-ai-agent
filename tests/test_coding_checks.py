from agent.coding.verify.checks import (
    CheckStatus,
    check_pytest,
    check_syntax,
    ladder_passed,
    run_ladder,
)


def test_syntax_passes_on_valid(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n")
    result = check_syntax([f])
    assert result.passed


def test_syntax_fails_on_invalid(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n")
    result = check_syntax([f])
    assert result.status is CheckStatus.FAILED
    assert "bad.py" in result.detail


def test_syntax_ignores_non_python(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("def broken(:\n")
    assert check_syntax([f]).passed


def test_pytest_passes(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_it():\n    assert 1 + 1 == 2\n")
    result = check_pytest(tmp_path)
    assert result.passed


def test_pytest_fails(tmp_path):
    (tmp_path / "test_bad.py").write_text("def test_it():\n    assert False\n")
    result = check_pytest(tmp_path)
    assert result.status is CheckStatus.FAILED


def test_pytest_no_tests_is_skipped(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n")
    result = check_pytest(tmp_path)
    assert result.status is CheckStatus.SKIPPED


def test_ladder_stops_at_syntax(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n")
    results = run_ladder([f], tmp_path)
    assert len(results) == 1
    assert not ladder_passed(results)


def test_ladder_passes_end_to_end(tmp_path):
    (tmp_path / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    results = run_ladder([tmp_path / "mod.py"], tmp_path)
    assert ladder_passed(results)
