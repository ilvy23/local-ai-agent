from pathlib import Path

from agent.coding.verify.guards import (
    build_judge_prompt,
    coverage_gate,
    is_test_file,
    parse_judge_verdict,
    reject_test_edits,
)


def test_is_test_file_by_name():
    assert is_test_file("test_foo.py")
    assert is_test_file("foo_test.py")
    assert is_test_file("conftest.py")


def test_is_test_file_by_dir():
    assert is_test_file("tests/helpers.py")
    assert not is_test_file("agent/foo.py")


def test_reject_test_edits_allows_source():
    assert reject_test_edits(["agent/foo.py"]).ok


def test_reject_test_edits_blocks_tests():
    result = reject_test_edits(["agent/foo.py", "tests/test_foo.py"])
    assert not result.ok
    assert "test_foo.py" in result.reason


def test_parse_judge_solves():
    result = parse_judge_verdict('{"verdict": "solves", "reason": "clean fix"}')
    assert result.ok


def test_parse_judge_games():
    result = parse_judge_verdict('{"verdict": "games", "reason": "hardcoded value"}')
    assert not result.ok
    assert "hardcoded" in result.reason


def test_parse_judge_unparseable_allows():
    assert parse_judge_verdict("hmm, not sure").ok


def test_parse_judge_keyword_fallback():
    assert not parse_judge_verdict("this clearly games the test").ok


def test_build_judge_prompt_contains_task_and_diff():
    prompt = build_judge_prompt("fix the bug", "- old\n+ new")
    assert "fix the bug" in prompt
    assert "+ new" in prompt


def test_coverage_gate_no_python_files_passes(tmp_path):
    assert coverage_gate([tmp_path / "notes.txt"], tmp_path).ok


def test_coverage_gate_detects_untouched_change(tmp_path):
    mod = tmp_path / "mod.py"
    mod.write_text("def used():\n    return 1\n\ndef unused():\n    return 2\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import used\n\ndef test_used():\n    assert used() == 1\n"
    )
    result = coverage_gate([mod], tmp_path)
    assert result.ok


def test_coverage_gate_skips_without_coverage(tmp_path, monkeypatch):
    from agent.coding.verify import guards
    from agent.coding.verify.runner import RunResult

    monkeypatch.setattr(
        guards, "run_python", lambda *a, **k: RunResult(127, "", "No module named coverage")
    )
    mod = tmp_path / "mod.py"
    mod.write_text("x = 1\n")
    result = coverage_gate([mod], tmp_path)
    assert result.ok
    assert "skipped" in result.reason
