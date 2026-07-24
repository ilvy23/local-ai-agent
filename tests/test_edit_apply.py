from agent.coding.edit.apply import apply_edits
from agent.coding.edit.format import Edit, EditKind


def test_whole_file_creates_file(tmp_path):
    edit = Edit(path="new.py", kind=EditKind.WHOLE_FILE, replace="x = 1")
    result = apply_edits([edit], tmp_path)
    assert result.ok
    assert (tmp_path / "new.py").read_text() == "x = 1"
    assert result.changed == ("new.py",)


def test_whole_file_creates_parent_dirs(tmp_path):
    edit = Edit(path="pkg/mod.py", kind=EditKind.WHOLE_FILE, replace="y = 2")
    result = apply_edits([edit], tmp_path)
    assert result.ok
    assert (tmp_path / "pkg" / "mod.py").read_text() == "y = 2"


def test_search_replace_applies(tmp_path):
    (tmp_path / "a.py").write_text("value = 1\n")
    edit = Edit(path="a.py", kind=EditKind.SEARCH_REPLACE, search="value = 1", replace="value = 2")
    result = apply_edits([edit], tmp_path)
    assert result.ok
    assert (tmp_path / "a.py").read_text() == "value = 2\n"


def test_search_not_found_rejected(tmp_path):
    (tmp_path / "a.py").write_text("value = 1\n")
    edit = Edit(path="a.py", kind=EditKind.SEARCH_REPLACE, search="missing", replace="x")
    result = apply_edits([edit], tmp_path)
    assert not result.ok
    assert "not found" in result.errors[0]


def test_ambiguous_search_rejected(tmp_path):
    (tmp_path / "a.py").write_text("x\nx\n")
    edit = Edit(path="a.py", kind=EditKind.SEARCH_REPLACE, search="x", replace="y")
    result = apply_edits([edit], tmp_path)
    assert not result.ok
    assert "2 times" in result.errors[0]


def test_path_escape_rejected(tmp_path):
    edit = Edit(path="../evil.py", kind=EditKind.WHOLE_FILE, replace="bad")
    result = apply_edits([edit], tmp_path)
    assert not result.ok
    assert "outside sandbox" in result.errors[0]


def test_all_or_nothing(tmp_path):
    good = Edit(path="good.py", kind=EditKind.WHOLE_FILE, replace="ok")
    bad = Edit(path="bad.py", kind=EditKind.SEARCH_REPLACE, search="nope", replace="x")
    result = apply_edits([good, bad], tmp_path)
    assert not result.ok
    assert not (tmp_path / "good.py").exists()


def test_missing_search_target_rejected(tmp_path):
    edit = Edit(path="nofile.py", kind=EditKind.SEARCH_REPLACE, search="a", replace="b")
    result = apply_edits([edit], tmp_path)
    assert not result.ok
    assert "does not exist" in result.errors[0]
