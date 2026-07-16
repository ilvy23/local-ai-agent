from pathlib import Path

import pytest

from agent.tools.files import (
    LIST_DIR_TOOL,
    SEARCH_FILES_TOOL,
    list_dir,
    read_file,
    search_files,
    write_file,
)


def test_read_file_round_trip(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    out = read_file(str(f))
    # Headed with the file it came from: raw text alone is unattributable, so
    # the model can't tell two files apart or notice it read the wrong one.
    assert out == f"File {f.resolve()} (11 chars):\nhello world"


def test_read_file_truncates(tmp_path: Path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 100)
    out = read_file(str(f), max_chars=10)
    # The header says it's partial — a truncated read must not look whole.
    assert out.startswith(f"File {f.resolve()} (first 10 of 100 chars):")
    assert "x" * 10 in out
    assert "[truncated]" in out


def test_read_file_missing_returns_error_string(tmp_path: Path):
    out = read_file(str(tmp_path / "nope.txt"))
    assert "Error" in out or "not found" in out.lower()


def test_read_file_binary_returns_error_string(tmp_path: Path):
    f = tmp_path / "bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe")
    out = read_file(str(f))
    assert "binary" in out.lower()


def test_list_dir_names_and_types(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hi")
    (tmp_path / "sub").mkdir()
    out = list_dir(str(tmp_path))
    assert "a.txt" in out
    assert "sub" in out
    assert "dir" in out
    assert "file" in out


def test_list_dir_missing_returns_error_string(tmp_path: Path):
    out = list_dir(str(tmp_path / "nope"))
    assert "Error" in out or "not" in out.lower()


def test_list_dir_caps_entries(tmp_path: Path):
    for i in range(510):
        (tmp_path / f"f{i:04d}.txt").write_text("x")
    out = list_dir(str(tmp_path))
    assert "500" in out  # cap noted


def test_search_files_finds_substring(tmp_path: Path):
    (tmp_path / "a.txt").write_text("the needle is here\nother line")
    (tmp_path / "b.txt").write_text("nothing relevant")
    out = search_files("needle", str(tmp_path))
    assert "a.txt" in out
    assert "needle" in out
    assert "b.txt" not in out


def test_search_files_skips_excluded_dirs(tmp_path: Path):
    excluded = tmp_path / ".git"
    excluded.mkdir()
    (excluded / "secret.txt").write_text("needle in git")
    data = tmp_path / "data"
    data.mkdir()
    (data / "db.txt").write_text("needle in data")
    (tmp_path / "keep.txt").write_text("needle in keep")
    out = search_files("needle", str(tmp_path))
    assert "keep.txt" in out
    assert ".git" not in out
    assert "db.txt" not in out


def test_search_files_skips_binary_silently(tmp_path: Path):
    (tmp_path / "binfile").write_bytes(b"\x00needle\x00")
    (tmp_path / "keep.txt").write_text("needle here")
    out = search_files("needle", str(tmp_path))
    assert "keep.txt" in out
    assert "binfile" not in out


def test_write_file_creates_and_parent_dirs(tmp_path: Path):
    target = tmp_path / "nested" / "out.txt"
    out = write_file(str(target), "content here")
    assert target.read_text() == "content here"
    assert "Error" not in out


def test_write_file_append(tmp_path: Path):
    target = tmp_path / "out.txt"
    write_file(str(target), "one\n")
    write_file(str(target), "two\n", append=True)
    assert target.read_text() == "one\ntwo\n"


def test_write_file_refuses_data_dir(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("agent.tools.files.DEFAULT_DATA_DIR", data_dir)
    out = write_file(str(data_dir / "agent.db"), "malicious")
    assert "Error" in out or "refus" in out.lower()
    assert not (data_dir / "agent.db").exists()


def test_write_file_refuses_data_dir_via_traversal(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("agent.tools.files.DEFAULT_DATA_DIR", data_dir)
    sneaky = tmp_path / "sub" / ".." / "data" / "x.txt"
    out = write_file(str(sneaky), "malicious")
    assert "Error" in out or "refus" in out.lower()
    assert not (data_dir / "x.txt").exists()


def test_listing_names_the_directory_it_listed(tmp_path):
    """The model only sees bare filenames otherwise, so it can't tell which
    folder a listing came from — that's how it ends up describing the wrong one."""
    (tmp_path / "song.mp3").write_text("x")
    out = list_dir(str(tmp_path))
    assert out.splitlines()[0] == f"Contents of {tmp_path.resolve()} (1 entry):"
    assert "- song.mp3 (file, 1 bytes)" in out


def test_listing_counts_entries(tmp_path):
    for n in ("a", "b", "c"):
        (tmp_path / n).write_text("")
    assert f"({3} entries)" in list_dir(str(tmp_path))


def test_empty_directory_says_so_with_its_path(tmp_path):
    out = list_dir(str(tmp_path))
    assert str(tmp_path.resolve()) in out and "empty" in out


def test_errors_name_the_resolved_path(tmp_path):
    missing = tmp_path / "nope"
    assert str(missing.resolve()) in list_dir(str(missing))


def test_path_is_required_so_the_model_cannot_silently_list_the_cwd():
    # "." is wherever the CLI was launched — the model has no idea what that is,
    # and defaulting to it is how a request for ~/Music listed the home dir.
    assert LIST_DIR_TOOL.parameters["required"] == ["path"]


def test_search_says_where_it_looked(tmp_path):
    (tmp_path / "a.txt").write_text("the needle is here")
    out = search_files("needle", str(tmp_path))
    assert out.splitlines()[0] == f"Searched 'needle' under {tmp_path.resolve()} — 1 match:"


def test_no_matches_still_names_the_directory(tmp_path):
    """'No matches' with no location lets the model conclude the thing doesn't
    exist anywhere, when it only ever looked in one (possibly wrong) place."""
    out = search_files("nowhere", str(tmp_path))
    assert str(tmp_path.resolve()) in out
    assert "No matches" in out


def test_search_path_is_required():
    assert set(SEARCH_FILES_TOOL.parameters["required"]) == {"pattern", "path"}
