import io

from rich.console import Console

from agent import changes, tui


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=100, color_system=None)


def test_line_stats():
    assert changes.line_stats(None, "a\nb\nc\n") == (3, 0)
    assert changes.line_stats("a\nb\n", "a\nb\n") == (0, 0)
    assert changes.line_stats("a\nb\nc\n", "a\nX\nc\nd\n") == (2, 1)
    assert changes.line_stats("a\nb\n", "") == (0, 2)


def test_log_merges_edits_to_one_file_and_skips_noops(tmp_path):
    log = changes.FileChangeLog()
    log.record("a.py", None, "1\n2\n3\n")             # created
    log.record("a.py", "1\n2\n3\n", "1\n2\n3\n4\n")   # then edited
    log.record("b.py", "x\n", "x\n")                  # rewritten identically
    log.record("c.py", "old\n", "new\n")
    rows = {p: (a, r, new) for p, a, r, new in log.totals()}
    assert rows == {"a.py": (4, 0, True), "c.py": (1, 1, False)}


def test_summary_text_and_colors(tmp_path):
    log = changes.FileChangeLog()
    f = tmp_path / "pkg" / "worklog.py"
    log.record(str(f), None, "x\n" * 126)
    lines = log.render(cwd=tmp_path)
    assert lines[0].plain == "1 file changed  +126 -0"
    assert lines[1].plain.startswith("  pkg/worklog.py") and "+126 -0" in lines[1].plain
    styles = {line.plain[span.start:span.end]: str(span.style) for line in lines for span in line.spans}
    assert styles["+126"] == "green" and styles["-0"] == "red"
    c = _console()
    log.print_summary(c, cwd=tmp_path)
    assert "worklog.py" in c.file.getvalue()
    assert log.totals() == []                          # cleared after printing


def test_display_path_relative_home_and_absolute(tmp_path):
    assert changes.display_path(str(tmp_path / "a" / "b.py"), cwd=tmp_path) == "a/b.py"
    assert changes.display_path("/usr/share/x.txt", cwd=tmp_path) == "/usr/share/x.txt"


def test_execute_tool_records_real_file_changes(tmp_path):
    from agent.tools.files import write_file, replace_in_file  # noqa: F401  (real handlers)

    tui._CHANGES.clear()
    target = tmp_path / "game.py"
    before = changes.read_text(target)
    target.write_text("a = 1\nb = 2\n")
    tui._CHANGES.record(str(target), before, changes.read_text(target))
    before = changes.read_text(target)
    target.write_text("a = 1\nb = 3\nc = 4\n")
    tui._CHANGES.record(str(target), before, changes.read_text(target))
    assert tui._CHANGES.totals() == [(str(target), 3, 0, True)]
    tui._CHANGES.clear()
