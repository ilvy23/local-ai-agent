"""Phase-4 edit engine fixes: patches never delete the line after a change, tabs survive re-indentation, files keep their encoding
through an edit, /undo handles CRLF files, deletions and moves, and every numbered view counts lines the same way."""

from __future__ import annotations

import codecs

from agent.codeagent import edits
from tests.test_codeagent_engine import FakeLLM, call, engine, say

SRC = "a = 1\nx = f()\nb = 2\nc = 3\n"


def hunks(patch: str, path: str = "f.py"):
    return edits.parse_patch(patch, path)[0][1]


def test_a_patch_ending_in_a_blank_line_keeps_the_next_line():
    res = edits.apply_hunks(SRC, hunks("@@\n a = 1\n-x = f()\n+x = g()\n\n"))
    assert res.text == "a = 1\nx = g()\nb = 2\nc = 3\n"


def test_leading_blank_context_and_a_blank_removed_line_at_the_edge():
    src = "a = 1\n\nx = f()\nb = 2\n"
    assert edits.apply_hunks(src, hunks("@@\n \n-x = f()\n+x = g()\n")).text == "a = 1\n\nx = g()\nb = 2\n"
    assert edits.apply_hunks(src, hunks("@@\n a = 1\n-\n x = f()\n")).text == "a = 1\nx = f()\nb = 2\n"


def test_a_wrongly_indented_hunk_keeps_the_files_context_lines():
    src = "def f():\n    if x:\n        return 1\n    return 2\n"
    res = edits.apply_hunks(src, hunks("@@\n  if x:\n-     return 1\n+     return 3\n  return 2\n"))
    assert res.text == "def f():\n    if x:\n        return 3\n    return 2\n" and "whitespace-tolerant" in res.how


def test_an_unfound_hunk_shows_the_real_lines():
    res = edits.apply_hunks(SRC, hunks("@@\n-x = f(1)\n+x = g()\n b = 2\n"))
    assert res.text is None and "hunk 1" in res.error and "x = f()" in res.error


def test_tabs_survive_when_the_model_indented_with_spaces():
    go = "func f() {\n\tif x {\n\t\treturn\n\t}\n}\n"
    res = edits.replace_once(go, "    if x {\n        return\n    }", "    if y {\n        return 1\n    }")
    assert res.text == "func f() {\n\tif y {\n\t\treturn 1\n\t}\n}\n"
    res = edits.apply_hunks(go, hunks("@@\n     if x {\n-        return\n+        return 2\n     }\n", "f.go"))
    assert res.text == "func f() {\n\tif x {\n\t\treturn 2\n\t}\n}\n"
    make = "all:\n\tcc -o x x.c\n"
    assert edits.replace_once(make, "    cc -o x x.c", "    cc -O2 -o x x.c").text == "all:\n\tcc -O2 -o x x.c\n"


def test_an_edit_keeps_a_cp1252_file_byte_exact(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    f = repo / "notes.txt"
    f.write_bytes("Größe: 1\r\nPreis: 5 €\r\n".encode("cp1252"))
    msg = eng.execute("edit_file", {"path": "notes.txt", "search": "Größe: 1", "replace": "Größe: 2"})
    assert "error" not in msg.lower(), msg
    assert f.read_bytes() == "Größe: 2\r\nPreis: 5 €\r\n".encode("cp1252")
    assert "stored as cp1252" in eng.execute("read_file", {"path": "notes.txt"})


def test_text_the_encoding_cannot_store_is_refused_and_nothing_written(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    f = repo / "notes.txt"
    raw = "Größe: 1\n".encode("cp1252")
    f.write_bytes(raw)
    msg = eng.execute("edit_file", {"path": "notes.txt", "search": "Größe: 1", "replace": "Größe → 2"})
    assert "cp1252" in msg and "cannot hold" in msg
    assert f.read_bytes() == raw


def test_a_bom_is_kept_and_python_with_a_bom_passes_the_syntax_gate(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    f = repo / "m.py"
    f.write_bytes(codecs.BOM_UTF8 + b"x = 1\n")
    msg = eng.execute("edit_file", {"path": "m.py", "search": "x = 1", "replace": "x = 2"})
    assert "error" not in msg.lower(), msg
    assert f.read_bytes() == codecs.BOM_UTF8 + b"x = 2\n"


def test_undo_restores_crlf_files_deletions_and_moves(tmp_path):
    steps = [call("edit_file", path="w.txt", search="one", replace="two"), call("delete_file", path="gone.txt"),
             call("move_file", src="old.txt", dst="new.txt"), say("Done.")]
    eng, repo, _ = engine(tmp_path, FakeLLM(steps))
    (repo / "w.txt").write_bytes(b"one\r\nline\r\n")
    (repo / "gone.txt").write_text("keep me\n")
    (repo / "old.txt").write_text("moved\n")
    eng.turn("do it")
    assert (repo / "w.txt").read_bytes() == b"two\r\nline\r\n" and not (repo / "gone.txt").exists() and (repo / "new.txt").exists()
    lines = eng.undo.undo()
    assert not any(line.startswith("skipped") for line in lines), lines
    assert (repo / "w.txt").read_bytes() == b"one\r\nline\r\n"
    assert (repo / "gone.txt").read_text() == "keep me\n"
    assert (repo / "old.txt").read_text() == "moved\n" and not (repo / "new.txt").exists()


def test_read_file_numbers_lines_like_replace_lines_even_with_a_form_feed(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "m.py").write_text("a = 1\n\x0c\nb = 2\nc = 3\n")
    shown = eng.execute("read_file", {"path": "m.py"})
    assert "(4 lines" in shown                                  # splitlines() said 5: the form feed counted as a line break
    eng.execute("replace_lines", {"path": "m.py", "start": 3, "end": 3, "new_text": "b = 20"})
    assert (repo / "m.py").read_text() == "a = 1\n\x0c\nb = 20\nc = 3\n"
