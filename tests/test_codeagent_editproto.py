"""Edit protocol for prose-heavy files (measured live: 687 lines of dialogue, 10 minutes, 21 lines rewritten):
  1. an edit that was already applied and is re-sent answers ONE identical 'ALREADY APPLIED' message (a loop-guard repeat, not an error),
  2. a stale stamp that matches exactly ONE shifted position applies there, two positions or none reject as before,
  3. replace_lines / edit_file / insert_after never turn a valid py/json/toml/yaml file into an invalid one (coding.syntax_reject)."""

from __future__ import annotations

import json

from agent.codeagent import profile as PR, symbols
from agent.codeagent.context import ContextManager
from agent.codeagent.eventlog import EventLog
from agent.codeagent.tools import CodeTools

from tests.test_codeagent_engine import FakeLLM, engine  # noqa: F401
from tests.test_codeagent_hardening import _isolated  # noqa: F401

LINES = [f'LINE_{i:02d} = "{"words about the lake and the bell " * 2}number {i}"' for i in range(1, 31)]
SRC = "\n".join(LINES) + "\n"


def tools(tmp_path, src=SRC, name="dialogue.py", **kw):
    (tmp_path / name).write_text(src)
    ctx = ContextManager(None, {"models": {"background": "m"}}, 32768, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    return CodeTools(tmp_path, PR.detect(tmp_path), ctx, **kw)


def stamp(text, a, b):
    return symbols.range_hash(text.splitlines()[a - 1:b])


def new_block(a, b, tag="NEW"):
    return "\n".join(f'LINE_{i:02d} = "{tag} sentence {i}, cleaner and shorter"' for i in range(a, b + 1)) + "\n"


# ── 1. already applied ──────────────────────────────────────────────────────

def test_friction_sequence_edit_applied_then_resent_with_the_old_stamp(tmp_path):
    t = tools(tmp_path)
    old = stamp(SRC, 5, 9)
    first = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=old)
    assert first.startswith("edited dialogue.py")
    after = (tmp_path / "dialogue.py").read_text()
    again = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=old)           # the stamp of the text from BEFORE its own edit
    assert again.startswith("ALREADY APPLIED: these lines already contain your new text (lines 5-9). Nothing to do here.")
    assert "Next line of your window that is still unchanged: 10." in again
    assert not again.startswith(("error", "NOT CHANGED")) and (tmp_path / "dialogue.py").read_text() == after
    third = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=old)
    assert third == again                                                                   # identical for every repeat


def test_already_applied_without_a_stamp_and_with_the_current_stamp(tmp_path):
    t = tools(tmp_path)
    t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=stamp(SRC, 5, 9))
    now = (tmp_path / "dialogue.py").read_text()
    no_stamp = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9))                       # file changed since its last numbered view? no - it saw the new numbers
    assert no_stamp.startswith("ALREADY APPLIED") and "(lines 5-9)" in no_stamp
    ident = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=stamp(now, 5, 9))   # a correct stamp, identical text: the old 'no change'
    assert ident == no_stamp and "no change" not in ident


def test_already_applied_finds_the_text_a_few_lines_away(tmp_path):
    t = tools(tmp_path)
    t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=stamp(SRC, 5, 9))
    out = t.replace_lines("dialogue.py", 8, 12, new_block(5, 9), expect="deadbe")          # the model's numbers drifted by 3
    assert out.startswith("ALREADY APPLIED") and "(lines 5-9)" in out


def test_already_applied_uses_the_cursor_hint_and_reports_a_finished_window(tmp_path):
    t = tools(tmp_path)
    t.next_hint = lambda rel: 21
    t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=stamp(SRC, 5, 9))
    out = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect="abc123")
    assert "Next line of your window that is still unchanged: 21." in out
    t.next_hint = lambda rel: -1                                                              # the stage window is done
    assert "has no unchanged lines left" in t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect="abc123")
    t.next_hint = lambda rel: 3                                                               # a nonsense hint (before the range): the line after Y
    assert "still unchanged: 10." in t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect="abc123")


def test_a_different_new_text_is_not_called_already_applied(tmp_path):
    t = tools(tmp_path)
    t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=stamp(SRC, 5, 9))
    out = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9, "OTHER"), expect=stamp(SRC, 5, 9))   # stale stamp, new text not in the file
    assert out.startswith("NOT CHANGED: lines 5-9 are not the lines you saw")


def test_a_tiny_block_is_never_taken_for_already_applied(tmp_path):
    t = tools(tmp_path)
    out = t.replace_lines("dialogue.py", 5, 5, "\n", expect="abc123")
    assert out.startswith("NOT CHANGED")


def test_edit_file_already_applied(tmp_path):
    t = tools(tmp_path)
    old = LINES[4]
    new = 'LINE_05 = "the bell rang once, and nobody turned around to look"'
    assert t.edit_file("dialogue.py", [{"search": old, "replace": new}]).startswith("edited")
    out = t.edit_file("dialogue.py", [{"search": old, "replace": new}])                        # the search text is gone now
    assert out.startswith("ALREADY APPLIED: these lines already contain your new text (lines 5-5).")
    assert t.edit_file("dialogue.py", [{"search": old, "replace": new}]) == out
    same = t.edit_file("dialogue.py", [{"search": new, "replace": new}])                       # old 'no change: identical'
    assert same == out
    assert t.edit_file("dialogue.py", [{"search": "nothing like it xyz", "replace": "other text entirely"}]).startswith("edit_file failed")


def test_already_applied_counts_as_a_repeat_for_the_loop_guard_and_is_a_look(tmp_path):
    from agent.codeagent import lookact

    t = tools(tmp_path)
    t.replace_lines("dialogue.py", 5, 9, new_block(5, 9), expect=stamp(SRC, 5, 9))
    args = {"path": "dialogue.py", "start": 5, "end": 9, "new_text": new_block(5, 9), "expect": stamp(SRC, 5, 9)}
    res = t.replace_lines(**args)
    assert lookact.classify("replace_lines", args, res) == "look"
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 30
    for i in range(1, 4):
        eng._loop_guard("replace_lines", args, res)
        assert eng._abort_loop == 0, i
    eng._loop_guard("replace_lines", args, res)
    assert eng._abort_loop >= 4                                                               # the ladder still works


# ── 2. relocate a stale stamp ───────────────────────────────────────────────

def test_shifted_stamp_is_relocated_and_applied(tmp_path):
    t = tools(tmp_path)
    want = stamp(SRC, 20, 22)
    t.replace_lines("dialogue.py", 1, 2, new_block(1, 2) + "EXTRA_A = 1\nEXTRA_B = 2\nEXTRA_C = 3\n", expect=stamp(SRC, 1, 2))   # +3 lines: the block is at 23-25
    out = t.replace_lines("dialogue.py", 20, 22, new_block(20, 22, "FIX"), expect=want)
    assert out.startswith("edited dialogue.py") and "applied at lines 23-25 (the file had shifted by +3 since you read it)" in out
    rows = (tmp_path / "dialogue.py").read_text().splitlines()
    assert rows[22].startswith("LINE_20 = \"FIX") and rows[24].startswith("LINE_22 = \"FIX") and len(rows) == 33
    assert rows[19].startswith("LINE_17 =")                                                  # nothing at the requested range was touched


def test_shifted_negative_and_far_away_are_handled(tmp_path):
    t = tools(tmp_path)
    want = stamp(SRC, 10, 11)
    t.replace_lines("dialogue.py", 1, 4, "", expect=stamp(SRC, 1, 4))                          # -4: the block moved up to 6-7
    out = t.replace_lines("dialogue.py", 10, 11, new_block(10, 11, "UP"), expect=want)
    assert "applied at lines 6-7 (the file had shifted by -4 since you read it)" in out
    far = "\n".join(f"L{i} = {i}" for i in range(1, 300)) + "\nTARGET = 'x'\nTARGET2 = 'y'\n"
    t2 = tools(tmp_path, far, "far.py")
    miss = t2.replace_lines("far.py", 5, 6, "A = 1\nB = 2\n", expect=symbols.range_hash(["TARGET = 'x'", "TARGET2 = 'y'"]))   # 295 lines away: no relocation
    assert miss.startswith("NOT CHANGED") and (tmp_path / "far.py").read_text() == far + ""


def test_ambiguous_shifted_stamp_is_rejected(tmp_path):
    twin = ["alpha line number one of the twin block", "beta line number two of the twin block"]
    src = "\n".join(["head = 1"] * 4 + twin + ["mid = 2"] * 6 + twin + ["tail = 3"] * 4) + "\n"
    t = tools(tmp_path, src, "twin.py")
    out = t.replace_lines("twin.py", 1, 2, "X = 1\nY = 2\n", expect=symbols.range_hash(twin))   # matches at 5-6 and 13-14
    assert out.startswith("NOT CHANGED: lines 1-2 are not the lines you saw")
    assert (tmp_path / "twin.py").read_text() == src


def test_stamp_that_matches_nowhere_is_rejected_as_before(tmp_path):
    t = tools(tmp_path)
    out = t.replace_lines("dialogue.py", 5, 9, new_block(5, 9, "NEW"), expect="abc123")
    assert out.startswith("NOT CHANGED: lines 5-9 are not the lines you saw (#abc123; now #")
    assert "Lines 5-9 are #" in out and (tmp_path / "dialogue.py").read_text() == SRC          # the current lines are still shown


def test_a_correct_stamp_is_never_relocated(tmp_path):
    t = tools(tmp_path)
    out = t.replace_lines("dialogue.py", 5, 6, new_block(5, 6), expect=stamp(SRC, 5, 6))
    assert out.startswith("edited") and "applied at" not in out


# ── 3. syntax-break reject ──────────────────────────────────────────────────

PY = 'def talk(who):\n    line = "hello"\n    if who:\n        line = "hello " + who\n    return line\n\n\nprint(talk("a"))\n'


def test_replace_lines_that_breaks_a_valid_python_file_is_not_applied(tmp_path):
    t = tools(tmp_path, PY, "talk.py")
    out = t.replace_lines("talk.py", 4, 4, '        line = "hello " + (who\n', expect=stamp(PY, 4, 4))
    assert out.startswith("NOT APPLIED: this edit would break the syntax of talk.py (")
    assert "The file is unchanged." in out and "apply_patch" in out and "coding.syntax_reject" in out
    assert "line = \"hello \" + (who" in out                                              # the lines around the error as they would look
    assert (tmp_path / "talk.py").read_text() == PY
    kept = tmp_path / ".scratch" / "unsaved" / "talk.py.1.txt"
    assert kept.read_text() == '        line = "hello " + (who\n' and ".scratch/unsaved/talk.py.1.txt" in out
    t.replace_lines("talk.py", 4, 4, '        line = "hello " + (who\n', expect=stamp(PY, 4, 4))
    assert (tmp_path / ".scratch" / "unsaved" / "talk.py.2.txt").exists()                  # numbered, never overwritten
    assert not t.syntax_bad and not t.dirty                                                 # nothing was written


def test_a_range_that_cuts_a_block_in_half_is_the_typical_case(tmp_path):
    t = tools(tmp_path, PY, "talk.py")
    out = t.replace_lines("talk.py", 3, 4, '    if who:\n', expect=stamp(PY, 3, 4))          # drops the body of the if
    assert out.startswith("NOT APPLIED") and (tmp_path / "talk.py").read_text() == PY


def test_a_good_edit_on_a_valid_file_is_applied(tmp_path):
    t = tools(tmp_path, PY, "talk.py")
    out = t.replace_lines("talk.py", 2, 2, '    line = "hi"\n', expect=stamp(PY, 2, 2))
    assert out.startswith("edited talk.py") and "Syntax OK" in out and '    line = "hi"' in (tmp_path / "talk.py").read_text()


def test_edit_file_and_insert_after_are_gated_too(tmp_path):
    t = tools(tmp_path, PY, "talk.py")
    out = t.edit_file("talk.py", [{"search": '    return line', "replace": '    return (line'}])
    assert out.startswith("NOT APPLIED: this edit would break the syntax of talk.py") and (tmp_path / "talk.py").read_text() == PY
    assert (tmp_path / ".scratch" / "unsaved" / "talk.py.1.txt").read_text() == "    return (line"
    out = t.insert_after("talk.py", "    x = (\n", line=5, expect=stamp(PY, 5, 5))
    assert out.startswith("NOT APPLIED") and (tmp_path / "talk.py").read_text() == PY
    assert (tmp_path / ".scratch" / "unsaved" / "talk.py.2.txt").exists()


def test_an_already_broken_file_is_edited_as_before(tmp_path):
    broken = PY + "def oops(:\n    pass\n"
    t = tools(tmp_path, broken, "talk.py")
    out = t.replace_lines("talk.py", 4, 4, '        line = "hello " + (who\n', expect=stamp(broken, 4, 4))
    assert out.startswith("edited talk.py") and "SYNTAX GATE FAILED" in out
    assert '(who' in (tmp_path / "talk.py").read_text() and not (tmp_path / ".scratch").exists()
    t2 = tools(tmp_path, broken, "talk2.py")
    assert t2.edit_file("talk2.py", [{"search": "    return line", "replace": "    return (line"}]).startswith("edited")


def test_json_and_toml_are_gated_and_prose_files_are_not(tmp_path):
    js = '{\n  "a": 1,\n  "b": 2\n}\n'
    t = tools(tmp_path, js, "d.json")
    out = t.replace_lines("d.json", 2, 2, '  "a": 1,,\n', expect=stamp(js, 2, 2))
    assert out.startswith("NOT APPLIED: this edit would break the syntax of d.json") and (tmp_path / "d.json").read_text() == js
    assert json.loads((tmp_path / "d.json").read_text()) == {"a": 1, "b": 2}
    ok = t.replace_lines("d.json", 2, 2, '  "a": 5,\n', expect=stamp(js, 2, 2))
    assert ok.startswith("edited") and json.loads((tmp_path / "d.json").read_text())["a"] == 5
    toml = 'name = "x"\nvalue = 3\n'
    t = tools(tmp_path, toml, "c.toml")
    out = t.replace_lines("c.toml", 2, 2, "value = = 3\n", expect=stamp(toml, 2, 2))
    assert out.startswith("NOT APPLIED") and (tmp_path / "c.toml").read_text() == toml
    md = "# title\n\nsome ( unbalanced prose [ here\n"
    t = tools(tmp_path, md, "notes.md")
    assert t.replace_lines("notes.md", 3, 3, "another ( unbalanced prose\n", expect=stamp(md, 3, 3)).startswith("edited")


def test_yaml_is_gated_when_pyyaml_is_present(tmp_path):
    import pytest

    pytest.importorskip("yaml")
    y = "a: 1\nb:\n  - x\n  - y\n"
    t = tools(tmp_path, y, "c.yaml")
    out = t.replace_lines("c.yaml", 2, 2, "b: [x\n", expect=stamp(y, 2, 2))
    assert out.startswith("NOT APPLIED") and (tmp_path / "c.yaml").read_text() == y


def test_config_off_keeps_the_old_behaviour(tmp_path):
    t = tools(tmp_path, PY, "talk.py", syntax_reject=False)
    out = t.replace_lines("talk.py", 4, 4, '        line = "hello " + (who\n', expect=stamp(PY, 4, 4))
    assert out.startswith("edited talk.py") and "SYNTAX GATE FAILED" in out


def test_engine_reads_coding_syntax_reject(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    assert eng.tools.syntax_reject is True


def test_write_file_and_apply_patch_keep_their_syntax_gate_behaviour(tmp_path):
    t = tools(tmp_path, PY, "talk.py")
    out = t.write_file("talk.py", 'def broken(:\n    pass\n', overwrite=True)
    assert out.startswith("wrote talk.py") and "SYNTAX GATE FAILED" in out and "def broken(:" in (tmp_path / "talk.py").read_text()
    (tmp_path / "talk.py").write_text(PY)
    patch = "--- a/talk.py\n+++ b/talk.py\n@@ -2,3 +2,3 @@\n     line = \"hello\"\n-    if who:\n+    if (who:\n         line = \"hello \" + who\n"
    out = t.apply_patch(patch)
    assert out.startswith("patched 1 file(s)") and "SYNTAX GATE FAILED" in out and "if (who:" in (tmp_path / "talk.py").read_text()
