"""Duplicate-block protection of the edit tools. Seen live (a long run on a prose-heavy dialogue.py): replace_lines got an end line
that was too small, so the new, longer `ELLA_OPEN = {...}` block was inserted in front of the old one; Python accepted the second
assignment, every test stayed green, the file kept growing. Two deterministic checks: static.duplicate_bindings (module-level name bound
twice, final gate) and edits.duplicate_block (a block just written that is still in the file elsewhere, a line on the edit result)."""

from __future__ import annotations

import time
from pathlib import Path

from agent.codeagent import edits, static as ST

from tests.test_codeagent_engine import FakeLLM, call, engine, say  # noqa: F401
from tests.test_codeagent_static import _t

DATA = Path(__file__).parent / "data" / "duplicate_incident"      # a synthetic file before / after a run that went wrong (stored as .txt: not a module, not collected)


def entries(n, tag="a"):
    return [f'    "{tag}{i}": "this is the line number {i} of the {tag} block",' for i in range(n)]


def block(name, n, tag="a"):
    return [f"{name} = {{", *entries(n, tag), "}"]


def build(*parts):
    return "\n".join(parts) + "\n"


# ── 1: duplicate top-level bindings ──────────────────────────────────────────

def test_a_data_block_bound_twice_names_both_lines():
    src = build("import os", "", *block("ELLA_OPEN", 12), "", "def f():", "    return 1", "", *block("ELLA_OPEN", 15, "b"), "", "print(ELLA_OPEN)")
    found = ST.duplicate_bindings(src)
    assert found == [("ELLA_OPEN", 3, 21)]
    probs = ST.check("/x", "m.py", src)
    assert any("ELLA_OPEN is defined twice at module level (line 3 and line 21)" in p and "end line was too small" in p
               and "Remove the stale copy" in p for p in probs)


def test_a_def_followed_by_an_assignment_of_the_same_name_counts_but_two_defs_stay_with_duplicate_defs():
    assert ST.duplicate_bindings("def f():\n    return 1\n\nf = {'a': 1, 'b': 2}\n") == [("f", 1, 4)]
    assert ST.duplicate_bindings("def f():\n    return 1\n\ndef f():\n    return 2\n") == []
    assert ST.duplicate_defs("def f():\n    return 1\n\ndef f():\n    return 2\n") == [("f", 1, 4)]


def test_exempt_conditional_self_referencing_augmented_overload_setter_dunder_and_read_between():
    cases = [
        "import sys\nif sys.platform == 'x':\n    A = {1: 2, 3: 4}\nelse:\n    A = {5: 6}\n",
        "try:\n    B = {1: 2}\nexcept ImportError:\n    B = {3: 4}\n",
        "C = {1: 2, 3: 4}\nC = {**C, 5: 6}\n",
        "D = [1, 2, 3]\nD = D + [4]\n",
        "E = [1, 2, 3]\nE += [4]\nE = [9, 9]\n",
        "F = {'a': 1}\nF.update({'b': 2})\nF = {'c': 3}\n",
        "G = {'a': 1}\nprint(G)\nG = {'b': 2}\n",
        "H = {'a': 1}\ndef use():\n    return H\nH = {'b': 2}\n",
        "__all__ = ['a']\n__all__ = ['a', 'b']\n_ = {1: 2}\n_ = {3: 4}\n__version__ = '1'\n__version__ = '2'\n",
        "from typing import overload\n@overload\ndef o(x: int) -> int: ...\n@overload\ndef o(x: str) -> str: ...\ndef o(x):\n    return x\n",
        "class C:\n    @property\n    def v(self):\n        return 1\n    @v.setter\n    def v(self, x):\n        pass\n",
        "X = None\nX = {'a': 1}\n",                                           # a constant placeholder is not a dead data block
        "J = {'a': 1}\nfor J in range(3):\n    pass\n",
        "K = {'a': 1}\ndef set_k():\n    global K\n    K = {'b': 2}\nK = {'c': 3}\n",
    ]
    for src in cases:
        assert ST.duplicate_bindings(src) == [], src


def test_dynamic_modules_and_syntax_errors_are_not_checked():
    assert ST.duplicate_bindings("from os import *\nA = {1: 2}\nA = {3: 4}\n") == []
    assert ST.duplicate_bindings("A = {\n") == []


def test_a_correct_block_replacement_has_no_finding():
    src = build(*block("ELLA_OPEN", 12), "", "def f():", "    return 1")
    assert ST.duplicate_bindings(src) == []


def test_real_incident_files():
    before = (DATA / "dialogue_before.txt").read_text()
    after = (DATA / "dialogue_corrupted.txt").read_text()
    assert ST.duplicate_bindings(before) == []
    # the corrupted file holds three annotated data blocks twice (ELLA_OPEN, ELLA_CHAPTER and ELLA_TREE are the same damage)
    assert ST.duplicate_bindings(after) == [("ELLA_CHAPTER", 123, 217), ("ELLA_OPEN", 146, 240), ("ELLA_TREE", 161, 255)]
    assert not any("ELLA_OPEN" in p for p in ST.check("/x", "dialogue.py", before))
    assert any("ELLA_OPEN is defined twice at module level (line 146 and line 240)" in p for p in ST.check("/x", "dialogue.py", after))
    # the block detector sees one edit (one region): the pure 94-line insertion of the run (difflib: insert orig 119 -> new 119-212)
    rows, bad = before.split("\n"), after.split("\n")
    single = "\n".join(rows[:118] + bad[118:212] + rows[118:])
    w = edits.duplicate_block(before, single)
    # the model re-wrote the block, so only runs of identical lines survive in the old copy: the longest one is reported
    assert w is not None and w[0] >= 8 and 119 <= w[1] <= w[2] <= 212 and w[3] > 212, w
    assert "duplicate lines" in edits.duplicate_text(*w)
    assert edits.duplicate_block(before, before) is None


# ── 2: duplicate block warning ───────────────────────────────────────────────

def lines_of(n, tag="x"):
    return [f"sentence {tag}{i} says something unique about the harbour at night" for i in range(n)]


def test_duplicate_block_in_any_text_file_is_found_with_both_ranges():
    old = lines_of(30) + ["", "# end"]
    new = lines_of(30)[:5] + lines_of(10, "n") + lines_of(30)[5:] + ["", "# end"]           # 10 new lines inserted: no duplicate of them
    assert edits.duplicate_block("\n".join(old) + "\n", "\n".join(new) + "\n") is None
    dup = old[:20] + lines_of(30)[10:20] + old[20:]                                            # lines 11-20 typed a second time
    got = edits.duplicate_block("\n".join(old) + "\n", "\n".join(dup) + "\n")
    assert got is not None
    n, a, b, c, d = got
    assert (n, b - a + 1) == (10, 10)
    assert edits.duplicate_text(*got) == (f"WARNING: the {n} lines you just wrote at {a}-{b} duplicate lines {c}-{d} that are still in the file. "
                                          "If you meant to replace that block, your end line was too small; fix it now.")


def test_replace_lines_with_too_small_end_warns_once_on_markdown_json_and_text(tmp_path):
    for name in ("notes.md", "data.json", "story.txt"):
        body = [f'  "key{i}": "a long enough value for key number {i}",' if name.endswith("json") else f"Paragraph {i} of the story is long enough to count."
                for i in range(45)]
        t = _t(tmp_path, "\n".join(body) + "\n", name)
        t.read_file(name)
        # rewrite lines 10-20 but stop at 12: the old tail 13-20 stays behind the new text
        out = t.replace_lines(name, 10, 12, "\n".join(body[9:20]), expect=None)
        assert out.count("WARNING: the ") == 1, out
        assert "duplicate lines" in out and "your end line was too small" in out
        text = (tmp_path / name).read_text()
        assert len(text.splitlines()) == 45 + 11 - 3


def test_a_correct_replacement_gives_no_warning(tmp_path):
    body = lines_of(60)
    t = _t(tmp_path, "\n".join(body) + "\n", "doc.txt")
    t.read_file("doc.txt")
    out = t.replace_lines("doc.txt", 10, 25, "\n".join(lines_of(20, "fresh")))
    assert "WARNING" not in out and "NOTE: your replacement" not in out


def test_no_false_positives_on_short_repeated_stanzas_and_punctuation():
    boiler = ['"",', "}", "),", "]", "},", "{", "  ),", "});"] * 20
    old = lines_of(40) + boiler
    new = lines_of(40) + boiler + ['"",', "}", "),", "]", "},", "{", "  ),", "});", '"",', "}", "),"]
    assert edits.duplicate_block("\n".join(old) + "\n", "\n".join(new) + "\n") is None
    stanza = ["    if x:", "        do_a()", "        do_b()"]
    base = lines_of(40) + stanza * 2
    assert edits.duplicate_block("\n".join(base) + "\n", "\n".join(base + stanza) + "\n") is None    # one more 3-line stanza: far below 8 lines
    short = [f"a{i}" for i in range(20)]                                                              # 60%+ lines shorter than 12 chars
    assert edits.duplicate_block("\n".join(lines_of(40) + short) + "\n", "\n".join(lines_of(40) + short + short) + "\n") is None


def test_small_files_and_pure_deletes_and_moves_are_skipped():
    small = lines_of(20)
    assert edits.duplicate_block("\n".join(small[:10]) + "\n", "\n".join(small[:10] + small[:10]) + "\n") is None           # < 40 lines
    big = lines_of(60)
    assert edits.duplicate_block("\n".join(big + big[:20]) + "\n", "\n".join(big) + "\n") is None                            # delete
    moved = big[30:45] + big[:30] + big[45:]
    assert edits.duplicate_block("\n".join(big) + "\n", "\n".join(moved) + "\n") is None                                    # a move


def test_edit_file_insert_after_append_and_write_file_warn_too(tmp_path):
    body = lines_of(50)
    t = _t(tmp_path, "\n".join(body) + "\n", "doc.txt")
    chunk = "\n".join(body[5:15])
    out = t.edit_file("doc.txt", [{"search": body[49], "replace": body[49] + "\n" + chunk}])
    assert out.count("WARNING: the 10 lines") == 1 and "duplicate lines 6-15" in out
    t = _t(tmp_path, "\n".join(body) + "\n", "doc2.txt")
    t.read_file("doc2.txt")
    out = t.insert_after("doc2.txt", chunk, line=30)
    assert out.count("WARNING: the 10 lines") == 1
    t = _t(tmp_path, "\n".join(body) + "\n", "doc3.txt")
    assert t.append_file("doc3.txt", chunk).count("WARNING: the 10 lines") == 1
    t = _t(tmp_path, "\n".join(body) + "\n", "doc4.txt")
    assert t.write_file("doc4.txt", "\n".join(body + body[5:15]) + "\n", overwrite=True).count("WARNING: the 10 lines") == 1
    t = _t(tmp_path, "x = 1\n", "doc5.txt")
    assert "WARNING" not in t.write_file("doc5.txt", "\n".join(lines_of(50)) + "\n", overwrite=True)


def test_the_python_incident_shape_through_replace_lines(tmp_path):
    old = block("ELLA_OPEN", 20)
    src = build("import os", "", *old, "", "def f():", "    return ELLA_OPEN", "", *[f"# filler line {i} of the module" for i in range(30)])
    t = _t(tmp_path, src, "dialogue.py")
    t.read_file("dialogue.py")
    new = block("ELLA_OPEN", 20)
    new[10] = '    "changed": "this entry was rewritten by the model",'
    out = t.replace_lines("dialogue.py", 1, 2, "\n".join(["import os", "", *new, ""]))      # range too small: the old dict stays behind
    assert "ELLA_OPEN is defined twice at module level (line 3 and line 25)" in out and out.count("STATIC CHECK") == 1
    assert out.count("WARNING: the ") == 1 and "your end line was too small" in out
    assert "dialogue.py" in t.static_bad
    # fixing it (deleting the stale copy) clears the finding
    t.read_file("dialogue.py")
    out = t.replace_lines("dialogue.py", 25, 46, "")
    assert "defined twice" not in out and "WARNING" not in out and "dialogue.py" not in t.static_bad


# ── 3: size sanity ───────────────────────────────────────────────────────────

BODY = [f"statement number {i} does some distinct work here" for i in range(30)]


def replaced(tmp_path, name, s, e, new):
    t = _t(tmp_path, "\n".join(BODY) + "\n", name)
    t.read_file(name)
    return t.replace_lines(name, s, e, "\n".join(new))


def test_replacement_that_retypes_the_line_after_the_range_gets_a_note(tmp_path):
    fresh = [f"fresh line {i} that replaces the short range" for i in range(8)]
    out = replaced(tmp_path, "a.txt", 10, 10, fresh + [BODY[10]])             # 9 lines for 1, BODY[10] is line 11 = the line after the range
    assert out.count("NOTE: your replacement re-includes the line that follows the range (statement number 10 does some distinct work here)") == 1
    assert "check the end line." in out


def test_no_size_note_for_long_ranges_small_factors_or_trivial_lines(tmp_path):
    fresh = [f"fresh line {i} that replaces the short range" for i in range(8)]
    assert "NOTE: your replacement" not in replaced(tmp_path, "b.txt", 1, 12, fresh + [BODY[12]])            # range of 12 lines
    assert "NOTE: your replacement" not in replaced(tmp_path, "c.txt", 10, 10, ["a", BODY[10]])              # only 2x
    assert "NOTE: your replacement" not in replaced(tmp_path, "d.txt", 10, 10, fresh + [BODY[12]])           # not the line after
    t = _t(tmp_path, "\n".join(BODY[:10] + ["}"] + BODY[10:]) + "\n", "e.txt")
    t.read_file("e.txt")
    assert "NOTE: your replacement" not in t.replace_lines("e.txt", 10, 10, "\n".join(fresh + ["}"]))         # a bare brace


# ── 4: the final gate ────────────────────────────────────────────────────────

DUP = build(*block("OPEN", 12), "", "def f():", "    return 1", "", *block("OPEN", 14, "b"))


def test_the_final_gate_sends_a_duplicate_binding_back_and_then_labels_not_verified(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([call("write_file", path="game.py", content=DUP), *[say("Done.")] * 6]))
    res = eng.turn("add the data")
    notes = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "static check finds errors" in str(m["content"])]
    assert len(notes) == 2 and "OPEN is defined twice at module level" in notes[0]          # the static gate cap (2) applies
    assert res.text.startswith("NOT VERIFIED: ") and "duplicate definition: game.py (OPEN)" in res.text
    assert res.verified is False


def test_fixing_the_duplicate_clears_the_gate(tmp_path):
    fixed = build(*block("OPEN", 14, "b"), "", "def f():", "    return 1")
    eng, repo, _ = engine(tmp_path, FakeLLM([call("write_file", path="game.py", content=DUP), say("Done."),
                                             call("write_file", path="game.py", content=fixed), say("Fixed.")]))
    res = eng.turn("add the data")
    assert res.text == "Fixed." and eng.tools.static_bad == {} and not res.not_verified


# ── performance ──────────────────────────────────────────────────────────────

def test_a_5000_line_file_is_checked_in_under_100_ms():
    old = [f"line {i} of a very long generated file with unique content {i * 7}" for i in range(5000)]
    new = old[:2500] + [f"fresh text number {i} that was newly written here" for i in range(40)] + old[2500:]
    a, b = "\n".join(old) + "\n", "\n".join(new) + "\n"
    t0 = time.perf_counter()
    assert edits.duplicate_block(a, b) is None
    dup = old[:2500] + old[100:140] + old[2500:]
    assert edits.duplicate_block(a, "\n".join(dup) + "\n") is not None
    assert (time.perf_counter() - t0) / 2 < 0.5   # generous: the full suite runs next to a model server
    src = "\n".join(f"V{i} = {{'a': {i}}}" for i in range(5000)) + "\n"
    t0 = time.perf_counter()
    assert ST.duplicate_bindings(src) == []
    assert time.perf_counter() - t0 < 0.5
