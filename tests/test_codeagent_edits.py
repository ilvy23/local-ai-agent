from __future__ import annotations

import pytest

from agent.codeagent import edits as E

SRC = '''import os


def alpha(x):
    """Doubles."""
    return x * 2


class Box:
    def put(self, item):
        self.items.append(item)
        return len(self.items)


MAX_ITEMS = 5
'''


def test_syntax_gate_points_at_the_line_and_hints_on_quotes():
    bad = 'x = "he said "hi" to me"\nprint(x)\n'
    msg = E.check_syntax("a.py", bad)
    assert msg and "line 1" in msg and ">>" in msg and "Hint" not in (E.check_syntax("a.py", "def f(:\n  pass\n")[:5] or "")
    open_str = E.check_syntax("a.py", 'print("abc)\n')
    assert open_str and "Hint" in open_str
    assert E.check_syntax("a.py", SRC) is None
    assert E.check_syntax("a.json", '{"a": 1,}') and E.check_syntax("a.json", '{"a": 1}') is None
    assert E.check_syntax("a.toml", "a = [") and E.check_syntax("a.txt", "anything (") is None
    assert E.check_syntax("a.yaml", "a: [1, 2") and E.check_syntax("a.yaml", "a: 1\n") is None


def test_replace_exact_tolerant_and_ambiguous():
    r = E.replace_once(SRC, "return x * 2", "return x * 3")
    assert r.how == "exact" and "x * 3" in r.text
    # whitespace-tolerant: wrong indentation and trailing spaces in the model's search block
    r = E.replace_once(SRC, "self.items.append(item)  \nreturn len(self.items)", "self.items.insert(0, item)\nreturn 0")
    assert r.text and r.how == "whitespace-tolerant"
    assert "        self.items.insert(0, item)\n        return 0\n" in r.text          # re-indented to the file
    amb = E.replace_once("a = 1\nb = 1\n", " = 1", " = 2")
    assert amb.text is None and "2 places" in amb.error
    assert E.replace_once("a = 1\nb = 1\n", " = 1", " = 2", replace_all=True).text == "a = 2\nb = 2\n"
    missing = E.replace_once(SRC, "def alpha(y):", "def alpha(z):")
    assert missing.text is None and "closest match starts at line" in missing.error and "def alpha(x)" in missing.error


def test_apply_edits_is_all_or_nothing():
    res = E.apply_edits(SRC, [{"search": "MAX_ITEMS = 5", "replace": "MAX_ITEMS = 9"}, {"search": "nope", "replace": "x"}])
    assert res.text is None and "edit 2/2" in res.error
    ok = E.apply_edits(SRC, [{"search": "MAX_ITEMS = 5", "replace": "MAX_ITEMS = 9"}, {"search": "import os", "replace": "import os\nimport sys"}])
    assert "MAX_ITEMS = 9" in ok.text and "import sys" in ok.text


def test_unified_diff_applies_by_content_not_line_numbers():
    patch = """--- a/m.py
+++ b/m.py
@@ -99,3 +99,3 @@
 def alpha(x):
     \"\"\"Doubles.\"\"\"
-    return x * 2
+    return x * 4
"""
    files = E.parse_patch(patch)
    assert files[0][0] == "m.py"
    res = E.apply_hunks(SRC, files[0][1])
    assert "x * 4" in res.text and res.text.count("def alpha") == 1
    bad = E.apply_hunks(SRC, [[" nothing like this", "-gone", "+new"]])
    assert bad.text is None and "not found" in bad.error


def test_outline_and_repo_map(tmp_path):
    out = E.outline("m.py", SRC)
    assert any("def alpha(x)" in o and "Doubles" in o for o in out) and any("class Box" in o for o in out)
    assert any("def put(self, item)" in o for o in out) and any("MAX_ITEMS" in o for o in out)
    (tmp_path / "m.py").write_text(SRC)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("function nope() {}")
    (tmp_path / "web.js").write_text("export function render() {}\nclass App {}\n")
    m = E.repo_map(tmp_path, 2000)
    assert "m.py" in m and "def alpha" in m and "render" in m and "nope" not in m
    assert len(E.repo_map(tmp_path, 60)) <= 90                       # budget respected (plus the truncation note)
    assert E.numbered("a\nb\nc\n", 2, 3) == "    2| b\n    3| c"


def test_patches_in_the_shapes_local_models_really_send():
    unified = "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    assert E.parse_patch(unified)[0][0] == "m.py"
    codex = "*** Begin Patch\n*** Update File: m.py\n@@\n def f():\n-    return 1\n+    return 2\n*** End Patch\n"
    files = E.parse_patch(codex)
    assert files[0][0] == "m.py" and E.apply_hunks("def f():\n    return 1\n", files[0][1]).text == "def f():\n    return 2\n"
    headerless = "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    assert E.parse_patch(headerless) == []                                        # no file named -> nothing to apply...
    assert E.parse_patch(headerless, default_path="m.py")[0][0] == "m.py"         # ...unless path=... is given
    gitdiff = "diff --git a/m.py b/m.py\nindex 1..2 100644\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert [f[0] for f in E.parse_patch(gitdiff)] == ["m.py"]


# ── tolerant levels after exact and whitespace: typographic variants, block anchor (clean-room version of opencode's replacer chain) ──

PROSE = '''def intro():
    print("It’s a “quiet” night — the lamp flickers…")
    print("Nothing moves.")
    return 1
'''


def test_curly_quotes_in_the_file_straight_in_the_search():
    r = E.replace_once(PROSE, "It's a \"quiet\" night", "It is a \"loud\" night")
    assert r.text and r.how == E.HOW_TYPO == "matched ignoring typographic quotes/dashes"
    assert 'print("It is a "loud" night — the lamp' in r.text                  # the rest of the line is untouched, the replacement is verbatim


def test_straight_quotes_in_the_file_curly_in_the_search():
    src = 'say("It\'s a \\"quiet\\" night")\n'
    r = E.replace_once(src, "It’s a \\“quiet\\” night", "Hello")
    assert r.text == 'say("Hello")\n' and r.how == E.HOW_TYPO


def test_em_dash_ellipsis_and_nbsp_are_folded():
    assert E.replace_once(PROSE, "night - the lamp", "night, the lamp").how == E.HOW_TYPO
    r = E.replace_once(PROSE, "lamp flickers...", "lamp burns.")
    assert r.how == E.HOW_TYPO and "lamp burns.\")" in r.text
    r = E.replace_once(PROSE, "Nothing moves.", "Silence.")                         # the no-break space: still found, whichever level reports it
    assert r.text and "Silence." in r.text
    r = E.replace_once("x = 'a b'\n", "'a b'", "'c'")
    assert r.how == E.HOW_TYPO and r.text == "x = 'c'\n"
    assert E.replace_once("a – b\n", "a — b", "a + b").text == "a + b\n"          # en dash in the file, em dash in the search


def test_the_replacement_is_written_as_given_not_folded():
    r = E.replace_once(PROSE, "night - the lamp", "night — the “lamp”")
    assert "night — the “lamp” flickers" in r.text


def test_an_ellipsis_is_never_cut_in_half():
    assert E.replace_once("wait… what\n", "wait.. what", "x").text is None


def test_typographic_folding_and_an_indentation_shift_together():
    src = "def f():\n    if ok:\n        say(\"Don’t go\")\n"
    r = E.replace_once(src, "if ok:\nsay(\"Don't go\")", "if ok:\n    stay()")
    assert r.how == E.HOW_TYPO and "        stay()" in r.text


def test_two_typographic_candidates_are_refused_with_line_numbers():
    src = "a = ‘x’\nb = 1\nc = ‘x’\n"
    r = E.replace_once(src, "'x'", "'y'")
    assert r.text is None and "2 places" in r.error and "lines 1, 3" in r.error
    assert E.replace_once(src, "'x'", "'y'", replace_all=True).text is None             # no fuzzy replace_all


def test_exact_ambiguity_names_the_lines():
    r = E.replace_once("a = 1\nb = 1\n", " = 1", " = 2")
    assert r.text is None and "2 places" in r.error and "lines 1, 2" in r.error


BLOCK = '''def total(items):
    result = 0
    for item in items:
        result += item.price * item.qty
    return result


def other():
    pass
'''


def test_block_anchor_with_a_changed_middle_line():
    search = "def total(items):\n    result = 0\n    for it in items:\n        result += it.price * it.qty\n    return result"
    r = E.replace_once(BLOCK, search, "def total(items):\n    return sum(i.price * i.qty for i in items)")
    assert r.text and r.how.startswith(E.HOW_ANCHOR)
    assert r.text.startswith("def total(items):\n    return sum(") and "def other():" in r.text and "result += item" not in r.text


def test_block_anchor_needs_three_lines_and_a_similar_middle():
    assert E.replace_once(BLOCK, "def total(items):\n    return result", "x").text is None                    # 2 lines: never fuzzy
    far = "def total(items):\n    zzzzzzzzzzzzzz()\n    qqqqqqqqqqqqqqqqqq()\n    return result"
    assert E.replace_once(BLOCK, far, "x").text is None                                                  # middle too different
    assert E.replace_once(BLOCK, "def total(items):\n    result = 0\n    for item in items:\n        result += 1\n    return resultt", "x").text is None  # last line must match


def test_block_anchor_with_two_candidates_is_refused():
    src = BLOCK + "\n\n" + BLOCK.replace("price", "cost")
    search = "def total(items):\n    result = 0\n    for it in items:\n        result += it.value\n    return result"
    r = E.replace_once(src, search, "x")
    assert r.text is None and "2 places" in r.error and "lines 1, 12" in r.error


def test_uniform_indentation_shift_is_still_matched_and_reindented():
    r = E.replace_once(BLOCK, "        result = 0\n        for item in items:", "        total = 0\n        for item in items:")
    assert r.text and r.how == "whitespace-tolerant" and "    total = 0\n    for item in items:\n" in r.text


def test_nothing_fuzzy_matches_still_reports_the_closest_lines():
    r = E.replace_once(BLOCK, "def total(things):\n    return 5", "x")
    assert r.text is None and "closest match starts at line 1" in r.error and "def total(items):" in r.error


# ── through edit_file: the result names the level, markers stay refused, hash-stamped line edits are a separate path ──

def _tools(tmp_path, src, name="story.py"):
    from agent.codeagent import profile as PR
    from agent.codeagent.context import ContextManager
    from agent.codeagent.eventlog import EventLog
    from agent.codeagent.tools import CodeTools

    (tmp_path / name).write_text(src, encoding="utf-8")
    ctx = ContextManager(None, {"models": {"background": "m"}}, 32768, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    return CodeTools(tmp_path, PR.detect(tmp_path), ctx)


def test_edit_file_result_says_which_level_matched(tmp_path):
    t = _tools(tmp_path, PROSE)
    out = t.edit_file("story.py", search="It's a \"quiet\" night", replace="It is a night")
    assert out.startswith("edited story.py") and "(matched ignoring typographic quotes/dashes)" in out
    assert "It is a night — the lamp" in (tmp_path / "story.py").read_text(encoding="utf-8")


def test_edit_file_reports_the_anchor_level(tmp_path):
    t = _tools(tmp_path, BLOCK)
    out = t.edit_file("story.py", search="def total(items):\n    result = 0\n    for it in items:\n        result += it.price\n    return result", replace="def total(items):\n    return 0")
    assert "matched by its first and last line" in out and "similarity" in out


def test_markers_and_placeholders_are_refused_also_on_a_fuzzy_match(tmp_path):
    t = _tools(tmp_path, PROSE)
    for bad in ("[12 lines, 340 chars - now on disk]", "x...[cut]", "[… 7 more matches - narrow the search]"):
        with pytest.raises(ValueError, match="harness marker"):                       # engine.execute turns this refusal into the tool answer
            t.edit_file("story.py", search="It's a \"quiet\" night", replace=bad)
        assert (tmp_path / "story.py").read_text(encoding="utf-8") == PROSE, bad


def test_hash_stamped_line_range_edits_do_not_use_the_fuzzy_levels(tmp_path):
    from agent.codeagent import symbols

    t = _tools(tmp_path, PROSE)
    t.read_file("story.py")
    h = symbols.range_hash(PROSE.splitlines()[1:2])
    out = t.replace_lines("story.py", 2, 2, '    print("fresh")\n', expect=h)
    assert out.startswith("edited story.py: lines 2-2 replaced") and "typographic" not in out
    assert t.replace_lines("story.py", 2, 2, "x = 1\n", expect="abc123").startswith("NOT CHANGED")
