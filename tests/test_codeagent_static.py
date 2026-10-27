"""Item 4 (B4): static checks on every Python write and at the final gate - undefined names (the missing `_advance_chapter`),
duplicate definitions, imports of missing names/modules inside the project - without false alarms on dynamic modules."""

from __future__ import annotations

from agent.codeagent import static as ST

from tests.test_codeagent_engine import FakeLLM, call, engine, say  # noqa: F401


def test_the_missing_advance_chapter_is_reported_with_its_line():
    src = "def examine(s):\n    s.seen = True\n    _advance_chapter(s)\n\n\ndef ok():\n    return len([1])\n"
    assert ST.undefined_names(src) == [("_advance_chapter", 3)]


def test_scopes_globals_classes_and_comprehensions_are_understood():
    src = ("import os\nfrom typing import TYPE_CHECKING\nX = 1\n\n\ndef outer():\n    y = 2\n    def inner():\n        return y + X\n"
           "    return [z * y for z in range(3)], inner\n\n\ndef setter():\n    global LATE\n    LATE = 3\n\n\ndef use():\n    return LATE, os.sep\n\n\n"
           "class A:\n    k = 1\n    def m(self, a: 'Unknown') -> 'Nope':\n        return self.k, __class__, a\n\n\n"
           "try:\n    import json\nexcept ImportError:\n    json = None\nfor i in range(2):\n    pass\nprint(i, json, __file__)\n")
    assert ST.undefined_names(src) == []


def test_annotations_and_dynamic_modules_are_not_false_alarms():
    assert ST.undefined_names("from __future__ import annotations\ndef f(x: Missing) -> Other:\n    return x\n") == []
    assert ST.undefined_names("from os import *\nprint(path)\n") == []
    assert ST.undefined_names("globals()['a'] = 1\nprint(a)\n") == []
    assert ST.undefined_names("def f(:\n") == []                                   # a syntax error is the syntax gate's job


def test_method_called_without_self_is_undefined():
    src = "class G:\n    def a(self):\n        return helper()\n\n    def helper(self):\n        return 1\n"
    assert ST.undefined_names(src) == [("helper", 3)]


def test_duplicate_definitions_but_not_property_setters_or_overloads():
    src = ("def f():\n    return 1\n\n\ndef f():\n    return 2\n\n\nclass C:\n    @property\n    def v(self):\n        return 1\n\n"
           "    @v.setter\n    def v(self, x):\n        pass\n\n    def m(self):\n        pass\n\n    def m(self):\n        pass\n")
    assert ST.duplicate_defs(src) == [("f", 1, 5), ("C.m", 18, 21)]


def test_project_imports_missing_names_and_modules(tmp_path):
    pkg = tmp_path / "game"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "story.py").write_text("def chapter():\n    pass\nTITLE = 'x'\n")
    main = tmp_path / "main.py"
    src = "from game.story import chapter, ending\nfrom game import nothere\nimport game.missing\nimport json\nfrom os import path\n"
    probs = ST.import_problems(tmp_path, main, src)
    assert any("does not define 'ending'" in p for p in probs) and any("nothere" in p for p in probs)
    assert any("game/missing.py" in p for p in probs) and len(probs) == 3
    (pkg / "util.py").write_text("from .story import TITLE, gone\n")
    assert ST.import_problems(tmp_path, pkg / "util.py", (pkg / "util.py").read_text()) == [
        "line 1: `from .story import gone`: game/story.py does not define 'gone'"]


def test_write_result_reports_static_problems_and_the_final_gate_sends_back(tmp_path):
    bad = "def examine():\n    _advance_chapter()\n"
    llm = FakeLLM([call("write_file", path="game.py", content=bad),
                   say("Done."),
                   call("write_file", path="game.py", content=bad + "\n\ndef _advance_chapter():\n    pass\n"),
                   say("Done, defined it.")])
    eng, repo, _ = engine(tmp_path, llm)
    res = eng.turn("add examine")
    first = [m["content"] for m in eng.ctx.messages if m["role"] == "tool"][0]
    assert "STATIC CHECK" in first and "undefined name '_advance_chapter'" in first and "line 2" in first
    assert any("static check finds errors" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "system")
    assert res.text == "Done, defined it." and eng.tools.static_bad == {}


def test_edits_elsewhere_are_swept_at_the_final_gate(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.execute("write_file", {"path": "a.py", "content": "def f():\n    return g()\n\n\ndef g():\n    return 1\n"})
    assert eng.tools.static_bad == {}
    (repo / "a.py").write_text("def f():\n    return g()\n")                     # g removed behind the tool's back
    eng._static_sweep()
    assert "a.py" in eng.tools.static_bad


# ── C: the write note shows only NEW problems (Claude Code's "Found N new diagnostic issues"), earlier ones in one line ──────────

OLD = "def a():\n    return missing_one()\n\n\ndef b():\n    return 1\n"


def _t(tmp_path, src=OLD, name="m.py"):
    from agent.codeagent import profile as PR
    from agent.codeagent.context import ContextManager
    from agent.codeagent.eventlog import EventLog
    from agent.codeagent.tools import CodeTools

    (tmp_path / name).write_text(src)
    ctx = ContextManager(None, {"models": {"background": "m"}}, 32768, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    return CodeTools(tmp_path, PR.detect(tmp_path), ctx)


def test_the_first_write_shows_every_problem(tmp_path):
    t = _t(tmp_path, "x = 1\n")
    out = t.write_file("m.py", OLD, overwrite=True)
    assert "undefined name 'missing_one'" in out and "earlier issue" not in out


def test_a_problem_from_before_the_edit_is_not_repeated_after_an_unrelated_edit(tmp_path):
    t = _t(tmp_path)                                                                 # the file already has the problem on disk
    out = t.edit_file("m.py", search="return 1", replace="return 2")
    assert out.startswith("edited m.py") and "undefined name" not in out and "STATIC CHECK" not in out
    assert "1 earlier issue in this file unchanged" in out
    assert "m.py" in t.static_bad                                                    # the final gate still knows the whole list


def test_a_line_shift_does_not_make_an_old_problem_new(tmp_path):
    t = _t(tmp_path)
    out = t.edit_file("m.py", search="def a():", replace="import os\nimport sys\n\n\ndef a():")
    assert "undefined name" not in out and "1 earlier issue in this file unchanged" in out
    assert "line 6" in t.static_bad["m.py"][0]                                       # it really moved


def test_a_new_problem_is_shown_and_the_old_one_is_summarised(tmp_path):
    t = _t(tmp_path)
    out = t.edit_file("m.py", search="return 1", replace="return other_missing()")
    assert "undefined name 'other_missing'" in out and "missing_one" not in out
    assert "1 earlier issue in this file unchanged" in out


def test_fixing_a_problem_removes_it_from_the_baseline(tmp_path):
    t = _t(tmp_path)
    t.edit_file("m.py", search="return 1", replace="return 2")
    out = t.edit_file("m.py", search="missing_one()", replace="1")
    assert "STATIC CHECK" not in out and "earlier issue" not in out and "m.py" not in t.static_bad
    out = t.edit_file("m.py", search="return 1\n", replace="return missing_one()\n")           # the same message comes back: new again
    assert "undefined name 'missing_one'" in out and "earlier issue" not in out


def test_a_syntax_error_is_always_shown_whatever_the_baseline(tmp_path):
    t = _t(tmp_path)
    t.syntax_reject = False                                   # changed on purpose: edit_file refuses to break a valid file by default (test_codeagent_editproto.py)
    t.edit_file("m.py", search="return 1", replace="return 2")
    out = t.edit_file("m.py", search="return 2", replace="return (2")
    assert "SYNTAX GATE FAILED" in out and "m.py" in t.syntax_bad


def test_chunked_files_compare_with_the_last_checked_state(tmp_path):
    t = _t(tmp_path, "x = 1\n")
    t.write_file("m.py", "def a():\n    return helper()\n", overwrite=True)                  # problem seen once
    out = t.append_file("m.py", "def b():\n    return 2\n" * 1, final=True)
    assert "undefined name 'helper'" not in out and "1 earlier issue in this file unchanged" in out


def test_signatures_ignore_line_numbers_but_not_content():
    text = "a = 1\nb = nope()\n"
    sig = ST.signatures(["line 2: undefined name 'nope' (NameError when this line runs)"], text)
    shifted = ST.signatures(["line 5: undefined name 'nope' (NameError when this line runs)"], "\n\n\na = 1\nb = nope()\n")
    other = ST.signatures(["line 2: undefined name 'nope' (NameError when this line runs)"], "a = 1\nc = nope()\n")
    assert sig == shifted and sig != other
    assert ST.note([], 2) == "\n2 earlier issues in this file unchanged" and ST.note([]) == ""
