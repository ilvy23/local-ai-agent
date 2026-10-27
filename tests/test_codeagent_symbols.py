"""D1: edits that need no verbatim old text - show_symbol / where_used / replace_lines / insert_after - and the stale-line
protection (a range hash or the file hash of the last numbered view must match, else the current lines come back)."""

from __future__ import annotations

from agent.codeagent import profile as PR, symbols
from agent.codeagent.context import ContextManager
from agent.codeagent.eventlog import EventLog
from agent.codeagent.tools import CodeTools

from tests.test_codeagent_engine import FakeLLM, call, engine, say  # noqa: F401

SRC = '''"""Game."""
import os

TABLE = {
    "a": 1,
    "b": 2,
}


class Room:
    def enter(self):
        return "in"

    @property
    def name(self):
        return "room"


def start():
    return Room().enter()
'''


def tools(tmp_path, src=SRC, name="game.py"):
    (tmp_path / name).write_text(src)
    ctx = ContextManager(None, {"models": {"background": "m"}}, 32768, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    return CodeTools(tmp_path, PR.detect(tmp_path), ctx)


def test_find_symbol_gives_one_definition_with_decorators_and_qualified_names():
    span, _ = symbols.find_symbol("g.py", SRC, "Room.name")
    assert (span.start, span.end, span.kind) == (14, 16, "def")              # the decorator line is included
    span, _ = symbols.find_symbol("g.py", SRC, "TABLE")
    assert (span.start, span.end, span.kind) == (4, 7, "assign")
    span, _ = symbols.find_symbol("g.py", SRC, "enter")                         # unique tail name
    assert span.name == "Room.enter"
    span, cands = symbols.find_symbol("g.py", SRC, "nothing_like_it")
    assert span is None and cands == []
    js = "function a() {\n  return 1;\n}\nfunction b() {\n  if (x) {\n    y();\n  }\n}\n"
    span, _ = symbols.find_symbol("x.js", js, "b")
    assert (span.start, span.end) == (4, 8)


def test_show_symbol_numbers_lines_and_gives_a_hash(tmp_path):
    t = tools(tmp_path)
    out = t.show_symbol("start", "game.py")
    assert "lines 19-20" in out and "   19| def start():" in out and "#" + symbols.range_hash(SRC.splitlines()[18:20]) in out
    assert "replace_lines" in out
    assert "lines 19-20" in t.show_symbol("start")                               # path optional when the name is unique
    assert "no symbol" in t.show_symbol("missing", "game.py")


def test_replace_lines_with_the_shown_hash_changes_only_those_lines(tmp_path):
    t = tools(tmp_path)
    t.show_symbol("start", "game.py")
    h = symbols.range_hash(SRC.splitlines()[18:20])
    out = t.replace_lines("game.py", 19, 20, "def start():\n    return 'go'\n", expect=h)
    assert out.startswith("edited game.py: lines 19-20 replaced by 2 line(s)") and "Syntax OK" in out
    text = (tmp_path / "game.py").read_text()
    assert text.endswith("def start():\n    return 'go'\n") and "class Room" in text and "game.py" in t.dirty


def test_a_stale_hash_changes_nothing_and_returns_the_current_lines(tmp_path):
    t = tools(tmp_path)
    out = t.replace_lines("game.py", 19, 20, "x = 1\n", expect="abc123")
    assert out.startswith("NOT CHANGED") and "19| def start():" in out and (tmp_path / "game.py").read_text() == SRC
    # without expect: the file must be unchanged since the last numbered view
    t.show_symbol("start", "game.py")
    (tmp_path / "game.py").write_text("# new first line\n" + SRC)                # someone else edited it: the numbers moved
    out = t.replace_lines("game.py", 19, 20, "x = 1\n")
    assert out.startswith("NOT CHANGED") and "changed since" in out
    out = t.replace_lines("game.py", 20, 21, "def start():\n    return 2\n")       # the reply showed the current lines: now it works
    assert out.startswith("edited") and (tmp_path / "game.py").read_text().endswith("return 2\n")


def test_never_seen_line_numbers_are_shown_first(tmp_path):
    t = tools(tmp_path)
    out = t.replace_lines("game.py", 2, 2, "import sys")
    assert out.startswith("NOT CHANGED") and "not seen line numbers" in out
    assert t.replace_lines("game.py", 2, 2, "import sys").startswith("edited")


def test_replace_lines_syntax_gate_and_copied_numbers_are_stripped(tmp_path):
    t = tools(tmp_path)
    t.read_file("game.py", 1, 20)
    out = t.replace_lines("game.py", 19, 20, "   19| def start():\n   20|     return 3\n")
    assert "Syntax OK" in out and "def start():\n    return 3" in (tmp_path / "game.py").read_text()
    out = t.replace_lines("game.py", 19, 19, "def start(:")                       # changed on purpose: a valid file is no longer broken by default
    assert out.startswith("NOT APPLIED") and "game.py" not in t.syntax_bad and "def start():\n    return 3" in (tmp_path / "game.py").read_text()
    t.syntax_reject = False                                                     # coding.syntax_reject: false keeps the old behaviour
    out = t.replace_lines("game.py", 19, 19, "def start(:", expect=symbols.range_hash(["def start():"]))
    assert "SYNTAX GATE FAILED" in out and "game.py" in t.syntax_bad


def test_insert_after_a_symbol_and_at_the_top(tmp_path):
    t = tools(tmp_path)
    out = t.insert_after("game.py", "\n\ndef stop():\n    return 0", symbol="start")
    assert out.startswith("edited game.py: inserted") and (tmp_path / "game.py").read_text().endswith("def stop():\n    return 0\n")
    out = t.insert_after("game.py", "# header", line=0)
    assert out.startswith("edited") and (tmp_path / "game.py").read_text().startswith("# header\n")
    (tmp_path / "game.py").write_text("# moved\n" + (tmp_path / "game.py").read_text())   # changed outside: the numbers it saw are stale
    assert t.insert_after("game.py", "x = 1", line=5).startswith("NOT CHANGED")
    assert "no single definition" in t.insert_after("game.py", "x", symbol="nope")


def test_where_used_marks_definitions(tmp_path):
    t = tools(tmp_path)
    (tmp_path / "main.py").write_text("from game import start\nstart()\n")
    out = t.where_used("start")
    assert "3 line(s) in 2 file(s)" in out and out.splitlines()[1].endswith("[definition]")
    assert "not used anywhere" in t.where_used("zzz_nothing")
    assert t.line_views.get("main.py")                                            # line numbers were shown: replace_lines may use them


def test_engine_runs_the_new_tools_and_counts_them_as_look_and_write(tmp_path):
    show = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "show_symbol", "arguments": {"name": "start", "path": "game.py"}}}]}
    llm = FakeLLM([show,
                   call("replace_lines", path="game.py", start=19, end=20, new_text="def start():\n    return 1\n"),
                   say("Changed start().")])
    eng, repo, _out = engine(tmp_path, llm)
    (repo / "game.py").write_text(SRC)
    res = eng.turn("make start return 1")
    assert res.files == ["game.py"] and (repo / "game.py").read_text().endswith("return 1\n")
    stored = [m["content"] for m in eng.ctx.messages if m["role"] == "assistant"]
    assert '"replace_lines"' in stored[1] and eng._look_streak == 0
