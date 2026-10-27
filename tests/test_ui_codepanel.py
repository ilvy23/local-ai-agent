"""CodePanel: pure rendering of the live code-agent state at 40/60/100 columns, unicode and plain."""

import pytest

from agent.ui import codepanel as cp
from tests._screen_code import render

WIDTHS = (40, 60, 100)

SNAP = {"state": "running", "now": "editing game.py - a rather long description of what is going on right now", "step": 7, "budget_now": 40,
        "step_ceiling": 120, "budget_source": "estimate", "ctx_used": 9000, "ctx_limit": 32768, "ctx_percent": 27.5, "tps": 41.234,
        "model": "qwen3.5:27b", "mode": "auto", "elapsed_s": 125.0, "queued": 2}

PLAN = {"source": "headings", "task": "t", "current": 1, "now": "", "dead_ends": 2, "checklist_confirmed": 1, "checklist_total": 2,
        "stages": [
            {"id": 1, "title": "Scaffold the project layout", "status": "done", "budget_used_s": 30, "budget_limit_s": 120, "budget_used_steps": 4,
             "budget_limit_steps": 20, "budget_share": 0.25, "reason": ""},
            {"id": 2, "title": "Implement the game loop and input handling", "status": "active", "budget_used_s": 90, "budget_limit_s": 120,
             "budget_used_steps": 12, "budget_limit_steps": 20, "budget_share": 0.75, "reason": ""},
            {"id": 3, "title": "Tests", "status": "pending", "budget_used_s": 0, "budget_limit_s": 0, "budget_used_steps": 0, "budget_limit_steps": 0,
             "budget_share": 0, "reason": ""},
            {"id": 4, "title": "Docs", "status": "blocked", "budget_used_s": 0, "budget_limit_s": 0, "budget_used_steps": 0, "budget_limit_steps": 0,
             "budget_share": 0, "reason": "needs a human"}],
        "checklist": [{"id": 1, "text": "game starts", "done": True, "evidence": "pytest test_start passed"},
                      {"id": 2, "text": "score is shown", "done": False, "evidence": ""}]}

EVENTS = [
    {"kind": "tool_start", "name": "read_file", "summary": "read game.py", "args": '{"path": "game.py"}'},
    {"kind": "tool_result", "name": "read_file", "summary": "read game.py", "result": "line1\nline2", "seconds": 0.4, "ok": True},
    {"kind": "tool_start", "name": "run_command", "summary": "pytest -q", "args": '{"command": "pytest -q"}'},
    {"kind": "tool_result", "name": "run_command", "summary": "pytest -q", "result": "1 failed", "seconds": 12.5, "ok": False},
    {"kind": "edit", "path": "game.py", "added": 3, "removed": 1, "created": False, "diff": "--- a/game.py\n+++ b/game.py\n@@\n-x\n+y"},
    {"kind": "check", "id": "c1", "check_kind": "tests", "status": "pass", "seconds": 3.0},
    {"kind": "compact", "before": 30000, "after": 9000, "folded": 12},
    {"kind": "loop", "tool": "list_dir", "repeats": 3},
    {"kind": "budget_extend", "to": 60, "steps": 20},
    {"kind": "tool_start", "name": "write_file", "summary": "write notes.md", "args": "{}"},
    {"kind": "now", "text": "ignored"},
    {"kind": "thinking", "text": "ignored"},
]
FILES = [{"path": "game.py", "added": 30, "removed": 4, "created": False},
         {"path": "README.md", "added": 5, "removed": 0, "created": True}]


def _fits(text, width):
    assert all(len(line) <= width for line in text.splitlines()), [ln for ln in text.splitlines() if len(ln) > width]


@pytest.mark.parametrize("plain", [True, False])
@pytest.mark.parametrize("width", WIDTHS)
def test_header(width, plain):
    out = render(cp.render_header(SNAP, width=width, plain=plain), width, plain=plain)
    _fits(out, width)
    assert "editing game.py" in out and "step 7/40" in out and "ctx 28%" in out and "41.2 tok/s" in out
    assert "2:05" in out or "2m 05s" in out
    assert "qwen3.5:27b" in out and "auto" in out and "2 queued" in out
    if width >= 60:
        assert "(estimate)" in out and "max 120" in out
    else:
        assert "(estimate)" not in out


def test_header_states_and_empty():
    assert "paused" in render(cp.render_header({"state": "paused", "now": "paused"}, plain=True), 80)
    assert "idle" in render(cp.render_header(None, plain=True), 80)
    assert "*" in render(cp.render_header({"state": "running", "now": "x"}, plain=True), 80)


@pytest.mark.parametrize("plain", [True, False])
@pytest.mark.parametrize("width", WIDTHS)
def test_plan(width, plain):
    out = render(cp.render_plan(PLAN, width=width, plain=plain), width, plain=plain)
    _fits(out, width)
    assert "PLAN 1/4" in out and "ACCEPTANCE 1/2" in out and "dead ends: 2" in out and "needs a human" in out
    assert "Scaffold" in out and "score is shown" in out
    if plain:
        assert "[x]" in out and "[ ]" in out and "+ 1" in out and "> 2" in out and "x 4" in out
    else:
        assert "✓" in out and "▶" in out and "☑" in out
    if width >= 100:
        assert "90s/120s 12/20 steps (75%)" in out and "pytest test_start passed" in out
    if width == 40:
        assert "75%" in out and "pytest test_start" not in out


def test_plan_empty():
    assert "no plan yet" in render(cp.render_plan(None, plain=True), 40)
    assert "no plan yet" in render(cp.render_plan({"stages": [], "checklist": []}, plain=True), 40)


@pytest.mark.parametrize("plain", [True, False])
@pytest.mark.parametrize("width", WIDTHS)
def test_timeline_collapsed(width, plain):
    out = render(cp.render_timeline(EVENTS, width=width, plain=plain), width, plain=plain)
    _fits(out, width)
    lines = out.splitlines()
    assert sum("read_file" in ln for ln in lines) == 1 and sum("run_command" in ln for ln in lines) == 1   # start+result collapse to one line
    assert any("run_command" in ln and "12s" in ln for ln in lines) and any("read_file" in ln and "0.4s" in ln for ln in lines)
    assert any("write_file" in ln for ln in lines)                      # still running: no duration
    for needle in ("check", "compact", "loop guard", "budget"):
        assert any(needle in ln for ln in lines), needle
    assert "ignored" not in out and "1 failed" not in out
    marks = ("x ", "+ ") if plain else ("✗", "✓")
    assert marks[0] in out and marks[1] in out


def test_timeline_verbose_and_limit():
    out = render(cp.render_timeline(EVENTS, verbose=True, width=100, plain=True), 100)
    assert '"path": "game.py"' in out and "= 1 failed" in out and "+y" in out and "-x" in out
    short = render(cp.render_timeline(EVENTS, width=100, plain=True, limit=2), 100)
    assert len(short.splitlines()) == 2
    assert "no activity yet" in render(cp.render_timeline([], plain=True), 60)


def test_timeline_persisted_tool_events_and_uievents():
    from agent.ui.events import UiEvent

    rows = cp.timeline_rows([{"kind": "tool", "name": "read_file", "seconds": 1.0, "ok": True}])
    assert rows[0][1] == "read_file" and rows[0][3] == "1.0s"
    rows = cp.timeline_rows([UiEvent("tool_result", {"name": "x", "ok": False, "seconds": 2})])
    assert rows[0][0] == "fail"


@pytest.mark.parametrize("plain", [True, False])
@pytest.mark.parametrize("width", WIDTHS)
def test_files_and_diff(width, plain):
    diffs = {"README.md": "--- /dev/null\n+++ b/README.md\n+hello\n+world"}
    out = render(cp.render_files(FILES, diffs, width=width, plain=plain), width, plain=plain)
    _fits(out, width)
    assert "FILES CHANGED 2" in out and "+35 -4" in out and "+30" in out and "-4" in out and "README.md" in out
    assert "+hello" not in out                                    # collapsed by default
    assert "game.py" in out
    out = render(cp.render_files(FILES, diffs, expand=["README.md"], width=width, plain=plain), width, plain=plain)
    assert "+hello" in out and "+world" in out
    assert " A " in out and " M " in out


def test_diff_truncation_and_empty():
    big = "\n".join(f"+line{i}" for i in range(100))
    out = render(cp.render_diff(big, width=60, plain=True, max_lines=10), 60)
    assert "+line9" in out and "+line10" not in out and "90 more lines" in out
    assert "(empty diff)" in render(cp.render_diff("", plain=True), 60)
    assert "+line0" in render(cp.render_diff(big, width=60, plain=False, max_lines=3), 60, plain=False)
    assert "no files changed" in render(cp.render_files([], plain=True), 40)


@pytest.mark.parametrize("plain", [True, False])
@pytest.mark.parametrize("width", WIDTHS)
def test_summary(width, plain):
    res = {"text": "Built the game.\nIt starts.", "stopped": "", "verified": False, "not_verified": ["pytest: 1 failing", "lint not run"],
           "files": ["README.md"]}
    out = render(cp.render_summary(res, {"files": FILES}, width=width, plain=plain), width, plain=plain)
    _fits(out, width)
    assert "DONE" in out and "not verified" in out and "NOT VERIFIED" in out and "lint not run" in out and "FILES CHANGED 2" in out
    assert "Built the game." in out and "next" in out and "fix what is not verified" in out
    stopped = render(cp.render_summary({"text": None, "stopped": "budget", "files": []}, width=width, plain=plain), width, plain=plain)
    assert "STOPPED (budget)" in stopped and "continue" in stopped
    assert "custom" in render(cp.render_summary(res, next_cmd="custom thing", width=100, plain=True), 100)


def test_panel_wrapper_feed_and_render():
    p = cp.CodePanel(width=60, plain=True)
    for e in EVENTS:
        p.feed(e)
    p.update({**SNAP, "plan": PLAN, "files": FILES})
    assert len(p.events) == len(EVENTS) - 2 and "game.py" in p.diffs
    out = render(p.render(), 60)
    assert "editing game.py" in out and "PLAN 1/4" in out and "FILES CHANGED 2" in out and "read_file" in out
    assert "-x" in render(p.diff("game.py"), 60)
    assert "+y" in render(p.files(expand=True), 60) and "+y" not in render(p.files(), 60)
    assert "…" in render(cp.render_files([{"path": "a/" * 30 + "x.py", "added": 1, "removed": 0}], width=40, plain=False), 40)
    p.finish({"text": "ok", "stopped": "", "verified": True, "not_verified": [], "files": ["a"]})
    out = render(p.render(), 60)
    assert "DONE" in out and "read_file" not in out


def test_no_ansi_in_plain_mode():
    from agent.ui import style

    buf = __import__("io").StringIO()
    style.make_console(60, plain=True, file=buf).print(cp.render_diff("+a\n-b", plain=True))
    style.make_console(60, plain=True, file=buf).print(cp.render_plan(PLAN, width=60, plain=True))
    assert "\x1b" not in buf.getvalue()
