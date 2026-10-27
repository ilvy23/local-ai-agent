import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from agent import plan as planmod
from agent import projectmem, toolargs, undo
from agent.tools.registry import Tool


# ------------------------------------------------------------------------------------- toolargs

def _tool(handler=None, props=None, required=("path",)):
    return Tool(name="t", description="d", handler=handler or (lambda path, **k: "ok"),
                parameters={"type": "object", "properties": props or {"path": {"type": "string"}, "n": {"type": "integer"},
                                                                     "flag": {"type": "boolean"}, "items": {"type": "array"},
                                                                     "mode": {"type": "string", "enum": ["a", "b"]}},
                            "required": list(required)})


def test_toolargs_coerces_unambiguous_mistakes():
    fixed, err = toolargs.validate(_tool(), {"path": "/x", "n": "5", "flag": "true", "items": '["a","b"]'})
    assert err is None and fixed == {"path": "/x", "n": 5, "flag": True, "items": ["a", "b"]}
    fixed, err = toolargs.validate(_tool(), {"path": 123})
    assert err is None and fixed["path"] == "123"


def test_toolargs_precise_errors():
    _, err = toolargs.validate(_tool(), {})
    assert err.startswith("Error: invalid arguments for tool 't': missing required argument path") and "path (string, required)" in err
    _, err = toolargs.validate(_tool(), {"path": "/x", "n": "many"})
    assert "argument 'n' must be integer, got string" in err
    _, err = toolargs.validate(_tool(), {"path": "/x", "mode": "c"})
    assert "must be one of ['a', 'b']" in err
    strict = _tool(handler=lambda path: "ok")                     # no **kwargs: unknown arguments are errors
    _, err = toolargs.validate(strict, {"path": "/x", "bogus": 1})
    assert "unknown argument 'bogus'" in err
    _, err = toolargs.validate(_tool(), "not a dict")
    assert err.startswith("Error:")
    lenient_fixed, err = toolargs.validate(_tool(), {"path": "/x", "extra": 1})
    assert err is None and lenient_fixed["extra"] == 1             # handler takes **kwargs: passed through
    schemaless = Tool(name="s", description="d", handler=lambda **k: "ok", parameters={})
    assert toolargs.validate(schemaless, {"anything": 1}) == ({"anything": 1}, None)


# ----------------------------------------------------------------------------------------- undo

def test_undo_restores_edits_and_trashes_new_files(tmp_path, monkeypatch):
    monkeypatch.setattr(undo, "TRASH_DIR", tmp_path / "trash")
    monkeypatch.setattr(undo.shutil, "which", lambda name: None)         # force the plain move fallback
    a, b = tmp_path / "a.py", tmp_path / "new.py"
    a.write_text("old\n")
    a.write_text("new\n")
    b.write_text("fresh\n")
    stack = undo.UndoStack()
    stack.push({str(a): ("old\n", "new\n"), str(b): (None, "fresh\n")}, checkpoint="refs/agent/checkpoints/x")
    assert "2 file(s)" in stack.describe()[0] and "refs/agent/checkpoints/x" in stack.describe()[0]
    out = stack.undo()
    assert a.read_text() == "old\n" and not b.exists() and any("moved to trash" in line for line in out)
    assert list((tmp_path / "trash").rglob("new.py"))
    assert stack.undo() == ["nothing to undo"]


def test_undo_leaves_files_changed_afterwards_alone(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("user edit\n")
    stack = undo.UndoStack()
    stack.push({str(f): ("old\n", "agent edit\n")})
    assert "changed after" in stack.undo()[0] and f.read_text() == "user edit\n"


def test_git_checkpoint_snapshots_without_touching_the_repo(tmp_path):
    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "-q", "-m", "init")
    (tmp_path / "keep.txt").write_text("v1\n")
    (tmp_path / "untracked.txt").write_text("u\n")
    status_before = git("status", "--porcelain")
    ref = undo.git_checkpoint(str(tmp_path / "keep.txt"))
    assert ref and ref.startswith("refs/agent/checkpoints/")
    assert git("status", "--porcelain") == status_before                  # index and tree untouched
    assert "keep.txt" in git("ls-tree", "-r", "--name-only", ref) and "untracked.txt" in git("ls-tree", "-r", "--name-only", ref)
    assert undo.git_checkpoint(str(tmp_path.parent / "nowhere" / "x.py")) is None


# ----------------------------------------------------------------------------------------- plan

def test_should_create_only_for_substantial_build_tasks():
    long_task = "Refactor ~/Downloads/proj/rock_paper_scissors.py into a small package with game logic, leaderboard storage and a terminal UI, then add a Lizard-Spock mode and write pytest tests."
    assert planmod.should_create(long_task)
    assert not planmod.should_create("hi")
    assert not planmod.should_create("what is the weather like in Berlin today and tomorrow, and the weekend?")
    assert not planmod.should_create("/auto-/home/x/y")
    assert planmod.should_create("Please do these things:\n- first thing to do now\n- second thing to do\n- third thing to do")


def test_parse_items_json_and_bullets():
    assert planmod.parse_items('Sure! ["a file exists", "tests pass"]') == ["a file exists", "tests pass"]
    assert planmod.parse_items("1. one thing\n2. other thing") == ["one thing", "other thing"]
    assert planmod.parse_items("nothing useful") == []
    assert len(planmod.parse_items(json.dumps([f"i{i}" for i in range(20)]))) == planmod.MAX_ITEMS


def test_plan_mark_done_needs_evidence_and_tracks_progress():
    p = planmod.Plan("t", [planmod.Item(1, "x exists"), planmod.Item(2, "tests pass")])
    assert p.mark_done(9, "pytest: 3 passed").startswith("Error: there is no item 9")
    assert p.mark_done(1, "ok").startswith("Error: give concrete evidence")
    assert "1 of 2" in p.mark_done("1", "ls shows /p/x.py exists")
    assert "[x] x exists" in p.system_message() and "[ ] tests pass" in p.system_message()
    assert "2. tests pass" in p.reminder() and "1. x exists" not in p.reminder()
    assert "All items are confirmed" in p.mark_done(2, "pytest -q: 3 passed in 0.1s") and not p.open_items()


def test_generate_uses_the_background_model_and_survives_failures():
    class LLM:
        def chat(self, messages, model, stream=True, **kw):
            assert kw.get("background") is True and model == "bg"
            return iter(['["package split", "tests pass", "old scores.json loads"]'])

    plan = planmod.generate(LLM(), {"models": {"background": "bg"}}, "big task")
    assert [i.text for i in plan.items] == ["package split", "tests pass", "old scores.json loads"]

    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("down")

    assert planmod.generate(Broken(), {"models": {"background": "bg"}}, "t") is None

    class OneItem:
        def chat(self, *a, **k):
            return iter(['["only one"]'])

    assert planmod.generate(OneItem(), {"models": {"background": "bg"}}, "t") is None


# ------------------------------------------------------------------------------------ projectmem

def test_project_memory_scopes_dedupes_and_limits(tmp_path):
    proj = tmp_path / "w" / "proj"
    (proj / "tests").mkdir(parents=True)
    key = projectmem.project_key(str(proj / "pkg" / "a.py"))
    assert key == str(proj.resolve()) and projectmem.project_key(str(proj)) == key
    assert projectmem.project_key(str(tmp_path / "w" / "plain" / "a.py")) is None
    mem = projectmem.ProjectMemory(sqlite3.connect(":memory:"))
    assert mem.add(key, "Tests run with: uv run --with pytest python -m pytest -q")
    assert not mem.add(key, "Tests run with: uv run --with pytest python -m pytest -q")      # duplicate
    assert not mem.add(key, "short")
    mem.add("/other/project", "A fact about another project entirely")
    assert [n for _i, n in mem.notes(key)] == ["Tests run with: uv run --with pytest python -m pytest -q"]
    msg = mem.system_message(key)
    assert "PROJECT MEMORY for proj" in msg and "uv run" in msg
    assert mem.system_message("/nothing") is None
    assert mem.clear(key) == 1 and mem.notes(key) == []


def test_project_note_tool_needs_an_active_project(tmp_path):
    import io
    from rich.console import Console

    mem = projectmem.ProjectMemory(sqlite3.connect(":memory:"))
    active = projectmem.ActiveProject()
    tool = projectmem.make_tool(mem, active, Console(file=io.StringIO()))
    assert tool.handler(note="Always run the linter first").startswith("Error: no active project")
    proj = tmp_path / "a" / "proj"
    (proj / ".git").mkdir(parents=True)
    active.set_from(str(proj / "x.py"))
    assert "Saved for the project proj" in tool.handler(note="Always run the linter first")
    assert tool.handler(note="Always run the linter first").startswith("Not saved")
