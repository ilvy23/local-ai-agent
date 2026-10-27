"""File changes made through the `run` tool (shell edits, redirects, rename-based writes) are accounted for like tool writes:
ledger, dirty set, syntax gate, progress; read-only files survive a command; move/delete keep the ledger honest."""

from __future__ import annotations

import os
import stat
import time

import pytest

from agent.codeagent import lookact, tools as T

from tests.test_codeagent_engine import FakeLLM, engine


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def _eng(tmp_path):
    eng, repo, _out = engine(tmp_path, FakeLLM([]))
    return eng, repo


def test_a_shell_edit_is_recorded_like_a_tool_write(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "main.py").write_text("x = 1\n")
    (repo / "lena.py").write_text("y = 1\n")
    res = eng.tools.run("sed -i 's/1/2/' main.py lena.py && echo done > new.txt")
    assert "NOTE: this command changed lena.py, main.py, new.txt (recorded)." in res
    led = eng.ctx.ledger.files
    assert led["main.py"].action == "run" and led["lena.py"].action == "run" and "new.txt" in led
    assert {"main.py", "lena.py", "new.txt"} <= eng.tools.dirty
    assert str(repo / "main.py") in eng.tools.changes.changed()


def test_a_command_that_changes_nothing_adds_no_note(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "a.py").write_text("x = 1\n")
    assert "NOTE" not in eng.tools.run("python3 -c 'print(1)'")
    assert "NOTE" not in eng.tools.run("touch -d '2001-01-01' /dev/null; ls")
    assert eng.ctx.ledger.files == {}


def test_a_shell_edit_that_breaks_a_file_is_reported(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "ok.py").write_text("x = 1\n")
    eng.tools.run("true")
    res = eng.tools.run("printf 'def f(:\\n' > ok.py")
    assert "NOTE: this command changed ok.py (recorded)." in res and "SYNTAX GATE FAILED" in res
    assert "ok.py" in eng.tools.syntax_bad
    eng.tools.run("printf 'x = 2\\n' > ok.py")
    assert "ok.py" not in eng.tools.syntax_bad


def test_a_shell_delete_leaves_the_ledger(tmp_path):
    eng, repo = _eng(tmp_path)
    eng.tools.write_file("gone.py", "x = 1\n")
    assert "gone.py" in eng.ctx.ledger.files and "gone.py" in eng.tools.dirty
    res = eng.tools.run("rm gone.py")
    assert "gone.py" in res and "gone.py" not in eng.ctx.ledger.files and "gone.py" not in eng.tools.dirty
    assert "MISSING" not in eng.ctx.ledger.render(repo)


def test_skipped_dirs_are_not_tracked(tmp_path):
    eng, repo = _eng(tmp_path)
    res = eng.tools.run("mkdir -p .scratch node_modules/x __pycache__ && echo a > .scratch/t.py && echo b > node_modules/x/i.js && echo c > __pycache__/z.py")
    assert "NOTE" not in res and eng.ctx.ledger.files == {}


def test_progress_counts_a_shell_edit_but_not_a_scratch_copy_or_a_broken_file(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "tmp").mkdir()
    eng._run_call({"name": "run", "arguments": {"command": "echo 'x = 1' > main.py"}})
    assert eng._progress == 1
    eng._run_call({"name": "run", "arguments": {"command": "echo 'x = 1' > tmp/t.py"}})
    assert eng._progress == 1                                       # a scratch folder is not real work
    eng._run_call({"name": "run", "arguments": {"command": "echo 'def (' > main.py"}})
    assert eng._progress == 1                                       # left a syntax error: not progress
    eng._run_call({"name": "run", "arguments": {"command": "echo ok"}})
    assert eng._progress == 1


def test_lookact_sees_a_shell_edit_as_a_real_edit():
    assert lookact.real_edit("run", {"command": "sed -i s/a/b/ x.py"}, "exit code 0  (0.0s)  (no output)\nNOTE: this command changed x.py (recorded).")
    assert not lookact.real_edit("run", {"command": "echo"}, "exit code 0  (0.0s)\nNOTE: this command changed tmp/x.py (recorded).")
    assert not lookact.real_edit("run", {"command": "ls"}, "exit code 0  (0.0s)\nx.py")
    la = lookact.LookAct()
    la.edit_gap, la.edit_nudges, la.probing = 7, 1, True
    la.observe("run", {"command": "sed -i s/a/b/ x.py"}, "exit code 0  (0.0s)\nNOTE: this command changed x.py (recorded).")
    assert (la.edit_gap, la.edit_nudges, la.probing) == (0, 0, False)


# ── read-only files ─────────────────────────────────────────────────────────

def _ro(repo, name="ro.txt", text="precious\n"):
    f = repo / name
    f.write_text(text)
    f.chmod(0o444)
    return f


def test_a_rename_replace_of_a_read_only_file_is_undone(tmp_path):
    eng, repo = _eng(tmp_path)
    f = _ro(repo)
    res = eng.tools.run("python3 -c \"import os;open('x.tmp','w').write('z');os.replace('x.tmp','ro.txt')\"")
    assert f.read_text() == "precious\n" and not os.access(f, os.W_OK)
    assert stat.S_IMODE(f.stat().st_mode) == 0o444
    assert "NOTE: this command tried to change the read-only file ro.txt; it was restored." in res
    assert "ro.txt" not in eng.ctx.ledger.files and "ro.txt" not in eng.tools.dirty


def test_a_deleted_or_chmodded_and_edited_read_only_file_is_restored(tmp_path):
    eng, repo = _eng(tmp_path)
    a, b = _ro(repo, "a.dat", "A\n"), _ro(repo, "b.dat", "B\n")
    res = eng.tools.run("rm -f a.dat; chmod u+w b.dat; echo changed > b.dat")
    assert a.read_text() == "A\n" and b.read_text() == "B\n"
    assert stat.S_IMODE(a.stat().st_mode) == 0o444 and stat.S_IMODE(b.stat().st_mode) == 0o444
    assert "read-only file a.dat; it was restored" in res and "read-only file b.dat; it was restored" in res
    assert "a.dat" not in eng.tools.dirty and "b.dat" not in eng.tools.dirty


def test_a_read_only_file_that_was_only_read_is_left_alone(tmp_path):
    eng, repo = _eng(tmp_path)
    f = _ro(repo)
    before = f.stat().st_mtime_ns
    assert "NOTE" not in eng.tools.run("cat ro.txt > /dev/null; python3 -c \"open('ro.txt').read()\"")
    assert f.stat().st_mtime_ns == before


# ── move / delete keep the ledger honest ────────────────────────────────────

def test_move_file_moves_the_ledger_entry(tmp_path):
    eng, repo = _eng(tmp_path)
    eng.tools.write_file("old.py", "def f():\n    return 1\n")
    entry = eng.ctx.ledger.files["old.py"]
    eng.tools.move_file("old.py", "pkg/new.py")
    led = eng.ctx.ledger.files
    assert "old.py" not in led and led["pkg/new.py"].names == entry.names and led["pkg/new.py"].action == "moved"
    assert "MISSING" not in eng.ctx.ledger.render(repo)
    assert "pkg/new.py" in eng.tools.dirty and "old.py" not in eng.tools.dirty


def test_move_of_a_directory_moves_every_entry_below_it(tmp_path):
    eng, repo = _eng(tmp_path)
    eng.tools.write_file("src/a.py", "x = 1\n")
    eng.tools.write_file("src/b.py", "y = 1\n")
    eng.tools.move_file("src", "lib")
    assert set(eng.ctx.ledger.files) == {"lib/a.py", "lib/b.py"}
    assert "MISSING" not in eng.ctx.ledger.render(repo)


def test_delete_file_drops_the_ledger_entry_and_its_problems(tmp_path):
    eng, repo = _eng(tmp_path)
    eng.tools.write_file("bad.py", "def (:\n")
    assert "bad.py" in eng.tools.syntax_bad
    eng.tools.delete_file("bad.py")
    assert "bad.py" not in eng.ctx.ledger.files and "bad.py" not in eng.tools.syntax_bad
    assert "MISSING" not in eng.ctx.ledger.render(repo)


# ── cost ────────────────────────────────────────────────────────────────────

def test_the_snapshot_of_2000_files_is_cheap(tmp_path):
    for d in range(20):
        (tmp_path / f"d{d}").mkdir()
        for i in range(100):
            (tmp_path / f"d{d}" / f"f{i}.py").write_text("x = 1\n")
    t0 = time.perf_counter()
    snap = T.snapshot_tree(tmp_path)
    assert time.perf_counter() - t0 < 0.05 and len(snap[0]) == 2000


def test_the_snapshot_is_skipped_above_the_cap(tmp_path, monkeypatch):
    eng, repo = _eng(tmp_path)
    for i in range(10):
        (repo / f"f{i}.txt").write_text("x")
    monkeypatch.setattr(T, "SNAPSHOT_CAP", 5)
    assert T.snapshot_tree(repo) is None
    assert "NOTE" not in eng.tools.run("echo hi > f0.txt")            # silently untracked


def test_a_read_only_command_takes_no_snapshot(tmp_path, monkeypatch):
    eng, repo = _eng(tmp_path)
    seen = []
    real = T.snapshot_tree
    monkeypatch.setattr(T, "snapshot_tree", lambda *a, **k: seen.append(1) or real(*a, **k))
    eng.tools.run("ls | head -3")
    eng.tools.run("git log --oneline")
    assert seen == []
    eng.tools.run("echo hi > x.txt")
    assert seen                                                         # a real command does


# ── files the agent owns (for the rm policy) ──────────────────────────────────────────────────────

def test_files_created_by_a_run_or_a_tool_write_are_owned(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "pre.json").write_text("{}")
    eng.tools.run("echo [] > todos.json && echo y >> pre.json")
    eng.tools.write_file("main.py", "x = 1\n")
    assert eng.tools.owned.get("todos.json") == "run" and eng.tools.owned.get("main.py") == "write"
    assert "pre.json" not in eng.tools.owned                      # modified, not created
    eng.tools.run("rm todos.json")
    assert "todos.json" not in eng.tools.owned


def test_the_owned_set_is_capped(tmp_path):
    eng, _repo = _eng(tmp_path)
    for i in range(T.MAX_OWNED + 20):
        eng.tools._own(f"f{i}.json", "run")
    assert len(eng.tools.owned) == T.MAX_OWNED and "f0.json" not in eng.tools.owned and f"f{T.MAX_OWNED + 19}.json" in eng.tools.owned
