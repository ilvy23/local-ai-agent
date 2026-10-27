"""A1 + A2: harness markers are never written (one central MARKERS list), and the look/act classification has no shell, path or
neutral-act bypasses. The model copies what it sees, so every marker the history can show is refused where it would be written."""

from __future__ import annotations

import pytest

from agent.codeagent import context as C, lookact as L, tools as T

from tests.test_codeagent_checks import sup_dir
from tests.test_codeagent_engine import FakeLLM, call, engine, say  # noqa: F401


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


# ── A1: MARKERS ─────────────────────────────────────────────────────────────

SAMPLES = {
    "on_disk": "[12 lines, 340 chars - now on disk]",
    "cut": "x = 'abc...[cut]",
    "chars_omitted": "[… 5000 chars omitted …]",
    "bytes_omitted": "[… 123456 bytes of output omitted …]",
    "truncated": "[truncated - use a smaller range]",
    "more_matches": "[… 12 more matches - narrow the search]",
    "old_read": "[old read of a.py omitted to save context (5 chars).",
    "old_output": "[old tool output omitted: x (5 chars) - run the tool again if you need it]",
    "old_command": "[old command output trimmed]",
    "no_result": "[no result recorded: the session was interrupted while this call ran.",
}


def test_markers_cover_every_emitted_form_and_new_markers_counts():
    names = {n for n, _ in C.MARKERS}
    assert set(SAMPLES) <= names
    for name, text in SAMPLES.items():
        assert C.new_markers(None, text) == [name] or name in C.new_markers(None, text), name
        assert C.new_markers(text, text) == []                       # same count: may stay
        assert name in C.new_markers(text, text + "\n" + text)        # one more: new
    assert C.new_markers("a\nb", "a\nc") == []
    # the real emitters produce text the list recognises
    assert "chars_omitted" in C.new_markers(None, T.cap_output("x" * 20000))
    assert "no_result" in C.new_markers(None, __import__("agent.codeagent.sessionstore", fromlist=["x"]).NO_RESULT)
    assert "cut" in C.new_markers(None, C._clip_text("y" * 50, 10))


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_commit_write_refuses_every_marker_and_allows_an_existing_one(tmp_path, name):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "n.py").write_text("A = 1\n")
    res = eng.execute("write_file", {"path": "n.py", "content": f"A = 1\n# {SAMPLES[name]}\n", "overwrite": True})
    assert ("placeholder" if name == "on_disk" else "harness marker") in res and (repo / "n.py").read_text() == "A = 1\n"
    if name != "on_disk":
        assert name in res
    (repo / "k.py").write_text(f"# {SAMPLES[name]}\nB = 1\n")                  # already in the file: an edit that adds none passes
    eng.execute("read_file", {"path": "k.py"})
    res = eng.execute("replace_lines", {"path": "k.py", "start": 2, "end": 2, "new_text": "B = 2"})
    assert res.startswith("edited k.py") and "B = 2" in (repo / "k.py").read_text()


@pytest.mark.parametrize("tool,extra", [("write_file", {"overwrite": True}), ("append_file", {})])
def test_placeholder_anywhere_refuses_and_writes_nothing(tmp_path, tool, extra):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "d.py").write_text("X = 1\n")
    res = getattr(eng.tools, tool)(path="d.py", content="Y = 2\n[1 lines, 33 chars - now on disk]\nZ = 3", **extra)
    assert res.startswith("error") and "placeholder" in res and (repo / "d.py").read_text() == "X = 1\n"
    res = getattr(eng.tools, tool)(path="d.py", content="Y = 2  # [1 lines, 33 chars - now on disk]", **extra)
    assert res.startswith("error") and (repo / "d.py").read_text() == "X = 1\n"


def test_placeholder_anywhere_refuses_line_edits(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "d.py").write_text("X = 1\nY = 2\n")
    eng.execute("read_file", {"path": "d.py"})
    ph = "Z = 3\n[1 lines, 33 chars - now on disk]\nW = 4"
    for tool, args in (("replace_lines", {"start": 1, "end": 1}), ("insert_after", {"line": 1})):
        res = eng.tools.__getattribute__(tool)(path="d.py", new_text=ph, **args)
        assert res.startswith("error") and "placeholder" in res, tool
    assert (repo / "d.py").read_text() == "X = 1\nY = 2\n"
    assert not hasattr(T, "strip_placeholder")


@pytest.mark.parametrize("cmd,refused", [
    ("cat > f.py <<'EOF'\nx = 1 ...[cut]\nEOF", True),
    ("python3 -c \"open('f','w').write('[3 lines, 9 chars - now on disk]')\"", True),
    ("sed -i 's/a/[truncated - use a smaller range]/' f.py", True),
    ("echo '[old command output trimmed]' | tee f.txt", True),
    ("cp a b # [no result recorded", True),
    ("grep -n '...[cut]' f.py", False),
    ("grep -rn 'now on disk' .", False),
    ("cat f.py | grep '[old read of'", False),
])
def test_run_refuses_a_marker_only_when_the_command_can_write(tmp_path, cmd, refused):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "f.py").write_text("x = 1\n")
    res = eng.tools.run(command=cmd)
    assert res.startswith("error: ") == refused and (not refused or "harness marker" in res)
    assert (repo / "f.py").read_text() == "x = 1\n"


def test_not_verified_prefix_is_shown_but_not_stored(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="X = 1\n")] + [say("Done.")] * 5)
    eng, repo, _ = engine(tmp_path, llm)
    (sup_dir(repo) / "never.sh").write_text("echo nope; exit 1\n")
    res = eng.turn("set X")
    assert res.text.startswith("NOT VERIFIED:")
    last = [m for m in eng.ctx.messages if m["role"] == "assistant"][-1]
    assert last["content"] == "Done." and not any("NOT VERIFIED:" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "assistant")


# ── A2: read-only shell ─────────────────────────────────────────────────────

RO = [
    "cat a.py", "head -n 5 a.py", "sed -n '1,40p' a.py", "sed -n '/def f/,/^def /p' a.py", "grep -n x a.py | head", "rg -n foo src",
    "ls -la | wc -l", "find . -name '*.py' | sort | uniq -c", "git log --oneline -5", "git diff HEAD~1", "git -C . status", "git show HEAD:a.py",
    "git blame a.py", "git ls-files | head", "git grep -n foo", "git rev-parse HEAD", "git branch -a", "git branch --show-current",
    "jq .a x.json", "sort a.txt | uniq", "xxd a.bin | head", "od -c a.bin", "strings a.bin | grep foo", "base64 -d a.b64", "cut -d: -f1 a.txt",
    "tr a-z A-Z < a.txt", "column -t a.txt", "fold -w 80 a.txt", "fd py", "tree -L 2", "realpath a.py", "dirname a/b.py", "basename a/b.py",
    "echo hello", "printf '%s\\n' hi", "grep -n '->' a.py", "cat a.py 2>&1 | head", "ls nope 2>/dev/null",
    "awk '{print $1}' a.txt", "awk -F: '{print $1}' a.txt", "python3 -c \"print(open('f').read())\"", "python -c 'import json; print(json.load(open(\"f\")))'",
    "find . -type f -name '*.py'", "sed -E 's/a/b/' a.py", "diff a b", "echo 'a > b'",
]
ACT = [
    "python3 a.py", "python3 -c \"open('f','w').write('x')\"", "python3 -c \"open('f', 'a')\"", "python3 -c \"from pathlib import Path; Path('f').write_text('x')\"",
    "python3 -c 'import shutil; shutil.rmtree(\"d\")'", "python3 -c 'import os; os.remove(\"f\")'", "python3 -c 'import os; os.system(\"ls\")'",
    "python3 -c 'exec(\"1\")'", "python3 -c \"open('f', mode)\"", "python3 -m pytest", "python3",
    "find . -delete", "find . -name x -exec rm {} \;", "find . -execdir rm {} +", "find . -ok rm {} \;", "find . -fprint out.txt",
    "sed -i 's/a/b/' a.py", "sed -ni 'p' a.py", "sed -i.bak 's/a/b/' a.py", "sed --in-place 's/a/b/' a.py", "sed 's/a/b/w out.txt' a.py",
    "sed -n '1w out.txt' a.py", "sed -e 'w out.txt' a.py", "sed 's/a/b/e' a.py", "sed '1e ls' a.py", "sed -f s.sed a.py",
    "awk 'BEGIN{system(\"ls\")}'", "awk '{print > \"out\"}' a.txt", "awk '{print >> \"out\"}' a.txt", "awk '{print | \"sh\"}' a.txt",
    "sort -o out a.txt", "sort a.txt --output=out", "cat a.py > b.py", "cat a.py >> b.py", "cat a.py | tee b.py", "echo hi > f", "echo $(rm f)", "echo `ls`",
    "cat a.py | xargs rm", "git checkout main", "git reset --hard", "git restore a.py", "git clean -fd", "git commit -m x", "git add .", "git rm a.py",
    "git mv a b", "git stash", "git pull", "git push", "git merge x", "git rebase x", "git apply p.diff", "git cherry-pick x", "git config a b",
    "git branch -d x", "git branch -D x", "git branch -m x y", "git branch newbranch", "base64 -o out a.b64", "base64 a --output=out",
    "cat a.py | grep x && rm a.py", "ls; rm a.py", "cp a b", "mv a b", "rm a", "touch a", "mkdir d", "cat a.py | sed -i s/a/b/ b",
]


@pytest.mark.parametrize("cmd", RO)
def test_readonly_shell_true(cmd):
    assert L.readonly_shell(cmd), cmd


@pytest.mark.parametrize("cmd", ACT)
def test_readonly_shell_false(cmd):
    assert not L.readonly_shell(cmd), cmd


def test_table_is_large_enough():
    assert len(RO) + len(ACT) >= 40


def test_there_is_one_list_of_readonly_commands():
    assert T._FILTERS == L.FILTER_CMDS and L.FILTER_CMDS <= L.READONLY_CMDS
    assert all(L.readonly_shell(f"{c} x") for c in L.FILTER_CMDS if c not in ("sed", "awk"))


def test_paused_run_blocks_a_python_read_but_not_a_python_write(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "f").write_text("hello\n")
    eng._look_streak = L.LOOK_BLOCK_AT
    assert eng.execute("run", {"command": "python3 -c \"print(open('f').read())\""}).startswith("NOT RUN")
    res = eng.execute("run", {"command": "python3 -c \"open('g','w').write('x')\""})
    assert not res.startswith("NOT RUN") and (repo / "g").read_text() == "x"


@pytest.mark.parametrize("name", ["project_note", "add_check", "mark_done"])
def test_neutral_acts_neither_reset_nor_raise_the_look_streak(name):
    la = L.LookAct()
    for _ in range(5):
        la.observe("read_file", {"path": "a.py"}, "text")
    assert L.classify(name, {}, "ok") == "neutral"
    assert la.observe(name, {}, "ok") == [] and la.look_streak == 5
    assert la.observe(name, {}, "NOT CONFIRMED: checks failing") == [] and la.look_streak == 5
    la.pause()
    la.observe(name, {}, "ok")
    assert la.blocked("read_file", {"path": "a.py"})                 # still paused


@pytest.mark.parametrize("target,scratch", [
    (".scratch/x.py", True), ("scratch/x.py", True), ("tmp/x.py", True), (".tmp/x.py", True), ("./tmp/x.py", True),
    ("__pycache__/x.pyc", True), (".pytest_cache/x", True),
    ("tmpl/x.py", False), ("scratchpad/x.py", False), ("src/tmp/x.py", True), ("tmp_utils.py", False), ("src/x.py", False),
])
def test_scratch_is_decided_by_a_whole_path_segment(target, scratch):
    assert L.is_scratch_path(target) is scratch
    assert L.real_edit("write_file", {"path": target}, f"wrote {target}") is (not scratch)
    from agent.codeagent.engine import _real_write
    assert _real_write("write_file", {"path": target}, f"wrote {target}") is (not scratch)
