"""The 'unchanged since step N - already in your context' placeholder of read_file may only be returned while the earlier FULL result is still
in the context (not masked, compacted, from another stage, session or process), was not cut by a read cap, and the file is the same
(inode, mtime_ns, size). Modelled on qwen-code's fileReadCache (full-read flag + history residency + inode/fingerprint). FakeLLM only."""

from __future__ import annotations

import os

import pytest

from agent.codeagent import profile as PR, sessionstore as SS, tools as T
from agent.codeagent.tools import CodeTools

from tests.test_codeagent_context import cm
from tests.test_codeagent_engine import FakeLLM
from tests.test_codeagent_stages import new_engine

UNCHANGED = "unchanged since step"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def make(tmp_path, name="w.py", n=100, **kw):
    (tmp_path / name).write_text("".join(f"line {i:04d} of the text\n" for i in range(1, n + 1)))
    ctx = cm(tmp_path)
    ctx.step = 3
    return CodeTools(tmp_path, PR.detect(tmp_path), ctx, **kw), ctx


# ── what was really shown ───────────────────────────────────────────────────

def test_control_a_whole_file_read_twice_is_a_placeholder_the_second_time(tmp_path):
    tools, _ctx = make(tmp_path)
    assert "line 0100" in tools.read_file("w.py")
    assert UNCHANGED in tools.read_file("w.py")


def test_a_whole_read_cut_by_the_read_cap_is_never_answered_with_the_placeholder(tmp_path):
    tools, _ctx = make(tmp_path, n=400, read_max_chars=2000)
    first = tools.read_file("w.py")
    assert "continue with start=" in first and "line 0400" not in first
    again = tools.read_file("w.py")                                    # the model never saw the later part: a real answer with the pointer
    assert UNCHANGED not in again and "continue with start=" in again


def test_a_whole_read_cut_by_the_line_limit_is_never_answered_with_the_placeholder(tmp_path):
    tools, _ctx = make(tmp_path, n=T.READ_MAX_LINES + 50, read_max_chars=10_000_000)
    assert "continue with start=" in tools.read_file("w.py")
    assert UNCHANGED not in tools.read_file("w.py")


def test_a_range_read_cut_by_the_range_cap_is_never_answered_with_the_placeholder(tmp_path):
    (tmp_path / "w.py").write_text("".join(f"{i:05d} " + "x" * 200 + "\n" for i in range(1, 301)))
    ctx = cm(tmp_path)
    ctx.step = 3
    tools = CodeTools(tmp_path, PR.detect(tmp_path), ctx)
    first = tools.read_file("w.py", start=1, end=300)
    assert "[truncated - use a smaller range]" in first
    assert UNCHANGED not in tools.read_file("w.py", start=1, end=300)


def test_a_range_is_not_a_whole_file_and_a_whole_file_is_not_another_range(tmp_path):
    tools, _ctx = make(tmp_path)
    tools.read_file("w.py", start=1, end=40)
    whole = tools.read_file("w.py")
    assert UNCHANGED not in whole and "line 0100" in whole           # only a part was seen: the whole file is a real read
    tools.read_file("w.py", start=50, end=60)
    assert UNCHANGED not in tools.read_file("w.py", start=50, end=61)
    assert UNCHANGED in tools.read_file("w.py", start=50, end=60)


def test_an_outline_is_never_a_placeholder_and_never_arms_one(tmp_path):
    tools, _ctx = make(tmp_path)
    tools.read_file("w.py", outline=True)
    assert UNCHANGED not in tools.read_file("w.py")
    assert "OUTLINE" in tools.read_file("w.py", outline=True)


# ── the file is the same file ───────────────────────────────────────────────

def test_a_replaced_file_with_the_same_size_and_mtime_is_a_real_read(tmp_path):
    tools, _ctx = make(tmp_path)
    p = tmp_path / "w.py"
    tools.read_file("w.py")
    st = p.stat()
    other = tmp_path / "other.tmp"
    other.write_text(p.read_text().replace("line 0001", "LINE 0001"))
    os.utime(other, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(other, p)                                               # a new inode, identical size and mtime_ns
    assert p.stat().st_size == st.st_size and p.stat().st_mtime_ns == st.st_mtime_ns and p.stat().st_ino != st.st_ino
    assert "LINE 0001" in tools.read_file("w.py")


def test_a_tool_write_invalidates_even_when_size_and_mtime_look_unchanged(tmp_path, monkeypatch):
    tools, _ctx = make(tmp_path)
    p = tmp_path / "w.py"
    tools.read_file("w.py")
    st = p.stat()

    def in_place(path, text):                                          # a coarse clock: same inode, same size, same mtime_ns
        path.write_text(text)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))

    monkeypatch.setattr(T, "atomic_write", in_place)
    tools.replace_lines("w.py", start=1, end=1, new_text="LINE 0001 of the text\n")
    assert p.stat().st_mtime_ns == st.st_mtime_ns and p.stat().st_size == st.st_size
    assert "LINE 0001" in tools.read_file("w.py")


def test_a_shell_command_that_changes_a_big_file_invalidates(tmp_path):
    tools, _ctx = make(tmp_path)
    tools.read_file("w.py")
    tools.run(command="python3 -c \"open('w.py','a').write('more\\n')\"")
    assert "more" in tools.read_file("w.py")


def test_a_command_whose_effects_could_not_be_measured_forgets_the_reads(tmp_path, monkeypatch):
    tools, _ctx = make(tmp_path)
    p = tmp_path / "w.py"
    tools.read_file("w.py")
    st = p.stat()
    monkeypatch.setattr(T, "snapshot_tree", lambda *a, **k: None)      # no snapshot (too many files): nobody knows what the command did
    tools.run(command=f"python3 -c \"import os; p='w.py'; t=open(p).read().replace('0001','9999'); open(p,'w').write(t); "
                      f"os.utime(p, ns=({st.st_atime_ns},{st.st_mtime_ns}))\"")
    assert "9999 of the text" in tools.read_file("w.py")


def test_a_read_only_shell_look_keeps_the_placeholder(tmp_path):
    tools, _ctx = make(tmp_path)
    tools.read_file("w.py")
    tools.run(command="ls")
    assert UNCHANGED in tools.read_file("w.py")


# ── the earlier result is still in the context ──────────────────────────────

def test_masking_and_compaction_make_the_next_read_real(tmp_path):
    tools, ctx = make(tmp_path)
    tools.read_file("w.py")
    ctx._cache_reset()                                                 # what every mask / compaction / fold does
    assert UNCHANGED not in tools.read_file("w.py")
    assert UNCHANGED in tools.read_file("w.py")


def test_a_fresh_stage_context_makes_the_next_read_real(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    (repo / "w.py").write_text("x = 1\n" * 20)
    eng.tools.read_file("w.py")
    assert UNCHANGED in eng.tools.read_file("w.py")
    eng.begin_stage("goal", "VERIFIED STATE\nx", "")
    assert UNCHANGED not in eng.tools.read_file("w.py")


def test_a_new_session_in_the_same_engine_forgets_the_reads(tmp_path):
    from agent.codeagent import session

    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    (repo / "w.py").write_text("x = 1\n" * 20)
    eng.tools.read_file("w.py")
    session.reset_engine(eng)
    assert UNCHANGED not in eng.tools.read_file("w.py")


def test_restoring_a_saved_session_forgets_the_reads(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    (repo / "w.py").write_text("x = 1\n" * 20)
    data = SS.snapshot(eng)
    eng.tools.read_file("w.py")
    assert UNCHANGED in eng.tools.read_file("w.py")
    SS.restore(eng, data)                                              # the restored history holds no such read (or another one)
    assert UNCHANGED not in eng.tools.read_file("w.py")


def test_a_new_engine_never_starts_with_a_cache(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    (repo / "w.py").write_text("x = 1\n" * 20)
    assert eng.tools.read_cache == {} and UNCHANGED not in eng.tools.read_file("w.py")
