"""Saved coding sessions: many per folder, resumed in a NEW engine exactly as they were (messages, ledger, plan,
settings), repaired after a kill, robust against damaged files, legacy format migrated, stale locks impossible."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console

from agent.codeagent import policy as P, session as S, sessionstore as SS
from agent.codeagent.engine import CodeEngine
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_engine import CFG, FakeLLM, call, say

TASK = ("Write a small python module a.py that prints hi when run, and make sure that running it works. Keep the code "
        "simple and readable, add a module docstring, and verify the program by running it once at the end.")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def new_engine(tmp_path, llm, name="repo"):
    """A brand-new engine object, as a new process would build it."""
    repo = tmp_path / name
    repo.mkdir(exist_ok=True)
    P.save_repo(repo, mode="auto")
    out = io.StringIO()
    eng = CodeEngine(repo, CFG, llm, Console(file=out, width=140, force_terminal=False), interactive=False,
                     events=EventLog("t", tmp_path / "ev.jsonl"))
    return eng, repo, out


def first_turn(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content='"""Says hi."""\nprint("hi")\n'),
                   call("run", command="python3 a.py"),
                   call("mark_done", ids=[1, 2], evidence="python3 a.py -> exit code 0, printed hi"),
                   say("Wrote a.py; `python3 a.py` prints hi.")],
                  background=['["a.py exists with a docstring", "python3 a.py prints hi"]'])
    eng, repo, out = new_engine(tmp_path, llm)
    eng.session_id = SS.new_id()
    eng.start()
    eng.on_step = lambda: S.save_session(eng)
    res = eng.turn(TASK)
    S.save_session(eng)
    return eng, repo, res


def test_a_session_resumes_in_a_new_engine_exactly_and_the_next_turn_works(tmp_path):
    eng, repo, res = first_turn(tmp_path)
    assert res.stopped == "" and eng.plan.plan and eng.plan.plan.confirmed() == 2
    eng.num_ctx = eng.ctx.num_ctx = 24576                    # a per-session setting that differs from the policy default
    eng.model = "other-model"
    S.save_session(eng)
    saved_msgs = json.loads(json.dumps(eng.ctx.messages))

    llm2 = FakeLLM([say("Nothing else to do: a.py prints hi (ran it before).")])
    eng2, _r, out2 = new_engine(tmp_path, llm2)              # nothing shared with the first engine but the disk
    meta = SS.latest(repo)
    assert meta.id == eng.session_id and meta.state == "finished" and meta.title.startswith("Write a small python module")
    info = S.resume_into(eng2, meta)
    assert info and info["changes"] == [] and not info["repaired"]
    assert eng2.ctx.messages == saved_msgs                   # stable prefix byte-identical: nothing re-summarised
    assert eng2.ctx.user_requests == eng.ctx.user_requests and eng2.ctx.step == eng.ctx.step
    assert {p: e.__dict__ for p, e in eng2.ctx.ledger.files.items()} == {p: e.__dict__ for p, e in eng.ctx.ledger.files.items()}
    assert eng2.ctx.ledger.runs == eng.ctx.ledger.runs and eng2.ctx.ledger.runs[0][2] == 0
    assert [(i.id, i.text, i.done, i.evidence) for i in eng2.plan.plan.items] == [(i.id, i.text, i.done, i.evidence) for i in eng.plan.plan.items]
    assert (eng2.model, eng2.num_ctx, eng2.ctx.num_ctx, eng2.session_id) == ("other-model", 24576, 24576, eng.session_id)
    assert "Every file in the ledger is unchanged" in eng2.resume_note
    res2 = eng2.turn("anything left?")
    assert res2.stopped == "" and res2.text.startswith("Nothing else")
    sent = llm2.calls[0]["snapshot"]
    assert sent[: len(saved_msgs)] == saved_msgs             # the server sees the old prompt as its prefix (cache reuse)
    assert "SESSION RESUMED" in sent[len(saved_msgs)]["content"] and sent[len(saved_msgs) + 1]["content"] == "anything left?"
    assert eng2.resume_note is None                          # sent once


class Killed(BaseException):
    """Stands in for kill -9 / a power cut: nothing after it runs."""


def test_a_run_killed_mid_turn_resumes_cleanly(tmp_path):
    def die(_messages):
        raise Killed()

    llm = FakeLLM([call("write_file", path="b.py", content="x = 1\n"), call("run", command="python3 b.py"), die])
    eng, repo, _ = new_engine(tmp_path, llm)
    eng.session_id = SS.new_id()
    eng.on_step = lambda: S.save_session(eng)
    with pytest.raises(Killed):
        eng.turn("make b.py")
    # no save after the kill: the file on disk is the autosave from before the third model call
    meta = SS.latest(repo)
    assert meta.state == "in progress" and meta.shown_state() == "interrupted (killed)"     # nobody holds the folder lock
    (repo / "b.py").write_text("x = 2\ny = 3\n")              # changed meanwhile (say, by hand)
    eng2, _r, _o = new_engine(tmp_path, FakeLLM([say("b.py defines x and y.")]))
    info = S.resume_into(eng2, meta)
    assert info["state"] == "in progress" and info["changes"] == ["b.py changed (now 3 lines)"]
    roles = [m["role"] for m in eng2.ctx.messages]
    assert roles[-4:] == ["assistant", "tool", "assistant", "tool"]                  # every call has its result
    assert "cut off mid-step" in eng2.resume_note and "b.py changed (now 3 lines)" in eng2.resume_note
    assert eng2.turn("continue").stopped == ""


def test_repair_gives_a_dangling_tool_call_a_result_and_drops_junk():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": json.dumps({"tool": "run", "arguments": {"command": "ls"}})},
            "junk", {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "a"}}, {"function": {"name": "b"}}]},
            {"role": "user", "content": None}]
    fixed, notes = SS.repair(msgs)
    assert [m["role"] for m in fixed] == ["system", "user", "assistant", "tool", "assistant", "tool", "tool", "user"]
    assert fixed[3]["content"] == SS.NO_RESULT and fixed[-1]["content"] == ""
    assert any("damaged" in n for n in notes) and any("3 tool call" in n for n in notes)
    ok = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "done"}]
    assert SS.repair(ok) == (ok, [])


def test_two_sessions_in_one_folder_and_prefix_lookup(tmp_path):
    eng, repo, _ = first_turn(tmp_path)
    first = eng.session_id
    S.reset_engine(eng)
    eng.llm.script = [say("second answer")]
    eng.turn("a second, unrelated question")
    S.save_session(eng)
    metas = SS.list_sessions(repo)
    assert [m.id for m in metas] == [eng.session_id, first]
    assert SS.find(first[:5], repo)[0].id == first
    other, _r, _o = new_engine(tmp_path, FakeLLM([]))
    S.resume_into(other, SS.find(first)[0])
    assert other.ctx.user_requests == [TASK] and other.plan.plan is not None
    S.resume_into(other, SS.find(eng.session_id)[0])
    assert other.ctx.user_requests == ["a second, unrelated question"] and other.plan.plan is None
    assert SS.find("zzzz")[0] is None and "no saved session" in SS.find("zzzz")[1]


def test_damaged_session_files_are_skipped_with_a_warning(tmp_path):
    eng, repo, _ = first_turn(tmp_path)
    d = SS.folder_dir(repo)
    (d / "deadbeef.json").write_text('{"id": "deadbeef", "messages": [{"role": "us')         # truncated mid-write
    (d / "cafe0001.json").write_text("not json at all")
    good = d / f"{eng.session_id}.json"
    warn: list[str] = []
    metas = SS.list_sessions(None, warn)
    assert [m.id for m in metas] == [eng.session_id] and len(warn) == 2 and all("damaged" in w for w in warn)
    meta = metas[0]
    good.write_text(good.read_text()[:200])                  # the real one truncated after listing
    eng2, _r, out = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng2, meta) is None and "can't resume" in out.getvalue()
    assert eng2.ctx.messages == []                           # nothing half-loaded


def test_resume_of_a_session_whose_folder_is_gone_is_refused_politely(tmp_path, monkeypatch, capsys):
    eng, repo, _ = first_turn(tmp_path)
    sid = eng.session_id
    import shutil

    shutil.rmtree(repo)
    monkeypatch.setattr("agent.config.load_config", lambda: CFG)
    monkeypatch.chdir(tmp_path)
    assert S.run_session(resume=sid[:4]) == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "no longer exists" in out and "stays saved" in out
    assert SS.find(sid)[0] is not None                       # still saved
    assert S.sessions_cli() == 0 and "folder gone" in capsys.readouterr().out


def test_legacy_single_file_is_migrated_never_lost(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    old = SS.state_root() / f"{SS.folder_key(repo.resolve())}.json"
    old.parent.mkdir(parents=True)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "old task"}, {"role": "assistant", "content": "old answer"}]
    old.write_text(json.dumps({"saved": time.time() - 3600, "root": str(repo.resolve()), "messages": msgs, "requests": ["old task"],
                               "files": {"x.py": {"action": "write", "step": 1, "lines": 1, "size": 6, "names": []}}, "runs": [[1, "ls", 0, ""]]}))
    metas = SS.list_sessions(repo.resolve())
    assert len(metas) == 1 and metas[0].title == "old task" and metas[0].state == "finished"
    assert not old.exists() and old.with_name(old.name + ".migrated").exists()
    eng, _r, _o = new_engine(tmp_path, FakeLLM([]))
    info = S.resume_into(eng, metas[0])
    assert eng.ctx.messages == msgs and eng.ctx.ledger.runs == [(1, "ls", 0, "")] and info["changes"] == ["x.py deleted"]


def test_default_start_continues_only_a_recent_session_resume_takes_any_age(tmp_path):
    eng, repo, _ = first_turn(tmp_path)
    con = Console(file=io.StringIO())
    assert S.pick_session(con, repo, None, False).id == eng.session_id
    assert S.pick_session(con, repo, None, True) is None                             # --new
    f = SS.folder_dir(repo) / f"{eng.session_id}.meta.json"
    m = json.loads(f.read_text())
    m["updated"] = time.time() - 13 * 3600
    f.write_text(json.dumps(m))
    assert S.pick_session(con, repo, None, False) is None                            # too old for the implicit continue
    assert S.pick_session(con, repo, "latest", False).id == eng.session_id           # --resume: any age
    assert S.pick_session(con, repo, eng.session_id[:4], False).id == eng.session_id


def test_a_lock_of_a_killed_process_never_blocks(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    code = ("import os, signal, sys; from pathlib import Path; from agent.codeagent import session as S\n"
            f"ok, _ = S.acquire_lock(Path({str(repo)!r})); print('GOT' if ok else 'NO', flush=True); os.kill(os.getpid(), signal.SIGKILL)")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ})
    assert p.stdout.startswith("GOT") and p.returncode == -signal.SIGKILL
    assert S.lock_file(repo).exists()                        # the file stays (with a dead pid in it) ...
    assert not SS.folder_busy(repo)                          # ... but the kernel lock died with the process
    ok, _ = S.acquire_lock(repo)
    assert ok and SS.folder_busy(repo)


def test_sessions_cli_lists_all_folders_filters_deletes_and_prunes(tmp_path, monkeypatch, capsys):
    eng, repo, _ = first_turn(tmp_path)
    eng2, repo2, _o = new_engine(tmp_path, FakeLLM([say("ok")]), name="other")
    eng2.session_id = SS.new_id()
    eng2.turn("hello from the other folder")
    S.save_session(eng2)
    assert S.sessions_cli() == 0
    out = capsys.readouterr().out
    assert eng.session_id in out and eng2.session_id in out and "hello from the other" in out
    monkeypatch.chdir(repo2)
    assert S.sessions_cli(here=True) == 0
    out = capsys.readouterr().out
    assert eng2.session_id in out and eng.session_id not in out
    assert S.sessions_cli(delete=eng2.session_id) == 0 and SS.find(eng2.session_id)[0] is None
    f = SS.folder_dir(repo) / f"{eng.session_id}.meta.json"
    m = json.loads(f.read_text())
    m["updated"] = time.time() - 40 * 86400
    f.write_text(json.dumps(m))
    assert S.sessions_cli(prune=30) == 0 and SS.list_sessions() == []


def test_repl_commands_new_sessions_resume_delete(tmp_path):
    eng, repo, _ = first_turn(tmp_path)
    out = io.StringIO()
    con = Console(file=out, width=160)
    first = eng.session_id
    S.handle_command(con, eng, "/new")
    assert eng.session_id != first and eng.ctx.user_requests == [] and eng.plan.plan is None and f"/resume {first}" in out.getvalue()
    eng.llm.script = [say("fine")]
    eng.turn("second chat")
    S.handle_command(con, eng, "/sessions")
    assert first in out.getvalue() and "this one" in out.getvalue()
    S.handle_command(con, eng, f"/resume {first[:4]}")
    assert eng.session_id == first and eng.ctx.user_requests == [TASK] and "resumed" in out.getvalue()
    second = SS.list_sessions(repo)[0].id                    # the second chat was saved before switching
    assert second != first
    S.handle_command(con, eng, f"/delete {first}")
    assert "that is the session you are in" in out.getvalue()
    S.handle_command(con, eng, f"/delete {second}")
    assert SS.find(second)[0] is None


def test_cli_resume_flag_takes_an_id_or_a_task(monkeypatch):
    from typer.testing import CliRunner

    from agent import main

    seen = []
    monkeypatch.setattr(S, "run_session", lambda folder, task, new, files, test, which=None: seen.append((task, which)) or 0)
    monkeypatch.setattr(SS, "id_exists", lambda prefix: prefix == "3fa9c1")           # an id only counts when a saved session has it (B5)
    r = CliRunner()
    r.invoke(main.app, ["code", "--old-ui", "--resume", "3fa9c1"])
    r.invoke(main.app, ["code", "--old-ui", "--resume"])
    r.invoke(main.app, ["code", "--old-ui", "--resume", "add a health route"])
    r.invoke(main.app, ["code", "--old-ui", "fix it"])
    assert seen == [(None, "3fa9c1"), (None, "latest"), ("add a health route", "latest"), ("fix it", None)]


def test_home_topic_resumes_the_picked_session_in_its_folder(tmp_path, monkeypatch):
    from agent import main
    from agent.ui import launcher

    eng, repo, _ = first_turn(tmp_path)
    ran = []
    monkeypatch.setattr(main, "app", lambda argv, standalone_mode=False: ran.append((argv, os.getcwd())))
    con = Console(file=io.StringIO(), width=160)
    monkeypatch.setattr(con, "input", lambda *_a, **_k: "1")
    here = os.getcwd()
    try:
        launcher.code_sessions(con)
    finally:
        os.chdir(here)
    assert ran == [(["code", "--resume", eng.session_id, "--folder", str(repo.resolve())], str(repo.resolve()))]


def test_look_streak_survives_a_resume_but_leaves_a_few_reads(tmp_path):
    from agent.codeagent.engine import LOOK_BLOCK_AT

    eng, repo, res = first_turn(tmp_path)
    eng._look_streak = 40
    S.save_session(eng)
    eng2, _r, _o = new_engine(tmp_path, FakeLLM([say("ok")]))
    assert S.resume_into(eng2, SS.latest(repo))
    assert eng2._look_streak == LOOK_BLOCK_AT - 5              # not a fresh allowance of 25 reads, but a few stay allowed
    eng.ctx.step = eng.ctx.step
    eng._look_streak = 3
    S.save_session(eng)
    eng3, _r3, _o3 = new_engine(tmp_path, FakeLLM([say("ok")]))
    assert S.resume_into(eng3, SS.latest(repo)) and eng3._look_streak == 3


# ── counters and flags that used to live only in memory ─────────────────────

def _saved(tmp_path, llm=None, name="repo"):
    eng, repo, _ = new_engine(tmp_path, llm or FakeLLM([]), name)
    eng.session_id = SS.new_id()
    eng.start()
    eng.ctx.add_user("make things")
    return eng, repo


def _resume(tmp_path, eng, repo, llm=None):
    S.save_session(eng)
    eng2, _r, _o = new_engine(tmp_path, llm or FakeLLM([]))
    assert S.resume_into(eng2, SS.latest(repo))
    return eng2


def test_loop_progress_lookact_and_watcher_state_round_trip(tmp_path):
    eng, repo = _saved(tmp_path)
    eng.ctx.step = 30
    for _ in range(2):
        eng._loop_guard("run", {"command": "python3 x.py"}, "exit code 1\nboom")
    eng._progress, eng._loop_note_at, eng._in_place_failures, eng.ctx._compact_retry_at = 7, 21, 1, 1234.5
    la = eng.lookact
    la.look_streak, la.wander_nudges, la.edit_gap, la.edit_nudges, la.probing = 7, 1, 4, 1, True
    w = eng.watcher
    w.lookups, w.streak, w.last_call, w.identical = 2, 2, ("run", "[('command', 'x')]"), 2
    w.seen, w.looked_up = {"boom": 2}, {"boom"}
    eng.tools.dirty = {"b.py", "a.py"}
    eng2 = _resume(tmp_path, eng, repo)
    assert (eng2._progress, eng2._loop_note_at, eng2._in_place_failures, eng2.ctx._compact_retry_at) == (7, 21, 1, 1234.5)
    assert eng2.lookact.state() == la.state() and la.state() == {"look_streak": 7, "wander_nudges": 1, "edit_gap": 4, "edit_nudges": 1, "probing": True}
    assert (eng2.watcher.lookups, eng2.watcher.streak, eng2.watcher.last_call, eng2.watcher.identical) == (2, 2, ("run", "[('command', 'x')]"), 2)
    assert eng2.watcher.seen == {"boom": 2} and eng2.watcher.looked_up == {"boom"}
    assert eng2.tools.dirty == {"a.py", "b.py"}
    assert [h[:5] for h in eng2._sigs] == [h[:5] for h in eng._sigs] and len(eng2._sigs) == 2
    assert all(len(h[1]) == 40 and len(h[2]) == 40 for h in eng2._sigs)           # sha1 hex, not the long strings


def test_the_loop_guard_keeps_counting_after_a_resume(tmp_path):
    eng, repo = _saved(tmp_path)
    eng.ctx.step = 10
    for _ in range(2):
        eng._loop_guard("search", {"pattern": "zzz"}, "no matches")
    eng2 = _resume(tmp_path, eng, repo)
    assert not [m for m in eng2.ctx.messages if m["role"] == "system" and "identical" in str(m["content"])]
    eng2._loop_guard("search", {"pattern": "zzz"}, "no matches")                    # the 3rd copy overall
    assert [m for m in eng2.ctx.messages if m["role"] == "system" and "identical `search` call 3 times" in str(m["content"])]


def test_a_session_file_without_the_new_keys_still_loads(tmp_path):
    eng, repo, res = first_turn(tmp_path)
    meta = SS.latest(repo)
    data = json.loads(meta.path.read_text())
    for k in ("progress", "sigs", "loop_note_at", "lookact", "watcher", "in_place_failures", "compact_retry_at", "dirty"):
        data["engine_state"].pop(k, None)
    data["engine_state"]["declined"] = [["run", "cd /elsewhere"]]                 # the old 2-element shape
    meta.path.write_text(json.dumps(data))
    eng2, _r, _o = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng2, meta)
    assert (eng2._progress, eng2._sigs, eng2._loop_note_at, eng2._in_place_failures, eng2.ctx._compact_retry_at) == (0, [], -99, 0, 0.0)
    assert eng2.lookact.state() == {"look_streak": 0, "wander_nudges": 0, "edit_gap": 0, "edit_nudges": 0, "probing": False}
    assert eng2.watcher.lookups == 0 and eng2.tools.syntax_bad == {}
    eng2.llm.script = [say("nothing more")]
    assert eng2.turn("anything?").stopped == ""


def test_declined_calls_are_keyed_on_the_whole_call_not_its_first_300_characters(tmp_path, monkeypatch):
    import sys as _sys
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False, raising=False)
    eng, repo = _saved(tmp_path)
    eng._explicit_approve = False
    head = "cd /srv/luke/repos/Projects && python3 -c '" + "x" * 400
    one, two = {"command": head + "1'"}, {"command": head + "2'"}
    assert eng.execute("run", one).startswith("NOT RUN")
    assert eng.execute("run", one).startswith("NOT RUN (asked before)")
    assert not eng.execute("run", two).startswith("NOT RUN (asked before)")        # same first 300 chars, a different call
    assert all(len(h) == 40 for _n, h in eng._declined)
    eng2 = _resume(tmp_path, eng, repo)
    eng2._explicit_approve = False
    assert eng2._declined == eng._declined
    assert eng2.execute("run", one).startswith("NOT RUN (asked before)") and eng2.execute("run", two).startswith("NOT RUN (asked before)")


def test_resume_rebuilds_syntax_bad_and_dirty_from_the_ledger(tmp_path):
    eng, repo = _saved(tmp_path)
    eng.tools.write_file("a.py", "x = 1\n")
    eng.tools.write_file("b.py", "y = 1\n")
    eng.tools.write_file("c.py", "z = 1\n")
    eng.tools.dirty.discard("c.py")
    S.save_session(eng)
    (repo / "b.py").write_text("def broken(:\n    pass\n")                          # a killed run left it broken
    (repo / "c.py").write_text("def alsobroken(:\n")                                # changed since the save, not in the saved dirty set
    eng2, _r, _o = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng2, SS.latest(repo))
    assert set(eng2.tools.syntax_bad) == {"b.py", "c.py"} and "a.py" not in eng2.tools.syntax_bad
    assert {"a.py", "b.py", "c.py"} <= eng2.tools.dirty


def test_resume_rebuilds_static_problems_of_changed_files(tmp_path):
    eng, repo = _saved(tmp_path)
    eng.tools.write_file("m.py", "x = 1\n")
    S.save_session(eng)
    (repo / "m.py").write_text("def f():\n    return undefined_thing\n")
    eng2, _r, _o = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng2, SS.latest(repo))
    assert "m.py" in eng2.tools.static_bad and "m.py" not in eng2.tools.syntax_bad


def test_the_resume_check_is_capped(tmp_path, monkeypatch):
    eng, repo = _saved(tmp_path)
    for i in range(6):
        eng.tools.write_file(f"f{i}.py", "x = 1\n")
    S.save_session(eng)
    for i in range(6):
        (repo / f"f{i}.py").write_text("def (:\n")
    monkeypatch.setattr(SS, "RESUME_CHECK_MAX", 4)
    eng2, _r, _o = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng2, SS.latest(repo))
    assert len(eng2.tools.syntax_bad) == 4


def test_a_rubric_is_judged_again_after_a_resume_when_a_file_changed(tmp_path):
    from agent.codeagent import checks as K

    seen = []
    eng, repo = _saved(tmp_path)
    eng.tools.write_file("lena.py", "line = 'hmph'\n")
    eng._review = lambda crit, out: seen.append(out) or {"status": "pass", "evidence": [], "problems": []}
    eng.checks.add(K.Check("tone", "warm", "rubric", {"criterion": "warm", "scenario": ["echo hi"]}))
    eng.run_checks(quiet=True)
    assert len(seen) == 1
    eng2 = _resume(tmp_path, eng, repo)                                           # nothing changed: still fresh (progress AND stamp match)
    eng2._review = lambda crit, out: seen.append(out) or {"status": "pass", "evidence": [], "problems": []}
    eng2.run_checks(quiet=True)
    assert len(seen) == 1
    eng2.tools.write_file("lena.py", "line = 'baka'\n")                           # an edit that did not go through the progress counter
    eng3 = _resume(tmp_path, eng2, repo)
    eng3._review = lambda crit, out: seen.append(out) or {"status": "pass", "evidence": [], "problems": []}
    assert eng3._progress == eng._progress
    eng3.run_checks(quiet=True)
    assert len(seen) == 2


def test_a_rubric_result_without_a_content_stamp_is_judged_again(tmp_path):
    from agent.codeagent import checks as K

    seen = []
    eng, repo = _saved(tmp_path)
    eng._review = lambda crit, out: seen.append(out) or {"status": "pass", "evidence": [], "problems": []}
    c = K.Check("tone", "warm", "rubric", {"criterion": "warm", "scenario": ["echo hi"]})
    c.last_result = {"status": "pass", "at": eng._progress, "seconds": 0.1}       # an old session's result
    eng.checks.add(c)
    eng.run_checks(quiet=True)
    assert len(seen) == 1


def test_owned_files_round_trip_and_old_sessions_load_empty(tmp_path):
    eng, repo = _saved(tmp_path)
    eng.tools.owned = {"todos.json": "run", "a.py": "write"}
    eng2 = _resume(tmp_path, eng, repo)
    assert eng2.tools.owned == {"todos.json": "run", "a.py": "write"}
    meta = SS.latest(repo)
    data = json.loads(meta.path.read_text())
    data["engine_state"].pop("owned", None)
    meta.path.write_text(json.dumps(data))
    eng3, _r, _o = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng3, meta) and eng3.tools.owned == {}
    data["engine_state"]["owned"] = {"x": "bogus", "y": 3, "ok.json": "run"}
    meta.path.write_text(json.dumps(data))
    eng4, _r, _o = new_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng4, meta) and eng4.tools.owned == {"ok.json": "run"}
