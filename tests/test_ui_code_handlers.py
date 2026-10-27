"""Slash commands of the code UI (agent/ui/code_handlers.py): the command table's `when`, the classic code-REPL commands bound to the engine,
the run-view commands, and the guarantee that nothing reads stdin."""

from __future__ import annotations

import builtins
import sys

import pytest

from agent.codeagent import policy as P
from agent.codeagent import sessionstore as SS
from agent.ui import commands as C
from agent.ui.code_handlers import build_registry
from agent.ui.commands import REGISTRY
from tests.test_codeagent_engine import FakeLLM, _isolated, call, say  # noqa: F401 - _isolated is autouse
from tests.test_ui_code_app import Gate, idle, make, wait_for


def run(sess, line):
    sess.app.handle_line(line)


def out(buf):
    return buf.getvalue()


# ── the table ──────────────────────────────────────────────────────────────

def test_registry_is_a_copy_and_code_mode_remaps_the_two_clashing_aliases():
    r = build_registry()
    assert r.get("think").name == "think" and r.get("ctx").name == "ctx" and r.get("context").name == "context"
    assert REGISTRY.get("think").name == "thinking" and REGISTRY.get("ctx").name == "context"       # the shared table is untouched
    for gone in ("web", "verify", "doctor", "notes", "project", "access", "auto"):
        assert r.get(gone) is None
    for name in ("say", "compact", "steps", "mode", "undo", "plan", "pause", "resume", "stop", "why-slow", "events", "diff", "files", "network"):
        assert r.get(name) is not None
    assert all(c.when in C.WHENS for c in r.commands())


@pytest.mark.parametrize("line,busy,action", [
    ("/status", True, C.RUN_NOW), ("/diff", True, C.RUN_NOW), ("/files", True, C.RUN_NOW), ("/policy", True, C.RUN_NOW),
    ("/why-slow", True, C.RUN_NOW), ("/events", True, C.RUN_NOW), ("/sessions", True, C.RUN_NOW), ("/stages", True, C.RUN_NOW),
    ("/undo list", True, C.RUN_NOW), ("/plan", True, C.RUN_NOW), ("/network", True, C.RUN_NOW), ("/allow", True, C.RUN_NOW),
    ("/compact", True, C.STEP_ACTION), ("/say hi", True, C.STEP_ACTION), ("/steps 20", True, C.STEP_ACTION), ("/mode ask", True, C.STEP_ACTION),
    ("/network on", True, C.DEFER), ("/allow pytest*", True, C.DEFER), ("/deny rm*", True, C.DEFER), ("/commit auto", True, C.DEFER),
    ("/think on", True, C.DEFER), ("/ctx 16", True, C.DEFER), ("/undo", True, C.DEFER), ("/stages run", True, C.DEFER), ("/state", True, C.DEFER),
    ("/model qwen", True, C.DEFER), ("/new", True, C.DEFER), ("/plan clear", True, C.DEFER), ("/green", True, C.DEFER),
    ("/network on", False, C.RUN_NOW), ("/undo", False, C.RUN_NOW), ("/web x", False, C.REFUSE),
])
def test_when_for_a_running_agent(line, busy, action):
    assert build_registry().classify(line, busy).action == action


# ── classic commands, bound to the engine ──────────────────────────────────

def test_policy_commands_persist_and_report(tmp_path):
    sess, eng, repo, buf = make(tmp_path, FakeLLM([]))
    run(sess, "/network on")
    run(sess, "/installs on")
    run(sess, "/allow pytest*")
    run(sess, "/deny rm -rf*")
    run(sess, "/commit ask")
    run(sess, "/think on")
    run(sess, "/steps 25")
    run(sess, "/ctx 16")
    pol = P.for_repo(repo)
    assert pol.network and pol.installs and "pytest*" in pol.allow and "rm -rf*" in pol.deny
    assert pol.commit == "ask" and pol.think == "on" and pol.max_steps == 25 and pol.ctx == 16 * 1024
    assert eng.pol.network and eng.num_ctx == 16 * 1024
    run(sess, "/policy")
    assert "network" in out(buf).lower()
    run(sess, "/unallow pytest*")
    run(sess, "/allow")
    assert "pytest*" not in P.for_repo(repo).allow and "allow: -" in out(buf)


def test_mode_and_model(tmp_path):
    sess, eng, repo, buf = make(tmp_path, FakeLLM([]))
    run(sess, "/mode medium")
    assert eng.pol.mode == "medium" and P.for_repo(repo).mode == "medium"
    run(sess, "/mode sideways")
    assert "mode is one of" in out(buf) and eng.pol.mode == "medium"
    run(sess, "/model other-model:7b")
    assert eng.model == "other-model:7b" and "model other-model:7b" in out(buf)


def test_mode_steps_say_while_busy_use_the_control_object(tmp_path):
    gate = Gate(say("done"))
    sess, eng, repo, buf = make(tmp_path, FakeLLM([gate]))
    run(sess, "do it")
    assert wait_for(lambda: gate.entered.is_set())
    run(sess, "/mode medium")
    run(sess, "/steps 30")
    run(sess, "/mode sideways")
    assert eng.pol.mode == "medium" and eng.pol.max_steps == 30
    assert "mode medium (saved" in out(buf) and "steps per prompt: 30" in out(buf) and "mode is one of" in out(buf)
    run(sess, "/network on")                              # deferred: nothing changes yet
    assert not eng.pol.network and "will run after the current turn" in out(buf)
    gate.release.set()
    idle(sess)
    assert wait_for(lambda: eng.pol.network)              # ... and runs once the turn is over


def test_files_diff_undo_after_a_run(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="x = 1\ny = 2\n"), say("Wrote a.py.")])
    sess, eng, repo, buf = make(tmp_path, llm)
    run(sess, "make a.py")
    idle(sess)
    assert "+x = 1" not in out(buf)                       # collapsed in the stream ...
    run(sess, "/diff a.py")
    assert "+x = 1" in out(buf) and "+y = 2" in out(buf)  # ... expanded on request
    run(sess, "/diff zzz.py")
    assert "no recorded edit for zzz.py" in out(buf)
    run(sess, "/files")
    assert "FILES CHANGED 1" in out(buf)
    run(sess, "/undo list")
    assert (repo / "a.py").exists()
    run(sess, "/undo")
    assert not (repo / "a.py").exists()


def test_status_context_plan_events_whyslow(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="x = 1\n"), say("Wrote a.py.")])
    sess, eng, repo, buf = make(tmp_path, llm)
    run(sess, "make a.py")
    idle(sess)
    for line in ("/status", "/context", "/plan", "/events", "/why-slow", "/log", "/usage"):
        run(sess, line)
    o = out(buf)
    assert "session   " + eng.session_id in o and "steps " in o and "policy" in o and "auto mode" in o
    assert "tokens" in o and "mask at" in o              # /context (classic numbers)
    assert "no acceptance list" in o or "ACCEPTANCE" in o
    assert "steps 1" in o or "steps 2" in o or "no steps yet" in o


def test_note_and_sessions_and_new_and_resume(tmp_path):
    llm = FakeLLM([say("hello"), say("again")])
    sess, eng, repo, buf = make(tmp_path, llm)
    run(sess, "first task")
    idle(sess)
    first = eng.session_id
    run(sess, "/sessions")
    assert first in out(buf) and "first task" in out(buf)
    run(sess, "/new")
    assert eng.session_id != first and eng.ctx.user_requests == [] and "fresh session" in out(buf)
    assert sess.runner.panel.result is None
    run(sess, "/resume " + first[:4])
    assert eng.session_id == first and "resumed" in out(buf)
    run(sess, "/delete " + first[:4])
    assert "that is the session you are in" in out(buf)
    assert SS.find(first)[0] is not None


def test_resume_without_id_resumes_the_paused_run_and_with_id_refuses_while_busy(tmp_path):
    gate = Gate(call("list_dir", path="."))
    sess, eng, repo, buf = make(tmp_path, FakeLLM([gate, say("done")]))
    run(sess, "go")
    assert wait_for(lambda: gate.entered.is_set())
    run(sess, "/pause")
    run(sess, "/resume abcd")
    assert "/stop it first" in out(buf) and sess.app.worker.state == "paused"
    gate.release.set()
    assert wait_for(lambda: eng.now == "paused")
    run(sess, "/resume")
    idle(sess)
    assert not eng.live.paused


def test_stages_listing_and_run(tmp_path):
    from tests.test_codeagent_stages import TASK3, new_engine, three_stage_script

    llm = FakeLLM(three_stage_script())
    eng, repo, _ = new_engine(tmp_path, llm)
    sess, eng, repo, buf = make(tmp_path, llm, eng=eng, cfg=eng.config)
    run(sess, "/stages")
    assert "no staged task" in out(buf)
    from agent.codeagent import stages as ST

    plan = ST.split(TASK3, llm=eng.llm, cfg=eng.config, model=eng.model, cache=eng.split_cache, root=eng.root)
    eng.stage_plan = plan
    eng.ctx.user_requests.append(TASK3)
    run(sess, "/stages")
    assert "stages (" in out(buf)
    run(sess, "/stages run")
    assert sess.app.worker.wait_idle(30)
    assert [s.status for s in eng.stage_plan.stages] == ["done"] * 3


def test_say_without_text_and_note(tmp_path):
    sess, eng, repo, buf = make(tmp_path, FakeLLM([]))
    run(sess, "/say")
    assert "usage: /say" in out(buf)
    run(sess, "/note remember the build uses make")
    assert "saved" in out(buf) or "not saved" in out(buf)       # no notes store in the fixture engine: reported, not raised


def test_nothing_reads_stdin(tmp_path, monkeypatch):
    class NoStdin:
        def read(self, *a):
            raise AssertionError("stdin was read")

        readline = read

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", NoStdin())
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(AssertionError("input() was called")))
    llm = FakeLLM([say("ok")])
    sess, eng, repo, buf = make(tmp_path, llm, mode="ask")
    for line in ("/help", "/status", "/policy", "/mode medium", "/network off", "/sessions", "/events", "/why-slow", "/context", "/diff",
                 "/files", "/plan", "/stages", "/undo", "/state", "/new", "/compact", "/map", "/think auto"):
        run(sess, line)
    run(sess, "hello")
    idle(sess)
    assert " failed:" not in out(buf)
    assert "unknown command" not in out(buf)
