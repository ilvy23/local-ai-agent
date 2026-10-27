"""The real code agent on the terminal UI (agent/ui/code_app.py), driven by fake-LLM engines: no model, no stdin. Most tests feed lines to the app
directly (`app.handle_line`); the `Dock` ones use a prompt_toolkit pipe input with hard timeouts."""

from __future__ import annotations

import contextlib
import io
import threading
import time

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from agent.codeagent import session as S
from agent.codeagent import sessionstore as SS
from agent.ui.code_app import CodeSession, run_code_ui
from agent.ui.code_runner import CodeRunner
from tests.test_codeagent_engine import CFG, FakeLLM, _isolated, call, engine, say  # noqa: F401 - _isolated is autouse
from tests.test_codeagent_sessions import new_engine as session_engine
from tests.test_codeagent_stages import TASK3, new_engine, three_stage_script

OFF = {**CFG, "coding": {**CFG["coding"], "stages": "off"}}


def wait_for(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def make(tmp_path, llm, *, goal=None, mode="auto", eng=None, cfg=OFF, width=100, **kw):
    if eng is None:
        eng, repo, _ = engine(tmp_path, llm, mode=mode)
    else:
        repo = eng.root
    eng.config = cfg
    if not eng.session_id:
        eng.session_id = SS.new_id()
        eng.start()
    eng.on_step = lambda: S.save_session(eng)
    buf = io.StringIO()
    con = Console(file=buf, width=width, force_terminal=False, color_system=None, highlight=False)
    sess = CodeSession(eng, cfg, goal=goal, plain=True, console=con, history_path=tmp_path / "hist", **kw)
    return sess, eng, repo, buf


def idle(sess, t=8.0):
    assert sess.app.worker.wait_idle(t)


class Gate:
    """A scripted model step that waits for the test."""

    def __init__(self, then):
        self.release, self.entered, self.then = threading.Event(), threading.Event(), then

    def __call__(self, messages):
        self.entered.set()
        assert self.release.wait(8)
        return self.then


class Dock:
    """Runs sess.run() on a pipe input in a thread."""

    def __init__(self, sess):
        self.sess = sess

    def __enter__(self):
        self._cm = create_pipe_input()
        self.inp = self._cm.__enter__()
        a = self.sess.app
        a._input, a._output, a._width, a._live_guard = self.inp, DummyOutput(), (lambda: 100), "suppress"
        self.rc = []

        def go():
            with create_app_session(input=self.inp, output=DummyOutput()):
                self.rc.append(self.sess.run())

        self.t = threading.Thread(target=go, daemon=True)
        self.t.start()
        return self

    def type(self, text, pause=0.15):
        time.sleep(pause)
        self.inp.send_text(text)

    def __exit__(self, *a):
        self.t.join(10)
        alive = self.t.is_alive()
        self._cm.__exit__(*a)
        assert not alive, "the UI did not exit"


# ── a goal through the dock ────────────────────────────────────────────────

def test_goal_run_renders_events_prints_summary_and_saves_the_session(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="x = 1\nprint(x)\n"), say("Wrote a.py.")])
    sess, eng, repo, buf = make(tmp_path, llm, goal="make a.py")
    with Dock(sess) as d:
        assert wait_for(lambda: (repo / "a.py").exists() and not sess.app.worker.busy and "DONE" in buf.getvalue())
        d.type("/quit\r")
    out = buf.getvalue()
    assert d.rc == [0]                                           # the goal run's exit code
    assert "> make a.py" in out and "write_file" in out          # the goal is echoed, the tool line printed
    assert "edit a.py +2 -0" in out or "created a.py +2 -0" in out
    assert "+x = 1" not in out                                   # edits are collapsed one-liners
    assert "DONE" in out and "FILES CHANGED 1" in out and "a.py" in out
    meta = SS.latest(repo)
    assert meta is not None and meta.id == eng.session_id        # saved
    assert [a.state for a in sess.activities.list() if a.kind == "code"] == ["done"]


def test_banner_is_one_plain_line_with_the_policy_mode(tmp_path):
    sess, eng, repo, buf = make(tmp_path, FakeLLM([]), mode="medium", width=220)
    sess.print_banner(None)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines[0].startswith("nightage code") and "medium mode" in lines[0] and str(repo) in lines[0] and "(new)" in lines[0]
    assert "#" not in "".join(lines[:1]) and len(lines) == 2           # one banner line + the key hints


def test_header_rows_come_from_the_engine(tmp_path):
    gate = Gate(say("done"))
    sess, eng, repo, buf = make(tmp_path, FakeLLM([gate]))
    sess.app.handle_line("do it")
    assert wait_for(lambda: gate.entered.is_set())
    snap = sess.app.activity_snapshot()
    assert snap["busy"] and snap["now"].startswith("step ") and "(" in snap["now"]       # step n/budget (source) ...
    fs = sess.app.footer_state()
    assert fs.model == eng.model and fs.access == "medium" or fs.access.startswith("auto")
    assert fs.ctx_limit == eng.num_ctx and fs.folder == str(repo)
    gate.release.set()
    idle(sess)


# ── follow-ups, queue, steering ────────────────────────────────────────────

def test_follow_up_prompt_continues_on_the_same_engine(tmp_path):
    llm = FakeLLM([say("first answer"), say("second answer")])
    sess, eng, repo, buf = make(tmp_path, llm)
    sess.app.handle_line("one")
    idle(sess)
    sess.app.handle_line("two")
    idle(sess)
    assert len(llm.calls) == 2
    prior = " ".join(str(m.get("content")) for m in llm.calls[1]["snapshot"])
    assert "one" in prior and "first answer" in prior                   # same conversation, not a fresh engine
    assert [r for r in eng.ctx.user_requests] == ["one", "two"]


def test_prompts_typed_while_busy_queue_and_run_in_order(tmp_path):
    gate = Gate(say("a"))
    llm = FakeLLM([gate, say("b"), say("c")])
    sess, eng, repo, buf = make(tmp_path, llm)
    sess.app.handle_line("p1")
    assert wait_for(lambda: gate.entered.is_set())
    sess.app.handle_line("p2")
    sess.app.handle_line("p3")
    assert "queued (2): p3" in buf.getvalue() and len(sess.app.queue) == 2
    gate.release.set()
    assert wait_for(lambda: len(llm.calls) == 3 and not sess.app.worker.busy)
    assert eng.ctx.user_requests == ["p1", "p2", "p3"]


def test_say_reaches_the_next_model_call(tmp_path):
    gate = Gate(call("list_dir", path="."))
    llm = FakeLLM([gate, say("done")])
    sess, eng, repo, buf = make(tmp_path, llm)
    sess.app.handle_line("explore")
    assert wait_for(lambda: gate.entered.is_set())
    sess.app.handle_line("/say please also add a README")
    assert "will apply at the next step" in buf.getvalue()
    gate.release.set()
    idle(sess)
    assert any("add a README" in str(m["content"]) for m in llm.calls[1]["snapshot"])
    assert sess.app.queue.items() == []


def test_say_when_idle_starts_a_turn(tmp_path):
    llm = FakeLLM([say("ok")])
    sess, eng, repo, buf = make(tmp_path, llm)
    sess.app.handle_line("/say hello agent")
    idle(sess)
    assert eng.ctx.user_requests == ["hello agent"]


def test_compact_request_is_taken_at_the_step_boundary(tmp_path):
    gate = Gate(call("list_dir", path="."))
    sess, eng, repo, buf = make(tmp_path, FakeLLM([gate, say("done")]))
    sess.app.handle_line("go")
    assert wait_for(lambda: gate.entered.is_set())
    sess.app.handle_line("/compact")
    assert eng.live._compact is True
    gate.release.set()
    idle(sess)
    assert eng.live._compact is False and eng.events.of("compact_requested")


def test_pause_then_resume(tmp_path):
    gate = Gate(call("list_dir", path="."))
    llm = FakeLLM([gate, say("finished")])
    sess, eng, repo, buf = make(tmp_path, llm)
    sess.app.handle_line("go")
    assert wait_for(lambda: gate.entered.is_set())
    sess.app.handle_line("/pause")
    assert sess.app.worker.state == "paused" and eng.live.paused
    gate.release.set()
    assert wait_for(lambda: eng.now == "paused")
    act = [a for a in sess.activities.list() if a.kind == "code"][0]
    assert act.state == "paused"
    sess.app.handle_line("/resume")
    idle(sess)
    assert len(llm.calls) == 2 and not eng.live.paused


def test_stop_keeps_the_work_and_pauses_the_queue(tmp_path):
    gate = Gate(call("list_dir", path="."))
    llm = FakeLLM([call("write_file", path="keep.py", content="a = 1\n"), gate, say("never")])
    sess, eng, repo, buf = make(tmp_path, llm)
    sess.app.handle_line("write then stop")
    assert wait_for(lambda: gate.entered.is_set())
    sess.app.handle_line("later")
    sess.app.handle_line("/stop")
    gate.release.set()
    idle(sess)
    assert (repo / "keep.py").exists()
    out = buf.getvalue()
    assert "STOPPED (interrupted)" in out and "/undo" in out and "keep.py" in out
    assert sess.app.worker.last_outcome == "interrupted" and sess.app.queue.paused and len(sess.app.queue) == 1


def test_tasks_stop_goes_through_the_app(tmp_path):
    gate = Gate(call("list_dir", path="."))
    sess, eng, repo, buf = make(tmp_path, FakeLLM([gate, say("never")]))
    sess.app.handle_line("go")
    assert wait_for(lambda: gate.entered.is_set())
    sess.app.handle_line("/tasks")
    assert "code-" in buf.getvalue() and "stop,pause" in buf.getvalue()
    act = [a for a in sess.activities.list() if a.kind == "code"][0]
    sess.app.handle_line(f"/tasks stop {act.id}")
    gate.release.set()
    idle(sess)
    assert sess.app.worker.last_outcome == "interrupted"


# ── approvals ──────────────────────────────────────────────────────────────

def _approval_turn(tmp_path, answer):
    llm = FakeLLM([call("write_file", path="w.py", content="a = 1\n"), say("ok"), say("ok")])
    sess, eng, repo, buf = make(tmp_path, llm, mode="ask")
    sess.app.handle_line("write w.py")
    seen = set()
    end = time.time() + 10
    while sess.app.worker.busy and time.time() < end:
        req = sess.app.worker.pending_approval
        if req is not None and id(req) not in seen:
            seen.add(id(req))
            sess.app.handle_line(answer)
        time.sleep(0.01)
    idle(sess)
    return sess, repo, buf


def test_approval_round_trip_allow(tmp_path):
    sess, repo, buf = _approval_turn(tmp_path, "y")
    assert (repo / "w.py").exists()
    assert "approval needed" in buf.getvalue() and "w.py" in buf.getvalue()


def test_approval_round_trip_deny(tmp_path):
    sess, repo, buf = _approval_turn(tmp_path, "n")
    assert not (repo / "w.py").exists() and "denied" in buf.getvalue()


def test_dangerous_approval_needs_a_typed_yes(tmp_path):
    result = []

    def ask(messages):
        result.append(eng._approve("run  rm -rf build", "dangerous command"))
        return say("done")

    sess, eng, repo, buf = make(tmp_path, FakeLLM([ask]))
    sess.app.handle_line("clean up")
    assert wait_for(lambda: sess.app.worker.pending_approval is not None)
    req = sess.app.worker.pending_approval
    assert req.risk == "DANGEROUS" and "rm -rf build" in req.command
    sess.app.handle_line("y")                          # a stray y never runs it
    assert "type  yes" in buf.getvalue() and sess.app.worker.pending_approval is not None
    sess.app.handle_line("yes")
    idle(sess)
    assert result == [True]


# ── plan-first, resume ─────────────────────────────────────────────────────

def test_plan_first_decline_executes_nothing(tmp_path):
    llm = FakeLLM([])
    eng, repo, _ = new_engine(tmp_path, llm)
    sess, eng, repo, buf = make(tmp_path, llm, goal=TASK3, plan_first=True, eng=eng, cfg=eng.config)
    sess.app.worker.submit(TASK3)
    assert wait_for(lambda: "Run this plan" in buf.getvalue())
    assert "PLAN 0/3" in buf.getvalue()
    sess.app.handle_line("n")
    idle(sess)
    assert llm.calls == [] and eng.stage_plan is None and "plan declined - nothing was run" in buf.getvalue()
    assert "DONE" not in buf.getvalue()


def test_plan_first_accept_runs_the_stages(tmp_path):
    llm = FakeLLM(three_stage_script())
    eng, repo, _ = new_engine(tmp_path, llm)
    sess, eng, repo, buf = make(tmp_path, llm, goal=TASK3, plan_first=True, eng=eng, cfg=eng.config)
    sess.app.worker.submit(TASK3)
    assert wait_for(lambda: sess.app.worker.pending_approval is not None)
    sess.app.handle_line("y")
    assert sess.app.worker.wait_idle(30)
    assert [s.status for s in eng.stage_plan.stages] == ["done"] * 3
    assert "stage" in buf.getvalue() and sess.exit_code == 0


def test_resume_of_a_saved_session_continues_it(tmp_path):
    llm1 = FakeLLM([call("write_file", path="a.py", content="print('hi')\n"), say("Wrote a.py.")])
    sess1, eng1, repo, buf1 = make(tmp_path, llm1)
    sess1.app.handle_line("make a.py")
    idle(sess1)
    S.save_session(eng1)
    first_id = eng1.session_id

    llm2 = FakeLLM([say("a.py is already there.")])
    eng2, _repo, _ = session_engine(tmp_path, llm2)
    info = S.prepare_engine(Console(file=io.StringIO()), eng2, eng2.root, None, "latest", False)
    assert info is not None and eng2.session_id == first_id
    sess2, eng2, _, buf2 = make(tmp_path, llm2, eng=eng2)
    sess2._resume_info, sess2.resumed = info, True
    sess2.print_banner(info)
    assert "resumed:" in buf2.getvalue() and "make a.py" in buf2.getvalue() and first_id in buf2.getvalue()
    sess2.app.handle_line("anything left?")
    idle(sess2)
    prior = " ".join(str(m.get("content")) for m in llm2.calls[0]["snapshot"])
    assert "make a.py" in prior


def test_requeued_steer_becomes_the_next_prompt(tmp_path):
    holder = {}

    def last(messages):
        holder["s"].runner.say("one more thing")        # arrives after the final model call
        return say("finished")

    llm = FakeLLM([last, say("handled")])
    sess, eng, repo, buf = make(tmp_path, llm)
    holder["s"] = sess
    sess.app.handle_line("go")
    assert wait_for(lambda: len(llm.calls) == 2 and not sess.app.worker.busy)
    assert eng.ctx.user_requests[-1] == "one more thing" and "queued as the next prompt" in buf.getvalue()


def test_engine_console_lines_are_shown_but_tool_echo_is_not(tmp_path):
    sess, eng, repo, buf = make(tmp_path, FakeLLM([]))
    eng.console.print("  [cyan]›[/cyan] read a.py")
    eng.console.print("    [dim]12 lines[/dim]")
    eng.console.print("[yellow]stopped after 5 steps - say 'continue'[/yellow]")
    eng.console.file.flush()
    out = buf.getvalue()
    assert "stopped after 5 steps" in out and "read a.py" not in out and "12 lines" not in out


# ── real startup (run_code_ui) ─────────────────────────────────────────────

def test_run_code_ui_startup_and_exit_code(tmp_path, monkeypatch):
    llm = FakeLLM([call("write_file", path="b.py", content="b = 2\n"), say("Wrote b.py.")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.config = OFF
    monkeypatch.setattr(CodeRunner, "open_engine", staticmethod(lambda cfg, root, console, test=None: (eng, llm, contextlib.nullcontext())))
    monkeypatch.setattr(S, "acquire_lock", lambda root: (True, ""))
    monkeypatch.setattr(S, "free_gpu", lambda *a: None)
    monkeypatch.setattr(S, "release_gpu_on_exit", lambda *a: None)
    monkeypatch.setattr("agent.foreground.session", lambda cfg: contextlib.nullcontext())
    hist = tmp_path / "h"
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        rc = run_code_ui(str(repo), "make b.py", config=OFF, plain=True, input=inp, output=DummyOutput(), once=True, history_path=hist,
                         width=lambda: 100)
    assert rc == 0 and (repo / "b.py").exists() and SS.latest(repo) is not None
    assert eng.session_id and eng.events.session.endswith(eng.session_id)
