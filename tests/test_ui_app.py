"""NightageApp: input routing, instant/step/deferred commands, queue editing, approvals, export, and a few
end-to-end runs through the real dock with a pipe input. No model: turn functions are fakes."""

import io
import json
import threading

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from agent.ui.app import NightageApp
from agent.ui.demo import FakeAgent
from tests._screen import final_screen


def nap(s):
    threading.Event().wait(s)


def wait_for(cond, timeout=4.0):
    end = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if cond():
            return True
        end.wait(0.01)
    return False


class Gate:
    def __init__(self):
        self.release = threading.Event()
        self.ran = []

    def __call__(self, prompt, emit, stop, approve):
        self.ran.append(prompt)
        emit({"type": "status", "state": "thinking", "label": "waiting for the model"})
        while not (self.release.is_set() or stop.is_set()):
            nap(0.005)
        if not stop.is_set():
            emit({"type": "token", "text": f"re: {prompt}\n\n"})
            emit({"type": "done", "reply": f"re: {prompt}"})


class Control:
    def __init__(self):
        self.calls = []

    def request_compact(self):
        self.calls.append(("compact",))

    def queue_user_message(self, text):
        self.calls.append(("say", text))

    def request_steps(self, n):
        self.calls.append(("steps", n))

    def request_mode(self, m):
        self.calls.append(("mode", m))

    def request_stop(self):
        self.calls.append(("stop",))


def make(turn=None, width=80, plain=True, **kw):
    buf = io.StringIO()
    con = Console(file=buf, width=width, force_terminal=False, color_system=None, highlight=False)
    kw.setdefault("history_path", None)
    app = NightageApp(turn or Gate(), console=con, plain=plain, banner=False, exports_dir=kw.pop("exports_dir", None), **kw)
    return app, buf


def busy_app(**kw):
    g = Gate()
    app, buf = make(g, **kw)
    app.handle_line("first")
    assert wait_for(lambda: g.ran == ["first"] and app.worker.busy)
    return app, buf, g


def finish(app, g):
    g.release.set()
    assert app.worker.wait_idle(4)


# ---------------------------------------------------------------- routing


def test_plain_text_starts_a_turn_when_idle_and_streams_output():
    g = Gate()
    g.release.set()
    app, buf = make(g)
    app.handle_line("hello there")
    assert app.worker.wait_idle(4) and g.ran == ["hello there"]
    assert "re: hello there" in buf.getvalue()
    assert app.transcript.turns[0].prompt == "hello there"


def test_enter_while_busy_queues_with_visible_notice():
    app, buf, g = busy_app()
    app.handle_line("second prompt")
    app.handle_line("third")
    assert [i.text for i in app.queue.items()] == ["second prompt", "third"]
    assert "queued (1): second prompt" in buf.getvalue() and "queued (2): third" in buf.getvalue()
    assert app.footer_state().queue == 2
    g.release.set()
    assert wait_for(lambda: g.ran == ["first", "second prompt", "third"] and not app.worker.busy)
    assert "> second prompt" in buf.getvalue()                      # queued prompts are echoed when they start


def test_queue_command_lists_removes_clears_and_moves():
    app, buf, g = busy_app()
    for t in ("a", "b", "c"):
        app.handle_line(t)
    buf.truncate(0), buf.seek(0)
    app.handle_line("/queue")
    out = buf.getvalue()
    assert "queued (3):" in out and "1. a" in out and "3. c" in out
    app.handle_line("/queue rm 2")
    assert [i.text for i in app.queue.items()] == ["a", "c"]
    app.handle_line("/queue now 2")
    assert [i.text for i in app.queue.items()] == ["c", "a"] and "moved to the front" in buf.getvalue()
    app.handle_line("/queue clear")
    assert len(app.queue) == 0 and "cleared 2" in buf.getvalue()
    app.handle_line("/queue rm 9")
    assert "no queued item 9" in buf.getvalue()
    app.handle_line("/queue bogus")
    assert "usage: /queue" in buf.getvalue()
    finish(app, g)


def test_queue_now_when_idle_resumes_and_runs_that_item():
    g = Gate()
    g.release.set()
    app, buf = make(g)
    app.queue.enqueue("one"), app.queue.enqueue("two")
    app.queue.pause()
    app.handle_line("/queue now 2")
    assert wait_for(lambda: "two" in g.ran)
    assert g.ran[0] == "two"


def test_stop_interrupts_pauses_queue_and_resume_continues():
    app, buf, g = busy_app(control=Control())
    app.handle_line("later")
    app.handle_line("/stop")
    assert app.worker.wait_idle(4)
    assert app.queue.paused and len(app.queue) == 1 and g.ran == ["first"]
    out = buf.getvalue()
    assert "interrupted - partial answer kept" in out and "queue paused (1 waiting)" in out
    assert ("stop",) in app.control.calls
    g.release.set()
    app.handle_line("/resume")
    assert wait_for(lambda: g.ran == ["first", "later"])
    assert not app.queue.paused
    assert app.transcript.turns[0].outcome == "interrupted"


def test_stop_when_idle_says_so():
    app, buf = make()
    app.handle_line("/stop")
    assert "nothing is running" in buf.getvalue()


# ---------------------------------------------------------------- classification paths


def test_instant_commands_run_immediately_while_busy():
    app, buf, g = busy_app()
    for cmd in ("/status", "/tasks", "/queue", "/usage", "/help", "/log", "/transcript", "/thinking off", "/context"):
        app.handle_line(cmd)
    out = buf.getvalue()
    assert "state     running" in out and "chat-1" in out and "queue is empty" in out and "Inspect" in out
    assert len(app.queue) == 0 and app.worker.busy
    finish(app, g)


def test_deferred_command_waits_with_notice_then_runs_after_the_turn():
    ran = []
    app, buf, g = busy_app(handlers={"clear": lambda args, a: ran.append(("clear", args)) or "cleared!"})
    app.handle_line("/clear")
    assert ran == [] and "will run after the current turn" in buf.getvalue()
    item = app.queue.items()[0]
    assert item.kind == "command" and item.text == "/clear"
    finish(app, g)
    assert wait_for(lambda: ran == [("clear", "")])
    assert "cleared!" in buf.getvalue()


def test_deferred_command_keeps_fifo_order_with_prompts():
    order = []
    g = Gate()
    app, buf = make(g, handlers={"model": lambda args, a: order.append("model " + args)})
    app.handle_line("p1")
    assert wait_for(lambda: app.worker.busy)
    app.handle_line("p2"), app.handle_line("/model big"), app.handle_line("p3")
    g.release.set()
    assert wait_for(lambda: g.ran == ["p1", "p2", "p3"] and not app.worker.busy)
    assert order == ["model big"]


def test_step_boundary_commands_call_the_control_object():
    ctl = Control()
    app, buf, g = busy_app(control=ctl)
    app.handle_line("/compact")
    app.handle_line("/steps 12")
    app.handle_line("/mode careful")
    app.handle_line("/say please also add tests")
    assert ctl.calls == [("compact",), ("steps", 12), ("mode", "careful"), ("say", "please also add tests")]
    assert "will apply at the next step" in buf.getvalue() and len(app.queue) == 0
    app.handle_line("/steps lots")
    app.handle_line("/say")
    assert "expected a number" in buf.getvalue() and "usage: /say" in buf.getvalue()
    finish(app, g)


def test_step_command_without_control_is_reported_not_fatal():
    app, buf, g = busy_app()
    app.handle_line("/compact")
    assert "no step controls" in buf.getvalue()
    finish(app, g)


def test_unknown_command_is_refused_and_paths_are_prompts():
    g = Gate()
    g.release.set()
    app, buf = make(g)
    app.handle_line("/nonsense")
    assert "unknown command /nonsense" in buf.getvalue() and g.ran == []
    app.handle_line("/home/me/file.txt what is this")
    assert app.worker.wait_idle(4) and g.ran == ["/home/me/file.txt what is this"]


def test_registered_but_unbound_command_gives_a_clear_message():
    app, buf = make()
    app.handle_line("/doctor")
    assert "/doctor is not available in this build yet" in buf.getvalue()


def test_injected_handler_overrides_local_and_may_return_text_or_start_a_turn():
    g = Gate()
    g.release.set()
    app, buf = make(g, handlers={"status": lambda a, app: "custom status", "web": lambda a, app: app.start_turn("searched: " + a)})
    app.handle_line("/status")
    assert "custom status" in buf.getvalue()
    app.handle_line("/web rust release")
    assert wait_for(lambda: g.ran == ["searched: rust release"])


def test_handler_exception_does_not_crash_the_app():
    def boom(a, app):
        raise ValueError("kaput")
    app, buf = make(handlers={"doctor": boom})
    app.handle_line("/doctor")
    assert "/doctor failed: kaput" in buf.getvalue()


def test_help_is_grouped_and_per_command():
    app, buf = make()
    app.handle_line("/help")
    out = buf.getvalue()
    for group in ("Chat", "Code", "Inspect", "Safety & access", "Session"):
        assert group in out
    assert out.index("Chat") < out.index("Inspect") < out.index("Session")
    buf.truncate(0), buf.seek(0)
    app.handle_line("/help queue")
    assert "/queue [rm N | clear | now N]" in buf.getvalue() and "while busy: instant" in buf.getvalue()
    app.handle_line("/help zzz")
    assert "no command zzz" in buf.getvalue()


def test_question_mark_alone_shows_keys():
    app, buf = make()
    app.handle_line("?")
    assert "Alt+Enter" in buf.getvalue() and "Esc / Ctrl-C" in buf.getvalue()


def test_pause_resume_commands():
    app, buf, g = busy_app(control=Control())
    app.handle_line("/pause")
    assert app.worker.state == "paused" and "paused - /resume" in buf.getvalue()
    assert "|| paused" in __import__("agent.ui.dock", fromlist=["x"]).activity_row(app.worker.snapshot(), 80, True)
    app.handle_line("/resume")
    assert app.worker.state == "running"
    app.handle_line("/resume")
    assert "nothing to resume" in buf.getvalue()
    finish(app, g)


def test_status_tasks_and_footer_reflect_the_running_turn():
    app, buf, g = busy_app(footer_provider=lambda: {"model": "m1", "backend": "ollama", "access": "ask"},
                           status_provider=lambda: {"extra": "yes"})
    app.worker.emit({"type": "context", "used": 500, "limit": 1000, "percent": 50, "tokens_per_s": 20})
    fs = app.footer_state()
    assert (fs.model, fs.ctx_pct, fs.queue, fs.agents, fs.state, fs.tok_s) == ("m1", 50, 0, 1, "running", 20)
    app.handle_line("/status")
    out = buf.getvalue()
    assert "model     m1" in out and "context   500/1.0k (50%)" in out and "extra     yes" in out
    app.handle_line("/tasks")
    assert "chat-1" in buf.getvalue() and "[stop,pause]" in buf.getvalue()
    app.handle_line("/tasks stop chat-1")
    assert app.worker.wait_idle(4) and app.worker.last_outcome == "interrupted"


def test_mood_glyph_follows_events_and_can_be_switched_off():
    g = Gate()
    g.release.set()
    app, _ = make(g)
    app.handle_line("hi")
    app.worker.wait_idle(4)
    assert app.footer_state().mood == "*\\0/*"
    off, _ = make(config={"ui": {"mood_emotes": False}})
    off.worker.emit({"type": "error", "message": "x"})
    assert off.footer_state().mood == ""


# ---------------------------------------------------------------- approvals


def approval_app(risk):
    got = []

    def turn(prompt, emit, stop, approve):
        got.append(approve({"tool_name": "run_command", "command": "rm -r build", "risk_level": risk, "reason": "deletes"}))
        emit({"type": "done", "reply": "x"})
    app, buf = make(turn)
    app.handle_line("go")
    assert wait_for(lambda: app.worker.pending_approval is not None)
    return app, buf, got


@pytest.mark.parametrize("answer,approved", [("y", True), ("n", False)])
def test_approval_round_trip_caution(answer, approved):
    app, buf, got = approval_app("CAUTION")
    assert "approval needed [CAUTION]" in buf.getvalue()
    app.handle_line(answer)                                         # while an approval is pending, input answers it
    assert app.worker.wait_idle(4) and bool(got[0]) is approved
    assert ("allowed: run_command" if approved else "denied: run_command") in buf.getvalue()


def test_approval_explain_keeps_waiting():
    app, buf, got = approval_app("CAUTION")
    app.handle_line("e")
    assert "risk CAUTION - deletes" in buf.getvalue() and app.worker.pending_approval is not None
    app.handle_line("and also add logging")                  # not an answer: queued as the next prompt, the approval stays open
    assert "still waiting for your answer" in buf.getvalue() and app.worker.pending_approval is not None
    assert [i.text for i in app.queue.items()] == ["and also add logging"]
    app.handle_line("a")
    assert app.worker.wait_idle(4) and got[0].approve and got[0].always


def test_dangerous_needs_typed_yes():
    app, buf, got = approval_app("DANGEROUS")
    assert "type  yes" in buf.getvalue()
    for wrong in ("y", "a", "Y", "ok"):
        app.handle_line(wrong)
        assert app.worker.pending_approval is not None, wrong
    assert got == []
    app.handle_line("yes")
    assert app.worker.wait_idle(4) and got[0].approve and not got[0].always


def test_dangerous_can_be_denied():
    app, buf, got = approval_app("DANGEROUS")
    app.handle_line("n")
    assert app.worker.wait_idle(4) and not got[0]


# ---------------------------------------------------------------- transcript, thinking, export, log, usage


def run_demo_turn(app):
    app.handle_line("build a game please")
    for _ in range(400):
        if app.worker.pending_approval:
            app.handle_line("y")
        if app.worker.wait_idle(0.02) and not app.worker.busy:
            break


def test_demo_agent_end_to_end_headless():
    app, buf = make(FakeAgent(0))
    run_demo_turn(app)
    out = buf.getvalue()
    for needle in ("plan 3/3", "read_file", "approval needed [CAUTION]", "edit rps/game.py +11 -2", "+def play() -> str:",
                   "2 sources", "thinking (410 tokens)", "What I did", "3 tool calls"):
        assert needle in out, needle


def test_thinking_command_modes():
    app, buf = make(FakeAgent(0))
    run_demo_turn(app)
    assert "The user wants a change" not in buf.getvalue()
    app.handle_line("/thinking show")
    assert "The user wants a change" in buf.getvalue()
    app.handle_line("/thinking on")
    assert app.renderer.show_thinking and "collapsed" not in buf.getvalue().split("/thinking")[-1]
    app.handle_line("/thinking off")
    assert not app.renderer.show_thinking


def test_export_markdown_and_json_files_exclude_thinking(tmp_path):
    app, buf = make(FakeAgent(0), exports_dir=tmp_path / "exp")
    run_demo_turn(app)
    app.handle_line("/export")
    app.handle_line("/export json")
    files = sorted((tmp_path / "exp").iterdir())
    assert [f.suffix for f in files] == [".json", ".md"]
    md = next(f for f in files if f.suffix == ".md").read_text()
    js = json.loads(next(f for f in files if f.suffix == ".json").read_text())
    assert "pytest -q -x" in md and "```diff" in md and "The user wants a change" not in md
    assert js["turns"][0]["outcome"] == "done" and "The user wants" not in json.dumps(js)
    assert all((f.stat().st_mode & 0o777) == 0o600 for f in files)
    target = tmp_path / "mine.md"
    app.handle_line(f"/export md {target}")
    assert target.exists()


def test_log_filters_events_and_usage_lists_turns():
    app, buf = make(FakeAgent(0))
    run_demo_turn(app)
    buf.truncate(0), buf.seek(0)
    app.handle_line("/log tool_call")
    out = buf.getvalue()
    assert out.count("tool_call:") == 3 and "run_command" in out and "token" not in out
    buf.truncate(0), buf.seek(0)
    app.handle_line("/log pytest")
    assert "pytest" in buf.getvalue()
    app.handle_line("/events zzzz")
    assert "no matching events" in buf.getvalue()
    buf.truncate(0), buf.seek(0)
    app.handle_line("/usage")
    out = buf.getvalue()
    assert "done" in out and "1 turn(s)" in out and "9.8k" in out


def test_usage_empty_and_transcript_view():
    app, buf = make(FakeAgent(0))
    app.handle_line("/usage")
    assert "no turns yet" in buf.getvalue()
    run_demo_turn(app)
    buf.truncate(0), buf.seek(0)
    app.handle_line("/transcript")
    out = buf.getvalue()
    assert "build a game please" in out and "import random" in out and "4 passed" in out        # FULL tool results
    assert app.renderer.detail == "normal"


def test_context_uses_provider_or_last_numbers():
    app, buf = make(context_provider=lambda: "system 1k\nfacts 2k")
    app.handle_line("/context")
    assert "system 1k" in buf.getvalue()
    app2, buf2 = make()
    app2.handle_line("/context")
    assert "no context information yet" in buf2.getvalue()
    app2.worker.emit({"type": "context", "used": 100, "limit": 400, "percent": 25})
    app2.handle_line("/context")
    assert "ctx 100/400 (25%)" in buf2.getvalue()


def test_banner_is_a_plain_wordmark_and_a_tips_line():
    buf = io.StringIO()
    con = Console(file=buf, width=100, force_terminal=False, color_system=None)
    app = NightageApp(Gate(), console=con, plain=True, banner=True)
    app.print_banner()
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2 and lines[0].startswith("nightage") and "Esc stop" in lines[1] and buf.getvalue().isascii()
    narrow = io.StringIO()
    NightageApp(Gate(), console=Console(file=narrow, width=40, color_system=None), plain=True).print_banner()
    assert all(len(ln) <= 40 for ln in narrow.getvalue().splitlines())


@pytest.mark.parametrize("width", [40, 60, 100])
def test_demo_output_fits_every_width_in_plain_and_colour(width):
    for plain in (True, False):
        app, buf = make(FakeAgent(0), width=width, plain=plain)
        run_demo_turn(app)
        for cmd in ("/help", "/status", "/usage", "/log", "/tasks", "/queue"):
            app.handle_line(cmd)
        screen = final_screen(buf.getvalue())
        bad = [ln for ln in screen.splitlines() if len(ln) > width]
        assert not bad, (width, plain, bad[:3])
        if plain:
            assert buf.getvalue().isascii(), [c for c in buf.getvalue() if not c.isascii()][:5]


# ---------------------------------------------------------------- through the real dock


class Session:
    """app.run() in a thread with a pipe for keystrokes."""

    def __init__(self, app):
        self.app, self.inp = app, None

    def __enter__(self):
        self._cm = create_pipe_input()
        self.inp = self._cm.__enter__()
        self.app._input, self.app._output = self.inp, DummyOutput()
        self.app._width = lambda: 80
        self.app._live_guard = "raise"
        self.rc = []

        def go():
            with create_app_session(input=self.inp, output=DummyOutput()):
                self.rc.append(self.app.run())
        self.t = threading.Thread(target=go, daemon=True)
        self.t.start()
        return self

    def type(self, text, pause=0.12):
        nap(pause)
        self.inp.send_text(text)

    def __exit__(self, *a):
        self.t.join(6)
        alive = self.t.is_alive()
        self._cm.__exit__(*a)
        assert not alive, "app.run() did not return"


def test_dock_end_to_end_queue_pull_back_stop_and_quit():
    g = Gate()
    app, buf = make(g)
    with Session(app) as s:
        s.type("first\r")
        assert wait_for(lambda: g.ran == ["first"] and app.worker.busy)
        s.type("second\r")
        assert wait_for(lambda: len(app.queue) == 1)
        assert "queued (1): second" in buf.getvalue()
        s.type("\x1b[A")                                            # Up with an empty editor pulls it back
        assert wait_for(lambda: len(app.queue) == 0)
        assert "pulled back from the queue" in buf.getvalue()
        assert wait_for(lambda: app.dock.session.default_buffer.text == "second")
        s.type("\x1b")                                              # Esc stops the turn
        assert app.worker.wait_idle(4) and app.worker.last_outcome == "interrupted"
        s.type("\x03")                                              # clears the pulled-back draft
        assert wait_for(lambda: app.dock.session.default_buffer.text == "")
        nap(2.1)
        s.type("/quit\r")
    assert app.quit_requested


def test_dock_ctrl_c_while_busy_stops_second_ctrl_c_quits():
    g = Gate()
    app, buf = make(g)
    with Session(app) as s:
        s.type("work\r")
        assert wait_for(lambda: app.worker.busy)
        s.type("\x03")
        assert app.worker.wait_idle(4) and app.worker.last_outcome == "interrupted"
        s.type("\x03")                                              # second one within the window leaves the app
    assert not app.worker.busy


def test_dock_slash_quit_while_idle_returns_zero():
    app, _ = make()
    with Session(app) as s:
        s.type("/help\r")
        s.type("/quit\r")
    assert app.quit_requested and s.rc == [0]


def test_dock_deferred_quit_runs_after_the_turn():
    g = Gate()
    app, buf = make(g)
    with Session(app) as s:
        s.type("work\r")
        assert wait_for(lambda: app.worker.busy)
        s.type("/quit\r")
        assert wait_for(lambda: len(app.queue) == 1) and app.worker.busy
        g.release.set()
    assert s.rc == [0] and app.quit_requested


def test_dock_approval_needs_the_word_and_enter_and_commands_still_work():
    got = []

    def turn(prompt, emit, stop, approve):
        got.append(approve({"tool_name": "edit", "command": "x.py", "risk_level": "CAUTION"}))
        emit({"type": "done", "reply": "ok"})
    app, buf = make(turn)
    with Session(app) as s:
        s.type("do\r")
        assert wait_for(lambda: app.worker.pending_approval is not None)
        s.type("y")                                                 # a single key is only text now ...
        nap(0.2)
        assert app.worker.pending_approval is not None and app.dock.session.default_buffer.text == "y"
        s.type("\x7f/status\r")                                    # ... and commands run while the approval waits
        assert wait_for(lambda: "state     waiting_approval" in buf.getvalue())
        assert app.worker.pending_approval is not None
        s.type("y\r")
        assert app.worker.wait_idle(4) and got[0].approve
        s.type("/quit\r")


def test_dock_dangerous_needs_the_typed_word():
    got = []

    def turn(prompt, emit, stop, approve):
        got.append(approve({"tool_name": "run_command", "command": "rm -r x", "risk_level": "DANGEROUS"}))
    app, buf = make(turn)
    with Session(app) as s:
        s.type("do\r")
        assert wait_for(lambda: app.worker.pending_approval is not None)
        s.type("y")
        nap(0.2)
        assert app.worker.pending_approval is not None and app.dock.session.default_buffer.text == "y"
        s.type("\x7fyes\r")
        assert app.worker.wait_idle(4) and got[0].approve
        s.type("/quit\r")
