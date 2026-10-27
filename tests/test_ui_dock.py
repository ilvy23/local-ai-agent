"""Dock: footer/activity row layouts, completion, and real key handling through prompt_toolkit's pipe input."""

import os
import threading

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent.ui import dock as dockmod
from agent.ui.dock import QUIT, Dock, DockHooks, FooterState, NightageCompleter, activity_row, fit_footer


def nap(s):
    threading.Event().wait(s)


FULL = FooterState(model="qwen3.5:9b-instruct-q4", backend="ollama", gpu="dual-gpu", ctx_used=41200, ctx_limit=98304,
                   tok_s=36.1, access="ask", folder=os.path.expanduser("~/code/rps"), queue=2, agents=2,
                   mood="^_^")


@pytest.mark.parametrize("width", [40, 60, 100])
@pytest.mark.parametrize("plain", [True, False])
def test_footer_never_exceeds_width_and_keeps_the_essentials(width, plain):
    line = fit_footer(FULL, width, plain)
    assert 0 < len(line) < width
    assert "qwen3.5" in line and "ctx 42%" in line.replace("41.2k/98.3k ", "") and "queue 2" in line or "q2" in line
    if plain:
        assert line.isascii()


def test_footer_adapts_to_layout_classes():
    wide, normal, narrow = fit_footer(FULL, 160), fit_footer(FULL, 70), fit_footer(FULL, 40)
    assert "41.2k/98.3k 42%" in wide and "ollama dual-gpu" in wide and "36 tok/s" in wide and "~/code/rps" in wide
    assert "ctx 42%" in normal and "41.2k" not in normal
    assert narrow.startswith("qwen3.5") and "ollama" not in narrow and "q2" in narrow
    assert len(wide) > len(normal) > len(narrow)


def test_footer_handles_empty_and_tiny_widths():
    assert fit_footer(FooterState(), 80) == ""
    assert len(fit_footer(FULL, 12)) <= 12
    assert "(paused)" in fit_footer(FooterState(model="m", queue=1, queue_paused=True), 100)


def test_ctx_pct_math():
    assert FooterState(ctx_used=50, ctx_limit=200).ctx_pct == 25
    assert FooterState(ctx_used=50).ctx_pct is None


def test_activity_row_states():
    busy = {"busy": True, "state": "running", "now": "reading foo.py", "elapsed": 12.4, "tokens": 1500, "tok_s": 33.0, "queue": 1}
    row = activity_row(busy, 100, tick=1.0)
    assert "reading foo.py" in row and "12s" in row and "1.5k tok" in row and "33 tok/s" in row and "1 queued" in row
    assert "paused" in activity_row({**busy, "state": "paused"}, 80)
    assert activity_row({**busy, "state": "waiting_approval", "now": "waiting for approval: run_command"}, 80).startswith("?")
    assert "queue paused (2 waiting)" in activity_row({"busy": False, "queue_paused": True, "queue": 2}, 80)
    assert "? keys" in activity_row({"busy": False}, 80)
    for w in (40, 60, 100):
        assert len(activity_row(busy, w, plain=True)) < w and activity_row(busy, w, plain=True).isascii()


def test_spinner_advances_with_time_but_row_stays_ascii_in_plain():
    busy = {"busy": True, "state": "running", "now": "x", "elapsed": 1}
    assert activity_row(busy, 80, tick=0.0) != activity_row(busy, 80, tick=0.15)
    assert activity_row(busy, 80, plain=True, tick=0.0)[0] in "-\\|/"


def completions(text, tmp=None):
    c = NightageCompleter(cwd=(lambda: str(tmp)) if tmp else None)
    return list(c.get_completions(Document(text, len(text)), CompleteEvent()))


def test_slash_completion_is_fuzzy_with_summaries():
    out = completions("/qu")
    names = [c.text for c in out]
    assert "/queue" in names and "/quit" in names
    assert next(c for c in out if c.text == "/queue").display_meta_text
    assert "/compact" in [c.text for c in completions("/cmpt")]               # subsequence
    assert completions("/queue rm 1") == [] and completions("hello") == []


def test_at_file_completion_relative_to_workdir(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "node_modules").mkdir()
    top = [c.text for c in completions("look at @", tmp_path)]
    assert top[0] == "@src/" and "@README.md" in top and "@.hidden" not in top and "@node_modules/" not in top
    assert [c.text for c in completions("see @src/ma", tmp_path)] == ["@src/main.py"]
    assert [c.text for c in completions("@readme", tmp_path)] == ["@README.md"]
    assert completions("@nodir/x", tmp_path) == []


# ----------------------------------------------------------------------------- real key handling


class Probe:
    """Hooks that record what the dock asked for."""

    def __init__(self, busy=False):
        self.busy = busy
        self.calls = []
        self.pull = []
        self.approval = False
        self.dangerous = False
        self.hooks = DockHooks(
            busy=lambda: self.busy, on_stop=lambda: self.calls.append("stop"),
            on_toggle_verbose=lambda: self.calls.append("verbose"), on_keys_help=lambda: self.calls.append("keys"),
            on_clear=lambda: self.calls.append("clear"), pull_back=lambda: self.pull.pop() if self.pull else None,
            approval_pending=lambda: self.approval, approval_dangerous=lambda: self.dangerous,
            on_approval_key=lambda k: self.calls.append("approval:" + k), on_notice=lambda t: self.calls.append("notice:" + t),
            footer=lambda: FULL, activity=lambda: {"busy": self.busy, "state": "running", "now": "working", "elapsed": 3},
            live_tail=lambda: "partial answer so far")


def with_dock(probe, sends, reads=1, pause=0.0, **kw):
    """Type `sends` (chunks) into a pipe while a reader thread calls dock.read_once() `reads` times."""
    results, box = [], []
    with create_pipe_input() as inp:
        def reader():
            with create_app_session(input=inp, output=DummyOutput()):
                d = Dock(probe.hooks, plain=True, input=inp, output=DummyOutput(), width=lambda: 80, live_guard="off", **kw)
                box.append(d)
                with d.active():
                    for _ in range(reads):
                        results.append(d.read_once())
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        for chunk in sends:
            nap(pause)
            inp.send_text(chunk)
        t.join(8)
        assert not t.is_alive(), "dock did not return - missing submit/quit key in the test input?"
    return results, box[0]


def test_enter_sends_and_alt_enter_makes_a_newline():
    res, _ = with_dock(Probe(), ["hello\r", "line1\x1b\rline2\r", "a\x0ab\r"], reads=3, pause=0.2)
    assert res == ["hello", "line1\nline2", "a\nb"]


def test_bracketed_paste_keeps_newlines_in_one_submission():
    res, _ = with_dock(Probe(), ["\x1b[200~first\nsecond\x1b[201~\r"])
    assert res == ["first\nsecond"]


def test_empty_enter_does_not_submit():
    res, _ = with_dock(Probe(), ["\r\rreal\r"])
    assert res == ["real"]


def test_enter_works_while_busy_so_the_app_can_queue():
    p = Probe(busy=True)
    res, _ = with_dock(p, ["queue me\r"])
    assert res == ["queue me"]


def test_up_pulls_back_last_queued_prompt_only_when_busy_and_empty():
    p = Probe(busy=True)
    p.pull = ["queued prompt"]
    res, _ = with_dock(p, ["\x1b[A\r"])
    assert res == ["queued prompt"]
    q = Probe(busy=False)
    q.pull = ["nope"]
    res, _ = with_dock(q, ["\x1b[A\rx\r"])                                    # idle: Up is plain history navigation
    assert res == ["x"] and q.pull == ["nope"]
    r = Probe(busy=True)
    r.pull = ["nope"]
    res, _ = with_dock(r, ["typed\x1b[A\r"])                                 # editor not empty: no pull
    assert r.pull == ["nope"]


def test_up_with_empty_queue_falls_back_to_history():
    p = Probe(busy=True)
    res, _ = with_dock(p, ["old\r", "\x1b[A\r"], reads=2, pause=0.3)
    assert res == ["old", "old"]


def test_escape_stops_only_when_busy():
    p = Probe(busy=True)
    with_dock(p, ["\x1b", "x\r"], pause=0.2)
    assert "stop" in p.calls
    idle = Probe(busy=False)
    with_dock(idle, ["\x1b", "x\r"], pause=0.2)
    assert "stop" not in idle.calls


def test_ctrl_c_stops_when_busy_and_second_ctrl_c_quits():
    p = Probe(busy=True)
    res, _ = with_dock(p, ["\x03\x03"])
    assert p.calls[0] == "stop" and res == [QUIT]


def test_ctrl_c_idle_clears_then_hints_then_quits():
    p = Probe()
    res, _ = with_dock(p, ["draft\x03x\r"])
    assert res == ["x"]                                                       # first Ctrl-C only cleared the draft
    q = Probe()
    res, _ = with_dock(q, ["\x03", "z\r"], pause=0.05, quit_window=0.0)
    assert any(c.startswith("notice:press Ctrl-C again") for c in q.calls)
    res, _ = with_dock(Probe(), ["\x03\x03"])
    assert res == [QUIT]


def test_ctrl_d_quits_on_empty_line_only():
    res, _ = with_dock(Probe(), ["\x04"])
    assert res == [QUIT]
    res, _ = with_dock(Probe(), ["ab\x04\r"])
    assert res == ["ab"]


def test_ctrl_o_ctrl_l_and_question_mark():
    p = Probe()
    with_dock(p, ["\x0f", "\x0c", "?", "x\r"], pause=0.05)
    assert "verbose" in p.calls and "clear" in p.calls and "keys" in p.calls
    q = Probe()
    res, _ = with_dock(q, ["why?\r"])
    assert res == ["why?"] and "keys" not in q.calls                          # ? inside text is just a character


def test_approval_is_never_a_single_key():
    """A message that merely starts with y/n/a/e while an approval is open must not answer it."""
    p = Probe()
    p.approval = True
    res, _ = with_dock(p, ["a", "n", "d", " also add logging\r"], pause=0.05)
    assert res == ["and also add logging"] and not any(c.startswith("approval:") for c in p.calls)
    d = Probe()
    d.approval, d.dangerous = True, True
    res, _ = with_dock(d, ["yes\r"])
    assert res == ["yes"] and not any(c.startswith("approval:") for c in d.calls)


def test_history_file_persists_prompts(tmp_path):
    hp = tmp_path / "state" / "history"
    with_dock(Probe(), ["remember me\r"], history_path=hp)
    assert "remember me" in hp.read_text()


def test_message_has_tail_activity_and_prompt_rows():
    p = Probe(busy=True)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        d = Dock(p.hooks, plain=True, input=inp, output=DummyOutput(), width=lambda: 80, live_guard="off")
        text = "".join(t for _, t in d._message())
        assert "partial answer so far" in text and "working" in text and text.rstrip().endswith(">")
        assert "ctx 42%" in "".join(t for _, t in d._toolbar())


def test_live_guard_blocks_rich_live_while_active():
    from rich.console import Console
    from rich.live import Live
    p = Probe()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        raising = Dock(p.hooks, plain=True, input=inp, output=DummyOutput(), live_guard="raise")
        assert not dockmod.dock_active()
        with raising.active():
            assert dockmod.dock_active()
            with pytest.raises(RuntimeError):
                Live(console=Console(file=open(os.devnull, "w"))).start()
        assert not dockmod.dock_active()
        quiet = Dock(p.hooks, plain=True, input=inp, output=DummyOutput(), live_guard="suppress")
        with quiet.active():
            lv = Live(console=Console(file=open(os.devnull, "w")), transient=True)
            lv.start()                                                          # silently does nothing
            assert not lv._started
            lv.stop()
        orig = Live.start
        assert Live.start is orig and Live.start.__name__ == "start"            # restored after the dock closes


def test_dock_modules_do_not_use_rich_live():
    import inspect

    import agent.ui.app as app
    import agent.ui.render as render
    import agent.ui.worker as worker
    for mod in (app, render, worker):
        src = inspect.getsource(mod)
        assert "rich.live" not in src and "Live(" not in src and ".status(" not in src, mod.__name__
    src = inspect.getsource(dockmod).replace("from rich.live import Live", "")
    assert "Live(" not in src and ".status(" not in src
