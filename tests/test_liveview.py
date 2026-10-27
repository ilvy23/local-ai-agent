"""Rotating status words, the live Markdown view and the streamed tool-aware chat.

Presentation only: the finished screen must equal a plain print, and nothing raw (tool-call JSON,
hidden reasoning) may ever reach the user's terminal.
"""

import io
import json
import re
from typing import ClassVar

import httpx
import pytest
from rich.console import Console
from rich.markdown import Markdown

from agent import liveview, llm, tui
from agent.liveview import (
    LiveMarkdown,
    LiveReply,
    ThinkingText,
    split_blocks,
)

DOC = (
    "# Titel\n\nErster Absatz mit **fett** und `code`.\n\n"
    "- eins\n- zwei\n- drei\n\n"
    "1. a\n2. b\n\n"
    "```python\nprint('x')\n\nprint('y')\n```\n\nSchluss.\n"
)


def _console(width=60, terminal=False, height=24):
    return Console(file=io.StringIO(), width=width, height=height, force_terminal=terminal, color_system=None)


def _lines(text):
    return [ln.rstrip() for ln in text.rstrip().splitlines() if ln.strip()]


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_CTRL = re.compile(r"\x1b\[(\d*)([A-Za-z])|(\r)|(\n)")


def _final_screen(raw: str) -> str:
    """What a real terminal would actually show after `raw` plays out, line by line.

    Rich's `Live(transient=True)` no longer clears its frame with an explicit erase
    before the caller prints the real thing (as of rich 15); it moves the cursor back
    up and lets the next print overwrite the row in place - correct on a real terminal,
    but `ANSI.sub("", raw)` alone just concatenates both frames since a plain string has
    no notion of "the same row". This replays cursor-up (`\\x1b[nA`), clear-line
    (`\\x1b[2K`/`\\x1b[K`), `\\r` and `\\n` against a line buffer instead, so a count on
    the result reflects what ends up on screen, not what was ever written to the pipe."""
    def write(row: int, col: int, text: str) -> int:
        line = lines[row]
        if len(line) < col:
            line += " " * (col - len(line))
        lines[row] = line[:col] + text + line[col + len(text):]
        return col + len(text)

    lines, row, col = [""], 0, 0
    pos = 0
    for m in _CTRL.finditer(raw):
        col = write(row, col, ANSI.sub("", raw[pos:m.start()]))
        pos = m.end()
        count, letter, cr, lf = m.groups()
        if cr:
            col = 0
        elif lf:
            row += 1
            col = 0
            if row == len(lines):
                lines.append("")
        else:
            n = int(count) if count else 1
            if letter == "A":
                row = max(0, row - n)
            elif letter == "K":  # Rich only ever clears the whole current line here (plain \x1b[2K)
                lines[row] = lines[row][:col] if count == "0" else ""
        # other CSI codes (cursor hide/show, etc.) carry no screen effect we need
    write(row, col, ANSI.sub("", raw[pos:]))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- status line


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_status_line_is_label_dots_and_seconds_only():
    clock = Clock()
    status = ThinkingText("warten", clock=clock)
    assert status.render().plain == "warten."
    clock.t += 3
    plain = status.render().plain
    assert plain.startswith("warten") and plain.endswith("  3s") and "·" not in plain and "…" not in plain


def test_dots_cycle_and_seconds_appear_after_two():
    clock = Clock()
    status = ThinkingText("w", clock=clock)
    seen = []
    for step in (0, 0.6, 0.5, 0.5, 0.5):          # t = 0, 0.6, 1.1, 1.6, 2.1
        clock.t += step
        seen.append(status.render().plain)
    assert [p.split(" ")[0] for p in seen] == ["w.", "w..", "w...", "w.", "w.."]
    assert not any("s" in p for p in seen[:4]) and seen[4].endswith("2s")


# ----------------------------------------------------------------------------- block splitting


def test_blocks_end_at_blank_lines_and_the_last_line_stays_open():
    blocks, tail = split_blocks("Eins.\n\nZwei.\n\nDrei noch nicht fer")
    assert blocks == ["Eins.", "Zwei."] and tail == "Drei noch nicht fer"


def test_a_blank_line_inside_a_code_block_does_not_end_it():
    blocks, tail = split_blocks("```py\na\n\nb\n```\n\nNach")
    assert blocks == ["```py\na\n\nb\n```"] and tail == "Nach"


def test_an_unclosed_code_block_stays_in_the_tail():
    blocks, tail = split_blocks("Text.\n\n```py\na\n\nb")
    assert blocks == ["Text."] and tail.startswith("```py") and "b" in tail


def test_a_list_stays_open_until_a_non_list_block_starts():
    blocks, tail = split_blocks("- a\n- b\n\n")
    assert blocks == [] and "- a" in tail
    blocks, tail = split_blocks("- a\n- b\n\nDan")                      # a non-list block has started
    assert blocks == ["- a\n- b"] and tail == "Dan"
    blocks, tail = split_blocks("- a\n- b\n\nDanach.\n\n")
    assert blocks == ["- a\n- b", "Danach."] and tail == ""


def test_loose_lists_are_not_cut_in_two():
    blocks, tail = split_blocks("1. a\n\n2. b\n\nEnde\n\n")
    assert blocks == ["1. a\n\n2. b", "Ende"] and tail == ""


# ----------------------------------------------------------------------------- live markdown


class FakeLive:
    """Stands in for rich.live.Live (whose frames cannot be inspected without a terminal).
    Like a transient Rich display it erases itself on stop(). Records whether anything gets
    printed while it is on screen - that would corrupt the real display."""

    running: ClassVar[list] = []
    frames: int = 0

    def __init__(self, renderable, console, transient=False, **_kw):
        assert transient, "the live frame must be transient, or finished blocks would show twice"
        self.renderable, self.console = renderable, console

    def start(self):
        FakeLive.running.append(self)

    def update(self, renderable, refresh=False):
        self.renderable = renderable
        FakeLive.frames += 1

    def stop(self):
        FakeLive.running.remove(self)


@pytest.fixture
def fake_live(monkeypatch):
    FakeLive.running, FakeLive.frames = [], 0
    monkeypatch.setattr(liveview, "Live", FakeLive)
    return FakeLive


class GuardedConsole(Console):
    """Fails if something is printed while a (fake) Live display is running."""

    def print(self, *args, **kwargs):
        if FakeLive.running:
            raise AssertionError("printed while the live block was on screen")
        return super().print(*args, **kwargs)


def _guarded(width=60, height=24):
    return GuardedConsole(file=io.StringIO(), width=width, height=height, force_terminal=False, color_system=None)


def _feed(console, text, step):
    live = LiveMarkdown(console)
    for i in range(0, len(text), step):
        live.feed(text[i:i + step])
    live.finish()


@pytest.mark.parametrize("step", [1, 3, 17, 1000])
def test_live_markdown_ends_up_identical_to_a_plain_print(fake_live, step):
    plain = _console()
    plain.print(Markdown(DOC))
    live = _console()
    _feed(live, DOC, step)
    assert _lines(live.file.getvalue()) == _lines(plain.file.getvalue())


def test_finished_blocks_are_printed_once_and_only_the_open_block_is_live(fake_live):
    console = _console()
    live = LiveMarkdown(console)
    live.feed("Erster Absatz.\n\nZweiter Absatz ist noch of")
    assert console.file.getvalue().count("Erster Absatz") == 1 and "Zweiter" not in console.file.getvalue()
    assert len(fake_live.running) == 1                       # the open block is on screen as the live part
    live.feed("fen.\n\nDritter")
    assert console.file.getvalue().count("Zweiter Absatz ist noch offen.") == 1
    live.finish()
    assert fake_live.running == [] and console.file.getvalue().count("Dritter") == 1


def test_live_display_never_runs_while_a_block_is_printed(fake_live):
    console = _guarded()
    _feed(console, DOC, 4)                                   # GuardedConsole raises if that ever happens


def test_live_markdown_on_a_real_terminal_shows_all_text_without_error():
    console = _console(terminal=True)
    _feed(console, DOC, 5)
    shown = ANSI.sub("", console.file.getvalue())
    for needle in ("Titel", "Erster Absatz", "eins", "drei", "print('y')", "Schluss."):
        assert needle in shown


def test_a_very_tall_unfinished_block_is_flushed_instead_of_redrawn_off_screen(fake_live):
    console = _console(height=12)
    text = "\n".join(f"Zeile{n:03d} " + "wort " * 8 for n in range(80))     # one 80-line paragraph, no blank line
    _feed(console, text, 40)
    out = console.file.getvalue()
    assert all(out.count(f"Zeile{n:03d}") == 1 for n in range(80))        # every line exactly once


def test_cutting_inside_a_code_block_keeps_it_a_code_block(fake_live):
    console = _console(height=10)
    body = "\n".join(f"zeile{n:03d} = {n}" for n in range(60))
    _feed(console, f"```python\n{body}\n```\n", 50)
    out = console.file.getvalue()
    assert all(out.count(f"zeile{n:03d}") == 1 for n in range(60))


# ----------------------------------------------------------------------------- live reply gate


def _reply():
    console = _console()
    stopped = []
    return LiveReply(console, "Agent", lambda: stopped.append(1)), console, stopped


def test_a_normal_answer_is_shown_live_and_stops_the_status_line_once():
    reply, console, stopped = _reply()
    for piece in ["Hallo,", " das ist", " eine ganz normale Antwort."]:
        reply.on_content(piece)
    assert reply.started and stopped == [1]
    assert reply.finish("[dim]ctx[/dim]") is True
    out = console.file.getvalue()
    assert out.count("Agent>") == 1 and "normale Antwort" in out and "ctx" in out


@pytest.mark.parametrize("start", ['{"tool": "read_file"', '[{"name"', "<tool_call>", "```json\n{", "<think>hm", "<function=x>"])
def test_tool_calls_and_thinking_text_are_never_shown(start):
    reply, console, stopped = _reply()
    reply.on_content(start)
    reply.on_content(' ... more json ...')
    assert reply.state == "tool" and not stopped
    assert reply.finish() is False and console.file.getvalue() == ""


def test_a_plain_code_block_answer_is_shown():
    reply, _, _ = _reply()
    reply.on_content("```python\nprint(1)\n```\nfertig, das war's schon.")
    assert reply.started


def test_a_very_short_reply_stays_unshown_so_the_caller_prints_it():
    reply, _, stopped = _reply()
    reply.on_content("Hi!")
    assert reply.state == "pending" and reply.finish() is False and not stopped


# ----------------------------------------------------------------------------- streamed chat


def _stream_client(chunks, status=200, error=None):
    seen = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["calls"] += 1
        seen["payload"] = json.loads(request.content)
        if status != 200 and seen["calls"] == 1:
            return httpx.Response(status, json={"error": error})
        return httpx.Response(200, text="\n".join(json.dumps(c) for c in chunks))

    return llm.OllamaClient(transport=httpx.MockTransport(handler)), seen


def _chunk(content="", thinking="", tool_calls=None, done=False, **extra):
    msg = {"role": "assistant", "content": content}
    if thinking:
        msg["thinking"] = thinking
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "done": done, **extra}


def test_stream_delivers_pieces_in_order_and_assembles_the_message():
    client, seen = _stream_client([
        _chunk(thinking="denke"), _chunk("Hal"), _chunk("lo"), _chunk(" Welt"),
        _chunk(done=True, prompt_eval_count=11, eval_count=4),
    ])
    pieces = []
    message = client.chat_with_tools_stream([{"role": "user", "content": "x"}], "m:1", [], num_ctx=65536,
                                            on_content=pieces.append)
    assert pieces == ["Hal", "lo", " Welt"] and message["content"] == "Hallo Welt"
    assert message["_prompt_eval_count"] == 11 and message["_eval_count"] == 4 and "tool_calls" not in message
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["options"] == {"num_predict": 16384, "num_ctx": 65536}


def test_stream_collects_structured_tool_calls():
    call = {"function": {"name": "read_file", "arguments": {"path": "/x"}}}
    client, _ = _stream_client([_chunk("Ich lese. "), _chunk(tool_calls=[call]), _chunk(done=True)])
    message = client.chat_with_tools_stream([], "m:1", [{"type": "function"}], on_content=lambda p: None)
    assert message["tool_calls"] == [call] and message["content"] == "Ich lese. "


def test_stream_retries_without_tools_for_models_that_reject_them():
    client, seen = _stream_client([_chunk("ok"), _chunk(done=True)], status=400,
                                  error="registry.ollama.ai/x does not support tools")
    message = client.chat_with_tools_stream([], "m:1", [{"type": "function"}])
    assert message["content"] == "ok" and seen["calls"] == 2 and "tools" not in seen["payload"]
    client.chat_with_tools_stream([], "m:1", [{"type": "function"}])           # remembered: no doomed first try
    assert seen["calls"] == 3


def test_stream_other_errors_propagate():
    client, _ = _stream_client([], status=500, error="boom")
    with pytest.raises(httpx.HTTPStatusError):
        client.chat_with_tools_stream([], "m:1", [])


def test_deepseek_r1_is_not_streamed_live():
    pieces = []
    client = llm.OllamaClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"message": {"role": "assistant", "content": "<think>x</think>antwort"}})))
    message = client.chat_with_tools_stream([], "deepseek-r1:7b", [], on_content=pieces.append)
    assert pieces == [] and message["content"] == "antwort"


# ----------------------------------------------------------------------------- the agent turn


class _Store:
    def add_message(self, *a, **k):
        pass

    def add_audit_log(self, **k):
        pass


class _StreamingClient:
    """Streams `pieces`, then returns what a real chat_with_tools_stream would."""

    def __init__(self, pieces, tool_calls=None):
        self.pieces, self.tool_calls = pieces, tool_calls
        self.buffered_calls = 0

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None):
        for p in self.pieces:
            on_content(p)
        msg = {"role": "assistant", "content": "".join(self.pieces), "_prompt_eval_count": 100, "_eval_count": 20}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def chat_with_tools(self, messages, model, tools, num_ctx=None):
        self.buffered_calls += 1
        return {"role": "assistant", "content": "gepuffert", "_prompt_eval_count": 1, "_eval_count": 1}


CONFIG = {"models": {"chat": "qwen3.5:9b"}, "tools": {"max_iterations": 4},
          "safety": {"blocked_patterns": [], "safe_commands": [], "max_timeout_s": 5}}


def _turn(client, console):
    from agent.tools.registry import ToolRegistry

    return tui._run_agent_turn(client, CONFIG, console, _Store(), 1, ToolRegistry(),
                               [{"role": "user", "content": "x"}], "Agent")


def test_agent_turn_writes_the_reply_live_exactly_once():
    console = _console(terminal=True)
    reply, tools = _turn(_StreamingClient(["Das ist ", "eine lange, ", "ganz normale Antwort."]), console)
    # Not a plain ANSI-strip: rich's Live(transient=True) now overwrites its frame in place
    # on stop() (cursor-up + redraw) rather than erasing it first, so the raw byte stream
    # legitimately contains the text more than once - only the simulated final screen
    # tells us what actually ends up visible.
    shown = _final_screen(console.file.getvalue())
    assert reply == "Das ist eine lange, ganz normale Antwort." and tools == set()
    assert shown.count("ganz normale Antwort") == 1 and shown.count("Agent>") == 1 and "context:" in shown


def test_agent_turn_without_a_terminal_uses_the_buffered_call():
    console = _console(terminal=False)
    client = _StreamingClient(["x" * 30])
    reply, _ = _turn(client, console)
    assert reply == "gepuffert" and client.buffered_calls == 1


def test_agent_turn_does_not_stream_when_ui_live_is_off(monkeypatch):
    monkeypatch.setattr(tui, "_ui_settings", lambda: {"live": False, "wait_label": "w"})
    console = _console(terminal=True)
    client = _StreamingClient(["x" * 30])
    _turn(client, console)
    assert client.buffered_calls == 1


def test_text_form_tool_call_json_is_never_printed_live():
    console = _console(terminal=True)
    call_json = '{"tool": "no_such_tool", "arguments": {}}'
    client = _StreamingClient([call_json[:20], call_json[20:]])
    # the stream is the model's tool call; after the (failing) tool the next call answers normally
    answers = iter([client.chat_with_tools_stream, lambda *a, **k: {
        "role": "assistant", "content": "Fertig, alles klar mit dir.", "_prompt_eval_count": 5, "_eval_count": 5}])
    client.chat_with_tools_stream = lambda *a, **k: next(answers)(*a, **k)
    reply, _tools = _turn(client, console)
    shown = ANSI.sub("", console.file.getvalue())
    assert '"tool"' not in shown and "no_such_tool" not in shown.replace("no such tool", "")
    assert reply == "Fertig, alles klar mit dir."
