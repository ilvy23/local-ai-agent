"""Renderer cards, transcript export, widths and plain/ASCII mode. Headless: a Console writing to a StringIO."""

import io
import json
import re

import pytest
from rich.console import Console

from agent.ui.events import UiEvent
from agent.ui.render import Renderer, Transcript, next_step_for, to_event
from tests._screen import final_screen


def make(width=80, plain=False, detail="normal", term=False):
    buf = io.StringIO()
    con = Console(file=buf, width=width, force_terminal=term, color_system="standard" if term else None, highlight=False)
    return Renderer(con, Transcript(), detail, plain), buf


def feed(r, *events):
    for e in events:
        r.handle(e)


def lines(buf):
    return [ln.rstrip() for ln in buf.getvalue().splitlines() if ln.strip()]


DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n-old\n+new\n+more\n"


def tool_events(name="read_file", args=None, result="ok", risk="SAFE"):
    return [{"type": "tool_call", "id": "1", "name": name, "arguments": args or {"path": "a.py"}, "risk": risk},
            {"type": "tool_result", "id": "1", "name": name, "result": result, "ok": True}]


def test_to_event_accepts_dicts_and_events():
    e = to_event({"type": "token", "text": "hi"})
    assert e.kind == "token" and e.data == {"text": "hi"}
    u = UiEvent("done", {})
    assert to_event(u) is u
    assert to_event("weird").kind == "log"


def test_text_streams_as_complete_blocks_and_tail_is_exposed():
    r, buf = make(plain=True)
    r.begin_turn("hi")
    feed(r, {"type": "token", "text": "First para"})
    assert buf.getvalue() == "" and r.tail() == "First para"          # nothing printed, partial only in the live tail
    feed(r, {"type": "token", "text": " done.\n\nSecond"})
    assert "First para done." in buf.getvalue() and "Second" not in buf.getvalue()
    assert r.tail() == "Second"
    feed(r, {"type": "done", "reply": "x"})
    assert "Second" in buf.getvalue() and r.tail() == ""


def test_done_prints_reply_when_nothing_was_streamed():
    r, buf = make(plain=True)
    r.begin_turn("hi")
    feed(r, {"type": "done", "reply": "whole answer"})
    assert "whole answer" in buf.getvalue()


def test_tool_card_collapsed_normal_and_expanded_verbose():
    r, buf = make(plain=True)
    feed(r, *tool_events("run_command", {"command": "ls -la"}, "total 0\nfile", risk="CAUTION"))
    out = lines(buf)
    assert len(out) == 1 and "run_command" in out[0] and "ls -la" in out[0] and "[CAUTION]" in out[0] and "total 0" not in out[0]
    v, vbuf = make(plain=True, detail="verbose")
    feed(v, *tool_events("run_command", {"command": "ls -la"}, "total 0\nfile", risk="CAUTION"))
    text = vbuf.getvalue()
    assert '"command": "ls -la"' in text and "total 0" in text and "file" in text      # FULL args and result


def test_tool_error_is_marked_and_quiet_hides_ok_calls():
    r, buf = make(plain=True)
    feed(r, {"type": "tool_call", "id": "1", "name": "web_search", "arguments": {"query": "x"}},
         {"type": "tool_result", "id": "1", "name": "web_search", "result": "BLOCKED by safety policy: no", "ok": False})
    line = lines(buf)[0]
    assert line.startswith("x ") and "BLOCKED" in line
    q, qbuf = make(plain=True, detail="quiet")
    feed(q, *tool_events())
    assert qbuf.getvalue() == ""


def test_edit_card_shows_inline_diff_and_counts():
    r, buf = make(plain=True)
    feed(r, {"type": "edit", "path": "x.py", "added": 2, "removed": 1, "diff": DIFF})
    out = buf.getvalue()
    assert "edit x.py +2 -1" in out and "+new" in out and "-old" in out
    c, cbuf = make(term=True)                           # colour: Syntax highlighting emits colour codes for +/-
    feed(c, {"type": "edit", "path": "x.py", "diff": DIFF})
    assert "\x1b[" in cbuf.getvalue() and "+more" in re.sub(r"\x1b\[[0-9;]*m", "", cbuf.getvalue())


def test_edit_card_replaces_the_tool_card_of_the_same_call_but_other_tools_keep_theirs():
    r, buf = make(plain=True)
    feed(r, {"type": "tool_call", "name": "edit_file", "arguments": {"path": "x.py"}},
         {"type": "edit", "path": "x.py", "added": 2, "removed": 1, "diff": DIFF},
         {"type": "tool_result", "name": "edit_file", "result": "edited x.py", "ok": True},
         {"type": "tool_call", "name": "grep", "arguments": {"pattern": "a"}},
         {"type": "tool_result", "name": "grep", "result": "x.py:1: a", "ok": True})
    out = buf.getvalue()
    assert "edit x.py +2 -1" in out and "edit_file" not in out and "grep" in out        # one line per edit, still one per other tool
    v, vbuf = make(plain=True, detail="verbose")
    feed(v, {"type": "tool_call", "name": "edit_file", "arguments": {"path": "x.py"}},
         {"type": "edit", "path": "x.py", "added": 1, "removed": 0, "diff": DIFF},
         {"type": "tool_result", "name": "edit_file", "result": "edited x.py", "ok": True})
    assert "edit_file" in vbuf.getvalue()                                               # verbose keeps everything


def test_long_diff_is_cut_in_normal_but_complete_in_verbose():
    big = "\n".join(f"+line{i}" for i in range(60))
    r, buf = make(plain=True)
    feed(r, {"type": "edit", "path": "big.py", "diff": big})
    assert "line10" in buf.getvalue() and "line59" not in buf.getvalue() and "more lines" in buf.getvalue()
    v, vbuf = make(plain=True, detail="verbose")
    feed(v, {"type": "edit", "path": "big.py", "diff": big})
    assert "line59" in vbuf.getvalue()


def test_thinking_is_collapsed_kept_in_memory_and_expandable():
    r, buf = make(plain=True)
    feed(r, {"type": "thinking", "tokens": 1240, "text": "secret reasoning"}, {"type": "token", "text": "Answer\n\n"})
    first = lines(buf)[0]
    assert "thinking (1.2k tokens)" in first and "secret reasoning" not in buf.getvalue()
    assert r.last_thinking() == [(1240, "secret reasoning")]
    assert r.render_thinking() and "secret reasoning" in buf.getvalue()
    s, sbuf = make(plain=True)
    s.show_thinking = True
    feed(s, {"type": "thinking", "text": "visible now"}, {"type": "done", "reply": "x"})
    assert "visible now" in sbuf.getvalue()


def test_plan_stage_verify_stuck_lines():
    r, buf = make(plain=True)
    feed(r, {"type": "plan", "items": [{"text": "a", "status": "confirmed"}, {"text": "b", "status": "open"}], "confirmed": 1, "total": 2},
         {"type": "stage_start", "name": "build"}, {"type": "verify", "state": "failed", "command": "pytest", "attempt": 1, "max_attempts": 2},
         {"type": "stuck", "kind": "lookup", "query": "q"})
    out = buf.getvalue()
    assert "plan 1/2" in out and "[x] a" in out and "[ ] b" in out and "stage build" in out
    assert "verify failed (1/2) pytest" in out and "stuck - repeating the same lookup: q" in out


def test_sources_and_verdict():
    r, buf = make(plain=True)
    feed(r, {"type": "sources", "query": "rust", "items": [
        {"n": 1, "domain": "blog.rust-lang.org", "trust": {"level": "high"}},
        {"n": 2, "domain": "spam.example", "trust": {"level": "low"}, "flags": ["suspicious_instructions"]}]},
         {"type": "verdict", "label": "single_source", "summary": "1 independent", "warnings": ["figure 43 appears nowhere"]})
    out = buf.getvalue()
    assert "2 sources for rust" in out and "[1] high" in out and "!suspicious_instructions" in out
    assert "single source: 1 independent" in out and "figure 43" in out


def test_approval_card_levels():
    r, buf = make(plain=True)
    feed(r, {"type": "awaiting_approval", "tool_name": "run_command", "command": "rm -r build", "risk_level": "DANGEROUS",
             "reason": "deletes files"})
    out = buf.getvalue()
    assert "approval needed [DANGEROUS]" in out and "rm -r build" in out and "type  yes" in out and "deletes files" in out
    c, cbuf = make()
    feed(c, {"type": "awaiting_approval", "tool_name": "edit", "command": "x.py", "risk_level": "CAUTION", "diff": DIFF})
    out = cbuf.getvalue()
    assert "y allow once, n deny, a always for this folder, e explain" in out and "+new" in out and "approval needed [CAUTION]" in out


@pytest.mark.parametrize("msg,hint", [("Connection refused", "ollama serve"), ("model 'x' not found", "ollama pull"),
                                       ("context length exceeded", "/compact"), ("boom", "/log")])
def test_error_card_gives_a_next_step(msg, hint):
    assert hint in next_step_for(msg)
    r, buf = make(plain=True)
    feed(r, {"type": "error", "message": msg})
    assert msg in buf.getvalue() and "next:" in buf.getvalue() and hint in buf.getvalue()


def test_usage_line_after_done_and_quiet_mode():
    r, buf = make(plain=True)
    r.begin_turn("hi")
    feed(r, {"type": "context", "used": 4200, "limit": 8000, "percent": 52, "tokens_per_s": 33.0}, *tool_events(), {"type": "done", "reply": "yo"})
    assert "ctx 4.2k (52%)" in buf.getvalue() and "33 tok/s" in buf.getvalue() and "1 tool call" in buf.getvalue()
    q, qbuf = make(plain=True, detail="quiet")
    q.begin_turn("hi")
    feed(q, {"type": "thinking", "tokens": 5}, {"type": "plan", "items": []}, {"type": "done", "reply": "yo"})
    assert lines(qbuf) == ["yo"]


def test_cycle_verbose_toggles():
    r, _ = make()
    assert r.cycle_verbose() == "verbose" and r.cycle_verbose() == "normal"


@pytest.mark.parametrize("width", [40, 60, 100])
@pytest.mark.parametrize("plain", [True, False])
def test_every_card_fits_the_width(width, plain):
    r, buf = make(width, plain)
    r.begin_turn("x")
    long = "very/long/path/" * 8 + "file.py"
    feed(r, {"type": "thinking", "tokens": 300, "text": "t"},
         *tool_events("read_file", {"path": long}, "res" * 80),
         {"type": "edit", "path": long, "diff": DIFF},
         {"type": "plan", "items": [{"text": "word " * 30, "status": "open"}], "confirmed": 0, "total": 1},
         {"type": "sources", "query": "q" * 90, "items": [{"n": 1, "domain": "d" * 70, "trust": {"level": "high"}}]},
         {"type": "awaiting_approval", "tool_name": "run_command", "command": "echo " * 40, "risk_level": "CAUTION"},
         {"type": "error", "message": "Connection refused " * 5},
         {"type": "token", "text": "Some **markdown** answer " * 10 + "\n\n"},
         {"type": "done", "reply": ""})
    screen = final_screen(buf.getvalue())
    assert all(len(ln) <= width for ln in screen.splitlines()), [ln for ln in screen.splitlines() if len(ln) > width]
    assert "file.py" in screen or "…" in screen or "..." in screen


def test_plain_mode_is_pure_ascii_and_has_no_escape_codes():
    r, buf = make(60, True)
    r.begin_turn("x")
    feed(r, {"type": "thinking", "tokens": 9, "text": "t"}, *tool_events("grep", {"pattern": "p" * 90}),
         {"type": "edit", "path": "a.py", "diff": DIFF}, {"type": "plan", "items": [{"text": "z" * 90}], "total": 1},
         {"type": "sources", "query": "q", "items": [{"n": 1, "domain": "d" * 80, "trust": {"level": "high"}}]},
         {"type": "awaiting_approval", "tool_name": "t", "command": "c", "risk_level": "DANGEROUS"},
         {"type": "error", "message": "Connection refused"}, {"type": "done", "reply": "ok"})
    out = buf.getvalue()
    assert out.isascii() and "\x1b" not in out


def test_transcript_records_everything_and_exports_without_thinking():
    r, _ = make(plain=True)
    r.begin_turn("do it")
    feed(r, {"type": "thinking", "tokens": 50, "text": "PRIVATE-THOUGHT"}, *tool_events("run_command", {"command": "ls"}, "FULLRESULT"),
         {"type": "edit", "path": "a.py", "added": 1, "removed": 0, "diff": DIFF}, {"type": "token", "text": "The answer"},
         {"type": "done", "reply": "The answer"})
    tr = r.transcript
    assert [e.kind for e in tr.events][:2] == ["user", "thinking"] and tr.turns[0].outcome == "done" and tr.turns[0].tools == 1
    md = tr.to_markdown()
    assert "do it" in md and "FULLRESULT" in md and '"command": "ls"' in md and "```diff" in md and "The answer" in md
    assert "PRIVATE-THOUGHT" not in md and "thinking omitted" in md
    js = tr.to_json()
    assert "PRIVATE-THOUGHT" not in js and json.loads(js)["turns"][0]["tools"] == 1
    assert "PRIVATE-THOUGHT" in tr.to_markdown(include_thinking=True) and "PRIVATE-THOUGHT" in tr.to_json(include_thinking=True)


def test_transcript_select_filters_by_kind_and_text():
    tr = Transcript()
    for ev in ({"type": "log", "text": "searching foo"}, {"type": "error", "message": "bad thing"}, {"type": "log", "text": "other"}):
        tr.record(to_event(ev))
    assert [e.kind for e in tr.select("error")] == ["error"]
    assert len(tr.select("foo")) == 1 and len(tr.select("")) == 3


def test_print_transcript_replays_with_full_detail():
    r, buf = make(plain=True)
    r.begin_turn("go")
    feed(r, *tool_events("run_command", {"command": "ls"}, "LONGOUTPUT"), {"type": "done", "reply": "fin"})
    buf.truncate(0), buf.seek(0)
    r.print_transcript()
    assert "LONGOUTPUT" in buf.getvalue() and "go" in buf.getvalue()
    assert r.detail == "normal"


def test_renderer_never_uses_rich_live():
    import inspect

    import agent.ui.render as mod
    src = inspect.getsource(mod)
    assert "rich.live" not in src and "Live(" not in src and ".status(" not in src


def test_consecutive_thinking_pieces_make_one_card_with_one_index():
    r, buf = make(plain=True)
    evs = []
    for i in range(8):
        evs += [{"type": "thinking", "text": f"step {i} "}, {"type": "status", "state": "thinking", "tokens": i}]
    feed(r, *evs, {"type": "token", "text": "Answer\n\n"})
    out = lines(buf)
    assert sum("thinking (" in ln for ln in out) == 1 and "/thinking show 1" in buf.getvalue()
    assert "/thinking show 2" not in buf.getvalue()
    assert len(r.last_thinking()) == 1 and "step 0 step 1" in r.last_thinking()[0][1]
    assert r.render_thinking(1) and "step 7" in buf.getvalue()


def test_thinking_without_counts_is_estimated_from_the_joined_text():
    r, buf = make(plain=True)
    feed(r, {"type": "thinking", "text": "a" * 40}, {"type": "thinking", "text": "b" * 40}, {"type": "done", "reply": "x"})
    assert r.last_thinking() == [(20, "a" * 40 + "b" * 40)]


def test_a_tool_call_between_thinking_makes_two_cards():
    r, buf = make(plain=True)
    feed(r, {"type": "thinking", "text": "one"}, *tool_events(), {"type": "thinking", "text": "two"}, {"type": "done", "reply": "x"})
    assert len(r.last_thinking()) == 2 and "/thinking show 2" in buf.getvalue()
