"""Redirect to tui_smoke at run time (tuihint.py), the prompt / add_check wording for terminal repos, the sprawl guard (sprawl.py) and the
8192-token output cap. No model: FakeLLM / a token-by-token stream; programs are written to a temp folder by the tests."""

from __future__ import annotations

import json

import pytest

from agent.codeagent import engine as E, sprawl as SP, tuihint as TH, tuismoke as TS
from agent.llm import GenerationAborted

from tests.test_codeagent_engine import FakeLLM, call, engine, say

FAKE_HEREDOC = """python3 - <<'EOF'
class FakeScreen:
    def getmaxyx(self):
        return (24, 80)
    def addstr(self, *a):
        print(*a)
print("fake screen ready")
EOF"""

TINY_CURSES = """import curses

def main(stdscr):
    stdscr.addstr(0, 0, "SNAKE TITLE SCREEN")
    stdscr.refresh()
    stdscr.getch()

curses.wrapper(main)
"""


def new(tmp_path, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return engine(tmp_path, FakeLLM([]), **kw)


# ── (a) a fake screen ────────────────────────────────────────────────────────

def test_a_heredoc_that_fakes_a_screen_gets_the_hint_once_then_the_short_form(tmp_path):
    eng, _repo, _ = new(tmp_path)
    first = eng.execute("run", {"command": FAKE_HEREDOC})
    assert "fake screen ready" in first and first.count(TH.LONG_HINT) == 1 and "instead of faking a screen" in first
    second = eng.execute("run", {"command": FAKE_HEREDOC.replace("ready", "again")})
    assert TH.LONG_HINT not in second and TH.SHORT_HINT in second and "use tui_smoke" in second


def test_a_monkeypatched_curses_wrapper_is_a_fake_screen_too(tmp_path):
    eng, repo, _ = new(tmp_path)
    (repo / "t.py").write_text("import curses\ncurses.wrapper = lambda f: f(None)\nprint('patched')\n")
    out = eng.execute("run", {"command": "python3 t.py"})
    assert "patched" in out and TH.LONG_HINT in out and "tui_smoke (automatic" not in out          # a fake is not run again


def test_fake_screen_detection_needs_both_parts():
    assert TH.fakes_screen("class MockScreen:\n    def getmaxyx(self): return 24, 80\n")
    assert TH.fakes_screen("monkeypatch.setattr(curses, 'wrapper', fake)") and TH.fakes_screen("mock.patch('curses.initscr')")
    assert not TH.fakes_screen("class Screen:\n    def getmaxyx(self): return 24, 80\n")                # a real class of the game itself
    assert not TH.fakes_screen("class FakeScreen:\n    pass\n")


# ── (b) a terminal program through run ──────────────────────────────────────

def test_a_curses_program_run_without_a_terminal_gets_the_hint_and_its_screen(tmp_path):
    pytest.importorskip("curses")
    eng, repo, _ = new(tmp_path)
    (repo / "snake.py").write_text(TINY_CURSES)
    out = eng.execute("run", {"command": "python3 snake.py"})
    assert not out.startswith("exit code 0"), out
    assert TH.LONG_HINT in out, out
    assert "tui_smoke (automatic, no keys, 3 s timeout):" in out and "SNAKE TITLE SCREEN" in out, out


def test_the_hint_and_the_automatic_screen_come_once_per_program(tmp_path):
    pytest.importorskip("curses")
    eng, repo, _ = new(tmp_path)
    (repo / "snake.py").write_text(TINY_CURSES)
    (repo / "other.py").write_text(TINY_CURSES.replace("SNAKE", "OTHER"))
    one = eng.execute("run", {"command": "python3 snake.py"})
    again = eng.execute("run", {"command": "python3 snake.py"})
    assert TH.LONG_HINT in one and TH.LONG_HINT not in again and TH.SHORT_HINT in again and "tui_smoke (automatic" not in again
    other = eng.execute("run", {"command": "python3 other.py"})
    assert TH.LONG_HINT in other and "OTHER TITLE SCREEN" in other                                  # a different program: the long form again


def test_a_program_that_imports_curses_but_runs_fine_is_not_touched(tmp_path):
    eng, repo, _ = new(tmp_path)
    (repo / "g.py").write_text("import curses\nprint('--help: a curses game')\n")
    out = eng.execute("run", {"command": "python3 g.py"})
    assert out.startswith("exit code 0") and "tui_smoke" not in out


def test_the_terminal_error_texts_are_recognised():
    for text in ("_curses.error: setupterm: could not find terminal", "nocbreak() returned ERR", "Redirection is not supported.",
                 "stdin is not a tty", "OSError: [Errno 25] Inappropriate ioctl for device", "curses.error: cbreak() returned ERR"):
        assert TH.has_terminal_error(text), text
    assert not TH.has_terminal_error("ZeroDivisionError: division by zero")


def test_program_of_follows_one_simple_python_command(tmp_path):
    (tmp_path / "snake.py").write_text("import curses\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__main__.py").write_text("import urwid\n")
    assert TH.program_of("python3 snake.py --seed 3", tmp_path)[0] == "snake.py"
    assert TH.program_of("cd . && python3 -u snake.py", tmp_path)[0] == "snake.py"
    assert TH.program_of("python -m pkg", tmp_path)[0] == "pkg/__main__.py"
    assert TH.program_of("python3 missing.py", tmp_path) is None and TH.program_of("pytest -q", tmp_path) is None
    assert TH.is_tui_source("import curses\n") and not TH.is_tui_source("import os\n")


# ── (c) other programs ───────────────────────────────────────────────────────

def test_a_plain_python_script_is_unaffected(tmp_path):
    eng, repo, _ = new(tmp_path)
    (repo / "hello.py").write_text("print('hello')\n")
    out = eng.execute("run", {"command": "python3 hello.py"})
    assert out.startswith("exit code 0") and "hello" in out and "tui_smoke" not in out
    (repo / "bad.py").write_text("raise SystemExit('boom')\n")
    assert "tui_smoke" not in eng.execute("run", {"command": "python3 bad.py"})
    assert "tui_smoke" not in eng.execute("run", {"command": "python3 - <<'EOF'\nprint(1)\nEOF"})


# ── (f) policy ───────────────────────────────────────────────────────────────

def test_no_automatic_run_when_the_policy_would_ask_or_the_command_pipes(tmp_path):
    pytest.importorskip("curses")
    eng, repo, _ = new(tmp_path)
    (repo / "snake.py").write_text(TINY_CURSES)
    eng.tools.smoke_allow = lambda command: False                                                  # the repo policy would ask for it
    out = eng.execute("run", {"command": "python3 snake.py"})
    assert TH.LONG_HINT in out and "tui_smoke (automatic" not in out
    eng2, repo2, _ = new(tmp_path / "b")
    (repo2 / "snake.py").write_text(TINY_CURSES)
    piped = eng2.execute("run", {"command": "python3 snake.py 2>&1 | head -5"})
    assert TH.LONG_HINT in piped and "tui_smoke (automatic" not in piped
    assert not TH.auto_run_ok("python3 snake.py > out.txt") and not TH.auto_run_ok("python3 snake.py | tee log") and TH.auto_run_ok("python3 snake.py")


def test_a_denied_command_never_runs_and_gets_no_hint(tmp_path):
    from agent.codeagent.engine import DENIED

    seen = []
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="ask", approve=lambda d, why: seen.append(d) or False)
    (repo / "snake.py").write_text(TINY_CURSES)
    (repo / "build").mkdir()
    out = eng.execute("run", {"command": "rm -rf build; python3 snake.py"})
    assert out == DENIED and (repo / "build").is_dir() and "tui_smoke" not in out


def test_the_engine_hands_its_policy_to_the_tools(tmp_path):
    eng, _repo, _ = new(tmp_path)
    assert eng.tools.smoke_allow == eng._allowed_without_asking
    assert eng.tools.smoke_allow("python3 snake.py") is True and eng.tools.smoke_allow("rm -rf /") is False


# ── (e) the prompt and add_check wording ────────────────────────────────────

def test_the_prompt_sentences_and_the_add_check_advice_appear_only_for_a_terminal_repo(tmp_path):
    plain, prepo, _ = new(tmp_path / "plain")
    (prepo / "app.py").write_text("print(1)\n")
    plain.start()
    (tmp_path / "t" / "repo").mkdir(parents=True)                       # before the engine: add_check's description is part of the tool list built at start-up
    (tmp_path / "t" / "repo" / "snake.py").write_text("import curses\n")
    term, _trepo, _ = new(tmp_path / "t")
    term.start()
    head = term.ctx.messages[0]["content"]
    assert TS.PROMPT_SENTENCE in head and "cannot be run with `run` (no terminal)" in head and "the text the player must SEE" in head
    assert len(TS.PROMPT_SENTENCE.split()) <= 60
    assert "tui_smoke" not in plain.ctx.messages[0]["content"]

    def desc(eng):
        return eng.registry.get("add_check").description

    assert desc(term).startswith("For terminal programs prefer kind=tui.")
    assert "prefer kind=tui" not in desc(plain) and "kind=tui" not in desc(plain).split("tui (")[0]        # the kind itself is described for every repo
    assert "tui (command + keys + expect" in desc(plain) and "must SEE" in desc(plain)


def test_the_tui_smoke_description_says_what_to_expect(tmp_path):
    eng, _repo, _ = new(tmp_path)
    tool = eng.registry.get("tui_smoke")
    assert "must SEE" in tool.description and "game over banner" in tool.description and len(tool.description.split()) <= 60


# ── (g)(h) output cap and sprawl guard ──────────────────────────────────────

class TokenLLM(FakeLLM):
    """Streams scripted replies one piece per token (like the server), obeys num_predict like the server (a cut reply is `length`)."""

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, think=None, extra_options=None):
        self.calls.append({"n": len(messages), "think": think, "opts": extra_options})
        pieces = self.script.pop(0)
        cap = (extra_options or {}).get("num_predict", 10 ** 9)
        got = []
        try:
            for i, piece in enumerate(pieces):
                if i >= cap:
                    return {"role": "assistant", "content": "".join(got), "_done_reason": "length", "_eval_count": i}
                got.append(piece)
                on_content(piece)
        except GenerationAborted as stop:
            return {"role": "assistant", "content": "".join(got), "_aborted": str(stop)}
        return {"role": "assistant", "content": "".join(got), "_done_reason": "stop", "_eval_count": len(pieces)}


def prose(n):
    return [f"word{i} " for i in range(n)]


def write_stream(path, n):
    """One write_file call as JSON text of n pieces (one line of code per piece)."""
    return ['{"tool": "write_file", "arguments": {"path": "' + path + '", "content": "'] + [f"v{i} = {i}\\n" for i in range(n - 2)] + ['"}}']


def test_the_default_output_cap_is_8192_and_configurable(tmp_path):
    assert E.NUM_PREDICT == 8192
    eng, _repo, _ = engine(tmp_path, TokenLLM([[ "x" ]]))
    assert eng._limit("num_predict", E.NUM_PREDICT) == 8192
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "num_predict": 12000}}
    eng.llm = TokenLLM([["a"]])
    eng._call_model(False)
    assert eng.llm.calls[0]["opts"]["num_predict"] == 12000


def test_a_prose_sprawl_is_stopped_at_the_threshold_with_the_smaller_steps_instruction(tmp_path):
    llm = TokenLLM([prose(9000), [json.dumps({"tool": "list_dir", "arguments": {"path": "."}})], ["Listed the folder."]])
    eng, _repo, out = engine(tmp_path, llm)
    res = eng.turn("look around")
    assert res.text == "Listed the folder."
    ev = eng.events.of("gen_sprawl")
    assert len(ev) == 1 and ev[0]["pieces"] == SP.SPRAWL_TOKENS + 1 and not eng.events.of("cut_off")
    notes = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "cut off after" in m["content"]]
    assert len(notes) == 1 and "several smaller write_file/append_file calls of at most ~150 lines each" in notes[0] and "6001 tokens" in notes[0]
    assert not any("word10 " in str(m.get("content")) for m in eng.ctx.messages)                  # the sprawl never enters the history
    assert "without a tool call" in out.getvalue()


def test_a_long_single_write_file_of_7000_tokens_is_not_a_sprawl_but_the_hard_cap_stops_it_at_8192(tmp_path):
    llm = TokenLLM([write_stream("big.py", 7000), ["Wrote big.py."]])
    eng, repo, _ = engine(tmp_path, llm)
    res = eng.turn("write big.py")
    assert not eng.events.of("gen_sprawl") and not eng.events.of("cut_off") and res.text == "Wrote big.py."
    assert (repo / "big.py").read_text().count("\n") == 6998
    llm2 = TokenLLM([write_stream("huge.py", 9000), ["Gave up."]])
    (tmp_path / "b").mkdir()
    eng2, repo2, _ = engine(tmp_path / "b", llm2)
    eng2.turn("write huge.py")
    assert not eng2.events.of("gen_sprawl") and eng2.events.of("cut_off")[0]["gen"] == 8192
    assert not (repo2 / "huge.py").exists()                                                         # a cut-off call never runs, not even a part of it
    assert any("cut off after 8192 tokens" in m["content"] for m in eng2.ctx.messages if m["role"] == "system")


def test_the_quiet_window_decides_after_a_call_and_an_open_call_or_fence_is_activity():
    call_text = ['{"tool": "list_dir", "arguments": {"path": "x"', '}}']
    g = SP.SprawlGuard()
    flags = [g.feed(p) for p in prose(5900) + call_text + prose(3000)]
    assert flags.index(True) + 1 == 5902 + SP.SPRAWL_QUIET + 1                  # the call closed at 5902: stopped 2000 quiet tokens later
    g2 = SP.SprawlGuard()
    assert [g2.feed(p) for p in call_text + prose(7000)].index(True) + 1 == SP.SPRAWL_TOKENS + 1      # a call at the start does not protect 6000 tokens of prose
    assert not any(SP.SprawlGuard().feed(p) for p in prose(10))
    g3 = SP.SprawlGuard()
    assert not any(g3.feed(p) for p in ["```python\n"] + prose(7500))           # an OPEN code fence is activity
    g4 = SP.SprawlGuard()
    assert any(g4.feed(p, "thinking") for p in prose(7000))                    # thinking counts and is never structure
    g5 = SP.SprawlGuard()
    assert not any(g5.feed(p) for p in ['{"tool": "write_file", "arguments": {"content": "a } and ``` and \\" quote'] + prose(7500))   # braces / fences inside a string do not end it


def test_a_cut_off_reply_never_runs_its_call_whole_or_half(tmp_path):
    whole = {"role": "assistant", "content": json.dumps({"tool": "write_file", "arguments": {"path": "a.py", "content": "x = 1\n"}}),
             "_done_reason": "length", "_eval_count": 8192}
    half = {"role": "assistant", "content": '{"tool": "write_file", "arguments": {"path": "b.py", "content": "def f():\\n    return', "_done_reason": "length",
            "_eval_count": 8192}
    eng, repo, out = engine(tmp_path, FakeLLM([whole, half, say("Told the user.")]))
    res = eng.turn("write a.py and b.py")
    assert res.text == "Told the user." and not (repo / "a.py").exists() and not (repo / "b.py").exists()
    assert "output limit" in out.getvalue() and len(eng.events.of("cut_off")) == 2
