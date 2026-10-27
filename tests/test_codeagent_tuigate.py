"""The harness-run crash gate for terminal programs / games (agent/codeagent/tuigate.py): seeded random play on a real pty at the final gate.
Real ptys and tiny curses programs written by the tests; no model (FakeLLM)."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.codeagent import tuigate as TG, tuismoke as TS, verified

from tests.test_codeagent_engine import CFG, FakeLLM, call, engine, say
from tests.test_codeagent_tuismoke import pid_gone

pytest.importorskip("curses")


def _curses_starts_small() -> bool:
    """Can curses start on a 40x12 pty with this environment's TERM? Sandboxes without a usable terminfo entry cannot."""
    import fcntl
    import struct
    import subprocess
    import sys
    import termios
    try:
        master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 12, 40, 0, 0))
        code = "import curses; curses.wrapper(lambda s: s.getmaxyx())"
        r = subprocess.run([sys.executable, "-c", code], stdin=slave, stdout=slave, stderr=slave, timeout=20)   # the environment as the gate sees it
        os.close(master), os.close(slave)
        return r.returncode == 0
    except Exception:  # noqa: BLE001 - any failure to even probe means "cannot run curses here"
        return False


_NEEDS_SMALL_CURSES = pytest.mark.skipif(not _curses_starts_small(), reason="curses cannot initialise on a 40x12 pty here (no usable terminfo entry for TERM)")

BANNER = "GAME OVER - the snake hit the wall"          # 34 characters: wider than a 30-column grid, narrower than 80 columns
SNAKE = '''
import curses

BANNER = "GAME OVER - the snake hit the wall"


def over(scr):
    h, w = scr.getmaxyx()
    PLACE
    scr.refresh()
    while scr.getch() != ord("q"):
        pass


def main(scr):
    curses.curs_set(0)
    scr.keypad(True)
    moves = 0
    while True:
        scr.erase()
        scr.addstr(0, 0, "SCORE %d" % moves)
        scr.refresh()
        if scr.getch() == -1:
            continue
        moves += 1
        if moves >= 3:                      # the snake hits the wall after three moves, whatever the keys were
            over(scr)
            return


if __name__ == "__main__":
    curses.wrapper(main)
'''
CRASH_PLACE = "scr.addstr(h // 2, (30 - len(BANNER)) // 2, BANNER)"                  # the measured bug: negative x -> curses.error
SAFE_PLACE = "scr.addstr(h // 2, max(0, (w - len(BANNER)) // 2), BANNER[: w - 1])"
SWALLOW_PLACE = "try:\n        scr.addstr(h // 2, (30 - len(BANNER)) // 2, BANNER)\n    except curses.error:\n        pass"
SMALL_ONLY_PLACE = "scr.addstr(h // 2, (w - 50) // 2, 'GAME OVER' + ' ' * 41)"        # fine at 80 columns, negative x at 40


def snake(place: str) -> str:
    return SNAKE.replace("PLACE", place)


IGNORES_Q = '''
import curses, os, signal
signal.signal(signal.SIGINT, signal.SIG_IGN)
open("pid.txt", "w").write(str(os.getpid()))


def main(scr):
    scr.keypad(True)
    n = 0
    while True:
        scr.erase()
        scr.addstr(0, 0, "running %d" % n)
        n += 1
        scr.refresh()
        scr.getch()


if __name__ == "__main__":
    curses.wrapper(main)
'''

PY = shlex.quote(sys.executable)


def runner_in(root: Path):
    def run(command, keys, size, timeout):
        return TS.run_smoke(command, keys, size, None, timeout, 50, cwd=root)
    return run


def gate(tmp_path, source, name="snake.py", seconds=2, banners=()):
    (tmp_path / name).write_text(source)
    return TG.run_gate(name, runner_in(tmp_path), seconds=seconds, source=source, banners=list(banners), command=f"{PY} {name}")


def gate_engine(tmp_path, script, seconds=2, **coding):
    tmp_path.mkdir(parents=True, exist_ok=True)
    eng, repo, out = engine(tmp_path, FakeLLM(script))
    eng.config = {**CFG, "coding": {"claims_check": False, "tui_gate_seconds": seconds, **coding}}
    return eng, repo, out


def runs(eng, cached=False):
    return [e for e in eng.events.of("tui_gate") if bool(e.get("cached")) is cached]


# ── the pieces: keys, seed, entry, verdict ──────────────────────────────────

def test_the_key_sequence_is_seeded_bounded_and_ends_with_the_closing_keys():
    a, b = TG.key_sequence(TG.seed_for("print(1)"), 8), TG.key_sequence(TG.seed_for("print(1)"), 8)
    assert a == b and a != TG.key_sequence(TG.seed_for("print(2)"), 8)                 # same content, same keys; other content, other keys
    assert TG.seed_for("print(1)") == TG.seed_for("print(1)") != TG.seed_for("print(2)")
    played = [k for k in a if not k.startswith("wait:")]
    assert played[-3:] == ["Esc", "q", "Ctrl-C"] and len([k for k in played if k in TG.KEYS]) in range(20, 31)
    assert {k for k in played[:-3]} <= set(TG.KEYS)
    waits = [int(k[5:]) for k in a if k.startswith("wait:")]
    assert min(waits) >= 60 and max(waits) <= 200
    assert TG.estimated_seconds(a) <= 8 + 1.5                                          # random part within 8 s, the closing keys a little more
    short = TG.key_sequence(TG.seed_for("x"), 8, small=True)
    assert len([k for k in short if k in TG.KEYS]) < len([k for k in a if k in TG.KEYS]) and short[-1] == "Ctrl-C"
    tiny = TG.key_sequence(5, 1)
    assert TG.estimated_seconds(tiny) < 2.5 and len([k for k in tiny if k in TG.KEYS]) >= 3          # a small budget gives a shorter sequence


def test_three_runs_with_fixed_sizes_and_reproducible_keys(tmp_path):
    seen = []

    def fake(command, keys, size, timeout):
        seen.append((command, list(keys), size))
        return SimpleNamespace(error="", status="exited", exit_code=0, traceback=False, tail=[], screen="hi", last_frame=None, startup_bytes=9,
                               seconds=1.0, signal="", history="hi", cols=80, rows=24)

    first = TG.run_gate("snake.py", fake, seconds=8, source="SRC", command="python3 snake.py")
    again = TG.run_gate("snake.py", fake, seconds=8, source="SRC", command="python3 snake.py")
    assert first.status == "pass" and first.runs == 3 and first.line == "TUI SMOKE: pass (3 runs, 3 s)"
    assert [s for _c, _k, s in seen[:3]] == ["80x24", "80x24", "40x12"] and seen[:3] == seen[3:]          # reproducible
    assert seen[0][1] != seen[1][1] and all(c == "python3 snake.py" for c, _k, _s in seen)
    assert all(k[-5:] == ["q", "wait:120", "Ctrl-C"][-5:] or k[-1] == "Ctrl-C" for _c, k, _s in seen) and again.status == "pass"


def judge_of(**kw):
    base = dict(error="", status="exited", exit_code=0, traceback=False, tail=[], screen="screen", last_frame=None, startup_bytes=50, seconds=1.0,
                signal="", history="", cols=80, rows=24)
    return TG.judge(SimpleNamespace(**{**base, **kw}))


def test_the_verdict_per_run():
    assert judge_of() == ("ok", "")
    kind, why = judge_of(traceback=True, exit_code=1, tail=["Traceback (most recent call last):", "  File x", "_curses.error: addwstr() returned ERR"])
    assert kind == "fail" and why == "_curses.error: addwstr() returned ERR"
    assert judge_of(exit_code=3)[0] == "fail" and "exit code 3" in judge_of(exit_code=3)[1]
    assert judge_of(exit_code=-11, signal="SIGSEGV")[0] == "fail"
    # Ctrl-C is a closing key: the KeyboardInterrupt it raises is not a crash, neither is the exit it causes
    ki = ["Traceback (most recent call last):", "  File x", "KeyboardInterrupt"]
    assert judge_of(traceback=True, exit_code=1, tail=ki) == ("ok", "")
    assert judge_of(exit_code=130)[0] == "ok" and judge_of(exit_code=-2, signal="SIGINT")[0] == "ok"
    assert judge_of(status="running", exit_code=None)[0] == "warn"
    assert judge_of(status="timeout", exit_code=None)[0] == "warn"
    assert judge_of(status="running", exit_code=None, startup_bytes=0, screen="")[0] == "fail"          # killed and never printed anything
    assert judge_of(status="running", exit_code=None, traceback=True, screen="Traceback (most recent call last):\nValueError: bad")[0] == "fail"
    assert judge_of(status="output_cap", exit_code=None)[0] == "warn"


def test_the_entry_is_the_main_file_that_uses_a_terminal_library_newest_edit_first(tmp_path):
    import os
    import time

    (tmp_path / "lib.py").write_text("import curses\n\ndef f():\n    pass\n")                                    # no main guard
    (tmp_path / "plain.py").write_text("print(1)\nif __name__ == '__main__':\n    pass\n")                    # main guard, no terminal library
    (tmp_path / "snake.py").write_text("import curses\nif __name__ == '__main__':\n    pass\n")
    (tmp_path / "pong.py").write_text("import pygame\nif __name__ == \"__main__\":\n    pass\n")
    old = time.time() - 100
    os.utime(tmp_path / "snake.py", (old, old))
    assert TG.find_entry(tmp_path) == "pong.py"                                                                   # newest by mtime
    assert TG.find_entry(tmp_path, edited=[tmp_path / "snake.py"]) == "snake.py"                                  # an edited file beats a newer one
    assert TG.find_entry(tmp_path, override="plain.py") == "plain.py"                                             # coding.tui_entry wins
    assert TG.find_entry(tmp_path, override="missing.py") == "pong.py"                                            # a wrong override is ignored
    (tmp_path / "main.py").write_text("from ui import run\nif __name__ == '__main__':\n    run()\n")
    (tmp_path / "ui.py").write_text("import curses\n\ndef run():\n    pass\n")
    assert TG.find_entry(tmp_path, edited=[tmp_path / "main.py"]) == "main.py"                                    # one level of local import
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "tool.py").write_text("print('hi')\n")
    assert TG.find_entry(empty) is None


def test_banner_words_are_found_only_in_string_literals_the_edit_added():
    before = "print('GAME OVER')\nx = 1\n"
    after = "print('GAME OVER')\nx = 1\nscr.addstr(0, 0, 'You died!')\n# Victory in a comment\nprint(f'Press any key to go on')\n"
    assert TG.banner_words({"a.py": (before, after)}) == ["You died", "Press any key"]
    assert TG.banner_words({"a.py": (None, "print('Game over')\nprint('GAME OVER')\n")}) == ["Game over"]            # one per word, case-insensitive
    assert TG.banner_words({"a.py": (before, before)}) == []
    assert TG.banner_words({"a.md": (None, "GAME OVER")}) == []
    assert TG.banner_words({"a.py": (None, "print('Level cleared!')\n")}) == []                                        # not on the list
    assert TG.banner_words({"a.py": (None, "print('Level cleared!')\n")}, extra=["Level cleared"]) == ["Level cleared"]
    assert TG.banner_words({"a.py": (None, "def f(:\n  'GAME OVER'")}) == []                                        # unparsable: nothing, never an exception


# ── real programs, no engine ────────────────────────────────────────────────

def test_a_banner_wider_than_the_grid_crashes_the_gate(tmp_path):
    r = gate(tmp_path, snake(CRASH_PLACE))
    assert r.status == "fail" and r.crash_line == "_curses.error: addwstr() returned ERR" and r.runs == 1
    assert r.line == "TUI SMOKE: FAIL - _curses.error: addwstr() returned ERR"
    assert "Traceback (most recent call last):" in r.detail and "scr.addstr(h // 2, (30 - len(BANNER)) // 2, BANNER)" in r.detail     # the traceback excerpt
    assert "screen:\nSCORE 2" in r.detail                                                                        # the last frame before the crash
    assert r.detail.startswith("run at 80x24, keys: ") and r.text == r.line + "\n" + r.detail                      # how to reproduce


def test_the_same_program_with_a_safe_banner_passes(tmp_path):
    r = gate(tmp_path, snake(SAFE_PLACE), banners=["GAME OVER"])
    assert r.status == "pass" and r.line == f"TUI SMOKE: pass (3 runs, {r.seconds:.0f} s)", r.line
    assert r.notes == [] and r.warnings == []                                                                      # the banner WAS on a screen


def test_a_swallowed_error_passes_but_the_banner_note_names_the_text(tmp_path):
    r = gate(tmp_path, snake(SWALLOW_PLACE), banners=["GAME OVER"])
    assert r.status == "pass"
    note = 'The text "GAME OVER" exists in the code but never appeared on screen in the random-play runs - check that the player can actually see it.'
    assert r.notes == [note] and r.line.endswith(f" - NOTE: {note}")


@_NEEDS_SMALL_CURSES
def test_a_crash_only_at_40x12_fails_from_the_small_run(tmp_path):
    r = gate(tmp_path, snake(SMALL_ONLY_PLACE))
    assert r.status == "fail" and r.runs == 3 and r.detail.startswith("run at 40x12, keys: ")                      # both 80x24 runs were fine
    assert r.crash_line == "_curses.error: addwstr() returned ERR"


def test_a_program_that_ignores_q_is_a_warning_and_nothing_survives(tmp_path):
    r = gate(tmp_path, IGNORES_Q, name="idle.py")
    assert r.status == "pass" and r.warnings and "still running" in r.warnings[0] and "WARNING" in r.line
    assert pid_gone(int((tmp_path / "pid.txt").read_text()))


def test_a_program_that_never_prints_and_never_ends_fails(tmp_path):
    r = gate(tmp_path, "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(60)\n", name="hang.py")
    assert r.status == "fail" and "no output" in r.crash_line


def test_a_clean_quit_on_the_first_key_is_a_pass(tmp_path):
    r = gate(tmp_path, "import curses\n\ndef main(scr):\n    scr.getch()\n\nif __name__ == '__main__':\n    curses.wrapper(main)\n", name="quick.py")
    assert r.status == "pass" and not r.warnings


# ── in the engine ───────────────────────────────────────────────────────────

def test_the_final_gate_sends_the_model_back_with_the_compact_failure(tmp_path):
    eng, repo, out = gate_engine(tmp_path, [call("write_file", path="snake.py", content=snake(CRASH_PLACE)), say("The snake game is done."), say("Fixed."), say("Fixed.")])
    res = eng.turn("make a snake game")
    nudge = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "crashed when played with random keys" in m["content"]]
    assert len(nudge) == 2 and nudge[0] == nudge[1]                                              # the same failure again: the code was not edited, so not re-run
    assert nudge[0].startswith("Your program crashed when played with random keys. Reproduce with tui_smoke and fix the cause (not by hiding the error).")
    assert "TUI SMOKE: FAIL" in nudge[0] and "curses.error" in nudge[0] and "SCORE" in nudge[0]
    assert len(nudge[0]) < 2500
    assert eng.tui_last.status == "fail" and res.verified is False


def test_it_is_not_run_again_without_a_real_edit_and_the_cap_ends_as_not_verified(tmp_path):
    eng, repo, out = gate_engine(tmp_path, [call("write_file", path="snake.py", content=snake(CRASH_PLACE))] + [say("Everything works now.")] * 6)
    res = eng.turn("make a snake game")
    assert len(runs(eng)) == 1 and len(runs(eng, cached=True)) >= 1                              # the same code is judged once, then from the cache
    nudges = [m for m in eng.ctx.messages if m["role"] == "system" and "crashed when played with random keys" in m["content"]]
    assert len(nudges) == 2                                                                       # per-gate cap 2
    assert res.text.startswith("NOT VERIFIED: program crashes under random play: ") and "curses.error" in res.text.split("\n")[0]
    assert res.verified is False and any(x.startswith("program crashes under random play: ") for x in res.not_verified)


def test_a_fix_is_judged_again_and_passes(tmp_path):
    fix = call("edit_file", path="snake.py", edits=[{"search": CRASH_PLACE, "replace": SAFE_PLACE}])
    eng, repo, out = gate_engine(tmp_path, [call("write_file", path="snake.py", content=snake(CRASH_PLACE)), say("Done."), fix, say("Fixed the banner.")])
    res = eng.turn("make a snake game")
    assert [e["status"] for e in runs(eng)] == ["fail", "pass"] and res.text == "Fixed the banner." and res.verified is True
    assert eng.tui_last.status == "pass" and "TUI SMOKE: pass" in out.getvalue()
    assert "NOT VERIFIED" not in (res.text or "")


def test_a_pass_is_not_run_again_for_the_same_code(tmp_path):
    eng, repo, _ = gate_engine(tmp_path, [call("write_file", path="snake.py", content=snake(SAFE_PLACE)), say("Done."), say("Really done.")])
    eng.turn("make a snake game")
    assert [e["status"] for e in runs(eng)] == ["pass"]
    eng.turn("anything else?")                                                                     # nothing was edited in this prompt: not even judged
    assert len(runs(eng)) == 1


def test_a_non_terminal_repo_skips_the_gate_silently(tmp_path):
    eng, repo, out = gate_engine(tmp_path, [call("write_file", path="tool.py", content="if __name__ == '__main__':\n    print('hi')\n"), say("Done.")])
    res = eng.turn("write a tool")
    assert eng.events.of("tui_gate") == [] and eng.tui_last is None and res.text == "Done." and "TUI SMOKE" not in out.getvalue()


def test_a_terminal_repo_without_an_entry_skips_silently(tmp_path):
    eng, repo, out = gate_engine(tmp_path, [call("write_file", path="widgets.py", content="import curses\n\ndef draw(scr):\n    scr.addstr(0, 0, 'x')\n"), say("Done.")])
    res = eng.turn("write the widgets")
    assert eng.events.of("tui_gate") == [] and res.text == "Done."


def test_editing_only_a_test_file_or_scratch_does_not_run_the_gate(tmp_path):
    eng, repo, _ = gate_engine(tmp_path, [call("write_file", path="test_snake.py", content="def test_x():\n    assert True\n"),
                                          call("write_file", path=".scratch/probe.py", content="print(1)\n"), say("Done.")])
    (repo / "snake.py").write_text(snake(CRASH_PLACE))                                              # a terminal repo with a crashing entry, untouched
    res = eng.turn("add a test")
    assert eng.events.of("tui_gate") == [] and res.text == "Done."


def test_the_switches_in_the_config(tmp_path):
    script = [call("write_file", path="snake.py", content=snake(CRASH_PLACE)), say("Done.")]
    eng, repo, _ = gate_engine(tmp_path / "off", script, tui_gate=False)
    assert eng.turn("make a snake game").text == "Done." and eng.events.of("tui_gate") == []
    (tmp_path / "pick").mkdir()
    eng, repo, _ = gate_engine(tmp_path / "pick", [call("write_file", path="snake.py", content=snake(CRASH_PLACE)),
                                                   call("write_file", path="other.py", content="import curses\nif __name__ == '__main__':\n    pass\n"),
                                                   say("Done.")], tui_entry="other.py")
    eng.turn("two programs")
    assert runs(eng)[0]["entry"] == "other.py" and runs(eng)[0]["status"] == "pass"


def test_banner_words_of_the_config_extend_the_list(tmp_path):
    src = snake(SWALLOW_PLACE).replace("GAME OVER - the snake hit the wall", "LEVEL CLEARED - the snake hit the wall")
    eng, repo, _ = gate_engine(tmp_path, [call("write_file", path="snake.py", content=src), say("Done.")], tui_banner_words=["Level cleared"])
    eng.turn("make a snake game")
    assert eng.tui_last.status == "pass" and 'The text "LEVEL CLEARED" exists in the code but never appeared' in eng.tui_last.line


def test_the_note_reaches_the_result_line_but_never_blocks(tmp_path):
    eng, repo, out = gate_engine(tmp_path, [call("write_file", path="snake.py", content=snake(SWALLOW_PLACE)), say("Done.")])
    res = eng.turn("make a snake game")
    assert res.text == "Done." and res.verified is True and 'The text "GAME OVER" exists in the code' in out.getvalue().replace("\n", " ")


def test_a_save_file_the_program_writes_is_restored(tmp_path):
    src = snake(SAFE_PLACE).replace("def main(scr):", "def main(scr):\n    open('save.json', 'w').write('{\"best\": 0}')")
    eng, repo, _ = gate_engine(tmp_path, [call("write_file", path="snake.py", content=src), say("Done.")])
    (repo / "save.json").write_text('{"best": 41}\n')
    eng.turn("make a snake game")
    assert eng.tui_last.status == "pass" and (repo / "save.json").read_text() == '{"best": 41}\n'


def test_a_command_the_policy_would_ask_about_is_skipped_with_a_note(tmp_path):
    asked = []
    eng, repo, out = engine(tmp_path, FakeLLM([call("write_file", path="snake.py", content=snake(CRASH_PLACE)), say("Done.")]), mode="ask",
                            approve=lambda d, why: asked.append(d) or True)
    eng.config = {**CFG, "coding": {"claims_check": False, "tui_gate_seconds": 2}}
    res = eng.turn("make a snake game")
    assert not [d for d in asked if "python3 snake.py" in d and "py_compile" not in d]                 # the gate never asks
    assert eng.tui_last.status == "skipped" and "policy" in eng.tui_last.line and res.text == "Done."


def test_verified_state_shows_the_cached_line_and_does_not_rerun(tmp_path):
    eng, repo, _ = gate_engine(tmp_path, [call("write_file", path="snake.py", content=snake(SAFE_PLACE)), say("Done.")])
    before = verified.build(eng, run_tests=False)
    assert "TUI SMOKE" not in before.text and "TUI SMOKE" not in before.sections
    eng.turn("make a snake game")
    n = len(eng.events.of("tui_gate"))
    state = verified.build(eng, run_tests=False)
    assert state.sections["TUI SMOKE"]["status"] == "pass" and "TUI SMOKE: pass (3 runs," in state.text
    assert len(eng.events.of("tui_gate")) == n                                                     # cached, not re-run
    again = verified.build(eng, run_tests=False, run_tui=True)
    assert len(eng.events.of("tui_gate")) == n + 1 and again.sections["TUI SMOKE"]["status"] == "pass"


def test_a_failed_smoke_is_a_red_item_for_the_supervisor():
    from agent.codeagent import supervisor

    state = SimpleNamespace(sections={"TUI SMOKE": {"status": "fail", "text": "TUI SMOKE: FAIL - _curses.error: addwstr() returned ERR"}})
    assert supervisor.red_items(state) == ["tui smoke: FAIL - _curses.error: addwstr() returned ERR"]
    assert supervisor.red_items(SimpleNamespace(sections={"TUI SMOKE": {"status": "pass", "text": "TUI SMOKE: pass (3 runs, 9 s)"}})) == []
