"""tui_smoke: a real pty, a small VT emulator, keys, answers to terminal queries, the policy of `run`, the loop guard's family, the `tui` check kind.
No model: programs are written to a temp folder by the tests; the engine runs with FakeLLM."""

from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path

import pytest

from agent.codeagent import checks as checks_mod, tuismoke as TS
from agent.codeagent import loopguard as LG

from tests.test_codeagent_engine import FakeLLM, call, engine, say
from tests.test_codeagent_loopguard import fresh, orient, playt

PY = shlex.quote(sys.executable)


def prog(tmp_path: Path, name: str, text: str) -> str:
    """Write a program, return the command that runs it."""
    (tmp_path / name).write_text(text)
    return f"{PY} {name}"


def smoke(tmp_path, command, **kw):
    kw.setdefault("timeout", 8)
    return TS.run_smoke(command, cwd=tmp_path, **kw)


def pid_gone(pid: int, wait: float = 3.0) -> bool:
    """True when the process is gone (a zombie that a non-reaping init has not collected counts as gone)."""
    end = time.monotonic() + wait
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except (OSError, IndexError):
            return True
        time.sleep(0.05)
    return False


def screen_of(text: str, cols: int = 20, rows: int = 6) -> TS.Screen:
    s = TS.Screen(cols, rows)
    s.feed(text)
    return s


# ── the emulator ────────────────────────────────────────────────────────────

def test_printing_wrapping_and_the_deferred_wrap_at_the_last_column():
    s = screen_of("abcdefghij" "klmnopqrst" "u", cols=10, rows=3)
    assert s.text() == "abcdefghij\nklmnopqrst\nu"
    s = screen_of("abcdefghij", cols=10, rows=3)
    assert (s.y, s.x) == (0, 9)                                     # the cursor stays on the last column until the next character
    s.feed("\r\n")
    assert s.text() == "abcdefghij" and (s.y, s.x) == (1, 0)        # a CRLF after a full line adds no blank line


def test_cr_lf_backspace_tab_and_bell():
    s = screen_of("ab\bX\x07\tY\rZ")
    assert s.text() == "ZX      Y" and s.x == 1


@pytest.mark.parametrize("seq, pos", [
    ("\x1b[3;5H", (2, 4)), ("\x1b[3;5f", (2, 4)), ("\x1b[H", (0, 0)), ("\x1b[2;3H\x1b[A", (0, 2)), ("\x1b[2;3H\x1b[2B", (3, 2)),
    ("\x1b[2;3H\x1b[3C", (1, 5)), ("\x1b[2;8H\x1b[3D", (1, 4)), ("\x1b[2;3H\x1b[E", (2, 0)), ("\x1b[3;3H\x1b[F", (1, 0)),
    ("\x1b[9G", (0, 8)), ("\x1b[4d", (3, 0)), ("\x1b[2;3H\x1b[0A", (0, 2)), ("\x1b[99;99H", (5, 19)), ("\x1b[3;3H\x1b[99A", (0, 2)),
])
def test_cursor_movement(seq, pos):
    s = screen_of(seq)
    assert (s.y, s.x) == pos


def test_erase_in_display_and_line():
    full = "".join(f"{c * 10}\r\n" for c in "abcd")
    s = screen_of(full + "\x1b[2;4H\x1b[J", cols=10, rows=6)
    assert s.text().splitlines() == ["aaaaaaaaaa", "bbb"]
    s = screen_of(full + "\x1b[2;4H\x1b[1J", cols=10, rows=6)
    assert s.text().splitlines() == ["", "    bbbbbb", "cccccccccc", "dddddddddd"]
    s = screen_of(full + "\x1b[2J", cols=10, rows=6)
    assert s.text() == "" and (s.y, s.x) == (4, 0)                  # ED 2 does not move the cursor
    s = screen_of("abcdef\x1b[1;4H\x1b[K", cols=10, rows=2)
    assert s.text() == "abc"
    s = screen_of("abcdef\x1b[1;4H\x1b[1K", cols=10, rows=2)
    assert s.text() == "    ef"
    s = screen_of("abcdef\x1b[2K", cols=10, rows=2)
    assert s.text() == ""


def test_insert_delete_lines_and_chars():
    base = "1\r\n2\r\n3\r\n4"
    assert screen_of(base + "\x1b[2;1H\x1b[L", cols=5, rows=4).text() == "1\n\n2\n3"
    assert screen_of(base + "\x1b[2;1H\x1b[M", cols=5, rows=4).text() == "1\n3\n4"
    assert screen_of(base + "\x1b[2;1H\x1b[2M", cols=5, rows=4).text() == "1\n4"
    assert screen_of("abcdef\x1b[1;2H\x1b[2P", cols=10, rows=2).text() == "adef"
    assert screen_of("abcdef\x1b[1;2H\x1b[2@", cols=10, rows=2).text() == "a  bcdef"
    assert screen_of("abcdef\x1b[1;2H\x1b[3X", cols=10, rows=2).text() == "a   ef"


def test_scroll_region_scrolling_and_the_sequences_s_and_t():
    s = screen_of("1\r\n2\r\n3\r\n4\r\n5\x1b[2;4r", cols=5, rows=5)        # region rows 2-4, cursor homed
    assert (s.y, s.x) == (0, 0)
    s.feed("\x1b[4;1H\n")                                           # a line feed at the bottom of the region scrolls the region only
    assert s.text().splitlines() == ["1", "3", "4", "", "5"]
    s.feed("\x1b[S")
    assert s.text().splitlines() == ["1", "4", "", "", "5"]
    s.feed("\x1b[T")
    assert s.text().splitlines() == ["1", "", "4", "", "5"]
    s.feed("\x1b[2;1H\x1bM")                                        # reverse index at the top of the region scrolls it down
    assert s.text().splitlines() == ["1", "", "", "4", "5"]
    s.feed("\x1b[r")                                                # reset: the whole screen again
    assert (s.top, s.bot) == (0, 4)


def test_scrolling_the_main_screen_keeps_a_transcript():
    s = TS.Screen(10, 3)
    s.feed("".join(f"line{i}\r\n" for i in range(8)))
    lines = s.transcript_lines()
    assert lines[:3] == ["line0", "line1", "line2"] and lines[-1] == "line7"


def test_save_restore_cursor_both_forms_and_reverse_index_at_the_top():
    s = screen_of("\x1b[3;4H\x1b7\x1b[1;1Hx\x1b8y")
    assert s.grid[2][3] == "y" and s.grid[0][0] == "x"
    s = screen_of("\x1b[2;2H\x1b[s\x1b[5;5H\x1b[u")
    assert (s.y, s.x) == (1, 1)
    s = screen_of("a\r\nb\x1b[1;1H\x1bM")
    assert s.text().splitlines() == ["", "a", "b"]


def test_alternate_screen_has_its_own_buffer_and_the_main_one_comes_back():
    s = screen_of("MAIN\r\n\x1b[?1049h\x1b[HALT\x1b[2;1Hmore")
    assert s.alt and s.text().splitlines() == ["ALT", "more"] and s.alt_seen
    s.feed("\x1b[?1049l")
    assert not s.alt and s.text() == "MAIN" and s.last_alt_frame == "ALT\nmore"
    assert (s.y, s.x) == (1, 0)                                     # 1049 restores the cursor saved when it was entered
    s.feed("\x1b[?47h")
    assert s.alt and "MAIN" not in s.text()
    s.feed("x\x1b[?47l")
    assert s.text() == "MAIN"
    assert s.transcript_lines() == ["MAIN"]                         # the alternate screen never reaches the transcript


def test_line_drawing_charset_maps_to_box_characters():
    s = screen_of("\x1b(0lqqk\r\nx  x\r\nmqqj\x1b(B lqk")
    assert s.text() == "┌──┐\n│  │\n└──┘ lqk"
    s = screen_of("\x0e" "q" "\x0f" "q")                              # SO / SI with G1 = line drawing
    assert s.text() == "qq"
    s = screen_of("\x1b)0\x0eq\x0fq")
    assert s.text() == "─q"


def test_wide_characters_take_two_cells_and_wrap_whole():
    s = screen_of("日本a", cols=10)
    assert s.text() == "日本a" and s.x == 5 and s.grid[0][1] == "" and s.grid[0][4] == "a"
    s = screen_of("abc日", cols=4, rows=3)                            # does not fit in the last column: wraps as a whole
    assert s.text().splitlines() == ["abc", "日"]
    s = screen_of("日\x1b[1;2H\x1b[X", cols=6)                         # erasing half a wide character leaves no orphan cell
    assert s.text() == "" and s.grid[0][1] == " "
    s = screen_of("e\u0301x")                                         # a combining mark joins the previous cell
    assert s.text() == "e\u0301x" and s.x == 2


def test_unknown_and_broken_sequences_are_ignored_and_never_wedge_the_parser():
    s = screen_of("a\x1b[?9999z\x1b[99;99;99q\x1b]0;a title\x07b\x1bP1$rjunk\x1b\\c\x1b[1;31;48;5;9mD\x1b%G\x1b=\x1b>E")
    assert s.text() == "abcDE"
    s = screen_of("x\x1b[" + "9" * 200 + "y")                         # an absurdly long parameter list is dropped
    assert s.text().startswith("x")
    s.feed("\x1b[\x18ok")                                             # CAN cancels
    assert s.text().endswith("ok")


def test_utf8_is_decoded_incrementally_across_reads():
    s = TS.Screen(10, 2)
    data = "é日".encode()
    for b in data:
        s.feed_bytes(bytes([b]))
    assert s.text() == "é日"


def test_modes_and_queries_are_recorded_and_answered():
    s = screen_of("\x1b[?1h\x1b[?25l\x1b[?7l")
    assert s.app_cursor and not s.cursor_visible and not s.autowrap
    s.feed("\x1b[?1l")
    assert not s.app_cursor
    s = screen_of("\x1b[5;7H\x1b[6n\x1b[c\x1b[>c\x1b[5n\x1b[18t\x1b[?u\x1b[?25$p")
    assert b"".join(s.responses) == b"\x1b[5;7R\x1b[?1;2c\x1b[>0;10;0c\x1b[0n\x1b[8;6;20t\x1b[?25;0$y"


def test_the_screen_text_strips_trailing_spaces_and_blank_rows():
    s = screen_of("a  \r\n\r\nb   ")
    assert s.text() == "a\n\nb"


# ── keys ────────────────────────────────────────────────────────────────────

def keys_bytes(tokens, app=False):
    steps, err = TS.parse_keys(tokens)
    assert err is None
    return [TS.encode_step(s, app) for s in steps if s.kind != "wait"]


def test_key_names_and_text():
    assert keys_bytes(["Enter", "Esc", "Tab", "Space", "Backspace", "hello", "q"]) == [b"\r", b"\x1b", b"\t", b" ", b"\x7f", b"hello", b"q"]
    assert keys_bytes(["enter", "<Enter>", "ESC", "PageUp", "page_down", "Delete", "Shift-Tab"]) == [b"\r", b"\r", b"\x1b", b"\x1b[5~", b"\x1b[6~", b"\x1b[3~", b"\x1b[Z"]
    assert keys_bytes(["F1", "F4", "F5", "F12"]) == [b"\x1bOP", b"\x1bOS", b"\x1b[15~", b"\x1b[24~"]
    assert keys_bytes(["Ctrl-C", "ctrl+d", "C-x", "^Z", "Ctrl-[", "Ctrl-Space", "Ctrl-?"]) == [b"\x03", b"\x04", b"\x18", b"\x1a", b"\x1b", b"\x00", b"\x7f"]
    assert keys_bytes(["Alt-x"]) == [b"\x1bx"]
    assert keys_bytes(["text:Enter", "text:line\n"]) == [b"Enter", b"line\r"]       # a literal; a newline in text is Enter
    assert keys_bytes(["héllo", "c", "F13"]) == ["héllo".encode(), b"c", b"F13"]       # not a key name: typed as text


def test_cursor_keys_follow_decckm():
    assert keys_bytes(["Up", "Down", "Right", "Left", "Home", "End"]) == [b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D", b"\x1b[H", b"\x1b[F"]
    assert keys_bytes(["Up", "Down", "Right", "Left", "Home", "End"], app=True) == [b"\x1bOA", b"\x1bOB", b"\x1bOC", b"\x1bOD", b"\x1bOH", b"\x1bOF"]


def test_wait_tokens_and_bad_tokens():
    steps, err = TS.parse_keys(["wait:250", "wait:1s", "wait:99999", "x"])
    assert err is None and [s.data for s in steps if s.kind == "wait"] == [250, 1000, TS.MAX_WAIT_MS]
    assert TS.parse_keys(["wait:soon"])[1] and TS.parse_keys(["Ctrl-"])[0][0].data == b"Ctrl-"      # looks like a control key but is not: typed as text
    assert TS.parse_keys(["x"] * (TS.MAX_KEYS + 1))[1] and TS.parse_keys("a b")[0][1].data == b"b"
    assert TS.parse_keys(None) == ([], None) and TS.parse_keys(5)[1]


def test_size_parsing():
    assert TS.parse_size("80x24") == (80, 24) and TS.parse_size("100X30") == (100, 30) and TS.parse_size("1x1") == (TS.MIN_COLS, TS.MIN_ROWS)
    assert TS.parse_size("9999x999") == (TS.MAX_COLS, TS.MAX_ROWS)
    with pytest.raises(ValueError):
        TS.parse_size("wide")


# ── the pty ─────────────────────────────────────────────────────────────────

CURSES_GAME = '''
import curses
def main(scr):
    curses.curs_set(0)
    scr.keypad(True)
    x = 5
    while True:
        scr.erase()
        scr.addstr(0, 0, "SCORE %d" % (x - 5))
        scr.addstr(2, x, "@")
        scr.refresh()
        k = scr.getch()
        if k == curses.KEY_RIGHT: x += 1
        elif k == curses.KEY_LEFT: x -= 1
        elif k == ord("q"): break
curses.wrapper(main)
print("FINAL SCORE bye")
'''


def test_a_curses_game_is_played_with_arrow_keys_and_quits_cleanly(tmp_path):
    pytest.importorskip("curses")
    r = smoke(tmp_path, prog(tmp_path, "game.py", CURSES_GAME), keys=["Right", "Right", "q"], expect=["SCORE 2", "FINAL SCORE bye", "re:^FINAL"])
    assert r.status == "exited" and r.exit_code == 0 and r.keys_sent == 3 and not r.traceback
    assert [(e, ok) for e, ok, _w in r.expects] == [("SCORE 2", True), ("FINAL SCORE bye", True), ("re:^FINAL", True)]
    assert r.passed(0) and r.passed()
    assert "SCORE 2" in (r.last_frame or "") and "@" in (r.last_frame or "")        # the last full-screen frame survives the exit from the alternate screen
    text = TS.format_result(r)
    assert text.startswith("tui_smoke: exited with code 0") and "expect: 3/3 PASS" in text and "FINAL SCORE bye" in text and "cursor at (" in text
    assert len(text) <= TS.MAX_CHARS + 200


def test_a_failing_expect_is_reported_and_fails_the_verdict(tmp_path):
    r = smoke(tmp_path, prog(tmp_path, "p.py", "print('hello world')\n"), expect=["hello", "goodbye", "re:^hel+o"])
    assert [ok for _e, ok, _w in r.expects] == [True, False, True] and not r.passed()
    assert "FAIL  'goodbye'" in TS.format_result(r)


def test_a_crash_is_reported_prominently_with_the_transcript_tail(tmp_path):
    src = "print('starting')\nfor i in range(30):\n    print('row', i)\nraise ZeroDivisionError('boom in the game loop')\n"
    r = smoke(tmp_path, prog(tmp_path, "crash.py", src))
    assert r.status == "exited" and r.exit_code == 1 and r.traceback and not r.passed()
    text = TS.format_result(r)
    assert "!! NON-ZERO EXIT: code 1" in text and "!! a traceback" in text
    assert "ZeroDivisionError: boom in the game loop" in text and "transcript tail" in text and "row 29" in text


def test_a_crash_after_leaving_the_alternate_screen_shows_the_traceback_from_the_main_buffer(tmp_path):
    src = ("import sys, time\nsys.stdout.write('\\x1b[?1049hGAME SCREEN\\r\\n'); sys.stdout.flush(); time.sleep(0.5)\n"
           "sys.stdout.write('\\x1b[?1049l'); sys.stdout.flush()\nraise RuntimeError('late failure')\n")
    r = smoke(tmp_path, prog(tmp_path, "alt.py", src), expect=["GAME SCREEN"])
    assert r.exit_code == 1 and r.traceback and r.expects[0][1] and "alternate" not in r.expects[0][2]
    assert "RuntimeError: late failure" in "\n".join(r.tail) and r.last_frame == "GAME SCREEN"


def test_a_nonzero_exit_without_a_traceback(tmp_path):
    r = smoke(tmp_path, prog(tmp_path, "x.py", "import sys\nprint('bye')\nsys.exit(3)\n"))
    assert r.exit_code == 3 and not r.traceback and not r.passed() and r.passed(3)
    assert "NON-ZERO EXIT: code 3" in TS.format_result(r)


def test_a_program_that_waits_forever_is_killed_at_the_timeout_and_nothing_survives(tmp_path):
    pidfile = tmp_path / "pids"
    src = (f"import os, subprocess, time\nchild = subprocess.Popen(['sleep', '300'])\nopen({str(pidfile)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
           "print('waiting', flush=True)\ntime.sleep(300)\n")
    t0 = time.monotonic()
    r = smoke(tmp_path, prog(tmp_path, "hang.py", src), expect=["never appears"], timeout=2)
    assert time.monotonic() - t0 < 6
    assert r.status == "timeout" and not r.passed() and r.expects == [("never appears", False, "not on any settled screen nor in the transcript")]
    assert "KILLED BY TIMEOUT after 2s" in TS.format_result(r)
    leader, child = (int(x) for x in pidfile.read_text().split())
    assert pid_gone(leader) and pid_gone(child)                       # the whole process group is gone


def test_a_program_still_running_after_the_keys_is_stopped_and_reported_as_running(tmp_path):
    r = smoke(tmp_path, prog(tmp_path, "idle.py", "import time\nprint('menu', flush=True)\ntime.sleep(60)\n"), keys=["Down"], expect=["menu"])
    assert r.status == "running" and r.passed() and not r.passed(0) and r.seconds < 5
    assert "still running after the last key" in TS.format_result(r)


def test_application_cursor_keys_are_sent_as_esc_o_when_the_program_asked_for_them(tmp_path):
    src = ("import os, sys, tty\ntty.setraw(0)\nif sys.argv[1] == 'app':\n    sys.stdout.write('\\x1b[?1h')\n"
           "sys.stdout.write('READY\\r\\n'); sys.stdout.flush()\ndata = os.read(0, 16)\n"
           "sys.stdout.write('GOT ' + repr(data) + '\\r\\n'); sys.stdout.flush()\n")
    base = prog(tmp_path, "keys.py", src)
    app = smoke(tmp_path, f"{base} app", keys=["Right"], expect=["GOT b'\\x1bOC'"])
    plain = smoke(tmp_path, f"{base} plain", keys=["Right"], expect=["GOT b'\\x1b[C'"])
    assert app.passed(0) and plain.passed(0), (TS.format_result(app), TS.format_result(plain))


def test_terminal_queries_are_answered_so_the_program_does_not_hang(tmp_path):
    src = ("import os, sys, tty\ntty.setraw(0)\n\ndef ask(q, end):\n    sys.stdout.write(q); sys.stdout.flush()\n    buf = b''\n"
           "    while not buf.endswith(end):\n        buf += os.read(0, 1)\n    return buf\n\n"
           "pos = ask('\\x1b[5;7H\\x1b[6n', b'R')\nda = ask('\\x1b[c', b'c')\n"
           "sys.stdout.write('\\x1b[2J\\x1b[H' + 'POS ' + repr(pos) + '\\r\\n' + 'DA ' + repr(da) + '\\r\\n')\n")
    r = smoke(tmp_path, prog(tmp_path, "query.py", src), expect=["POS b'\\x1b[5;7R'", "DA b'\\x1b[?1;2c'"], timeout=5)
    assert r.status == "exited" and r.passed(0), TS.format_result(r)


def test_the_program_runs_with_a_controlling_terminal(tmp_path):
    r = smoke(tmp_path, prog(tmp_path, "tty.py", "with open('/dev/tty', 'w') as t:\n    t.write('TTYOK')\n"), expect=["TTYOK"])
    assert r.passed(0), TS.format_result(r)


def test_the_window_size_and_the_environment_are_set(tmp_path):
    src = "import os\nprint(os.get_terminal_size(), os.environ['TERM'], os.environ['COLUMNS'], os.environ['LINES'], os.environ.get('SDL_VIDEODRIVER'))\n"
    r = smoke(tmp_path, prog(tmp_path, "size.py", src), size="100x30", expect=["columns=100, lines=30) xterm-256color 100 30"])
    assert r.passed(0), TS.format_result(r)


def test_a_big_output_is_capped_and_the_program_stopped(tmp_path):
    src = "import sys\nwhile True:\n    sys.stdout.write('y' * 999 + '\\n')\n    sys.stdout.flush()\n"
    t0 = time.monotonic()
    r = TS.run_smoke(prog(tmp_path, "flood.py", src), cwd=tmp_path, timeout=8, max_bytes=200_000)
    assert r.status == "output_cap" and not r.passed() and time.monotonic() - t0 < 6
    assert "KILLED: more than 2 MB of output" in TS.format_result(r)


def test_a_command_that_does_not_exist_and_shell_syntax(tmp_path):
    r = smoke(tmp_path, "no-such-program-xyz --flag")
    assert r.status == "exited" and r.exit_code == 127 and "no-such-program-xyz" in r.screen
    r = smoke(tmp_path, "echo A && echo B | tr B C", expect=["A", "C"])
    assert r.passed(0)
    r = smoke(tmp_path, "FOO=bar sh -c 'echo $FOO'", expect=["bar"])
    assert r.passed(0)


def test_bad_arguments_come_back_as_errors(tmp_path):
    assert TS.format_result(TS.run_smoke("echo x", size="huge")).startswith("error: tui_smoke size must look like 80x24")
    assert "wait:<milliseconds>" in TS.format_result(TS.run_smoke("echo x", keys=["wait:soon"]))
    assert "needs a command" in TS.format_result(TS.run_smoke("  "))
    assert TS.run_smoke("echo x", expect=["re:("], cwd=tmp_path).expects[0][1] is False


def test_the_result_stays_small_for_a_big_screen_and_the_full_text_is_kept_by_the_tool(tmp_path):
    src = "\n".join(["import sys", "for i in range(60):", "    print(f'{i:02d} ' + 'abcdefghij' * 18)"])
    r = smoke(tmp_path, prog(tmp_path, "big.py", src), size="200x70")
    text = TS.format_result(r)
    assert len(text) <= TS.MAX_CHARS + 100 and "rows omitted" in text and "00 abcdefghij" in text and "59 abcdefghij" in text


def test_the_sdl_dummy_drivers_are_set_only_when_the_user_did_not_choose(tmp_path, monkeypatch):
    assert TS.headless_env({})["SDL_VIDEODRIVER"] == "dummy" and TS.headless_env({})["SDL_AUDIODRIVER"] == "dummy"
    assert TS.headless_env({"SDL_VIDEODRIVER": "x11"})["SDL_VIDEODRIVER"] == "x11"
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    src = "import os\nprint('V=' + os.environ['SDL_VIDEODRIVER'], 'A=' + os.environ['SDL_AUDIODRIVER'], 'D=' + str(os.environ.get('DISPLAY')))\n"
    assert smoke(tmp_path, prog(tmp_path, "sdl.py", src), expect=["V=dummy A=dummy D=None"]).passed(0)


# ── the tool in the engine ──────────────────────────────────────────────────

def test_the_tool_is_registered_with_a_short_description(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    tool = eng.registry.get("tui_smoke")
    assert tool is not None and len(tool.description.split()) <= 60
    assert tool.description.startswith("Run an interactive terminal program (curses, TUI, text game) in a real terminal")
    props = tool.parameters["properties"]
    assert set(props) == {"command", "keys", "size", "expect", "timeout", "settle_ms"} and tool.parameters["required"] == ["command"]


def test_engine_runs_a_tool_call_and_returns_the_screen(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "menu.py").write_text("import sys\nsys.stdout.write('MENU\\r\\n1) play\\r\\n> '); sys.stdout.flush()\nline = sys.stdin.readline()\nprint('chosen', line.strip())\n")
    llm = FakeLLM([call("tui_smoke", command="python3 menu.py", keys=["1", "Enter"], expect=["MENU", "chosen 1"]), say("The menu works.")])
    eng, repo, _ = engine(tmp_path, llm)
    res = eng.turn("check the menu")
    tool = [m["content"] for m in eng.ctx.messages if m["role"] == "tool"][0]
    assert "tui_smoke: exited with code 0" in tool and "expect: 2/2 PASS" in tool and "chosen 1" in tool, tool
    assert res.text == "The menu works."
    assert eng.events.of("tool")[0]["cmd"] == "python3 menu.py"


def test_the_forbidden_path_jail_applies(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    out = eng.execute("tui_smoke", {"command": "cat ~/.ssh/id_rsa"})
    assert out.startswith("NO:")
    again = eng.execute("tui_smoke", {"command": "cat ~/.ssh/id_rsa"})
    assert again == out


def test_a_protected_file_the_program_changes_is_restored_like_after_run(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "save.json").write_text('{"gold": 5}\n')
    (repo / "bad.py").write_text("open('save.json', 'w').write('{}')\nopen('made.txt', 'w').write('x')\nprint('wrote')\n")
    out = eng.execute("tui_smoke", {"command": "python3 bad.py", "expect": ["wrote"]})
    assert (repo / "save.json").read_text() == '{"gold": 5}\n'
    assert "save.json is protected; this command changed it and the change was undone" in out and "expect: 1/1 PASS" in out
    assert "NOTE: this command changed made.txt (recorded)." in out


def test_a_protected_file_the_program_creates_is_deleted_again(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "mk.py").write_text("open('savegame1.sav', 'w').write('x')\nprint('ok')\n")
    out = eng.execute("tui_smoke", {"command": "python3 mk.py"})
    assert not (repo / "savegame1.sav").exists() and "this command created it and it was deleted" in out


def test_the_command_goes_through_the_same_approval_and_block_rules_as_run(tmp_path):
    from agent.codeagent.engine import DENIED

    seen = []
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="ask", approve=lambda d, why: seen.append((d, why)) or False)
    (repo / "build").mkdir()
    assert eng.execute("tui_smoke", {"command": "rm -rf build"}) == DENIED
    assert seen[0][0].startswith("tui  rm -rf build") and (repo / "build").is_dir()                 # asked, declined, never ran
    blocked = eng.execute("tui_smoke", {"command": "rm -rf /"})
    assert blocked.startswith("BLOCKED:") and len(seen) == 1                                          # a blocked command is not even asked


def test_a_long_result_is_spilled_with_one_locator_line(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "wide.py").write_text("for i in range(60):\n    print(f'{i:02d} ' + 'abcdefghij' * 18)\n")
    out = eng.execute("tui_smoke", {"command": "python3 wide.py", "size": "200x70"})
    assert "rows omitted" in out and "full output: .scratch/spill/tui_smoke-1.txt" in out and "59 abcdefghij" in (repo / ".scratch/spill/tui_smoke-1.txt").read_text()


def test_the_timeout_and_size_of_the_answer_come_from_the_config(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    assert eng.tools.smoke_timeout == 10 and eng.tools.smoke_max_chars == 3500
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "tui_smoke_timeout": 3, "tui_smoke_max_chars": 900}}
    from agent.codeagent.engine import CodeEngine

    eng2 = CodeEngine(repo, eng.config, FakeLLM([]), eng.console, interactive=False, events=eng.events)
    assert eng2.tools.smoke_timeout == 3 and eng2.tools.smoke_max_chars == 900
    (repo / "hang.py").write_text("import time\nprint('x', flush=True)\ntime.sleep(60)\n")
    t0 = time.monotonic()
    out = eng2.execute("tui_smoke", {"command": "python3 hang.py", "expect": ["never"]})
    assert time.monotonic() - t0 < 7 and "KILLED BY TIMEOUT after 3s" in out


# ── look / act, the loop guard, the prompt ──────────────────────────────────

def test_tui_smoke_is_an_act_like_run(tmp_path):
    from agent.codeagent import lookact

    assert lookact.classify("tui_smoke", {"command": "python3 g.py"}, "tui_smoke: exited") == "act"
    assert "tui_smoke" in lookact.SHELL_TOOLS


def test_family_key_of_smoke_tests_ignores_the_keys():
    a = LG.family_key("tui_smoke", {"command": "python3 snake.py", "keys": ["Up"]})
    b = LG.family_key("tui_smoke", {"command": "python3 ./snake.py --seed 3", "keys": ["Down", "q"], "expect": ["x"]})
    assert a == b == "tui python3 snake.py"
    assert LG.family_key("tui_smoke", {"command": "python3 -m snake"}) == "tui python3 -m snake"
    assert LG.family_key("tui_smoke", {"command": "python3 a.py"}) != LG.family_key("tui_smoke", {"command": "python3 b.py"})
    assert LG.family_key("tui_smoke", {"command": " "}) is None


def test_twelve_identical_smoke_tests_are_caught_by_the_family_guard(tmp_path):
    eng = fresh(tmp_path, "fam")
    orient(eng)
    cmd = "python3 snake.py"
    for i in range(1, 5):
        playt(eng, "tui_smoke", {"command": cmd, "keys": ["Up"] * i}, f"tui_smoke: exited with code 0\nscreen {i} " + "x" * i)
    playt(eng, "tui_smoke", {"command": cmd, "keys": ["Up"] * 5}, "tui_smoke: exited with code 0\nscreen 5 xxxxx")
    assert "Same kind of command as your last 4 (tui python3 snake.py)" in [m["content"] for m in eng.ctx.messages if m["role"] == "tool"][-1]
    for i in range(6, 9):
        playt(eng, "tui_smoke", {"command": cmd, "keys": ["Up"] * i}, f"tui_smoke: exited with code 0\nscreen {i} " + "y" * i)
    assert eng._abort_loop == 8
    assert eng.events.of("loop_abort")[-1]["shape"] == "family" and eng.events.of("loop_abort")[-1]["tool"] == "tui_smoke"


def test_the_prompt_mentions_tui_smoke_only_for_a_repo_that_imports_a_terminal_ui_library(tmp_path):
    for i, (src, expected) in enumerate([("import curses\n", True), ("from textual.app import App\n", True), ("import pygame\n", True),
                                         ("  import prompt_toolkit\n", True), ("from blessed import Terminal\n", True), ("import urwid\n", True),
                                         ("import os\nx = 'curses'  # not an import\n", False), ("import cursesfoo\n", False)]):
        sub = tmp_path / f"r{i}"
        sub.mkdir()
        (sub / "main.py").write_text(src)
        assert TS.repo_uses_tui(sub) is expected, src
        eng, _repo, _ = engine(sub, FakeLLM([]))
        (sub / "repo" / "main.py").write_text(src)
        eng.start()
        head = eng.ctx.messages[0]["content"]
        assert (TS.PROMPT_SENTENCE in head) is expected, src
    assert TS.PROMPT_SENTENCE.count("`run`") == 1 and len(TS.PROMPT_SENTENCE.split()) <= 60         # two short sentences (tuihint tests check the wording)
    deep = tmp_path / "deep"
    (deep / "pkg").mkdir(parents=True)
    (deep / "pkg" / "ui.py").write_text("import curses\n")
    assert TS.repo_uses_tui(deep) is False                         # only top-level files are looked at


def test_the_system_prompt_is_unchanged_for_a_plain_repo(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    eng.start()
    from agent.codeagent.engine import SYSTEM

    assert eng.ctx.messages[0]["content"].startswith(SYSTEM + "\n\n") and "tui_smoke" not in eng.ctx.messages[0]["content"]


# ── the `tui` check kind ────────────────────────────────────────────────────

def test_the_tui_check_kind_validates_its_spec():
    assert "tui" in checks_mod.KINDS
    assert checks_mod.validate("tui", {}) and checks_mod.validate("tui", {"command": "python3 g.py"})
    assert checks_mod.validate("tui", {"command": "python3 g.py", "expect": ["TITLE"]}) is None
    assert checks_mod.validate("tui", {"command": "python3 g.py", "exit_code": 0}) is None


def test_a_tui_check_passes_and_fails_through_the_same_function(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "game.py").write_text(CURSES_GAME)
    spec = {"command": "python3 game.py", "keys": ["Right", "q"], "expect": ["SCORE 1", "FINAL SCORE bye"], "exit_code": 0}
    c = checks_mod.Check("starts", "game starts, shows the score, quits on q", "tui", spec)
    assert eng.checks.add(c) == ""
    pytest.importorskip("curses")
    eng.run_checks([c], quiet=True)
    assert c.status() == "pass", c.last_result
    bad = checks_mod.Check("title", "title", "tui", {**spec, "expect": ["TITLE SCREEN"]})
    eng.checks.add(bad)
    eng.run_checks([bad], quiet=True)
    assert bad.status() == "fail" and "FAIL  'TITLE SCREEN'" in bad.last_result["output"]


def test_a_tui_check_fails_when_the_program_crashes_or_does_not_exit_when_it_must(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "crash.py").write_text("print('TITLE')\nraise SystemExit(2)\n")
    (repo / "idle.py").write_text("import time\nprint('TITLE', flush=True)\ntime.sleep(60)\n")
    crash = checks_mod.Check("c", "c", "tui", {"command": "python3 crash.py", "expect": ["TITLE"]})
    idle_ok = checks_mod.Check("i", "i", "tui", {"command": "python3 idle.py", "expect": ["TITLE"]})
    idle_exit = checks_mod.Check("e", "e", "tui", {"command": "python3 idle.py", "expect": ["TITLE"], "exit_code": 0})
    for c in (crash, idle_ok, idle_exit):
        eng.checks.add(c)
    eng.run_checks([crash, idle_ok, idle_exit], quiet=True)
    assert (crash.status(), idle_ok.status(), idle_exit.status()) == ("fail", "pass", "fail")


def test_a_tui_check_obeys_the_repo_policy_and_the_add_check_tool_accepts_it(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "hello.py").write_text("print('HELLO')\n")
    out = eng.execute("add_check", {"id": "hello", "kind": "tui", "command": "python3 hello.py", "expect": ["HELLO"], "exit_code": 0,
                                   "description": "prints hello"})
    assert "check hello added" in out and "PASS" in out, out
    assert eng.checks.model["hello"].spec == {"command": "python3 hello.py", "expect": ["HELLO"], "exit_code": 0}
    assert "error:" in eng.execute("add_check", {"id": "nope", "kind": "tui"})
    c = checks_mod.Check("jail", "j", "tui", {"command": "cat ~/.ssh/id_rsa", "expect": ["x"]})
    eng.checks.add(c)
    eng.run_checks([c], quiet=True)
    assert c.status() == "skipped"


# ── run: headless defaults ──────────────────────────────────────────────────

def test_run_sets_the_sdl_dummy_drivers_unless_the_user_did(tmp_path, monkeypatch):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)
    assert "dummy dummy" in eng.tools.run("echo $SDL_VIDEODRIVER $SDL_AUDIODRIVER")
    monkeypatch.setenv("SDL_VIDEODRIVER", "wayland")
    assert "wayland dummy" in eng.tools.run("echo $SDL_VIDEODRIVER $SDL_AUDIODRIVER")
