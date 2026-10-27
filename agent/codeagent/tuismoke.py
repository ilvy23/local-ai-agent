"""`tui_smoke`: run an interactive terminal program (curses, a TUI, a text game) in a REAL terminal, send keystrokes, read the screen.

Seen live: asked to test a curses snake game, the model wrote a fake-screen script (its own renderer, a stand-in for `curses`) 12 times with
the same result and the loop guard ended the run. `run` gives a program no tty (stdin is a pipe or /dev/null), so `curses.initscr()` fails and
the only way left to "show a screen" is to fake one. One call that does the real thing removes the reason:

    tui_smoke(command, keys=[...], size="80x24", expect=[...], timeout=10, settle_ms=300)

How it works (stdlib only; ideas from agent-tty, pi-tmux-harness and pyte, no code taken from them):
  * a pty pair (`os.openpty`), the window size set with TIOCSWINSZ, the program in its own session / process group;
  * the program is started through a tiny `python -c` wrapper that makes the slave its CONTROLLING terminal (TIOCSCTTY on fd 0) and then
    `exec`s the command. Pitfall behind that choice: `subprocess.Popen(start_new_session=True)` alone gives a session WITHOUT a controlling
    terminal, so `/dev/tty`, `getpass` and some curses setups fail; `preexec_fn` could do the ioctl but is unsafe in a threaded process
    (the engine uses threads); `pty.fork` / `os.forkpty` fork the multi-threaded interpreter. The wrapper costs one extra interpreter start (~20 ms);
  * what the program prints goes through `Screen`, a ~300-line VT100/xterm subset (cursor movement, erase, insert/delete, scroll regions, alternate
    screen, save/restore, line-drawing charset, wide characters; unknown sequences are dropped, the parser can never wedge);
  * queries the program sends are answered (cursor position, device attributes, ...), so TUIs do not wait for a terminal that never answers;
  * keys are sent after output has been quiet for `settle_ms` (at most MAX_KEY_WAIT per key) - not after guessed sleeps; arrow keys follow the
    program's DECCKM state (ESC O A when it asked for application cursor keys, else ESC [ A);
  * the whole process group is stopped at the end (TERM, then KILL), output is capped (MAX_BYTES), nothing outlives the call.
`pyte` is NOT used (not installed on the target machine, LGPL); the built-in emulator is the only backend."""

from __future__ import annotations

import codecs
import contextlib
import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:                                                  # a pty needs a POSIX kernel; on Windows run_smoke() reports that instead of failing the import
    import fcntl
    import select
    import termios
    PTY_OK = True
except ImportError:                                   # pragma: no cover - exercised by tests/test_windows_import.py
    fcntl = select = termios = None                   # type: ignore[assignment]
    PTY_OK = False

DEFAULT_SIZE = "80x24"
DEFAULT_TIMEOUT = 10             # seconds for the whole call (coding.tui_smoke_timeout)
MAX_TIMEOUT = 120
DEFAULT_SETTLE_MS = 300
MAX_BYTES = 2_000_000            # output read from the program; more than this and it is stopped
MAX_KEY_WAIT = 2.0               # seconds to wait for quiet after one key (an animated program never goes quiet)
STARTUP_WAIT = 2.5               # seconds to wait for the first output of a program that is slow to start
EXIT_GRACE = 1.0                 # after the last key: how long a still-running program may take to exit before the call ends
MAX_WAIT_MS = 5000               # one `wait:<ms>` token
MAX_KEYS = 200
SCROLLBACK = 500                 # lines of the main buffer kept after they scrolled off
MAX_SNAPSHOTS = 64               # distinct settled screens remembered for `expect`
MAX_CHARS = 3500                 # size of the text the model gets (coding.tui_smoke_max_chars)
TAIL_LINES = 15                  # lines of the main-buffer transcript shown after the program exited
MAX_EXPECT_CHARS = 300
HISTORY_CHARS = 60000            # SmokeResult.history: all settled screens + transcript, cut here
MIN_COLS, MAX_COLS, MIN_ROWS, MAX_ROWS = 20, 250, 5, 80

_TRACEBACK = re.compile(r"Traceback \(most recent call last\)|Segmentation fault|core dumped")


def headless_env(env: dict[str, str]) -> dict[str, str]:
    """Defaults that let a game run without a display or a sound card: SDL's dummy video and audio drivers. Only when the user did not set
    them (an SDL program the user runs on purpose with another driver keeps it). DISPLAY is neither required nor touched: with the dummy
    driver SDL never opens an X connection, so an unset DISPLAY is fine. Used for `run` and for `tui_smoke`."""
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    return env


# ── the screen emulator ─────────────────────────────────────────────────────

_LINE = {"j": "┘", "k": "┐", "l": "┌", "m": "└", "n": "┼", "q": "─", "t": "├", "u": "┤", "v": "┴", "w": "┬", "x": "│",
         "a": "▒", "`": "◆", "f": "°", "g": "±", "o": "⎺", "p": "⎻", "r": "⎼", "s": "⎽", "y": "≤", "z": "≥", "{": "π", "|": "≠",
         "}": "£", "~": "·", "0": "█", "h": "▒", "i": "#", ",": "←", "+": "→", ".": "↓", "-": "↑"}
_PRINTABLE = re.compile(r"[^\x00-\x1f\x7f-\x9f]+")
_CSI_BODY = re.compile(r"([<=>?]?)([0-9;:]*)([ -/]*)$")
_BLANK = " "


@lru_cache(maxsize=4096)
def char_width(ch: str) -> int:
    """Columns a character takes: 0 (combining), 2 (east-asian wide / fullwidth), else 1."""
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


class Screen:
    """A rows x cols grid fed with the program's output. A wide character occupies its cell plus a "" cell. `responses` collects the bytes the
    program's queries must be answered with (the caller writes them to the pty). Unknown sequences are ignored silently."""

    def __init__(self, cols: int = 80, rows: int = 24) -> None:
        self.cols, self.rows = cols, rows
        self.scrollback: list[str] = []
        self.responses: list[bytes] = []
        self.last_alt_frame: str | None = None          # the alternate screen as it was when the program left it
        self.alt_seen = False
        self.cursor_moves = 0                           # absolute cursor addressing seen (a line-based program has none)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._reset()

    def _blank_row(self) -> list[str]:
        return [_BLANK] * self.cols

    def _blank_grid(self) -> list[list[str]]:
        return [self._blank_row() for _ in range(self.rows)]

    def _reset(self) -> None:
        self.main = self._blank_grid()
        self.alt_grid = self._blank_grid()
        self.grid = self.main
        self.alt = False
        self.x = self.y = 0
        self.pending = False                            # the last column was written: the next printable character wraps first
        self.top, self.bot = 0, self.rows - 1
        self.saved = (0, 0, False)                      # ESC 7 / CSI s
        self.saved_main = (0, 0, False)                 # the main screen's cursor while the alternate screen is up (?1049)
        self.autowrap, self.app_cursor, self.cursor_visible, self.insert_mode = True, False, True, False
        self.modes: set[int] = set()
        self.charsets: list[str | None] = [None, None]  # G0, G1: "0" = line drawing
        self.gl = 0
        self.last_char = ""
        self._state, self._buf = "ground", ""

    # ── input ───────────────────────────────────────────────────────────
    def feed_bytes(self, data: bytes) -> None:
        """UTF-8 is decoded incrementally: a character split between two reads is not lost."""
        self.feed(self._decoder.decode(data))

    def feed(self, text: str) -> None:
        i, n = 0, len(text)
        while i < n:
            if self._state == "ground":
                m = _PRINTABLE.match(text, i)
                if m:
                    for ch in m.group():
                        self._put(ch)
                    i = m.end()
                    continue
            self._step(text[i])
            i += 1

    def _step(self, ch: str) -> None:
        st = self._state
        if st == "ground":
            if ch == "\x1b":
                self._state = "esc"
            elif ch < " " or ch == "\x7f":
                self._control(ch)
            return
        if ch in "\x18\x1a":                            # CAN / SUB cancel any sequence
            self._state = "ground"
            return
        if st == "esc":
            self._esc(ch)
        elif st == "esc_int":
            if " " <= ch <= "/":
                return                                  # more intermediates
            if self._buf == "(":
                self.charsets[0] = "0" if ch == "0" else None
            elif self._buf == ")":
                self.charsets[1] = "0" if ch == "0" else None
            self._state = "ground"
        elif st == "csi":
            if ch == "\x1b":
                self._state = "esc"
            elif ch < " ":
                self._control(ch)
            elif "@" <= ch <= "~":
                buf, self._buf, self._state = self._buf, "", "ground"
                self._csi(buf, ch)
            else:
                self._buf += ch
                if len(self._buf) > 64:                 # garbage: swallow the rest of the sequence
                    self._buf, self._state = "", "csi_ignore"
        elif st == "csi_ignore":
            if ch == "\x1b":
                self._state = "esc"
            elif "@" <= ch <= "~":
                self._state = "ground"
        elif st == "str":                               # OSC / DCS / APC / PM / SOS: swallowed until BEL or ST
            if ch == "\x07":
                self._state = "ground"
            elif ch == "\x1b":
                self._state = "str_esc"
        elif st == "str_esc":
            self._state = "ground" if ch == "\\" else "esc"
            if self._state == "esc":
                self._esc(ch)
        else:
            self._state = "ground"

    def _esc(self, ch: str) -> None:
        self._state = "ground"
        if ch == "[":
            self._state, self._buf = "csi", ""
        elif ch in "]PX^_":
            self._state = "str"
        elif ch in "()*+#%":
            self._state, self._buf = "esc_int", ch
        elif ch == "7":
            self.saved = (self.x, self.y, self.pending)
        elif ch == "8":
            self.x, self.y, self.pending = self.saved
        elif ch == "D":
            self._index()
        elif ch == "E":
            self.x, self.pending = 0, False
            self._index()
        elif ch == "M":
            self._reverse_index()
        elif ch == "c":
            self._reset()
        elif ch == "\x1b":
            self._state = "esc"
        elif ch < " ":
            self._control(ch)
            self._state = "esc"

    def _control(self, ch: str) -> None:
        if ch == "\r":
            self.x, self.pending = 0, False
        elif ch in "\n\x0b\x0c":
            self.pending = False
            self._index()
        elif ch == "\b":
            self.x, self.pending = max(0, self.x - 1), False
        elif ch == "\t":
            self.x, self.pending = min(self.cols - 1, (self.x // 8 + 1) * 8), False
        elif ch == "\x0e":
            self.gl = 1
        elif ch == "\x0f":
            self.gl = 0

    # ── drawing ─────────────────────────────────────────────────────────
    def _put(self, ch: str) -> None:
        if self.charsets[self.gl] == "0":
            ch = _LINE.get(ch, ch)
        w = char_width(ch)
        if w == 0:                                      # a combining mark joins the cell before the cursor
            px = self.x if self.pending else self.x - 1
            row = self.grid[self.y]
            while px > 0 and row[px] == "":
                px -= 1
            if px >= 0:
                row[px] += ch
            return
        if self.pending:
            self.pending = False
            if self.autowrap:
                self.x = 0
                self._index()
        if w == 2 and self.x >= self.cols - 1:
            if not self.autowrap or self.cols < 2:
                return
            self.grid[self.y][self.x] = _BLANK
            self.x = 0
            self._index()
        row = self.grid[self.y]
        if self.insert_mode:
            row[self.x:] = ([_BLANK] * w + row[self.x:])[:self.cols - self.x]
        row[self.x] = ch
        if w == 2:
            row[self.x + 1] = ""
        self._fix_row(self.y)
        self.last_char = ch
        self.x += w
        if self.x >= self.cols:
            self.x, self.pending = self.cols - 1, self.autowrap

    def _fix_row(self, y: int) -> None:
        """Repair the wide characters of a row after cells were overwritten, erased or shifted: no orphan "" cell, no wide cell without its "" partner."""
        row = self.grid[y]
        for i, c in enumerate(row):
            if c == "" and not (i > 0 and row[i - 1] and char_width(row[i - 1][0]) == 2):
                row[i] = _BLANK
            elif c and char_width(c[0]) == 2 and not (i + 1 < self.cols and row[i + 1] == ""):
                row[i] = _BLANK

    def _index(self) -> None:
        if self.y == self.bot:
            self._scroll_up(1)
        elif self.y < self.rows - 1:
            self.y += 1

    def _reverse_index(self) -> None:
        if self.y == self.top:
            self._scroll_down(1)
        elif self.y > 0:
            self.y -= 1

    def _scroll_up(self, n: int) -> None:
        n = min(n, self.bot - self.top + 1)
        for _ in range(n):
            gone = self.grid.pop(self.top)
            if not self.alt and self.top == 0 and self.bot == self.rows - 1:
                self.scrollback.append("".join(gone).rstrip())
            self.grid.insert(self.bot, self._blank_row())
        del self.scrollback[:-SCROLLBACK]

    def _scroll_down(self, n: int) -> None:
        n = min(n, self.bot - self.top + 1)
        for _ in range(n):
            self.grid.pop(self.bot)
            self.grid.insert(self.top, self._blank_row())

    def _erase(self, y: int, a: int, b: int) -> None:
        """Blank the cells a..b-1 of row y."""
        a, b = max(0, a), min(self.cols, b)
        row = self.grid[y]
        for i in range(a, b):
            row[i] = _BLANK
        self._fix_row(y)

    # ── alternate screen ────────────────────────────────────────────────
    def _enter_alt(self, save: bool, clear: bool) -> None:
        if self.alt:
            return
        if save:
            self.saved_main = (self.x, self.y, self.pending)
        self.alt, self.alt_seen = True, True
        if clear:
            self.alt_grid = self._blank_grid()
        self.grid = self.alt_grid
        self.top, self.bot = 0, self.rows - 1

    def _leave_alt(self, restore: bool) -> None:
        if not self.alt:
            return
        self.last_alt_frame = self.text()
        self.alt, self.grid = False, self.main
        self.top, self.bot = 0, self.rows - 1
        if restore:
            self.x, self.y, self.pending = self.saved_main

    # ── CSI ─────────────────────────────────────────────────────────────
    def _csi(self, buf: str, final: str) -> None:
        m = _CSI_BODY.match(buf)
        if not m:
            return
        priv, digits, inter = m.groups()
        nums: list[int | None] = []
        for part in re.split(r"[;:]", digits)[:16]:
            nums.append(min(int(part), 65535) if part else None)

        def arg(i: int, default: int = 1) -> int:       # 0 and a missing parameter both mean the default (movement, counts)
            v = nums[i] if i < len(nums) else None
            return default if not v else v

        def raw(i: int, default: int = 0) -> int:
            v = nums[i] if i < len(nums) else None
            return default if v is None else v

        if inter:
            if inter == "$" and final == "p" and priv == "?":    # DECRQM: "mode not recognised" is a truthful, harmless answer
                self.responses.append(f"\x1b[?{raw(0)};0$y".encode())
            return
        if priv == "?":
            if final in "hl":
                for v in nums:
                    if v is not None:
                        self._dec_mode(v, final == "h")
            elif final == "n" and raw(0) == 6:
                self.responses.append(f"\x1b[?{self.y + 1};{self.x + 1}R".encode())
            return                                       # `CSI ? u` (kitty keyboard query) and friends get no answer: that is the "unsupported" reply
        if priv == ">":
            if final == "c":
                self.responses.append(b"\x1b[>0;10;0c")
            return
        if priv:
            return
        cmd = final
        if cmd in "ABeaCDEFGHfd`":
            self._move(cmd, arg, nums)
        elif cmd == "J":
            self._erase_display(raw(0))
        elif cmd == "K":
            mode = raw(0)
            if mode == 0:
                self._erase(self.y, self.x, self.cols)
            elif mode == 1:
                self._erase(self.y, 0, self.x + 1)
            elif mode == 2:
                self._erase(self.y, 0, self.cols)
        elif cmd in "LM":
            self._insert_delete_lines(cmd == "L", arg(0))
        elif cmd in "P@X":
            self._chars(cmd, arg(0))
        elif cmd == "S":
            self._scroll_up(arg(0))
        elif cmd == "T":
            self._scroll_down(arg(0))
        elif cmd == "r":
            top, bot = arg(0) - 1, arg(1, self.rows) - 1
            if 0 <= top < bot < self.rows:
                self.top, self.bot = top, bot
                self.x = self.y = 0
                self.pending = False
        elif cmd == "s" and not digits:
            self.saved = (self.x, self.y, self.pending)
        elif cmd == "u":
            self.x, self.y, self.pending = self.saved
        elif cmd == "b" and self.last_char:
            for _ in range(min(arg(0), 1000)):
                self._put(self.last_char)
        elif cmd == "I":
            self.x, self.pending = min(self.cols - 1, (self.x // 8 + arg(0)) * 8), False
        elif cmd == "Z":
            self.x, self.pending = max(0, ((self.x - 1) // 8 - arg(0) + 1) * 8), False
        elif cmd in "hl":
            for v in nums:
                if v == 4:
                    self.insert_mode = cmd == "h"
        elif cmd == "c" and raw(0) == 0:
            self.responses.append(b"\x1b[?1;2c")
        elif cmd == "n":
            if raw(0) == 5:
                self.responses.append(b"\x1b[0n")
            elif raw(0) == 6:
                self.responses.append(f"\x1b[{self.y + 1};{self.x + 1}R".encode())
        elif cmd == "t" and raw(0) == 18:
            self.responses.append(f"\x1b[8;{self.rows};{self.cols}t".encode())
        # SGR (m), cursor style, soft reset, window manipulation ...: parsed and ignored

    def _dec_mode(self, v: int, on: bool) -> None:
        (self.modes.add if on else self.modes.discard)(v)
        if v == 1:
            self.app_cursor = on
        elif v == 7:
            self.autowrap = on
        elif v == 25:
            self.cursor_visible = on
        elif v in (47, 1047, 1049):
            if on:
                self._enter_alt(save=v == 1049, clear=v != 47)
            else:
                self._leave_alt(restore=v == 1049)

    def _move(self, cmd: str, arg: Any, nums: list[int | None]) -> None:
        self.pending = False
        n = arg(0)
        if cmd in "AF":
            self.y = max(self.top if self.y >= self.top else 0, self.y - n)
            if cmd == "F":
                self.x = 0
        elif cmd in "BEe":
            self.y = min(self.bot if self.y <= self.bot else self.rows - 1, self.y + n)
            if cmd == "E":
                self.x = 0
        elif cmd in "Ca":
            self.x = min(self.cols - 1, self.x + n)
        elif cmd == "D":
            self.x = max(0, self.x - n)
        elif cmd in "G`":
            self.x = min(self.cols - 1, n - 1)
        elif cmd == "d":
            self.y = min(self.rows - 1, n - 1)
        else:                                            # H, f
            self.y, self.x = min(self.rows - 1, arg(0) - 1), min(self.cols - 1, arg(1) - 1)
            self.cursor_moves += 1

    def _erase_display(self, mode: int) -> None:
        if mode == 0:
            self._erase(self.y, self.x, self.cols)
            for y in range(self.y + 1, self.rows):
                self._erase(y, 0, self.cols)
        elif mode == 1:
            for y in range(self.y):
                self._erase(y, 0, self.cols)
            self._erase(self.y, 0, self.x + 1)
        elif mode in (2, 3):
            for y in range(self.rows):
                self._erase(y, 0, self.cols)
            if mode == 3:
                self.scrollback.clear()

    def _insert_delete_lines(self, insert: bool, n: int) -> None:
        if not self.top <= self.y <= self.bot:
            return
        n = min(n, self.bot - self.y + 1)
        for _ in range(n):
            if insert:
                self.grid.pop(self.bot)
                self.grid.insert(self.y, self._blank_row())
            else:
                self.grid.pop(self.y)
                self.grid.insert(self.bot, self._blank_row())
        self.x, self.pending = 0, False

    def _chars(self, cmd: str, n: int) -> None:
        row, x = self.grid[self.y], self.x
        n = min(n, self.cols - x)
        if cmd == "P":
            row[x:] = row[x + n:] + [_BLANK] * n
        elif cmd == "@":
            row[x:] = ([_BLANK] * n + row[x:])[:self.cols - x]
        else:
            for i in range(x, x + n):
                row[i] = _BLANK
        self._fix_row(self.y)
        self.pending = False

    # ── output ──────────────────────────────────────────────────────────
    @staticmethod
    def _rows_text(grid: list[list[str]]) -> list[str]:
        return ["".join(r).rstrip() for r in grid]

    def text(self) -> str:
        """The visible screen: rows without trailing spaces, no blank rows after the content."""
        rows = self._rows_text(self.grid)
        while rows and not rows[-1]:
            rows.pop()
        return "\n".join(rows)

    def transcript_lines(self) -> list[str]:
        """The main buffer's history: what scrolled off plus its visible rows (a program that left the alternate screen - or never used it - prints
        its traceback here), no blank lines at the end."""
        rows = [*self.scrollback, *self._rows_text(self.main)]
        while rows and not rows[-1]:
            rows.pop()
        return rows


# ── keys ────────────────────────────────────────────────────────────────────

_NAMED = {"enter": "\r", "return": "\r", "esc": "\x1b", "escape": "\x1b", "tab": "\t", "space": " ", "backspace": "\x7f", "bs": "\x7f",
          "bspace": "\x7f", "delete": "\x1b[3~", "del": "\x1b[3~", "insert": "\x1b[2~", "ins": "\x1b[2~", "pageup": "\x1b[5~",
          "pgup": "\x1b[5~", "pagedown": "\x1b[6~", "pgdn": "\x1b[6~", "pgdown": "\x1b[6~", "btab": "\x1b[Z", "shifttab": "\x1b[Z", "stab": "\x1b[Z"}
_CURSOR = {"up": "A", "down": "B", "right": "C", "left": "D", "home": "H", "end": "F"}
_FKEYS = {1: "\x1bOP", 2: "\x1bOQ", 3: "\x1bOR", 4: "\x1bOS", 5: "\x1b[15~", 6: "\x1b[17~", 7: "\x1b[18~", 8: "\x1b[19~", 9: "\x1b[20~",
          10: "\x1b[21~", 11: "\x1b[23~", 12: "\x1b[24~"}
_CTRL = re.compile(r"^(?:ctrl|control|c)[-+](space|.)$|^\^(.)$", re.I)
_ALT = re.compile(r"^(?:alt|meta|m)[-+](.)$", re.I)
_WAIT = re.compile(r"^wait:\s*(\d+(?:\.\d+)?)\s*(ms|s)?$", re.I)
_FKEY = re.compile(r"^f(\d{1,2})$", re.I)
KEY_NAMES = "Enter Esc Tab Space Backspace Delete Up Down Left Right Home End PageUp PageDown F1-F12 Ctrl-<x> Alt-<x> wait:<ms> text:<literal>"


@dataclass
class Step:
    kind: str                  # "bytes" (fixed), "cursor" (an arrow / Home / End: the encoding depends on DECCKM) or "wait"
    data: Any                  # bytes | the final letter (A-D, H, F) | milliseconds
    label: str = ""


def _ctrl_char(c: str) -> str | None:
    c = c.lower()
    if c == "space" or c == "@":
        return "\x00"
    if len(c) == 1 and c.isalpha() and c.isascii():
        return chr(ord(c) & 0x1F)
    return {"[": "\x1b", "\\": "\x1c", "]": "\x1d", "^": "\x1e", "_": "\x1f", "?": "\x7f"}.get(c)


def parse_keys(keys: Any) -> tuple[list[Step], str | None]:
    """Tokens -> steps. A token is a key name (Enter, Esc, Tab, Space, Backspace, Delete, Up, Down, Left, Right, Home, End, PageUp, PageDown,
    F1..F12, Ctrl-C / C-c / ^C, Alt-x, any case, optionally in <angle brackets>), `wait:<ms>` or `text:<literal>`; anything else is typed as text.
    Returns (steps, error message or None)."""
    if keys in (None, "", []):
        return [], None
    if isinstance(keys, str):
        keys = keys.split()
    if not isinstance(keys, (list, tuple)):
        return [], f"keys must be a list of tokens, got {type(keys).__name__}"
    if len(keys) > MAX_KEYS:
        return [], f"too many keys ({len(keys)}; at most {MAX_KEYS})"
    steps: list[Step] = []
    for raw in keys:
        tok = str(raw)
        if tok == "":
            continue
        step = _parse_token(tok)
        if isinstance(step, str):
            return [], step
        steps.append(step)
    return steps, None


def _parse_token(tok: str) -> Step | str:
    if tok.startswith("text:"):
        return Step("bytes", tok[5:].replace("\n", "\r").encode("utf-8"), tok)
    m = _WAIT.match(tok)
    if m:
        ms = float(m.group(1)) * (1000 if (m.group(2) or "").lower() == "s" else 1)
        return Step("wait", int(min(ms, MAX_WAIT_MS)), tok)
    if tok.lower().startswith("wait:"):
        return f"bad wait token {tok!r}: use wait:<milliseconds>, e.g. wait:500"
    name = tok[1:-1] if len(tok) > 2 and tok[0] == "<" and tok[-1] == ">" else tok
    m = _CTRL.match(name)
    if m:
        ch = _ctrl_char(m.group(1) or m.group(2))
        return Step("bytes", ch.encode(), tok) if ch is not None else f"unknown control key {tok!r}"
    m = _ALT.match(name)
    if m:
        return Step("bytes", ("\x1b" + m.group(1)).encode(), tok)
    flat = re.sub(r"[-_ ]", "", name.lower())
    if flat in _CURSOR:
        return Step("cursor", _CURSOR[flat], tok)
    if flat in _NAMED:
        return Step("bytes", _NAMED[flat].encode(), tok)
    m = _FKEY.match(name)
    if m and int(m.group(1)) in _FKEYS:
        return Step("bytes", _FKEYS[int(m.group(1))].encode(), tok)
    return Step("bytes", tok.replace("\n", "\r").encode("utf-8"), tok)


def encode_step(step: Step, app_cursor: bool) -> bytes:
    """The bytes of one step. Arrows, Home and End follow DECCKM: ESC O x while the program asked for application cursor keys (curses
    `keypad(True)` does), else ESC [ x."""
    if step.kind == "cursor":
        return (b"\x1bO" if app_cursor else b"\x1b[") + str(step.data).encode()
    return step.data if isinstance(step.data, bytes) else b""


# ── the pty session ─────────────────────────────────────────────────────────

_WRAPPER = (
    "import os, sys, fcntl, termios\n"
    "try:\n    fcntl.ioctl(0, termios.TIOCSCTTY, 0)\n"
    "except OSError:\n    pass\n"
    "try:\n    os.execvp(sys.argv[1], sys.argv[1:])\n"
    "except OSError as exc:\n"
    "    sys.stderr.write('%s: %s\\n' % (sys.argv[1], exc.strerror))\n"
    "    os._exit(127 if exc.errno == 2 else 126)\n")
_NEEDS_SHELL = re.compile(r"[|&;<>()$`\\*?\[\]{}~\n]|^\s*[A-Za-z_]\w*=")


def argv_for(command: str) -> list[str]:
    """A plain command is split with shlex and exec'd directly; anything with shell syntax (pipes, &&, redirects, VAR=x, globs) goes through
    `bash -lc` like `run` does."""
    if _NEEDS_SHELL.search(command):
        return ["bash", "-lc", command]
    try:
        return shlex.split(command)
    except ValueError:
        return ["bash", "-lc", command]


def parse_size(size: Any) -> tuple[int, int]:
    """'80x24' -> (cols, rows), clamped to a sane range. Raises ValueError for anything else."""
    m = re.match(r"^\s*(\d{1,5})\s*[xX*,]\s*(\d{1,5})\s*$", str(size or DEFAULT_SIZE))
    if not m:
        raise ValueError(f"size must look like 80x24 (columns x rows), got {size!r}")
    return max(MIN_COLS, min(MAX_COLS, int(m.group(1)))), max(MIN_ROWS, min(MAX_ROWS, int(m.group(2))))


def build_env(cols: int, rows: int, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update({"TERM": "xterm-256color", "COLUMNS": str(cols), "LINES": str(rows), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if not any("utf" in env.get(k, "").lower() for k in ("LC_ALL", "LC_CTYPE", "LANG")):
        env["LC_ALL"] = "C.UTF-8"                       # box characters and wide text need a UTF-8 locale in the program's curses
    return headless_env(env)


class _Term:
    """The program on a pty: reading its output into a Screen, writing keys, waiting for quiet, stopping the whole process group."""

    def __init__(self, argv: list[str], cols: int, rows: int, cwd: Path | str | None, env: dict[str, str], max_bytes: int) -> None:
        self.screen = Screen(cols, rows)
        self.max_bytes = max_bytes
        self.total = 0
        self.capped = False
        self.eof = False
        self.last_out = time.monotonic()
        self.got_output = False
        self.returncode: int | None = None
        self.master, slave = os.openpty()
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            self.proc = subprocess.Popen([sys.executable, "-I", "-S", "-c", _WRAPPER, *argv], stdin=slave, stdout=slave, stderr=slave,
                                         cwd=cwd, env=env, close_fds=True, start_new_session=True)
        except BaseException:
            os.close(self.master)
            raise
        finally:
            os.close(slave)
        os.set_blocking(self.master, False)

    def alive(self) -> bool:
        if self.returncode is None:
            self.returncode = self.proc.poll()
        return self.returncode is None

    def done(self) -> bool:
        return self.capped or not self.alive()

    def pump(self, wait: float) -> int:
        """Read what the program printed (waiting at most `wait` seconds for the first byte), feed the screen, answer its queries."""
        if self.eof or self.capped:
            if wait > 0:
                time.sleep(min(wait, 0.02))
            return 0
        readable, _w, _x = select.select([self.master], [], [], max(0.0, wait))
        if not readable:
            return 0
        got = 0
        while got < 262_144:
            try:
                data = os.read(self.master, 65536)
            except BlockingIOError:
                break
            except OSError:                              # EIO: every holder of the slave has gone
                self.eof = True
                break
            if not data:
                self.eof = True
                break
            got += len(data)
            self.total += len(data)
            self.screen.feed_bytes(data)
            if self.total > self.max_bytes:
                self.capped = True
                break
        if got:
            self.last_out, self.got_output = time.monotonic(), True
            self._answer()
        return got

    def _answer(self) -> None:
        if self.screen.responses:
            data = b"".join(self.screen.responses)
            self.screen.responses.clear()
            self.write(data, time.monotonic() + 0.5)

    def write(self, data: bytes, deadline: float) -> bool:
        """Write to the program's terminal; False when it is gone or does not read."""
        view = memoryview(data)
        while view:
            try:
                view = view[os.write(self.master, view[:512]):]
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return False
                select.select([], [self.master], [], 0.05)
            except OSError:
                return False
        return True

    def wait_first_output(self, limit: float, deadline: float) -> None:
        end = min(time.monotonic() + limit, deadline)
        while not self.got_output and not self.done() and time.monotonic() < end:
            self.pump(min(0.05, max(0.0, end - time.monotonic())))

    def settle(self, quiet: float, deadline: float, max_wait: float = MAX_KEY_WAIT) -> None:
        """Return when no output arrived for `quiet` seconds, the program ended, `max_wait` passed or the deadline came."""
        end = min(time.monotonic() + max_wait, deadline)
        while True:
            now = time.monotonic()
            if self.done() or now >= end or now - self.last_out >= quiet:
                break
            self.pump(min(quiet - (now - self.last_out), end - now, 0.05))
        if not self.alive():
            self.drain()

    def idle(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.done():
            self.pump(min(0.05, max(0.0, end - time.monotonic())))
        if not self.alive():
            self.drain()

    def wait_exit(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while self.alive() and not self.capped and time.monotonic() < end:
            self.pump(min(0.05, max(0.0, end - time.monotonic())))
        self.drain()

    def drain(self) -> None:
        """What the program printed just before it ended (a traceback) is still in the pty buffer."""
        for _ in range(8):
            if not self.pump(0.02):
                break

    def close(self) -> None:
        """TERM, a short grace, then KILL to the whole group; the leader is reaped, the master closed. Nothing started by the program survives."""
        pgid = self.proc.pid
        if self.alive():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGTERM)
            end = time.monotonic() + 0.4
            while self.alive() and time.monotonic() < end:
                time.sleep(0.02)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)              # leftovers too (a child that outlived the leader, one that ignored TERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=5)
        with contextlib.suppress(OSError):
            os.close(self.master)


# ── the call ────────────────────────────────────────────────────────────────

@dataclass
class SmokeResult:
    command: str = ""
    cols: int = 80
    rows: int = 24
    status: str = ""                       # "exited" | "running" | "timeout" | "output_cap" | "error"
    exit_code: int | None = None
    signal: str = ""
    seconds: float = 0.0
    timeout: int = DEFAULT_TIMEOUT
    screen: str = ""
    cursor: tuple[int, int] = (1, 1)       # (row, column), 1-based
    alt: bool = False
    last_frame: str | None = None          # the alternate screen as it was when the program left it
    tail: list[str] = field(default_factory=list)       # last lines of the main-buffer transcript (only when the program exited)
    expects: list[tuple[str, bool, str]] = field(default_factory=list)       # (expectation, passed, where it was seen / why not)
    keys_sent: int = 0
    keys_total: int = 0
    traceback: bool = False
    line_based: bool = False
    startup_bytes: int = 0                 # what the program printed on its own before the first key (0: nothing; typed keys that the tty echoes do not count)
    history: str = ""                      # every settled screen of the run and the main-buffer transcript, for a text search after the run
    error: str = ""

    def passed(self, exit_code: int | None = None) -> bool:
        """The verdict a check uses: no error, every expect found, no timeout / output cap, and the exit as wanted - by default exit code 0 or
        still running; `exit_code=N` requires the program to have exited with N. A Python traceback fails unless a non-zero exit was asked for."""
        if self.error or not all(ok for _e, ok, _w in self.expects) or self.status in ("timeout", "output_cap", "error"):
            return False
        if exit_code is not None:
            return self.status == "exited" and self.exit_code == exit_code and (exit_code != 0 or not self.traceback)
        return not self.traceback and (self.status == "running" or self.exit_code == 0)


def _matcher(expect: str) -> tuple[Any, str]:
    """(callable text -> bool, error text). `re:` prefix = regex (multi-line, case-sensitive), else a plain substring."""
    if expect.startswith("re:"):
        try:
            rx = re.compile(expect[3:], re.M)
        except re.error as exc:
            return None, f"bad regex ({exc})"
        return (lambda text: rx.search(text) is not None), ""
    return (lambda text: expect in text), ""


def _evaluate(expects: list[str], snaps: list[tuple[str, str]], final: str, transcript: str) -> list[tuple[str, bool, str]]:
    out = []
    for e in expects:
        match, err = _matcher(e)
        if match is None:
            out.append((e, False, err))
            continue
        where = "final screen" if match(final) else next((label for label, text in snaps if match(text)), "") or ("transcript" if match(transcript) else "")
        out.append((e, bool(where), where or "not on any settled screen nor in the transcript"))
    return out


def run_smoke(command: str, keys: Any = None, size: Any = DEFAULT_SIZE, expect: Any = None, timeout: Any = DEFAULT_TIMEOUT,
              settle_ms: Any = DEFAULT_SETTLE_MS, cwd: Path | str | None = None, env: dict[str, str] | None = None,
              max_bytes: int = MAX_BYTES) -> SmokeResult:
    """Run `command` on a pty of `size`, send `keys`, evaluate `expect`. Never raises; a problem with the arguments or the start comes back as
    `error`. `expect` entries are plain substrings or `re:` regexes; each is looked for on the final screen, on every settled screen the run
    showed (initial, after each key) and in the main-buffer transcript."""
    res = SmokeResult(command=str(command))
    try:
        res.cols, res.rows = parse_size(size)
    except ValueError as exc:
        res.status, res.error = "error", str(exc)
        return res
    steps, bad = parse_keys(keys)
    wanted = [str(e) for e in ([expect] if isinstance(expect, str) else expect or []) if str(e)][:20]
    if not PTY_OK:
        res.status, res.error = "error", "the terminal-UI smoke test needs a POSIX pty (Linux, macOS, Termux or WSL); it is not available on Windows"
        return res
    argv = argv_for(str(command))
    if bad or not argv:
        res.status, res.error = "error", bad or "needs a command"
        return res
    if any(len(e) > MAX_EXPECT_CHARS for e in wanted):
        res.status, res.error = "error", f"an expect entry is longer than {MAX_EXPECT_CHARS} characters"
        return res
    try:
        res.timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
        quiet = max(50, min(int(settle_ms or DEFAULT_SETTLE_MS), 2000)) / 1000
    except (TypeError, ValueError):
        res.timeout, quiet = DEFAULT_TIMEOUT, DEFAULT_SETTLE_MS / 1000
    res.keys_total = len(steps)
    t0 = time.monotonic()
    deadline = t0 + res.timeout
    try:
        term = _Term(argv, res.cols, res.rows, cwd, build_env(res.cols, res.rows, env), max_bytes)
    except OSError as exc:
        res.status, res.error = "error", f"could not start the program on a pty: {exc}"
        return res
    snaps: list[tuple[str, str]] = []
    scr = term.screen

    def snap(label: str) -> None:
        text = scr.text()
        if text and (not snaps or snaps[-1][1] != text) and len(snaps) < MAX_SNAPSHOTS:
            snaps.append((label, text))

    def unmet() -> bool:
        return not all(ok for _e, ok, _w in _evaluate(wanted, snaps, scr.text(), "\n".join(scr.transcript_lines())))

    try:
        term.wait_first_output(STARTUP_WAIT, deadline)
        term.settle(quiet, deadline)
        snap("initial screen")
        res.startup_bytes = term.total
        for i, step in enumerate(steps, 1):
            if term.done() or time.monotonic() >= deadline:
                break
            if step.kind == "wait":
                term.idle(min(int(step.data) / 1000, max(0.0, deadline - time.monotonic())))
                label = f"after wait {i}"
            else:
                if not term.write(encode_step(step, scr.app_cursor), deadline):
                    break
                term.last_out = time.monotonic()
                term.settle(quiet, deadline)
                label = f"after key {i} ({step.label})"
            res.keys_sent = i
            snap(label)
        if not term.done():
            if wanted and unmet():                       # expect-with-timeout: poll the screen until every entry was seen or time is up
                seen = scr.text()
                while not term.done() and time.monotonic() < deadline and unmet():
                    term.pump(0.05)
                    if scr.text() != seen:
                        seen = scr.text()
                        snap("later")
            else:
                term.wait_exit(min(EXIT_GRACE, max(0.0, deadline - time.monotonic())))
        term.drain()
        snap("final screen")
        alive = term.alive()
        res.seconds = time.monotonic() - t0
        res.screen, res.alt = scr.text(), scr.alt
        res.cursor = (scr.y + 1, scr.x + 1)
        res.last_frame = scr.last_alt_frame
        res.line_based = not scr.alt_seen and scr.cursor_moves == 0
        if term.capped:
            res.status = "output_cap"
        elif alive:
            res.status = "timeout" if time.monotonic() >= deadline else "running"
        else:
            res.status, res.exit_code = "exited", term.returncode
            if res.exit_code is not None and res.exit_code < 0:
                with contextlib.suppress(ValueError):
                    res.signal = signal.Signals(-res.exit_code).name
            res.tail = scr.transcript_lines()[-TAIL_LINES:]
        transcript = "\n".join(scr.transcript_lines())
        res.expects = _evaluate(wanted, snaps, res.screen, transcript)
        res.history = "\n".join([text for _label, text in snaps] + [transcript])[:HISTORY_CHARS]
        res.traceback = bool(_TRACEBACK.search(transcript) or _TRACEBACK.search(res.screen) or (res.last_frame and _TRACEBACK.search(res.last_frame)))
    finally:
        term.close()
    return res


# ── what the model reads ────────────────────────────────────────────────────

def _crop(text: str, limit: int) -> str:
    """Keep the top and the bottom rows (title and status bar) of a screen that does not fit, drop the middle."""
    if len(text) <= limit:
        return text
    rows = text.split("\n")
    head, tail, used = [], [], 0
    for r in rows:
        if used + len(r) + 1 > limit * 0.6:
            break
        head.append(r)
        used += len(r) + 1
    for r in reversed(rows[len(head):]):
        if used + len(r) + 1 > limit - 30:
            break
        tail.insert(0, r)
        used += len(r) + 1
    return "\n".join([*head, f"[… {len(rows) - len(head) - len(tail)} rows omitted …]", *tail])


def _status_text(r: SmokeResult) -> str:
    if r.status == "exited":
        extra = f" (killed by {r.signal})" if r.signal else ""
        return f"exited with code {r.exit_code}{extra}" if not r.signal else f"ended{extra}"
    if r.status == "timeout":
        return f"KILLED BY TIMEOUT after {r.timeout}s (the program was still running)"
    if r.status == "output_cap":
        return f"KILLED: more than {MAX_BYTES // 1_000_000} MB of output"
    return "still running after the last key (stopped by the harness)"


def format_result(r: SmokeResult, max_chars: int = MAX_CHARS) -> str:
    """The compact answer: status, loud problems, expect verdicts, cursor, the final screen, the last screen of an alternate-screen program, and the
    transcript tail of a program that exited. About `max_chars` characters (the screen is cropped first)."""
    if r.error:
        return f"error: tui_smoke {r.error}"
    top = [f"tui_smoke: {_status_text(r)}  ({r.seconds:.1f}s, {r.cols}x{r.rows})"]
    if r.status == "exited" and r.exit_code not in (0, None):
        top.append(f"!! NON-ZERO EXIT: code {r.exit_code}" + (f" ({r.signal})" if r.signal else ""))
    if r.traceback:
        top.append("!! a traceback / crash message was printed (see the transcript below)")
    if r.keys_sent < r.keys_total:
        top.append(f"!! only {r.keys_sent} of {r.keys_total} keys were sent: the program " + ("ended" if r.status == "exited" else "stopped reading") + " earlier")
    if r.expects:
        top.append(f"expect: {sum(1 for _e, ok, _w in r.expects if ok)}/{len(r.expects)} PASS")
        top += [f"  {'PASS' if ok else 'FAIL'}  {e[:100]!r}  ({where})" for e, ok, where in r.expects]
    top.append(f"cursor at ({r.cursor[0]},{r.cursor[1]}), {'alternate' if r.alt else 'main'} screen"
               + ("; no cursor addressing or alternate screen was used: a line-based program, this is its plain output" if r.line_based else ""))
    shown = "\n".join(ln[:110] for ln in r.tail)
    tail = ""
    if r.tail and (r.exit_code not in (0, None) or r.traceback or any(ln.strip() and ln.strip() not in r.screen for ln in r.tail)):
        tail = f"\ntranscript tail (last {len(r.tail)} lines of the main buffer):\n{shown}"
    head = "\n".join(top)
    room = max(500, max_chars - len(head) - len(tail) - 40)
    screen = _crop(r.screen, room) if r.screen else "(blank screen)"
    out = f"{head}\nscreen:\n{screen}"
    left = room - len(screen)
    if r.last_frame and r.last_frame != r.screen and left > 300 and (len(r.screen.splitlines()) <= 3):
        out += f"\nlast full-screen frame before the program left the alternate screen:\n{_crop(r.last_frame, left - 80)}"
    return out + tail


# ── repo detection for the system prompt ────────────────────────────────────

_TUI_IMPORT = re.compile(r"^\s*(?:import|from)\s+(?:curses|textual|urwid|pygame|prompt_toolkit|blessed)\b", re.M)
PROMPT_SENTENCE = ("Terminal/curses/game programs cannot be run with `run` (no terminal): test them with tui_smoke(command, keys, expect). "
                   "To prove a fix, make `expect` contain the text the player must SEE (e.g. the game over banner), not only that nothing crashes.")


def repo_uses_tui(root: Path | str, max_files: int = 40, max_bytes: int = 65536) -> bool:
    """Cheap check at session start: does a top-level .py file import curses / textual / urwid / pygame / prompt_toolkit / blessed?"""
    try:
        files = sorted(Path(root).glob("*.py"))[:max_files]
    except OSError:
        return False
    for f in files:
        try:
            with open(f, "rb") as fh:
                text = fh.read(max_bytes).decode("utf-8", errors="replace")
        except OSError:
            continue
        if _TUI_IMPORT.search(text):
            return True
    return False
