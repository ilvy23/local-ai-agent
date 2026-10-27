"""Run-time redirect to `tui_smoke`: the model got the prompt sentence and still never called it (0 calls in 49 steps of a real run on a
crashing curses game); it wrote scripts that fake the screen or drive the logic functions and "proved" a symptom patch with them.
The harness therefore looks at what `run` was asked to do and answers on the spot, one factual line, no model call:

  (a) the command starts a Python program that imports curses / textual / urwid / blessed / prompt_toolkit / pygame, and the output shows
      the missing terminal (setupterm, nocbreak() returned ERR, not a tty ...): the hint, and tools.py runs the program once through
      tui_smoke itself (no keys, 3 s) so the model sees what the program shows;
  (b) a script (heredoc, `-c`, stdin or a file) that fakes a screen (a Fake/Mock/Dummy screen class with getmaxyx, or a patched
      curses.wrapper / initscr): the hint.

Pure functions; the once-per-program bookkeeping and the automatic run live in CodeTools."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from agent.codeagent import tuismoke

LONG_HINT = ("This is a terminal program. Use tui_smoke(command=..., keys=[...], expect=[...]) instead of faking a screen: it runs the program in a "
             "real terminal and returns the final screen text.")
SHORT_HINT = "Terminal program: use tui_smoke(command, keys, expect)."
AUTO_TIMEOUT = 3                  # seconds of the automatic run

TERMINAL_ERRORS = re.compile(r"setupterm|nocbreak\(\) returned ERR|Redirection is not supported|not a tty|Inappropriate ioctl for device|curses\.error|"
                             r"error opening terminal|Error opening terminal", re.I)
_PY_PROG = re.compile(r"""^\s*(?:(?:cd\s+\S+|source\s+\S+|export\s+\S+)\s*&&\s*)*(?:uv\s+run\s+(?:--\S+\s+)*)?(?:\S*/)?python[\d.]*\s+(?:-[uOB]\s+)*"""
                      r"""(?:-m\s+(?P<mod>[\w.]+)|(?P<file>[\w./+-]+\.py))(?=\s|$)""")
_FAKE_NAME = re.compile(r"(?:fake|mock|dummy|stub)\w*(?:screen|scr|win|window|stdscr)|(?:screen|scr|win|window|stdscr)\w*(?:fake|mock|dummy|stub)", re.I)
_PATCHED = re.compile(r"\bcurses\.(?:wrapper|initscr)\s*=(?!=)|\bpatch(?:\.object)?\(\s*['\"]?[\w.]*curses[\w.'\",\s]*(?:wrapper|initscr)|"
                      r"monkeypatch\.setattr\([^)]*curses[^)]*(?:wrapper|initscr)", re.I)
_FILE_WRITING_SHELL = re.compile(r"[|<>]|\btee\b")      # a pipe or a redirection: no automatic second run
_MAX_READ = 65536


def _read(root: Path, name: str) -> str:
    try:
        p = Path(name)
        p = p if p.is_absolute() else root / p
        with open(p, "rb") as fh:
            return fh.read(_MAX_READ).decode("utf-8", errors="replace")
    except OSError:
        return ""


def program_of(command: str, root: Path) -> tuple[str, str] | None:
    """(repo-relative or given name, source text) of the program in a simple `python3 file.py [args]` / `python3 -m mod` command, else None."""
    m = _PY_PROG.match(command)
    if not m:
        return None
    if m.group("file"):
        name = m.group("file")
        text = _read(root, name)
        return (name, text) if text else None
    mod = m.group("mod").replace(".", "/")
    for name in (f"{mod}.py", f"{mod}/__main__.py"):
        text = _read(root, name)
        if text:
            return name, text
    return None


def is_tui_source(text: str) -> bool:
    return bool(tuismoke._TUI_IMPORT.search(text))


def fakes_screen(text: str) -> bool:
    """A script that stands in for the terminal: a fake screen class together with getmaxyx, or a patched curses.wrapper / initscr."""
    return bool(_PATCHED.search(text) or ("getmaxyx" in text and _FAKE_NAME.search(text)))


def has_terminal_error(output: str) -> bool:
    return bool(TERMINAL_ERRORS.search(output))


def auto_run_ok(command: str) -> bool:
    """The command may be run a second time through tui_smoke: no pipe, no redirection, no tee (nothing it writes besides the program's own work)."""
    try:
        shlex.split(command)
    except ValueError:
        return False
    return not _FILE_WRITING_SHELL.search(command)


def scan(command: str, stdin: str | None, output: str, root: Path) -> tuple[str, str] | None:
    """What `run` should tell the model: (key, kind) with kind "fake" or "tui", or None. The key names the program (once-per-program hints)."""
    prog = program_of(command, root)
    text = f"{command}\n{stdin or ''}\n{prog[1] if prog else ''}"
    if fakes_screen(text):
        return "fake", "fake"
    if prog and is_tui_source(prog[1]) and has_terminal_error(output):
        return f"prog:{prog[0]}", "tui"
    return None
