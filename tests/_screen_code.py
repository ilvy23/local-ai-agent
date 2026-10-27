"""Screen-replay helper for the code panel tests (copied from tests/test_liveview.py::_final_screen)."""

import io
import re

from rich.console import Console

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


def render(renderable, width, plain=True) -> str:
    """Print a renderable on a pipe console `width` columns wide and return the final screen text."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=not plain, color_system=None if plain else "standard", highlight=False).print(renderable)
    return _final_screen(buf.getvalue())
