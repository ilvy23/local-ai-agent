"""Shared test helper: replay a raw terminal byte stream into the screen a user would see.

Copied from tests/test_liveview.py::_final_screen so UI tests can reuse it."""

import re

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_CTRL = re.compile(r"\x1b\[(\d*)([A-Za-z])|(\r)|(\n)")


def final_screen(raw: str) -> str:
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
            elif letter == "K":
                lines[row] = lines[row][:col] if count == "0" else ""
    write(row, col, ANSI.sub("", raw[pos:]))
    return "\n".join(lines)
