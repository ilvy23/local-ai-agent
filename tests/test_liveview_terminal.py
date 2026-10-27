"""The live view on a real (emulated) terminal: cursor movement and scrollback included.

Needs the `pyte` terminal emulator (`uv run --with pyte pytest ...`); skipped when it is missing.
"""

import io

import pytest
from rich.console import Console
from rich.markdown import Markdown

pyte = pytest.importorskip("pyte")

from agent.liveview import LiveMarkdown

DOC = (
    "# Titel\n\nErster Absatz mit **fett** und `code`.\n\n- eins\n- zwei\n- drei\n\n1. a\n2. b\n\n"
    "```python\nprint('x')\n\nprint('y')\n```\n\nSchluss.\n"
)


def _screen_text(text, step, width=60, height=12):
    buf = io.StringIO()
    console = Console(file=buf, width=width, height=height, force_terminal=True, color_system="truecolor")
    live = LiveMarkdown(console)
    for i in range(0, len(text), step):
        live.feed(text[i:i + step])
    live.finish()
    screen = pyte.HistoryScreen(width, height, history=5000, ratio=1.0)
    pyte.Stream(screen).feed(buf.getvalue().replace("\n", "\r\n"))
    rows = list(screen.history.top)
    lines = ["".join(row[x].data for x in range(width)).rstrip() for row in rows]
    lines += [line.rstrip() for line in screen.display]
    return [line for line in lines if line.strip()]


@pytest.mark.parametrize("step", [1, 5, 200])
def test_final_screen_equals_a_plain_print(step):
    plain = Console(file=io.StringIO(), width=60, force_terminal=False, color_system=None)
    plain.print(Markdown(DOC))
    want = [ln.rstrip() for ln in plain.file.getvalue().splitlines() if ln.strip()]
    assert _screen_text(DOC, step) == want


def test_a_paragraph_taller_than_the_screen_shows_every_line_once():
    text = "\n".join(f"Zeile{n:03d} " + "wort " * 8 for n in range(80))
    joined = "\n".join(_screen_text(text, 40))
    assert all(joined.count(f"Zeile{n:03d}") == 1 for n in range(80))


def test_a_code_block_taller_than_the_screen_shows_every_line_once():
    code = "```python\n" + "\n".join(f"zeile{n:03d} = {n}" for n in range(60)) + "\n```\n"
    joined = "\n".join(_screen_text(code, 50, height=10))
    assert all(joined.count(f"zeile{n:03d}") == 1 for n in range(60))
