"""What you see while the model works: a plain status line, then the reply written live.

* `ThinkingText` builds the status line shown while nothing can be displayed yet:
  "<label>...  12s" - the label, growing dots and the seconds waited. Nothing else.
* `LiveMarkdown` renders a reply as Markdown *while it is still being generated*: finished
  blocks (paragraphs, headings, lists, code blocks) are printed once and stay in the
  scrollback; only the block currently being written is redrawn. That keeps a long answer
  from ever redrawing the whole screen.
* `LiveReply` decides, from the first few characters, whether the stream is a normal answer
  (show it live) or the model's text-form tool call (never show raw JSON).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

class ThinkingText:
    """Builds the status line: label with growing dots, then the seconds waited."""

    def __init__(
        self,
        label: str = "thinking",
        clock: Callable[[], float] = time.monotonic,
        style: str = "dim",
        suffix: str = "",
    ) -> None:
        """`style` colours label and dots; `suffix` is shown after them, e.g. "→ demo.py"."""
        self.style = style
        self.suffix = suffix
        self.label = label
        self._clock = clock
        self._started = clock()

    def render(self) -> Text:
        now = self._clock()
        elapsed = int(now - self._started)
        dots = "." * (int((now - self._started) / 0.5) % 3 + 1)
        text = Text()
        text.append(f"{self.label}{dots}", style=self.style)
        if self.suffix:
            text.append(f" → {self.suffix}", style=self.style)
        if elapsed >= 2:
            text.append(f"  {elapsed}s", style="dim")
        return text


# ----------------------------------------------------------------------------- live markdown

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def _is_list(block: str) -> bool:
    first = block.lstrip("\n").split("\n", 1)[0]
    return bool(_LIST_RE.match(first))


def split_blocks(text: str) -> tuple[list[str], str]:
    """Split streamed Markdown into finished blocks and the unfinished tail.

    A block ends at a blank line outside a fenced code block. Consecutive list blocks are
    kept together (so numbering survives), and a trailing list stays in the tail until a
    non-list block starts - otherwise a loose list would be cut in two. Returns
    (blocks, tail) where blocks + tail reproduce `text` up to blank-line separators."""
    lines = text.split("\n")
    complete, partial = lines[:-1], lines[-1]
    blocks: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in complete:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    tail_lines = current + [partial]

    merged: list[str] = []
    for block in ("\n".join(b) for b in blocks):
        if merged and _is_list(block) and _is_list(merged[-1]):
            merged[-1] = merged[-1] + "\n\n" + block
        else:
            merged.append(block)
    tail = "\n".join(tail_lines)
    if merged and _is_list(merged[-1]) and (not tail.strip() or _is_list(tail)):
        tail = merged.pop() + "\n\n" + tail
    return merged, tail


class LiveMarkdown:
    """Renders a Markdown reply progressively while tokens are still arriving."""

    def __init__(self, console: Console, refresh_per_second: int = 15) -> None:
        self.console = console
        self.text = ""
        self._refresh = refresh_per_second
        self._live: Live | None = None

    # -- internals
    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _show_tail(self, tail: str) -> None:
        if not tail.strip():
            self._stop_live()
            return
        renderable = Markdown(tail)
        if self._live is None:
            # transient: when the block is finished the live frame is erased and the block is
            # printed for good (printing it as well as leaving the frame would show it twice)
            self._live = Live(renderable, console=self.console, refresh_per_second=self._refresh,
                              transient=True, vertical_overflow="visible")
            self._live.start()
        else:
            self._live.update(renderable, refresh=True)

    def _too_tall(self, tail: str) -> bool:
        if len(tail) < 600:
            return False
        height = len(self.console.render_lines(Markdown(tail), self.console.options, pad=False))
        return height > max(self.console.height - 4, 8)

    def _flush_tail_head(self, tail: str) -> str:
        """The unfinished block no longer fits on screen: print everything up to its last
        complete line for good and keep only the newest line live."""
        cut = tail.rfind("\n")
        if cut <= 0:
            return tail
        head, rest = tail[:cut], tail[cut + 1:]
        fences = [ln for ln in head.split("\n") if _FENCE_RE.match(ln)]
        if len(fences) % 2:                      # cut inside a code block: close it, reopen below
            opener = fences[-1].strip()
            head += "\n" + opener[:3]
            rest = opener + "\n" + rest
        self._stop_live()
        self.console.print(Markdown(head))
        return rest

    # -- public
    def feed(self, piece: str) -> None:
        self.text += piece
        blocks, tail = split_blocks(self.text)
        if blocks:
            self._stop_live()
            for block in blocks:
                self.console.print(Markdown(block))
            self.text = tail.lstrip("\n")
            tail = self.text
        if self._too_tall(tail):
            tail = self._flush_tail_head(tail)
            self.text = tail
        self._show_tail(tail)

    def finish(self) -> None:
        """Show whatever is left for good and leave the cursor on a fresh line."""
        self._stop_live()
        if self.text.strip():
            self.console.print(Markdown(self.text))
        self.text = ""


# ----------------------------------------------------------------------------- reply glue

_TOOLISH_RE = re.compile(r"^(\{|\[|<tool|<function|<\||<think)|^```\s*(json|tool)", re.IGNORECASE)
_DECIDE_AFTER = 14     # characters needed before we trust the first-token guess


class LiveReply:
    """Streams a reply to the console unless it is (or looks like) a tool call.

    The model's text-form tool calls ({"tool": ..., "arguments": ...}) must never be shown as
    raw JSON, so the first characters decide: JSON-ish -> stay silent and let the normal
    tool path handle it; anything else -> stop the status line and write the reply live."""

    def __init__(self, console: Console, persona_name: str, stop_status: Callable[[], None]) -> None:
        self.console = console
        self.persona = persona_name
        self._stop_status = stop_status
        self.state = "pending"            # pending | live | tool
        self._buf = ""
        self._md: LiveMarkdown | None = None

    @property
    def started(self) -> bool:
        return self.state == "live"

    def on_content(self, piece: str) -> None:
        if self.state == "tool" or not piece:
            return
        self._buf += piece
        if self.state == "live":
            assert self._md is not None
            self._md.feed(piece)
            return
        head = self._buf.lstrip()
        if not head:
            return
        if _TOOLISH_RE.match(head):
            self.state = "tool"
            return
        if len(head) < _DECIDE_AFTER and "\n" not in head and not head.startswith("`"):
            return
        if head.startswith("`") and len(head) < 8:
            return
        self.state = "live"
        self._stop_status()
        self.console.print(f"[bold cyan]{self.persona}>[/bold cyan]")
        self._md = LiveMarkdown(self.console)
        self._md.feed(self._buf.lstrip())

    def finish(self, context_bar: str = "") -> bool:
        """Close the live view. True if a reply was shown (the caller must not print it again)."""
        if self.state != "live":
            return False
        assert self._md is not None
        self._md.finish()
        if context_bar:
            self.console.print(context_bar)
        return True

    def abandon(self) -> None:
        """A tool call followed the text: just end the live block cleanly."""
        if self.state == "live" and self._md is not None:
            self._md.finish()


def stream_settings(ui: dict[str, Any]) -> dict[str, Any]:
    return {
        "live": bool(ui.get("live", True)),
    }

