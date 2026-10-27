"""Clean terminal output for web research: plain words, one line per source, no emoji."""

from __future__ import annotations

import contextlib
import urllib.parse
from collections.abc import Iterator
from typing import Any

from rich.console import Console
from rich.markup import escape


def short_url(url: str, limit: int) -> str:
    p = urllib.parse.urlparse(url)
    text = (p.netloc.removeprefix("www.") + p.path).rstrip("/")
    return text if len(text) <= limit else text[: max(limit - 1, 8)] + "…"


class Narrator:
    """Prints what the research is doing:

        search  rust async runtimes
          1  tokio.rs/tokio/tutorial        read 0.7s
          2  docs.rs/async-std              cached
          3  example.com/report             skipped (pdf file)
        3 sources · 1 read · 1 cached · 2.3s
    """

    def __init__(self, console: Any) -> None:
        self.console = console

    @contextlib.contextmanager
    def busy(self, label: str, suffix: str = "") -> Iterator[None]:
        """Animated `searching. .. ...` while waiting (only on a real terminal)."""
        if not getattr(self.console, "is_terminal", False):
            yield
            return
        try:
            from agent.tui import _ThinkingSpinner
        except Exception:  # noqa: BLE001 - cosmetic
            yield
            return
        with _ThinkingSpinner(self.console, label=label, suffix=suffix):
            yield

    def header(self, kind: str, subject: str) -> None:
        self.console.print(f"[bold]{escape(kind)}[/bold]  {escape(subject)}")

    def row(self, index: int, url: str, state: str, detail: str = "") -> None:
        width = max(int(getattr(self.console, "width", 80)) - 30, 24)
        color = {"read": "green", "cached": "cyan", "skipped": "yellow", "failed": "red", "timeout": "red"}.get(state, "dim")
        label = state + (f" {detail}" if detail and state in ("read",) else f" ({detail})" if detail else "")
        self.console.print(f"  [dim]{index}[/dim]  {escape(short_url(url, width)):<{width}}  [{color}]{escape(label)}[/{color}]")

    def footer(self, text: str) -> None:
        self.console.print(f"[dim]{escape(text)}[/dim]")

    def note(self, text: str) -> None:
        self.console.print(f"[dim]{escape(text)}[/dim]")
