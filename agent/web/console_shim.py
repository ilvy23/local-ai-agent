"""A minimal stand-in for rich.console.Console, for the four tool
constructors that only ever call `.print(...)` on it for narration
(make_web_tools, make_diagram_tool, and tui.py's _make_remember_tool /
_make_take_note_tool). Their actual handler logic needs no change - they
just don't have a terminal to narrate into here, so narration becomes a
best-effort SSE `log` event instead.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_MARKUP_RE = re.compile(r"\[/?[a-zA-Z0-9_ ]*\]")


def _strip_markup(text: str) -> str:
    """Rich markup like "[dim]...[/dim]" means nothing outside a terminal."""
    return _MARKUP_RE.sub("", text)


class WebConsole:
    """Pass one of these wherever tui.py would pass a real Console to a
    tool-constructor function - `make_web_tools(console)`,
    `make_diagram_tool(console)`, etc.

    Tool handlers run inside `asyncio.to_thread` (agent_runtime.py), so a
    handler's own `console.print(...)` call happens on a worker thread with
    no event loop of its own to await the SSE `emit` coroutine onto.
    `asyncio.run_coroutine_threadsafe` is the correct bridge for exactly
    this - schedule the emit onto the loop that was running when this
    WebConsole was constructed (i.e. the route handler's loop), from
    whichever thread `.print()` happens to be called on. Best-effort only:
    a lost narration line is never as bad as a broken chat turn, so any
    failure here is swallowed, not raised.
    """

    def __init__(
        self,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._emit = emit
        self._loop = loop

    def print(self, *args: Any, **_kwargs: Any) -> None:
        text = _strip_markup(" ".join(str(a) for a in args)).strip()
        if not text:
            return
        logger.info("web console: %s", text)
        self.emit_event("log", text=text)

    def emit_event(self, event_type: str, **fields: Any) -> None:
        """Push an arbitrary structured SSE event directly, bypassing the
        model's own reply entirely - used by tools (send_file) that need to
        guarantee the frontend renders something specific (a player, a
        card) regardless of what the model says afterward. Same
        thread-safe bridge as .print(); best-effort, never raises."""
        if self._emit is None or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._emit({"type": event_type, **fields}), self._loop
            )
        except RuntimeError:
            pass  # loop already closed/shutting down

    @contextmanager
    def status(self, text: str = "", **_kwargs: Any):
        """No-op stand-in for Console.status(...) - a couple of reused CLI
        functions (e.g. tui.py's _distill_on_exit) wrap work in a terminal
        spinner; there's no terminal here, so just run the wrapped code."""
        self.print(text)
        yield self
