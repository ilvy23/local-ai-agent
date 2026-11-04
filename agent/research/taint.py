"""Remember what this turn read from the web, so an action built from it can be recognised.

A hostile page cannot run anything by itself, but it can try to talk the model into a command or a path.
When a tool call that would normally need approval carries text that came straight from a web page, the
approval prompt must be asked even where a folder grant (/auto, /medium, /greenlist) would skip it.

shortcut: plain substring/shingle matching on the page text; a model that rewrites the command enough
slips past it, but the approval gate for risky tools stays on regardless.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any

_lock = threading.Lock()


class _State:
    """What ONE turn read from the web (and what its calc tool computed)."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.computed: list[str] = []          # results of the calc tool this turn


# Per turn, not per process: each web request (an asyncio task, whose to_thread workers inherit its context) and the chat's turn
# get their own record, so one session starting a turn no longer wipes the record another session's approval check relies on.
# A plain thread that inherits no context uses the record of the turn begun last - exactly what the single shared record did.
_CURRENT: contextvars.ContextVar[_State | None] = contextvars.ContextVar("web_taint", default=None)
_LAST = [_State()]


def _st() -> _State:
    return _CURRENT.get() or _LAST[0]


def begin() -> None:
    """A fresh record for the turn that starts in this context (call once per NEW turn, never on a resume)."""
    st = _State()
    _CURRENT.set(st)
    _LAST[0] = st
_MAX_TEXTS = 40
_WINDOW = 24         # characters of an argument that must reappear verbatim in a page to count as copied


def add(text: str) -> None:
    if text and text.strip():
        with _lock:
            _st().texts.append(text)
            del _st().texts[:-_MAX_TEXTS]


def clear() -> None:
    with _lock:
        _st().texts.clear()
        _st().computed.clear()


def add_computed(text: str) -> None:
    """Record what the calc tool returned, so a block claiming "computed" can be checked against it."""
    if text and text.strip():
        with _lock:
            _st().computed.append(text)
            del _st().computed[:-_MAX_TEXTS]


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def copied(arguments: dict[str, Any]) -> bool:
    """True if a stretch of any string argument appears verbatim in text read from the web this turn."""
    with _lock:
        pages = list(_st().texts)
    if not pages:
        return False
    blob = "\n".join(pages)
    for text in _strings(arguments):
        text = " ".join(text.split())
        if len(text) < _WINDOW:
            continue
        step = max(_WINDOW // 2, 1)
        for i in range(0, len(text) - _WINDOW + 1, step):
            if text[i:i + _WINDOW] in " ".join(blob.split()):
                return True
    return False


def has_number(value: float) -> bool:
    """True if `value` appears as a number in text read from the web this turn."""
    from agent.research.verdict import _has_number

    with _lock:
        blob = "\n".join(_st().texts)
    return _has_number(value, blob)


def has_computed_number(value: float) -> bool:
    """True if `value` appears in a calc result of this turn."""
    from agent.research.verdict import _has_number

    with _lock:
        blob = "\n".join(_st().computed)
    return _has_number(value, blob)
