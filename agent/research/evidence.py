"""What this turn's web research established, kept so memory can decide what is worth remembering.

web_search / web_fetch record their verdict and the passages they showed the model. At the end of the turn the chat
hands the record to the fact saver (tui._save_learned_facts): only claims backed by a verified or likely verdict, with
the sources next to them, may become memory. The record is cleared at the start of every user turn.
"""

from __future__ import annotations

import threading

_RANK = {"verified": 4, "likely": 3, "single_source": 2, "conflicting": 1, "low_trust": 0, "from_memory": 0}
_lock = threading.Lock()
_last: dict = {"items": [], "verdict": {}}
_state: dict = {"label": "", "sources": []}
SAVEABLE = frozenset({"verified", "likely"})


def clear() -> None:
    """Forget this turn's research (called at the start of every user turn), including the last search's structured result."""
    with _lock:
        _state.update(label="", sources=[])
        _last["items"], _last["verdict"] = [], {}


def record(label: str, sources: list[dict]) -> None:
    """`sources`: dicts with domain, date, level and passage (only the ones actually shown to the model)."""
    with _lock:
        if _RANK.get(label, 0) >= _RANK.get(_state["label"], -1):
            _state["label"] = label
        _state["sources"] = (_state["sources"] + [s for s in sources if s.get("passage")])[-6:]


def take() -> dict | None:
    """The record of this turn (or None when nothing was read from the web)."""
    with _lock:
        if not _state["label"]:
            return None
        return {"label": _state["label"], "sources": [dict(s) for s in _state["sources"]]}


def set_last_search(items: list[dict], vd: dict) -> None:
    """The structured result of the most recent web_search (what `verify` reads after each search it runs)."""
    with _lock:
        _last["items"], _last["verdict"] = list(items), dict(vd)


def last_search() -> dict:
    with _lock:
        return {"items": list(_last["items"]), "verdict": dict(_last["verdict"])}
