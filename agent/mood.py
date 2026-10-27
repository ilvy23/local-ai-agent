"""Tiny rule-based mood/status side-channel (Phase 5 "emotes").

Maps a small, fixed set of engine/status event names to a kaomoji, with no
model call involved - this is cosmetic flavor on top of real output, never a
replacement for it. Callers (agent/tui.py for the terminal, the web status
event in agent/web/routes/chat.py) add the result as a small additive field
on a signal that already exists; nothing here invents a new event stream.

The canonical event names are deliberately few. A future caller that wants
more granularity passes `extra` (e.g. read from config, see
`config.yaml`'s `ui.mood_map`) to merge in its own event -> kaomoji entries
without touching this module.
"""

from __future__ import annotations

from typing import Mapping

# Canonical event name -> kaomoji. Keep this list small; it is the whole
# public contract of this module, so document additions here.
#   searching     - looking something up (web search, memory recall, ...)
#   error         - a tool call or step failed
#   test_passed   - a test/check came back green
#   stuck         - the loop-guard or stuck-watcher tripped (same call/error repeating)
#   waiting       - idle, nothing to show yet (generic fallback wait state)
#   done          - the turn/task finished successfully
MOOD_MAP: dict[str, str] = {
    "searching": "O_o",
    "error": "=_=",
    "test_passed": "^_^",
    "stuck": "=_=",
    "waiting": "(*_*)",
    "done": "*\\0/*",
}


def kaomoji_for(event: str, extra: Mapping[str, str] | None = None) -> str:
    """The kaomoji for `event`, or "" if the event isn't one we know about.

    Never raises: an unknown or malformed event name just falls back to "".
    `extra` (optional) is checked first so a caller-supplied/config-driven
    mapping can add or override entries without editing MOOD_MAP itself.
    """
    try:
        key = str(event or "").strip().lower()
    except Exception:  # noqa: BLE001 - a weird event value must never crash a status line
        return ""
    if not key:
        return ""
    if extra:
        try:
            if key in extra:
                return str(extra[key])
        except Exception:  # noqa: BLE001 - a broken `extra` mapping is not fatal
            pass
    return MOOD_MAP.get(key, "")


def mood_enabled(config: Mapping | None) -> bool:
    """Whether the mood side-channel is switched on (`ui.mood_emotes`,
    default on). Missing/malformed config just means "on" - this is a
    cosmetic feature, never worth failing a chat turn over."""
    try:
        return bool((config or {}).get("ui", {}).get("mood_emotes", True))
    except Exception:  # noqa: BLE001
        return True


def mood_for(event: str, config: Mapping | None = None) -> str:
    """kaomoji_for(event), but "" when the config switch is off. This is the
    one-stop call most insertion points want."""
    if not mood_enabled(config):
        return ""
    extra = None
    try:
        extra = (config or {}).get("ui", {}).get("mood_map")
    except Exception:  # noqa: BLE001
        extra = None
    return kaomoji_for(event, extra if isinstance(extra, Mapping) else None)
