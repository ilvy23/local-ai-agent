"""Phase 5 emotes side-channel: event -> kaomoji mapping (agent/mood.py), the
on/off switch, and its two additive insertion points - the terminal's
narration prints (agent/tui.py's `_mood` helper) and the web status event
(agent/web/routes/chat.py's `_add_mood`). No model call anywhere here."""

from __future__ import annotations

import copy

import pytest

from agent import mood
from agent.config import DEFAULT_CONFIG

# The whole canonical event set this module promises to handle (see module docstring).
_CANONICAL_EVENTS = ("searching", "error", "test_passed", "stuck", "waiting", "done")


def test_mapping_is_total_over_the_canonical_event_set():
    for event in _CANONICAL_EVENTS:
        assert mood.kaomoji_for(event), f"{event!r} should map to a non-empty kaomoji"
    # and the public MOOD_MAP has exactly these keys documented - nothing silently extra
    assert set(mood.MOOD_MAP) == set(_CANONICAL_EVENTS)


@pytest.mark.parametrize("bad", [None, "", "   ", "NOT-A-REAL-EVENT", 123, object(), ["list"]])
def test_unknown_or_malformed_event_falls_back_to_empty_never_raises(bad):
    assert mood.kaomoji_for(bad) == ""
    assert mood.mood_for(bad, {}) == ""


def test_case_and_whitespace_insensitive():
    assert mood.kaomoji_for("  Searching  ") == mood.kaomoji_for("searching")


def test_switch_off_suppresses_every_event():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ui"]["mood_emotes"] = False
    for event in _CANONICAL_EVENTS:
        assert mood.mood_for(event, config) == ""


def test_switch_defaults_on_even_with_missing_config():
    assert mood.mood_enabled(None) is True
    assert mood.mood_enabled({}) is True
    for event in _CANONICAL_EVENTS:
        assert mood.mood_for(event, None) == mood.kaomoji_for(event)


def test_extra_mapping_can_add_or_override_without_touching_mood_map():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ui"]["mood_map"] = {"compacting": "o.O", "searching": "!!!"}
    assert mood.mood_for("compacting", config) == "o.O"       # new event, not in MOOD_MAP at all
    assert mood.mood_for("searching", config) == "!!!"        # override of an existing canonical entry
    assert mood.MOOD_MAP["searching"] == "O_o"                 # the module's own table is untouched

    config["ui"]["mood_emotes"] = False
    assert mood.mood_for("compacting", config) == ""          # the switch still wins over `extra`


def test_default_config_ships_the_switch_on_and_an_empty_extra_map():
    assert DEFAULT_CONFIG["ui"]["mood_emotes"] is True
    assert DEFAULT_CONFIG["ui"]["mood_map"] == {}


# ------------------------------------------------------------------ terminal side (agent/tui.py)


def test_tui_mood_helper_prefixes_with_a_trailing_space(monkeypatch):
    from agent import tui

    monkeypatch.setattr("agent.config.load_config", lambda: copy.deepcopy(DEFAULT_CONFIG))
    assert tui._mood("error") == "=_= "
    assert tui._mood("not-a-real-event") == ""


def test_tui_mood_helper_respects_the_switch(monkeypatch):
    from agent import tui

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ui"]["mood_emotes"] = False
    monkeypatch.setattr("agent.config.load_config", lambda: config)
    assert tui._mood("error") == ""


def test_tui_mood_helper_never_raises_when_config_loading_breaks(monkeypatch):
    from agent import tui

    def boom():
        raise RuntimeError("no config.yaml here")

    monkeypatch.setattr("agent.config.load_config", boom)
    assert tui._mood("searching") == ""  # falls back to nothing, never bubbles up


# ------------------------------------------------------------------ web side (agent/web/routes/chat.py)


def test_add_mood_sets_additive_field_on_status_events():
    pytest.importorskip("fastapi")
    from agent.web.routes.chat import _add_mood

    config = copy.deepcopy(DEFAULT_CONFIG)
    event = {"type": "status", "state": "searching", "label": "searching", "detail": ""}
    _add_mood(event, config)
    assert event["mood"] == "O_o"
    assert event["state"] == "searching" and event["label"] == "searching"  # nothing existing changed


def test_add_mood_maps_verify_and_stuck_and_done_events():
    pytest.importorskip("fastapi")
    from agent.web.routes.chat import _add_mood

    config = copy.deepcopy(DEFAULT_CONFIG)
    passed = {"type": "verify", "state": "passed"}
    _add_mood(passed, config)
    assert passed["mood"] == "^_^"

    failed = {"type": "verify", "state": "failed"}
    _add_mood(failed, config)
    assert failed["mood"] == "=_="

    stuck = {"type": "stuck", "kind": "lookup"}
    _add_mood(stuck, config)
    assert stuck["mood"] == "=_="

    done = {"type": "done", "reply": "ok"}
    _add_mood(done, config)
    assert done["mood"] == "*\\0/*"


def test_add_mood_leaves_unmapped_or_unknown_events_alone():
    pytest.importorskip("fastapi")
    from agent.web.routes.chat import _add_mood

    config = copy.deepcopy(DEFAULT_CONFIG)
    event = {"type": "token", "text": "hello"}
    _add_mood(event, config)
    assert "mood" not in event


def test_add_mood_suppressed_when_switch_is_off():
    pytest.importorskip("fastapi")
    from agent.web.routes.chat import _add_mood

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ui"]["mood_emotes"] = False
    event = {"type": "status", "state": "searching"}
    _add_mood(event, config)
    assert "mood" not in event


def test_add_mood_never_raises_with_missing_or_odd_config():
    pytest.importorskip("fastapi")
    from agent.web.routes.chat import _add_mood

    event = {"type": "status", "state": "searching"}
    _add_mood(event, None)  # missing config -> no mood, no crash
    assert "mood" not in event
    event2 = {"type": "status", "state": "searching"}
    _add_mood(event2, "not-a-dict")  # odd config -> must not raise either way


def test_settings_exposes_the_switch():
    from agent.settings import EDITABLE

    assert EDITABLE["mood_emotes"]["path"] == "ui.mood_emotes"
    assert EDITABLE["mood_emotes"]["kind"] == "bool"
