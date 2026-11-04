"""A local model's way of writing an enum value is mapped onto the allowed one when unambiguous ("30d" -> month)."""

from __future__ import annotations

from agent.toolargs import _enum_alias

FRESH = ["any", "day", "week", "month", "year"]


def test_durations_and_spellings_map_to_the_allowed_value():
    assert _enum_alias("30d", FRESH) == "month"
    assert _enum_alias("7d", FRESH) == "week"
    assert _enum_alias("24h", FRESH) == "day"
    assert _enum_alias("1y", FRESH) == "year"
    assert _enum_alias("Month", FRESH) == "month"
    assert _enum_alias("past_month", FRESH) == "month"


def test_anything_unclear_is_left_for_the_normal_refusal():
    assert _enum_alias("soon", FRESH) == "soon"
    assert _enum_alias("3000d", FRESH) == "3000d"
    assert _enum_alias(5, FRESH) == 5
