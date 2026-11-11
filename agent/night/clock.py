"""Wall-clock planning: `--until 07:30` and `--start-at 01:00` instead of counting hours. Pure functions on datetimes so they can be tested."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

_HM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def next_occurrence(hhmm: str, now: datetime) -> datetime:
    """The next time the wall clock shows HH:MM (today if it is still ahead, else tomorrow). Raises ValueError for a malformed value."""
    m = _HM.match(hhmm or "")
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"not a time of day: {hhmm!r} (use HH:MM)")
    t = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def hours_until(hhmm: str, now: datetime) -> float:
    return (next_occurrence(hhmm, now) - now).total_seconds() / 3600


def seconds_to_start(hhmm: str | None, now: datetime) -> float:
    return 0.0 if not hhmm else (next_occurrence(hhmm, now) - now).total_seconds()
