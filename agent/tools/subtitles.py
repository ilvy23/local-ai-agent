"""Prepares an external subtitle file for use as a real, user-toggleable
HTML5 <track> on a streamed video - not burned in (which would be a
permanent, unremovable transcode), but a proper subtitle track Safari's
native player can show/hide on its own.

Safari's <video> only accepts WebVTT tracks, not .srt/.ass directly, so a
non-.vtt file is converted first (text-based, no video re-encoding - fast).
Same caching pattern as remux.py/album_art.py: keyed by (path, size, mtime),
so the same subtitle file is never converted twice.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR

SUBTITLE_CACHE_DIR = DEFAULT_DATA_DIR / "subtitle-cache"

_CONVERTIBLE_SUFFIXES = {".srt", ".ass", ".ssa", ".sub"}
_CONVERT_TIMEOUT_S = 30


def _cache_path(source: Path, stat: Any) -> Path:
    key = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return SUBTITLE_CACHE_DIR / f"{digest}.vtt"


def prepare_subtitle_vtt(source: Path) -> Path | None:
    """Returns a WebVTT file usable as a <track> src for `source`, converting
    it first if needed. None if the format isn't supported or conversion
    fails - callers should just skip attaching a subtitle track in that case,
    not fail the whole file send over it.
    """
    if source.suffix.lower() == ".vtt":
        return source
    if source.suffix.lower() not in _CONVERTIBLE_SUFFIXES:
        return None
    if shutil.which("ffmpeg") is None:
        return None

    stat = source.stat()
    cached = _cache_path(source, stat)
    if cached.exists():
        return cached

    SUBTITLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cached.with_suffix(".vtt.part")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(source), "-f", "webvtt", str(tmp_path)],
            capture_output=True, timeout=_CONVERT_TIMEOUT_S, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        tmp_path.unlink(missing_ok=True)
        return None
    tmp_path.rename(cached)
    return cached
