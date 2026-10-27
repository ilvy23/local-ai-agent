"""Extracts embedded cover art from an audio file's own metadata (ID3 for
MP3, FLAC picture blocks, MP4/M4A `covr` atoms) so the chat can show a
"now playing"-style card instead of a bare player. Not every audio file
has embedded art - this returns None rather than fabricating anything.

Results are cached by (source path, size, mtime), same pattern as
agent/tools/remux.py, so the same file's art is never re-extracted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR

ALBUM_ART_CACHE_DIR = DEFAULT_DATA_DIR / "album-art-cache"


def _cache_path(source: Path, stat: Any, ext: str) -> Path:
    key = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return ALBUM_ART_CACHE_DIR / f"{digest}.{ext}"


def _mime_to_ext(mime: str) -> str:
    return "png" if "png" in mime else "jpg"


def _extract_raw(source: Path) -> tuple[bytes, str] | None:
    """Returns (image_bytes, mime_type) from whichever tag format the file
    actually uses, or None if there's no embedded picture / mutagen can't
    open it."""
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3
    from mutagen.mp4 import MP4

    try:
        audio = mutagen.File(source)
    except Exception:  # noqa: BLE001 - any parse failure just means "no art"
        return None
    if audio is None:
        return None

    if isinstance(audio, FLAC):
        if audio.pictures:
            pic = audio.pictures[0]
            return pic.data, pic.mime or "image/jpeg"
        return None

    if isinstance(audio, MP4):
        covers = audio.tags.get("covr") if audio.tags else None
        if covers:
            cover = covers[0]
            mime = "image/png" if cover.imageformat == cover.FORMAT_PNG else "image/jpeg"
            return bytes(cover), mime
        return None

    # MP3/ID3 and anything else ID3-taggable.
    try:
        id3 = audio.tags if isinstance(audio.tags, ID3) else ID3(source)
    except Exception:  # noqa: BLE001
        id3 = None
    if id3:
        apics = id3.getall("APIC")
        if apics:
            return apics[0].data, apics[0].mime or "image/jpeg"
    return None


def read_track_tags(source: Path) -> dict[str, str]:
    """Best-effort {title, artist, album} from the file's own tags via
    mutagen's format-agnostic "easy" interface - any field the file doesn't
    have is simply absent from the returned dict, never guessed."""
    import mutagen

    try:
        audio = mutagen.File(source, easy=True)
    except Exception:  # noqa: BLE001
        return {}
    if audio is None or not audio.tags:
        return {}
    tags: dict[str, str] = {}
    for key in ("title", "artist", "album"):
        values = audio.tags.get(key)
        if values:
            tags[key] = str(values[0])
    return tags


def extract_album_art_cached(source: Path) -> Path | None:
    """Returns the cached path to `source`'s embedded cover art, extracting
    it first if needed. None if the file has no embedded art or can't be
    read."""
    stat = source.stat()
    # Check both possible extensions before extracting, since we don't know
    # the mime type until we've actually parsed the file once.
    for ext in ("jpg", "png"):
        existing = _cache_path(source, stat, ext)
        if existing.exists():
            return existing

    result = _extract_raw(source)
    if result is None:
        return None
    data, mime = result
    if not data:
        return None

    ALBUM_ART_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(source, stat, _mime_to_ext(mime))
    cached.write_bytes(data)
    return cached
