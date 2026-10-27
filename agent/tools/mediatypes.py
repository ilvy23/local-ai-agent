"""Content types for shared files. One explicit table (the system mime database differs per machine and misses
.opus/.flac/.m4a/.mkv on some), plus the short list of types that may be shown inline. Everything else - html, svg,
scripts, unknown - is only ever served as a download, so a shared file can never run in the app's origin."""

from __future__ import annotations

import mimetypes
from pathlib import Path

MEDIA_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
    ".ogv": "video/ogg", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".opus": "audio/ogg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
    ".avif": "image/avif", ".bmp": "image/bmp", ".vtt": "text/vtt",
}
INLINE_SAFE = frozenset(MEDIA_TYPES.values())        # no svg, no html, no text/plain


def content_type(name: str) -> str:
    ext = Path(name).suffix.lower()
    return MEDIA_TYPES.get(ext) or mimetypes.guess_type(name)[0] or "application/octet-stream"


def media_kind(name: str) -> str | None:
    """'video' | 'audio' | 'image' for the inline-safe media types, else None."""
    mime = MEDIA_TYPES.get(Path(name).suffix.lower())
    return mime.split("/")[0] if mime and mime != "text/vtt" else None
