"""get_media_info: a clean, single answer for "what is this file" (duration,
resolution, codecs, subtitle tracks, etc.) via ffprobe - so the model has a
real tool for basic media questions instead of reaching for the shell tool
and trying to parse raw ffprobe/mediainfo output itself, or fabricating a
description. Read-only, SAFE risk.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent.tools.registry import Tool

_PROBE_TIMEOUT_S = 20


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _human_duration(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def get_media_info(path: str = "", **_kwargs: Any) -> str:
    if not path:
        return "Error: no path given."
    resolved = _resolve(path)
    if not resolved.exists():
        return f"Error: no such file: {resolved}"
    if not resolved.is_file():
        return f"Error: {resolved} is a directory, not a file."
    if shutil.which("ffprobe") is None:
        return "Error: ffprobe is not installed, can't inspect media files."

    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(resolved),
            ],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=True,
        )
        probe = json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        return f"Error reading {resolved.name}: {exc}"

    lines = [resolved.name, f"Size: {_human_size(resolved.stat().st_size)}"]
    fmt = probe.get("format", {})
    if fmt.get("duration"):
        lines.append(f"Duration: {_human_duration(float(fmt['duration']))}")

    for stream in probe.get("streams", []):
        kind = stream.get("codec_type")
        idx = stream.get("index")
        codec = stream.get("codec_name", "?")
        if kind == "video" and stream.get("disposition", {}).get("attached_pic"):
            # Audio containers (FLAC/MP3/etc.) embed cover art as a "video"
            # stream with this flag set - it's a still image, not footage.
            w, h = stream.get("width"), stream.get("height")
            lines.append(f"Cover art: embedded image, {w}x{h}")
        elif kind == "video":
            w, h = stream.get("width"), stream.get("height")
            fps = stream.get("r_frame_rate", "")
            bit_depth = stream.get("bits_per_raw_sample")
            depth_note = f", {bit_depth}-bit" if bit_depth else ""
            lines.append(f"Video #{idx}: {codec}{depth_note}, {w}x{h}, {fps} fps")
        elif kind == "audio":
            lang = stream.get("tags", {}).get("language", "")
            lang_note = f" ({lang})" if lang else ""
            channels = stream.get("channels")
            rate = stream.get("sample_rate")
            lines.append(f"Audio #{idx}{lang_note}: {codec}, {channels}ch, {rate} Hz")
        elif kind == "subtitle":
            lang = stream.get("tags", {}).get("language", "")
            lang_note = f" ({lang})" if lang else ""
            lines.append(f"Subtitle #{idx}{lang_note}: {codec}")

    return "\n".join(lines)


GET_MEDIA_INFO_TOOL = Tool(
    name="get_media_info",
    description=(
        "Inspect a local audio/video file and get a clean summary: duration, "
        "resolution, video/audio codecs, and any subtitle tracks embedded in "
        "the container (with their language if tagged). Use this to answer "
        "questions about a media file, or to check what's available before "
        "deciding what to do with it - never try to parse raw ffprobe/shell "
        "output yourself for this."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or ~-relative path to the media file.",
            },
        },
        "required": ["path"],
    },
    handler=get_media_info,
    risk="safe",
)
