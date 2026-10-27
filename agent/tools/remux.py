"""On-the-fly conversion of a video into something iOS Safari can actually
play - and picking which audio track ships in it.

Four independent things this handles, each producing a single clean .mp4:

1. Container: Safari refuses Matroska/AVI/etc. outright regardless of the
   codecs inside. Fixed by a REMUX (`-c copy` into .mp4) when the streams
   are otherwise fine - no re-encoding, no quality loss, bounded by disk
   I/O not codec work.
2. 10-bit video: Safari's decoder support for 10-bit HEVC (Main10) is
   inconsistent across devices - the observed failure mode is exactly
   "audio plays, scrubber works, picture is blank," not an outright
   rejection. Fixed by transcoding the video (only) down to 8-bit H.264,
   using NVENC hardware encoding when available since that's a real
   re-encode, not a free container swap.
3. Incompatible audio (AC3/DTS/etc., common in movie rips): Safari can't
   decode it at all. Fixed by transcoding just the audio track to AAC.
4. Audio track choice: a plain <video> only ever plays ONE audio track -
   there is no in-player track switcher, regardless of how many the source
   container has. "Give me the Japanese audio" means picking that stream
   at mux time, not something fixable after the fact.

Each is handled independently - a file might need all, some, or none.
Results are cached by (source path, size, mtime, audio track selection) so
the same file+track combination is never processed twice.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR

REMUX_CACHE_DIR = DEFAULT_DATA_DIR / "remux-cache"

_SAFARI_NATIVE_SUFFIXES = {".mp4", ".m4v", ".mov"}
_SAFE_VIDEO_CODECS = {"h264", "hevc"}
_SAFE_AUDIO_CODECS = {"aac", "mp3"}

_PROBE_TIMEOUT_S = 20
_REMUX_TIMEOUT_S = 1800


_FINGERPRINT_SAMPLE_BYTES = 1_048_576  # 1 MiB


def _content_fingerprint(source: Path, stat: Any) -> str:
    """A fast, path-independent fingerprint - the same video existing at
    two different paths (perfectly normal: a media library synced between
    a local folder and an external drive, exactly like this system's
    Downloads copy and HDD copy of the same episode) used to hash by PATH,
    so identical content at a different path always missed the cache and
    re-transcoded from scratch, even though the expensive work had already
    been done once. Full-file hashing would fix this reliably but costs
    real seconds on every request for a multi-gigabyte video, just to
    check the cache. Sampling a few chunks (start/middle/end) plus the
    total size is near-instant regardless of file size, and combined with
    size, two genuinely different videos colliding by chance is
    astronomically unlikely.
    """
    size = stat.st_size
    n = _FINGERPRINT_SAMPLE_BYTES
    hasher = hashlib.sha256()
    hasher.update(str(size).encode())
    with source.open("rb") as f:
        hasher.update(f.read(n))
        if size > n:
            f.seek(max(0, size // 2 - n // 2))
            hasher.update(f.read(n))
        if size > n * 2:
            f.seek(max(0, size - n))
            hasher.update(f.read(n))
    return hasher.hexdigest()


def _cache_path(source: Path, stat: Any, audio_track: str | None) -> Path:
    try:
        fingerprint = _content_fingerprint(source, stat)
    except OSError:
        # Fall back to the old path+mtime key rather than fail outright -
        # a cache miss on a genuine read error is a minor inefficiency,
        # not worth losing the whole remux over.
        fingerprint = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    key = f"{fingerprint}:{audio_track or ''}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return REMUX_CACHE_DIR / f"{digest}.mp4"


def _probe_all_streams(source: Path) -> list[dict[str, Any]] | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_entries",
                "stream=index,codec_type,codec_name,profile,pix_fmt,bits_per_raw_sample:stream_tags=language,title",
                str(source),
            ],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False,
        )
        return json.loads(proc.stdout).get("streams", [])
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None


# Stream language tags are ISO 639-2 codes ("jpn", "ger"), but a user (and
# the model relaying their words) says the actual name ("Japanese",
# "German") - without this, "japanese" is not a substring of "jpn" or vice
# versa, and matching would silently never work for the common case.
_LANGUAGE_ALIASES = {
    "japanese": "jpn", "german": "ger", "deutsch": "ger", "english": "eng",
    "hindi": "hin", "french": "fre", "spanish": "spa", "italian": "ita",
    "korean": "kor", "chinese": "chi", "mandarin": "chi", "russian": "rus",
    "portuguese": "por", "arabic": "ara", "dutch": "dut", "polish": "pol",
    "turkish": "tur", "swedish": "swe", "norwegian": "nor", "danish": "dan",
    "finnish": "fin", "greek": "gre", "hebrew": "heb", "thai": "tha",
    "vietnamese": "vie", "indonesian": "ind",
}


def _normalize_track_needle(audio_track: str) -> str:
    needle = audio_track.strip().lower()
    return _LANGUAGE_ALIASES.get(needle, needle)


def _stream_matches(stream: dict[str, Any], audio_track: str) -> bool:
    if audio_track.strip().isdigit():
        return stream["index"] == int(audio_track.strip())
    needle = _normalize_track_needle(audio_track)
    tags = stream.get("tags", {})
    return needle in tags.get("language", "").lower() or needle in tags.get("title", "").lower()


def list_audio_tracks(source: Path) -> list[dict[str, Any]]:
    """Every audio stream's (index, language, title, codec) - for telling the
    model/user what's actually available to pick from."""
    streams = _probe_all_streams(source) or []
    return [
        {
            "index": s["index"],
            "language": s.get("tags", {}).get("language", ""),
            "title": s.get("tags", {}).get("title", ""),
            "codec": s.get("codec_name", ""),
        }
        for s in streams
        if s.get("codec_type") == "audio"
    ]


def audio_track_exists(source: Path, audio_track: str) -> bool:
    """True if `audio_track` (a language name/code, title substring, or
    absolute stream index) actually matches one of `source`'s audio
    streams - used to tell the model honestly whether the requested track
    was found, rather than silently falling back to the default and saying
    nothing."""
    audio_streams = [s for s in (_probe_all_streams(source) or []) if s.get("codec_type") == "audio"]
    return any(_stream_matches(s, audio_track) for s in audio_streams)


def _select_audio_stream(
    audio_streams: list[dict[str, Any]], audio_track: str | None
) -> dict[str, Any] | None:
    """Picks the requested audio stream by language name/code (substring,
    case-insensitive - "japanese" or "jpn" both match a "jpn" tag), by
    title, or by a plain numeric absolute stream index. Falls back to the
    first audio stream if `audio_track` is None or nothing matches."""
    if not audio_streams:
        return None
    if audio_track:
        match = next((s for s in audio_streams if _stream_matches(s, audio_track)), None)
        if match:
            return match
    return audio_streams[0]


def _is_he_aac(audio: dict[str, Any]) -> bool:
    """True for HE-AAC/HE-AACv2 (SBR/PS) - technically still "aac", but iOS
    Safari's decoder frequently can't play it at all inside an MP4 <video>
    (the failure looks like a totally broken/unplayable file, not a quality
    issue), unlike plain AAC-LC which is fine. Needs a real transcode down
    to AAC-LC, not a stream copy, even though codec_name alone says "aac"."""
    return "he-aac" in (audio.get("profile") or "").lower()


def _is_10bit(video: dict[str, Any]) -> bool:
    try:
        if int(video.get("bits_per_raw_sample") or 8) > 8:
            return True
    except (TypeError, ValueError):
        pass
    return "10" in (video.get("pix_fmt") or "")


@functools.lru_cache(maxsize=1)
def _nvenc_available() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return "h264_nvenc" in proc.stdout


def _run_ffmpeg_with_progress(
    args: list[str],
    total_duration_s: float | None,
    progress_cb: Callable[[float], None] | None,
    timeout_s: float,
) -> bool:
    """Runs ffmpeg to completion. Returns True on success (clean exit),
    False on any failure/timeout.

    This used to also read `-progress pipe:1` output line-by-line to report
    live percent-complete via progress_cb. Reverted: in real use, that
    pipe-reading loop (running inside asyncio.to_thread, reporting back
    through the same thread-safe console bridge other tools use) caused
    the ENTIRE web app to become totally unresponsive for the whole
    duration of a transcode - not just slow, completely dead, confirmed by
    killing the ffmpeg process and watching the app respond again
    instantly. The exact mechanism wasn't worth chasing further under
    time pressure when a plain, already-proven-reliable blocking
    subprocess.run() (used for months before today with zero such
    issues) does the same job safely. `total_duration_s` and
    `progress_cb` are kept as parameters so callers don't need changing,
    but are currently unused - a live-progress feature can come back once
    it's built without this failure mode.
    """
    del total_duration_s, progress_cb  # currently unused - see docstring
    try:
        subprocess.run(args, capture_output=True, timeout=timeout_s, check=True)
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def remux_for_safari(
    source: Path,
    audio_track: str | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> Path | None:
    """If `source` needs container/codec fixes to play in Safari, or a
    specific audio track was requested, produce a compatible .mp4 (cached)
    and return its path. Returns None if nothing needs doing, ffmpeg/
    ffprobe are unavailable, or processing fails - callers should fall back
    to serving the original file unchanged in that case.

    `audio_track` is a language substring ("japanese", "jpn"), a title
    substring, or an absolute ffprobe stream index as a string - resolved
    via list_audio_tracks/_select_audio_stream, never trusted as a raw
    ffmpeg map argument.
    """
    if shutil.which("ffmpeg") is None:
        return None

    streams = _probe_all_streams(source)
    if not streams:
        return None
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    audio = _select_audio_stream(audio_streams, audio_track)
    if video is None or audio is None:
        return None

    # Keyed on the RESOLVED stream index, not the raw audio_track string -
    # "Japanese", "japanese", and "jpn" all resolve to the same stream, and
    # should all hit the same cache entry instead of each re-transcoding.
    stat = source.stat()
    cached = _cache_path(source, stat, str(audio["index"]))
    if cached.exists():
        return cached
    video_codec, audio_codec = video.get("codec_name"), audio.get("codec_name")
    if video_codec not in _SAFE_VIDEO_CODECS:
        return None  # VP9/AV1/etc: broader re-encode, out of scope for now

    video_needs_transcode = _is_10bit(video)
    audio_needs_transcode = audio_codec not in _SAFE_AUDIO_CODECS or _is_he_aac(audio)
    container_needs_remux = source.suffix.lower() not in _SAFARI_NATIVE_SUFFIXES
    # A non-default track pick always needs a real mux pass, even if the
    # container/codecs were otherwise already fine - there's no way to
    # "select a track" after the fact on a single-audio-stream <video>.
    track_pick_needed = audio_track is not None and audio is not audio_streams[0]

    if not (video_needs_transcode or audio_needs_transcode or container_needs_remux or track_pick_needed):
        return None  # already fine as-is

    if video_needs_transcode:
        video_args = (
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-pix_fmt", "yuv420p"]
            if _nvenc_available()
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        )
    else:
        video_args = ["-c:v", "copy"]
    audio_args = ["-c:a", "aac", "-b:a", "192k"] if audio_needs_transcode else ["-c:a", "copy"]

    REMUX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cached.with_suffix(".mp4.part")
    ok = _run_ffmpeg_with_progress(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-map", f"0:{video['index']}", "-map", f"0:{audio['index']}",
            *video_args, *audio_args,
            "-movflags", "+faststart",
            # ffmpeg picks its muxer from the output filename's final
            # extension - the ".part" suffix (for atomic rename on
            # success) would otherwise make it refuse to guess a
            # container at all. Force it explicitly instead.
            "-f", "mp4",
            str(tmp_path),
        ],
        None, progress_cb, _REMUX_TIMEOUT_S,
    )
    if not ok:
        tmp_path.unlink(missing_ok=True)
        return None
    tmp_path.rename(cached)
    return cached
