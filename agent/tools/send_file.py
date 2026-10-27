"""send_file: hands the user a real file (a video, a document, anything on
disk) as a download link or an inline player in chat, instead of reading its
contents into the model's context - which would be pointless for a movie and
would blow the context window for anything but a small text file.

The player/link itself is delivered via a direct `file_ready` SSE event
(console.emit_event), not by asking the model to relay a snippet verbatim -
a small local model repeatedly failed to do that faithfully (dropping the
tag, saying something else entirely). The tool decides everything about
HOW the file gets shown; the model's only job is deciding WHICH file and
telling the user something short afterward - it can't get the mechanics
wrong because it never has to reproduce them.

Bound to one (store, session_id, console) at registry-build time, same
shape as _make_take_note_tool - a download token is meaningless without
knowing which session it belongs to, and emitting the ready-event needs the
same web-only console bridge make_web_tools uses for narration.
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR
from agent.memory.store import Store
from agent.textio import human_size
from agent.tools.album_art import extract_album_art_cached, read_track_tags
from agent.tools.registry import Tool
from agent.tools.remux import audio_track_exists, list_audio_tracks, remux_for_safari
from agent.tools.subtitles import prepare_subtitle_vtt

FOLDER_ZIP_CACHE_DIR = DEFAULT_DATA_DIR / "folder-zip-cache"


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


_ZIP_MAX_BYTES = 4 * 1024**3  # 4 GB
_ZIP_MAX_ENTRIES = 50_000


def _zip_directory_cached(directory: Path) -> Path | str | None:
    """Zips `directory` into a single file for sending, cached by directory
    path + total content size + latest mtime among its files (a cheap,
    good-enough "has anything in here changed" signal - a full recursive
    hash would be more precise but is real work for a big folder, and this
    tool is about sending a folder someone just pointed at, not detecting
    subtle in-place edits to its contents).

    Returns a Path on success, None on an OS-level failure, or a str error
    message if the folder is too big to reasonably zip and send - without
    this, pointing it at e.g. a home directory would grind for a long time
    and fill the disk with a multi-GB archive, and the approval prompt the
    human sees beforehand shows only a bare path with no hint of that."""
    try:
        entries = list(directory.rglob("*"))
        files = [f for f in entries if f.is_file()]
        total_size = sum(f.stat().st_size for f in files)
        latest_mtime = max((f.stat().st_mtime_ns for f in files), default=0)
    except OSError:
        return None
    if total_size > _ZIP_MAX_BYTES or len(files) > _ZIP_MAX_ENTRIES:
        return (
            f"folder too large to zip and send ({len(files)} files, "
            f"{human_size(total_size)} - limit is {_ZIP_MAX_ENTRIES} files / "
            f"{human_size(_ZIP_MAX_BYTES)})"
        )
    key = f"{directory}:{total_size}:{latest_mtime}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    cached = FOLDER_ZIP_CACHE_DIR / f"{digest}.zip"
    if cached.exists():
        return cached
    FOLDER_ZIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # A unique per-call temp name, not just the content digest: two
    # concurrent sends of the same unchanged folder would otherwise both
    # target the identical .tmp.zip, and whichever renamed second would hit
    # a FileNotFoundError racing the first's already-completed rename.
    tmp_base = FOLDER_ZIP_CACHE_DIR / f"{digest}.{uuid.uuid4().hex}.tmp"
    try:
        archive = shutil.make_archive(str(tmp_base), "zip", root_dir=str(directory))
        Path(archive).rename(cached)
    except (OSError, shutil.Error):
        return None
    return cached


def make_send_file_tool(store: Store, session_id: int, console: Any) -> Tool:
    def handler(
        path: str = "", mode: str = "auto",
        subtitle_path: str = "", subtitle_label: str = "Subtitles",
        audio_track: str = "",
        **_kwargs: Any,
    ) -> str:
        if not path:
            return "Error: no path given."
        resolved = _resolve(path)
        if not resolved.exists():
            return f"Error: no such file: {resolved}"
        folder_display_name: str | None = None
        if resolved.is_dir():
            # A whole folder can't be "sent" as-is over HTTP - zip it into
            # one real file and send THAT, same delivery mechanism as
            # everything else below. Errors (empty dir, permission issues)
            # surface as a plain string, same convention as the rest of
            # this handler. The zip lives in a hash-named cache file, so
            # the folder's own name is captured here to use as the
            # user-facing display name instead of that hash.
            folder_display_name = f"{resolved.name}.zip"
            zipped = _zip_directory_cached(resolved)
            if isinstance(zipped, str):
                return f"Error: {zipped}."
            if zipped is None:
                return f"Error: couldn't zip {resolved} (is it empty or unreadable?)."
            resolved = zipped
        elif not resolved.is_file():
            return f"Error: {resolved} is neither a file nor a directory."
        size = resolved.stat().st_size

        media_type, _ = mimetypes.guess_type(resolved.name)
        media_type = media_type or "application/octet-stream"
        is_video = media_type.startswith("video/")
        is_audio = media_type.startswith("audio/")
        is_image = media_type.startswith("image/")

        if mode not in ("auto", "download", "stream"):
            mode = "auto"
        wants_stream = mode == "stream" or (mode == "auto" and (is_video or is_audio or is_image))
        disposition = "inline" if wants_stream else "attachment"

        display_name = folder_display_name or resolved.name
        serve_path = resolved
        audio_track_note = ""
        if wants_stream and is_video:
            # A plain <video> only ever plays ONE audio track - there is no
            # in-player track switcher, no matter how many tracks the
            # source has. "Give me the Japanese audio" means picking that
            # stream at mux time, which is why this always goes through
            # remux_for_safari when a track is requested, even if the
            # container/codecs would otherwise already be fine.
            if audio_track and not audio_track_exists(resolved, audio_track):
                available = ", ".join(
                    t["language"] or t["title"] or f"track {t['index']}"
                    for t in list_audio_tracks(resolved)
                ) or "none found"
                audio_track_note = (
                    f" (requested audio track '{audio_track}' not found - "
                    f"available: {available}; used the default track instead)"
                )
            # Safari refuses Matroska/etc. containers outright and silently
            # drops the picture (audio + scrubber still work) on 10-bit HEVC
            # - remux_for_safari fixes both, transcoding only what actually
            # needs it. Falls back to the original file untouched if
            # nothing needs fixing, or fixing isn't possible.
            #
            # A real video re-encode (not just a container/audio swap) can
            # take several minutes for a long file - a movie, not a TV
            # episode - so this narrates that ONCE, up front, rather than
            # leaving a long wait with zero explanation. A previous version
            # of this tried reporting LIVE percent-complete during the
            # transcode instead - that repeatedly called back into the
            # console bridge from inside ffmpeg's progress-reading loop,
            # and in real use that made the entire web app completely
            # unresponsive for the whole transcode (confirmed: killing the
            # ffmpeg process instantly restored it). Reverted in favor of
            # this simpler, proven-safe single notification.
            console.print("Transcoding video - this can take a few minutes for a longer file…")
            remuxed = remux_for_safari(resolved, audio_track=audio_track or None)
            if remuxed is not None:
                serve_path = remuxed
                display_name = f"{resolved.stem}.mp4"
                size = remuxed.stat().st_size

        token = store.add_file_share(
            session_id, str(serve_path), disposition=disposition, display_name=display_name
        )
        url = f"/api/files/{token}"

        if wants_stream and is_video:
            kind = "video"
        elif wants_stream and is_audio:
            kind = "audio"
        elif wants_stream and is_image:
            kind = "image"
        elif wants_stream:
            kind = "link"  # streaming asked/implied but not a playable media type
        else:
            kind = "download"

        extra: dict[str, Any] = {}
        if kind == "audio":
            tags = read_track_tags(resolved)
            extra.update(tags)
            art_path = extract_album_art_cached(resolved)
            if art_path is not None:
                art_token = store.add_file_share(
                    session_id, str(art_path), disposition="inline",
                    display_name=art_path.name,
                )
                extra["art_url"] = f"/api/files/{art_token}"

        subtitle_note = ""
        if kind == "video" and subtitle_path:
            sub_resolved = _resolve(subtitle_path)
            if not sub_resolved.exists():
                subtitle_note = f" (couldn't attach subtitles: no such file {sub_resolved})"
            else:
                vtt_path = prepare_subtitle_vtt(sub_resolved)
                if vtt_path is None:
                    subtitle_note = (
                        f" (couldn't attach subtitles: {sub_resolved.suffix} isn't a "
                        "supported subtitle format, or conversion failed)"
                    )
                else:
                    sub_token = store.add_file_share(
                        session_id, str(vtt_path), disposition="inline",
                        display_name=f"{subtitle_label}.vtt",
                    )
                    extra["subtitle_url"] = f"/api/files/{sub_token}"
                    extra["subtitle_label"] = subtitle_label

        console.emit_event(
            "file_ready", kind=kind, url=url, filename=display_name,
            size=human_size(size), **extra,
        )

        return (
            f"File sent: {display_name} ({human_size(size)}){subtitle_note}"
            f"{audio_track_note}. It is already showing in the chat as a "
            "player/link - do NOT include any URL, HTML tag, or markdown link "
            "in your reply, and do NOT claim the player itself has a menu to "
            "switch audio tracks or subtitles (it doesn't - a browser video "
            "player only ever plays the one audio track it was given). If "
            "the requested audio track wasn't found, say so plainly using "
            "the available list above - don't guess at what tracks exist. "
            "Just say something short and natural confirming what was sent."
        )

    return Tool(
        name="send_file",
        description=(
            "Send a file from the local filesystem to the user in chat - use "
            "this whenever they ask you to send/share/give them an actual file "
            "(a video, photo, document, archive, etc.), rather than reading or "
            "describing its contents. Not for text you're pulling into your own "
            "context (use read_file for that). Set mode='stream' when the user "
            "wants to watch/listen/preview it directly in the chat (best for "
            "video/audio - it becomes an inline player they can scrub through "
            "without downloading the whole file first). Set mode='download' "
            "when they want to save it to their device. Leave mode='auto' "
            "(the default) to pick automatically: video/audio stream inline, "
            "everything else downloads. If the user doesn't say which they "
            "want and the file is a video or audio file, ask them first "
            "(stream or download?) rather than guessing. The player/link "
            "appears in the chat automatically the moment this tool runs - "
            "never write your own link/tag for it, and never try to do any "
            "part of this with run_command/ffmpeg/shell commands - this tool "
            "handles the whole process (container/codec fixes, audio track "
            "selection, subtitle conversion) by itself. A browser video "
            "player has NO menu to switch audio tracks or subtitles after "
            "the fact - if the user wants different audio or subtitles on a "
            "video you already sent, just call this tool again on the SAME "
            "path with the new audio_track/subtitle_path - that's it, don't "
            "improvise anything else. For a streamed video, pass "
            "subtitle_path to attach an external subtitle file (.srt/.ass/"
            ".ssa/.vtt - converted to WebVTT automatically if needed) as a "
            "real toggleable subtitle track, NOT burned into the picture - "
            "set subtitle_label to say what it is (e.g. 'German') since you "
            "know that from context and the tool doesn't. ALWAYS pass "
            "audio_track (a language name like 'Japanese', or a track "
            "index) whenever the user names a specific audio language for "
            "a video - never leave it out and assume the source's default "
            "track happens to match. A multi-audio file's first/default "
            "track is picked essentially at random with respect to what "
            "the user actually wants (often not the one they asked for), "
            "and omitting audio_track silently ships that wrong track "
            "with no warning - it does NOT mean 'use whichever track the "
            "user mentioned,' it means 'I don't care which.' Images (PNG/"
            "JPEG/etc.) stream inline as a real preview by default, same as "
            "video/audio. If `path` is a DIRECTORY rather than a file, it's "
            "automatically zipped into one archive and sent as a download - "
            "call this the same way for 'send me that folder' requests, no "
            "need to zip it yourself with run_command first."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute or ~-relative path to the file to send - or a "
                        "directory, which gets zipped automatically."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "stream", "download"],
                    "description": (
                        "'stream' for an inline player (video/audio), "
                        "'download' for a save prompt, 'auto' to decide by file type."
                    ),
                },
                "subtitle_path": {
                    "type": "string",
                    "description": (
                        "Optional: path to a subtitle file (.srt/.ass/.ssa/.vtt) to "
                        "attach to a streamed video as a toggleable track. Only "
                        "used when mode results in streaming a video."
                    ),
                },
                "subtitle_label": {
                    "type": "string",
                    "description": "What to call the subtitle track, e.g. 'German'.",
                },
                "audio_track": {
                    "type": "string",
                    "description": (
                        "Optional: which audio track to include in a streamed video, "
                        "when the source has more than one (e.g. 'Japanese', 'eng', "
                        "or a numeric track index). Only used for streamed video."
                    ),
                },
            },
            "required": ["path"],
        },
        handler=handler,
        risk="caution",
    )
