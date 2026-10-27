"""send_album: shares a folder or a list of media files as ONE group - an audio playlist, a photo gallery or a video
list - delivered as a single `file_ready` event (kind "album") that the chat renders with a real player/lightbox.

Registered like send_file: always CAUTION (one approval for the whole group; no folder grant lifts it). The approval
text comes from plan_album(), so the person sees the file count and total size before saying yes. Only media files
of the album's kind are shared - everything else in a folder is skipped, never served. Files stream from disk with
Range support (routes/files.py); nothing is transcoded or zipped here."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from agent.memory.store import Store
from agent.textio import human_size
from agent.tools.album_art import extract_album_art_cached, read_track_tags
from agent.tools.mediatypes import content_type, media_kind
from agent.tools.registry import Tool

MAX_ITEMS = 200
_COVER_NAMES = ("cover", "folder", "front", "album")


def natural_key(p: Path) -> list[Any]:
    """'track2' sorts before 'track10'; case-insensitive."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p).lower())]


def plan_album(path: str = "", paths: list[str] | None = None) -> tuple[dict[str, Any] | None, str]:
    """-> ({name, kind, files, size, source}, "") or (None, error). Pure filesystem read, no side effects."""
    files: list[Path] = []
    if path:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return None, f"{root} is not a folder (pass `paths` for a list of files)."
        for f in root.rglob("*"):
            rel = f.relative_to(root)
            if any(part.startswith(".") for part in rel.parts) or not f.is_file():
                continue
            try:
                inside = f.resolve().is_relative_to(root)        # a symlink out of the folder is not part of it
            except OSError:
                inside = False
            if inside:
                files.append(f.resolve())
        files.sort(key=natural_key)
        name, source = root.name or str(root), str(root)
    elif paths:
        files = [Path(str(p)).expanduser().resolve() for p in paths]      # explicit list: the caller's order is kept
        files = [f for f in files if f.is_file()]
        name, source = (files[0].parent.name if files else "") or "Album", f"{len(paths)} listed files"
    else:
        return None, "give `path` (a folder) or `paths` (a list of files)."
    kinds = [media_kind(f.name) for f in files]
    counts = {k: kinds.count(k) for k in ("audio", "video", "image") if k in kinds}
    if not counts:
        return None, "no audio, video or image files found."
    kind = max(counts, key=lambda k: counts[k])
    files = [f for f, k in zip(files, kinds) if k == kind]
    if len(files) > MAX_ITEMS:
        return None, f"{len(files)} {kind} files is more than the {MAX_ITEMS} limit - share a subfolder or a shorter list."
    seen: set[Path] = set()
    files = [f for f in files if not (f in seen or seen.add(f))]
    return {"name": name, "kind": kind, "files": files, "size": sum(f.stat().st_size for f in files), "source": source}, ""


def describe(arguments: dict[str, Any]) -> str:
    """Approval-card line: what exactly is about to be shared."""
    plan, err = plan_album(str(arguments.get("path", "")), arguments.get("paths") or None)
    if plan is None:
        return f"share album: {err}"
    return f"share album '{plan['name']}': {len(plan['files'])} {plan['kind']} files, {human_size(plan['size'])} (from {plan['source']})"


def _duration(f: Path) -> float | None:
    try:
        import mutagen
        info = mutagen.File(f).info
        return round(float(info.length), 1)
    except Exception:  # noqa: BLE001 - no duration is fine, the player reads it from the stream
        return None


def make_send_album_tool(store: Store, session_id: int, console: Any) -> Tool:
    def handler(path: str = "", paths: list[str] | None = None, title: str = "", **_kwargs: Any) -> str:
        plan, err = plan_album(path, paths if isinstance(paths, list) else None)
        if plan is None:
            return f"Error: {err}"
        kind, files = plan["kind"], plan["files"]
        name = (title or plan["name"]).strip()[:120] or "Album"
        gid = uuid.uuid4().hex
        items: list[dict[str, Any]] = []
        for f in files:
            tok = store.add_file_share(session_id, str(f), disposition="inline", display_name=f.name, group_id=gid, group_name=name)
            item: dict[str, Any] = {"token": tok, "url": f"/api/files/{tok}", "title": f.stem, "filename": f.name,
                                    "size": f.stat().st_size, "mime": content_type(f.name)}
            if kind == "audio":
                tags = read_track_tags(f)
                item["title"] = tags.get("title") or f.stem
                if tags.get("artist"):
                    item["artist"] = tags["artist"]
                dur = _duration(f)
                if dur:
                    item["duration"] = dur
            items.append(item)
        cover = None
        if kind == "audio":
            cover = next((f for f in sorted(files[0].parent.iterdir()) if f.is_file() and f.stem.lower() in _COVER_NAMES
                          and media_kind(f.name) == "image"), None)
            cover = cover or next((a for a in (extract_album_art_cached(f) for f in files[:3]) if a), None)
        extra: dict[str, Any] = {}
        if cover is not None:
            ct = store.add_file_share(session_id, str(cover), disposition="inline", display_name=cover.name, group_id=gid, group_name=name)
            extra["cover_url"] = f"/api/files/{ct}"
        total = sum(i["size"] for i in items)
        console.emit_event("file_ready", kind="album", album_kind=kind, name=name, filename=name, count=len(items),
                           size=human_size(total), items=items, **extra)
        return (f"Album sent: '{name}', {len(items)} {kind} files ({human_size(total)}). It is already showing in the chat "
                "as a player/gallery - do NOT write any URL, tag or track list yourself; just say something short.")

    return Tool(
        name="send_album",
        description=(
            "Share SEVERAL media files as one album in chat: a music folder becomes a playlist player, a photo folder a "
            "gallery with lightbox, a video folder a list with a player. Use it whenever the user asks for a whole folder/"
            "album/playlist/photo set or a handful of files to play or browse (send_file is for one file or a zip download). "
            "Pass `path` (a folder, files are sorted by track number/name, non-media skipped) OR `paths` (an explicit "
            "list, kept in your order). Max 200 files. The user is shown the file count and total size and must approve. "
            "The player appears automatically - never write links or tags yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~-relative folder to share."},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Alternative to path: explicit file list."},
                "title": {"type": "string", "description": "Optional album name; defaults to the folder name."},
            },
        },
        handler=handler,
        risk="caution",
    )
