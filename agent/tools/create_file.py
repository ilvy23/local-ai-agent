"""create_file: the assistant writes a file it just authored (script, csv, markdown, config...) into a per-session
outbox and hands it to the user as a download card. For files that already exist on disk use send_file instead.

Safe by construction rather than by asking: the name is reduced to a plain basename, the file lives in a fresh
outbox/<session>/<random>/ folder (nothing existing is ever overwritten, nothing outside the outbox is touched), it is
capped at MAX_BYTES, never made executable, and is always served as an attachment (see routes/files.py). The gate in
agent/tui.py still asks when the content was copied from a web page this turn.
"""

from __future__ import annotations

import base64
import binascii
import re
import uuid
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR
from agent.memory.store import Store
from agent.textio import human_size
from agent.tools.registry import Tool

OUTBOX_DIR = DEFAULT_DATA_DIR / "outbox"
MAX_BYTES = 10 * 1024 * 1024


def safe_filename(name: str) -> str:
    """Basename only, no control or path characters, no leading dots (hidden files), at most 120 chars."""
    base = re.split(r"[\\/]", str(name or "").strip())[-1]
    base = re.sub(r"[\x00-\x1f\x7f<>:\"|?*]", "_", base).strip(" .")
    base = base.lstrip(".")
    stem, dot, ext = base.rpartition(".")
    if dot and len(ext) > 12:
        stem, dot, ext = base, "", ""
    return (stem + dot + ext)[:120].rstrip(" .")


def make_create_file_tool(store: Store, session_id: int, console: Any) -> Tool:
    def handler(filename: str = "", content: str = "", encoding: str = "utf-8", **_kwargs: Any) -> str:
        name = safe_filename(filename)
        if not name:
            return "Error: give a plain file name such as notes.md (no folders, no leading dot)."
        if encoding == "base64":
            try:
                data = base64.b64decode(content, validate=False)
            except (binascii.Error, ValueError):
                return "Error: content is not valid base64."
        else:
            data = str(content).encode("utf-8")
        if len(data) > MAX_BYTES:
            return f"Error: file too large ({human_size(len(data))}, limit {human_size(MAX_BYTES)})."
        folder = OUTBOX_DIR / str(session_id) / uuid.uuid4().hex[:12]
        try:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / name
            target.write_bytes(data)
            target.chmod(0o644)
        except OSError as exc:
            return f"Error: couldn't write the file: {exc}"
        token = store.add_file_share(session_id, str(target), disposition="attachment", display_name=name)
        console.emit_event("file_ready", kind="download", url=f"/api/files/{token}", filename=name,
                           size=human_size(len(data)))
        return (f"File created and sent: {name} ({human_size(len(data))}). It is already showing in the chat as a "
                "download card - do NOT include any link or path in your reply, just say briefly what the file is.")

    return Tool(
        name="create_file",
        description=(
            "Create a NEW file from text you write and give it to the user as a download card in the chat. Use it "
            "only when the user asks for a file/download (e.g. 'write me a script / csv / markdown / config and "
            "give it to me as a file'). Do NOT use it for ordinary answers, code you are just showing, or notes "
            "(reply in text), and do not use it to edit project files (write_file) or to send a file that already "
            "exists on disk (send_file). filename is a plain name with extension, no folders. Max 10 MB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Plain file name with extension, e.g. report.csv."},
                "content": {"type": "string", "description": "The full file content."},
                "encoding": {"type": "string", "enum": ["utf-8", "base64"],
                             "description": "'utf-8' (default) for text; 'base64' only for binary content."},
            },
            "required": ["filename", "content"],
        },
        handler=handler,
        risk="safe",
    )
