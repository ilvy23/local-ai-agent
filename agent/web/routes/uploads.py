"""Attachments: the GUI sends a file here, the chat message then names its path and the assistant reads it with read_file.

Raw body upload (`PUT /api/uploads/{session_id}/{filename}`) - no multipart dependency. Files land in
~/Documents/agent-uploads/<session>/ (AGENT_UPLOAD_DIR overrides), owner-only, never executed. A size cap protects the disk.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

MAX_BYTES = 25 * 1024 * 1024
_SAFE = re.compile(r"[^\w.\- ()+]+")


def upload_dir() -> Path:
    return Path(os.environ.get("AGENT_UPLOAD_DIR") or Path.home() / "Documents" / "agent-uploads").expanduser()


def _clean_name(name: str) -> str:
    base = _SAFE.sub("_", Path(name).name).strip(" .") or "file"
    return base[:120]


@router.put("/{session_id}/{filename:path}")
async def upload(session_id: int, filename: str, request: Request) -> dict[str, object]:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // (1024 * 1024)} MB).")
    folder = upload_dir() / str(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    name = _clean_name(filename)
    target = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    size = 0
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"File too large (max {MAX_BYTES // (1024 * 1024)} MB).")
                out.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(422, "The file is empty.")
    return {"path": str(target), "name": name, "size": size}
