"""Serves files handed out by the send_file tool.

Uses Starlette's FileResponse directly rather than reading the file into
memory ourselves: it supports the ASGI `http.response.pathsend` extension
(zero-copy sendfile(2) on servers that implement it, uvicorn included) and
answers HTTP Range requests out of the box - both matter here, since a
"send me this movie" file can be gigabytes and the phone may be seeking
around in a video rather than downloading it start-to-finish.
"""

from __future__ import annotations

from pathlib import Path

import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response

from agent.tools.mediatypes import INLINE_SAFE, content_type
from agent.web.deps import get_store

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/{token}")
def download_file(token: str, request: Request, store=Depends(get_store)) -> Response:
    share = store.get_file_share(token)
    if share is None:
        raise HTTPException(status_code=404, detail="No such download.")
    resolved = Path(share["path"])
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File no longer exists.")

    # Starlette answers a malformed Range with 400; the HTTP answer is 416 (unsatisfiable ranges it already handles itself).
    rng = request.headers.get("range")
    if rng is not None:
        specs = re.fullmatch(r"bytes=\s*(\d*)-(\d*)(?:\s*,\s*(\d*)-(\d*))*", rng.strip())
        bad = specs is None or any(a and b and int(a) > int(b) for a, b in re.findall(r"(\d*)-(\d*)", rng.split("=", 1)[1]))
        if bad:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{resolved.stat().st_size}"})

    media_type = content_type(resolved.name)
    disposition = share["disposition"]
    if disposition == "inline" and media_type not in INLINE_SAFE:
        disposition = "attachment"        # html/svg/unknown never render inside the app's origin
    if disposition == "attachment":
        # iOS Safari intercepts links to a recognized audio/video Content-Type
        # with its own built-in media chooser overlay ("Download" vs. "open in
        # ...") even when Content-Disposition says attachment - the real fix
        # media servers use is to stop declaring it as playable media at all
        # for a real download. The filename (and its real extension) in
        # Content-Disposition below is untouched, so it still saves correctly.
        media_type = "application/octet-stream"

    return FileResponse(
        resolved,
        media_type=media_type,
        filename=share["display_name"] or resolved.name,
        content_disposition_type=disposition,
        headers={"X-Content-Type-Options": "nosniff"},
    )
