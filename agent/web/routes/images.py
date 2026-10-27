"""Polling endpoint for generate_image's background jobs - the frontend
hits this every few seconds after an `image_job` SSE event to update its
progress UI. See agent/tools/image_gen.py for why this is poll-based
rather than a live SSE push: the generation runs well past the chat
turn's own request/response cycle, so there's no open connection left to
push through by the time it finishes.

Delivery itself (creating the file-share token, persisting the file_card
message) happens directly in image_gen.py's background thread the moment
a generation succeeds - NOT here. This route is a pure, side-effect-free
status read: whether or not anyone is polling, the image still gets
delivered, and a reload just picks it up from history like any other
message.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.tools import image_gen
from agent.tools.image_gen import get_job_status
from agent.web.deps import get_client, get_store

router = APIRouter(prefix="/api/images", tags=["images"])


# The Images screen. These fixed paths come before "/{job_id}" so they are not taken for a job id.


@router.get("/info")
def info() -> dict[str, Any]:
    """Whether the local image model is installed, how long a picture takes on this machine, and the busy state."""
    busy = image_gen._GEN_SEMAPHORE._value == 0  # noqa: SLF001 - the tool's own one-at-a-time lock
    return {"installed": image_gen.IMAGE_GEN_CHECKPOINT.is_dir(), "model": "HiDream-O1-Image", "busy": busy, "eta_s": image_gen.estimate_eta_s(),
            "output_dir": str(image_gen.IMAGE_OUTPUT_DIR), "history": image_gen._load_duration_history()}  # noqa: SLF001


@router.get("/gallery")
def gallery(limit: int = 60, store=Depends(get_store)) -> list[dict[str, Any]]:
    """Delivered images (newest first), from the file links the generator created. The prompt shown is the saved link name (the first 40 characters)."""
    out_dir = Path(image_gen.IMAGE_OUTPUT_DIR)
    rows = [r for r in store._conn.execute("SELECT token, path, created_at, display_name FROM file_shares ORDER BY id DESC LIMIT 2000").fetchall()  # noqa: SLF001
            if Path(r["path"]).parent == out_dir][: max(1, min(limit, 200))]
    out = []
    for r in rows:
        try:
            size = Path(r["path"]).stat().st_size
        except OSError:
            size = None
        out.append({"url": f"/api/files/{r['token']}", "title": (r["display_name"] or "image").removesuffix(".png"), "created_at": r["created_at"], "size": size, "exists": size is not None})
    return out


@router.get("/queue")
def queue() -> list[dict[str, Any]]:
    """Generations started since the server came up (running first). Progress is the elapsed share of the median recent duration, not a measured step count."""
    now = time.time()
    with image_gen._JOBS_LOCK:  # noqa: SLF001
        jobs = [(k, dict(v)) for k, v in image_gen._JOBS.items()]  # noqa: SLF001
    out = []
    for k, v in jobs:
        el = now - v["started_at"]
        out.append({"id": k, "prompt": v["prompt"], "status": v["status"], "error": v.get("error"), "elapsed_s": round(el), "eta_s": round(v["eta_s"]),
                    "pct": 100 if v["status"] == "done" else min(95, round(100 * el / max(v["eta_s"], 1))) if v["status"] == "running" else 0,
                    "url": v.get("url"), "started_at": v["started_at"]})
    return sorted(out, key=lambda j: (j["status"] != "running", -j["started_at"]))


class GenerateIn(BaseModel):
    prompt: str
    session_id: int
    size: int = 1024


@router.post("/generate")
def generate(body: GenerateIn, client=Depends(get_client), store=Depends(get_store)) -> dict[str, Any]:
    """Start one generation, delivered into the chosen chat like one asked for in chat. Only one can run at a time on this machine."""
    prompt = body.prompt.strip()
    if not prompt or len(prompt) > 2000:
        raise HTTPException(422, "Give a prompt of 1-2000 characters.")
    if body.size not in (512, 768, 1024):
        raise HTTPException(422, "size must be 512, 768 or 1024.")
    if store._conn.execute("SELECT 1 FROM sessions WHERE id = ?", (body.session_id,)).fetchone() is None:  # noqa: SLF001
        raise HTTPException(404, "No such chat to deliver the image to.")
    if not image_gen.IMAGE_GEN_CHECKPOINT.is_dir():
        raise HTTPException(409, f"The image model is not installed at {image_gen.IMAGE_GEN_CHECKPOINT}.")
    if not image_gen._GEN_SEMAPHORE.acquire(blocking=False):  # noqa: SLF001
        raise HTTPException(409, "A generation is already running; this machine can only run one at a time.")
    try:
        return image_gen.start_generation(prompt, body.session_id, client, store, height=body.size, width=body.size)
    except Exception:
        image_gen._GEN_SEMAPHORE.release()  # noqa: SLF001 - the job thread releases it normally; it never started
        raise


@router.get("/{job_id}")
def image_job_status(job_id: str) -> dict:
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(404, "No such image job.")
    return {
        "status": job["status"], "prompt": job["prompt"], "error": job["error"],
        "url": job.get("url"), "filename": job.get("filename"), "size": job.get("size_str"),
    }
