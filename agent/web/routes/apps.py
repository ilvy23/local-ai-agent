"""Apps screen extras: the files the assistant shared (chat file links) and the prompt studio's saved chats, refine loop and
Max-anonymise trace. Slow model calls run as jobs (start, poll, stop)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent import promptlab
from agent.tools.mediatypes import content_type
from agent.web.deps import get_client, get_config, get_store
from agent.web.routes import _jobs

router = APIRouter(prefix="/api", tags=["apps"])

_KIND = {"image": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif"}, "video": {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"},
         "audio": {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".opus"}}


def _kind(name: str) -> str:
    ext = Path(name).suffix.lower()
    return next((k for k, v in _KIND.items() if ext in v), "file")


@router.get("/apps/shared-files")
def shared_files(limit: int = 200, store=Depends(get_store)) -> list[dict[str, Any]]:
    """Every link the assistant handed out (send_file / generated images), newest first. The product has no expiry: a link works as long as the file exists."""
    rows = store._conn.execute("SELECT token, path, created_at, disposition, display_name, group_id, group_name FROM file_shares ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
    out: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    for r in reversed(rows):                                  # oldest first so album tracks keep their order
        p = Path(r["path"])
        try:
            size = p.stat().st_size
        except OSError:
            size = None
        name = r["display_name"] or p.name
        if r["group_id"]:
            g = groups.setdefault(r["group_id"], {"name": r["group_name"] or "Album", "created_at": r["created_at"], "files": []})
            g["files"].append({"name": name, "kind": _kind(name), "size": size, "url": f"/api/files/{r['token']}", "mime": content_type(name)})
            continue
        out.append({"name": name, "kind": _kind(name), "created_at": r["created_at"], "exists": size is not None, "size": size,
                    "url": f"/api/files/{r['token']}", "inline": r["disposition"] == "inline"})
    for g in groups.values():
        kinds = [f["kind"] for f in g["files"]]
        kind = max(set(kinds), key=kinds.count)
        items = [f for f in g["files"] if f["kind"] == kind]       # drops the cover image of an audio album
        out.append({"name": g["name"], "kind": "album", "album_kind": kind, "created_at": g["created_at"], "count": len(items),
                    "exists": all(f["size"] is not None for f in items), "size": sum(f["size"] or 0 for f in items), "inline": True,
                    "items": [{"url": f["url"], "title": Path(f["name"]).stem, "filename": f["name"], "size": f["size"] or 0, "mime": f["mime"]} for f in items]})
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


# --------------------------------------------------------------------------------------------------- prompt studio: saved chats


def _session(sid: int) -> dict[str, Any]:
    s = next((x for x in promptlab._load_sessions() if isinstance(x, dict) and x.get("id") == sid), None)
    if s is None:
        raise HTTPException(404, "No such saved prompt chat.")
    return s


@router.get("/prompt/sessions/{sid}")
def prompt_session(sid: int) -> dict[str, Any]:
    s = _session(sid)
    msgs = [m for m in s.get("messages") or [] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    return {"id": s["id"], "title": s.get("title", ""), "mode": s.get("mode", ""), "created": s.get("created", ""), "messages": msgs}


@router.delete("/prompt/sessions/{sid}")
def prompt_session_delete(sid: int) -> dict[str, bool]:
    _session(sid)
    promptlab._write_sessions([s for s in promptlab._load_sessions() if not (isinstance(s, dict) and s.get("id") == sid)])
    return {"ok": True}


# --------------------------------------------------------------------------------------------------- prompt studio: jobs


class PromptJob(BaseModel):
    op: str                       # improve | plan | anon | refine | questions | chat
    text: str = ""
    answers: str = ""
    strength: str = "heavy"
    current: str = ""             # refine: the prompt to revise
    feedback: str = ""            # refine: what to change
    mode: str = "improve"         # chat: improve | plan | anon
    sid: int | None = None        # chat: resume this saved chat
    message: str = ""             # chat: the next message


_OPS = {"improve", "plan", "anon", "refine", "questions", "chat"}


def _need(v: str, what: str) -> str:
    v = v.strip()
    if not v or len(v) > 20_000:
        raise HTTPException(422, f"Give 1-20000 characters of {what}.")
    return v


@router.post("/prompt/jobs")
def prompt_job(body: PromptJob, client=Depends(get_client), config=Depends(get_config)) -> dict[str, str]:
    if body.op not in _OPS:
        raise HTTPException(422, "op must be one of " + ", ".join(sorted(_OPS)) + ".")
    if body.strength not in ("light", "heavy", "poison", "max"):
        raise HTTPException(422, "strength must be light, heavy, poison or max.")
    if body.op == "refine":
        cur, fb = _need(body.current, "the prompt to revise"), _need(body.feedback, "feedback")
    elif body.op == "chat":
        if body.mode not in promptlab._CHAT_SYS:
            raise HTTPException(422, "mode must be improve, plan or anon.")
        msg = _need(body.message, "message")
        if body.sid is not None:
            _session(body.sid)
    else:
        text = _need(body.text, "text")

    def work(job: dict[str, Any]) -> dict[str, Any]:
        if body.op == "improve":
            return {"text": promptlab.improve(client, config, text, body.answers)}
        if body.op == "plan":
            return {"text": promptlab.plan(client, config, text, body.answers)}
        if body.op == "questions":
            return {"questions": promptlab.clarifying_questions(client, config, text)}
        if body.op == "refine":
            return {"text": promptlab.refine(client, config, cur, fb)}
        if body.op == "anon":
            if body.strength == "max":
                final, trace = promptlab.anonymize_max(client, config, text, trace=True)
                return {"text": final, "trace": [{"pass": k, "text": trace.get(k, "")} for k in ("core", "keep", "strip", "draft")]}
            return {"text": promptlab.anonymize(client, config, text, body.strength)}
        # chat: same flow as `agent prompt` chat mode, autosaved every turn
        if body.sid is None:
            messages = [{"role": "system", "content": promptlab._CHAT_SYS[body.mode]}]
            mode = body.mode
        else:
            s = _session(body.sid)
            messages, mode = list(s.get("messages") or []), s.get("mode", body.mode)
        messages.append({"role": "user", "content": msg})
        reply = "".join(client.chat(messages, model=config["models"]["chat"], num_ctx=8192, temperature=0.4)).strip()
        messages.append({"role": "assistant", "content": reply})
        sid = promptlab._save_session(mode, messages, body.sid)
        return {"text": reply, "sid": sid}

    return {"job": _jobs.start("prompt", work)}


@router.get("/prompt/jobs/{jid}")
def prompt_job_state(jid: str) -> dict[str, Any]:
    j = _jobs.get(jid)
    if j is None or j["kind"] != "prompt":
        raise HTTPException(404, "No such job.")
    return {"id": j["id"], "status": j["status"], "error": j["error"], "elapsed_s": j["elapsed_s"], "result": j["result"] if j["status"] == "done" else None}


@router.post("/prompt/jobs/{jid}/stop")
def prompt_job_stop(jid: str) -> dict[str, bool]:
    j = _jobs.get(jid)
    if j is None or j["kind"] != "prompt":
        raise HTTPException(404, "No such job.")
    _jobs.stop(jid)                     # the model call itself cannot be interrupted: the result is dropped when it returns
    return {"ok": True}
