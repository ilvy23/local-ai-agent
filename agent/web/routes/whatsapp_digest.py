"""WhatsApp GUI: digest, daily note (availability brief), day summary, weekly summary, day log, "on this day", nudges.

Mirrors `agent wa digest|brief|plans|day-summary|weekly-summary|day-log|moments|nudges`. The product has no schedule or delivery
setting for these (the background worker writes one digest a day on its own), so the screen shows that as fact, not as a control.
Nudges are computed from your chat rhythm; "dismiss" and "snooze" only hide a nudge in this app (a note in wa_sync_state).
"""

from __future__ import annotations

import re
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.web import deps
from agent.web.deps import get_config
from agent.web.routes import whatsapp_core as core

router = APIRouter(prefix="/api/wa", tags=["whatsapp"])

DIRS = {"digests": "digests", "day-summaries": "day-summaries", "weekly-summaries": "weekly-summaries", "day-log": "day-log",
        "reflections": "reflections"}
_NAME = re.compile(r"^[\w.-]{1,120}\.md$")
BRIEF = [("Doing", "What are you up to today?"), ("Availability", "When are you free vs busy?"), ("Location", "Where are you / going anywhere?"),
         ("Plans", "Any plans with friends? (who / when)"), ("Mood", "How are you feeling / what's your energy today?"),
         ("Up for", "Anything you'd be up for if someone asks?"), ("Not up for", "Anything you'd turn down today?"),
         ("On your mind", "Anything happening or on your mind?"), ("Careful with", "Anything the AI should avoid or be careful about?")]


def _names(config: dict[str, Any], kind: str) -> list[str]:
    d = core.db_path(config).parent / DIRS[kind]
    try:
        return sorted((p.name for p in d.glob("*.md") if _NAME.match(p.name)), reverse=True)[:90]
    except OSError:
        return []


@router.get("/digest/overview")
def digest_overview(config=Depends(get_config)) -> dict[str, Any]:
    core.wa_config(config)
    cur = core.ro_one(config, "SELECT v FROM wa_sync_state WHERE k='digest_cursor'")
    note = core.ro_one(config, "SELECT note, updated_at FROM wa_daily_note WHERE date = ?", (time.strftime("%Y-%m-%d"),))
    return {"installed": core.installed(), "hint": core.CLI_HINT, "digest_cursor": int(cur["v"]) if cur and str(cur["v"]).isdigit() else None,
            "note": note, "files": {k: _names(config, k) for k in DIRS}, "today": time.strftime("%Y-%m-%d"),
            "brief": [{"label": a, "question": q} for a, q in BRIEF]}


@router.get("/digest/files/{kind}")
def digest_files(kind: str, config=Depends(get_config)) -> list[str]:
    if kind not in DIRS:
        raise HTTPException(404, "Unknown folder.")
    return _names(config, kind)


@router.get("/digest/files/{kind}/{name}")
def digest_file(kind: str, name: str, config=Depends(get_config)) -> dict[str, str]:
    if kind not in DIRS or name not in _names(config, kind):         # only files that are listed: no path tricks
        raise HTTPException(404, "No such file.")
    return {"name": name, "text": (core.db_path(config).parent / DIRS[kind] / name).read_text(errors="replace")}


class DigestIn(BaseModel):
    kind: str = Field(pattern="^(digest|day_summary|weekly_summary)$")
    preview: bool = False        # digest only: show it without moving the "since" mark
    force: bool = False          # summaries only: write again even if the file exists


@router.post("/digest/jobs")
def digest_job(body: DigestIn, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    core.require()
    core.wa_config(config)

    def work(job: core.Job) -> dict[str, Any]:
        with core.open_store(config) as store:
            if body.kind == "digest":
                n = [0]

                def on_item(_label: str) -> None:
                    n[0] += 1
                    job.say(f"digesting chat {n[0]}")

                text = core.mod("digest").daily_digest(store, client, config, mark=not body.preview, on_item=on_item)
                return {"text": text or "", "empty": text is None}
            if body.kind == "day_summary":
                p = core.mod("day_summary").day_summary(store, client, config, force=body.force, log=lambda m: job.say("writing the day summary"))
            else:
                p = core.mod("weekly_summary").weekly_summary(store, client, config, force=body.force, log=lambda m: job.say("writing the weekly summary"))
        return {"file": p.name if p else "", "empty": p is None}

    label = {"digest": "Digest" + (" (preview)" if body.preview else ""), "day_summary": "Day summary", "weekly_summary": "Weekly summary"}[body.kind]
    return core.start_job(body.kind, label, work).view()


# ------------------------------------------------------------------------------------------------ daily note


class NoteIn(BaseModel):
    note: str | None = Field(None, max_length=4000)
    brief: dict[str, str] | None = None        # label -> answer, composed like `agent wa brief`


@router.put("/daily-note")
def set_note(body: NoteIn, config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    text = (body.note or "").strip()
    if body.brief:
        labels = {a for a, _ in BRIEF}
        parts = [f"- {k}: {v.strip()[:400]}" for k, v in body.brief.items() if k in labels and v.strip()]
        text = ("Today's brief:\n" + "\n".join(parts)) if parts else text
    if not text:
        raise HTTPException(400, "Nothing to save.")
    with core.open_store(core.wa_config(config)) as store:
        core.mod("daily_note").set_note(store, text)
    return {"saved": True, "note": text}


# ------------------------------------------------------------------------------------------------ moments, nudges


@router.get("/moments")
def moments(config=Depends(get_config)) -> list[dict[str, Any]]:
    core.require()
    with core.open_store(core.wa_config(config)) as store:
        return core.mod("moments").on_this_day(store)


@router.get("/nudges")
def nudges(config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    core.wa_config(config)
    with core.open_store(config) as store:
        rows = core.mod("nudges").going_quiet(store)
        hidden = {r["k"][len("gui_nudge:"):]: int(r["v"]) for r in
                  (dict(x) for x in store.conn.execute("SELECT k, v FROM wa_sync_state WHERE k LIKE 'gui_nudge:%'").fetchall()) if str(r["v"]).isdigit()}
    now = time.time()
    shown = [{**r, "hidden_until": None} for r in rows if hidden.get(r["jid"], 0) <= now]
    return {"nudges": shown, "hidden": sum(1 for r in rows if hidden.get(r["jid"], 0) > now)}


class SnoozeIn(BaseModel):
    hours: int = Field(1, ge=1, le=24 * 90)


@router.post("/nudges/{jid}/hide")
def nudge_hide(jid: str, body: SnoozeIn, config=Depends(get_config)) -> dict[str, Any]:
    """Snooze (a few hours) or dismiss (30 days = 720 h): hides the nudge in this app only."""
    core.require()
    core.safe_jid(jid)
    until = int(time.time()) + body.hours * 3600
    with core.open_store(core.wa_config(config)) as store:
        store.conn.execute("INSERT INTO wa_sync_state (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (f"gui_nudge:{jid}", str(until)))
        store.conn.commit()
    return {"until": until}
