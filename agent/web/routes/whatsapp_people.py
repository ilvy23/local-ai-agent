"""WhatsApp GUI: people and chats -> summaries, profiles, deep analysis, vibe, health, red flags, queue, self-reflection.

Mirrors `agent wa summarize|profile|speech-profile|vibe|health|repair|redflags|predict|me|queue|summaries|profiles|
forget-summary|forget-profile`. The model work is a job (see whatsapp_core). The big-model actions (deep profile, red flags,
predict) need `confirm: true` on the server as well as in the dialog. Nothing here sends anything: `repair` and `predict`
produce text for you to read.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from agent.web import deps
from agent.web.deps import get_config
from agent.web.routes import whatsapp_core as core

router = APIRouter(prefix="/api/wa", tags=["whatsapp"])

PERSON_ONLY = {"profile", "deep", "speech", "health", "repair", "redflags", "predict"}
NEEDS_CONFIRM = {"deep", "redflags", "predict"}
LABELS = {"profile": "Profile", "deep": "Deep analysis", "summary": "Chat summary", "speech": "Speech profile", "vibe": "Vibe check",
          "health": "Friendship health", "repair": "Repair suggestions", "redflags": "Red-flag scan", "predict": "Predict reaction"}


def _set(config: dict[str, Any], sql: str) -> set[str]:
    return {next(iter(r.values())) for r in core.ro_rows(config, sql)}


@router.get("/people")
def people(config=Depends(get_config)) -> list[dict[str, Any]]:
    """Chats with what has been built for them, newest first. Tolerates tables that do not exist yet."""
    core.wa_config(config)
    min_n = int(config["whatsapp"].get("min_messages", 5))
    rows = core.ro_rows(config, """
        SELECT c.jid AS jid, c.name AS name, COALESCE(c.is_group, 0) AS is_group, c.last_message_at AS last_message_at,
               COUNT(m.id) AS total, COALESCE(SUM(m.from_me = 0), 0) AS theirs
        FROM wa_chats c JOIN wa_messages m ON m.chat_jid = c.jid
        WHERE c.jid NOT LIKE '%@broadcast' AND c.jid NOT LIKE '%@newsletter'
        GROUP BY c.jid ORDER BY c.last_message_at DESC LIMIT 300""")
    prof = _set(config, "SELECT contact_jid FROM wa_profiles")
    deep = _set(config, "SELECT contact_jid FROM wa_unrestricted_profiles")
    summ = _set(config, "SELECT chat_jid FROM wa_chat_summaries")
    for r in rows:
        r["is_group"] = bool(r["is_group"])
        r["profile"], r["deep"], r["summary"] = r["jid"] in prof, r["jid"] in deep, r["jid"] in summ
        r["too_few"] = (not r["is_group"]) and r["theirs"] < min_n
    return rows


@router.get("/people/{jid}")
def person(jid: str, config=Depends(get_config)) -> dict[str, Any]:
    core.safe_jid(jid)
    core.wa_config(config)
    stats = core.ro_one(config, "SELECT COUNT(*) AS total, COALESCE(SUM(from_me = 0), 0) AS theirs, MIN(timestamp) AS first_at, "
                                "MAX(timestamp) AS last_at FROM wa_messages WHERE chat_jid = ?", (jid,)) or {}
    prof = core.ro_one(config, "SELECT profile, based_on_count, updated_at FROM wa_profiles WHERE contact_jid = ?", (jid,))
    deep = core.ro_one(config, "SELECT profile, based_on_count, effort, updated_at FROM wa_unrestricted_profiles WHERE contact_jid = ?", (jid,))
    speech = core.ro_one(config, "SELECT profile, based_on_count, updated_at FROM wa_speech_profiles WHERE contact_jid = ?", (jid,))
    summ = core.ro_one(config, "SELECT summary, covered_through, updated_at FROM wa_chat_summaries WHERE chat_jid = ?", (jid,))
    queue = core.ro_one(config, "SELECT v FROM wa_sync_state WHERE k = 'unrestricted_queue'")
    return {"jid": jid, "name": core.chat_name(config, jid), "is_group": jid.endswith("@g.us"), "stats": stats, "profile": prof, "deep": deep,
            "speech": speech, "summary": summ, "queued": jid in (queue or {}).get("v", ""),
            "min_messages": int(config["whatsapp"].get("min_messages", 5))}


# ------------------------------------------------------------------------------------------------ analysis jobs


class AnalysisIn(BaseModel):
    action: str
    depth: int | None = Field(None, ge=1, le=3000)
    effort: str = Field("medium", pattern="^(low|medium|high|max)$")
    text: str = Field("", max_length=2000)             # predict: the draft to judge; vibe: "as if they had just sent"
    confirm: bool = False


def _analysis(config: dict[str, Any], client: Any, jid: str, a: AnalysisIn):
    """The function a job runs for one action. Returns work(job) -> {"text": ...}."""
    depth = a.depth

    def work(job: core.Job) -> dict[str, Any]:
        job.say(f"{LABELS[a.action]}: reading the chat")
        with core.open_store(config) as store:
            text: str | None
            if a.action == "profile":
                text = core.mod("profile").build_profile(store, client, config, jid, depth=depth)
            elif a.action == "deep":
                U = core.mod("unrestricted")
                text = U.build_unrestricted_profile(store, client, config, jid, depth=depth, effort=a.effort)
                text = (U.IMPRESSIONS_HEADER + text) if text else None
            elif a.action == "summary":
                text = core.mod("summarize").summarize_chat(store, client, config, jid, depth=depth)
            elif a.action == "speech":
                text = core.mod("speech").build_speech_profile(store, client, config, jid)
            elif a.action == "health":
                text = core.mod("health").friendship_health(store, client, config, jid, depth=depth)
            elif a.action == "repair":
                text = core.mod("health").repair(store, client, config, jid, depth=depth)
            elif a.action == "redflags":
                text = core.mod("unrestricted").redflags(store, client, config, jid, depth=depth)
            elif a.action == "predict":
                text = core.mod("unrestricted").predict_reaction(store, client, config, jid, a.text, depth=depth)
            else:                                                       # vibe
                V = core.mod("vibe")
                v = V.read_vibe(store, client, config, jid, a.text or None)
                text = V.format_vibe(v) if v else None
        if not text:
            raise RuntimeError(f"Not enough messages yet (the minimum is {config['whatsapp'].get('min_messages', 5)} from them) or the model gave no answer.")
        return {"text": text}

    return work


@router.post("/people/{jid}/jobs")
def run_analysis(jid: str, body: AnalysisIn, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    core.require()
    core.safe_jid(jid)
    core.wa_config(config)
    if body.action not in LABELS:
        raise HTTPException(400, "Unknown action.")
    if body.action in PERSON_ONLY and jid.endswith("@g.us"):
        raise HTTPException(400, "That works on one person, not on a group. Use the chat summary for groups.")
    if body.action in NEEDS_CONFIRM and not body.confirm:
        raise HTTPException(400, "This runs the big model on the chat. Confirm it first.")
    if body.action == "predict" and not body.text.strip():
        raise HTTPException(400, "Enter the draft message to judge.")
    if body.action in ("profile", "deep", "summary") and body.depth and body.depth > (3000 if body.action == "summary" else 1000):
        raise HTTPException(400, "Depth is too large for this action.")
    return core.start_job(body.action, LABELS[body.action], _analysis(config, client, jid, body), target=jid).view()


class SummarizeAllIn(BaseModel):
    confirm: bool = False


@router.post("/summaries/run")
def summarize_everything(body: SummarizeAllIn, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    """Rolling summaries for every eligible chat (`agent wa summarize --all`, background model)."""
    core.require()
    core.wa_config(config)
    if not body.confirm:
        raise HTTPException(400, "Confirm first: this can take a long time.")

    def work(job: core.Job) -> dict[str, Any]:
        n = [0]

        def on_item(_name: str) -> None:
            n[0] += 1
            job.say(f"summarizing chat {n[0]}")

        with core.open_store(config) as store:
            res = core.mod("summarize").summarize_all(store, client, config, on_item=on_item)
        return {"summarized": res["summarized"], "skipped": res["skipped"]}

    return core.start_job("summaries", "Summarize all chats", work).view()


@router.delete("/people/{jid}/summary")
def forget_summary(jid: str, config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    core.safe_jid(jid)
    with core.open_store(core.wa_config(config)) as store:
        n = store.conn.execute("DELETE FROM wa_chat_summaries WHERE chat_jid = ?", (jid,)).rowcount
        store.conn.execute("DELETE FROM wa_sync_state WHERE k = ?", (f"summ_cursor:{jid}",))
        store.conn.commit()
    return {"deleted": n}


@router.delete("/people/{jid}/profile")
def forget_profile(jid: str, deep: bool = Query(False), config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    core.safe_jid(jid)
    with core.open_store(core.wa_config(config)) as store:
        n = store.conn.execute("DELETE FROM wa_profiles WHERE contact_jid = ?", (jid,)).rowcount
        if deep:
            n += store.conn.execute("DELETE FROM wa_unrestricted_profiles WHERE contact_jid = ?", (jid,)).rowcount
        store.conn.commit()
    return {"deleted": n}


# ------------------------------------------------------------------------------------------------ overnight queue


@router.get("/queue")
def queue_list(config=Depends(get_config)) -> list[dict[str, Any]]:
    core.require()
    with core.open_store(core.wa_config(config)) as store:
        return core.mod("queue").list_queue(store)


@router.post("/queue/{jid}")
def queue_add(jid: str, config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    core.safe_jid(jid)
    with core.open_store(core.wa_config(config)) as store:
        ok = core.mod("queue").add(store, jid)
    if not ok:
        raise HTTPException(409, "Already queued, or it is a group (deep profiles are for one person).")
    return {"queued": True}


@router.delete("/queue/{jid}")
def queue_remove(jid: str, config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    core.safe_jid(jid)
    with core.open_store(core.wa_config(config)) as store:
        return {"removed": bool(core.mod("queue").remove(store, jid))}


class DrainIn(BaseModel):
    effort: str = Field("high", pattern="^(low|medium|high|max)$")
    force: bool = False                      # skip the wait for a sustained idle machine
    confirm: bool = False


@router.post("/queue/run/drain")
def queue_drain(body: DrainIn, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    core.require()
    core.wa_config(config)
    if not body.confirm:
        raise HTTPException(400, "This runs the big model on every queued chat. Confirm it first.")

    def work(job: core.Job) -> dict[str, Any]:
        with core.open_store(config) as store:
            return core.mod("queue").drain(store, client, config, effort=body.effort, force=body.force, log=lambda m: job.say("queue: working"))

    return core.start_job("drain", "Deep-analysis queue", work).view()


# ------------------------------------------------------------------------------------------------ self-reflection


class ReflectIn(BaseModel):
    chat: str | None = None
    depth: int | None = Field(None, ge=1, le=2500)


@router.post("/reflect")
def reflect(body: ReflectIn, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    """How you show up in your chats (`agent wa me`). The result is journalled by the plugin like the CLI does."""
    core.require()
    core.wa_config(config)
    if body.chat:
        core.safe_jid(body.chat)

    def work(job: core.Job) -> dict[str, Any]:
        R = core.mod("reflect")
        job.say("reflecting")
        with core.open_store(config) as store:
            out = R.reflect_self(store, client, config, chat_jid=body.chat, depth=body.depth)
        if not out:
            raise RuntimeError("Not enough of your own messages yet.")
        R.save_reflection(config, core.chat_name(config, body.chat) if body.chat else "global", out)
        return {"text": out}

    return core.start_job("reflect", "Self-reflection", work, target=body.chat or "").view()
