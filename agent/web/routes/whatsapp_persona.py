"""WhatsApp GUI: persona clones (style card, evaluation, training data, LoRA training, simulated chat), auto-reply STATE, safety.

Mirrors `agent wa persona build|say|eval|show|export|list`, `agent wa clone|clones|clone-rename`, `agent wa auto status|sent|allow list`
and the policy around sending. Everything a persona says is a local simulation: nothing is sent, the person is not contacted.

Auto-reply is state only. The only changes reachable from here make it SAFER: mode off or draft (never auto), loop off, and
removing a contact from the allowlist. Turning auto-reply on, adding contacts, loop on and instant speed are not offered.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.web import deps
from agent.web.deps import get_config
from agent.web.routes import whatsapp_core as core
from agent.web.routes.whatsapp_sync import auto_state

router = APIRouter(prefix="/api/wa", tags=["whatsapp"])


def _pkill(pattern: str) -> None:
    """`pkill -f` where pkill exists (procps); a no-op elsewhere (Termux / minimal images may not ship it)."""
    try:
        subprocess.run(["pkill", "-f", pattern], check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def _count_lines(p: Path) -> int | None:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _env() -> dict[str, Any]:
    try:
        pc = core.mod("persona_clone")
        return {"training": bool(Path(pc.PYTHON).exists() and (Path(pc.MOOD_LLM_DIR) / "clone_pipeline.py").exists()), "dir": str(pc.MOOD_LLM_DIR)}
    except core.PluginMissing:
        return {"training": False, "dir": ""}


@router.get("/persona/overview")
def persona_overview(config=Depends(get_config)) -> dict[str, Any]:
    core.wa_config(config)
    cards = core.ro_rows(config, "SELECT contact_jid AS jid, name, n_messages, built_at, fingerprint, eval FROM wa_persona_cards ORDER BY built_at DESC")
    clones = core.ro_rows(config, "SELECT name, contact_jid AS jid, created_at FROM wa_persona_clones ORDER BY created_at DESC")
    ad = None
    if core.installed():
        try:
            ad = core.mod("persona_adapter")
        except core.PluginMissing:
            ad = None
    for c in cards:
        meta = core.jsonable_card(c)
        c["eval"], c["facts_hint"] = meta["eval"], None
        c.pop("fingerprint", None)
        c["train"] = c["val"] = None
        c["lora"] = False
        if ad:
            try:
                d = ad.persona_dir(c["name"] or "", c["jid"])
                c["train"], c["val"] = _count_lines(d / "train.jsonl"), _count_lines(d / "val.jsonl")
                c["lora"] = bool(ad.available(c["name"] or "", c["jid"]))
            except Exception:  # noqa: BLE001 - a missing folder is just "no dataset yet"
                pass
    cands: list[dict[str, Any]] = []
    if core.installed():
        try:
            with core.open_store(config) as store:
                cands = core.mod("persona_clone").list_candidate_contacts(store)[:40]
        except Exception:  # noqa: BLE001
            cands = []
    return {"installed": core.installed(), "hint": core.CLI_HINT, "cards": cards, "clones": clones, "candidates": cands, "env": _env(),
            "note": "Personas are local simulations. Nothing they write is sent."}


@router.get("/persona/{jid}")
def persona_detail(jid: str, config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    core.safe_jid(jid)
    with core.open_store(core.wa_config(config)) as store:
        PE = core.mod("persona")
        card = PE.load_card(store, jid)
        if not card:
            raise HTTPException(404, "No persona built for this chat yet.")
        rules = PE.P.rules_text(card["fingerprint"]) if card["fingerprint"] else ""
    fp = card["fingerprint"]
    keep = ("n_msgs", "n_turns", "len_p50", "len_p10", "len_p90", "burst_mean", "burst_max", "lower_start", "ends_period", "emoji_share", "laugh_share",
            "latency_p25", "latency_p50", "latency_p75")
    return {"jid": jid, "name": card["name"], "built_at": card["built_at"], "n_messages": card["n_messages"], "card": card["card"], "rules": rules,
            "facts": len(card["facts"]), "fingerprint": {k: fp.get(k) for k in keep}, "eval": card["eval"]}


class PersonaJobIn(BaseModel):
    action: str = Field(pattern="^(build|eval|export|say|clone)$")
    name: str = Field("", max_length=60)
    k: int = Field(24, ge=4, le=100)
    n: int = Field(4, ge=1, le=10)
    epochs: int = Field(4, ge=1, le=20)
    text: str = Field("", max_length=1500)
    history: list[tuple[str, str]] = Field(default_factory=list, max_length=40)
    explain: bool = False
    confirm: bool = False


def _progress_pct(msg: str) -> float | None:
    m = re.search(r"(\d+)/(\d+)", msg)
    return 100 * int(m[1]) / max(1, int(m[2])) if m else None


@router.post("/persona/{jid}/jobs")
def persona_job(jid: str, body: PersonaJobIn, config=Depends(get_config), client=Depends(deps.get_client)) -> dict[str, Any]:
    core.require()
    core.safe_jid(jid)
    core.wa_config(config)
    if jid.endswith("@g.us"):
        raise HTTPException(400, "Personas are for one person, not a group.")
    if body.action != "say" and not body.confirm:
        raise HTTPException(400, "Confirm first: this reads the whole chat on the big model" + (" and trains a model on your GPU." if body.action == "clone" else "."))
    name = body.name.strip() or core.chat_name(config, jid).split("@")[0]
    on_stop = None

    if body.action == "clone":
        if not _env()["training"]:
            raise HTTPException(409, f"The training environment is not installed ({_env()['dir'] or 'mood-llm'}). Nothing was started.")
        pat = f"clone_pipeline.py.*--jid {re.escape(jid)}"
        on_stop = lambda: _pkill(pat)

    def work(job: core.Job) -> dict[str, Any]:
        PE = core.mod("persona")
        prog = lambda m: job.say(m, _progress_pct(m))              
        if body.action == "clone":
            pc = core.mod("persona_clone")

            def cb(label: str, pct: float) -> None:
                job.say(label, pct)

            try:
                path = pc.run_pipeline_with_progress(str(core.db_path(config)), jid, name, cb, epochs=body.epochs)
            except RuntimeError:
                if job.stop.is_set():
                    raise core.Stopped() from None
                raise
            with core.open_store(config) as store:
                pc.save_clone(store, name, jid, path)
            return {"model_path": path, "name": name}
        with core.open_store(config) as store:
            PE.ensure_tables(store)
            if body.action == "build":
                built = PE.build_card(store, client, config, jid, name, on_progress=prog)
                PE.save_card(store, jid, name=name, n_messages=built["fingerprint"]["n_msgs"], fingerprint=built["fingerprint"], card=built["card"], facts=built["facts"])
                return {"stats": built["stats"], "text": built["card"]}
            if body.action == "eval":
                res = PE.evaluate(store, client, config, jid, k=body.k, n=body.n, on_progress=prog)
                card = PE.load_card(store, jid)
                PE.save_card(store, jid, name=card["name"], n_messages=card["n_messages"], fingerprint=card["fingerprint"], card=card["card"], facts=card["facts"],
                             eval_={"summary": res["summary"], "k": res["k"], "cutoff": res["cutoff"]})
                return {"text": PE.format_eval(res), "summary": res["summary"]}
            if body.action == "export":
                info = PE.export_sft(store, jid)
                return {"train": info["train"], "val": info["val"], "dir": info["dir"]}
            r = PE.reply_as(store, client, config, jid, body.text, history=[tuple(h) for h in body.history] or None, n=body.n, judge=body.n > 2, on_progress=prog)
            out: dict[str, Any] = {"messages": r["messages"], "text": "\n".join(r["messages"])}
            if body.explain:
                out["candidates"] = [{"src": c.get("src", "27B"), "style": c.get("style"), "sem": c.get("sem"), "ai": c.get("ai"), "judge": c.get("judge"),
                                      "messages": c.get("messages"), "chosen": i == r["chosen"]} for i, c in enumerate(r["candidates"])]
            return out

    if body.action == "say" and not body.text.strip():
        raise HTTPException(400, "Write what you would say to them.")
    labels = {"build": "Build persona card", "eval": "Evaluate persona", "export": "Prepare training data", "say": "Persona reply (simulation)", "clone": "Train persona (LoRA)"}
    return core.start_job("persona_" + body.action, labels[body.action], work, target=jid, on_stop=on_stop).view()


class RenameIn(BaseModel):
    new_name: str = Field(min_length=1, max_length=60)


@router.post("/persona/clones/{name}/rename")
def clone_rename(name: str, body: RenameIn, config=Depends(get_config)) -> dict[str, Any]:
    core.require()
    with core.open_store(core.wa_config(config)) as store:
        if not core.mod("persona_clone").rename_clone(store, name, body.new_name.strip()):
            raise HTTPException(404, "No saved clone with that name.")
    return {"name": body.new_name.strip()}


# ------------------------------------------------------------------------------------------------ auto-reply (state)


def _bridge_allowlist() -> set[str]:
    try:
        return {ln.strip() for ln in (core.plugin_root() / "bridge" / "allowed-send.txt").read_text().split() if ln.strip()}
    except OSError:
        return set()


def _allow_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = list((config.get("whatsapp") or {}).get("auto_reply_contacts") or [])
    bridge = _bridge_allowlist()
    out = []
    for e in entries:
        jid = e if "@" in str(e) else None
        if jid is None and core.installed():
            try:
                with core.open_store(config) as store:
                    jid = core.mod("tools").resolve_contact(store, str(e)).jid
            except Exception:  # noqa: BLE001
                jid = None
        out.append({"entry": str(e), "jid": jid, "name": core.chat_name(config, jid) if jid else str(e), "in_bridge_allowlist": bool(jid and jid in bridge)})
    return out


@router.get("/autoreply")
def autoreply(config=Depends(get_config)) -> dict[str, Any]:
    core.wa_config(config)
    st = auto_state(config)
    allow = _allow_entries(config)
    pending = core.ro_rows(config, "SELECT id, contact_jid, status, reason, send_at, created_at FROM wa_autoreply WHERE status IN ('held','scheduled') ORDER BY id DESC LIMIT 50")
    per = {r["contact_jid"]: r["n"] for r in core.ro_rows(config, "SELECT contact_jid, COUNT(*) AS n FROM wa_autoreply WHERE status IN ('held','scheduled') GROUP BY contact_jid")}
    for a in allow:
        a["pending"] = per.get(a["jid"], 0)
    for p in pending:
        p["name"] = core.chat_name(config, p["contact_jid"])
    ac = config["whatsapp"].get("auto_reply") or {}
    return {**st, "allowlist": allow, "pending_rows": pending, "send": core.send_state(),
            "delay_min": ac.get("delay_min_s"), "delay_max": ac.get("delay_max_s"), "quiet_hours": (ac.get("loop") or {}).get("quiet_hours")}


class ModeIn(BaseModel):
    mode: str


@router.put("/autoreply/mode")
def set_mode(body: ModeIn, config=Depends(get_config)) -> dict[str, str]:
    """Only the two modes that never send. Turning auto on is deliberately not possible from the app."""
    if body.mode not in ("off", "draft"):
        raise HTTPException(400, "Only 'off' and 'draft' can be set here. Auto-reply sending cannot be switched on from this app.")
    core.update_config_yaml(config, lambda d: d.setdefault("whatsapp", {}).setdefault("auto_reply", {}).__setitem__("mode", body.mode))
    return {"mode": body.mode}


class LoopIn(BaseModel):
    enabled: bool


@router.put("/autoreply/loop")
def set_loop(body: LoopIn, config=Depends(get_config)) -> dict[str, bool]:
    if body.enabled:
        raise HTTPException(400, "Loop mode texts people first, so it cannot be switched on from this app.")
    core.update_config_yaml(config, lambda d: d.setdefault("whatsapp", {}).setdefault("auto_reply", {}).setdefault("loop", {}).__setitem__("enabled", False))
    return {"enabled": False}


@router.delete("/autoreply/allow/{jid}")
def allow_remove(jid: str, config=Depends(get_config)) -> dict[str, Any]:
    """Stop the auto-responder from texting this person (`agent wa auto allow remove`): config list, the bridge's own file, queued drafts."""
    core.require()
    core.safe_jid(jid)
    core.wa_config(config)
    entries = list(config["whatsapp"].get("auto_reply_contacts") or [])
    with core.open_store(config) as store:
        resolve = core.mod("tools").resolve_contact

        def jid_of(e: str) -> str | None:
            return e if "@" in e else resolve(store, e).jid

        keep = [e for e in entries if jid_of(str(e)) != jid and e != jid]
        if len(keep) == len(entries):
            raise HTTPException(404, "That contact is not on the allowlist.")
        jids = [j for j in (jid_of(str(e)) for e in keep) if j]
        n = store.conn.execute("UPDATE wa_autoreply SET status='cancelled', reason='removed from allowlist' WHERE contact_jid=? AND status IN ('held','scheduled')", (jid,)).rowcount
        store.conn.commit()
    core.update_config_yaml(config, lambda d: d.setdefault("whatsapp", {}).__setitem__("auto_reply_contacts", jids))
    path = core.plugin_root() / "bridge" / "allowed-send.txt"
    try:
        path.write_text("\n".join(jids) + ("\n" if jids else ""))
    except OSError as exc:
        raise HTTPException(500, f"Config updated, but the bridge allowlist file could not be written: {exc}") from exc
    return {"removed": jid, "cancelled_drafts": n}


# ------------------------------------------------------------------------------------------------ safety


@router.get("/safety")
def safety(config=Depends(get_config)) -> dict[str, Any]:
    core.wa_config(config)
    tools: list[dict[str, str]] = []
    if core.installed():
        try:
            from agent.tools.registry import ToolRegistry

            reg = ToolRegistry()
            core.mod("tools").register_wa_tools(reg, None, None, None, config)
            tools = [{"name": t.name, "risk": str(t.risk), "description": (t.description or "")[:110]} for t in reg.list()]
        except Exception:  # noqa: BLE001 - listing is informational
            tools = []
    allow = _allow_entries(config)
    return {"installed": core.installed(), "send": core.send_state(), "auto": auto_state(config), "tools": tools,
            "allowlist": len(allow), "unsynced": [a["name"] for a in allow if not a["in_bridge_allowlist"]],
            "bridge_allowlist": len(_bridge_allowlist())}
