"""WhatsApp GUI routes beyond reading: jobs, analysis, digest, persona, auto-reply state, safety. Fake plugin, synthetic data."""

import sqlite3
import time
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("fastapi")
import yaml  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agent import settings  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402
from agent.web.routes import whatsapp_core as core  # noqa: E402

DDL = """
CREATE TABLE IF NOT EXISTS wa_chats (jid TEXT PRIMARY KEY, name TEXT, is_group INTEGER, last_message_at INTEGER);
CREATE TABLE IF NOT EXISTS wa_contacts (jid TEXT PRIMARY KEY, name TEXT, is_group INTEGER, updated_at TEXT);
CREATE TABLE IF NOT EXISTS wa_messages (id TEXT PRIMARY KEY, chat_jid TEXT, sender_jid TEXT, from_me INTEGER, timestamp INTEGER, text TEXT, type TEXT);
CREATE TABLE IF NOT EXISTS wa_sync_state (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS wa_chat_summaries (chat_jid TEXT PRIMARY KEY, summary TEXT, covered_through INTEGER, updated_at TEXT);
CREATE TABLE IF NOT EXISTS wa_profiles (contact_jid TEXT PRIMARY KEY, profile TEXT, based_on_count INTEGER, updated_at TEXT);
CREATE TABLE IF NOT EXISTS wa_autoreply (id INTEGER PRIMARY KEY, contact_jid TEXT, status TEXT, reason TEXT, send_at INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS wa_daily_note (date TEXT PRIMARY KEY, note TEXT, updated_at TEXT);
"""


def wait(c, job):
    for _ in range(100):
        j = c.get(f"/api/wa/jobs/{job['id']}").json()
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    monkeypatch.delenv("AGENT_WA_ALLOW_SEND", raising=False)
    db = tmp_path / "t.db"
    plug = tmp_path / "plug" / "whatsapp"
    (plug / "bridge").mkdir(parents=True)
    (plug / "bridge" / "allowed-send.txt").write_text("a@s.whatsapp.net\nb@s.whatsapp.net\n")
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text("whatsapp:\n  auto_reply:\n    mode: auto\n    loop: {enabled: true}\n")
    monkeypatch.setattr(settings, "DEFAULT_CONFIG_PATH", cfgfile)
    from agent.memory.store import Store
    Store(db).close()
    c = sqlite3.connect(db)
    c.executescript(DDL + """
        INSERT INTO wa_chats VALUES ('a@s.whatsapp.net','Test Alice',0,300),('g@g.us','Test Group',1,200);
        INSERT INTO wa_messages VALUES ('1','a@s.whatsapp.net','a',0,100,'hi','text'),('2','a@s.whatsapp.net',NULL,1,200,'yo','text'),('3','g@g.us','x',0,150,'grp','text');
        INSERT INTO wa_autoreply VALUES (1,'a@s.whatsapp.net','held','instant',NULL,'2026-01-01'),(2,'b@s.whatsapp.net','sent','x',1,'2026-01-01');""")
    c.commit(); c.close()
    cfg = {"data": {"db_path": str(db)}, "models": {"chat": "m"},
           "whatsapp": {"auto_reply": {"mode": "auto", "loop": {"enabled": True}}, "auto_reply_contacts": ["a@s.whatsapp.net", "b@s.whatsapp.net"]}}
    calls = []
    fake = {
        "schema": NS(ensure_schema=lambda conn: conn.executescript(DDL)),
        "bridge_client": NS(BridgeClient=lambda **kw: NS(health=lambda: (_ for _ in ()).throw(RuntimeError("bridge down")), backfill=lambda d: calls.append(("backfill", d)), lid_map=lambda: {"lid_map": []})),
        "ingest": NS(sync=lambda store, vec, llm, config, bridge, embed=True: calls.append(("sync", embed)) or {"new_messages": 3, "chats": 2, "contacts": 1},
                     _set_state=lambda store, k, v: calls.append(("state", k, v)), _embed_backlog=lambda *a: 0),
        "profile": NS(build_profile=lambda store, llm, config, jid, depth=None: "A kind person." if jid.startswith("a@") else None),
        "unrestricted": NS(IMPRESSIONS_HEADER="H:", build_unrestricted_profile=lambda *a, **k: "deep", redflags=lambda *a, **k: "none", predict_reaction=lambda *a, **k: "ok"),
        "tools": NS(resolve_contact=lambda store, e: NS(jid=e if "@" in e else None, name=e, ambiguous=False, matches=[]), register_wa_tools=lambda reg, *a: None),
        "nudges": NS(going_quiet=lambda store: [{"jid": "a@s.whatsapp.net", "label": "Test Alice", "silent_days": 9, "cadence_days": 1.0, "overdue": 9.0}]),
        "moments": NS(on_this_day=lambda store: []),
        "queue": NS(list_queue=lambda store: [], add=lambda s, j: True, remove=lambda s, j: True),
        "persona": NS(ensure_tables=lambda s: None, load_card=lambda s, j: None, reply_as=lambda *a, **k: {"messages": ["sure"], "candidates": [{"messages": ["sure"]}], "chosen": 0}),
        "persona_clone": NS(PYTHON="/nonexistent/python", MOOD_LLM_DIR="/nonexistent", list_candidate_contacts=lambda s: [], rename_clone=lambda *a: True),
    }
    monkeypatch.setattr(core, "installed", lambda: True)
    monkeypatch.setattr(core, "plugin_root", lambda: plug)
    monkeypatch.setattr(core, "mod", lambda n: fake[n] if n in fake else (_ for _ in ()).throw(core.PluginMissing(n)))
    monkeypatch.setattr("agent.memory.vectors.VectorIndex", lambda store: None)
    core.reset_jobs()
    app.dependency_overrides[deps.get_config] = lambda: cfg
    app.dependency_overrides[deps.get_client] = lambda: NS()
    yield TestClient(app), cfg, cfgfile, plug, calls
    app.dependency_overrides.clear()
    core.reset_jobs()


def test_plugin_missing_is_a_clean_state(env, monkeypatch):
    c, *_ = env
    monkeypatch.setattr(core, "installed", lambda: False)
    o = c.get("/api/wa/overview").json()
    assert o["installed"] is False and o["bridge"] is None and "nightage wa" in o["hint"]
    assert c.post("/api/wa/sync").status_code == 503
    assert c.post("/api/wa/people/a@s.whatsapp.net/jobs", json={"action": "profile"}).status_code == 503
    assert [r["name"] for r in c.get("/api/wa/chats").json()] == ["Test Alice", "Test Group"]


def test_overview_counts_and_safety_state(env):
    c, *_ = env
    o = c.get("/api/wa/overview").json()
    assert o["counts"]["messages"] == 3 and o["bridge"]["reachable"] is False
    assert o["send"] == {"blocked": True, "env_enabled": False} and o["auto"]["mode"] == "auto" and o["auto"]["pending"] == 1
    assert o["embed"]["embeddable"] == 3 and o["embed"]["missing"] == 3


def test_sync_job_runs_and_only_one_at_a_time(env):
    c, _, _, _, calls = env
    j = c.post("/api/wa/sync", json={"days": 7}).json()
    assert wait(c, j)["result"]["new_messages"] == 3
    assert ("backfill", 7) in calls and ("sync", True) in calls
    assert c.post("/api/wa/sync/reingest", json={}).status_code == 400
    assert wait(c, c.post("/api/wa/sync/reingest", json={"confirm": True}).json())["state"] == "done"
    assert ("state", "cursor", "0") in calls
    assert wait(c, c.post("/api/wa/sync/pull").json())["state"] == "done" and ("sync", False) in calls


def test_second_job_while_running_is_409(env):
    c, *_ = env
    import threading
    gate = threading.Event()
    core.start_job("x", "Slow", lambda job: gate.wait(2) and None)
    assert c.post("/api/wa/sync").status_code == 409
    gate.set()


def test_stop_job_is_cooperative(env):
    c, *_ = env
    import threading
    never = threading.Event()                    # time.sleep is stubbed in the suite, so wait on an event instead

    def work(job):
        while not never.wait(0.01):
            job.say("step")
    j = core.start_job("x", "Loop", work)
    assert c.post(f"/api/wa/jobs/{j.id}/stop").status_code == 200
    assert wait(c, {"id": j.id})["state"] == "stopped"


def test_analysis_rules(env):
    c, *_ = env
    a = "a@s.whatsapp.net"
    assert wait(c, c.post(f"/api/wa/people/{a}/jobs", json={"action": "profile"}).json())["result"]["text"] == "A kind person."
    assert c.post(f"/api/wa/people/{a}/jobs", json={"action": "deep"}).status_code == 400            # needs confirm
    assert c.post(f"/api/wa/people/{a}/jobs", json={"action": "predict", "confirm": True}).status_code == 400   # needs a draft
    assert c.post("/api/wa/people/g@g.us/jobs", json={"action": "profile"}).status_code == 400        # groups
    assert c.post(f"/api/wa/people/{a}/jobs", json={"action": "bogus"}).status_code == 400
    j = wait(c, c.post("/api/wa/people/b@s.whatsapp.net/jobs", json={"action": "profile"}).json())
    assert j["state"] == "failed" and "Not enough" in j["error"]
    deep = wait(c, c.post(f"/api/wa/people/{a}/jobs", json={"action": "deep", "confirm": True}).json())
    assert deep["result"]["text"] == "H:deep"


def test_people_flags(env):
    c, _, _, _, _ = env
    rows = {r["jid"]: r for r in c.get("/api/wa/people").json()}
    assert rows["a@s.whatsapp.net"]["too_few"] is True and rows["g@g.us"]["is_group"] is True


def test_autoreply_can_only_get_safer(env):
    c, cfg, cfgfile, plug, _ = env
    assert c.put("/api/wa/autoreply/mode", json={"mode": "auto"}).status_code == 400
    assert c.put("/api/wa/autoreply/loop", json={"enabled": True}).status_code == 400
    assert c.put("/api/wa/autoreply/mode", json={"mode": "draft"}).json() == {"mode": "draft"}
    assert c.put("/api/wa/autoreply/loop", json={"enabled": False}).status_code == 200
    saved = yaml.safe_load(cfgfile.read_text())["whatsapp"]["auto_reply"]
    assert saved["mode"] == "draft" and saved["loop"]["enabled"] is False and cfg["whatsapp"]["auto_reply"]["mode"] == "draft"
    r = c.delete("/api/wa/autoreply/allow/a@s.whatsapp.net").json()
    assert r["cancelled_drafts"] == 1
    assert (plug / "bridge" / "allowed-send.txt").read_text().split() == ["b@s.whatsapp.net"]
    assert yaml.safe_load(cfgfile.read_text())["whatsapp"]["auto_reply_contacts"] == ["b@s.whatsapp.net"]
    assert c.delete("/api/wa/autoreply/allow/a@s.whatsapp.net").status_code == 404
    st = c.get("/api/wa/autoreply").json()
    assert st["mode"] == "draft" and st["allowlist"][0]["name"] == "b@s.whatsapp.net" and "text" not in str(st["pending_rows"])


def test_config_numbers_are_validated(env):
    c, cfg, cfgfile, *_ = env
    assert c.put("/api/wa/config", json={"key": "min_messages", "value": 9}).status_code == 200
    assert yaml.safe_load(cfgfile.read_text())["whatsapp"]["min_messages"] == 9
    assert c.put("/api/wa/config", json={"key": "min_messages", "value": 0}).status_code == 400
    assert c.put("/api/wa/config", json={"key": "bridge_url", "value": 1}).status_code == 400


def test_digest_files_and_nudges(env, tmp_path):
    c, cfg, *_ = env
    d = tmp_path / "digests"; d.mkdir()
    (d / "2026-01-01.md").write_text("# Digest")
    assert c.get("/api/wa/digest/files/digests/2026-01-01.md").json()["text"] == "# Digest"
    assert c.get("/api/wa/digest/files/digests/..%2Ft.db").status_code == 404
    assert c.get("/api/wa/digest/files/nope/x.md").status_code == 404
    assert len(c.get("/api/wa/nudges").json()["nudges"]) == 1
    assert c.post("/api/wa/nudges/a@s.whatsapp.net/hide", json={"hours": 1}).status_code == 200
    n = c.get("/api/wa/nudges").json()
    assert n["nudges"] == [] and n["hidden"] == 1


def test_persona_needs_confirm_and_says_nothing_is_sent(env):
    c, *_ = env
    a = "a@s.whatsapp.net"
    assert c.post(f"/api/wa/persona/{a}/jobs", json={"action": "build"}).status_code == 400
    assert c.post(f"/api/wa/persona/{a}/jobs", json={"action": "clone", "confirm": True}).status_code == 409   # no training env
    j = wait(c, c.post(f"/api/wa/persona/{a}/jobs", json={"action": "say", "text": "free tonight?"}).json())
    assert j["result"]["messages"] == ["sure"]
    assert c.post("/api/wa/persona/g@g.us/jobs", json={"action": "say", "text": "x"}).status_code == 400


def test_nothing_in_the_gui_can_send():
    paths = [p for p in app.openapi()["paths"] if p.startswith("/api/wa")]
    assert len(paths) > 30 and not [p for p in paths if "send" in p.lower()]
    import pathlib
    src = "".join(p.read_text() for p in pathlib.Path("agent/web/routes").glob("whatsapp*.py"))
    for banned in (".send(", 'mod("autoreply")', 'mod("responder")', 'mod("watch")'):       # the drivers that can text people
        assert banned not in src
    js = pathlib.Path("agent/web/static/js/screens/whatsapp.js").read_text()
    assert "/send" not in js and "Send message" not in js.replace('"Send messages"', "")
