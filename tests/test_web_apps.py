"""Apps screen routes: images, shared files, prompt studio chats and jobs. Fake model, temp folders, no subprocess."""

from __future__ import annotations

import copy
import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import promptlab  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.tools import image_gen  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, model, **kw):
        self.calls.append(messages)
        yield "reply to " + messages[-1]["content"][:20]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "a.db")
    config["models"]["chat"] = "m:1"
    client = FakeClient()
    monkeypatch.setattr(deps, "_config", config)      # get_store() reads the config directly, not through Depends
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: client
    monkeypatch.setattr(promptlab, "_SESS_PATH", tmp_path / "sessions.json")
    monkeypatch.setattr(image_gen, "IMAGE_OUTPUT_DIR", tmp_path / "img")
    monkeypatch.setattr(image_gen, "IMAGE_GEN_CHECKPOINT", tmp_path / "ckpt")
    yield TestClient(app), tmp_path, client, Store(config["data"]["db_path"])
    app.dependency_overrides.clear()


def _wait(tc, jid, base="/api/prompt/jobs/"):
    for _ in range(100):
        j = tc.get(base + jid).json()
        if j["status"] != "running":
            return j
        time.sleep(0.03)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------------------------------------ shared files


def test_shared_files_list(env):
    tc, tmp, _, store = env
    sid = store.create_session("s") if hasattr(store, "create_session") else None
    real = tmp / "r.pdf"
    real.write_bytes(b"1234")
    store.add_file_share(sid, str(real), display_name="report.pdf")
    store.add_file_share(sid, str(tmp / "old.png"), disposition="inline")
    rows = tc.get("/api/apps/shared-files").json()
    assert [r["name"] for r in rows] == ["old.png", "report.pdf"]
    assert rows[0]["exists"] is False and rows[0]["kind"] == "image" and rows[1]["size"] == 4 and rows[1]["url"].startswith("/api/files/")
    assert "path" not in rows[1]


# ------------------------------------------------------------------------------------------------ images


def test_images_info_gallery_queue_and_generate(env, monkeypatch):
    tc, tmp, _, store = env
    assert tc.get("/api/images/info").json()["installed"] is False
    sid = store.create_session("s")
    (tmp / "img").mkdir()
    png = tmp / "img" / "abc.png"
    png.write_bytes(b"png")
    store.add_file_share(sid, str(png), disposition="inline", display_name="a lantern in rain.png")
    store.add_file_share(sid, str(tmp / "elsewhere.png"), disposition="inline")
    g = tc.get("/api/images/gallery").json()
    assert len(g) == 1 and g[0]["title"] == "a lantern in rain" and g[0]["exists"] and g[0]["url"].startswith("/api/files/")
    image_gen._JOBS["j1"] = {"status": "running", "prompt": "snow", "started_at": time.time() - 10, "eta_s": 100.0, "error": None, "session_id": sid}
    try:
        q = tc.get("/api/images/queue").json()
        assert q[0]["id"] == "j1" and 5 <= q[0]["pct"] <= 20
        assert tc.get("/api/images/j1").json()["status"] == "running"        # the existing route still works after the new fixed ones
    finally:
        image_gen._JOBS.pop("j1", None)
    assert tc.post("/api/images/generate", json={"prompt": "x", "session_id": sid}).status_code == 409     # model not installed
    (tmp / "ckpt").mkdir()
    started = []
    monkeypatch.setattr(image_gen, "start_generation", lambda *a, **k: started.append((a[0], k)) or {"job_id": "z", "eta_s": 9.0})
    assert tc.post("/api/images/generate", json={"prompt": "x", "session_id": 9999}).status_code == 404
    assert tc.post("/api/images/generate", json={"prompt": "x", "session_id": sid, "size": 100}).status_code == 422
    r = tc.post("/api/images/generate", json={"prompt": " a lamp ", "session_id": sid, "size": 768})
    assert r.json()["job_id"] == "z" and started == [("a lamp", {"height": 768, "width": 768})]
    assert tc.post("/api/images/generate", json={"prompt": "again", "session_id": sid}).status_code == 409      # one at a time
    image_gen._GEN_SEMAPHORE.release()


# ------------------------------------------------------------------------------------------------ prompt studio


def test_prompt_jobs_ops_and_validation(env):
    tc, _, client, _ = env
    assert tc.post("/api/prompt/jobs", json={"op": "nope", "text": "x"}).status_code == 422
    assert tc.post("/api/prompt/jobs", json={"op": "improve", "text": " "}).status_code == 422
    assert tc.post("/api/prompt/jobs", json={"op": "refine", "current": "p"}).status_code == 422
    j = _wait(tc, tc.post("/api/prompt/jobs", json={"op": "improve", "text": "make a game"}).json()["job"])
    assert j["status"] == "done" and j["result"]["text"].startswith("reply to")
    j = _wait(tc, tc.post("/api/prompt/jobs", json={"op": "refine", "current": "old prompt", "feedback": "shorter"}).json()["job"])
    assert "shorter" in client.calls[-1][0]["content"] and "old prompt" in client.calls[-1][0]["content"]
    j = _wait(tc, tc.post("/api/prompt/jobs", json={"op": "anon", "text": "I am Anna", "strength": "max"}).json()["job"])
    assert [t["pass"] for t in j["result"]["trace"]] == ["core", "keep", "strip", "draft"] and len(client.calls) >= 7
    assert tc.get("/api/prompt/jobs/zzz").status_code == 404


def test_prompt_chat_is_saved_resumed_and_deleted(env):
    tc, _, _, _ = env
    r = _wait(tc, tc.post("/api/prompt/jobs", json={"op": "chat", "mode": "improve", "message": "a nix flake review prompt"}).json()["job"])["result"]
    sid = r["sid"]
    assert tc.get("/api/prompt/sessions").json()[0]["title"].startswith("a nix flake")
    r2 = _wait(tc, tc.post("/api/prompt/jobs", json={"op": "chat", "sid": sid, "message": "make it shorter"}).json()["job"])["result"]
    assert r2["sid"] == sid
    d = tc.get(f"/api/prompt/sessions/{sid}").json()
    assert [m["role"] for m in d["messages"]] == ["user", "assistant", "user", "assistant"]
    assert tc.post("/api/prompt/jobs", json={"op": "chat", "sid": 99, "message": "x"}).status_code == 404
    assert tc.delete(f"/api/prompt/sessions/{sid}").json() == {"ok": True}
    assert tc.get(f"/api/prompt/sessions/{sid}").status_code == 404 and tc.get("/api/prompt/sessions").json() == []


def test_stopped_job_hides_its_result(env):
    tc, _, client, _ = env
    gate = threading.Event()

    def slow(messages, model, **kw):
        gate.wait(2)
        yield "late"
    client.chat = slow
    jid = tc.post("/api/prompt/jobs", json={"op": "improve", "text": "x"}).json()["job"]
    tc.post(f"/api/prompt/jobs/{jid}/stop")
    gate.set()
    j = _wait(tc, jid)
    assert j["status"] == "stopped" and j["result"] is None
