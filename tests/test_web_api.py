"""GUI API: settings, memory, skills, audit, status, GPU mode, folder access, source-trust rules. Needs the web group."""

from __future__ import annotations

import copy

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import gpumode, repoaccess, settings as S  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.research import trust  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


class FakeClient:
    def __init__(self):
        self.embedded = []

    def embed(self, texts, model, **kw):
        self.embedded.append(texts)
        return [[0.1] * 8 for _ in texts]

    def loaded_models(self):
        return ["qwen3.8:27b-text"]

    def vram_in_use_mb(self):
        return 0

    def installed_sizes(self):
        return {"qwen3.8:27b-text": 17000}

    def model_facts(self, model):
        return {"size_mb": 17000, "layers": 64, "approx": False, "kv_elems_per_token": 40000, "ctx_max": 262144}


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"persona": {"name": "Agent"}}))
    monkeypatch.setattr(S, "DEFAULT_CONFIG_PATH", cfg_file)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "web.db")
    client = FakeClient()
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: client
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    repoaccess.reset_session()
    yield TestClient(app), config, client, cfg_file
    app.dependency_overrides.clear()
    repoaccess.reset_session()
    gpumode.reset()


# ---------------------------------------------------------------------------------------- settings


def test_settings_list_has_everything_the_cli_has(env):
    tc, *_ = env
    items = tc.get("/api/settings").json()
    assert {i["key"] for i in items} == set(S.EDITABLE)
    chat = next(i for i in items if i["key"] == "chat_model")
    assert chat["kind"] == "model" and chat["category"] == "Models" and chat["default"] == DEFAULT_CONFIG["models"]["chat"]


def test_setting_change_is_validated_saved_and_applied_to_the_running_server(env):
    tc, config, _, cfg_file = env
    r = tc.put("/api/settings/tools_max_iterations", json={"value": 80})
    assert r.status_code == 200 and r.json()["value"] == 80 and config["tools"]["max_iterations"] == 80
    assert yaml.safe_load(cfg_file.read_text())["tools"]["max_iterations"] == 80
    bad = tc.put("/api/settings/tools_max_iterations", json={"value": 100000})
    assert bad.status_code == 422 and "between" in bad.json()["detail"] and config["tools"]["max_iterations"] == 80
    assert tc.put("/api/settings/nope", json={"value": 1}).status_code == 404
    flag = tc.put("/api/settings/unload_on_exit", json={"value": False})
    assert flag.json()["value"] is False and config["ollama"]["unload_on_exit"] is False


def test_setting_reset_restores_the_default(env):
    tc, config, *_ = env
    tc.put("/api/settings/tools_max_iterations", json={"value": 80})
    r = tc.post("/api/settings/tools_max_iterations/reset")
    assert r.json()["value"] == DEFAULT_CONFIG["tools"]["max_iterations"] == config["tools"]["max_iterations"]


# ---------------------------------------------------------------------------------------- memory, skills, audit


def test_facts_can_be_added_listed_filtered_and_forgotten(env):
    tc, _, client, _ = env
    a = tc.post("/api/memory/facts", json={"content": "likes black metal", "kind": "personal"}).json()
    tc.post("/api/memory/facts", json={"content": "The Eiffel Tower was completed in 1889. (web, 2026-10-18, toureiffel.paris)", "kind": "incidental"})
    assert a["kind"] == "personal" and a["origin"] == "user" and client.embedded
    allf = tc.get("/api/memory/facts").json()
    assert [f["origin"] for f in allf] == ["web", "user"]
    assert [f["content"] for f in tc.get("/api/memory/facts?kind=personal").json()] == ["likes black metal"]
    assert len(tc.get("/api/memory/facts?q=eiffel").json()) == 1
    assert tc.delete(f"/api/memory/facts/{a['id']}").json() == {"ok": True}
    assert tc.delete(f"/api/memory/facts/{a['id']}").status_code == 404
    assert tc.post("/api/memory/facts", json={"content": "  "}).status_code == 422
    assert tc.post("/api/memory/facts", json={"content": "x", "kind": "weird"}).status_code == 422


def test_skills_and_audit(env):
    tc, config, *_ = env
    store = Store(config["data"]["db_path"])
    sid = store.add_skill("Fix pip", "pip fails", ["use uv"], ["run_command"], ["python"])
    store.add_audit_log(kind="shell", detail="ls -la", approved=1, result="ok")
    store.add_audit_log(kind="tool", detail="write_file x", approved=0, result="User denied execution.")
    assert [r["detail"] for r in tc.get("/api/audit?kind=shell").json()] == ["ls -la"]
    assert len(tc.get("/api/audit?q=denied").json()) == 1
    assert tc.get("/api/skills").json()[0]["title"] == "Fix pip"
    assert tc.get(f"/api/skills/{sid}").json()["solution_steps"] == ["use uv"]
    assert tc.delete(f"/api/skills/{sid}").json() == {"ok": True}
    assert tc.get(f"/api/skills/{sid}").status_code == 404


# ---------------------------------------------------------------------------------------- status, gpu, access, trust


def test_status_reports_loaded_models_and_never_fails(env, monkeypatch):
    tc, *_ = env
    monkeypatch.setattr("agent.governor.game_running", lambda: "chess")
    monkeypatch.setattr("agent.governor.pause_reason", lambda cfg, **k: "game running")
    body = tc.get("/api/status").json()
    assert body["loaded_models"] == ["qwen3.8:27b-text"] and body["game_detected"] and body["paused_reason"] == "game running"


def test_gpu_table_and_choice(env, monkeypatch):
    tc, *_ = env
    gpus = [gpumode.Gpu("RX 7800 XT 16 GB", "amd", 16384, 500), gpumode.Gpu("RTX 4060 Ti 8 GB", "nvidia", 8192, 400)]
    monkeypatch.setattr(gpumode, "detect_gpus", lambda: gpus)
    monkeypatch.setattr(gpumode, "_remember", lambda mode: None)
    body = tc.get("/api/gpu").json()
    assert [o["mode"] for o in body["options"]] == ["single", "dual"] and body["table"][0]["model"] == "qwen3.8:27b-text"
    single, dual = body["table"][0]["estimates"]
    assert single["fits"] is False and dual["fits"] is True and dual["ctx"] > 4096
    assert tc.put("/api/gpu", json={"mode": "dual", "ctx": 65536}).json() == {"mode": "dual", "ctx": 65536}
    assert gpumode.current_mode() == "dual" and gpumode.chat_ctx_override() == 65536
    assert tc.put("/api/gpu", json={"mode": "quad"}).status_code == 422


def test_folder_access_levels_and_refusals(env, tmp_path):
    tc, *_ = env
    work = tmp_path / "proj"
    work.mkdir()
    r = tc.put("/api/access", json={"path": str(work), "level": "medium"})
    assert r.status_code == 200 and r.json()["session"][0]["level"] == "medium"
    assert tc.put("/api/access", json={"path": str(work), "level": "greenlist"}).json()["greenlist"] == [str(work)]
    assert tc.put("/api/access", json={"path": str(work), "level": "ungreenlist"}).json()["greenlist"] == []
    assert tc.put("/api/access", json={"path": "/etc", "level": "auto"}).status_code == 422          # system folders are never granted
    assert tc.put("/api/access", json={"path": str(work), "level": "root"}).status_code == 422


def test_trust_rules_round_trip_and_change_the_rating(env):
    tc, *_ = env
    assert tc.get("/api/trust/rules").json() == {"allow": [], "deny": [], "tiers": {}}
    doc = tc.put("/api/trust/rules", json={"domain": "https://www.slimy.example/page", "action": "deny"}).json()
    assert doc["deny"] == ["slimy.example"]
    assert trust.assess("https://slimy.example/x", rules=trust.load_rules(trust.RULES_PATH)).level == "avoid"
    doc = tc.put("/api/trust/rules", json={"domain": "slimy.example", "action": "tier", "tier": "T2"}).json()
    assert doc["deny"] == [] and doc["tiers"] == {"slimy.example": "T2"}
    assert tc.put("/api/trust/rules", json={"domain": "slimy.example", "action": "clear"}).json() == {"allow": [], "deny": [], "tiers": {}}
    assert tc.put("/api/trust/rules", json={"domain": "nodot", "action": "deny"}).status_code == 422
    assert tc.put("/api/trust/rules", json={"domain": "a.org", "action": "tier", "tier": "T9"}).status_code == 422
