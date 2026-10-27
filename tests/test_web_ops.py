"""Status detail and code-run timeline routes (read-only). Needs the web group."""

from __future__ import annotations

import copy
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.codeagent import sessionstore  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402
from agent.web.routes import ops  # noqa: E402


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "w.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    monkeypatch.setattr(ops.governor, "pause_reason", lambda c: None)
    monkeypatch.setattr(ops.governor, "game_running", lambda: None)
    monkeypatch.setattr(ops.worker_status, "read_activity", lambda c: None)
    monkeypatch.setattr(ops, "_systemctl", lambda *a: "inactive")
    monkeypatch.setattr(ops, "_probe", lambda url: {"reachable": False, "models": []})
    yield TestClient(app), tmp_path
    app.dependency_overrides.clear()


def test_status_detail_idle_machine(env):
    tc, _ = env
    d = tc.get("/api/status/detail").json()
    assert d["worker"]["state"] == "stopped" and d["worker"]["reason"] is None
    assert [s["id"] for s in d["servers"]] == ["default", "dual-gpu"] and not any(s["reachable"] for s in d["servers"])
    assert d["game"] is None and d["paused_reason"] is None


def test_status_detail_game_pauses_a_running_worker(env, monkeypatch):
    tc, _ = env
    monkeypatch.setattr(ops, "_systemctl", lambda *a: "active")
    monkeypatch.setattr(ops.governor, "pause_reason", lambda c: "game running (proton)")
    monkeypatch.setattr(ops.governor, "game_running", lambda: "proton")
    monkeypatch.setattr(ops.worker_status, "read_activity", lambda c: ("syncing whatsapp", 12))
    monkeypatch.setattr(ops, "_probe", lambda url: {"reachable": True, "models": [{"name": "qwen:27b", "vram_mb": 20000}]})
    w = tc.get("/api/status/detail").json()
    assert w["worker"]["state"] == "paused" and w["worker"]["reason"] == "game running (proton)"
    assert w["worker"]["activity"] == "syncing whatsapp" and w["game"] == "proton"
    assert w["servers"][0]["models"][0]["name"] == "qwen:27b"


def test_status_detail_without_systemd_is_unknown_not_an_error(env, monkeypatch):
    tc, _ = env
    monkeypatch.setattr(ops, "_systemctl", lambda *a: "")
    assert tc.get("/api/status/detail").json()["worker"]["state"] == "unknown"


def test_probe_reports_unreachable_server(monkeypatch):
    assert ops._probe("http://127.0.0.1:9") == {"reachable": False, "models": []}


def _save(root, sid="ab12cd34", **extra):
    created = time.time() - 3000
    data = {"version": 2, "id": sid, "root": str(root), "title": "Add stats", "created": created, "updated": created + 2700, "state": "interrupted",
            "model": "m", "step": 14, "num_ctx": 1000, "messages": [{"role": "user", "content": "x"}], "requests": ["add stats"],
            "ctx_state": {"last_prompt": 400},
            "ledger": {"files": {"stats.py": {"action": "wrote", "step": 2, "lines": 40, "size": 900, "names": []}},
                       "runs": [[5, "pytest -q", 1, "2 failed"], [9, "pytest -q", 0, ""]]},
            "plan": {"task": "t", "nudges": 0, "items": [{"id": 1, "text": "a", "done": True}, {"id": 2, "text": "b", "done": False}]},
            "pending": {"stopped": "budget", "task": "add stats"}, **extra}
    sessionstore.save(data)
    return data


def test_timeline_of_a_stopped_run(env, monkeypatch):
    tc, tmp = env
    root = tmp / "proj"
    root.mkdir()
    data = _save(root)
    import json
    ev = [{"ts": data["created"] + 60, "session": "proj-1", "kind": "check", "status": "fail", "check_kind": "tests", "id": "c1"},
          {"ts": data["created"] + 90, "session": "proj-1", "kind": "step", "n": 5}]
    (tmp / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    t = tc.get("/api/code/sessions/ab12/timeline").json()
    assert t["summary"]["edits"] == 1 and t["summary"]["failed_checks"] == 1 and t["summary"]["failed_commands"] == 1
    assert t["summary"]["progress"] == {"label": "Acceptance items", "done": 1, "total": 2}
    assert t["summary"]["context"] == {"used": 400, "limit": 1000}
    assert t["decision"]["stopped"] == "budget"
    titles = [i["title"] for i in t["items"]]
    assert {"Wrote", "Check failed", "Command failed", "Command passed"} <= set(titles)
    assert [i["ts"] for i in t["items"]] == sorted(i["ts"] for i in t["items"])
    assert tc.get("/api/code/sessions/zzzz/timeline").status_code == 404


def test_finished_run_has_no_pending_decision(env):
    tc, tmp = env
    root = tmp / "p2"
    root.mkdir()
    _save(root, state="finished", pending={})
    assert tc.get("/api/code/sessions/ab12/timeline").json()["decision"] is None
