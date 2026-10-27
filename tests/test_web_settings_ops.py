"""Settings safety guard and the Operations routes (governor, worker control, logs, backup/sync report). Needs the web group."""

from __future__ import annotations

import copy
import json
import subprocess

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import settings as S  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402
from agent.web.routes import ops, ops_control  # noqa: E402


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({}))
    monkeypatch.setattr(S, "DEFAULT_CONFIG_PATH", cfg_file)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "w.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    monkeypatch.setattr(ops_control.governor, "pause_reason", lambda c: None)
    monkeypatch.setattr(ops_control.governor, "game_running", lambda: None)
    monkeypatch.setattr(ops_control.governor, "gpu_utilization", lambda: 12)
    monkeypatch.setattr(ops, "_systemctl", lambda *a: "inactive")
    monkeypatch.setattr(ops_control, "_journal", lambda unit, user, lines=40: [{"ts": 1.0, "lv": "E", "t": "boom"}, {"ts": 2.0, "lv": "I", "t": "all fine"}, {"ts": 3.0, "lv": "W", "t": "slow"}])
    yield TestClient(app), config, cfg_file, tmp_path
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------------------- settings: safety guard


def test_every_editable_setting_is_listed_with_type_default_and_flags(env):
    tc, *_ = env
    items = {i["key"]: i for i in tc.get("/api/settings").json()}
    assert set(items) == set(S.EDITABLE)
    assert items["embed_model"]["restart_required"] and not items["recall_k"]["restart_required"]
    assert items["safe_commands"]["loosens_safety"] and not items["recall_k"]["loosens_safety"]
    assert "capped" in items["chat_context_window"]["note"]


def test_adding_a_safe_command_needs_confirmation(env):
    tc, config, cfg_file, _ = env
    r = tc.put("/api/settings/safe_commands", json={"value": ["make"]})
    assert r.status_code == 409 and config["safety"]["safe_commands"] == []
    ok = tc.put("/api/settings/safe_commands", json={"value": ["make", " make ", ""], "confirm": True})
    assert ok.status_code == 200 and config["safety"]["safe_commands"] == ["make"]
    assert yaml.safe_load(cfg_file.read_text())["safety"]["safe_commands"] == ["make"]
    # tightening again needs no confirmation
    assert tc.put("/api/settings/safe_commands", json={"value": []}).status_code == 200


def test_removing_a_blocked_pattern_or_raising_the_timeout_needs_confirmation(env):
    tc, config, *_ = env
    assert tc.put("/api/settings/blocked_patterns", json={"value": ["rm -rf"]}).status_code == 200      # adding a block is a tightening
    assert tc.put("/api/settings/blocked_patterns", json={"value": []}).status_code == 409
    assert config["safety"]["blocked_patterns"] == ["rm -rf"]
    assert tc.put("/api/settings/safety_max_timeout_s", json={"value": 600}).status_code == 409
    assert tc.put("/api/settings/safety_max_timeout_s", json={"value": 60}).status_code == 200
    assert tc.put("/api/settings/safety_max_timeout_s", json={"value": 100}).status_code == 409


def test_reset_that_loosens_safety_is_guarded_too(env):
    tc, config, *_ = env
    tc.put("/api/settings/blocked_patterns", json={"value": ["x"]})
    assert tc.post("/api/settings/blocked_patterns/reset").status_code == 409
    assert tc.post("/api/settings/blocked_patterns/reset?confirm=true").json()["value"] == []


def test_list_settings_reject_bad_entries(env):
    tc, *_ = env
    assert tc.put("/api/settings/blocked_patterns", json={"value": "not a list"}).status_code == 422
    assert tc.put("/api/settings/blocked_patterns", json={"value": ["a\nb"]}).status_code == 422
    assert tc.put("/api/settings/recall_k", json={"value": {"a": 1}}).status_code == 422


# ---------------------------------------------------------------------------------------- operations


def test_governor_reports_state_and_caps(env):
    tc, *_ = env
    g = tc.get("/api/ops/governor").json()
    assert g["gpu_percent"] == 12 and g["caps"]["gpu"] == DEFAULT_CONFIG["background"]["max_gpu_percent"] and g["paused_reason"] is None


def test_worker_control_runs_systemctl_user_and_reports_failures(env, monkeypatch):
    tc, *_ = env
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ops_control.subprocess, "run", fake_run)
    assert tc.post("/api/ops/worker/start").status_code == 200
    assert calls == [["systemctl", "--user", "start", "agent-watch.service"]]
    assert tc.post("/api/ops/worker/restart").status_code == 404
    monkeypatch.setattr(ops_control.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 5, "", "Unit not found"))
    r = tc.post("/api/ops/worker/stop")
    assert r.status_code == 502 and "Unit not found" in r.json()["detail"]


def test_worker_state_includes_journal(env):
    tc, *_ = env
    w = tc.get("/api/ops/worker").json()
    assert w["state"] == "stopped" and w["journal"][0]["t"] == "boom"


def test_logs_filter_by_level_and_search_and_reject_unknown_sources(env):
    tc, *_ = env
    assert [l["t"] for l in tc.get("/api/ops/logs?level=E").json()["lines"]] == ["boom"]
    assert [l["t"] for l in tc.get("/api/ops/logs?level=W").json()["lines"]] == ["boom", "slow"]
    assert [l["t"] for l in tc.get("/api/ops/logs?q=FINE").json()["lines"]] == ["all fine"]
    assert tc.get("/api/ops/logs?source=../../etc/passwd").status_code == 404
    assert tc.get("/api/ops/logs?level=x").status_code == 422


def test_journal_parser_maps_priorities(monkeypatch):
    out = "\n".join(json.dumps(e) for e in [{"MESSAGE": "a", "PRIORITY": "3", "__REALTIME_TIMESTAMP": "2000000"}, {"MESSAGE": [1, 2]}, {"MESSAGE": "b", "PRIORITY": "4"}]) + "\nnot json"
    monkeypatch.setattr(ops_control.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, out, ""))
    rows = ops_control._journal("u", True)
    assert [(r["lv"], r["t"]) for r in rows] == [("E", "a"), ("W", "b")] and rows[0]["ts"] == 2.0


def test_backup_and_sync_are_report_only(env, monkeypatch):
    tc, config, _, tmp = env
    (tmp / "w.db").write_bytes(b"x" * 2048)
    b = tc.get("/api/ops/backup").json()
    assert b["available"] is False and any(f["path"].endswith("w.db") and f["bytes"] == 2048 for f in b["files"])
    bindir = tmp / "bin"
    bindir.mkdir()
    (bindir / "x-pull-db.sh").write_text("#!/bin/sh\n")
    monkeypatch.setenv("XDG_BIN_HOME", str(bindir))
    s = tc.get("/api/ops/sync").json()
    assert s["available"] and [x["name"] for x in s["scripts"]] == ["x-pull-db.sh"] and s["database"]["exists"]
    monkeypatch.setenv("XDG_BIN_HOME", str(tmp / "none"))
    assert tc.get("/api/ops/sync").json()["available"] is False
