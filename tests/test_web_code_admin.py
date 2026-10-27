"""Code agent admin routes: policy + classifier, checkpoints, memory, sessions, stats, evals (fake subprocess)."""

from __future__ import annotations

import copy
import json
import subprocess
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.codeagent import sessionstore  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402
from agent.web.routes import code_admin  # noqa: E402


class Client:
    def list_models(self):
        return ["a"]

    def loaded_models(self):
        return []


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "ev.jsonl"))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "w.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: Client()
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    proj = tmp_path / "home" / "me" / "proj"
    proj.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=n", *a], cwd=proj, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    (proj / "a.txt").write_text("one\n")
    git("add", "-A")
    git("commit", "-q", "-m", "i")
    yield TestClient(app), proj, tmp_path
    app.dependency_overrides.clear()


def test_classifier_and_policy_rules(env):
    tc, proj, _ = env
    r = tc.post("/api/code/policy/classify", json={"command": "ls", "folder": str(proj)}).json()
    assert r["tier"] == "SAFE" and r["verdict"] == "allow"
    r = tc.post("/api/code/policy/classify", json={"command": "rm -rf /"}).json()
    assert r["tier"] == "BLOCKED" and r["why"]
    r = tc.post("/api/code/policy/classify", json={"command": "rm -r build", "folder": str(proj)}).json()
    assert r["tier"] == "DANGEROUS" and r["verdict"] == "ask"
    # loosening needs the typed word, tightening does not
    put = lambda changes, confirm=None: tc.put("/api/code/policy", json={"folder": str(proj), "changes": changes, "confirm": confirm})  # noqa: E731
    assert put({"mode": "auto"}).status_code == 422
    assert put({"mode": "auto"}, "auto").json()["mode"] == "auto"
    assert put({"mode": "ask"}).json()["mode"] == "ask"
    assert put({"allow": ["pytest*"]}).status_code == 422
    assert put({"allow": ["pytest*"]}, "yes").json()["allow"] == ["pytest*"]
    assert put({"max_steps": 1}).status_code == 422
    assert put({"bogus": 1}).status_code == 422
    assert tc.put("/api/code/policy", json={"folder": "/home", "changes": {}}).status_code == 422
    rows = tc.get("/api/code/policy").json()["repos"]
    assert rows[0]["allow"] == ["pytest*"]
    assert tc.delete("/api/code/policy", params={"folder": str(proj)}).json() == {"removed": True}
    assert tc.post("/api/code/policy/greenlist", json={"folder": str(proj)}).status_code == 422


def test_checkpoints_list_diff_restore(env):
    tc, proj, _ = env
    (proj / "a.txt").write_text("two\n")
    from agent import undo

    ref = undo.git_checkpoint(str(proj / "a.txt"))
    assert ref
    (proj / "a.txt").write_text("three\n")
    items = tc.get("/api/code/checkpoints", params={"folder": str(proj)}).json()["items"]
    assert [i["ref"] for i in items] == [ref]
    d = tc.get("/api/code/checkpoints/detail", params={"folder": str(proj), "ref": ref, "against": "now"}).json()
    assert d["files"][0]["path"] == "a.txt" and "-two" in d["diff"]
    assert tc.get("/api/code/checkpoints/detail", params={"folder": str(proj), "ref": "--output=x"}).status_code == 422
    body = {"folder": str(proj), "ref": ref}
    assert tc.post("/api/code/checkpoints/restore", json=body).status_code == 422
    r = tc.post("/api/code/checkpoints/restore", json={**body, "confirm": "restore"}).json()
    assert (proj / "a.txt").read_text() == "two\n" and r["safety_checkpoint"]


def test_memory_notes_and_check(env):
    tc, proj, _ = env
    p = {"folder": str(proj)}
    r = tc.put("/api/code/memory/notes", json={**p, "notes": "tests live in tests/\nuse uv for everything"}).json()
    assert len(r["notes"]) == 2
    r = tc.put("/api/code/memory/notes", json={**p, "notes": "use uv for everything"}).json()
    assert [n["note"] for n in r["notes"]] == ["use uv for everything"]
    assert tc.put("/api/code/memory/check", json={**p, "test_cmd": "pytest -q -x"}).json()["test_cmd"] == "pytest -q -x"
    m = tc.get("/api/code/memory", params=p).json()
    assert m["test_cmd"] == "pytest -q -x" and len(m["notes"]) == 1
    assert tc.delete("/api/code/memory", params=p).json()["forgotten"] == 1
    assert tc.get("/api/code/memory", params=p).json()["test_cmd"] == ""


def _save(root, sid, state="finished", updated=None):
    now = updated or time.time()
    sessionstore.save({"version": 2, "id": sid, "root": str(root), "title": "t", "created": now - 5, "updated": now, "state": state, "model": "m", "step": 3,
                       "messages": [{"role": "user", "content": f"hello from {root}"}], "requests": ["t"], "ledger": {}, "pending": {}})


def test_sessions_delete_prune_export_and_stats(env):
    tc, proj, tmp = env
    _save(proj, "aaaa1111", updated=time.time() - 40 * 86400)
    _save(proj, "bbbb2222")
    assert [m["id"] for m in tc.post("/api/code/sessions/prune", json={"days": 30, "dry_run": True}).json()["would_delete"]] == ["aaaa1111"]
    e = tc.get("/api/code/sessions/bbbb/export")
    rec = json.loads(e.text)
    assert rec["id"] == "bbbb2222" and "/home/user" in rec["messages"][0]["content"] and str(proj) not in e.text
    assert tc.post("/api/code/sessions/export-all", json={}).status_code == 422
    assert [m["id"] for m in tc.post("/api/code/sessions/prune", json={"days": 30}).json()["deleted"]] == ["aaaa1111"]
    assert tc.delete("/api/code/sessions/bbbb").json() == {"deleted": "bbbb2222"}
    assert tc.delete("/api/code/sessions/bbbb").status_code == 404
    s = tc.get("/api/code/stats").json()
    assert s["runs"] == 0 and s["median_tps"] is None
    assert tc.get("/api/code/why-slow").json()["empty"] is True
    assert tc.get("/api/code/stats/speed").json()["empty"] is True


def test_evals_need_confirm_and_use_a_fake_process(env, monkeypatch):
    tc, proj, tmp = env
    monkeypatch.setattr(code_admin, "eval_command", lambda body: ["python3", "-c", "print('pass@1            0.90  (18/20 solved incl. held-out)')"])
    assert tc.post("/api/code/evals", json={"kind": "fix"}).status_code == 422
    jid = tc.post("/api/code/evals", json={"kind": "fix", "confirm": "run"}).json()["id"]
    for _ in range(100):
        j = tc.get(f"/api/code/jobs/{jid}").json()
        if j["state"] != "running":
            break
        time.sleep(0.05)
    assert j["state"] == "done"
    time.sleep(0.1)
    runs = tc.get("/api/code/evals").json()["runs"]
    assert runs[0]["passed"] == 18 and runs[0]["total"] == 20
    assert tc.get(f"/api/code/evals/history/{jid}").json()["log"]
    cmp = tc.get("/api/code/model-compare").json()["rows"]
    assert cmp and cmp[0]["pass_rate"] == 90
    assert tc.post("/api/code/bench", json={"kind": "server"}).status_code == 422


def test_model_swap_keeps_routing_in_sync(env, monkeypatch, tmp_path):
    tc, proj, tmp = env
    cfgfile = tmp / "config.yaml"
    cfgfile.write_text("models:\n  coding: old\n")
    monkeypatch.setattr("agent.config.DEFAULT_CONFIG_PATH", cfgfile)
    m = tc.get("/api/code/models").json()
    assert m["installed"] == ["a"] and m["recommended"]
    tc.put("/api/code/models/coding", json={"model": "new:7b"})
    txt = cfgfile.read_text()
    assert "coding: new:7b" in txt and "model: new:7b" in txt


def test_real_command_builders_are_plain_argument_lists():
    from agent.web.routes import code_admin as ca

    cmd = ca.eval_command(ca.EvalIn(kind="agentic", trials=3))
    assert cmd[-4:] == ["code-eval", "--agentic", "--trials", "3"] and ca.eval_command(ca.EvalIn(seeds=5))[-2:] == ["--seeds", "5"]
    assert ca.bench_command(ca.BenchIn(kind="dry"))[-1] == "--dry-run" and ca.bench_command(ca.BenchIn(confirm="benchmark"))[-1] == "--yes"
