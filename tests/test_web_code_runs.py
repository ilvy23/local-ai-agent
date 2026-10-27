"""Goal runs of the code agent from the GUI: start, approval, stop, validation. Fake model, temp folders, no real run."""

from __future__ import annotations

import copy
import io
import subprocess
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from rich.console import Console  # noqa: E402

from agent.codeagent import policy as P  # noqa: E402
from agent.codeagent.engine import CodeEngine  # noqa: E402
from agent.codeagent.eventlog import EventLog  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402
from agent.web.routes import code_runs  # noqa: E402


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}], "_prompt_eval_count": 100, "_eval_count": 10}


def say(t):
    return {"role": "assistant", "content": t, "_prompt_eval_count": 100, "_eval_count": 10}


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)

    def chat_with_tools(self, messages, model, tools, num_ctx=None, think=None, extra_options=None):
        return dict(self.script.pop(0)) if self.script else say("done")

    def chat(self, *a, **k):
        yield "NONE"

    def loaded_models(self, vram_only=False):
        return []

    def list_models(self):
        return ["m1", "m2"]

    def unload_all(self):
        return []


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "ev.jsonl"))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "w.db")
    config["coding"] = {**config.get("coding", {}), "claims_check": False, "forced_summary": False}
    config["models"]["chat"] = "m1"
    llm = FakeLLM([])
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: llm
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    proj = tmp_path / "home" / "me" / "proj"
    proj.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)

    def build(run, root, cfg, client, console):
        eng = CodeEngine(root, cfg, llm, console, approve_fn=lambda d, r: code_runs._ask_human(run, d, r), interactive=False, events=EventLog("t", tmp_path / "ev.jsonl"))
        return eng, Store(config["data"]["db_path"])

    monkeypatch.setattr(code_runs, "_build_engine", build)
    monkeypatch.setattr("agent.codeagent.session.free_gpu", lambda *a, **k: None)
    monkeypatch.setattr("agent.codeagent.session.release_gpu_on_exit", lambda *a, **k: None)
    code_runs._RUNS.clear()
    yield TestClient(app), proj, llm
    for r in list(code_runs._RUNS.values()):
        r.stop_flag.set(); r.answered.set(); r.resume_flag.set()
        if r.thread:
            r.thread.join(5)
    code_runs._RUNS.clear()
    app.dependency_overrides.clear()


def wait(tc, rid, states, t=15):
    end = time.time() + t
    while time.time() < end:
        r = tc.get(f"/api/code/runs/{rid}").json()
        if r["state"] in states:
            return r
        time.sleep(0.05)
    raise AssertionError(f"run stays {r['state']}: {r['log'][-5:]}")


def test_a_run_writes_a_file_and_reports(env):
    tc, proj, llm = env
    P.save_repo(proj, mode="auto")
    llm.script = [call("write_file", path="a.py", content="x = 1\n"), say("Created a.py.")]
    rid = tc.post("/api/code/runs", json={"folder": str(proj), "goal": "make a.py"}).json()["id"]
    r = wait(tc, rid, {"done", "partial", "failed", "stopped"})
    assert r["state"] == "done" and r["result"]["files"] == ["a.py"] and r["session_id"]
    assert (proj / "a.py").read_text() == "x = 1\n"
    assert tc.get(f"/api/code/sessions/{r['session_id']}/timeline").status_code == 200


def test_ask_mode_blocks_until_the_gui_decides_and_danger_needs_yes(env):
    tc, proj, llm = env
    P.save_repo(proj, mode="ask")
    llm.script = [call("run", command="rm -r build"), say("ok")]
    rid = tc.post("/api/code/runs", json={"folder": str(proj), "goal": "clean"}).json()["id"]
    r = wait(tc, rid, {"waiting"})
    assert r["approval"]["risk_level"] == "DANGEROUS"
    assert tc.post(f"/api/code/runs/{rid}/decide", json={"approve": True}).status_code == 422
    assert tc.post(f"/api/code/runs/{rid}/decide", json={"approve": False}).json() == {"decided": False}
    assert wait(tc, rid, {"done", "partial"})["approval"] is None
    assert tc.post(f"/api/code/runs/{rid}/decide", json={"approve": False}).status_code == 409


def test_stop_ends_a_waiting_run(env):
    tc, proj, llm = env
    P.save_repo(proj, mode="ask")
    llm.script = [call("run", command="pip install rich"), say("ok")]
    rid = tc.post("/api/code/runs", json={"folder": str(proj), "goal": "x"}).json()["id"]
    wait(tc, rid, {"waiting"})
    tc.post(f"/api/code/runs/{rid}/stop")
    wait(tc, rid, {"stopped", "done", "partial"})
    for _ in range(100):
        if tc.get(f"/api/code/runs/{rid}").json()["ended"]:
            break
        time.sleep(0.05)
    assert tc.get(f"/api/code/runs/{rid}").json()["ended"]


def test_only_one_run_and_validation(env):
    tc, proj, llm = env
    P.save_repo(proj, mode="ask")
    llm.script = [call("run", command="pip install rich")]
    rid = tc.post("/api/code/runs", json={"folder": str(proj), "goal": "x"}).json()["id"]
    wait(tc, rid, {"waiting"})
    assert tc.post("/api/code/runs", json={"folder": str(proj), "goal": "y"}).status_code == 409
    assert tc.get("/api/code/runs").json()[0]["id"] == rid


def test_start_validation(env):
    tc, proj, _ = env
    post = lambda **b: tc.post("/api/code/runs", json={"folder": str(proj), "goal": "g", **b})  # noqa: E731
    assert post(mode="auto").status_code == 422                      # more access than the folder has: typed word
    assert post(mode="bogus").status_code == 422
    assert post(effort="huge", engine="classic").status_code == 422
    assert post(protect=["a\nb"]).status_code == 422
    assert post(files=["../x"]).status_code == 422
    assert post(engine="classic", autopilot=True).status_code == 422
    assert tc.post("/api/code/runs", json={"folder": "/home", "goal": "g"}).status_code == 422
    assert tc.post("/api/code/runs", json={"folder": str(proj), "goal": ""}).status_code == 422
    assert post(mode="auto", confirm_access="auto", remember=False, max_steps=5).status_code == 200


def test_options_and_defaults(env, monkeypatch, tmp_path):
    tc, proj, _ = env
    o = tc.get("/api/code/options", params={"folder": str(proj)}).json()
    assert o["installed"] == ["m1", "m2"] and o["coding"]["effort"] in ("min", "mid", "max") and o["modes"] == ["ask", "medium", "auto"]
    monkeypatch.setattr("agent.config.DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    r = tc.put("/api/code/defaults", json={"values": {"effort": "min", "protect": ["*.pem"], "budget_max_actions": 50}}).json()
    assert r["coding"]["effort"] == "min" and r["coding"]["protect"] == ["*.pem"] and r["coding"]["budget"]["max_actions"] == 50
    assert "effort: min" in (tmp_path / "config.yaml").read_text()
    assert tc.put("/api/code/defaults", json={"values": {"nope": 1}}).status_code == 422
    assert tc.put("/api/code/defaults", json={"values": {"effort": "x"}}).status_code == 422


def test_classic_run_uses_fake_builder(env, monkeypatch):
    tc, proj, _ = env
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=n", "commit", "-q", "--allow-empty", "-m", "i"], cwd=proj, check=True)
    from agent.coding.session import Outcome

    seen = {}

    def fake_build(root, goal, config=None, llm=None, events=None, **k):
        seen["effort"] = config["coding"]["build"]["effort"]
        events.emit("build_round", n=1, of=8, phase="generate")
        events.emit("edit", path="app.py")
        return Outcome(ok=True, stage="built", diff="+x", branch="agent-code/1", attempts=1)

    monkeypatch.setattr("agent.coding.build.build", fake_build)
    rid = tc.post("/api/code/runs", json={"folder": str(proj), "goal": "build", "engine": "classic", "effort": "max"}).json()["id"]
    r = wait(tc, rid, {"done", "partial", "failed"})
    assert r["state"] == "done" and seen["effort"] == "max" and any(i["kind"] == "edit" for i in r["items"]) and "agent-code/1" in r["summary"]
    assert tc.get(f"/api/code/runs/{rid}/diff").json()["diff"] == "+x"


def test_diff_includes_new_files(env):
    tc, proj, _ = env
    (proj / "new.py").write_text("x = 1\n")
    d = code_runs.git_diff(proj)
    assert "diff --git" in d and "+x = 1" in d
