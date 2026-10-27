"""Code-agent run viewer, project notes and prompt studio routes. Needs the web group."""

from __future__ import annotations

import copy
import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import projectmem, promptlab  # noqa: E402
from agent.codeagent import eventlog, sessionstore  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "w.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: object()
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app), tmp_path
    app.dependency_overrides.clear()


def _save(root, sid="ab12cd34", created=None, state="finished", **extra):
    created = created or time.time() - 600
    data = {"version": 2, "id": sid, "root": str(root), "title": "Add stats", "created": created, "updated": created + 300, "state": state,
            "model": "qwen3.8:27b-text", "step": 14, "num_ctx": 98304, "messages": [{"role": "user", "content": "x"}] * 5,
            "requests": ["add a stats module"], "ledger": {"files": {"stats.py": {"wrote": 1}, "test_stats.py": {"wrote": 1}}}, "pending": {}, **extra}
    sessionstore.save(data)
    return data


def test_sessions_are_listed_and_opened_by_id_prefix(env):
    tc, tmp = env
    root = tmp / "proj"
    root.mkdir()
    assert tc.get("/api/code/sessions").json() == []
    _save(root)
    (row,) = tc.get("/api/code/sessions").json()
    assert row["id"] == "ab12cd34" and row["title"] == "Add stats" and row["steps"] == 14 and row["files"] == 2 and row["state"] == "finished"
    one = tc.get("/api/code/sessions/ab12").json()
    assert one["requests"] == ["add a stats module"] and one["messages"] == 5 and one["num_ctx"] == 98304
    assert set(one["ledger"]["files"]) == {"stats.py", "test_stats.py"}
    assert tc.get("/api/code/sessions/zzzz").status_code == 404


def test_big_ledgers_are_summarised_not_dumped(env):
    tc, tmp = env
    root = tmp / "big"
    root.mkdir()
    _save(root, ledger={"files": {f"f{i}.py": {"note": "x" * 100} for i in range(400)}})
    assert tc.get("/api/code/sessions/ab12").json()["ledger"]["truncated"] is True


def test_events_are_attributed_by_folder_and_time(env):
    tc, tmp = env
    root = tmp / "proj"
    root.mkdir()
    data = _save(root)
    mine = eventlog.EventLog(f"{root.name}-{int(data['created'])}")
    other = eventlog.EventLog("elsewhere-1")
    now = data["created"] + 60
    for log, ts, kind in [(mine, now, "step"), (mine, now + 1, "tool"), (other, now, "step"), (mine, now + 9999, "step")]:
        rec = {"ts": ts, "session": log.session, "kind": kind, "n": 1, "prompt": 100, "gen": 5, "prefill_s": 1.0, "decode_s": 0.5}
        with open(log.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    body = tc.get("/api/code/sessions/ab12/events").json()
    assert body["total"] == 2 and {e["kind"] for e in body["events"]} == {"step", "tool"} and body["timing"] is not None
    assert [e["kind"] for e in tc.get("/api/code/sessions/ab12/events?kind=tool").json()["events"]] == ["tool"]


def test_project_notes_list_and_forget(env):
    tc, tmp = env
    store = Store(str(tmp / "w.db"))
    pm = projectmem.ProjectMemory(store.conn)
    pm.add("/home/x/game", "Check that passes: pytest -q -x", kind="verify")
    pm.add("/home/x/game", "Uses pygame-ce for rendering")
    pm.add("/home/x/other", "Written in Rust with tokio")
    listing = tc.get("/api/projects").json()
    assert {(p["name"], p["notes"]) for p in listing} == {("game", 2), ("other", 1)}
    notes = tc.get("/api/projects/notes", params={"key": "/home/x/game"}).json()
    assert {n["note"] for n in notes} == {"Check that passes: pytest -q -x", "Uses pygame-ce for rendering"}
    assert tc.delete("/api/projects/notes", params={"key": "/home/x/game"}).json() == {"forgotten": 2}
    assert tc.get("/api/projects/notes", params={"key": "/home/x/game"}).json() == []


def test_prompt_studio_calls_promptlab_and_validates(env, monkeypatch):
    tc, _ = env
    seen = {}
    monkeypatch.setattr(promptlab, "improve", lambda llm, cfg, text, answers="": seen.update(improve=(text, answers)) or "BETTER " + text)
    monkeypatch.setattr(promptlab, "anonymize", lambda llm, cfg, text, strength="heavy": seen.update(anon=(text, strength)) or "ANON " + text)
    assert tc.post("/api/prompt/improve", json={"text": "fix my code", "answers": "python"}).json() == {"text": "BETTER fix my code"}
    assert seen["improve"] == ("fix my code", "python")
    assert tc.post("/api/prompt/anonymize", json={"text": "I am Ann", "strength": "light"}).json() == {"text": "ANON I am Ann"}
    assert seen["anon"] == ("I am Ann", "light")
    assert tc.post("/api/prompt/anonymize", json={"text": "x", "strength": "gentle"}).status_code == 422
    assert tc.post("/api/prompt/improve", json={"text": "   "}).status_code == 422
    assert tc.post("/api/prompt/improve", json={"text": "x" * 20_001}).status_code == 422
