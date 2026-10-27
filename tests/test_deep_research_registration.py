"""Wiring smoke tests for Phase 4 (deep research): the CLI command exists and runs end to end on fakes, and
the `deep_research` chat tool is actually registered (web registry + terminal registry) with a safe risk
tier - the thing that's easy to silently break when another agent edits the same shared files
(agent/main.py, agent/web/deps.py, agent/tui.py) concurrently.
"""

from __future__ import annotations

import copy
import json

import pytest

from agent.config import DEFAULT_CONFIG
from agent.research import deep


class FakeClient:
    """Enough of OllamaClient's surface for `make_complete` and the `/api/tools` listing."""

    def __init__(self, replies: dict[str, str] | None = None):
        self.replies = replies or {}

    def chat(self, messages, model, background=False, **_kwargs):
        prompt = messages[-1]["content"]
        for needle, reply in self.replies.items():
            if needle in prompt:
                yield reply
                return
        yield json.dumps({"answer": "fake answer", "new_questions": []})

    def embed(self, texts, model, **kw):
        return [[1.0, 0.0] for _ in texts]


def _fake_make_web_tools(pages: dict[str, list[dict]]):
    from agent.research import evidence as turn_evidence
    from agent.tools.registry import Tool

    def search_handler(query: str, num_results: int = 5, read: bool = True, **_kwargs) -> str:
        items = pages.get(query, [])
        turn_evidence.set_last_search(items, {"label": "likely" if items else "low_trust", "summary": "", "conflicts": []})
        return f'Web search "{query}": {len(items)} sources.'

    def fetch_handler(url: str, focus: str = "", max_chars: int = 2500, **_kwargs) -> str:
        return f"(deep read of {url})"

    search_tool = Tool(name="web_search", description="d", parameters={"type": "object", "properties": {}}, handler=search_handler)
    fetch_tool = Tool(name="web_fetch", description="d", parameters={"type": "object", "properties": {}}, handler=fetch_handler)

    def make_web_tools(console, memory=None):
        return [search_tool, fetch_tool]
    return make_web_tools


# ----------------------------------------------------------------------------------------------------------- CLI


def test_cli_research_command_runs_end_to_end_on_fakes(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from agent.main import app

    pages = {"what is testland?": [{"n": 1, "url": "https://a.example/x", "domain": "a.example", "title": "t",
                                    "date": "", "trust": {"label": "T2 medium", "level": "medium"}, "status": "ok",
                                    "passage": "Testland is a fictional country used in examples."}]}
    client = FakeClient({"what is testland?": json.dumps({"answer": "Testland is fictional [1].", "new_questions": []})})

    monkeypatch.setattr("agent.main._open_llm", lambda: client)
    monkeypatch.setattr("agent.main._ask_gpu_mode", lambda: None)
    monkeypatch.setattr("agent.tools.web.make_web_tools", _fake_make_web_tools(pages))
    monkeypatch.chdir(tmp_path)

    res = CliRunner().invoke(app, ["research", "what is testland?", "--budget-chars", "5000"])
    assert res.exit_code == 0, res.output
    assert "deep_research" in res.output
    report = tmp_path / "deep_research" / "what-is-testland.md"
    assert report.exists()
    text = report.read_text()
    assert "# Deep research: what is testland?" in text
    assert "Testland is fictional [1]." in text


def test_cli_research_requires_a_question():
    from typer.testing import CliRunner

    from agent.main import app

    res = CliRunner().invoke(app, ["research"])
    assert res.exit_code != 0


# ------------------------------------------------------------------------------------------------- tool registration


@pytest.fixture()
def _tools_env(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from agent.memory.store import Store
    from agent.web import deps
    from agent.web.app import app

    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "t.db")
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: FakeClient()
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_deep_research_tool_is_registered_for_the_web_chat_with_safe_risk(_tools_env):
    tools = {t["name"]: t for t in _tools_env.get("/api/tools").json()}
    assert "deep_research" in tools
    assert tools["deep_research"]["risk"] == "SAFE"


def test_deep_research_tool_is_registered_for_the_terminal_chat():
    """`agent/tui.py::run_repl` registers the same tool for the CLI chat loop - check the factory it calls
    produces a well-formed, read-only tool without having to drive the whole interactive loop."""
    from agent.tools.deep_research import make_deep_research_tool

    tool = make_deep_research_tool(FakeClient(), copy.deepcopy(DEFAULT_CONFIG), console=None)
    assert tool.name == "deep_research" and tool.risk == "safe"
    assert tool.parameters["required"] == ["question"]
    assert "question" in tool.parameters["properties"]


def test_deep_research_tool_handler_starts_a_job_and_reports_its_id(monkeypatch, tmp_path):
    from agent.tools.deep_research import make_deep_research_tool
    from agent.web.routes import _jobs

    pages = {"a broad question?": []}
    monkeypatch.setattr("agent.tools.web.make_web_tools", _fake_make_web_tools(pages))
    monkeypatch.chdir(tmp_path)
    tool = make_deep_research_tool(FakeClient(), copy.deepcopy(DEFAULT_CONFIG), console=None)

    out = tool.handler(question="a broad question?", max_subquestions=3, budget_chars=5000)
    assert "Started deep research job" in out
    jid = out.split("job ")[1].split(" ")[0]
    import time

    deadline = time.monotonic() + 10  # the job runs in a real daemon thread; generous under a loaded test run
    job = _jobs.get(jid)
    while job and job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
        job = _jobs.get(jid)
    assert job is not None and job["status"] == "done", job


# ------------------------------------------------------------------------------------------------------ web route


def test_research_deep_route_starts_polls_and_stops(_tools_env, monkeypatch, tmp_path):
    """`agent/web/routes/research_deep.py`: start -> poll -> (idempotent re-start) -> stop, on fakes only."""
    import time

    from agent.web.routes import _jobs

    tc = _tools_env
    pages = {"route question?": []}
    monkeypatch.setattr("agent.tools.web.make_web_tools", _fake_make_web_tools(pages))
    monkeypatch.chdir(tmp_path)

    r = tc.post("/api/research/deep", json={"question": "route question?", "max_subquestions": 2, "budget_chars": 5000})
    assert r.status_code == 200 and r.json()["already_running"] is False
    jid = r.json()["id"]

    # starting the same question again while it may still be running is reported, not duplicated
    r2 = tc.post("/api/research/deep", json={"question": "route question?"})
    assert r2.status_code == 200

    deadline = time.monotonic() + 10
    status = tc.get(f"/api/research/deep/{jid}").json()
    while status["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
        status = tc.get(f"/api/research/deep/{jid}").json()
    assert status["status"] == "done", status
    assert status["result"]["questions_answered"] == 0  # no evidence found, but the job still completed cleanly

    assert tc.post(f"/api/research/deep/{jid}/stop").status_code == 200   # a no-op on a finished job, must not error
    assert tc.get("/api/research/deep/doesnotexist").status_code == 404
    assert tc.post("/api/research/deep", json={"question": "   "}).status_code == 422
