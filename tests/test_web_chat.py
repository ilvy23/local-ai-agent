"""The web chat endpoint end to end with a scripted model: SSE events, persistence, blocks, approvals. Needs the web group."""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


class ScriptedClient:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, **kw):
        self.prompts.append(list(messages))
        reply = self.replies.pop(0)
        for i in range(0, len(reply.get("content", "")), 9):
            on_content(reply["content"][i:i + 9])
        return {"_prompt_eval_count": 500, "_eval_count": 100, **reply}

    def chat(self, messages, model, **kw):
        return iter(["NONE"])

    def embed(self, texts, model, **kw):
        return [[0.1] * 8 for _ in texts]

    def list_models(self):
        return []


def _call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def _events(response):
    out = []
    for chunk in response.text.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines() if ": " in line)
        if "data" in lines:
            out.append(json.loads(lines["data"]))
    return out


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "web.db")
    config["tools"]["max_iterations"] = 6
    client = ScriptedClient()
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_client] = lambda: client
    app.dependency_overrides[deps.get_store] = lambda: Store(config["data"]["db_path"])
    yield TestClient(app), client, config
    app.dependency_overrides.clear()


def _session(tc):
    return tc.post("/api/sessions").json()["id"]


def test_plain_answer_streams_tokens_then_done_and_is_saved(env):
    tc, client, config = env
    client.replies = [{"role": "assistant", "content": "A calm, direct answer to your question."}]
    sid = _session(tc)
    events = _events(tc.post(f"/api/sessions/{sid}/messages", json={"text": "hello?"}))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and "token" in kinds and "context" in kinds and "status" in kinds
    assert "".join(e["text"] for e in events if e["type"] == "token") == "A calm, direct answer to your question."
    history = tc.get(f"/api/sessions/{sid}/messages").json()
    assert [(m["role"], m["content"]) for m in history] == [("user", "hello?"), ("assistant", "A calm, direct answer to your question.")]


def test_present_tool_emits_a_block_that_is_stored_but_never_sent_back_to_the_model(env):
    tc, client, _ = env
    spec = {"columns": [{"key": "a", "label": "A"}], "rows": [{"a": 1}]}
    client.replies = [_call("present", {"kind": "table", "title": "T", "spec": spec}), {"role": "assistant", "content": "Here is the table above."}]
    sid = _session(tc)
    events = _events(tc.post(f"/api/sessions/{sid}/messages", json={"text": "show a table"}))
    block = next(e for e in events if e["type"] == "block")
    assert block["kind"] == "table" and block["spec"]["rows"] == [{"a": 1}] and block["provenance"] == "model"
    roles = [m["role"] for m in tc.get(f"/api/sessions/{sid}/messages").json()]
    assert "block" in roles
    client.replies = [{"role": "assistant", "content": "Second answer here, plain text."}]
    tc.post(f"/api/sessions/{sid}/messages", json={"text": "and now?"})
    assert not any('"kind": "table"' in str(m.get("content")) and m["role"] == "block" for m in client.prompts[-1])


def test_approval_round_trip_resumes_the_turn(env, tmp_path):
    tc, client, _ = env
    target = tmp_path / "out.txt"
    client.replies = [_call("write_file", {"path": str(target), "content": "hello"}),
                      {"role": "assistant", "content": "The file has been written for you."}]
    sid = _session(tc)
    events = _events(tc.post(f"/api/sessions/{sid}/messages", json={"text": "write it"}))
    assert events[-1]["type"] == "awaiting_approval" and events[-1]["risk_level"] == "CAUTION" and not target.exists()
    pending = tc.get(f"/api/approvals/pending?session_id={sid}").json()
    assert len(pending) == 1
    resumed = _events(tc.post(f"/api/approvals/{pending[0]['id']}/decide", json={"approve": True}))
    assert resumed[-1] == {"type": "done", "reply": "The file has been written for you."}
    assert target.read_text() == "hello"
    assert tc.post(f"/api/approvals/{pending[0]['id']}/decide", json={"approve": True}).status_code == 409


def test_denied_approval_does_not_write_and_the_model_is_told(env, tmp_path):
    tc, client, _ = env
    target = tmp_path / "no.txt"
    client.replies = [_call("write_file", {"path": str(target), "content": "x"}), {"role": "assistant", "content": "Understood, I did not write it."}]
    sid = _session(tc)
    _events(tc.post(f"/api/sessions/{sid}/messages", json={"text": "write it"}))
    pid = tc.get(f"/api/approvals/pending?session_id={sid}").json()[0]["id"]
    resumed = _events(tc.post(f"/api/approvals/{pid}/decide", json={"approve": False}))
    assert resumed[-1]["type"] == "done" and not target.exists()
    assert "APPROVAL DENIED" in client.prompts[-1][-1]["content"]


def test_evidence_gated_background_learning_is_queued_only_for_strong_evidence(monkeypatch):
    from agent.memory import background

    queued = []
    monkeypatch.setattr(background, "submit", lambda fn, *a: queued.append(fn))
    assert background.learn_after_turn(None, {}, "q", "a", None) is None
    assert background.learn_after_turn(None, {}, "q", "a", {"label": "single_source", "sources": []}) is None
    assert background.learn_after_turn(None, {}, "q", "  ", {"label": "verified", "sources": []}) is None
    background.learn_after_turn(None, {}, "q", "a", {"label": "verified", "sources": []})
    assert len(queued) == 1


# ---------------------------------------------------------------------------------------- /verify


_OFFICIAL = ("<html><head><title>Python 3.14 release notes</title><meta name='author' content='Python team'>"
             "<meta property='article:published_time' content='2026-09-14T10:00:00Z'></head><body><article>"
             "<p>Python 3.14 startup time is 40 % faster than 3.13 according to the official benchmark suite results.</p>"
             + "".join(f"<p>Section {i}: the interpreter received further changes to imports, memory handling and the standard library, "
                       f"described here for release number {i}.</p>" for i in range(6)) + "</article></body></html>")


def test_verify_endpoint_checks_the_last_answer_and_streams_the_evidence(env, monkeypatch):
    from agent.research.fetch import Fetched

    tc, client, _ = env
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: [{"url": "https://docs.python.org/3.14/whatsnew", "title": "t", "snippet": ""}])
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, html=_OFFICIAL))
    sid = _session(tc)
    assert _events(tc.post(f"/api/sessions/{sid}/verify"))[-1]["type"] == "error"            # nothing to verify yet
    client.replies = [{"role": "assistant", "content": "Python 3.14 starts about 40 % faster than 3.13 [1]. It also adds 90 new tools overall."}]
    tc.post(f"/api/sessions/{sid}/messages", json={"text": "what changed?"})
    events = _events(tc.post(f"/api/sessions/{sid}/verify"))
    kinds = [e["type"] for e in events]
    assert "sources" in kinds and "verdict" in kinds and kinds[-1] == "done"
    result = next(e for e in events if e["type"] == "verify_result")
    assert result["claims"][0]["status"] == "supported" and result["searches"] >= 2
    assert "supported" in events[-1]["reply"]
    assert "evidence" in [m["role"] for m in tc.get(f"/api/sessions/{sid}/messages").json()]
