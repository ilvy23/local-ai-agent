"""Search-engine chain: block detection, failover, multi-query fusion and the "nothing matched" honesty. No network."""

from __future__ import annotations

import io

import httpx
import pytest
from rich.console import Console

from agent.research.fetch import Fetched
from agent.tools import web

PAGE = ("<html><head><title>Tokio</title><meta name='author' content='x'><meta property='article:published_time' content='2026-01-01'>"
        "</head><body><article>" + "".join(f"<p>Tokio is an asynchronous runtime for Rust, point {i}: it schedules tasks across threads "
                                         "with work stealing and offers timers and io.</p>" for i in range(8)) + "</article></body></html>")


def _tools(monkeypatch, pages=None):
    pages = pages or {}
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, html=pages.get(url, PAGE)))
    console = Console(file=io.StringIO(), force_terminal=False, width=100, color_system=None)
    return {t.name: t for t in web.make_web_tools(console)}


def _resp(status=200, text=""):
    return httpx.Response(status, text=text, request=httpx.Request("POST", "https://html.duckduckgo.com/html/"))


def test_ddg_bot_challenge_is_reported_as_blocked_not_as_no_results(monkeypatch):
    monkeypatch.setattr(web.httpx, "post", lambda *a, **k: _resp(202, "<html>challenge</html>"))
    with pytest.raises(web.SearchBlocked, match="202"):
        web._search_ddg("anything", 5)
    monkeypatch.setattr(web.httpx, "post", lambda *a, **k: _resp(200, "<html>Unfortunately, bots use DuckDuckGo too. captcha</html>"))
    with pytest.raises(web.SearchBlocked, match="bot check"):
        web._search_ddg("anything", 5)
    monkeypatch.setattr(web.httpx, "post", lambda *a, **k: _resp(200, "<html><body>No  results.</body></html>"))
    assert web._search_ddg("anything", 5) == []


def test_chain_falls_through_a_blocked_engine_to_the_next(monkeypatch):
    def blocked(q, k, df=None):
        raise web.SearchBlocked("nope")

    good = lambda q, k, df=None: [{"url": "https://tokio.rs/a", "title": "Tokio", "snippet": "s"}]
    monkeypatch.setattr(web, "_engines", lambda: [("ddg", blocked), ("brave", good)])
    hits, engine = web._search("tokio", 5)
    assert engine == "brave" and hits[0]["url"] == "https://tokio.rs/a"


def test_all_engines_blocked_says_so_and_is_not_confused_with_no_results(monkeypatch):
    def blocked(q, k, df=None):
        raise web.SearchBlocked("bot check")

    monkeypatch.setattr(web, "_engines", lambda: [("ddg", blocked)])
    out = _tools(monkeypatch)["web_search"].handler(query="tokio")
    assert out.startswith("web search failed") and "could not run" in out and "SEARXNG" in out and "No results" not in out
    monkeypatch.setattr(web, "_engines", lambda: [("ddg", lambda q, k, df=None: [])])
    assert _tools(monkeypatch)["web_search"].handler(query="tokio") == "No results found."


def test_engines_are_chosen_from_the_environment(monkeypatch):
    monkeypatch.delenv("AGENT_SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    assert [n for n, _ in web._engines()] == ["ddg"]
    monkeypatch.setenv("AGENT_SEARXNG_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    assert [n for n, _ in web._engines()] == ["searxng", "ddg", "brave"]


def test_searxng_and_brave_results_are_parsed(monkeypatch):
    monkeypatch.setenv("AGENT_SEARXNG_URL", "http://127.0.0.1:8888/")
    seen = {}

    def fake_get(url, **kw):
        seen["url"], seen["params"] = url, kw.get("params")
        if "8888" in url:
            body = {"results": [{"url": "https://a.org/1", "title": "<b>A</b>", "content": "snippet a"}, {"url": "ftp://x", "title": "bad"}]}
        else:
            body = {"web": {"results": [{"url": "https://b.org/1", "title": "B", "description": "snippet b"}]}}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(web.httpx, "get", fake_get)
    assert web._search_searxng("q", 5, df="w") == [{"url": "https://a.org/1", "title": "A", "snippet": "snippet a"}]
    assert seen["url"] == "http://127.0.0.1:8888/search" and seen["params"]["time_range"] == "week"
    assert web._search_brave("q", 5)[0]["url"] == "https://b.org/1"


def test_several_queries_are_fused_and_a_page_both_find_ranks_first(monkeypatch):
    lists = {"q1": [{"url": "https://a.org/1", "title": "a", "snippet": ""}, {"url": "https://shared.org/x", "title": "s", "snippet": ""}],
             "q2": [{"url": "https://shared.org/x/", "title": "s", "snippet": ""}, {"url": "https://c.org/1", "title": "c", "snippet": ""}]}
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: lists[q])
    tool = _tools(monkeypatch)["web_search"]
    out = tool.handler(query="q1", queries=["q2", "q3-ignored-if-more-than-max"], num_results=3, read=False)
    first_line = [ln for ln in out.splitlines() if ln.startswith("[1]")][0]
    assert "s" in first_line and "shared.org" in first_line
    assert "q3" not in out


def test_pages_that_do_not_address_the_question_are_flagged(monkeypatch):
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: [{"url": "https://a.org/1", "title": "t", "snippet": ""}])
    out = _tools(monkeypatch)["web_search"].handler(query="best sourdough hydration percentage")
    assert "nothing on this page matches" in out and "more specific query" in out
    ok = _tools(monkeypatch)["web_search"].handler(query="tokio work stealing scheduler")
    assert "nothing on this page matches" not in ok and "more specific query" not in ok
