"""Research connectors: which questions they pick up and how their API answers become sources. No network."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from agent.research import connectors as c
from agent.research.fetch import Fetched
from agent.tools import web


@pytest.mark.parametrize("query,names", [
    ("pypi httpx", ["pypi"]), ("pip install rich", ["pypi"]), ("latest version of httpx on pypi", ["pypi"]),
    ("npm express", ["npm"]), ("weather in Munich", ["weather"]), ("Wetter in München heute", ["weather"]),
    ("100 usd to eur", ["exchange"]), ("EUR in CHF", ["exchange"]),
    ("arxiv speculative decoding", ["arxiv"]), ("github rust async runtime", ["github"]),
    ("hacker news rust", ["hackernews"]), ("what is a monad", ["wikipedia"]), ("Wer ist Ada Lovelace?", ["wikipedia"]),
    ("how do I tune a kernel", []), ("rust release notes", []), ("usd to xyz", []), ("buy it to eat", []),
])
def test_routing(query, names):
    assert [n for n, _ in c.plan(query)] == names


def test_nothing_fires_for_ordinary_questions_and_failures_are_swallowed(monkeypatch):
    assert c.lookup("how do I tune a kernel") == []

    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(c, "_get_json", boom)
    assert c.lookup("pypi httpx") == []


def test_pypi_record(monkeypatch):
    monkeypatch.setattr(c, "_get_json", lambda url, params=None, timeout=0: {
        "info": {"name": "httpx", "version": "0.28.1", "summary": "The next generation HTTP client.", "requires_python": ">=3.8",
                 "license": "BSD-3-Clause", "home_page": "https://github.com/encode/httpx"},
        "urls": [{"upload_time_iso_8601": "2024-12-06T15:37:21Z"}]})
    (r,) = c.lookup("pypi httpx")
    assert r["url"] == "https://pypi.org/project/httpx/" and r["published"] == "2024-12-06"
    assert "0.28.1" in r["text"] and "Requires Python >=3.8" in r["text"] and r["signals"]["byline"]


def test_exchange_and_weather_records(monkeypatch):
    def fake(url, params=None, timeout=0):
        if "frankfurter" in url:
            assert params == {"from": "USD", "to": "EUR", "amount": 100.0}
            return {"amount": 100.0, "base": "USD", "date": "2026-10-16", "rates": {"EUR": 92.5}}
        if "geocoding" in url:
            return {"results": [{"name": "Munich", "latitude": 48.1, "longitude": 11.6, "country": "Germany", "admin1": "Bavaria"}]}
        return {"current": {"temperature_2m": 11.2, "wind_speed_10m": 9.0, "relative_humidity_2m": 70, "weather_code": 3},
                "daily": {"time": ["2026-10-18", "2026-10-19"], "temperature_2m_min": [6, 5], "temperature_2m_max": [14, 13],
                          "precipitation_sum": [1.2, 0.0]}}

    monkeypatch.setattr(c, "_get_json", fake)
    (fx,) = c.lookup("100 usd to eur")
    assert "100 USD = 92.5 EUR" in fx["text"] and fx["published"] == "2026-10-16"
    (w,) = c.lookup("weather in Munich")
    assert "Munich, Bavaria, Germany" in w["text"] and "11.2 °C" in w["text"] and "6 to 14 °C" in w["text"]


def test_wikipedia_arxiv_github_hn(monkeypatch):
    def fake_json(url, params=None, timeout=0):
        if "w/api.php" in url:
            return {"query": {"search": [{"title": "Monad (functional programming)"}]}}
        if "page/summary" in url:
            return {"title": "Monad (functional programming)", "extract": "In functional programming, a monad is a structure.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Monad_(functional_programming)"}}, "timestamp": "2026-09-01T00:00:00Z"}
        if "github" in url:
            return {"items": [{"html_url": "https://github.com/tokio-rs/tokio", "full_name": "tokio-rs/tokio", "description": "async runtime",
                               "stargazers_count": 28000, "pushed_at": "2026-10-10T00:00:00Z", "language": "Rust", "license": {"spdx_id": "MIT"}}]}
        return {"hits": [{"objectID": "1", "title": "Rust 1.90", "points": 300, "num_comments": 50, "created_at": "2026-09-14T10:00:00Z", "url": "https://x.org"}]}

    monkeypatch.setattr(c, "_get_json", fake_json)
    monkeypatch.setattr(c, "_get_text", lambda url, params=None, timeout=0: (
        "<feed><entry><id>http://arxiv.org/abs/2401.1</id><title> Speculative  Decoding </title><summary>We study drafting.</summary>"
        "<published>2024-01-02T00:00:00Z</published><author><name>A. One</name></author></entry></feed>"))
    assert c.lookup("what is a monad")[0]["url"].endswith("Monad_(functional_programming)")
    assert c.lookup("arxiv speculative decoding")[0]["url"] == "https://arxiv.org/abs/2401.1"
    assert "28000 stars" in c.lookup("github rust async runtime")[0]["text"]
    assert c.lookup("hacker news rust")[0]["url"].endswith("item?id=1")


def _tool(monkeypatch):
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, status="failed", note="should not be fetched"))
    return {t.name: t for t in web.make_web_tools(Console(file=io.StringIO(), force_terminal=False, width=100, color_system=None))}["web_search"]


def test_web_search_uses_connector_records_without_fetching_and_survives_blocked_engines(monkeypatch):
    monkeypatch.setattr(c, "_get_json", lambda url, params=None, timeout=0: {
        "amount": 100.0, "base": "USD", "date": "2026-10-16", "rates": {"EUR": 92.5}})

    def blocked(q, k, df=None):
        raise web.SearchBlocked("bot check")

    monkeypatch.setattr("agent.tools.web._search_ddg", blocked)
    out = _tool(monkeypatch).handler(query="100 usd to eur")
    assert out.startswith("Web search") and "100 USD = 92.5 EUR" in out and "T2" in out and "should not be fetched" not in out
    assert "web search failed" not in out


def test_connector_records_come_first_and_ordinary_results_follow(monkeypatch):
    monkeypatch.setattr(c, "_get_json", lambda url, params=None, timeout=0: {
        "info": {"name": "httpx", "version": "0.28.1", "summary": "HTTP client", "requires_python": ">=3.8"}, "urls": []})
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: [{"url": "https://blog.example.net/httpx", "title": "httpx blog", "snippet": "s"}])
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, html="<html><body><article>" + "<p>httpx is a fully featured http client for python with sync and async apis.</p>" * 6 + "</article></body></html>"))
    tool = {t.name: t for t in web.make_web_tools(Console(file=io.StringIO(), force_terminal=False, width=100, color_system=None))}["web_search"]
    out = tool.handler(query="latest version of httpx on pypi")
    assert out.index("pypi.org") < out.index("blog.example.net")
