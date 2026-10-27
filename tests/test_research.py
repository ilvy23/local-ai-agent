import io
import sqlite3
import time
from unittest.mock import MagicMock

from rich.console import Console

from agent.research import extract
from agent.research.fetch import Fetched, fetch_many
from agent.research.memory import WebMemory, query_key, ttl_for
from agent.tools import web

HTML = """<html><head><title>Tokio tutorial</title>
<meta property="article:published_time" content="2025-03-02T10:00:00Z">
<script>var junk = 1;</script></head><body>
<nav><a>Home</a><a>Docs</a></nav><header>Site header</header>
<div class="cookie">We use cookies. Accept all</div>
<article><h1>Tokio tutorial</h1>
<p>Tokio is an asynchronous runtime for the Rust programming language and it provides the building blocks.</p>
<h2>Installation</h2><p>Add tokio to Cargo.toml with the full feature flag to get the multi-threaded scheduler working.</p>
<p>Unrelated paragraph about gardening tomatoes and watering them every morning before the sun is high.</p>
<ul><li>Spawn tasks with tokio::spawn for concurrency</li></ul></article>
<footer>Copyright 2025 all rights reserved</footer></body></html>"""


def test_extract_keeps_article_drops_chrome_finds_title_and_date():
    page = extract.extract_page(HTML)
    assert page.title == "Tokio tutorial" and page.published == "2025-03-02"
    assert "asynchronous runtime" in page.text and "tokio::spawn" in page.text
    for junk in ("cookies", "Site header", "var junk", "all rights reserved", "Home"):
        assert junk not in page.text


def test_select_passages_prefers_matching_paragraphs_in_page_order():
    filler = "\n".join(f"Paragraph number {i} talks about completely different matters such as weather and cooking." * 2 for i in range(30))
    text = filler + "\nThe tokio scheduler uses work stealing across the multi-threaded runtime for fast task dispatch." + "\n" + filler
    out = extract.select_passages(text, "how does the tokio scheduler work", budget=700)
    assert "work stealing" in out and len(out) <= 900
    assert extract.select_passages("short text", "anything", 500) == "short text"
    lede = extract.select_passages(filler, "zzzqqq nothing matches", 300)
    assert lede.startswith("Paragraph number 0") and lede.endswith("…")


def test_fetch_many_runs_in_parallel_and_marks_stragglers_as_timeout():
    import threading

    gate = threading.Event()                      # (time.sleep is stubbed out by conftest)

    def fake(url):
        if "slow" in url:
            gate.wait(1.5)
        return Fetched(url, html="x")

    started = time.monotonic()
    out = fetch_many(["https://a.org/1", "https://b.org/2", "https://slow.org/3"], {}, deadline=0.4, fetch=fake)
    assert time.monotonic() - started < 1.2
    gate.set()
    assert out["https://a.org/1"].status == "ok" and out["https://slow.org/3"].status == "timeout"


def test_skip_reasons():
    from agent.research.fetch import _skip_reason

    assert _skip_reason("https://www.youtube.com/watch?v=1") and _skip_reason("https://x.org/paper.pdf")
    assert _skip_reason("https://docs.rs/tokio") == ""


def test_ttl_classes():
    assert ttl_for("latest rust release") == 1800
    assert ttl_for("rust tutorial for beginners") == 7 * 86400
    assert ttl_for("compare two crates") == 12 * 3600
    assert ttl_for("anything", "day") == 1800


class FakeVectors:
    available = True

    def __init__(self):
        self.added = []

    def add(self, kind, ref_id, text, embedding):
        self.added.append((kind, ref_id, text, embedding))


def _memory(clock=None, vectors=None, embed=None):
    conn = sqlite3.connect(":memory:")
    kwargs = {"clock": clock} if clock else {}
    return WebMemory(conn, vectors, embed, **kwargs)


def test_memory_caches_pages_and_hits_with_expiry():
    now = [1000.0]
    m = _memory(clock=lambda: now[0])
    m.put_page("https://a.org", "T", "2025-01-01", "text")
    key = query_key("Rust  Async", "any")
    m.put_hits(key, "Rust Async", [{"url": "https://a.org", "title": "T", "snippet": ""}])
    assert m.page("https://a.org", 60)["text"] == "text"
    assert m.hits(query_key("rust async"), 60)[0]["url"] == "https://a.org"
    now[0] += 120
    assert m.page("https://a.org", 60) is None and m.hits(key, 60) is None


def test_memory_remember_embeds_one_dated_note_once():
    vectors = FakeVectors()
    m = _memory(vectors=vectors, embed=lambda texts: [[0.1, 0.2]])
    key = query_key("tokio scheduler")
    m.put_hits(key, "tokio scheduler", [])
    src = [{"url": "https://tokio.rs/x", "title": "Tokio", "published": "2025-03-02", "best": "work stealing scheduler"}]
    assert m.remember(key, "tokio scheduler", src) is True
    assert m.remember(key, "tokio scheduler", src) is False            # only once
    kind, _ref, text, emb = vectors.added[0]
    assert kind == "web" and "tokio.rs" in text and "2025-03-02" in text and "work stealing" in text and emb == [0.1, 0.2]
    broken = _memory(vectors=FakeVectors(), embed=lambda t: 1 / 0)
    broken.put_hits(key, "q", [])
    assert broken.remember(key, "q", src) is False                     # never raises


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=100, color_system=None)


def _tool_env(monkeypatch, calls, memory=None, html=HTML):
    def fake_search(query, k, df=None):
        calls["search"] += 1
        return [{"url": f"https://site{i}.example.org/p", "title": f"R{i}", "snippet": "snip"} for i in range(k)]

    def fake_fetch(url):
        calls["fetch"] += 1
        return Fetched(url, html=html, seconds=0.4)

    monkeypatch.setattr("agent.tools.web._search_ddg", fake_search)
    monkeypatch.setattr("agent.tools.web._http_fetch", fake_fetch)
    console = _console()
    tools = {t.name: t for t in web.make_web_tools(console, memory)}
    return tools, console


def test_search_output_is_compact_dated_and_numbered(monkeypatch):
    calls = {"search": 0, "fetch": 0}
    tools, console = _tool_env(monkeypatch, calls)
    out = tools["web_search"].handler(query="tokio scheduler", num_results=3)
    assert "[1] Tokio tutorial - site0.example.org - 2025-03-02" in out and "Cite sources" in out
    assert "gardening" not in out or len(out) < 7000
    shown = console.file.getvalue()
    assert shown.startswith("search  tokio scheduler") and "read 0.4s" in shown and "3 sources" in shown
    assert not any(ord(c) > 0x2500 and c != "…" for c in shown)         # no emoji / symbols


def test_second_identical_search_comes_from_the_cache_without_network(monkeypatch):
    calls = {"search": 0, "fetch": 0}
    mem = _memory()
    tools, console = _tool_env(monkeypatch, calls, mem)
    tools["web_search"].handler(query="tokio scheduler", num_results=2)
    first = dict(calls)
    out = tools["web_search"].handler(query="Tokio  Scheduler", num_results=2)
    assert calls == first and "[1]" in out
    assert "cached" in console.file.getvalue() and "results from cache" in console.file.getvalue()


def test_unreadable_sources_are_reported_not_fatal(monkeypatch):
    calls = {"search": 0, "fetch": 0}
    tools, console = _tool_env(monkeypatch, calls)
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, status="timeout", note="timeout"))
    out = tools["web_search"].handler(query="x", num_results=2)
    assert "Not read: [1] timeout; [2] timeout" in out and "snip" in out
    assert "timeout" in console.file.getvalue()


def test_web_fetch_focus_and_offset_paging(monkeypatch):
    calls = {"search": 0, "fetch": 0}
    big = "<html><body><article>" + "".join(f"<p>Paragraph {i} about topic {i % 7} with enough words to count as text here.</p>" for i in range(400)) + "</article></body></html>"
    tools, _ = _tool_env(monkeypatch, calls, html=big)
    first = tools["web_fetch"].handler(url="https://a.org/doc", max_chars=1000)
    assert "call again with offset=" in first
    nxt = tools["web_fetch"].handler(url="https://a.org/doc", max_chars=1000, offset=1000)
    assert "characters 1000-2000" in nxt
    assert "Could not read" in tools["web_fetch"].handler(url="not a url")
    focused = tools["web_fetch"].handler(url="https://a.org/doc", focus="topic 3", max_chars=800)
    assert "passages about 'topic 3'" in focused


def test_search_failure_is_returned_to_the_model(monkeypatch):
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: 1 / 0)
    tools = {t.name: t for t in web.make_web_tools(_console())}
    assert tools["web_search"].handler(query="x").startswith("web search failed")


def test_freshness_is_passed_to_the_engine(monkeypatch):
    seen = {}
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: seen.setdefault("df", df) and [])
    tools = {t.name: t for t in web.make_web_tools(_console())}
    tools["web_search"].handler(query="news", freshness="week")
    assert seen["df"] == "w"


def test_recall_includes_labeled_recent_web_notes_and_drops_old_ones():
    from datetime import UTC, datetime, timedelta

    from agent.memory import recall

    today = datetime.now(UTC).date().isoformat()
    old = (datetime.now(UTC) - timedelta(days=90)).date().isoformat()

    class V:
        available = True
        def search(self, emb, k, kinds):
            assert "web" in kinds
            return [(f"Web research on {today} for 'tokio': Tokio (tokio.rs, 2025): x", "web", 1, 0.5),
                    (f"Web research on {old} for 'tokio': old", "web", 2, 0.5),
                    ("Web research on 2099-01-01 for 'far': far", "web", 3, 1.2),     # too dissimilar
                    ("user likes rust", "fact", 4, 0.5)]

    llm = MagicMock()
    llm.embed.return_value = [[0.0]]
    cfg = {"memory": {"recall_k": 6, "recall_min_similarity": 0.55}, "models": {"embed": "e"}}
    hits = recall._semantic_hits(V(), llm, cfg, "tokio?")
    assert len(hits) == 2 and hits[0].startswith("[from an earlier web search") and "user likes rust" in hits[1]


def test_blocked_sites_are_replaced_by_the_next_readable_results(monkeypatch):
    calls = {"search": 0, "fetch": 0}
    tools, console = _tool_env(monkeypatch, calls)

    def fetch(url):
        calls["fetch"] += 1
        if "site0" in url or "site1" in url:
            return Fetched(url, status="failed", note="http 403")
        return Fetched(url, html=HTML, seconds=0.2)

    monkeypatch.setattr("agent.tools.web._http_fetch", fetch)
    out = tools["web_search"].handler(query="tokio scheduler", num_results=3)
    read_rows = [line for line in console.file.getvalue().splitlines() if "read 0.2s" in line]
    assert len(read_rows) == 3 and "failed (http 403)" in console.file.getvalue()
    assert out.count(" - 2025-03-02") == 3                      # three sources with content
    assert "Not read: [1] http 403; [2] http 403" in out


def test_change_summary_paths_are_relative_to_their_project(tmp_path):
    from agent import changes

    proj = tmp_path / "w" / "proj1"
    (proj / ".git").mkdir(parents=True)
    f = proj / "tests" / "test_calc.py"
    f.parent.mkdir()
    f.write_text("x")
    assert changes.display_path(str(f), cwd=tmp_path / "elsewhere") == "proj1/tests/test_calc.py"
