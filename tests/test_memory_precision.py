"""What may become memory: web-learned facts need verified evidence and the sources, junk filter keeps real setup facts."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from agent import tui
from agent.memory.distill import JUNK_FACT_RE
from agent.memory.store import Store
from agent.research import evidence
from agent.research.fetch import Fetched
from agent.tools import web


class _Client:
    def __init__(self, reply="The Eiffel Tower was completed in 1889."):
        self.reply, self.calls = reply, []

    def chat(self, messages, model, stream=False, **kw):
        self.calls.append(messages)
        return iter([self.reply])

    def embed(self, texts, model):
        return [[0.1, 0.2]]


class _NoVectors:
    available = False


CONFIG = {"models": {"background": "bg", "embed": "e"}}
SOURCES = [{"domain": "toureiffel.paris", "date": "2026-01-01", "level": "high", "passage": "Completed in 1889 for the World's Fair."}]


@pytest.fixture(autouse=True)
def _fresh_evidence():
    evidence.clear()
    yield
    evidence.clear()


def _facts(store):
    """(content, kind, confidence) of every active fact; the store only filters by kind, so ask for each."""
    return [(f["content"], kind, f["confidence"]) for kind in ("personal", "incidental") for f in store.get_active_facts(kind=kind)]


def test_evidence_keeps_the_strongest_label_and_recent_sources():
    assert evidence.take() is None
    evidence.record("single_source", SOURCES)
    evidence.record("verified", SOURCES)
    evidence.record("low_trust", SOURCES)
    got = evidence.take()
    assert got["label"] == "verified" and len(got["sources"]) == 3
    evidence.clear()
    assert evidence.take() is None


@pytest.mark.parametrize("label", ["single_source", "conflicting", "low_trust"])
def test_weak_evidence_never_becomes_a_fact_and_costs_no_model_call(tmp_path, label):
    store, client = Store(tmp_path / "a.db"), _Client()
    tui._save_learned_facts(client, CONFIG, store, _NoVectors(), [("How tall is it?", "300 m", {"label": label, "sources": SOURCES})])
    assert _facts(store) == [] and client.calls == []


def test_verified_evidence_is_shown_to_the_extractor_and_the_fact_carries_its_origin(tmp_path):
    store, client = Store(tmp_path / "a.db"), _Client()
    tui._save_learned_facts(client, CONFIG, store, _NoVectors(),
                            [("When was it built?", "In 1889.", {"label": "verified", "sources": SOURCES})])
    prompt = client.calls[0][1]["content"]
    assert "Sources:" in prompt and "toureiffel.paris" in prompt and "Completed in 1889" in prompt
    assert "stated by the sources" in client.calls[0][0]["content"]
    ((content, kind, confidence),) = _facts(store)
    assert content.startswith("The Eiffel Tower was completed in 1889.") and "(web, " in content and "toureiffel.paris" in content
    assert kind == "incidental" and confidence == "verified"


def test_likely_evidence_is_saved_as_assumed(tmp_path):
    store = Store(tmp_path / "a.db")
    tui._save_learned_facts(_Client(), CONFIG, store, _NoVectors(), [("q", "a", {"label": "likely", "sources": SOURCES})])
    assert _facts(store)[0][2] == "assumed"


def test_urls_and_none_are_not_saved_and_old_two_part_exchanges_still_work(tmp_path):
    store = Store(tmp_path / "a.db")
    tui._save_learned_facts(_Client("See https://example.org for the height."), CONFIG, store, _NoVectors(), [("q", "a")])
    tui._save_learned_facts(_Client("NONE"), CONFIG, store, _NoVectors(), [("q", "a", {"label": "verified", "sources": SOURCES})])
    assert _facts(store) == []
    tui._save_learned_facts(_Client("Paris is the capital of France."), CONFIG, store, _NoVectors(), [("q", "a")])   # no web: as before
    assert _facts(store)[0][0] == "Paris is the capital of France."


@pytest.mark.parametrize("text,junk", [
    ("uses pytest for testing", False), ("main project is a Python game in game.py", False), ("my site is https://example.org", False),
    ("keeps notes in a README", False), ("/home/alice/Documents/x", True), ("~/projects/foo", True), ("C:\\Users\\a", True),
    ("list_dir output", True), ("saved at 2026-10-18T10:00:00", True), ("[image] deleted", True),
])
def test_junk_filter_keeps_real_setup_facts(text, junk):
    assert bool(JUNK_FACT_RE.search(text)) is junk


_PAGE = ("<html><head><title>Tower</title><meta name='author' content='a'><meta property='article:published_time' content='2026-01-01'></head>"
         "<body><article>" + "".join(f"<p>The Eiffel Tower in Paris was completed in 1889 for the World's Fair, detail number {i} about it.</p>" for i in range(8))
         + "</article></body></html>")


def test_web_search_records_the_evidence_the_chat_hands_to_memory(monkeypatch):
    pages = {"https://www.toureiffel.paris/a": _PAGE, "https://en.wikipedia.org/wiki/Eiffel_Tower": _PAGE.replace("detail number", "more detail")}
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: [{"url": u, "title": "t", "snippet": ""} for u in pages])
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, html=pages[url]))
    tool = {t.name: t for t in web.make_web_tools(Console(file=io.StringIO(), force_terminal=False, width=100))}["web_search"]
    tool.handler(query="eiffel tower completed")
    got = evidence.take()
    assert got and got["label"] in ("verified", "likely") and {s["domain"] for s in got["sources"]} >= {"en.wikipedia.org"}


def test_clearing_a_turn_also_forgets_the_last_search_so_it_cannot_ground_the_next_answer():
    evidence.set_last_search([{"n": 1, "status": "ok", "passage": "x"}], {"label": "likely"})
    evidence.clear()
    assert evidence.last_search() == {"items": [], "verdict": {}}
