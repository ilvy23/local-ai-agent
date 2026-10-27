"""/verify: claim picking, counter-queries and the supported / contradicted / unclear decision. No network, no model."""

from __future__ import annotations

from agent.research import evidence, verify as vf


def _item(n, domain, level, passage, status="ok"):
    return {"n": n, "url": f"https://{domain}/p", "domain": domain, "title": "t", "date": "2026-09-14", "status": status,
            "trust": {"tier": "T1", "level": level, "reasons": []}, "passage": passage}


def test_claims_prefer_sentences_with_figures_and_drop_citations():
    answer = ("Python is a language. Python 3.14 starts about 40 % faster than 3.13 [1]. It was released on 14 September 2026 [2]. "
              "Many people like it a lot in general. Memory use fell by 12 MB overall.")
    claims = vf.claims_from(answer)
    assert claims == ["Python 3.14 starts about 40 % faster than 3.13.", "It was released on 14 September 2026.",
                      "Memory use fell by 12 MB overall."]
    assert vf.claims_from("Short. Tiny.") == []
    assert len(vf.claims_from("This sentence has enough words in it to count. " * 3)) == 2         # no figures: first two long sentences


def test_counter_queries_keep_names_and_figures():
    qs = vf.counter_queries("Python 3.14 starts about 40 % faster than 3.13.")
    assert len(qs) == 2 and qs[0].endswith("official source") and qs[1].endswith("fact check")
    assert "Python" in qs[0] and "3.14" in qs[0] and "the" not in qs[0].split()


def test_a_figure_found_in_a_solid_source_is_supported():
    r = vf.judge("Startup is 40 % faster in 3.14.", [_item(1, "docs.python.org", "high", "Startup time is 40 % faster than 3.13."),
                                                    _item(2, "spam.example", "low", "Startup is 40 % faster, buy now")])
    assert r["status"] == "supported" and r["support"] == ["https://docs.python.org/p"]


def test_a_different_value_in_a_solid_source_contradicts():
    r = vf.judge("Startup is 90 % faster in 3.14.", [_item(1, "docs.python.org", "high", "Startup time is 40 % faster than 3.13.")])
    assert r["status"] == "contradicted" and r["against"][0]["claim"] == 90 and r["against"][0]["source"] == 40


def test_only_weak_or_unread_sources_leave_it_unclear():
    r = vf.judge("Startup is 40 % faster in 3.14.", [_item(1, "spam.example", "low", "Startup is 40 % faster"),
                                                    _item(2, "docs.python.org", "high", "x", status="blocked")])
    assert r["status"] == "unclear"


def test_a_claim_without_figures_needs_topic_words_in_a_solid_source():
    claim = "The interpreter received changes to imports and memory handling"
    yes = vf.judge(claim, [_item(1, "docs.python.org", "high", "The interpreter received many changes to imports and memory handling.")])
    no = vf.judge(claim, [_item(1, "docs.python.org", "high", "Gardening tomatoes in spring needs patience.")])
    assert yes["status"] == "supported" and no["status"] == "unclear"


def test_verify_runs_two_searches_per_claim_and_summarises():
    calls = []

    def run_search(q):
        calls.append(q)
        return [_item(1, "docs.python.org", "high", "Startup time is 40 % faster than 3.13.")], {"label": "likely"}

    out = vf.verify("Python 3.14 starts about 40 % faster than 3.13 [1]. Memory use fell by 12 MB overall.", run_search)
    assert out["searches"] == len(calls) == 4 and out["overall"] == "unclear"
    assert [c["status"] for c in out["claims"]] == ["supported", "unclear"]
    only_good = vf.verify("Python 3.14 starts about 40 % faster than 3.13 [1].", run_search)
    assert only_good["overall"] == "supported"
    bad = vf.verify("Python 3.14 starts about 90 % faster than 3.13 [1].", run_search)
    assert bad["overall"] == "contradicted"
    assert vf.verify("Hi.", run_search) == {"claims": [], "overall": "nothing_to_check", "searches": 0}


def test_last_search_is_exposed_by_the_evidence_module():
    evidence.set_last_search([{"n": 1}], {"label": "verified"})
    assert evidence.last_search() == {"items": [{"n": 1}], "verdict": {"label": "verified"}}


def test_terminal_verify_command_checks_the_last_answer(tmp_path, monkeypatch):
    from agent import tui
    from agent.memory.store import Store
    from agent.research.fetch import Fetched
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    official = ("<html><head><title>Notes</title><meta name='author' content='a'><meta property='article:published_time' content='2026-09-14'></head>"
                "<body><article><p>Python 3.14 startup time is 40 % faster than 3.13 according to the official benchmark results.</p>"
                + "".join(f"<p>Section {i}: further changes to imports and memory handling are described here for release {i}.</p>" for i in range(6))
                + "</article></body></html>")
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: [{"url": "https://docs.python.org/3.14/new", "title": "t", "snippet": ""}])
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, html=official))
    client = ScriptedToolClient([{"role": "assistant", "content": "Python 3.14 starts about 40 % faster than 3.13 according to benchmarks."}])
    console = _ScriptedConsole(["what is new", "/verify", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=Store(tmp_path / "a.db"))
    shown = console.file.getvalue()
    assert "Checked 1 claim with 2 searches: 1 supported." in shown and "confirmed by https://docs.python.org/3.14/new" in shown

    again = _ScriptedConsole(["/verify", "/quit"])
    tui.run_repl(ScriptedToolClient([]), CONFIG, console=again, store=Store(tmp_path / "b.db"))
    assert "nothing to check yet" in again.file.getvalue()
