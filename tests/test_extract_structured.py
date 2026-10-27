"""Tables keep their cell boundaries and JSON-LD contributes FAQ, prices and script-rendered article text."""

from __future__ import annotations

import json

from agent.research import extract, verdict


def _page(inner: str, head: str = "") -> str:
    return f"<html><head><title>T</title>{head}</head><body><article>{inner}</article></body></html>"


def test_table_rows_become_one_line_each_with_header():
    html = _page("<p>Benchmark results for the three runtimes measured on one machine for this comparison article.</p>"
                 "<table><tr><th>Language</th><th>Speed (ms)</th></tr><tr><td>Rust</td><td>12</td></tr><tr><td>Python</td><td>410</td></tr></table>")
    text = extract.extract_page(html).text
    assert "| Language | Speed (ms) |" in text and "| Rust | 12 |" in text and "| Python | 410 |" in text


def test_table_rows_survive_the_short_line_filter_and_nested_markup_is_cleaned():
    html = _page("<table><tr><td><a href='x'>Go</a></td><td><b>30</b> ms</td></tr></table>")
    assert "| Go | 30 ms |" in extract.extract_page(html).text


def test_large_tables_are_capped():
    rows = "".join(f"<tr><td>row{i}</td><td>{i}</td></tr>" for i in range(100))
    text = extract.extract_page(_page(f"<table>{rows}</table>")).text
    assert text.count("| row") == extract.MAX_TABLE_ROWS


def test_a_table_cell_value_can_answer_a_question():
    html = _page("<p>Pricing overview of the plans offered by the vendor, updated for this year with all taxes included.</p>"
                 "<table><tr><th>Plan</th><th>Price</th></tr><tr><td>Pro</td><td>49 EUR</td></tr></table>" + "<p>Filler paragraph about unrelated gardening topics. </p>" * 30)
    text = extract.extract_page(html).text
    assert "Pro" in extract.select_passages(text, "Pro plan price", 400) and "49 EUR" in extract.select_passages(text, "Pro plan price", 400)


def _ld(obj) -> str:
    return f'<script type="application/ld+json">{json.dumps(obj)}</script>'


def test_faq_and_product_structured_data_become_text():
    faq = {"@type": "FAQPage", "mainEntity": [{"@type": "Question", "name": "Is it free?", "acceptedAnswer": {"@type": "Answer", "text": "<p>Yes, for personal use.</p>"}}]}
    prod = {"@context": "https://schema.org", "@graph": [{"@type": "Product", "name": "Widget", "offers": {"@type": "Offer", "price": "19.99", "priceCurrency": "EUR"}}]}
    text = extract.extract_page(_page("<p>Some visible text that is long enough to stay in the page body for sure okay.</p>", _ld(faq) + _ld(prod))).text
    assert "Q: Is it free? A: Yes, for personal use." in text and "Product Widget: price 19.99 EUR" in text


def test_script_rendered_page_falls_back_to_the_article_body():
    body = "The release shipped on 14 September with faster startup. " * 12
    html = ("<html><head><title>JS page</title>" + _ld({"@type": "NewsArticle", "headline": "x", "articleBody": body, "datePublished": "2026-09-14"})
            + "</head><body><div id='root'></div></body></html>")
    page = extract.extract_page(html)
    assert "faster startup" in page.text and page.published == "2026-09-14"
    long_visible = _page("<p>" + "A long visible article paragraph about something else entirely. " * 12 + "</p>", _ld({"@type": "NewsArticle", "articleBody": body}))
    assert "faster startup" not in extract.extract_page(long_visible).text      # not duplicated when the page already has text


def test_broken_json_ld_is_ignored():
    html = _page("<p>Visible text that is long enough to be kept as the body of this page for the test.</p>", '<script type="application/ld+json">{not json</script>')
    assert "Visible text" in extract.extract_page(html).text and extract.jsonld_items(html) == []


def test_table_figures_feed_the_conflict_detection():
    a = {"n": 1, "url": "https://a.org/x", "status": "ok", "level": "high", "text": "| Startup | 40 % |"}
    b = {"n": 2, "url": "https://b.org/y", "status": "ok", "level": "medium", "text": "| Startup | 25 % |"}
    assert verdict.find_conflicts([a, b])
