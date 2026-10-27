"""Sponsored links are not search results, and hrefs arrive HTML-escaped.

DDG serves ads through the same `result__a` markup as real hits. One music
query returned two merch shops above Wikipedia — burning 2 of 5 slots on
things that answer nothing and don't even fetch.
"""

from __future__ import annotations

from agent.tools.web import _is_ad, _unwrap

AD_HREF = (
    "https://duckduckgo.com/y.js?ad_domain=emp.de&amp;ad_provider=bingv7aa"
    "&amp;ad_type=txad&amp;u3=https%3A%2F%2Fwww.bing.com%2Faclick"
)
REAL_HREF = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FOctober_Rust&amp;rut=abc"


def test_spots_a_sponsored_link():
    assert _is_ad(_unwrap(AD_HREF))


def test_does_not_flag_a_real_result():
    assert not _is_ad(_unwrap(REAL_HREF))
    assert not _is_ad("https://en.wikipedia.org/wiki/October_Rust")


def test_unwraps_a_real_result_through_the_redirect():
    assert _unwrap(REAL_HREF) == "https://en.wikipedia.org/wiki/October_Rust"


def test_html_entities_in_hrefs_are_decoded():
    """hrefs come out of the page escaped; &amp; must become & or the URL is
    wrong (and the fetch fails)."""
    href = "https://example.com/x?a=1&amp;b=2"
    assert _unwrap(href) == "https://example.com/x?a=1&b=2"
