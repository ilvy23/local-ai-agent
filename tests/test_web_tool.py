"""Web tool: URL-unwrap, tag-strip, and the /web trigger parse. No network."""

from unittest.mock import MagicMock

from agent.tools.web import _detag, _extract_next_page_params, _search_ddg, _unwrap, make_web_tools
from agent.tui import _split_web_trigger


def test_unwrap_ddg_redirect():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX&rut=abc"
    assert _unwrap(wrapped) == "https://en.wikipedia.org/wiki/X"
    # a bare protocol-relative link gets https:
    assert _unwrap("//example.com/p").startswith("https://example.com")
    # a normal absolute url is untouched
    assert _unwrap("https://example.com/p") == "https://example.com/p"


def test_detag():
    assert _detag("<b>hi</b> &amp; bye") == "hi & bye"


def test_web_trigger():
    assert _split_web_trigger("latest news /web") == ("latest news", True)
    assert _split_web_trigger("price of gold /search") == ("price of gold", True)
    assert _split_web_trigger("just talking") == ("just talking", False)
    # a path that merely contains a slash must NOT trigger
    assert _split_web_trigger("cat /etc/hosts") == ("cat /etc/hosts", False)


def _fake_hits(n: int) -> list[dict[str, str]]:
    return [
        {"url": f"https://site{i}.example.org/a", "title": f"Result {i}", "snippet": ""}
        for i in range(n)
    ]


_PAGE = "<html><head><title>T</title></head><body><article>" + ("<p>Some real sentence about x. </p>" * 30) + "</article></body></html>"


def _patch_web(monkeypatch, search):
    from agent.research.fetch import Fetched

    monkeypatch.setattr("agent.tools.web._search_ddg", search)
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(url, html=_PAGE))


def _sources(out: str) -> int:
    return sum(1 for line in out.splitlines() if line.startswith("[") and "] " in line[:6])


def test_num_results_reads_the_requested_page_count(monkeypatch):
    _patch_web(monkeypatch, lambda query, k: _fake_hits(k))
    tool = make_web_tools(MagicMock())[0]
    assert _sources(tool.handler(query="x", num_results=15)) == 15


def test_num_results_clamps_to_hard_max(monkeypatch):
    captured = {}

    def fake_search(query, k):
        captured["k"] = k
        return _fake_hits(k)

    _patch_web(monkeypatch, fake_search)
    tool = make_web_tools(MagicMock())[0]
    out = tool.handler(query="x", num_results=50)
    assert captured["k"] <= 23                      # 20 sources + a few spare for dedupe
    assert _sources(out) == 20


def test_extract_next_page_params_finds_the_next_button_fields():
    page = '''
    <form action="/html/" method="post">
      <input type="submit" class='btn btn--alt' value="Next" />
      <input type="hidden" name="q" value="my query" />
      <input type="hidden" name="s" value="10" />
      <input type="hidden" name="nextParams" value="" />
      <input type="hidden" name="v" value="l" />
      <input type="hidden" name="o" value="json" />
      <input type="hidden" name="dc" value="11" />
      <input type="hidden" name="api" value="d.js" />
      <input type="hidden" name="vqd" value="4-abc123" />
      <input name="kl" value="wt-wt" type="hidden" />
    </form>
    '''
    params = _extract_next_page_params(page)
    assert params == {
        "vqd": "4-abc123", "dc": "11", "nextParams": "",
        "v": "l", "o": "json", "api": "d.js", "kl": "wt-wt",
    }


def test_extract_next_page_params_returns_none_with_no_next_button():
    assert _extract_next_page_params("<html>no next button here</html>") is None


def _ddg_page(result_hrefs: list[str], has_next: bool) -> str:
    """Build a minimal fake DDG results page for _search_ddg to parse."""
    results = "".join(
        f'<a class="result__a" href="{href}">Title {i}</a>'
        f'<a class="result__snippet">Snippet {i}</a>'
        for i, href in enumerate(result_hrefs)
    )
    next_form = (
        '<form action="/html/" method="post">'
        '<input type="submit" value="Next" />'
        '<input type="hidden" name="vqd" value="4-xyz" />'
        "</form>"
        if has_next
        else ""
    )
    return f"<html>{results}{next_form}</html>"


def test_search_ddg_paginates_past_one_pages_worth_of_results(monkeypatch):
    # Page 1 has 10 results and a "Next" button; page 2 has 8 more. Asking
    # for 15 must pull from both pages, not silently cap at page 1's 10.
    page1_hrefs = [f"https://example.com/{i}" for i in range(10)]
    page2_hrefs = [f"https://example.com/{i}" for i in range(10, 18)]
    responses = [_ddg_page(page1_hrefs, has_next=True), _ddg_page(page2_hrefs, has_next=False)]

    calls = []

    def fake_post(url, data=None, **kwargs):
        calls.append(data)
        resp = MagicMock()
        resp.text = responses[len(calls) - 1]
        resp.raise_for_status = lambda: None
        return resp

    monkeypatch.setattr("agent.tools.web.httpx.post", fake_post)
    hits = _search_ddg("test query", 15)
    assert len(hits) == 15
    assert len(calls) == 2
    # Second request must carry the pagination token from the first page.
    assert calls[1]["vqd"] == "4-xyz"
    assert calls[1]["s"] == "10"


def test_search_ddg_stops_when_no_next_page_exists(monkeypatch):
    page1_hrefs = [f"https://example.com/{i}" for i in range(5)]

    def fake_post(url, data=None, **kwargs):
        resp = MagicMock()
        resp.text = _ddg_page(page1_hrefs, has_next=False)
        resp.raise_for_status = lambda: None
        return resp

    monkeypatch.setattr("agent.tools.web.httpx.post", fake_post)
    hits = _search_ddg("test query", 20)
    assert len(hits) == 5  # can't fetch more than actually exists


def test_num_results_defaults_to_five(monkeypatch):
    _patch_web(monkeypatch, lambda query, k: _fake_hits(k))
    tool = make_web_tools(MagicMock())[0]
    assert _sources(tool.handler(query="x")) == 5


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
    print("ok — unwrap, detag, /web trigger")
