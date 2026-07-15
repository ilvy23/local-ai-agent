"""Web search + page-read tool for the agent.

No API key, no new dependency: it scrapes DuckDuckGo's HTML endpoint with the
httpx already in the project, unwraps the redirect links, and pulls readable
text from each hit so the model has real content to answer from — not just
titles. It prints every site as it visits it, so the terminal shows the
research happening live.

ponytail: regex HTML parsing, not bs4/trafilatura. DDG's result markup is
stable and this dodges two dependencies; swap in a real parser only if the
scrape starts missing results.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from typing import Any

import httpx

from companion.tools.registry import Tool

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
# Full browser-ish header set — a bare User-Agent gets 403'd by Wikipedia and
# friends; these get most public pages to serve us the article.
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Referer": "https://duckduckgo.com/",
}
_DDG_HTML = "https://html.duckduckgo.com/html/"
_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)".*?>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.S)
_PAGE_CHARS = 2500      # readable text kept per page (enough to answer from)
_MAX_READ = 5           # never fetch more than this many pages in one search


def _detag(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def _unwrap(href: str) -> str:
    """DDG wraps results as //duckduckgo.com/l/?uddg=<real-url>. Unwrap it."""
    if "uddg=" in href:
        q = urllib.parse.urlparse(href).query
        got = urllib.parse.parse_qs(q).get("uddg")
        if got:
            return urllib.parse.unquote(got[0])
    return href if href.startswith("http") else f"https:{href}"


def _search_ddg(query: str, k: int) -> list[dict[str, str]]:
    r = httpx.post(_DDG_HTML, data={"q": query}, headers=_HEADERS,
                   timeout=15, follow_redirects=True)
    r.raise_for_status()
    titles = _RESULT_RE.findall(r.text)
    snippets = _SNIPPET_RE.findall(r.text)
    hits: list[dict[str, str]] = []
    for i, (href, title) in enumerate(titles[:k]):
        hits.append({
            "url": _unwrap(href),
            "title": _detag(title),
            "snippet": _detag(snippets[i]) if i < len(snippets) else "",
        })
    return hits


def _read_page(url: str) -> str:
    r = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
    r.raise_for_status()
    body = re.sub(r"(?is)<(script|style|noscript|nav|header|footer|svg)[^>]*>.*?</\1>",
                  " ", r.text)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", body))
    return re.sub(r"\s+", " ", text).strip()[:_PAGE_CHARS]


def make_web_tools(console: Any) -> list[Tool]:
    """Build the web_search tool bound to `console` so it can narrate live."""

    def web_search(query: str, num_results: int = 5, read: bool = True, **_: Any) -> str:
        console.print(f"\n[bold cyan]🌐 searching:[/bold cyan] {query}")
        try:
            hits = _search_ddg(query, min(int(num_results), 8))
        except Exception as exc:  # noqa: BLE001 - surface the failure to the model
            console.print(f"[red]  search failed: {exc}[/red]")
            return f"web search failed: {exc}"
        if not hits:
            console.print("[dim]  no results[/dim]")
            return "No results found."

        blocks: list[str] = []
        for i, h in enumerate(hits, 1):
            host = urllib.parse.urlparse(h["url"]).netloc or h["url"]
            console.print(f"  [green]{i}.[/green] {h['title']}  [dim]{host}[/dim]")
            block = f"[{i}] {h['title']}\n{h['url']}"
            if h["snippet"]:
                block += f"\n{h['snippet']}"
            if read and i <= _MAX_READ:
                console.print(f"     [dim]↳ reading {host} …[/dim]", end="")
                try:
                    text = _read_page(h["url"])
                    console.print(f"[dim] {len(text)} chars[/dim]")
                    block += f"\nPAGE TEXT: {text}"
                except Exception as exc:  # noqa: BLE001 - a dead link isn't fatal
                    console.print(f"[red] unreachable[/red]")
                    block += f"\n(could not open: {exc})"
            blocks.append(block)
        console.print()
        return "\n\n".join(blocks)

    tool = Tool(
        name="web_search",
        description=(
            "Search the live internet and read the top pages. Use this for "
            "anything current, factual, or beyond your training: news, prices, "
            "docs, people, events, 'latest', 'today'. Returns titles, URLs and "
            "the actual page text. Answer in detail and cite the source URLs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "num_results": {"type": "integer",
                                "description": "How many results (default 5)."},
                "read": {"type": "boolean",
                         "description": "Also fetch page text (default true)."},
            },
            "required": ["query"],
        },
        handler=web_search,
        risk="safe",
    )
    return [tool]
