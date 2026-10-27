"""Coding-only web access: search, filter to an allowlist, read full pages.

The agent may only reach documentation/Q&A sites relevant to coding, and unlike
the general web tool it returns the WHOLE cleaned article (not a 2500-char
snippet) so it can actually read the answer. Reuses the general tool's search +
headers so there's one HTTP path to maintain.
"""

from __future__ import annotations

import html
import re
import urllib.parse

import httpx

from agent.events import NULL, Events
from agent.tools import web as _web

# Config-editable allowlist. Hosts must equal or be a subdomain of these.
ALLOWLIST = (
    "stackoverflow.com", "stackexchange.com", "superuser.com", "askubuntu.com",
    "docs.python.org", "python.org", "developer.mozilla.org", "github.com",
    "readthedocs.io", "pypi.org", "devdocs.io", "man7.org", "docs.rs",
    "pkg.go.dev", "cppreference.com", "realpython.com", "geeksforgeeks.org",
    "wiki.archlinux.org", "kernel.org",
)
_FULL_CHARS = 15000


def _allowed(url: str, allowlist=ALLOWLIST) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in allowlist)


def _fetch_full(url: str, max_chars: int = _FULL_CHARS) -> str:
    r = httpx.get(url, headers=_web._HEADERS, timeout=20, follow_redirects=True)
    r.raise_for_status()
    body = re.sub(r"(?is)<(script|style|noscript|nav|header|footer|svg)[^>]*>.*?</\1>",
                  " ", r.text)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", body))
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def search_docs(query: str, events: Events = NULL, read: int = 3,
                allowlist=ALLOWLIST) -> str:
    try:
        hits = _web._search_ddg(query, 10)
    except Exception as exc:  # noqa: BLE001
        return f"web search failed: {exc}"
    allowed = [h for h in hits if _allowed(h["url"], allowlist)]
    if not allowed:
        return (f"No results on allowed coding sites for: {query}\n"
                "(allowed: StackOverflow, official docs, MDN, GitHub, PyPI, readthedocs, …)")
    blocks = []
    for i, h in enumerate(allowed[:read], 1):
        events.emit("web_docs", url=h["url"])
        block = f"[{i}] {h['title']}\n{h['url']}"
        try:
            block += "\n" + _fetch_full(h["url"])
        except Exception as exc:  # noqa: BLE001 - a dead link isn't fatal
            block += f"\n(could not open: {exc})"
        blocks.append(block)
    return "\n\n".join(blocks)
