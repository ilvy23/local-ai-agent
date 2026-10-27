"""Web research tools for the agent: web_search and web_fetch.

Search is DuckDuckGo's HTML endpoint (no API key). Everything after the search is `agent.research`:
pages are fetched in parallel, reduced to the main article, and cut down to the paragraphs that match the
question (not the first 4000 characters of a page); results and pages are cached in the memory database
(`WebMemory`), and every search leaves a short dated research note in the vector index so later turns can
recall it. Output is plain text with numbered sources and dates, small enough not to eat the context window.

ponytail: regex HTML extraction, not bs4/trafilatura - the DDG markup and article pages are simple enough,
and it avoids two dependencies. Swap in a real parser only if extraction starts missing content.
"""

from __future__ import annotations

import html
import os
import re
import time
import urllib.parse
from collections import Counter
from datetime import date
from typing import Any

import httpx

from agent.research import connectors, evidence as turn_evidence, extract, injection, taint, trust, verdict
from agent.research.display import Narrator
from agent.research.fetch import Fetched, fetch_many, fetch_one
from agent.research.memory import WebMemory, query_key, ttl_for
from agent.tools.registry import Tool

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
_PAGE_CHARS = 4000      # kept for agent/coding/webdocs.py; web_search itself now budgets by total size
_HARD_MAX_READ = 20     # absolute ceiling on pages read in one search, even
                        # if the model asks for more - num_results (default 5)
                        # is the real per-call count, clamped to this


def _detag(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


# DDG serves sponsored links through the same result markup. They're shops, not
# answers, and they burn the top slots — one query returned two merch stores
# above Wikipedia.
_AD_MARKERS = ("duckduckgo.com/y.js", "ad_provider=", "ad_domain=", "/y.js?ad_")


def _is_ad(url: str) -> bool:
    return any(marker in url for marker in _AD_MARKERS)


def _unwrap(href: str) -> str:
    """DDG wraps results as //duckduckgo.com/l/?uddg=<real-url>. Unwrap it."""
    href = html.unescape(href)  # hrefs arrive HTML-escaped (&amp; -> &)
    if "uddg=" in href:
        q = urllib.parse.urlparse(href).query
        got = urllib.parse.parse_qs(q).get("uddg")
        if got:
            return urllib.parse.unquote(got[0])
    return href if href.startswith("http") else f"https:{href}"


_NEXT_PAGE_FIELDS = ("vqd", "dc", "nextParams", "v", "o", "api", "kl")
_MAX_DDG_PAGES = 3  # DDG's HTML page yields ~10 results/page; 3 covers _HARD_MAX_READ (20)


def _extract_next_page_params(page_html: str) -> dict[str, str] | None:
    """Pull the hidden form fields DDG's own "Next" button submits to fetch
    the next page of results - a `vqd` continuation token plus a few fixed
    fields, none of which are stateless (no simple `s=<offset>` alone works)
    - without this, requesting more than ~10 results just re-reads the same
    single page's results over and over. Returns None if there's no next
    page (fewer results exist than one page holds).
    """
    if 'value="Next"' not in page_html and "value='Next'" not in page_html:
        return None
    params: dict[str, str] = {}
    for name in _NEXT_PAGE_FIELDS:
        m = re.search(rf'name=["\']?{name}["\']?\s+value=["\']([^"\']*)["\']', page_html)
        if m:
            params[name] = m.group(1)
    return params or None


class SearchBlocked(Exception):
    """A search engine refused to answer (bot check, rate limit). Tried-and-blocked is not the same as "no results"."""


_BLOCK_PAGE = re.compile(r"(?i)(anomaly|captcha|unusual traffic|are you a (human|robot)|bots use duckduckgo too|verify you are)")


def _search_ddg(query: str, k: int, df: str | None = None) -> list[dict[str, str]]:
    """`df` limits the age of results: "d" day, "w" week, "m" month, "y" year."""
    hits: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    extra_params: dict[str, str] = {}
    offset = 0
    for page in range(_MAX_DDG_PAGES):
        data = {"q": query, **extra_params}
        if df:
            data["df"] = df
        if offset:
            data["s"] = str(offset)
        r = httpx.post(_DDG_HTML, data=data, headers=_HEADERS,
                       timeout=15, follow_redirects=True)
        if r.status_code in (202, 403, 429):                       # DDG answers a bot challenge with 202, a hard block with 403/429
            raise SearchBlocked(f"duckduckgo answered http {r.status_code}")
        r.raise_for_status()
        titles = _RESULT_RE.findall(r.text)
        if not titles and page == 0 and _BLOCK_PAGE.search(r.text):
            raise SearchBlocked("duckduckgo showed a bot check")
        snippets = _SNIPPET_RE.findall(r.text)
        new_this_page = 0
        for i, (href, title) in enumerate(titles):
            url = _unwrap(href)
            if _is_ad(url) or url in seen_urls:
                continue  # sponsored link, or a dupe DDG sometimes repeats across pages
            seen_urls.add(url)
            hits.append({
                "url": url,
                "title": _detag(title),
                "snippet": _detag(snippets[i]) if i < len(snippets) else "",
            })
            new_this_page += 1
            if len(hits) >= k:
                return hits
        if new_this_page == 0 or page == _MAX_DDG_PAGES - 1:
            break
        next_params = _extract_next_page_params(r.text)
        if not next_params:
            break  # no more pages
        extra_params = next_params
        offset += 10
    return hits




_FRESH_SEARX = {"day": "day", "week": "week", "month": "month", "year": "year"}


def _search_searxng(query: str, k: int, df: str | None = None) -> list[dict[str, str]]:
    """Your own SearXNG instance (AGENT_SEARXNG_URL, e.g. http://127.0.0.1:8888): aggregates many engines, no bot checks.
    It must allow `format=json` in its settings.yml (search.formats)."""
    base = os.environ.get("AGENT_SEARXNG_URL", "").rstrip("/")
    params = {"q": query, "format": "json"}
    if df:
        params["time_range"] = {"d": "day", "w": "week", "m": "month", "y": "year"}[df]
    r = httpx.get(f"{base}/search", params=params, headers={"User-Agent": _UA}, timeout=20)
    if r.status_code in (403, 429):
        raise SearchBlocked(f"searxng answered http {r.status_code} (is format=json enabled?)")
    r.raise_for_status()
    return [{"url": x["url"], "title": _detag(x.get("title", "")), "snippet": _detag(x.get("content", ""))}
            for x in r.json().get("results", []) if x.get("url", "").startswith("http")][:k]


def _search_brave(query: str, k: int, df: str | None = None) -> list[dict[str, str]]:
    """Brave Search API (BRAVE_API_KEY): an independent index with a proper API instead of scraping."""
    params = {"q": query, "count": str(min(k, 20))}
    if df:
        params["freshness"] = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}[df]
    r = httpx.get("https://api.search.brave.com/res/v1/web/search", params=params, timeout=15,
                  headers={"X-Subscription-Token": os.environ.get("BRAVE_API_KEY", ""), "Accept": "application/json"})
    if r.status_code in (401, 403, 429):
        raise SearchBlocked(f"brave answered http {r.status_code}")
    r.raise_for_status()
    return [{"url": x["url"], "title": _detag(x.get("title", "")), "snippet": _detag(x.get("description", ""))}
            for x in (r.json().get("web") or {}).get("results", [])][:k]


def _engines() -> list[tuple[str, Any]]:
    """Engines to try, best first. Looked up at call time so tests (and the env) can swap them."""
    chain: list[tuple[str, Any]] = []
    if os.environ.get("AGENT_SEARXNG_URL", "").strip():
        chain.append(("searxng", _search_searxng))
    chain.append(("ddg", _search_ddg))
    if os.environ.get("BRAVE_API_KEY", "").strip():
        chain.append(("brave", _search_brave))
    return chain


def _search(query: str, k: int, df: str | None = None) -> tuple[list[dict[str, str]], str]:
    """(hits, engine that answered). Falls through blocked or failing engines; if none answers, raises with the reasons."""
    problems: list[str] = []
    for name, engine in _engines():
        try:
            hits = engine(query, k, df=df) if df else engine(query, k)
        except SearchBlocked as exc:
            problems.append(f"{name}: blocked ({exc})")
            continue
        except Exception as exc:  # noqa: BLE001 - one engine down must not end the search
            problems.append(f"{name}: {type(exc).__name__}")
            continue
        if hits:
            return hits, name
        problems.append(f"{name}: no results")
    if problems and all(p.endswith("no results") for p in problems):
        return [], ""
    raise SearchBlocked("; ".join(problems))


_FRESHNESS = {"day": "d", "week": "w", "month": "m", "year": "y"}
_TOTAL_BUDGET = 7000     # characters of page passages per search (the old reader sent up to 20000)
_MAX_PER_DOMAIN = 2
_SPARE_HITS = 3          # extra hits requested so unreadable or duplicate domains can be replaced
_READ_MAX_AGE = 7 * 86400


_MAX_QUERIES = 3


def _fuse(results: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """Merge several result lists (reciprocal-rank fusion): a page several queries agree on rises to the top."""
    if len(results) == 1:
        return results[0]
    score: dict[str, float] = {}
    first: dict[str, dict[str, str]] = {}
    for hits in results:
        for rank, h in enumerate(hits):
            url = h["url"].split("#")[0].rstrip("/")
            score[url] = score.get(url, 0.0) + 1.0 / (60 + rank)
            first.setdefault(url, h)
    return [first[u] for u in sorted(score, key=lambda u: -score[u])]


def _domain(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def _rank(hits: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Keep the engine's order, but at most two results per site so one domain can't fill the list."""
    per_domain: Counter[str] = Counter()
    kept = []
    for h in hits:
        d = _domain(h["url"])
        if per_domain[d] >= _MAX_PER_DOMAIN:
            continue
        per_domain[d] += 1
        kept.append(h)
        if len(kept) >= limit:
            break
    return kept


_WEIGH = ("Weigh each source by its rating (T1 official/primary > T2 reference or news > T3 community > T4 unknown; "
          "'avoid' sources are not evidence). Prefer the newest primary source for current facts, say so when sources "
          "disagree or the evidence is weak or from one source, and never state a figure that no source contains.")


def _rate(url: str, title: str, page: dict) -> trust.Trust:
    sig = page.get("signals") or {}
    return trust.assess(url, title, text=page.get("text", ""), published=page.get("published", ""),
                        signals=sig, injection=int(sig.get("injection", 0)) >= 2)


def _emit(console: Any, event: str, **fields: Any) -> None:
    """Structured event for the GUI (WebConsole.emit_event); the terminal console has none and ignores it."""
    emit = getattr(console, "emit_event", None)
    if callable(emit):
        try:
            emit(event, **fields)
        except Exception:  # noqa: BLE001 - a lost UI event must never break a search
            pass


def _http_fetch(url: str) -> Fetched:
    return fetch_one(url, _HEADERS)


def make_web_tools(console: Any, memory: WebMemory | None = None) -> list[Tool]:
    """Build web_search and web_fetch. `memory` (optional) adds the page/result cache and research notes."""
    view = Narrator(console)
    preloaded: dict[str, dict] = {}        # connector records of the current search: already read, nothing to fetch

    def _load_pages(urls: list[str]) -> tuple[dict[str, dict[str, str]], dict[str, Fetched], set[str]]:
        """Pages by URL: from the cache when fresh, the rest fetched in parallel. (pages, raw fetches, cached urls)"""
        pages: dict[str, dict[str, str]] = {}
        cached: set[str] = set()
        todo: list[str] = []
        fetched_pre: dict[str, Fetched] = {}
        for u in urls:
            if u in preloaded:
                pages[u], fetched_pre[u] = preloaded[u], Fetched(u)
                continue
            hit = memory.page(u, _READ_MAX_AGE) if memory else None
            if hit:
                pages[u] = hit
                cached.add(u)
            else:
                todo.append(u)
        fetched = fetch_many(todo, _HEADERS, fetch=_http_fetch) if todo else {}
        fetched.update(fetched_pre)
        for u, f in fetched.items():
            if u in fetched_pre:
                continue
            if f.status == "ok" and f.html:
                page = extract.extract_page(f.html)
                if page.text:
                    text, removed = injection.scrub(page.text)       # instructions aimed at the assistant never reach it
                    signals = {**trust.page_signals(f.html), "injection": removed}
                    pages[u] = {"title": page.title, "published": page.published, "text": text, "signals": signals}
                    if memory:
                        memory.put_page(u, page.title, page.published, text, signals)
                else:
                    f.status, f.note = "failed", "no readable text"
        return pages, fetched, cached

    def _read_with_replacements(hits, spare, wanted):
        """Read the hits; a site that blocks us (403/429), times out or has no text is replaced by the next
        unread result, so `wanted` pages come back whenever the results hold that many readable ones."""
        pages: dict[str, dict[str, str]] = {}
        fetched: dict[str, Fetched] = {}
        cached: set[str] = set()
        shown = list(hits)
        queue = list(spare)
        batch = list(hits)
        with view.busy("reading", suffix=f"{len(hits)} pages"):
            while batch:
                got_pages, got_fetched, got_cached = _load_pages([h["url"] for h in batch])
                pages.update(got_pages), fetched.update(got_fetched), cached.update(got_cached)
                missing = wanted - len(pages)
                batch = [queue.pop(0) for _ in range(min(missing, len(queue)))] if missing > 0 else []
                shown.extend(batch)
        return shown, pages, fetched, cached

    def _hits_for(q: str, freshness: str, wanted: int) -> tuple[list[dict[str, str]], str, bool]:
        """(hits, engine, from_cache) for one query; raises SearchBlocked / other errors to the caller."""
        key = query_key(q, freshness)
        cached_hits = memory.hits(key, ttl_for(q, freshness)) if memory else None
        if cached_hits is not None:
            return cached_hits, "cache", True
        hits, engine = _search(q, wanted, _FRESHNESS[freshness] if freshness != "any" else None)
        if memory and hits:
            memory.put_hits(key, q, hits)
        return hits, engine, False

    def web_search(query: str, num_results: int = 5, read: bool = True, freshness: str = "any",
                   queries: Any = None, **_: Any) -> str:
        started = time.monotonic()
        num_results = max(1, min(int(num_results), _HARD_MAX_READ))
        freshness = freshness if freshness in _FRESHNESS else "any"
        key = query_key(query, freshness)
        extra = [q.strip() for q in (queries if isinstance(queries, list) else []) if isinstance(q, str) and q.strip()]
        all_queries = list(dict.fromkeys([query, *extra]))[:_MAX_QUERIES]
        view.header("search", " | ".join(all_queries))
        wanted = min(num_results + _SPARE_HITS, _HARD_MAX_READ + _SPARE_HITS)
        results: list[list[dict[str, str]]] = []
        engine, from_cache, problems = "", True, []
        preloaded.clear()
        records = connectors.lookup(query) if read else []                  # APIs first: structured, with a known origin
        for r in records:
            preloaded[r["url"]] = {"title": r["title"], "published": r["published"], "text": r["text"], "signals": r["signals"]}
        with view.busy("searching"):
            for q in all_queries:
                try:
                    got, used, cached_q = _hits_for(q, freshness, wanted)
                except SearchBlocked as exc:
                    problems.append(str(exc))
                    continue
                except Exception as exc:  # noqa: BLE001 - surface the failure to the model
                    view.note(f"search failed: {type(exc).__name__}")
                    return f"web search failed: {exc}"
                results.append(got)
                engine = engine or used
                from_cache = from_cache and cached_q
        if not results and not records:
            view.note("search engines blocked or down")
            return (f"web search failed: every search engine refused or failed ({'; '.join(problems)}). This is not 'no "
                    "results': tell the user the search could not run, and do not answer as if it had. Try web_fetch on "
                    "a specific address, or ask the user to set AGENT_SEARXNG_URL / BRAVE_API_KEY.")
        hits = [{"url": r["url"], "title": r["title"], "snippet": r["text"][:200]} for r in records]
        hits += [h for h in (_fuse(results) if results else []) if h["url"] not in preloaded]
        engine = engine or ("api" if records else "")
        if not hits:
            view.note("no results")
            return "No results found."
        ranked = _rank(hits, num_results + _SPARE_HITS)
        hits, spare = ranked[:num_results], ranked[num_results:]

        pages: dict[str, dict[str, str]] = {}
        fetched: dict[str, Fetched] = {}
        cached: set[str] = set()
        if read:
            hits, pages, fetched, cached = _read_with_replacements(hits, spare, num_results)

        per_page = max(900, min(2200, _TOTAL_BUDGET // max(len(pages), 1)))
        blocks: list[str] = []
        unread: list[str] = []
        sources: list[dict[str, str]] = []
        evidence: list[dict] = []
        weak: list[int] = []
        for i, h in enumerate(hits, 1):
            page = pages.get(h["url"])
            title = (page or {}).get("title") or h["title"]
            when = (page or {}).get("published", "")
            rating = _rate(h["url"], title, page) if page else None
            head = f"[{i}] {title} - {_domain(h['url'])}" + (f" - {when}" if when else "") + (f" - {rating.label()}" if rating else "")
            item = {"n": i, "url": h["url"], "domain": _domain(h["url"]), "title": title, "date": when or None,
                    "trust": rating.as_dict() if rating else None, "status": "ok", "passage": ""}
            if page and rating and rating.level == "avoid":
                blocks.append(f"{head}\n{h['url']}\n(not used as evidence)")
                unread.append(f"[{i}] rated avoid")
                item.update(status="skipped", status_note="rated avoid")
                view.row(i, h["url"], "skipped", "rated avoid")
            elif page:
                best = extract.select_passages(page["text"], query, per_page)
                if extract.relevance(page["text"], query) <= extract.NO_MATCH:
                    weak.append(i)
                    best = "(nothing on this page matches the question closely; it starts:) " + best[:300].rsplit(" ", 1)[0] + " …"
                blocks.append(f"{head}\n{h['url']}\n{injection.frame(best)}")
                taint.add(best)
                sources.append({"url": h["url"], "title": title, "published": when, "best": best})
                item.update(status="cached" if h["url"] in cached else "ok", passage=best)
                view.row(i, h["url"], "cached" if h["url"] in cached else "read",
                         "" if h["url"] in cached else f"{fetched[h['url']].seconds:.1f}s")
            else:
                f = fetched.get(h["url"])
                blocks.append(f"{head}\n{h['url']}\n{h['snippet']}" if h["snippet"] else f"{head}\n{h['url']}")
                if read and f:
                    unread.append(f"[{i}] {f.note or f.status}")
                    item.update(status=f.status, status_note=f.note)
                    view.row(i, h["url"], f.status, f.note if f.note != f.status else "")
                else:
                    item.update(status="skipped", status_note="listed only")
                    view.row(i, h["url"], "listed")
            evidence.append(item)
        took = time.monotonic() - started
        view.footer(f"{len(hits)} sources - {len(pages) - len(cached & set(pages))} read - {len(cached & set(pages))} cached - "
                    f"{took:.1f}s" + (" - results from cache" if from_cache else ""))

        if memory and sources:
            memory.remember(key, query, sources)
        vd = verdict.verdict([{**e, "level": (e["trust"] or {}).get("level", "low"), "text": e["passage"]} for e in evidence])
        groups = verdict.group_origins([{"url": e["url"], "text": e["passage"]} for e in evidence])
        for e, g in zip(evidence, groups):
            e["origin_group"] = g
        turn_evidence.record(vd["label"], [
            {"domain": e["domain"], "date": e["date"] or "", "level": (e["trust"] or {}).get("level", "low"),
             "passage": e["passage"][:600]} for e in evidence if e["status"] in ("ok", "cached")])
        turn_evidence.set_last_search(evidence, vd)
        _emit(console, "sources", query=query, engine=engine or "ddg", searched_at=date.today().isoformat(), items=evidence)
        _emit(console, "verdict", **{k: v for k, v in vd.items() if k != "label"}, label=vd["label"])
        out = [f'Web search "{query}" on {date.today().isoformat()}: {len(hits)} sources.', "", "\n\n".join(blocks)]
        if unread:
            out += ["", "Not read: " + "; ".join(unread)]
        if weak and len(weak) == len(pages):
            out += ["", "None of the pages read matched the question closely. Search again with a more specific query "
                        "(exact names, version, year, the official source) instead of answering from these."]
        out += ["", f"Evidence check: {vd['label'].replace('_', ' ')} ({vd['summary']})."]
        for c in vd["conflicts"][:3]:
            out.append("Sources disagree on " + c["attribute"] + ": " + "; ".join(
                f"{v['value']:g} per " + ",".join(f"[{n}]" for n in v["sources"]) for v in c["values"]) + ".")
        out += ["", "Cite sources as [n]. " + _WEIGH + " " + injection.FRAME_NOTE]
        return "\n".join(out)

    def web_fetch(url: str, focus: str = "", max_chars: int = 6000, offset: int = 0, **_: Any) -> str:
        started = time.monotonic()
        max_chars = max(500, min(int(max_chars), 20000))
        offset = max(0, int(offset))
        view.header("read", url)
        pages, fetched, cached = _load_pages([url]) if url.startswith("http") else ({}, {}, set())
        page = pages.get(url)
        if not page:
            f = fetched.get(url)
            reason = (f.note or f.status) if f else "not a web address"
            view.row(1, url, (f.status if f else "failed"), reason if f and f.note != f.status else "")
            return f"Could not read {url}: {reason}"
        text = page["text"]
        if focus.strip():
            body = extract.select_passages(text, focus, max_chars)
            span = f"passages about {focus!r}"
        else:
            body = text[offset:offset + max_chars]
            end = offset + len(body)
            span = f"characters {offset}-{end} of {len(text)}" + (f"; call again with offset={end} for more" if end < len(text) else "")
        view.row(1, url, "cached" if url in cached else "read", "" if url in cached else f"{fetched[url].seconds:.1f}s")
        view.footer(f"{len(body)} of {len(text)} characters - {time.monotonic() - started:.1f}s")
        meta = f"{page['title']}" + (f" - published {page['published']}" if page["published"] else "")
        meta += f" - {_rate(url, page['title'], page).label()}"
        taint.add(body)
        level = _rate(url, page["title"], page).level
        turn_evidence.record(verdict.verdict([{"n": 1, "url": url, "status": "ok", "level": level, "date": page["published"],
                                               "text": body}])["label"],
                             [{"domain": _domain(url), "date": page["published"], "level": level, "passage": body[:600]}])
        return f"{meta}\n{url}\n({span})\n\n{injection.frame(body)}\n\n{injection.FRAME_NOTE}"

    search_tool = Tool(
        name="web_search",
        description=(
            "Search the live internet and read the best pages. Use it for anything current, factual or beyond "
            "your training: news, prices, docs, people, events, 'latest', 'today'. Returns numbered sources with "
            "dates and the passages of each page that match the query. Write a specific query (names, versions, "
            "year). Answer from the sources and cite their numbers and URLs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for - specific, like a search engine query."},
                "queries": {"type": "array", "items": {"type": "string"}, "description": (
                    "Up to 2 extra phrasings or sub-questions (e.g. with the year or version, or the official name). "
                    "Results are merged; use for broad or ambiguous questions.")},
                "num_results": {"type": "integer", "description": (
                    "Sources to read (default 5, max 20). Use more only for broad research; each page adds latency.")},
                "freshness": {"type": "string", "enum": ["any", "day", "week", "month", "year"],
                              "description": "Limit results to this age. Use for news and anything that changes."},
                "read": {"type": "boolean", "description": "Also read the pages (default true); false = titles only."},
            },
            "required": ["query"],
        },
        handler=web_search,
        risk="safe",
    )
    fetch_tool = Tool(
        name="web_fetch",
        description=(
            "Read one specific web page (a URL from a search result or one the user gave). With `focus` you get "
            "the passages about that topic; without it the page text from `offset`, up to `max_chars`, and a hint "
            "for the next offset. Use it to go deeper into a source instead of searching again."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full http(s) address."},
                "focus": {"type": "string", "description": "What you are looking for on the page (optional)."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return (500-20000, default 6000)."},
                "offset": {"type": "integer", "description": "Start position for reading a long page in parts."},
            },
            "required": ["url"],
        },
        handler=web_fetch,
        risk="safe",
    )
    return [search_tool, fetch_tool]
