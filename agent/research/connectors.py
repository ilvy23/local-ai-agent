"""Answers straight from public APIs - no page scraping, structured data, a known origin.

For some questions the right source is not a web page at all: a package version is on PyPI, an exchange rate comes
from the ECB, the weather from a forecast service, a definition from Wikipedia. `lookup(query)` recognises those
questions, asks the matching API and returns page-like records ({url, title, published, text}) that web_search
shows next to the ordinary results (their domains already carry a trust tier, see research/trust.py).

Each connector fails quietly (a down API must never break a search) and only runs when the query matches its
pattern. All network access goes through `_get_json`, the one seam tests replace.
"""

from __future__ import annotations

import re
import urllib.parse
from datetime import date
from typing import Any, Callable

import httpx

TIMEOUT = 6.0
_UA = {"User-Agent": "agent-research/1.0 (local assistant; contact: none)", "Accept": "application/json"}
_CURRENCIES = frozenset("EUR USD GBP JPY CHF CAD AUD NZD SEK NOK DKK PLN CZK HUF CNY INR BRL MXN KRW TRY ZAR SGD HKD".split())


def _get_json(url: str, params: dict[str, Any] | None = None, timeout: float = TIMEOUT) -> Any:
    r = httpx.get(url, params=params, headers=_UA, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _get_text(url: str, params: dict[str, Any] | None = None, timeout: float = TIMEOUT) -> str:
    r = httpx.get(url, params=params, headers={"User-Agent": _UA["User-Agent"]}, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.text


def _record(url: str, title: str, text: str, published: str = "") -> dict[str, Any]:
    return {"url": url, "title": title, "published": published, "text": text.strip(),
            "signals": {"byline": True, "ads": 0, "injection": 0}}


# --------------------------------------------------------------------------------------------------- connectors


def pypi(name: str) -> list[dict]:
    data = _get_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    info, files = data["info"], data.get("urls") or []
    uploaded = (files[0].get("upload_time_iso_8601") or "")[:10] if files else ""
    text = (f"{info['name']} {info['version']} on PyPI. {info.get('summary') or ''} "
            f"Requires Python {info.get('requires_python') or 'unspecified'}. Latest release uploaded {uploaded or 'date unknown'}. "
            f"License: {info.get('license') or (info.get('license_expression') or 'not stated')}. "
            f"Home page: {info.get('home_page') or (info.get('project_urls') or {}).get('Homepage') or 'not stated'}.")
    return [_record(f"https://pypi.org/project/{info['name']}/", f"{info['name']} {info['version']} - PyPI", text, uploaded)]


def npm(name: str) -> list[dict]:
    data = _get_json(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@')}/latest")
    text = (f"{data['name']} {data['version']} on npm. {data.get('description') or ''} License: {data.get('license') or 'not stated'}. "
            f"Node engine: {(data.get('engines') or {}).get('node', 'unspecified')}.")
    return [_record(f"https://www.npmjs.com/package/{data['name']}", f"{data['name']} {data['version']} - npm", text)]


def wikipedia(topic: str, lang: str = "en") -> list[dict]:
    found = _get_json(f"https://{lang}.wikipedia.org/w/api.php",
                      {"action": "query", "list": "search", "srsearch": topic, "format": "json", "srlimit": 1})
    hits = (found.get("query") or {}).get("search") or []
    if not hits:
        return []
    title = hits[0]["title"]
    summary = _get_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}")
    if not summary.get("extract"):
        return []
    page = ((summary.get("content_urls") or {}).get("desktop") or {}).get("page") or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    return [_record(page, f"{summary.get('title', title)} - Wikipedia", summary["extract"], (summary.get("timestamp") or "")[:10])]


def weather(city: str) -> list[dict]:
    geo = (_get_json("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1}).get("results") or [])
    if not geo:
        return []
    g = geo[0]
    w = _get_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum"})
    cur, day = w["current"], w["daily"]
    where = ", ".join(x for x in (g["name"], g.get("admin1"), g.get("country")) if x)
    lines = [f"Weather in {where}, read {date.today().isoformat()}: now {cur['temperature_2m']} °C, wind {cur['wind_speed_10m']} km/h, "
             f"humidity {cur['relative_humidity_2m']} %."]
    for i, d in enumerate(day["time"]):
        lines.append(f"{d}: {day['temperature_2m_min'][i]} to {day['temperature_2m_max'][i]} °C, precipitation {day['precipitation_sum'][i]} mm.")
    return [_record("https://open-meteo.com/en/docs", f"Weather forecast {g['name']} - Open-Meteo", " ".join(lines), date.today().isoformat())]


def exchange(amount: float, src: str, dst: str) -> list[dict]:
    data = _get_json("https://api.frankfurter.app/latest", {"from": src, "to": dst, "amount": amount})
    rate = data["rates"][dst]
    text = (f"{amount:g} {src} = {rate:g} {dst} (European Central Bank reference rate of {data['date']}). "
            f"1 {src} = {rate / amount:g} {dst}.")
    return [_record(f"https://api.frankfurter.app/latest?from={src}&to={dst}", f"{src}/{dst} exchange rate - ECB via Frankfurter", text, data["date"])]


def arxiv(terms: str) -> list[dict]:
    xml = _get_text("http://export.arxiv.org/api/query", {"search_query": f"all:{terms}", "max_results": 3, "sortBy": "relevance"})
    out = []
    for entry in re.findall(r"(?s)<entry>(.*?)</entry>", xml)[:3]:
        def tag(name: str) -> str:
            m = re.search(rf"(?s)<{name}[^>]*>(.*?)</{name}>", entry)
            return " ".join(m.group(1).split()) if m else ""
        authors = re.findall(r"(?s)<name>(.*?)</name>", entry)
        out.append(_record(tag("id").replace("http://", "https://"), tag("title") + " - arXiv",
                           f"{tag('title')}. {', '.join(authors[:4])}{' et al.' if len(authors) > 4 else ''}. {tag('summary')[:900]}",
                           tag("published")[:10]))
    return out


def github(terms: str) -> list[dict]:
    items = _get_json("https://api.github.com/search/repositories", {"q": terms, "per_page": 3, "sort": "stars"}).get("items") or []
    return [_record(i["html_url"], f"{i['full_name']} - GitHub",
                    f"{i['full_name']}: {i.get('description') or 'no description'}. {i['stargazers_count']} stars, last push "
                    f"{(i.get('pushed_at') or '')[:10]}, language {i.get('language') or 'n/a'}, license "
                    f"{(i.get('license') or {}).get('spdx_id') or 'none stated'}.", (i.get("pushed_at") or "")[:10]) for i in items]


def hackernews(terms: str) -> list[dict]:
    hits = _get_json("https://hn.algolia.com/api/v1/search", {"query": terms, "tags": "story", "hitsPerPage": 3}).get("hits") or []
    return [_record(f"https://news.ycombinator.com/item?id={h['objectID']}", f"{h['title']} - Hacker News",
                    f"{h['title']} ({h.get('points', 0)} points, {h.get('num_comments', 0)} comments). Link: {h.get('url') or 'discussion only'}",
                    (h.get("created_at") or "")[:10]) for h in hits if h.get("title")]


# --------------------------------------------------------------------------------------------------- routing

_NAME = r"([a-z0-9][a-z0-9._@/-]{1,60})"
_PYPI = re.compile(rf"(?:\bpypi\b\s+{_NAME}|\bpip install\s+{_NAME}|\bpython (?:package|library|module)\s+{_NAME}|{_NAME}\s+(?:on\s+)?pypi\b)")
_NPM = re.compile(rf"(?:\bnpm (?:install |i |package )?{_NAME}|{_NAME}\s+(?:on\s+)?npm\b)")
_WEATHER = re.compile(r"\b(?:weather|wetter|forecast|wettervorhersage)\b.*?\b(?:in|for|für|fuer|at|bei)\s+([a-zäöüß][a-zäöüß .'-]{1,40}?)(?:\s+(?:today|tomorrow|now|heute|morgen|jetzt))?\s*\??$", re.I)
_FX = re.compile(r"(?:(\d+(?:[.,]\d+)?)\s*)?\b([a-z]{3})\b\s*(?:to|in|nach|zu|into|->|→)\s*\b([a-z]{3})\b", re.I)
_DEFINE = re.compile(r"^(?:what is|what are|what was|who is|who was|who are|wer ist|wer war|was ist|was sind|was war|define|definition of|explain)\s+(?:an? |the |ein |eine |der |die |das )?(.{2,60}?)\s*\??$", re.I)
_ARXIV = re.compile(r"\b(?:arxiv|preprint|research paper|paper on|papers on)\b", re.I)
_GITHUB = re.compile(r"\b(?:github|git repo|repository|open[- ]source project)\b", re.I)
_HN = re.compile(r"\b(?:hacker ?news|hn discussion)\b", re.I)
_FILLER = re.compile(r"(?i)\b(?:arxiv|preprint|research|papers?|paper on|on|about|github|repos?|repository|open[- ]source|project|the|latest|new|hacker ?news|hn|discussion|find|show|me|a|an|for)\b")


def _topic(query: str) -> str:
    return " ".join(_FILLER.sub(" ", query).split())


def plan(query: str) -> list[tuple[str, Callable[[], list[dict]]]]:
    """The connectors that fit `query`: [(name, call)]. Pure matching, no network."""
    q = " ".join(query.strip().split())
    low = q.lower()
    calls: list[tuple[str, Callable[[], list[dict]]]] = []
    if m := _PYPI.search(low):
        name = next(g for g in m.groups() if g)
        if name not in ("on", "the", "a", "package", "for"):
            calls.append(("pypi", lambda n=name: pypi(n)))
    if m := _NPM.search(low):
        name = next(g for g in m.groups() if g)
        if name not in ("on", "the", "a", "package", "for"):
            calls.append(("npm", lambda n=name: npm(n)))
    if m := _WEATHER.search(q):
        calls.append(("weather", lambda c=m.group(1).strip(): weather(c)))
    if m := _FX.search(q):
        src, dst = m.group(2).upper(), m.group(3).upper()
        if src in _CURRENCIES and dst in _CURRENCIES and src != dst:
            amount = float((m.group(1) or "1").replace(",", "."))
            calls.append(("exchange", lambda a=amount, s=src, d=dst: exchange(a, s, d)))
    if _ARXIV.search(low) and _topic(q):
        calls.append(("arxiv", lambda t=_topic(q): arxiv(t)))
    if _GITHUB.search(low) and _topic(q):
        calls.append(("github", lambda t=_topic(q): github(t)))
    if _HN.search(low) and _topic(q):
        calls.append(("hackernews", lambda t=_topic(q): hackernews(t)))
    if (m := _DEFINE.match(q)) and not calls:
        calls.append(("wikipedia", lambda t=m.group(1): wikipedia(t, "de" if re.match(r"(?i)(wer|was)\b", q) else "en")))
    return calls


def lookup(query: str) -> list[dict]:
    """Records from every connector that fits `query`. Never raises."""
    out: list[dict] = []
    for _name, call in plan(query):
        try:
            out.extend(call())
        except Exception:  # noqa: BLE001 - an API that is down or changed must not break a search
            continue
    return out
