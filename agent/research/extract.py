"""Turn a web page into clean text and pick the passages that answer the question.

The old reader kept the first 4000 characters of every page, which are mostly menus, cookie banners
and teasers. This keeps the main article (<article>/<main>), drops boilerplate lines, finds the
publication date, and then scores paragraphs against the query (term overlap weighted by how rare the
word is on that page) so the model gets the few paragraphs that matter instead of a page head.
Pure functions, no network, no model: fast and testable.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass

MAX_TEXT_CHARS = 60_000
CHUNK_TARGET = 520

_DROP_BLOCKS = re.compile(
    r"(?is)<(script|style|noscript|svg|iframe|form|nav|header|footer|aside|template|button|select|dialog)\b[^>]*>.*?</\1>"
)
_COMMENTS = re.compile(r"(?s)<!--.*?-->")
# Text a browser would not show a reader (display:none, aria-hidden, zero font size ...) is a classic place to hide
# instructions for an AI. shortcut: a hidden block ends at the first closing tag of its own name, so a hidden <div>
# that nests another <div> leaves a little residue; the injection scan (research.injection) still sees that.
_HIDDEN = re.compile(
    r"""(?is)<(div|span|p|section|li|td|a|b|i|em|strong|small|font)\b[^>]*"""
    r"""(?:\shidden(?=[\s>=])|aria-hidden=["']true["']|style=["'][^"']*"""
    r"""(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?![.\d]*[1-9])|opacity\s*:\s*0(?![.\d]))"""
    r"""[^"']*["'])[^>]*>.*?</\1>"""
)
_TITLE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
_OG_TITLE = re.compile(r'(?is)<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']')
_DATE_META = re.compile(
    r'(?is)<meta[^>]+(?:property|name|itemprop)=["\'](?:article:published_time|datepublished|date|pubdate|'
    r'og:published_time|publishdate|dc\.date(?:\.issued)?)["\'][^>]+content=["\']([^"\']+)["\']'
)
_DATE_LD = re.compile(r'(?is)"datePublished"\s*:\s*"([^"]+)"')
_DATE_TIME_TAG = re.compile(r'(?is)<time[^>]+datetime=["\']([^"\']+)["\']')
_ISO_DATE = re.compile(r"(20\d{2}|19\d{2})-(\d{2})-(\d{2})")
_ARTICLE = re.compile(r"(?is)<article\b[^>]*>(.*?)</article>")
_MAIN = re.compile(r"(?is)<main\b[^>]*>(.*?)</main>")
_BLOCK_END = re.compile(r"(?i)</(p|div|li|tr|section|blockquote|pre|table|ul|ol|dd|dt|figure)>|<br\s*/?>|<hr\s*/?>")
_HEADING = re.compile(r"(?is)<h([1-4])\b[^>]*>(.*?)</h\1>")
_LI_OPEN = re.compile(r"(?i)<li\b[^>]*>")
_TAG = re.compile(r"(?s)<[^>]+>")
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

_BOILERPLATE = re.compile(
    r"(?i)(cookie|accept all|reject all|consent|subscribe|newsletter|sign in|log in|sign up|register now|"
    r"privacy policy|terms of (use|service)|all rights reserved|©|follow us|share this|read more|skip to|"
    r"advertisement|sponsored|related articles|you might also like|back to top|datenschutz|impressum|"
    r"alle rechte vorbehalten|jetzt anmelden|cookies zustimmen)"
)

_STOP = frozenset(
    "the a an and or of to in on for with is are was were be been it its this that these those as at by from "
    "what which who how why when where do does did can could should would will about into than then so if "
    "der die das und oder von zu in im auf für mit ist sind war waren sein es ein eine einen einem einer "
    "als bei aus nach wie was wer wo wann warum kann wird nicht auch noch nur mehr".split()
)


@dataclass
class Page:
    title: str
    published: str          # YYYY-MM-DD or ""
    text: str


def _clean_inline(s: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", html.unescape(_TAG.sub(" ", s))).strip()


def _normalize_date(raw: str) -> str:
    m = _ISO_DATE.search(raw or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def find_date(page_html: str) -> str:
    for pattern in (_DATE_META, _DATE_LD, _DATE_TIME_TAG):
        m = pattern.search(page_html)
        if m:
            d = _normalize_date(m.group(1))
            if d:
                return d
    return ""


def find_title(page_html: str) -> str:
    for pattern in (_OG_TITLE, _TITLE):
        m = pattern.search(page_html)
        if m:
            t = _clean_inline(m.group(1))
            if t:
                return t[:200]
    return ""


def _main_html(body: str) -> str:
    """The article if there is a substantial one, else <main>, else the whole body."""
    for pattern in (_ARTICLE, _MAIN):
        blocks = pattern.findall(body)
        if blocks:
            best = max(blocks, key=len)
            if len(_TAG.sub("", best)) > 400:
                return best
    return body


_TABLE = re.compile(r"(?is)<table\b[^>]*>(.*?)</table>")
_ROW = re.compile(r"(?is)<tr\b[^>]*>(.*?)</tr>")
_CELL = re.compile(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]>")
MAX_TABLE_ROWS = 40
_LD_SCRIPT = re.compile(r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>')


def _tables_to_text(fragment: str) -> str:
    """Each table row becomes one `| cell | cell |` line, so a number stays next to its label (the generic path glued
    all cells of a row into one run of words and lost the column boundaries)."""
    def render(match: re.Match) -> str:
        lines = []
        for row in _ROW.findall(match.group(1))[:MAX_TABLE_ROWS]:
            cells = [_clean_inline(c) for c in _CELL.findall(row)]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
        return "\n" + "\n".join(lines) + "\n" if lines else " "

    return _TABLE.sub(render, fragment)


def jsonld_items(page_html: str) -> list[dict]:
    """The JSON-LD objects of a page (flattening @graph and lists); broken blocks are skipped."""
    import json

    items: list[dict] = []

    def add(node: object) -> None:
        if isinstance(node, list):
            for n in node:
                add(n)
        elif isinstance(node, dict):
            if "@graph" in node:
                add(node["@graph"])
            items.append(node)

    for raw in _LD_SCRIPT.findall(page_html):
        try:
            add(json.loads(raw.strip()))
        except ValueError:
            continue
    return items


def _ld_types(item: dict) -> set[str]:
    t = item.get("@type")
    return {str(x) for x in t} if isinstance(t, list) else {str(t)} if t else set()


def ld_text(items: list[dict]) -> tuple[str, str]:
    """(facts, article body) from structured data: FAQ question/answer pairs and product prices as lines, and the
    `articleBody` some script-rendered pages carry instead of visible text."""
    facts, body = [], ""
    for it in items:
        types = _ld_types(it)
        if "FAQPage" in types:
            for q in it.get("mainEntity") or []:
                if isinstance(q, dict) and q.get("name") and isinstance(q.get("acceptedAnswer"), dict):
                    facts.append(f"Q: {_clean_inline(str(q['name']))} A: {_clean_inline(str(q['acceptedAnswer'].get('text', '')))}")
        if "Product" in types:
            offers = it.get("offers")
            for o in offers if isinstance(offers, list) else [offers] if isinstance(offers, dict) else []:
                if isinstance(o, dict) and o.get("price") is not None:
                    facts.append(f"Product {it.get('name', '')}: price {o['price']} {o.get('priceCurrency', '')}".strip())
        if types & {"Article", "NewsArticle", "BlogPosting", "TechArticle"} and isinstance(it.get("articleBody"), str):
            body = max(body, _clean_inline(it["articleBody"]), key=len)
    return "\n".join(facts), body


def _to_lines(fragment: str) -> list[str]:
    fragment = _tables_to_text(fragment)
    fragment = _HEADING.sub(lambda m: f"\n## {_clean_inline(m.group(2))}\n", fragment)
    fragment = _LI_OPEN.sub("\n- ", fragment)
    fragment = _BLOCK_END.sub("\n", fragment)
    text = html.unescape(_TAG.sub(" ", fragment))
    lines, seen = [], set()
    for raw in text.split("\n"):
        line = re.sub(r"[ \t\r\f\v]+", " ", raw).strip()
        if not line:
            continue
        heading = line.startswith("## ")
        if not heading:
            if _BOILERPLATE.search(line) and len(line) < 220:
                continue
            if len(line) < 25 and not line.startswith("| ") and not re.search(r"[.!?:;)]$|\d", line):     # menu items and labels
                continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def extract_page(page_html: str) -> Page:
    """Clean text of the main content, plus title and publication date."""
    title, published = find_title(page_html), find_date(page_html)
    body = _COMMENTS.sub(" ", page_html)
    head_cut = re.search(r"(?is)<body\b[^>]*>", body)
    body = body[head_cut.end():] if head_cut else body
    body = _DROP_BLOCKS.sub(" ", body)
    body = _HIDDEN.sub(" ", _HIDDEN.sub(" ", body))
    lines = _to_lines(_main_html(body))
    text = "\n".join(lines)
    facts, article = ld_text(jsonld_items(page_html))
    if len(text) < 400 and article:                          # a script-rendered page: the structured data holds the text
        text = (text + "\n" + article).strip()
    if facts:
        text = (text + "\n" + facts).strip()
    return Page(title, published, _ZERO_WIDTH.sub("", text)[:MAX_TEXT_CHARS])


# --------------------------------------------------------------------------- passage selection


def _terms(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[\w][\w.+#-]*", text.lower()):
        w = w.strip(".-")
        if len(w) < 2 or w in _STOP:
            continue
        out.append(_stem(w))
    return out


def _stem(w: str) -> str:
    for suffix in ("ing", "ed", "es", "en", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _chunks(text: str) -> list[str]:
    """Paragraph-sized chunks: short lines merged up to ~CHUNK_TARGET chars, a heading stays with
    the text that follows it."""
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        if line.startswith("## "):
            if cur:
                chunks.append(cur)
            cur = line[3:] + ": "
            continue
        if cur and len(cur) + len(line) > CHUNK_TARGET * 1.6:
            chunks.append(cur.strip())
            cur = ""
        cur = f"{cur}{line} " if len(line) < CHUNK_TARGET else f"{cur}{line}"
        if len(cur) >= CHUNK_TARGET:
            chunks.append(cur.strip())
            cur = ""
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 30]


def _scored(text: str, query: str) -> tuple[list[str], list[float]]:
    """Paragraph-sized chunks of `text` and how well each matches `query`.

    Score = sum over distinct query terms of (1 + log tf) * idf, idf computed over this page's own chunks, with a
    bonus for a chunk that has several different query terms and a small lead bonus (the first paragraphs usually
    define the topic)."""
    chunks = _chunks(text)
    q_terms = set(_terms(query))
    if not chunks or not q_terms:
        return chunks, []
    tokenized = [_terms(c) for c in chunks]
    n = len(chunks)
    df = {t: sum(1 for toks in tokenized if t in toks) for t in q_terms}
    scores = []
    for i, toks in enumerate(tokenized):
        score, hit = 0.0, 0
        for t in q_terms:
            tf = toks.count(t)
            if tf:
                hit += 1
                score += (1 + math.log(tf)) * math.log(1 + n / (1 + df[t]))
        if hit:
            score *= 1 + 0.25 * (hit - 1)
        score += max(0.0, 0.6 - 0.1 * i)                    # lead bonus fades after a few chunks
        scores.append(score)
    return chunks, scores


NO_MATCH = 0.6          # at or below this, a page has nothing on the question (the lead bonus alone is at most 0.6)


def relevance(text: str, query: str) -> float:
    """How well the best paragraph of `text` matches `query`; <= NO_MATCH means the page does not address it."""
    _, scores = _scored(text.strip(), query)
    return max(scores, default=0.0)


def select_passages(text: str, query: str, budget: int = 1800) -> str:
    """The paragraphs of `text` that best match `query`, in page order, within `budget` characters.
    No overlap at all -> the start of the page (use `relevance` to tell the caller that happened)."""
    text = text.strip()
    if len(text) <= budget:
        return text
    chunks, scores = _scored(text, query)
    if not chunks or not scores:
        return text[:budget].rsplit(" ", 1)[0] + " …"
    n = len(chunks)
    if max(scores) <= NO_MATCH:
        return text[:budget].rsplit(" ", 1)[0] + " …"
    chosen: list[int] = []
    used = 0
    for i in sorted(range(n), key=lambda k: -scores[k]):
        if scores[i] <= 0.6 and chosen:
            break
        size = len(chunks[i]) + 3
        if used + size > budget and chosen:
            continue
        chosen.append(i)
        used += size
        if used >= budget:
            break
    chosen.sort()
    parts: list[str] = []
    prev = -2
    for i in chosen:
        if parts and i != prev + 1:
            parts.append("…")
        parts.append(chunks[i] if len(chunks[i]) <= budget else chunks[i][:budget] + " …")
        prev = i
    return "\n".join(parts)
