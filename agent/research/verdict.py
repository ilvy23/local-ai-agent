"""Weigh what the sources say, in code, so the model only has to phrase the result.

* `group_origins`: copied or syndicated text counts once (word-shingle overlap), and so do several pages of
  one site - "three sources" that are one press release are not three confirmations.
* `find_conflicts`: the same measurable thing (a number with its unit and the words in front of it) given
  with clearly different values by different sources.
* `verdict`: verified / likely / single_source / conflicting / low_trust / from_memory.
* `ground`: after the model has answered, check that the figures in a sentence exist in the source it cites
  and that no figure appears out of nowhere.

Sources are plain dicts: {n, url, title, date, status, level, tier, text}. Pure functions, no network.
"""

from __future__ import annotations

import re
import urllib.parse
from collections import defaultdict

from agent.research.trust import registrable

_WORD = re.compile(r"[\w']+")
_NUM = re.compile(
    r"(?P<pre>(?:[\w'-]+\s+){0,3})(?P<num>\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|°c|°f|km/h|€|\$|£|eur\b|usd\b|gb\b|mb\b|tb\b|kb\b|ghz\b|mhz\b|tok/s\b|fps\b|w\b|kw\b|mah\b|mm\b|cm\b|km\b|kg\b|"
    r"ms\b|s\b|min\b|h\b|x\b|prozent\b|percent\b|million\w*|billion\w*|mrd\b|mio\b)?",
    re.I,
)
_STOP = frozenset("the a an of in on at to for is are was were be by with and or as from that this it its".split())


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + k]) for i in range(max(len(w) - k + 1, 0))}


def group_origins(sources: list[dict]) -> list[str]:
    """One group id per source: same site, or near-identical text, share a group."""
    groups: list[str] = []
    sh = [_shingles(s.get("text", "")) for s in sources]
    for i, s in enumerate(sources):
        reg = registrable(urllib.parse.urlparse(s.get("url", "")).netloc)
        gid = f"g{i + 1}"
        for j in range(i):
            same_site = reg and reg == registrable(urllib.parse.urlparse(sources[j].get("url", "")).netloc)
            a, b = sh[i], sh[j]
            overlap = len(a & b) / min(len(a), len(b)) if a and b else 0.0
            if same_site or overlap >= 0.5:
                gid = groups[j]
                break
        groups.append(gid)
    return groups


def parse_number(raw: str) -> float | None:
    """"1,234.5", "1.234,5", "1 234", "3,5" -> float. A lone "1.234" is read as a decimal (ambiguous in German text;
    the same reading is applied to answer and source, so comparisons stay consistent)."""
    s = raw.strip().replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _facts(text: str) -> list[tuple[str, str, float]]:
    """(attribute words, unit, value) for each number followed by a unit, with up to three words in front of it."""
    out = []
    for m in _NUM.finditer(text.replace("|", " ")):          # table rows read `| label | 40 % |`: the label is the attribute
        unit = (m.group("unit") or "").lower()
        value = parse_number(m.group("num"))
        if value is None or not unit:
            continue
        pre = " ".join(w for w in _WORD.findall(m.group("pre").lower()) if w not in _STOP)
        out.append((pre, unit, value))
    return out


def find_conflicts(sources: list[dict], tolerance: float = 0.05) -> list[dict]:
    """Same attribute and unit, values more than `tolerance` apart, from different sources."""
    seen: dict[tuple[str, str], list[tuple[float, int, str]]] = defaultdict(list)
    for s in sources:
        if s.get("status", "ok") not in ("ok", "cached", "read"):
            continue
        for pre, unit, value in _facts(s.get("text", "")):
            if pre:
                seen[(pre, unit)].append((value, s["n"], s.get("level", "low")))
    conflicts = []
    for (pre, unit), vals in seen.items():
        by_source = {n: (v, lvl) for v, n, lvl in vals}
        if len(by_source) < 2:
            continue
        values = sorted({v for v, _ in by_source.values()})
        if (values[-1] - values[0]) / (abs(values[-1]) or 1.0) <= tolerance:
            continue
        groups = defaultdict(list)
        for n, (v, lvl) in by_source.items():
            groups[v].append((n, lvl))
        conflicts.append({
            "attribute": f"{pre} ({unit})",
            "values": [{"value": v, "sources": sorted(n for n, _ in ns),
                        "trust": max((lvl for _, lvl in ns), key=_rank)} for v, ns in sorted(groups.items())],
        })
    return conflicts


def _rank(level: str) -> int:
    return {"avoid": 0, "low": 1, "medium": 2, "high": 3}.get(level, 1)


def _resolve(conflict: dict, by_n: dict) -> dict:
    """{"preferred", "why"} for one conflict: the value backed by the highest-trust source, ties broken by the newest date."""
    best, best_key = None, None
    for v in conflict["values"]:
        for n in v["sources"]:
            src = by_n.get(n, {})
            key = (_rank(src.get("level", "low")), src.get("date") or "")
            if best_key is None or key > best_key:
                best, best_key = v["value"], key
    others = [by_n.get(n, {}).get("date") or "" for v in conflict["values"] for n in v["sources"]]
    why = ("newest primary source" if best_key and best_key[0] >= 3 and best_key[1] >= max(others) and best_key[1]
           else "highest-trust source" if best_key and best_key[0] >= 2 else "no source is strong enough to decide")
    return {"preferred": best, "why": why}


def verdict(sources: list[dict], conflicts: list[dict] | None = None) -> dict:
    """Overall label for the evidence plus the counts the UI shows."""
    usable = [s for s in sources if s.get("status", "ok") in ("ok", "cached", "read") and s.get("level") != "avoid"]
    if not usable:
        return {"label": "from_memory" if not sources else "low_trust", "summary": "no usable sources" if sources else "no sources",
                "independent_sources": 0, "conflicts": [], "newest": "", "oldest": ""}
    groups = group_origins(usable)
    independent = len(set(groups))
    strong_groups = {g for g, s in zip(groups, usable) if _rank(s.get("level", "low")) >= 2}
    high_groups = {g for g, s in zip(groups, usable) if _rank(s.get("level", "low")) >= 3}
    conflicts = find_conflicts(usable) if conflicts is None else conflicts
    dates = sorted(s["date"] for s in usable if s.get("date"))
    # a conflict flips the label only when two values are each backed by a source of at least medium trust; a lone
    # weak page disagreeing with solid ones stays listed in `conflicts` (the UI shows it) but does not cancel them
    by_n = {u["n"]: u for u in usable}
    for c in conflicts:                                       # which value to believe: the most trusted source, newest first
        c.update(_resolve(c, by_n))
    material = [c for c in conflicts if sum(1 for v in c["values"] if _rank(v["trust"]) >= 2) >= 2]
    if material:
        label = "conflicting"
    elif not strong_groups:
        label = "low_trust"
    elif len(strong_groups) >= 2 and high_groups:
        label = "verified"
    elif len(strong_groups) >= 2 or high_groups:
        label = "likely"
    else:
        label = "single_source"
    parts = [f"{len(usable)} source{'s' if len(usable) != 1 else ''}", f"{independent} independent"]
    if dates:
        parts.append(f"newest {dates[-1]}")
    return {"label": label, "summary": ", ".join(parts), "independent_sources": independent, "conflicts": conflicts,
            "newest": dates[-1] if dates else "", "oldest": dates[0] if dates else ""}


# --------------------------------------------------------------------------- checking the finished answer

_CITE = re.compile(r"\[(\d{1,2})(?:\s*[,;]\s*(\d{1,2}))*\]")
_CITE_N = re.compile(r"\d{1,2}")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9\[])|\n+")


def _numbers(text: str) -> list[float]:
    vals = []
    for m in re.finditer(r"\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?", text):
        v = parse_number(m.group(0))
        if v is not None:
            vals.append(v)
    return vals


def _has_number(value: float, haystack: str) -> bool:
    return any(abs(v - value) <= 1e-9 * max(1.0, abs(value)) for v in _numbers(haystack))


def ground(answer: str, sources: list[dict]) -> dict:
    """Check the answer against its sources.

    * a sentence citing [n] whose figures are missing from source n -> `unsupported`
    * a figure anywhere in the answer that no source contains (and that looks like data, not a list number or a
      year) -> `warnings`
    Returns {"claims": [{"text", "support", "status"}], "warnings": [str]}."""
    by_n = {s["n"]: s.get("text", "") for s in sources}
    everything = "\n".join(by_n.values())
    claims, warnings, stray_figures = [], [], []
    for sent in (x.strip() for x in _SENT.split(answer) if x and x.strip()):
        cited = sorted({int(n) for m in _CITE.finditer(sent) for n in _CITE_N.findall(m.group(0))})
        plain = _CITE.sub("", sent)
        figures = [v for v in _numbers(plain) if not (1900 <= v <= 2100 and float(v).is_integer())]
        if cited:
            missing = [v for v in figures if not any(_has_number(v, by_n.get(n, "")) for n in cited)]
            if figures and missing:
                claims.append({"text": sent[:200], "support": cited, "status": "unsupported"})
                warnings.append(f"figure {missing[0]:g} is not in source {cited[0]}")
            else:
                claims.append({"text": sent[:200], "support": cited, "status": "supported"})
        elif figures:
            stray = [v for v in figures if not _has_number(v, everything)]
            if stray and by_n:
                warnings.append(f"figure {stray[0]:g} appears in no source")
                stray_figures.extend(stray)
    return {"claims": claims, "warnings": list(dict.fromkeys(warnings)), "stray_figures": list(dict.fromkeys(stray_figures))}
