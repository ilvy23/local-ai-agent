"""`/verify`: check the key claims of an answer against fresh, independent searches.

For each claim (a sentence with a figure or a date, at most three) it runs a primary-source search and a fact-check
search, then decides in code:

  supported     every figure of the claim appears in a source of at least medium trust (and the source is not the
                claim's only backer by a low-trust site)
  contradicted  a source of at least medium trust gives a clearly different value for the same measured thing
  unclear       nothing solid either way

No model call: the answer was written by the model, judging it with the same model would repeat its blind spots.
`run_search(query)` is injected (the chat's web_search tool plus `evidence.last_search()`), so this stays testable.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from agent.research import verdict as V

MAX_CLAIMS = 3
MAX_SEARCHES_PER_CLAIM = 2
_FIGURE = re.compile(r"\d")
_CITE = re.compile(r"\[\d{1,2}(?:\s*[,;]\s*\d{1,2})*\]")
_WORD = re.compile(r"[\w][\w.+#-]*", re.UNICODE)
_STOP = frozenset("the a an and or of to in on for with is are was were be been it its this that these those as at by from what "
                  "which who how why when where about into than then so if not also there their they them have has had will would can could "
                  "der die das und oder von zu im auf für mit ist sind war waren sein es ein eine einen einem einer als bei aus nach wie was wer "
                  "wo wann warum kann wird nicht auch noch nur mehr dass den dem des sich".split())


def claims_from(answer: str, max_claims: int = MAX_CLAIMS) -> list[str]:
    """The sentences of `answer` worth checking: those carrying figures or dates first, else the first long ones."""
    sentences = [re.sub(r"\s+([.,;:!?])", r"\1", _CITE.sub("", x)).strip() for x in V._SENT.split(answer) if x and x.strip()]
    sentences = [x for x in sentences if len(x.split()) >= 5]
    figured = [x for x in sentences if _FIGURE.search(x)]
    picked = figured[:max_claims] or sentences[:2]
    return [x[:220] for x in picked]


def counter_queries(claim: str) -> list[str]:
    """Two searches per claim: the primary source, and an explicit fact check."""
    words = [w for w in _WORD.findall(claim) if w.lower() not in _STOP and len(w) > 2]
    keyed = [w for w in words if w[0].isupper() or _FIGURE.search(w)] + [w for w in words if not (w[0].isupper() or _FIGURE.search(w))]
    base = " ".join(dict.fromkeys(keyed))[:90].strip()
    return [f"{base} official source", f"{base} fact check"][:MAX_SEARCHES_PER_CLAIM]


def _strong(item: dict) -> bool:
    level = (item.get("trust") or {}).get("level", "low")
    return item.get("status") in ("ok", "cached") and level in ("medium", "high")


def _clashes(claim_facts: list[tuple[str, str, float]], passage: str, tolerance: float = 0.05) -> list[dict]:
    """Measured figures of the claim that the passage states differently: same unit, a shared content word in front of the
    number ("startup" vs "startup time"), values more than `tolerance` apart."""
    def stems(pre: str) -> set[str]:
        return {w[:5] for w in pre.split() if len(w) >= 5}          # "starts" and "startup" share "start"

    found = []
    for pre, unit, value in claim_facts:
        words = stems(pre)
        for pre2, unit2, value2 in V._facts(passage):
            if unit2 == unit and words & stems(pre2) and abs(value - value2) / (abs(value2) or 1.0) > tolerance:
                found.append({"attribute": f"{pre or pre2} ({unit})", "claim": value, "source": value2})
    return found


def judge(claim: str, items: list[dict]) -> dict[str, Any]:
    """Decide one claim from the sources found for it."""
    strong = [i for i in items if _strong(i) and i.get("passage")]
    text = V._CITE.sub("", claim)
    measured = V._facts(text)                                  # numbers with a unit: the claim's real data points
    figures = [v for _, _, v in measured] or [v for v in V._numbers(text) if not (1900 <= v <= 2100 and float(v).is_integer())]
    support = [i for i in strong if figures and all(V._has_number(f, i["passage"]) for f in figures)]
    against = [{"url": i["url"], **c} for i in strong for c in _clashes(measured, i["passage"])]
    if against and not support:
        return {"status": "contradicted", "support": [], "against": against}
    if support:
        return {"status": "supported", "support": [i["url"] for i in support], "against": []}
    if not figures:                                           # a claim without figures: topic words found in a solid source
        terms = {w.lower() for w in _WORD.findall(claim) if w.lower() not in _STOP and len(w) > 3}
        for i in strong:
            hit = sum(1 for t in terms if t in i["passage"].lower())
            if terms and hit / len(terms) >= 0.6:
                return {"status": "supported", "support": [i["url"]], "against": []}
    return {"status": "unclear", "support": [], "against": []}


def verify(answer: str, run_search: Callable[[str], tuple[list[dict], dict]]) -> dict[str, Any]:
    """Check `answer`. `run_search(query)` returns (source items, evidence verdict) for one search."""
    out_claims, searches = [], 0
    for claim in claims_from(answer):
        items: list[dict] = []
        queries = counter_queries(claim)
        for q in queries:
            got, _vd = run_search(q)
            searches += 1
            items += got
        out_claims.append({"text": claim, "queries": queries, **judge(claim, items)})
    states = {c["status"] for c in out_claims}
    overall = ("contradicted" if "contradicted" in states else "supported" if states == {"supported"}
               else "unclear" if out_claims else "nothing_to_check")
    return {"claims": out_claims, "overall": overall, "searches": searches}


_STATUS_TEXT = {"supported": "supported", "contradicted": "CONTRADICTED", "unclear": "unclear"}


def summarize(result: dict[str, Any]) -> str:
    """Plain-text report of a `verify` result for a terminal or a chat bubble."""
    claims = result["claims"]
    if not claims:
        return "Nothing to check: the last answer has no concrete claims (figures, dates) to verify."
    counts = {k: sum(1 for c in claims if c["status"] == k) for k in _STATUS_TEXT}
    lines = [f"Checked {len(claims)} claim{'s' if len(claims) != 1 else ''} with {result['searches']} searches: "
             + ", ".join(f"{n} {_STATUS_TEXT[k]}" for k, n in counts.items() if n) + "."]
    for c in claims:
        lines.append(f"- {_STATUS_TEXT[c['status']]}: {c['text']}")
        if c["status"] == "supported":
            lines.append(f"    confirmed by {c['support'][0]}")
        elif c["status"] == "contradicted":
            a = c["against"][0]
            lines.append(f"    a source says {a['source']:g} where the answer says {a['claim']:g} ({a['attribute']}): {a['url']}")
        else:
            lines.append("    no solid source found either way - treat it as unconfirmed")
    return "\n".join(lines)
