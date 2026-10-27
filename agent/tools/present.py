"""`present`: show an answer visually - a table, chart, diagram, comparison, timeline, ... - instead of describing it.

One tool with a `kind` instead of one tool per kind keeps the tool list (and so every prompt) short. The spec is
validated here (a bad spec comes back to the model as an error it can fix), turned into a `block` event for the GUI
(see docs/design/02-event-component-contract.md, section 4) and drawn as plain text in the terminal.

Numbers are checked for provenance: a block that claims to show source data must contain figures that appear in the
web text read this turn; otherwise it is marked unverified (provenance "model") and the model is told so.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from agent.research import taint
from agent.tools.registry import Tool

KINDS = ("table", "chart", "diagram", "compare", "timeline", "stats", "callout", "quote", "steps", "code", "diff",
         "math", "map", "gallery", "html", "svg")
CHART_TYPES = ("bar", "stacked_bar", "line", "area", "pie", "donut", "scatter", "histogram", "heatmap", "radar")
DIAGRAM_STARTS = ("flowchart", "graph", "sequencediagram", "statediagram", "erdiagram", "classdiagram", "gantt", "mindmap",
                  "timeline", "journey", "quadrantchart", "pie", "gitgraph", "c4context", "sankey", "xychart")
MAX_ROWS, MAX_COLS, MAX_SERIES, MAX_POINTS = 500, 12, 10, 500
MAX_DIAGRAM, MAX_HTML, MAX_TEXT = 20_000, 100_000, 8_000

_UNSAFE_SVG = re.compile(r"(?is)<\s*(script|foreignobject|iframe|object|embed)|\son\w+\s*=|javascript:|(?:href|src)\s*=\s*[\"']\s*(?!#)")


class SpecError(ValueError):
    """The spec cannot be shown; the message tells the model what to fix."""


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise SpecError(message)


def _list(spec: dict, key: str, low: int, high: int, what: str) -> list:
    value = spec.get(key)
    _need(isinstance(value, list) and low <= len(value) <= high, f"{what}: `{key}` must be a list of {low}-{high} items")
    return value


def _text(spec: dict, key: str, limit: int = MAX_TEXT) -> str:
    value = spec.get(key)
    _need(isinstance(value, str) and value.strip(), f"`{key}` must be non-empty text")
    _need(len(value) <= limit, f"`{key}` is too long (max {limit} characters)")
    return value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _v_table(spec: dict) -> dict:
    cols = _list(spec, "columns", 1, MAX_COLS, "table")
    for c in cols:
        _need(isinstance(c, dict) and isinstance(c.get("key"), str) and c.get("key"), "each column needs a `key`")
        c.setdefault("label", c["key"])
    rows = _list(spec, "rows", 0, MAX_ROWS, "table")
    keys = {c["key"] for c in cols}
    for r in rows:
        _need(isinstance(r, dict) and r.keys() <= keys, "each row must be an object using only the column keys")
    return spec


def _v_chart(spec: dict) -> dict:
    _need(spec.get("type") in CHART_TYPES, f"chart `type` must be one of {', '.join(CHART_TYPES)}")
    series = _list(spec, "series", 1, MAX_SERIES, "chart")
    for s in series:
        _need(isinstance(s, dict) and isinstance(s.get("name"), str), "each series needs a `name`")
        pts = s.get("points")
        _need(isinstance(pts, list) and 1 <= len(pts) <= MAX_POINTS, f"series {s.get('name')!r}: `points` needs 1-{MAX_POINTS} items")
        for p in pts:
            _need(isinstance(p, (list, tuple)) and len(p) in (2, 3) and _number(p[-1]),
                  "each point must be [x, y] with a numeric y (x may be text or number)")
    return spec


def _v_diagram(spec: dict) -> dict:
    src = _text(spec, "source", MAX_DIAGRAM)
    first = re.sub(r"[^a-z0-9]", "", (re.split(r"[\s\n]", src.strip().lstrip("%").strip(), maxsplit=1)[0] or "").lower())
    _need(any(first.startswith(k) for k in DIAGRAM_STARTS), "diagram `source` must be Mermaid syntax starting with e.g. `flowchart LR`")
    spec["syntax"] = "mermaid"
    return spec


def _v_compare(spec: dict) -> dict:
    options = _list(spec, "options", 2, 6, "compare")
    criteria = _list(spec, "criteria", 1, 10, "compare")
    scores = spec.get("scores")
    _need(isinstance(scores, list) and len(scores) == len(criteria)
          and all(isinstance(r, list) and len(r) == len(options) and all(_number(x) for x in r) for r in scores),
          "`scores` must be one row per criterion with one number per option")
    for c in criteria:
        _need(isinstance(c, dict) and isinstance(c.get("name"), str), "each criterion needs a `name` (and optional numeric `weight`)")
        c.setdefault("weight", 1)
    totals = [sum(c["weight"] * scores[i][j] for i, c in enumerate(criteria)) for j in range(len(options))]
    spec["totals"] = totals
    spec.setdefault("winner", totals.index(max(totals)))
    return spec


def _v_timeline(spec: dict) -> dict:
    for e in _list(spec, "events", 1, 60, "timeline"):
        _need(isinstance(e, dict) and e.get("date") and e.get("title"), "each event needs `date` and `title`")
    return spec


def _v_stats(spec: dict) -> dict:
    for t in _list(spec, "tiles", 1, 12, "stats"):
        _need(isinstance(t, dict) and t.get("label") and t.get("value") is not None, "each tile needs `label` and `value`")
    return spec


def _v_callout(spec: dict) -> dict:
    _need(spec.get("level") in ("info", "warning", "danger", "tip"), "callout `level` must be info, warning, danger or tip")
    _text(spec, "text")
    return spec


def _v_quote(spec: dict) -> dict:
    _text(spec, "text", 2_000)
    return spec


def _v_steps(spec: dict) -> dict:
    for s in _list(spec, "steps", 1, 40, "steps"):
        _need(isinstance(s, dict) and s.get("title"), "each step needs a `title`")
    return spec


def _v_code(spec: dict) -> dict:
    _text(spec, "code", 30_000)
    return spec


def _v_diff(spec: dict) -> dict:
    for f in _list(spec, "files", 1, 30, "diff"):
        _need(isinstance(f, dict) and f.get("path") and isinstance(f.get("hunks"), list), "each file needs `path` and `hunks`")
    return spec


def _v_math(spec: dict) -> dict:
    _text(spec, "latex", 2_000)
    return spec


def _v_map(spec: dict) -> dict:
    for p in _list(spec, "points", 1, 100, "map"):
        _need(isinstance(p, dict) and _number(p.get("lat")) and _number(p.get("lon")) and -90 <= p["lat"] <= 90
              and -180 <= p["lon"] <= 180, "each point needs numeric `lat` (-90..90) and `lon` (-180..180)")
    return spec


def _v_gallery(spec: dict) -> dict:
    for i in _list(spec, "images", 1, 24, "gallery"):
        url = i.get("url", "") if isinstance(i, dict) else ""
        _need(isinstance(url, str) and (url.startswith("/api/files/") or url.startswith("https://")),
              "image `url` must be a file the agent shared (/api/files/...) or an https address")
    return spec


def _v_html(spec: dict) -> dict:
    _text(spec, "html", MAX_HTML)
    spec["height"] = max(120, min(int(spec.get("height") or 360), 900))
    return spec


def _v_svg(spec: dict) -> dict:
    svg = _text(spec, "svg", 60_000)
    _need(svg.lstrip().lower().startswith("<svg"), "`svg` must start with <svg")
    _need(not _UNSAFE_SVG.search(svg), "svg may not contain scripts, event handlers, foreignObject or external links")
    return spec


_VALIDATORS = {
    "table": _v_table, "chart": _v_chart, "diagram": _v_diagram, "compare": _v_compare, "timeline": _v_timeline,
    "stats": _v_stats, "callout": _v_callout, "quote": _v_quote, "steps": _v_steps, "code": _v_code, "diff": _v_diff,
    "math": _v_math, "map": _v_map, "gallery": _v_gallery, "html": _v_html, "svg": _v_svg,
}


def validate(kind: str, spec: Any, allow_html: bool = False) -> dict:
    """The cleaned spec, or SpecError."""
    _need(kind in KINDS, f"unknown kind {kind!r}; use one of: {', '.join(KINDS)}")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            raise SpecError("`spec` must be a JSON object") from None
    _need(isinstance(spec, dict), "`spec` must be a JSON object")
    _need(kind != "html" or allow_html, "interactive html blocks are switched off (ui.allow_html_blocks)")
    return _VALIDATORS[kind](spec)


# --------------------------------------------------------------------------- provenance


def _figures(kind: str, spec: dict) -> list[float]:
    """The data values a block shows (not axis ticks or ranks)."""
    if kind == "chart":
        return [float(p[-1]) for s in spec["series"] for p in s["points"]]
    if kind == "table":
        return [float(v) for r in spec["rows"] for v in r.values() if _number(v)]
    if kind == "stats":
        out = []
        for t in spec["tiles"]:
            try:
                out.append(float(str(t["value"]).replace(",", "")))
            except ValueError:
                pass
        return out
    return []


def provenance(kind: str, spec: dict, claimed: str, source_ids: list[int]) -> tuple[str, int]:
    """(provenance to record, how many figures are missing from this turn's web text). A block that claims to show source
    data but whose figures are not in the text read this turn is downgraded to "model"."""
    figures = [v for v in _figures(kind, spec) if not (1900 <= v <= 2100 and float(v).is_integer())]
    if claimed == "computed":
        # "computed" is only believed when the figures really came out of the calc tool this turn
        missing = [v for v in figures if not taint.has_computed_number(v)]
        return ("model", len(missing)) if figures and missing else ("computed", 0)
    if claimed == "sources" or source_ids:
        missing = [v for v in figures if not taint.has_number(v)]
        if figures and missing:
            return "model", len(missing)
        return "sources", 0
    return "model", 0


# --------------------------------------------------------------------------- terminal fallback


def render_text(kind: str, title: str, spec: dict) -> str:
    """Plain-text drawing for a terminal (or the model-visible log)."""
    head = f"{title}\n" if title else ""
    if kind == "table":
        keys = [c["key"] for c in spec["columns"]]
        rows = [[str(r.get(k, "")) for k in keys] for r in spec["rows"][:30]]
        widths = [max(len(str(c["label"])), *(len(r[i]) for r in rows), 1) for i, c in enumerate(spec["columns"])]
        line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        return head + "\n".join([line([str(c["label"]) for c in spec["columns"]]), line(["-" * w for w in widths]), *map(line, rows)])
    if kind == "chart":
        out = [head.rstrip()]
        for s in spec["series"][:4]:
            vals = [(str(p[0]) if len(p) > 1 else "", float(p[-1])) for p in s["points"][:20]]
            top = max((abs(v) for _, v in vals), default=1) or 1
            out.append(s["name"])
            out += [f"  {k[:14]:<14} {'█' * max(1, round(abs(v) / top * 30))} {v:g}" for k, v in vals]
        return "\n".join(out)
    if kind == "diagram":
        return head + spec["source"]
    if kind == "compare":
        names = [c["name"] for c in spec["criteria"]]
        out = [f"{head}options: {', '.join(map(str, spec['options']))}"]
        out += [f"  {n}: {spec['scores'][i]}" for i, n in enumerate(names)]
        return "\n".join(out + [f"  winner: {spec['options'][spec['winner']]}"])
    if kind == "timeline":
        return head + "\n".join(f"  {e['date']}  {e['title']}" for e in spec["events"])
    if kind == "stats":
        return head + "\n".join(f"  {t['label']}: {t['value']}{t.get('unit', '')}" for t in spec["tiles"])
    if kind == "steps":
        return head + "\n".join(f"  {i}. {s['title']}" + (f"  ({s['command']})" if s.get("command") else "") for i, s in enumerate(spec["steps"], 1))
    if kind in ("callout", "quote"):
        return head + spec["text"]
    if kind == "code":
        return head + spec["code"]
    if kind == "math":
        return head + spec["latex"]
    if kind == "map":
        return head + "\n".join(f"  {p.get('label', '')} {p['lat']:.4f}, {p['lon']:.4f}" for p in spec["points"])
    return head + f"[{kind} block - shown in the GUI]"


# --------------------------------------------------------------------------- the tool


_DESCRIPTION = (
    "Show information visually instead of describing it. Pick `kind`: table (compare things / list facts), chart (numbers: "
    "bar, line, pie, ...), diagram (Mermaid: flowchart, sequence, gantt, mindmap, state, er), compare (options x weighted "
    "criteria), timeline (dated events), stats (key numbers), callout (warning/tip), quote (verbatim source text), steps "
    "(guide/checklist), code, diff, math (LaTeX), map, gallery, svg, html. Use at most one or two per answer, always "
    "with a short written answer too. Use only numbers you were given, found in sources, or calculated with the calc tool "
    "(then set provenance_claim=computed); list the source numbers in `source_ids`. Specs: table {columns:[{key,label,unit?}], rows:[{key:value}]}; chart {type, series:[{name, "
    "points:[[x,y]]}], x?:{label}, y?:{label,unit}}; diagram {source}; compare {options, criteria:[{name,weight}], "
    "scores:[[one row per criterion, one number per option]]}; timeline {events:[{date,title,text?}]}; stats {tiles:[{label,"
    "value,unit?}]}; callout {level:info|warning|danger|tip, text}; quote {text}; steps {steps:[{title,text?,command?}]}; "
    "code {language, code}; math {latex}."
)


def make_present_tool(console: Any, allow_html: bool = False) -> Tool:
    def handler(kind: str = "", title: str = "", spec: Any = None, source_ids: Any = None, provenance_claim: str = "",
                **_: Any) -> str:
        try:
            clean = validate(str(kind), spec, allow_html=allow_html)
        except SpecError as exc:
            return f"Error: the {kind or 'block'} could not be shown - {exc}. Fix the spec and call present again."
        ids = [int(i) for i in (source_ids or []) if str(i).isdigit()] if isinstance(source_ids, list) else []
        prov, missing = provenance(kind, clean, provenance_claim, ids)
        block = {"id": "b_" + uuid.uuid4().hex[:6], "kind": kind, "title": str(title or "")[:120], "spec": clean,
                 "source_ids": ids, "provenance": prov}
        emit = getattr(console, "emit_event", None)
        if callable(emit):
            emit("block", **block)
        elif kind == "diagram":                                  # a terminal cannot draw it: save the browser page as before
            from agent.tools.diagram import make_diagram

            console.print(make_diagram(title=block["title"], diagram=clean["source"]), markup=False, highlight=False)
        else:
            console.print(render_text(kind, block["title"], clean), markup=False, highlight=False)
        note = ""
        if prov == "model" and provenance_claim == "computed":
            note = (f" {missing} figure(s) in it were not produced by the calc tool this turn, so it is marked unverified - "
                    "run the calculation with calc first, or say the numbers are estimates.")
        elif prov == "model" and (provenance_claim == "sources" or ids):
            note = (f" {missing} figure(s) in it are not in any source read this turn, so it is marked unverified - say so "
                    "in your answer, or search for the real numbers.")
        elif prov == "model":
            note = " It is marked as coming from your own knowledge (unverified)."
        return (f"The {kind} '{block['title']}' is now shown to the user. Do not repeat its contents in text; add one or "
                f"two sentences of interpretation.{note}")

    return Tool(
        name="present",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS)},
                "title": {"type": "string", "description": "Short title shown above the block."},
                "spec": {"type": "object", "description": "The data for this kind (see the tool description)."},
                "source_ids": {"type": "array", "items": {"type": "integer"},
                               "description": "Numbers [n] of the sources the data comes from."},
                "provenance_claim": {"type": "string", "enum": ["sources", "computed", "model"],
                                     "description": "Where the numbers come from."},
            },
            "required": ["kind", "spec"],
        },
        handler=handler,
        risk="safe",
    )
