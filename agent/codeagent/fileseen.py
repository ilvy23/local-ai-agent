"""FILES SO FAR: what a stage has already read and edited, per file, taken from the tool calls (never from model text), so a restart in a
fresh context does not read the same ranges again. Measured: 229 whole-file reads against 32 edits; the cursor (cursor.py) says where the
edits are, this says what was looked at. Deterministic, no model call.

The record lives on the Stage (`Stage.seen`, saved with the session): {file: {"read": [[a, b], ...], "reads": n, "last": "read" | "edited"}}.

    note_read(seen, rel, a, b) / note_edit(seen, rel)     one real read (a, b = the lines shown; none = a placeholder answer) / one write
    rows(seen, edited) / block(seen, edited)              'story.py  read 1-240,300-380 (3x), edited 1-120' / the section the model sees
    load(raw) -> seen                                     a saved record (absent or damaged: empty)"""

from __future__ import annotations

from typing import Any

from agent.codeagent import cursor

MAX_FILES = 12
MAX_CHARS = 700
HEADER = "FILES SO FAR (recorded from this stage's tool calls):"
FACT = "Files above were already read in this stage; read only the ranges you still need."


def _row(seen: dict[str, dict[str, Any]], rel: str) -> dict[str, Any]:
    return seen.setdefault(rel, {"read": [], "reads": 0, "last": ""})


def note_read(seen: dict[str, dict[str, Any]], rel: str, a: int | None = None, b: int | None = None) -> None:
    row = _row(seen, rel)
    row["reads"] += 1
    row["last"] = "read"
    if a is not None and b is not None:
        row["read"] = cursor.merge([*row["read"], [a, b]])


def note_edit(seen: dict[str, dict[str, Any]], rel: str) -> None:
    _row(seen, rel)["last"] = "edited"


def load(raw: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for rel, d in raw.items():
        if not isinstance(d, dict):
            continue
        try:
            reads = max(0, int(d.get("reads") or 0))
        except (TypeError, ValueError):
            reads = 0
        last = d.get("last")
        out[str(rel)] = {"read": cursor.merge(d.get("read") if isinstance(d.get("read"), list) else []), "reads": reads,
                         "last": last if last in ("read", "edited") else ""}
    return out


def rows(seen: dict[str, dict[str, Any]], edited: dict[str, Any] | None = None) -> list[str]:
    """One line per file, edited files first (by name), then the others by number of reads (most first)."""
    edited = edited or {}
    items = []
    for rel in {*seen, *(f for f, r in edited.items() if cursor.merge(r))}:
        row, ed = seen.get(rel) or {}, cursor.fmt(edited.get(rel))
        reads = cursor.fmt(row.get("read"))
        parts = ([f"read {reads} ({row.get('reads', 0)}x)"] if reads else []) + ([f"edited {ed}"] if ed else [])
        if parts:
            items.append((not ed, 0 if ed else -int(row.get("reads") or 0), rel, f"{rel}  {', '.join(parts)}"))
    return [text for *_k, text in sorted(items)]


def block(seen: dict[str, dict[str, Any]], edited: dict[str, Any] | None = None) -> list[str]:
    """The section ([] when nothing was read yet): at most MAX_FILES files and MAX_CHARS characters, the factual line last."""
    body = rows(seen, edited)
    if not body or not any(r.get("reads") for r in seen.values()):
        return []                                           # nothing read yet (a first segment): the cursor already shows the edits
    tail = [FACT] if any(r.get("read") for r in seen.values()) else []
    keep = body[:MAX_FILES]

    def render(n: int) -> list[str]:
        more = [f"  ... and {len(body) - n} more files"] if len(body) > n else []
        return [HEADER, *(f"  {x}" for x in keep[:n]), *more, *tail]

    n = len(keep)
    while n > 1 and len("\n".join(render(n))) > MAX_CHARS:
        n -= 1
    return render(n)
