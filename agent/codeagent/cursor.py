"""The stage cursor: which line ranges of its files a stage has edited, so a fresh context (after a restart or a cap) continues where the
last one stopped instead of surveying the files again. Deterministic, no model call.

Measured on a real run: a 'rewrite the texts' stage over four 400-700 line files read for ten minutes, edited nothing, and every
restart started the survey again. The ranges are recorded from the real writes (`CodeTools.on_write`: old text -> new text), in the
numbering of the file as it is now (later ranges move when an edit changes the line count).

    apply_change(ranges, before, after) -> ranges        one write
    cursor(st, root) -> {file: {"edited": "1-120,150-200", "next": 201, "total": 406}}
    lines(cur) / note(cur)                               the text the model sees (STAGE CURSOR section, begin_stage notes)
    next_chunk(cur) -> "Rewrite lines 201-280 of story.py now (read them first if needed)." or ''
    is_chunk_stage(st) / targets(st)                     the editing-by-passes stages that get the working-method note

A RANGED stage (`Stage.lines` = (first, last): "Part 2/5: dialogue.py lines 151-300", see stages.py) covers one line range of one file. Its
cursor speaks in that range ('dialogue.py lines 151-300: next unedited line 161') and `overlaps` tells the scope gate whether a change landed
inside it. Earlier parts add or remove lines, so `sync_range` keeps the range true: a stage keeps `lines_total`, the file length its range was
valid for; when the file is longer or shorter now, the difference moves BOTH ends of a part that has written nothing yet (the earlier parts
changed the file before it) and only the END of a part that already wrote (its own edits)."""

from __future__ import annotations

import bisect
import difflib
import re
from pathlib import Path
from typing import Any

CHUNK_LINES = 80
CHUNK_LOOK_BLOCK = 10                  # coding.chunk_look_block: the pause threshold of a chunk stage (the general one is 16)
MAX_FILES = 8                          # files named in the cursor text
CURSOR_CHARS = 500
INTENT = re.compile(r"\b(?:rewrit|rewrote|rephras|reword|polish|clarif|simplif|improv|clean(?:ing)? up|tidy|translat|refactor|renam|"
                    r"document|überarbeit|umschreib|verständlich|verbesser|aufräum)", re.I)
METHOD = ("WORKING METHOD: go through the files in order, one section at a time. Read at most 80 lines (read_file path start-end), "
          "rewrite them right away with replace_lines/edit_file, then continue with the next 80 lines. Do not survey the whole file "
          "or dump text with sed/python first. After each file run the tests once.")


def merge(ranges: Any) -> list[list[int]]:
    """Sorted, overlapping and adjacent ranges joined; junk entries are dropped."""
    rows = []
    for r in ranges or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if 1 <= a <= b:
            rows.append((a, b))
    out: list[list[int]] = []
    for a, b in sorted(rows):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _from_lines(lines: set[int]) -> list[list[int]]:
    return merge([[n, n] for n in lines])


def apply_change(ranges: Any, before: list[str] | None, after: list[str]) -> list[list[int]]:
    """`ranges` (numbering of `before`) after the write that turned `before` into `after`: lines the write kept keep their mark (moved to
    their new number), changed or new lines are marked. before=None = a new file: every line is marked."""
    n = len(after)
    if before is None:
        return [[1, n]] if n else []
    pre = 0
    while pre < min(len(before), n) and before[pre] == after[pre]:
        pre += 1
    suf = 0
    while suf < min(len(before), n) - pre and before[len(before) - 1 - suf] == after[n - 1 - suf]:
        suf += 1
    a, b = before[pre:len(before) - suf], after[pre:n - suf]
    eq: list[tuple[int, int, int]] = [(0, pre, 0)] if pre else []          # (old start, old end, new start) of the lines that stayed, 0-based
    new: set[int] = set()
    if a and b and len(a) * len(b) <= 4_000_000:
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == "equal":
                eq.append((pre + i1, pre + i2, pre + j1))
            else:
                new.update(range(pre + j1 + 1, pre + j2 + 1))
                if j1 == j2:                                      # lines removed: mark the line that now follows them
                    new.add(min(pre + j1 + 1, n))
    else:
        new.update(range(pre + 1, n - suf + 1))
        if not b:
            new.add(min(pre + 1, n))
    if suf:
        eq.append((len(before) - suf, len(before), n - suf))
    starts = [e[0] for e in eq]
    kept: set[int] = set()
    for a1, b1 in merge(ranges):
        for ln in range(a1, min(b1, len(before)) + 1):
            k = bisect.bisect_right(starts, ln - 1) - 1
            if k >= 0 and eq[k][0] <= ln - 1 < eq[k][1]:
                kept.add(eq[k][2] + (ln - 1 - eq[k][0]) + 1)
    return _from_lines({x for x in kept | new if 1 <= x <= n})


def fmt(ranges: Any) -> str:
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in merge(ranges))


def next_line(ranges: Any) -> int:
    """First line after the block that is edited contiguously from line 1 (1 when line 1 is not edited)."""
    rows = merge(ranges)
    return rows[0][1] + 1 if rows and rows[0][0] == 1 else 1


def targets(st: Any) -> list[str]:
    """The files a stage names: its scope (a sub-stage), else the files listed with it."""
    return list(getattr(st, "scope", None) or getattr(st, "files", None) or [])


def is_chunk_stage(st: Any) -> bool:
    """An editing-by-passes goal (rewrite, polish, translate ...) that names files."""
    return bool(targets(st)) and bool(INTENT.search(f"{getattr(st, 'title', '')}\n{getattr(st, 'goal', '')}"))


def overlaps(ranges: Any, lines: Any) -> bool:
    """True when one of the edited `ranges` touches the line range `lines` = (first, last)."""
    a, b = lines
    return any(x <= b and y >= a for x, y in merge(ranges))


def next_in(ranges: Any, lines: Any) -> int:
    """First line of `lines` = (first, last) that is not edited contiguously from its first line (last + 1 when the range is done)."""
    n = lines[0]
    for x, y in merge(ranges):
        if x <= n <= y:
            n = y + 1
    return n


def sync_range(st: Any, root: Any) -> None:
    """Keep a ranged stage's `lines` true for the file as it is now (see the module doc); a no-op for any other stage or an unreadable file."""
    rng = getattr(st, "lines", None)
    if not rng or root is None or not getattr(st, "scope", None):
        return
    f = st.scope[0]
    n = _total(root, f)
    if not n or st.lines_total <= 0:
        st.lines_total = st.lines_total or n
        return
    delta = n - st.lines_total
    if not delta:
        return
    a, b = rng
    if not merge((getattr(st, "edited", None) or {}).get(f)):
        a += delta
    b += delta
    a = min(max(1, a), n)
    b = min(max(a, b), n)
    old = f"{f} lines {rng[0]}-{rng[1]}"
    new = f"{f} lines {a}-{b}"
    st.title, st.goal = st.title.replace(old, new), st.goal.replace(old, new)
    st.lines, st.lines_total = (a, b), n


def _total(root: Any, rel: str) -> int:
    if root is None:
        return 0
    f = Path(rel) if Path(rel).is_absolute() else Path(root) / rel
    try:
        return len(f.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def cursor(st: Any, root: Any = None) -> dict[str, dict[str, Any]]:
    edited = getattr(st, "edited", None) or {}
    out: dict[str, dict[str, Any]] = {}
    for f in targets(st):
        if getattr(st, "lines", None) and list(st.scope[:1]) == [f]:
            sync_range(st, root)
            a, b = st.lines
            out[f] = {"edited": fmt([[max(x, a), min(y, b)] for x, y in merge(edited.get(f)) if x <= b and y >= a]), "next": next_in(edited.get(f), st.lines),
                      "total": b, "range": (a, b)}
        else:
            out[f] = {"edited": fmt(edited.get(f)), "next": next_line(edited.get(f)), "total": _total(root, f)}
    return out


def lines(cur: dict[str, dict[str, Any]]) -> list[str]:
    rows = []
    for f, c in list(cur.items())[:MAX_FILES]:
        if c.get("range"):
            rows.append(f"{f} lines {c['range'][0]}-{c['range'][1]}: " + ("done" if c["next"] > c["total"] else f"next unedited line {c['next']}"))
        elif not c["edited"]:
            rows.append(f"{f} not started" + (f" ({c['total']} lines)" if c["total"] else ""))
        elif c["total"] and c["next"] > c["total"]:
            rows.append(f"{f} done through line {c['total']}")
        else:
            rows.append(f"{f} line {c['next']} (edited {c['edited']}" + (f" of {c['total']})" if c["total"] else ")"))
    if len(cur) > MAX_FILES:
        rows.append(f"... and {len(cur) - MAX_FILES} more files")
    return rows


def note(cur: dict[str, dict[str, Any]]) -> str:
    """'Continue at: story.py line 201 (edited 1-200 of 406); dialogue.py not started' ('' when there are no files)."""
    return "Continue at: " + "; ".join(lines(cur)) if cur else ""


def next_chunk(cur: dict[str, dict[str, Any]]) -> str:
    """The next piece to write: the first file that is not finished, from its cursor line."""
    for f, c in cur.items():
        total = c["total"]
        if total and c["next"] > total:
            continue
        a = c["next"]
        b = min(a + CHUNK_LINES - 1, total) if total else a + CHUNK_LINES - 1
        return f"Rewrite lines {a}-{b} of {f} now (read them first if needed)."
    return ""
