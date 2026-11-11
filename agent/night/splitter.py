"""Split ONE big prompt into several prompts, one per stage, so each is revealed alone and verified alone.

Headers that start a part: `Stage 2: ...`, `Phase 1 - ...`, `Part 3.`, `Teil/Stufe N`, and Markdown headings `## ...` when there are at least two. Text before the
first header is shared context and is put at the top of every part. The same header words the coding agent's stage planner understands (stages.py)."""
from __future__ import annotations

import re
from pathlib import Path

_STAGE = re.compile(r"^[ \t]*[#>*_ \t]*(?:stage|stufe|phase|part|teil|step|schritt)[ \t]+(\d+)\b[ \t]*(?:[:.\-–—)]|$)", re.I | re.M)
_H2 = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.M)


def split(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(shared preamble, [(title, body)]). One part or none: the text is returned as the preamble and the list is empty."""
    marks = [(m.start(), text[m.start():text.find("\n", m.start()) if text.find("\n", m.start()) >= 0 else len(text)].strip(" #>*_\t")) for m in _STAGE.finditer(text)]
    if len(marks) < 2:
        marks = [(m.start(), m.group(1)) for m in _H2.finditer(text)]
    if len(marks) < 2:
        return text, []
    pre = text[:marks[0][0]].strip()
    parts = []
    for i, (start, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        parts.append((title, text[start:end].strip()))
    return pre, parts


def _slug(title: str, n: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    return f"{n:02d}-{s or 'part'}"


def write(text: str, folder: Path, chain: bool = True) -> list[Path]:
    """Write the parts as prompt files into `folder` (created). With `chain`, part N depends on part N-1. Returns the files; [] when the text has no parts."""
    pre, parts = split(text)
    if not parts:
        return []
    folder.mkdir(parents=True, exist_ok=True)
    files, prev = [], ""
    for i, (title, body) in enumerate(parts, 1):
        name = _slug(title, i)
        head = f"---\nname: {name}\n" + (f"depends: {prev}\n" if chain and prev else "") + "---\n"
        f = folder / f"{name}.md"
        f.write_text(head + (pre + "\n\n" if pre else "") + body + "\n")
        files.append(f)
        prev = name
    return files
