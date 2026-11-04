"""`@path` in a message adds that file's whole text to it, so a long prompt can be given as a file: `@prompt.md` in the chat, or
`nightage code @prompt.md`.

    expand(text, cwd=None, limit=None) -> Expansion(text, files, notes)

Only existing regular text files count. Anything else (an e-mail address, a name that is not a file) stays exactly as typed.
Only the typed text is scanned, never the text of a file that was added. A message that is nothing but `@file` becomes "the
file below is the request". All the added files together get at most `limit` characters (`limit_for`: most of the window the
turn is sent with). Beyond that the start of the file is kept and the cut is marked."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import textfile

MIN_LIMIT = 20_000                     # characters all added files may take together, at the least
SHARE = 0.7                            # else this share of the chat's history budget
MAX_FILE_BYTES = 8 * 2**20             # a prompt file is small; a bigger one is for read_file, not for one message
_MENTION = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|(\S+))')
_TRAILING = ".,;:!?)]}'\""
ONLY_FILES = "The user's message is the text of the file below. Read all of it and follow it as their request."
TASK_NOTE = "[This task is the text of the file {name} in the project folder: read it again for the details of any part.]"


@dataclass
class Expansion:
    text: str
    files: list[tuple[str, int]] = field(default_factory=list)     # (name as typed, characters of the file)
    notes: list[str] = field(default_factory=list)


def limit_for(config: dict[str, Any] | None = None) -> int:
    """How many characters added files may take: 70 % of the chat's history budget (the context the turn is sent with)."""
    try:
        if config is not None:
            from agent.memory.recall import history_char_budget

            return max(MIN_LIMIT, int(history_char_budget(config) * SHARE))
        from agent.tools.files import default_read_cap

        return max(MIN_LIMIT, default_read_cap())
    except Exception:  # noqa: BLE001 - sizing must never break a message
        return MIN_LIMIT


def _find(raw: str, cwd: Path) -> tuple[Path, str] | None:
    for cand in dict.fromkeys((raw, raw.rstrip(_TRAILING))):        # "see @notes.md." -> notes.md
        if not cand:
            continue
        p = Path(os.path.expanduser(cand))
        p = p if p.is_absolute() else cwd / p
        try:
            if p.is_file():
                return p, cand
        except OSError:
            continue
    return None


def expand(text: str, cwd: str | Path | None = None, limit: int | None = None, *, as_task: bool = False) -> Expansion:
    """`as_task` (the coding agent): a message that is only `@file` becomes the file's own text, so the task starts with the
    spec's first lines (stage titles, "create a new project" detection), followed by TASK_NOTE naming the file."""
    base = Path(cwd) if cwd is not None else Path.cwd()
    room = limit if limit is not None else limit_for()
    blocks: list[str] = []
    bodies: list[str] = []
    out = Expansion(text)
    used: list[tuple[int, int]] = []
    seen: set[str] = set()
    for m in _MENTION.finditer(text):
        hit = _find(m.group(1) or m.group(2), base)
        if hit is None:
            continue
        path, name = hit
        key = str(path.resolve())
        if key in seen:
            used.append(m.span())
            continue
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                out.notes.append(f"@{name} is too big for one message ({size / 2**20:.1f} MB) - it was not added; ask to read it instead")
                continue
            raw = path.read_bytes()
        except OSError as exc:
            out.notes.append(f"@{name} could not be read ({exc.strerror or exc}) - it was not added")
            continue
        if textfile.looks_binary(raw):
            out.notes.append(f"@{name} is not a text file - it was not added")
            continue
        if room <= 0:
            out.notes.append(f"@{name} was not added: the files before it already fill the context")
            continue
        content, _info = textfile.decode(raw)
        content = content.replace("\r\n", "\n")
        n = len(content)
        if n > room:
            out.notes.append(f"@{name} is longer than the context allows: the first {room:,} of {n:,} characters were added")
            content = (content[:room] + f"\n[... cut here: {name} has {n:,} characters, only the first {room:,} fit into the context."
                       " Read the rest with read_file.]")
        room -= min(n, room)
        seen.add(key)
        used.append(m.span())
        blocks.append(f'<file name="{name}" chars="{n}">\n{content}\n</file>')
        bodies.append(content)
        out.files.append((name, n))
    if not blocks:
        return out
    rest = text
    for start, end in sorted(used, reverse=True):
        rest = rest[:start] + rest[end:]
    if as_task and not rest.strip():
        out.text = "\n\n".join(bodies) + "\n\n" + " ".join(TASK_NOTE.format(name=name) for name, _n in out.files)
        return out
    lead = ONLY_FILES if not rest.strip() else text.rstrip()
    out.text = lead + "\n\n" + "\n\n".join(blocks)
    return out
