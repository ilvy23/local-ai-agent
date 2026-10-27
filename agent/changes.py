"""Per-turn record of the files the agent created or edited, shown as a clean summary.

After every prompt that touched files the chat prints one line per file with the net number of
added (green) and removed (red) lines, e.g. `rps/game.py  +126 -0`. Several edits to the same file
within a turn are merged: the diff is taken between the file as it was before the first edit and
as it is now. Only write_file / replace_in_file are tracked (those are the tools that know the path);
files changed by a shell command are not.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.text import Text

TRACKED_TOOLS = ("write_file", "replace_in_file")
_MAX_READ_BYTES = 5 * 1024 * 1024
_MAX_DIFF_LINES = 20000


def read_text(path: str | os.PathLike[str]) -> str | None:
    """File contents, or None if it does not exist / is not a readable text file / is too big."""
    try:
        p = Path(path).expanduser()
        if not p.is_file() or p.stat().st_size > _MAX_READ_BYTES:
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def line_stats(before: str | None, after: str) -> tuple[int, int]:
    """(added, removed) lines going from `before` (None = new file) to `after`."""
    new = after.splitlines()
    if before is None:
        return len(new), 0
    old = before.splitlines()
    if len(old) + len(new) > _MAX_DIFF_LINES:           # too big for a real diff: net size change
        return max(len(new) - len(old), 0), max(len(old) - len(new), 0)
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def display_path(path: str, cwd: Path | None = None) -> str:
    """Path relative to the working directory when inside it, else with ~ for the home directory."""
    p = Path(path).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    base = (cwd or Path.cwd()).resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        pass
    from agent import verify

    root = verify.project_root(str(resolved))                  # outside the cwd: show it relative to its project
    if root is not None:
        try:
            return f"{root.name}/{resolved.relative_to(root)}"
        except ValueError:
            pass
    try:
        return "~/" + str(resolved.relative_to(Path.home()))
    except ValueError:
        return str(resolved)


class FileChangeLog:
    def __init__(self) -> None:
        self._files: dict[str, tuple[str | None, str]] = {}    # path -> (content before first edit, now)
        self.checkpoint: str | None = None                     # git checkpoint ref made before the first edit
        self.checkpoint_tried = False

    def clear(self) -> None:
        self._files.clear()
        self.checkpoint, self.checkpoint_tried = None, False

    def changed(self) -> dict[str, tuple[str | None, str]]:
        """Files whose content really changed this turn: path -> (before, after)."""
        return {p: v for p, v in self._files.items() if v[1] is not None and v[0] != v[1]}

    def record(self, path: str, before: str | None, after: str | None) -> None:
        """Note one write. `before` is the content just before it (None = file did not exist)."""
        if after is None:
            return
        first = self._files[path][0] if path in self._files else before
        self._files[path] = (first, after)

    def totals(self) -> list[tuple[str, int, int, bool]]:
        """(path, added, removed, created) for every file whose content really changed."""
        rows = []
        for path, (before, after) in self._files.items():
            if before is not None and before == after:
                continue
            added, removed = line_stats(before, after)
            rows.append((path, added, removed, before is None))
        return rows

    def render(self, cwd: Path | None = None) -> list[Text]:
        rows = self.totals()
        if not rows:
            return []
        names = [display_path(p, cwd) for p, *_ in rows]
        names = [n if len(n) <= 48 else "…" + n[-47:] for n in names]
        width = max(len(n) for n in names)
        total_add = sum(r[1] for r in rows)
        total_rem = sum(r[2] for r in rows)
        head = Text()
        head.append(f"{len(rows)} file{'s' if len(rows) != 1 else ''} changed", style="bold")
        head.append("  ")
        head.append(f"+{total_add}", style="green")
        head.append(" ")
        head.append(f"-{total_rem}", style="red")
        lines = [head]
        for name, (_p, added, removed, created) in zip(names, rows):
            line = Text("  ")
            line.append(name.ljust(width))
            line.append("  ")
            line.append(f"+{added}", style="green")
            line.append(" ")
            line.append(f"-{removed}", style="red")
            if created:
                line.append("  new", style="dim")
            lines.append(line)
        return lines

    def print_summary(self, console: Console, cwd: Path | None = None) -> None:
        lines = self.render(cwd)
        if lines:
            console.print()
            for line in lines:
                console.print(line, crop=False)
        self.clear()
