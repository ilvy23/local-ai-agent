"""Shared terminal input helpers, used by both the interactive menu and the
settings browser so the two don't maintain separate copies of the same
subtle stdin-handling logic.
"""

from __future__ import annotations

import sys

try:
    import select
except ImportError:                       # pragma: no cover
    select = None                         # type: ignore[assignment]


def human_size(num_bytes: int) -> str:
    """Formats a byte count for display (e.g. "6.2 MB") - shared by every
    file-delivery path (send_file, images) so they show sizes consistently
    instead of each keeping its own copy of this."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def read_multiline(console, prompt: str) -> str:
    """Read a value that may span several lines. A pasted block hits stdin
    all at once, so after the first line we drain any lines already
    buffered - that keeps a copied multi-line block intact instead of
    leaking its tail into the next prompt. A normally typed single line
    has nothing buffered and returns as-is."""
    first = console.input(prompt)
    lines = [first]
    if sys.stdin.isatty() and sys.platform != "win32" and select is not None:    # select() on stdin is POSIX-only; on Windows a paste is one line
        try:
            while select.select([sys.stdin], [], [], 0.1)[0]:
                more = sys.stdin.readline()
                if not more:
                    break
                lines.append(more.rstrip("\n"))
        except (OSError, ValueError):
            pass  # stdin not selectable - just use what we have
    return "\n".join(lines).strip()
