from __future__ import annotations

import re
from pathlib import Path

_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')


def enrich_traceback(traceback_text: str, root: Path, *, context_lines: int = 3) -> str:
    root = Path(root).resolve()
    out: list[str] = []

    for raw in traceback_text.splitlines():
        out.append(raw)
        match = _FRAME_RE.match(raw)
        if match is None:
            continue
        snippet = _frame_context(
            match.group("file"), int(match.group("line")), root, context_lines
        )
        if snippet:
            indent = " " * (len(raw) - len(raw.lstrip()) + 4)
            out.extend(f"{indent}{line}" for line in snippet)

    return "\n".join(out)


def _frame_context(
    file_str: str, lineno: int, root: Path, context_lines: int
) -> list[str] | None:
    try:
        path = Path(file_str).resolve()
    except (OSError, ValueError):
        return None
    if path != root and root not in path.parents:
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None

    start = max(1, lineno - context_lines)
    end = min(len(lines), lineno + context_lines)
    width = len(str(end))
    rendered: list[str] = []
    for n in range(start, end + 1):
        marker = "->" if n == lineno else "  "
        rendered.append(f"{marker} {n:>{width}} | {lines[n - 1]}")
    return rendered


def summarize_failure(check_name: str, detail: str, *, max_chars: int = 2000) -> str:
    detail = detail.strip()
    if len(detail) > max_chars:
        detail = "[... earlier output trimmed ...]\n" + detail[-max_chars:]
    return f"[{check_name} failed]\n{detail}"
