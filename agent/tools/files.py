"""File tools: read, list, search, and write, exposed to the agent loop.

These are safer and more precise than shelling out for common file work. Each
resolves paths with expanduser + resolve and returns friendly error STRINGS on
failure rather than raising — a handler exception would otherwise be surfaced to
the model as a generic error. The approval gate in tui.py classifies and audits
these calls (read/list/search are "safe" and auto-run; write is "caution" and
prompts), so the handlers here do NOT write audit rows themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR
from agent.tools.registry import Tool

READ_MAX_CHARS = 20000
LIST_CAP = 500
SEARCH_MAX_HITS = 100
TRUNCATED_MARKER = "\n[truncated]"
EXCLUDED_DIRS = {".git", "node_modules", ".venv", "data"}


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def read_file(path: str, max_chars: int = READ_MAX_CHARS, **_kwargs: Any) -> str:
    """Return the text content of `path`, truncated to `max_chars`.

    Returns a friendly error string for missing, binary, or unreadable files.
    """
    p = _resolve(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if p.is_dir():
        return f"Error: {path} is a directory, not a file."
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return f"Error reading {path}: {exc}"
    if _is_binary(raw):
        return f"Error: {path} appears to be a binary file."
    text = raw.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + TRUNCATED_MARKER
    return text


def list_dir(path: str = ".", **_kwargs: Any) -> str:
    """List entries in `path` as `name  type  size`, one per line (cap 500)."""
    p = _resolve(path)
    if not p.exists():
        return f"Error: directory not found: {path}"
    if not p.is_dir():
        return f"Error: {path} is not a directory."
    try:
        entries = sorted(p.iterdir(), key=lambda e: e.name)
    except OSError as exc:
        return f"Error listing {path}: {exc}"

    capped = entries[:LIST_CAP]
    lines = []
    for entry in capped:
        if entry.is_dir():
            lines.append(f"{entry.name}\tdir\t-")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            lines.append(f"{entry.name}\tfile\t{size}")
    if len(entries) > LIST_CAP:
        lines.append(f"[showing first {LIST_CAP} of {len(entries)} entries]")
    return "\n".join(lines) if lines else "(empty directory)"


def search_files(pattern: str, path: str = ".", glob: str = "*", **_kwargs: Any) -> str:
    """Substring-search text files under `path`, returning `path:line: text` hits.

    `pattern` is a literal substring. Skips EXCLUDED_DIRS and binary/unreadable
    files silently. Capped at SEARCH_MAX_HITS hits.
    """
    root = _resolve(path)
    if not root.exists():
        return f"Error: path not found: {path}"

    hits: list[str] = []
    for candidate in sorted(root.rglob(glob)):
        if not candidate.is_file():
            continue
        if EXCLUDED_DIRS & set(candidate.relative_to(root).parts):
            continue
        try:
            raw = candidate.read_bytes()
        except OSError:
            continue
        if _is_binary(raw):
            continue
        text = raw.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                hits.append(f"{candidate}:{lineno}: {line.strip()}")
                if len(hits) >= SEARCH_MAX_HITS:
                    hits.append(f"[stopped at {SEARCH_MAX_HITS} hits]")
                    return "\n".join(hits)
    if not hits:
        return f"No matches for '{pattern}'."
    return "\n".join(hits)


def write_file(path: str, content: str, append: bool = False, **_kwargs: Any) -> str:
    """Write `content` to `path` (creating parent dirs). Refuses the data/ dir.

    The data/ refusal is based on the RESOLVED absolute path, so `../data/x`
    or a symlink into data/ cannot slip through.
    """
    p = _resolve(path)
    data_dir = Path(DEFAULT_DATA_DIR).expanduser().resolve()
    if p == data_dir or data_dir in p.parents:
        return f"Error: refusing to write under the protected data/ directory: {p}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        return f"Error writing {path}: {exc}"
    verb = "appended" if append else "wrote"
    return f"{verb} {len(content)} chars to {p}"


READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read a text file and return its contents (truncated if large).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
            "max_chars": {"type": "integer", "description": "Max chars to return."},
        },
        "required": ["path"],
    },
    handler=read_file,
    risk="safe",
)

LIST_DIR_TOOL = Tool(
    name="list_dir",
    description="List the entries (name, type, size) in a directory.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default '.')."},
        },
        "required": [],
    },
    handler=list_dir,
    risk="safe",
)

SEARCH_FILES_TOOL = Tool(
    name="search_files",
    description="Search text files under a directory for a literal substring.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Literal substring to find."},
            "path": {"type": "string", "description": "Directory to search (default '.')."},
            "glob": {"type": "string", "description": "Filename glob (default '*')."},
        },
        "required": ["pattern"],
    },
    handler=search_files,
    risk="safe",
)

WRITE_FILE_TOOL = Tool(
    name="write_file",
    description="Write text to a file, creating parent directories as needed.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to."},
            "content": {"type": "string", "description": "Text content to write."},
            "append": {"type": "boolean", "description": "Append instead of overwrite."},
        },
        "required": ["path", "content"],
    },
    handler=write_file,
    risk="caution",
)
