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


def safe_text(text: str) -> str:
    """Make `text` encodable as UTF-8, recovering legacy filename bytes.

    Filenames are bytes, not text, and nothing guarantees they're UTF-8. Python
    surfaces undecodable bytes as lone surrogates (os.listdir uses
    surrogateescape), and those blow up the moment anything encodes the string
    as UTF-8 — such as sending a tool result to Ollama as JSON. One music folder
    named `Alizée` in latin-1 was enough to kill a whole chat session.

    Recover the original bytes and read them as latin-1: it never fails and gets
    the common legacy case right. Anything still undecodable becomes U+FFFD.
    """
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        raw = text.encode("utf-8", "surrogateescape")
        return raw.decode("latin-1", "replace")


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def read_file(path: str, max_chars: int = READ_MAX_CHARS, **_kwargs: Any) -> str:
    """Return the text content of `path`, truncated to `max_chars`.

    Returns a friendly error string for missing, binary, or unreadable files.
    """
    p = _resolve(path)
    if not p.exists():
        return f"Error: file not found: {p}"
    if p.is_dir():
        return f"Error: {p} is a directory, not a file."
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return f"Error reading {p}: {exc}"
    if _is_binary(raw):
        return f"Error: {p} appears to be a binary file."
    text = raw.decode("utf-8", errors="replace")
    # Head the content with the file it came from: raw text alone is
    # unattributable, so the model can't tell two files apart or notice it read
    # the wrong one — and a truncated read looks identical to a whole file.
    if len(text) > max_chars:
        header = f"File {p} (first {max_chars} of {len(text)} chars):"
        return f"{header}\n{text[:max_chars]}{TRUNCATED_MARKER}"
    return f"File {p} ({len(text)} chars):\n{text}"


def list_dir(path: str = ".", **_kwargs: Any) -> str:
    """List entries in `path` as `name  type  size`, one per line (cap 500).

    The result is headed with the resolved absolute path. Without it the model
    only sees bare names and can't tell which directory a listing came from —
    which is how it ends up describing the wrong folder's contents.
    """
    p = _resolve(path)
    if not p.exists():
        return f"Error: directory not found: {p}"
    if not p.is_dir():
        return f"Error: {p} is not a directory."
    try:
        entries = sorted(p.iterdir(), key=lambda e: e.name)
    except OSError as exc:
        return f"Error listing {p}: {exc}"

    if not entries:
        return f"{safe_text(str(p))} is empty (0 entries)."

    capped = entries[:LIST_CAP]
    noun = "entry" if len(entries) == 1 else "entries"
    lines = [f"Contents of {safe_text(str(p))} ({len(entries)} {noun}):"]
    # One readable row per entry. A bare `name<TAB>dir<TAB>-` reads as ambiguous
    # to a small model — it mistook the name for part of the header above and
    # reported "no name specified". Naming the fields removes the guesswork.
    for entry in capped:
        name = safe_text(entry.name)
        if entry.is_dir():
            lines.append(f"- {name}/ (directory)")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            lines.append(f"- {name} (file, {size} bytes)")
    if len(entries) > LIST_CAP:
        lines.append(f"[showing first {LIST_CAP} of {len(entries)} entries]")
    return "\n".join(lines)


def search_files(pattern: str, path: str = ".", glob: str = "*", **_kwargs: Any) -> str:
    """Substring-search text files under `path`, returning `path:line: text` hits.

    `pattern` is a literal substring. Skips EXCLUDED_DIRS and binary/unreadable
    files silently. Capped at SEARCH_MAX_HITS hits.
    """
    root = _resolve(path)
    if not root.exists():
        return f"Error: path not found: {root}"

    hits: list[str] = []
    where = f"'{pattern}' under {safe_text(str(root))}" + (f" matching {glob}" if glob != "*" else "")
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
                hits.append(f"{safe_text(str(candidate))}:{lineno}: {line.strip()}")
                if len(hits) >= SEARCH_MAX_HITS:
                    return "\n".join(
                        [f"Searched {where} — stopped at {SEARCH_MAX_HITS} hits:", *hits]
                    )
    # Say where we looked, especially when we found nothing: "no matches" without
    # a location invites the model to conclude the thing doesn't exist at all.
    if not hits:
        return f"No matches for {where}."
    noun = "match" if len(hits) == 1 else "matches"
    return "\n".join([f"Searched {where} — {len(hits)} {noun}:", *hits])


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
    description=(
        "List the entries (name, type, size) in a directory. The reply is headed "
        "with the absolute path that was listed — report that folder, not another."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory to list. Give a full path like /home/user/Music. "
                    "'.' is the directory the agent was started in, which is "
                    "rarely what the user means."
                ),
            },
        },
        "required": ["path"],
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
            "path": {
                "type": "string",
                "description": (
                    "Directory to search. Give a full path like /home/user/project. "
                    "'.' is the directory the agent was started in, which is rarely "
                    "what the user means."
                ),
            },
            "glob": {"type": "string", "description": "Filename glob (default '*')."},
        },
        "required": ["pattern", "path"],
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
