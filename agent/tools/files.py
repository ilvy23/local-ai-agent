"""File tools: read, list, search, and write, exposed to the agent loop.

These are safer and more precise than shelling out for common file work. Each
resolves paths with expanduser + resolve and returns friendly error STRINGS on
failure rather than raising — a handler exception would otherwise be surfaced to
the model as a generic error. The approval gate in tui.py classifies and audits
these calls (read/list/search are "safe" and auto-run; write is "caution" and
prompts), so the handlers here do NOT write audit rows themselves.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR
from agent.tools.registry import Tool

READ_MAX_CHARS = 20000
LIST_CAP = 500
SEARCH_MAX_HITS = 100
SEARCH_MAX_FILE_BYTES = 2_000_000
FIND_MAX_HITS = 100
TRUNCATED_MARKER = "\n[truncated]"
EXCLUDED_DIRS = {".git", "node_modules", ".venv", "data", "$RECYCLE.BIN", "System Volume Information"}
# Where external drives actually show up on this system - /media and
# /run/media are one level deeper (<root>/<username>/<drive>), /mnt drives
# sit directly under it. Without this, "find it on the HDD" costs the model
# several guess-a-path, list_dir, guess-again round trips before it even
# starts looking for the actual file.
_DRIVE_MOUNT_ROOTS = ("/media", "/run/media", "/mnt")


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


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def default_read_cap() -> int:
    """How many characters one read_file call returns by default.

    20 000 chars (~5k tokens) silently cut off any longer document, which is fatal for
    "summarise these files exactly". When a bigger context window was chosen for the chat,
    allow about 1.2 characters per context token - roughly 40 % of the window at ~3 chars
    per token - so a 200k context reads ~245 000 chars; with the default 16k context the
    old 20 000 stays (16 384 * 1.2 is below it)."""
    try:
        from agent import gpumode

        ctx = gpumode.chat_ctx_override()
    except Exception:  # noqa: BLE001 - never let a sizing helper break a read
        ctx = None
    return max(READ_MAX_CHARS, int(ctx * 1.2)) if ctx else READ_MAX_CHARS


def read_file(path: str, max_chars: int | None = None, **_kwargs: Any) -> str:
    """Return the text content of `path`, truncated to `max_chars` (default: sized to the
    chat's context window, see default_read_cap).

    Returns a friendly error string for missing, binary, or unreadable files.
    """
    if not max_chars or max_chars <= 0:
        max_chars = default_read_cap()
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

    `pattern` is a literal substring searched in file CONTENTS, not filenames
    (use find_files for that). Skips EXCLUDED_DIRS, binary/unreadable files,
    and anything over SEARCH_MAX_FILE_BYTES (source/text files worth grepping
    are essentially always small - without this cap, an overly broad `path`
    that happens to include large media files would read each one fully into
    memory just to discover it's binary, which is slow at best and a real
    resource hazard for a many-GB video). Capped at SEARCH_MAX_HITS hits.
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
            if candidate.stat().st_size > SEARCH_MAX_FILE_BYTES:
                continue
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


def disk_usage(path: str = "/", **_kwargs: Any) -> str:
    """Report free/used/total space on the filesystem containing `path` -
    useful before a large remux/transcode/copy/download, where running out
    of space mid-operation is a worse failure than checking first.
    """
    root = _resolve(path)
    if not root.exists():
        return f"Error: path not found: {root}"
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        return f"Error checking disk usage for {root}: {exc}"
    pct_used = usage.used / usage.total * 100 if usage.total else 0
    return (
        f"Disk usage for {safe_text(str(root))}:\n"
        f"- Total: {_human_size(usage.total)}\n"
        f"- Used: {_human_size(usage.used)} ({pct_used:.1f}%)\n"
        f"- Free: {_human_size(usage.free)}"
    )


def list_drives(**_kwargs: Any) -> str:
    """List currently mounted external drives, so a request like "it's on
    the HDD" doesn't need several rounds of guess-a-path/list_dir first
    just to discover the drive's exact mount point and name.
    """
    found: list[str] = []
    for root_str in _DRIVE_MOUNT_ROOTS:
        root = Path(root_str)
        if not root.is_dir():
            continue
        try:
            entries = [e for e in root.iterdir() if e.is_dir()]
        except OSError:
            continue
        if root_str == "/mnt":
            found.extend(safe_text(str(e)) for e in entries)
            continue
        # /media and /run/media are one level deeper: <root>/<username>/<drive>.
        for user_dir in entries:
            try:
                found.extend(
                    safe_text(str(d)) for d in user_dir.iterdir() if d.is_dir()
                )
            except OSError:
                continue
    if not found:
        return "No mounted drives found under /media, /run/media, or /mnt."
    lines = [f"- {p}" for p in sorted(set(found))]
    return "Mounted drives:\n" + "\n".join(lines)


def _expand_brace_glob(pattern: str) -> list[str]:
    """Expand one shell-style {a,b,c} group in a glob into several plain
    globs, e.g. '*.{srt,vtt}' -> ['*.srt', '*.vtt'] - pathlib's rglob has
    no brace-expansion of its own, so without this a multi-extension
    pattern (the common case: "find any subtitle file") would silently
    match nothing instead of raising anything the model could notice."""
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    options = match.group(1).split(",")
    return [pattern[: match.start()] + opt.strip() + pattern[match.end() :] for opt in options]


def find_files(name: str = "", path: str = "/", glob: str = "*", **_kwargs: Any) -> str:
    """Recursively search under `path` for files/directories whose name
    contains `name` (case-insensitive substring) - the "where is it" tool,
    for when you know roughly what a file is called but not which folder
    it's actually in. Unlike search_files (which greps file CONTENTS),
    this matches the file NAME itself, so it works on binary files (videos,
    subtitles, images) too. `glob` supports one {a,b,c} group for matching
    several extensions at once. Capped at FIND_MAX_HITS, stops early once hit.
    """
    root = _resolve(path)
    if not root.exists():
        return f"Error: path not found: {root}"
    if not root.is_dir():
        return f"Error: {root} is not a directory."

    needle = name.lower()
    where = f"'{name}' under {safe_text(str(root))}" if name else f"{glob} under {safe_text(str(root))}"
    seen: set[Path] = set()
    hits: list[str] = []
    try:
        for pattern in _expand_brace_glob(glob):
            for candidate in root.rglob(pattern):
                if candidate in seen:
                    continue
                if EXCLUDED_DIRS & set(candidate.relative_to(root).parts):
                    continue
                if needle and needle not in candidate.name.lower():
                    continue
                seen.add(candidate)
                kind = "directory" if candidate.is_dir() else "file"
                hits.append(f"- {safe_text(str(candidate))} ({kind})")
                if len(hits) >= FIND_MAX_HITS:
                    return "\n".join(
                        [f"Searching {where} — stopped at {FIND_MAX_HITS} hits:", *hits]
                    )
    except OSError as exc:
        return f"Error searching {root}: {exc}"

    if not hits:
        return f"No matches for {where}."
    noun = "match" if len(hits) == 1 else "matches"
    return "\n".join([f"Searching {where} — {len(hits)} {noun}:", *hits])


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


def replace_in_file(path: str, search: str, replace: str, **_kwargs: Any) -> str:
    """Replace the first exact occurrence of `search` with `replace` in `path`.

    For editing part of an existing file without rewriting the whole thing
    through write_file - `search` must match the file's actual text exactly
    (whitespace included), which is why the tool description tells the model
    to read_file first rather than guess. Refuses the data/ dir like
    write_file, for the same reason.
    """
    p = _resolve(path)
    data_dir = Path(DEFAULT_DATA_DIR).expanduser().resolve()
    if p == data_dir or data_dir in p.parents:
        return f"Error: refusing to edit under the protected data/ directory: {p}"
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
    if search not in text:
        return (
            f"Error: the given search text was not found in {p} verbatim. "
            "Read the file first and copy the exact snippet, including "
            "whitespace and indentation."
        )
    new_text = text.replace(search, replace, 1)
    try:
        p.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        return f"Error writing {p}: {exc}"
    return f"Replaced 1 occurrence in {p} ({len(new_text)} chars now)."


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
    description=(
        "Search text files under a directory for a literal substring "
        "APPEARING INSIDE THEIR CONTENTS (like grep) - source code, config, "
        "notes, etc. NOT for finding a file by its name/extension (a "
        "subtitle, a video, an image) - use find_files for that instead, "
        "since this tool reads every candidate file's actual bytes and "
        "skips large/binary ones rather than matching on the filename."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Literal substring to find INSIDE file contents (not a filename).",
            },
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

DISK_USAGE_TOOL = Tool(
    name="disk_usage",
    description=(
        "Check free/used/total space on the drive containing a path - use "
        "this before a large operation (transcoding a video, downloading, "
        "copying) where running out of space partway through is worse than "
        "just checking first."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Any path on the drive to check (e.g. a mount point).",
            },
        },
        "required": ["path"],
    },
    handler=disk_usage,
    risk="safe",
)

LIST_DRIVES_TOOL = Tool(
    name="list_drives",
    description=(
        "List currently mounted external/removable drives (checks /media, "
        "/run/media, and /mnt). Call this FIRST when the user refers to a "
        "drive by a loose name ('the HDD', 'my USB stick') instead of "
        "guessing mount paths and calling list_dir repeatedly."
    ),
    parameters={"type": "object", "properties": {}},
    handler=list_drives,
    risk="safe",
)

FIND_FILES_TOOL = Tool(
    name="find_files",
    description=(
        "Recursively search a directory tree for files/folders by NAME - "
        "use this instead of repeated list_dir calls when you know roughly "
        "what something is called but not which folder it's in. Works on "
        "any file type, including videos/images/subtitles that search_files "
        "(which greps text CONTENTS) can't look inside of. Combine with "
        "list_drives first when the user pointed at 'the HDD' rather than a "
        "specific path.\n"
        "`name` and `glob` both narrow the results (a hit must match BOTH) "
        "- only set `name` to a piece of text you have real reason to think "
        "is literally in the filename (part of a movie title, a language "
        "name like 'german'), never a guessed generic word like 'subtitle' "
        "or 'video' that a file is unlikely to be named after. When you're "
        "searching by file TYPE instead (subtitles, videos, images), leave "
        "`name` empty and use `glob` alone, e.g. glob='*.{srt,ass,ssa,vtt}' "
        "for any subtitle file. If a search comes back empty, don't give up "
        "- retry once with a broader glob and no `name` filter before "
        "telling the user nothing was found."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Substring to match against file/folder names, case-insensitive. "
                    "Only for text you have real reason to think is in the actual "
                    "filename - leave empty when searching by file type instead."
                ),
            },
            "path": {
                "type": "string",
                "description": "Directory to search under (recursively). Give a full path.",
            },
            "glob": {
                "type": "string",
                "description": (
                    "Filename glob, e.g. '*.srt' for one extension or "
                    "'*.{srt,ass,ssa,vtt}' for several. Default '*' (everything)."
                ),
            },
        },
        "required": ["name", "path"],
    },
    handler=find_files,
    risk="safe",
)

WRITE_FILE_TOOL = Tool(
    name="write_file",
    description=(
        "Write text to a new or existing file, creating parent directories "
        "as needed. Works for anything text-based: code, a report or notes "
        "in Markdown, a config file, a plain document - not just code. Use "
        "this for a NEW file, or to replace an existing file's ENTIRE "
        "contents; for a small edit to part of an existing file, use "
        "replace_in_file instead so the rest of the file is left alone."
    ),
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

REPLACE_IN_FILE_TOOL = Tool(
    name="replace_in_file",
    description=(
        "Replace one exact snippet of text in an existing file with new "
        "text, without rewriting the whole file. Prefer this over "
        "write_file for editing part of a file that already exists - "
        "smaller, safer, and doesn't risk losing content you didn't mean "
        "to touch. The `search` text must match the file exactly, "
        "including whitespace - read_file first and copy the snippet byte "
        "for byte."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "search": {"type": "string", "description": "Exact existing text to find."},
            "replace": {"type": "string", "description": "Text to put in its place."},
        },
        "required": ["path", "search", "replace"],
    },
    handler=replace_in_file,
    risk="caution",
)
