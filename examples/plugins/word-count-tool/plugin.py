"""A useful read-only tool: word / line / character counts of a file."""

from pathlib import Path

MAX_BYTES = 5_000_000        # refuse huge files rather than loading them


def register(api):
    def count(path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"{path} is not a file."
        if p.stat().st_size > MAX_BYTES:
            return f"{path} is larger than {MAX_BYTES // 1_000_000} MB; not counting it."
        text = p.read_text(encoding="utf-8", errors="replace")
        return f"{p.name}: {len(text.split())} words, {len(text.splitlines())} lines, {len(text)} characters"

    api.tool(
        "count_words",
        "Count the words, lines and characters in a text file. Use for questions like 'how long is this file'.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "path of the text file"}}, "required": ["path"]},
        count,
        risk="safe",            # only reads one file; changes nothing
    )
