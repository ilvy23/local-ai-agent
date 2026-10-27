"""Turn raw command output into what a coding model needs: the failure, the code it points at, nothing else.

Two helpers used by the `run` tool:
  * `traceback_snippets` - for every project file named in a Python traceback, the lines around the failing one, so the model
    fixes the bug without spending a read step (each step costs a prefill plus a generation),
  * `summarize_pytest` - a long pytest report reduced to the failing tests, their assertion lines and the summary footer.
"""

from __future__ import annotations

import re
from pathlib import Path

_FRAME = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')
_PYTEST_HINT = re.compile(r"(=+ FAILURES =+|short test summary info|\d+ (?:failed|passed|error))")


def traceback_snippets(root: Path, output: str, context: int = 3, max_frames: int = 3, max_chars: int = 1800) -> str:
    root = Path(root).resolve()
    seen: list[tuple[Path, int, str]] = []
    for m in _FRAME.finditer(output):
        raw = m.group(1)
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        try:
            p = p.resolve()
        except OSError:
            continue
        if root not in p.parents or "site-packages" in p.parts or ".venv" in p.parts or not p.is_file():
            continue
        key = (p, int(m.group(2)), m.group(3) or "")
        if key not in seen:
            seen.append(key)
    if not seen:
        return ""
    blocks: list[str] = []
    for p, line, func in seen[-max_frames:]:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        lo, hi = max(1, line - context), min(len(lines), line + context)
        body = "\n".join(f"{'>>' if n == line else '  '} {n:4}| {lines[n - 1][:160]}" for n in range(lo, hi + 1))
        blocks.append(f"{p.relative_to(root)}:{line}" + (f" in {func}" if func else "") + "\n" + body)
    text = "\n".join(blocks)
    return ("\nSOURCE AT THE TRACEBACK (project files, so you can fix it without another read):\n" + text[:max_chars]) if text else ""


def summarize_pytest(output: str, max_chars: int = 2200) -> str:
    """Compact a long pytest report. Returns the input unchanged when it is short or not a pytest report."""
    if len(output) <= max_chars or not _PYTEST_HINT.search(output):
        return output
    keep: list[str] = []
    in_fail = False
    for line in output.splitlines():
        s = line.rstrip()
        if re.match(r"^_{3,} .+ _{3,}$", s):
            keep.append(s)
            in_fail = True
        elif s.startswith(("FAILED", "ERROR")) or s.startswith("E ") or re.match(r"^\S+\.py:\d+: ", s):
            keep.append(s[:220])
        elif in_fail and s.startswith(">"):
            keep.append(s[:220])
        elif re.search(r"\b\d+ (failed|passed|error)", s) or s.startswith("=") and ("passed" in s or "failed" in s):
            keep.append(s)
        elif s.startswith(("Interrupted", "collected ", "ModuleNotFoundError", "ImportError", "SyntaxError")):
            keep.append(s[:220])
    out = "\n".join(dict.fromkeys(keep))
    if not out.strip():
        return output
    return "PYTEST (condensed - failing tests, assertions, summary):\n" + out[:max_chars]
