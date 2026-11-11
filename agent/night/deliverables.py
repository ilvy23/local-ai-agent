"""Files a prompt asks for by name ("create `tests/test_wren.py`", "add `wren.py`") must exist when the prompt counts as done. Measured, not claimed:
only paths that did not exist at the start and are introduced with a creating verb count (a path mentioned as an example does not)."""
from __future__ import annotations

import re
from pathlib import Path

_CREATE = re.compile(r"\b(?:create|add|write|new|implement|make|generate)\b[^.\n`]{0,60}`([\w./-]+\.[A-Za-z0-9]{1,5})`", re.I)


def expected(text: str, root: Path) -> list[str]:
    """Relative paths the prompt asks to create that do not exist yet."""
    out: list[str] = []
    for m in _CREATE.finditer(text):
        p = m.group(1).lstrip("./")
        if p and not p.startswith("..") and not (root / p).exists() and p not in out:
            out.append(p)
    return out


def missing(paths: list[str], root: Path) -> list[str]:
    return [p for p in paths if not (root / p).exists()]
