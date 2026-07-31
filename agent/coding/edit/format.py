"""Parse a model's reply into structured file edits.

One canonical format the 7B is asked to produce:

    ### FILE: path/to/file.py
    ```python
    <full new file contents>
    ```

For a large file it may emit search/replace blocks inside the fence instead:

    ### FILE: path/to/file.py
    ```
    <<<<<<< SEARCH
    <exact existing lines>
    =======
    <replacement lines>
    >>>>>>> REPLACE
    ```

Anything we can't parse is reported as `malformed` so the caller can re-ask
without spending a repair attempt (Aider tracks this; so do we).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_BLOCK = re.compile(
    r"###\s*FILE:\s*(?P<path>\S+)\s*\n```[^\n]*\n(?P<body>.*?)\n?```",
    re.DOTALL,
)
# Fallback: a fenced block whose first line is `# file: path`
_INLINE = re.compile(
    r"```[^\n]*\n#\s*file:\s*(?P<path>\S+)\s*\n(?P<body>.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)
_SR = re.compile(
    r"<{5,}\s*SEARCH\s*\n(?P<search>.*?)\n={5,}\s*\n(?P<replace>.*?)\n>{5,}\s*REPLACE",
    re.DOTALL,
)
_FENCE = re.compile(r"```[^\n]*\n(?P<body>.*?)\n?```", re.DOTALL)


@dataclass
class Edit:
    path: str
    whole: str | None = None                       # full-file replacement
    replacements: list[tuple[str, str]] = field(default_factory=list)  # (search, replace)


@dataclass
class ParseResult:
    edits: list[Edit]
    malformed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.edits) and not self.malformed


def parse_edits(text: str, default_path: str | None = None) -> ParseResult:
    edits: list[Edit] = []
    malformed: list[str] = []

    matches = list(_BLOCK.finditer(text)) or list(_INLINE.finditer(text))
    if not matches:
        # Common with single-file edits: the model drops the `### FILE:` header and
        # emits one bare ```code``` block. If we know the single target, accept it.
        fences = _FENCE.findall(text)
        if default_path and len(fences) == 1:
            body = fences[0]
            reps = [(s.group("search"), s.group("replace")) for s in _SR.finditer(body)]
            if reps:
                return ParseResult([Edit(path=default_path, replacements=reps)])
            return ParseResult([Edit(path=default_path, whole=body)])
        return ParseResult([], ["no `### FILE:` edit blocks found in the reply"])

    for m in matches:
        path = m.group("path").strip().strip("`")
        body = m.group("body")
        if "<<<<<<" in body or "SEARCH" in body and "=====" in body:
            reps = [(s.group("search"), s.group("replace")) for s in _SR.finditer(body)]
            if not reps:
                malformed.append(f"{path}: looks like a search/replace block but couldn't parse it")
                continue
            edits.append(Edit(path=path, replacements=reps))
        else:
            edits.append(Edit(path=path, whole=body))

    if not edits and not malformed:
        malformed.append("no usable edits parsed")
    return ParseResult(edits, malformed)


if __name__ == "__main__":
    whole = """Here's the fix:

### FILE: mod.py
```python
def add(a, b):
    return a + b
```
Done."""
    r = parse_edits(whole)
    assert r.ok and len(r.edits) == 1 and r.edits[0].whole.strip() == "def add(a, b):\n    return a + b"

    sr = """### FILE: big.py
```
<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE
```"""
    r = parse_edits(sr)
    assert r.ok and r.edits[0].replacements == [("    return a - b", "    return a + b")], r.edits

    bad = parse_edits("just some prose, no edit blocks")
    assert not bad.ok and bad.malformed

    # single bare fenced block, no header → accepted when the target is known
    bare = "```python\ndef add(a, b):\n    return a + b\n```"
    assert not parse_edits(bare).ok  # ambiguous without a target
    r = parse_edits(bare, default_path="mod.py")
    assert r.ok and r.edits[0].path == "mod.py" and "a + b" in r.edits[0].whole

    print("edit-format self-check passed ✓")
