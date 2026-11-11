"""The prompt queue: files (a folder is read in natural order: prompt2 before prompt10) or inline text, one item each."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.night import frontmatter

RULES_NAMES = ("rules.md", "rules.txt")            # shared rules in a prompt folder: prepended to every prompt, never a prompt of their own


@dataclass
class Item:
    name: str
    text: str
    status: str = "pending"                        # pending | active | done | partial
    attempts: int = 0
    seconds: float = 0.0
    note: str = ""                                 # why it is partial / what the last attempt ended with
    kinds: list[str] = field(default_factory=list)         # the failure kinds of the attempts so far (analyzer.py)
    weight: float = 1.0                            # share of the night this prompt may use, relative to the others
    verify: str = ""                               # an extra command that must exit 0 for the prompt to count as done (prompt front matter)
    depends: list[str] = field(default_factory=list)   # names of prompts that must be done first
    kind: str = ""                                 # "bug": a failing test is written first and must flip
    plan: bool = False                             # write a checklist plan first (planfile.py), tick it by measurement after every attempt
    prepare: str = ""                             # a shell command run in the project before the prompt starts (an evaluation prompt injects its bugs / records its golden master here)
    score: str = ""                               # a shell command run after the prompt ended; the last JSON line it prints is the measured result (kept in the evidence)
    prepared: bool = False
    scored: dict | None = None
    grace_used: bool = False                       # the one extra stretch of time an unfinished-but-productive prompt gets after its slice (governor._grace)
    repro: list[str] = field(default_factory=list)     # kind "bug": the ids of the test that proves the bug (it must fail first, then pass)
    repro_sha: str = ""                            # the commit that holds that failing test: a roll back of the fix goes here, not before the test
    expected: list[str] = field(default_factory=list)  # files the prompt asks to create (deliverables.py), measured when the prompt starts


def _natural(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load(sources: list[str]) -> tuple[list[Item], str]:
    """(items in the order they will be revealed, the shared rules text)."""
    items: list[Item] = []
    rules: list[str] = []
    for src in sources:
        p = Path(src).expanduser()
        if p.is_dir():
            files = sorted((f for f in p.iterdir() if f.is_file() and f.suffix in (".md", ".txt")), key=_natural)
        elif p.is_file():
            files = [p]
        else:
            if src.strip():
                items.append(Item(f"inline-{len(items) + 1}", src.strip()))
            continue
        for f in files:
            text = f.read_text(errors="replace").strip()
            if f.name.lower() in RULES_NAMES:
                rules.append(text)
            elif text:
                meta, body = frontmatter.split(text)
                if body.strip():
                    items.append(Item(meta.get("name", f.stem), body.strip(), weight=frontmatter.weight(meta), verify=meta.get("verify", ""), prepare=meta.get("prepare", ""), score=meta.get("score", ""),
                                      depends=frontmatter.depends(meta), kind=meta.get("kind", "").lower(), plan=meta.get("plan", "").lower() in ("true", "yes", "on", "1")))
    return items, "\n\n".join(r for r in rules if r)
