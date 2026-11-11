"""Optional front matter at the top of a prompt file:

    ---
    name: characters        (default: the file name)
    weight: 2               (share of the night, relative to the other prompts; default 1)
    verify: python -m pytest tests/test_characters.py -q     (an extra command that must exit 0 for the prompt to count as done)
    depends: prompt1, prompt2                                (these must be done first)
    kind: bug               (a failing test is written first and must flip)
    plan: true              (a checklist plan is written first; the harness ticks its steps by measurement)
    prepare: <command>      (run in the project before the prompt starts, e.g. to inject the bugs an evaluation prompt must find)
    score: <command>        (run after the prompt ended; the last JSON line it prints is the measured result)
    ---

A tiny `key: value` reader: no YAML dependency, unknown keys are ignored, a malformed block means "no front matter" (the text stays as it is)."""
from __future__ import annotations

import re

_BLOCK = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?", re.S)
KEYS = ("name", "weight", "verify", "depends", "kind", "plan", "prepare", "score")


def split(text: str) -> tuple[dict[str, str], str]:
    """(metadata, the prompt text without the block)."""
    m = _BLOCK.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        key, sep, value = line.partition(":")
        key = key.strip().lower()
        if sep and key in KEYS and value.strip():
            meta[key] = value.strip()
        elif line.strip() and not sep:
            return {}, text                                 # not a key: value block: leave the text alone
    return meta, text[m.end():].lstrip("\n")


def weight(meta: dict[str, str]) -> float:
    try:
        return min(max(float(meta.get("weight", 1)), 0.1), 20.0)
    except ValueError:
        return 1.0


def depends(meta: dict[str, str]) -> list[str]:
    return [d.strip() for d in re.split(r"[,\s]+", meta.get("depends", "")) if d.strip()]
