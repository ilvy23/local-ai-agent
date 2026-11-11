"""Before the night starts: is every prompt something a machine can finish and prove? A prompt without a testable goal fails at 3 am instead of at 9 pm,
when you could still fix it. Findings are warnings (the run may still go): `check(items, root)` returns them as readable lines."""
from __future__ import annotations

import re
from pathlib import Path

from agent.night.queue import Item

_GOAL = re.compile(r"\b(test|tests|pytest|should|must|verify|verified|works?|passes?|green|returns?|prints?|raises?|exits?)\b", re.I)
_ASKS = re.compile(r"\b(ask me|let me know|check with me|tell me which|confirm with me)\b", re.I)
_PATH = re.compile(r"`([\w./-]+\.(?:py|js|ts|tsx|jsx|go|rs|md|json|toml|yaml|yml|sh|c|h|cpp|java))`")
MIN_CHARS, MAX_CHARS = 60, 14000


def check(items: list[Item], root: Path) -> list[str]:
    out: list[str] = []
    names = [i.name for i in items]
    for dup in {n for n in names if names.count(n) > 1}:
        out.append(f"two prompts are called {dup!r}: names must be unique (set `name:` in the front matter)")
    for it in items:
        t = it.text
        if len(t) < MIN_CHARS:
            out.append(f"[{it.name}] only {len(t)} characters: too vague to finish unattended")
        if len(t) > MAX_CHARS:
            out.append(f"[{it.name}] {len(t)} characters: split it into several prompts so each can be verified")
        if not _GOAL.search(t) and not it.verify:
            out.append(f"[{it.name}] nothing in it can be checked (no 'test', 'should', 'must', 'works' ...): add what proves it is done, or a `verify:` command")
        if _ASKS.search(t):
            out.append(f"[{it.name}] asks you something: nobody is there at night - decide it in the prompt")
        missing = [p for p in dict.fromkeys(_PATH.findall(t)) if not p.startswith(".night/") and not (root / p).exists()]       # .night/ files are the harness's own
        if missing:
            out.append(f"[{it.name}] mentions files that do not exist yet: {', '.join(missing[:4])} (fine if it should create them)")
        for d in it.depends:
            if d not in names:
                out.append(f"[{it.name}] depends on {d!r}, which is not in the queue")
            elif names.index(d) > names.index(it.name):
                out.append(f"[{it.name}] depends on {d!r}, which comes later in the queue: it would wait for the revisit round")
    return out
