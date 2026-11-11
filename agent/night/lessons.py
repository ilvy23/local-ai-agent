"""Lessons learned during the night, kept in .night/LESSONS.md and shown to later attempts (Reflexion-style, but only FACTS: a lesson is
built from a measured failure kind and a measured detail, never from the model's own reflection, which can be wrong).

Each lesson is one line and appears once per (prompt kind, detail): the file never grows with repeats. `recent()` returns the newest few
within a character budget so the prompt of the next attempt stays small."""
from __future__ import annotations

from pathlib import Path

TEMPLATES = {
    "no_edits": "A previous attempt only read files and changed nothing. Start writing within the first few steps; read only what the next edit needs.",
    "loop": "A previous attempt repeated the same call over and over ({detail}). If a call fails or gives the same answer twice, change the call or the approach.",
    "broke_tests": "A previous attempt broke passing tests ({detail}). Run the tests after every change and never leave a change that turns a green test red.",
    "almost": "A previous attempt was nearly finished but broke a test ({detail}). Run the whole suite after every change; fix the one that broke before adding more.",
    "timeout": "A previous attempt used its whole time slice. Work in small verified steps instead of one big change.",
    "idle": "A previous attempt read and thought for half an hour without changing a file. Write the first small change early, then refine it.",
    "tests_hang": "A previous attempt left the test suite hanging ({detail}). Look for endless loops and anything that waits for input.",
    "stalled": "A previous attempt hung without any progress. Avoid commands that wait for input or run forever.",
    "crash": "A previous attempt crashed ({detail}). Keep commands short and simple.",
    "tests_red": "A previous attempt left the tests red ({detail}). Read the failing test first, then fix exactly that.",
    "incomplete": "A previous attempt was green but did not create what the task names ({detail}). Create exactly those files.",
    "partial": "A previous attempt could not finish a stage. Pick the smallest remaining piece and complete it fully.",
}


class Lessons:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _lines(self) -> list[str]:
        try:
            return [ln for ln in self.path.read_text().splitlines() if ln.strip()]
        except OSError:
            return []

    def add(self, kind: str, detail: str = "", prompt: str = "") -> bool:
        """Record the lesson of a failure kind. Returns False when it is unknown or already recorded."""
        tpl = TEMPLATES.get(kind)
        if tpl is None:
            return False
        line = f"- [{prompt or 'any'}/{kind}] " + tpl.format(detail=(detail or "")[:100].replace("\n", " "))
        if line in self._lines():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        return True

    def recent(self, prompt: str = "", budget: int = 900) -> str:
        """The newest lessons of this prompt (and the general ones) that fit into `budget` characters, newest first."""
        out: list[str] = []
        used = 0
        for ln in reversed(self._lines()):
            if f"[{prompt}/" not in ln and "[any/" not in ln:
                continue
            if used + len(ln) > budget:
                break
            out.append(ln)
            used += len(ln)
        return "\n".join(out)
