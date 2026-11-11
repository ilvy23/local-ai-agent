"""Prompt templates for common night jobs: `nightage night init DIR` writes them as a prompt folder (with front matter and shared rules).
Each one is written the way the preflight wants it: a goal that can be checked, no questions to the user, a size the model can finish."""
from __future__ import annotations

from pathlib import Path

RULES = """Rules for every task of this night:
- Inspect the existing code before changing it. Do not rewrite working parts.
- Make small, incremental changes; run the tests after each one and fix what you broke.
- Never delete, skip or weaken a test to get green. Never edit CI files or lockfiles.
- Keep existing behaviour and saved data compatible unless the task says otherwise.
- Do not add dependencies or new abstractions the task does not need.
- Nobody is awake to answer: decide yourself, and write down the decision in the code or in a short comment.
"""

TEMPLATES: dict[str, str] = {
    "01-feature.md": """---
name: feature
weight: 2
---
Add this feature to the project: <DESCRIBE THE FEATURE IN 2-4 SENTENCES: what the user sees or can do afterwards>.

Do it in the style of the existing code (look at how a similar feature is built first). Add tests that show it works, including one edge case.
It is done when the whole test suite passes and the new tests exist.
""",
    "02-bug.md": """---
name: bug
kind: bug
---
There is a bug: <WHAT HAPPENS> instead of <WHAT SHOULD HAPPEN>. Steps to reproduce: <STEPS>.

Find the cause (not just the symptom), fix it, and make sure the new test fails without the fix. It is done when that test and the whole suite pass.
""",
    "03-coverage.md": """---
name: coverage
weight: 1.5
---
Raise the test coverage of the project's most important untested code. Find the modules with the least coverage that matter (look at what the program does),
and write meaningful tests for them: real behaviour, edge cases, error paths. Do not write tests that always pass.
It is done when at least 10 new tests exist, each of them fails if the code it checks is broken, and the whole suite passes.
""",
    "04-refactor.md": """---
name: refactor
weight: 1.5
---
Refactor <THE MODULE OR AREA>: <WHY, e.g. it is too long, duplicated, hard to follow>. Behaviour must not change at all.
Work in small steps, run the tests after every step. It is done when the whole suite passes unchanged (no test edited) and the code is shorter or clearer.
""",
    "05-docs.md": """---
name: docs
weight: 0.5
---
Bring the documentation in line with the code: the README must describe how to install, run and test the project as it really works now
(check every command you write by running it), and every public function that lacks a docstring gets a short, true one.
It is done when the commands in the README work and the tests still pass.
""",
}


def init(folder: Path, names: list[str] | None = None) -> list[Path]:
    """Write rules.md and the chosen templates (all by default) into `folder`; existing files are never overwritten. Returns the files written."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in {"rules.md": RULES, **{k: v for k, v in TEMPLATES.items() if not names or any(n in k for n in names)}}.items():
        f = folder / name
        if not f.exists():
            f.write_text(text)
            written.append(f)
    return written
