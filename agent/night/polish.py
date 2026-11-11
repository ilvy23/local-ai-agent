"""`nightage night polish PROMPT.md`: ask the model to rewrite a vague prompt into one a local 27B model can finish and the harness can verify - explicit stages with a
check each, files to touch, constraints, no questions to the user. The original is NEVER overwritten: the result goes to PROMPT.polished.md, and it is only offered when the
prompt check (preflight) finds fewer problems in it than in the original. `ask(text) -> reply` is one model call, injected so this can be tested without a model."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from agent.night import preflight
from agent.night.queue import Item

META = """You rewrite coding tasks for an autonomous coding agent that runs unattended for hours on a local 27B model. Rewrite the task below so that it can be finished and PROVEN:
- keep every requirement of the original, add none that were not meant;
- split it into numbered stages ("Stage 1: ...") of one concern each, in a sensible order, each stage small enough for one sitting;
- end every stage with a line "Check:" that names something a machine can verify (a test that must pass, a file that must exist, a command that must exit 0);
- name the files or modules to start from when the task says or implies them; do not invent file names;
- no questions to the user (decide), no vague words like "nice" or "better" without a measurable meaning;
- write plain Markdown, English or the language of the original, no preamble, no explanation, only the rewritten task.

TASK:
{task}
"""


def problems(text: str, root: Path) -> list[str]:
    return preflight.check([Item("prompt", text)], root)


def polish(text: str, root: Path, ask: Callable[[str], str]) -> tuple[str | None, list[str], list[str]]:
    """(the better text or None, problems before, problems after). None when the reply is empty, shorter than half the original, or not better by the prompt check."""
    before = problems(text, root)
    try:
        reply = (ask(META.format(task=text.strip())) or "").strip()
    except Exception:  # noqa: BLE001 - no model: nothing to offer
        return None, before, before
    if reply.startswith("```"):
        reply = reply.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if len(reply) < len(text.strip()) / 2:
        return None, before, before
    after = problems(reply, root)
    if len(after) >= max(len(before), 1) and before:
        return None, before, after
    return reply, before, after


def target(path: Path) -> Path:
    return path.with_name(path.stem + ".polished" + path.suffix)
