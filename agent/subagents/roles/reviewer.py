from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.subagents.spawn import SubagentSpec, spawn

_SYSTEM = (
    "You are a code reviewer. You review a diff for correctness bugs, missing "
    "tests, and unclear naming. You return concerns only — you cannot approve, "
    "apply, or edit anything. List each concern on its own line prefixed with "
    "'- '. If you find nothing, reply exactly 'NONE'."
)


def _build_prompt(inputs: dict[str, Any]) -> str:
    task = str(inputs.get("task", "")).strip()
    diff = str(inputs.get("diff", "")).strip()
    return f"## Task\n{task}\n\n## Diff\n{diff}\n\nList concerns, most severe first."


def _parse(output: str) -> list[str]:
    text = output.strip()
    if text.upper().startswith("NONE") or not text:
        return []
    concerns = []
    for raw in text.splitlines():
        line = raw.strip()
        bulleted = line.startswith(("- ", "* "))
        numbered = line[:1].isdigit() and line[1:2] in ".)"
        if bulleted or numbered:
            concerns.append(line[2:].strip())
    return [c for c in concerns if c]


REVIEWER_SPEC = SubagentSpec(
    role="reviewer",
    system=_SYSTEM,
    build_prompt=_build_prompt,
    parse=_parse,
)


@dataclass(frozen=True)
class Review:
    concerns: list[str]

    @property
    def clean(self) -> bool:
        return not self.concerns


def review_diff(
    client: Any,
    diff: str,
    task: str,
    *,
    model: str,
    num_ctx: int | None = None,
) -> Review:
    result = spawn(
        client,
        REVIEWER_SPEC,
        {"task": task, "diff": diff},
        model=model,
        num_ctx=num_ctx,
    )
    return Review(concerns=result.output)
