from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent.context.budget import estimate_tokens

KEEP_RECENT = 3
DEFAULT_THRESHOLD = 0.75


@dataclass(frozen=True)
class CompactResult:
    messages: list[dict]
    folded: int
    summary_tokens: int

    @property
    def changed(self) -> bool:
        return self.folded > 0


def _is_system(message: dict) -> bool:
    return message.get("role") == "system"


def fold_messages(
    messages: list[dict],
    *,
    keep_recent: int = KEEP_RECENT,
    summarize: Callable[[list[dict]], str] | None = None,
) -> CompactResult:
    leading_system = []
    rest = list(messages)
    while rest and _is_system(rest[0]):
        leading_system.append(rest.pop(0))

    if len(rest) <= keep_recent:
        return CompactResult(messages=list(messages), folded=0, summary_tokens=0)

    to_fold = rest[:-keep_recent] if keep_recent else rest
    recent = rest[-keep_recent:] if keep_recent else []

    summary_text = summarize(to_fold) if summarize else _default_summary(to_fold)
    summary_message = {"role": "system", "content": f"[summary of earlier turns]\n{summary_text}"}

    folded_messages = [*leading_system, summary_message, *recent]
    return CompactResult(
        messages=folded_messages,
        folded=len(to_fold),
        summary_tokens=estimate_tokens(summary_text),
    )


def _default_summary(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "?")
        content = " ".join(str(message.get("content", "")).split())
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def should_compact(total_tokens: int, limit: int, threshold: float = DEFAULT_THRESHOLD) -> bool:
    if limit <= 0:
        return False
    return total_tokens / limit >= threshold
