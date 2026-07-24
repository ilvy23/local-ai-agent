from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum


class Tier(int, Enum):
    EXERCISER = 0
    ACCUMULATING = 1
    FRESH_CONTEXT = 2
    ESCALATED = 3
    HANDBACK = 4


def normalized_ast(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return ast.dump(tree, annotate_fields=False)


@dataclass(frozen=True)
class Attempt:
    source: str
    error: str = ""
    blocker: str | None = None
    ast_dump: str | None = field(default=None)

    @classmethod
    def make(cls, source: str, error: str = "", blocker: str | None = None) -> Attempt:
        return cls(source=source, error=error, blocker=blocker, ast_dump=normalized_ast(source))


@dataclass(frozen=True)
class StallVerdict:
    stalled: bool
    reason: str = ""


def detect_stall(current: Attempt, previous: Attempt | None) -> StallVerdict:
    if previous is None:
        return StallVerdict(False)
    if current.ast_dump is not None and current.ast_dump == previous.ast_dump:
        return StallVerdict(True, "identical AST to previous attempt")
    if current.error and current.error == previous.error:
        return StallVerdict(True, "byte-identical error to previous attempt")
    return StallVerdict(False)


def is_specific_blocker(blocker: str | None) -> bool:
    if not blocker:
        return False
    text = blocker.strip()
    if len(text) < 4:
        return False
    vague = ("maybe", "may need", "might", "possibly", "refactor", "not sure", "unclear")
    low = text.lower()
    if any(word in low for word in vague):
        return False
    return any(ch in text for ch in "/._") or any(part[:1].isupper() for part in text.split())


@dataclass
class RepairState:
    max_attempts: int
    tier: Tier = Tier.ACCUMULATING
    attempts: list[Attempt] = field(default_factory=list)

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    @property
    def count(self) -> int:
        return len(self.attempts)

    @property
    def previous(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None


class Decision(str, Enum):
    RETRY = "retry"
    SATISFY_BLOCKER = "satisfy_blocker"
    ESCALATE = "escalate"
    HANDBACK = "handback"


@dataclass(frozen=True)
class RepairDecision:
    action: Decision
    tier: Tier
    reason: str = ""


def decide_next(state: RepairState, latest: Attempt) -> RepairDecision:
    previous = state.previous
    state.record(latest)

    if is_specific_blocker(latest.blocker):
        stall = detect_stall(latest, previous)
        if stall.stalled:
            return RepairDecision(Decision.ESCALATE, _next_tier(state.tier), latest.blocker or "")
        return RepairDecision(Decision.SATISFY_BLOCKER, state.tier, latest.blocker or "")

    stall = detect_stall(latest, previous)
    if stall.stalled:
        return RepairDecision(Decision.ESCALATE, _next_tier(state.tier), stall.reason)

    if state.count >= state.max_attempts:
        return RepairDecision(Decision.ESCALATE, _next_tier(state.tier), "attempt cap reached")

    return RepairDecision(Decision.RETRY, state.tier)


def _next_tier(tier: Tier) -> Tier:
    if tier is Tier.ACCUMULATING:
        return Tier.FRESH_CONTEXT
    if tier is Tier.FRESH_CONTEXT:
        return Tier.ESCALATED
    if tier is Tier.ESCALATED:
        return Tier.HANDBACK
    return Tier.HANDBACK
