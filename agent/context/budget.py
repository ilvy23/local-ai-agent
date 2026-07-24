from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 4.0

SECTION_ORDER = ("system", "skills", "repomap", "files", "history", "error")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class ContextBudget:
    limit: int
    sections: dict[str, int] = field(default_factory=dict)

    def add(self, section: str, text_or_tokens: str | int) -> int:
        tokens = (
            text_or_tokens
            if isinstance(text_or_tokens, int)
            else estimate_tokens(text_or_tokens)
        )
        self.sections[section] = self.sections.get(section, 0) + tokens
        return self.sections[section]

    def reset(self, section: str) -> None:
        self.sections.pop(section, None)

    @property
    def total(self) -> int:
        return sum(self.sections.values())

    @property
    def fraction(self) -> float:
        if self.limit <= 0:
            return 0.0
        return self.total / self.limit

    def over(self, threshold: float) -> bool:
        return self.fraction >= threshold

    def _ordered_items(self) -> list[tuple[str, int]]:
        known = [(s, self.sections[s]) for s in SECTION_ORDER if s in self.sections]
        extra = [(s, v) for s, v in self.sections.items() if s not in SECTION_ORDER]
        return known + sorted(extra)

    def render(self) -> str:
        items = self._ordered_items()
        label_w = max((len(s) for s, _ in items), default=6)
        value_w = max((len(f"{v:,}") for _, v in items), default=1)
        value_w = max(value_w, len(f"{self.total:,}"))

        lines = [f"{name:<{label_w}}  {count:>{value_w},}" for name, count in items]
        rule = "─" * (label_w + 2 + value_w)
        pct = round(self.fraction * 100)
        total_line = (
            f"{'total':<{label_w}}  {self.total:>{value_w},} / {self.limit:,}   ({pct}%)"
        )
        return "\n".join([*lines, rule, total_line])
