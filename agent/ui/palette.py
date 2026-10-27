"""Fuzzy command palette: rank registry commands for a query. Pure and cheap; used by the dock, the home screen
and `/help`'s "did you mean"."""

from __future__ import annotations

from dataclasses import dataclass

from agent.ui.commands import REGISTRY, Command, Registry


@dataclass(frozen=True)
class Match:
    command: Command
    score: int
    matched: str          # which name/alias/field matched ("" for an empty query)

    @property
    def name(self) -> str:
        return "/" + self.command.name + ("-<dir>" if self.command.pattern else "")

    @property
    def summary(self) -> str:
        return self.command.summary

    @property
    def insert(self) -> str:
        """Text to put into the input line for this command."""
        return "/" + self.command.name + ("-" if self.command.pattern else " ")


def _subseq_score(q: str, s: str) -> int:
    """0 when `q` is not a subsequence of `s`; else 1..30, higher = more compact and earlier."""
    pos, last, gaps = 0, -1, 0
    first = -1
    for ch in q:
        i = s.find(ch, pos)
        if i < 0:
            return 0
        if first < 0:
            first = i
        if last >= 0:
            gaps += i - last - 1
        last, pos = i, i + 1
    return max(1, 30 - gaps * 3 - first)


def score_command(cmd: Command, query: str) -> tuple[int, str]:
    q = query.strip().lstrip("/").lower()
    if not q:
        return 0, ""
    best, why = 0, ""

    def take(s: int, w: str) -> None:
        nonlocal best, why
        if s > best:
            best, why = s, w

    for n in (cmd.name, *cmd.aliases):
        nl = n.lower()
        if nl == q:
            take(100 if n == cmd.name else 95, n)
        elif nl.startswith(q):
            take(80 - min(10, len(nl) - len(q)), n)
        elif any(part.startswith(q) for part in nl.split("-")):
            take(70, n)
        elif q in nl:
            take(60, n)
        else:
            s = _subseq_score(q, nl)
            if s and len(q) >= 2:
                take(30 + s // 3, n)
    summ = cmd.summary.lower()
    words = summ.replace(",", " ").split()
    if len(q) >= 3:
        if any(w.startswith(q) for w in words):
            take(35, "summary")
        elif q in summ:
            take(25, "summary")
    if len(q) >= 3 and cmd.group.lower().startswith(q):
        take(20, cmd.group)
    return best, why


def pick_command(registry: Registry | None = None, query: str = "", limit: int | None = None) -> list[Match]:
    """Matches for `query`, best first (ties: shorter name, then alphabetical). Empty query = everything in
    registry order."""
    reg = registry or REGISTRY
    cmds = reg.commands()
    if not query.strip().lstrip("/"):
        out = [Match(c, 0, "") for c in cmds]
    else:
        scored = []
        for c in cmds:
            s, why = score_command(c, query)
            if s > 0:
                scored.append(Match(c, s, why))
        scored.sort(key=lambda m: (-m.score, len(m.command.name), m.command.name))
        out = scored
    return out[:limit] if limit else out
