"""Text checks for check scripts: sentence-level duplicates, per-group overlap, banned phrases.

Why sentences and not lines (D4): the author's own test compared lines and reported "all 4 tiers unique" while 35
sentences were shared - the same sentence wrapped differently is two different lines. Everything here collapses
whitespace first and compares normalised sentences, the way the supervisor's independent check did.

Standalone on purpose (standard library only), so a check script can use it with any interpreter:

    from agent.codeagent import textcheck as tc          # or copy this file next to the script
    units = {"open:steady": lena.LENA_OPEN["steady"], **{f"node:{k}": n["text"] for k, n in lena.LENA_TREE.items()}}
    fails = tc.duplicate_problems(units) + tc.banned_problems("\\n".join(units.values()), [r"\\bhollow\\b", "for a while"])
    tc.finish("LENA CHECK", fails)                        # prints PASS / FAIL with examples, exits 0 / 1

Also a tiny CLI:  python -m agent.codeagent.textcheck dupes FILE...   |   banned PATTERN_FILE FILE..."""

from __future__ import annotations

import collections
import itertools
import re
import sys
from pathlib import Path
from typing import Any, Iterable

MIN_WORDS = 3


def flat(x: Any) -> list[str]:
    """Every string inside nested dicts / lists / tuples (dict values only), in order."""
    if isinstance(x, dict):
        return [t for v in x.values() for t in flat(v)]
    if isinstance(x, (list, tuple, set)):
        return [t for v in x for t in flat(v)]
    return [x] if isinstance(x, str) else []


def sentences(text: str) -> list[str]:
    """Sentences of `text` with all whitespace collapsed first (hard wraps do not matter). Quotes end a sentence too, so
    `He said "Sit down." and left.` gives the spoken part on its own."""
    text = re.sub(r"\s+", " ", str(text or ""))
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+|[“”\"]", text) if s.strip()]


def norm(sentence: str) -> str:
    """Lower case letters, digits and single spaces only: punctuation and case never make two sentences 'different'."""
    return " ".join(re.sub(r"[^\w ]+", " ", sentence.lower()).split())


def sentence_set(text: str, min_words: int = MIN_WORDS) -> set[str]:
    return {n for n in (norm(s) for s in sentences(text)) if len(n.split()) >= min_words}


def duplicates(units: dict[str, str], min_words: int = MIN_WORDS) -> dict[str, set[str]]:
    """Normalised sentence -> names of the units (texts) it appears in, for sentences in 2+ units."""
    seen: dict[str, set[str]] = collections.defaultdict(set)
    for name, text in units.items():
        for s in sentence_set(text, min_words):
            seen[s].add(name)
    return {s: names for s, names in seen.items() if len(names) > 1}


def repeats_inside(units: dict[str, str], min_words: int = MIN_WORDS) -> list[tuple[str, str, int]]:
    """(unit, sentence, count) for a sentence that occurs more than once inside ONE text."""
    out = []
    for name, text in units.items():
        counts = collections.Counter(n for n in (norm(s) for s in sentences(text)) if len(n.split()) >= min_words)
        out += [(name, s, c) for s, c in counts.items() if c > 1]
    return out


def overlap(groups: dict[str, str], min_words: int = MIN_WORDS) -> list[tuple[str, str, set[str]]]:
    """(a, b, shared sentences) for every pair of groups that share at least one sentence (e.g. the four mood tiers)."""
    sets = {k: sentence_set(v, min_words) for k, v in groups.items()}
    return [(a, b, sets[a] & sets[b]) for a, b in itertools.combinations(sets, 2) if sets[a] & sets[b]]


def banned_hits(text: str, patterns: Iterable[str], regex: bool = True) -> list[tuple[str, int]]:
    """(pattern, count) for every pattern found in `text` (case-insensitive; plain phrases when regex=False)."""
    low = str(text or "")
    out = []
    for pat in patterns:
        rx = re.compile(pat if regex else re.escape(pat), re.I)
        n = len(rx.findall(low))
        if n:
            out.append((pat, n))
    return out


def duplicate_problems(units: dict[str, str], min_words: int = MIN_WORDS, limit: int = 12) -> list[str]:
    fails = [f"repeated inside {u}: '{s[:70]}' x{c}" for u, s, c in repeats_inside(units, min_words)]
    dups = sorted(duplicates(units, min_words).items(), key=lambda kv: -len(kv[1]))
    fails += [f"sentence shared by {len(n)} texts ({', '.join(sorted(n)[:4])}): '{s[:70]}'" for s, n in dups[:limit]]
    if len(dups) > limit:
        fails.append(f"... and {len(dups) - limit} more shared sentences")
    return fails


def overlap_problems(groups: dict[str, str], min_words: int = MIN_WORDS) -> list[str]:
    return [f"{a} / {b} share {len(sh)} sentence(s), e.g. '{sorted(sh)[0][:70]}'" for a, b, sh in overlap(groups, min_words)]


def banned_problems(text: str, patterns: Iterable[str], regex: bool = True) -> list[str]:
    return [f"banned phrase /{p}/ occurs {n}x" for p, n in banned_hits(text, patterns, regex)]


def finish(title: str, fails: list[str], limit: int = 40) -> None:
    """Print `TITLE: PASS` or `TITLE: FAIL (n problems)` plus the first problems, and exit 0 / 1."""
    print(f"{title}:", "PASS" if not fails else f"FAIL ({len(fails)} problems)")
    for f in fails[:limit]:
        print(" -", f)
    sys.exit(1 if fails else 0)


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "dupes":
        units = {f: Path(f).read_text(encoding="utf-8", errors="replace") for f in argv[1:]}
        finish("DUPLICATES", duplicate_problems(units))
    if len(argv) >= 3 and argv[0] == "banned":
        pats = [ln.strip() for ln in Path(argv[1]).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        text = "\n".join(Path(f).read_text(encoding="utf-8", errors="replace") for f in argv[2:])
        finish("BANNED", banned_problems(text, pats))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
