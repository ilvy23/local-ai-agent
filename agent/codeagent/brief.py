"""The brief card: the user's requirements, quoted VERBATIM from the request, pinned for the whole task.

Seen live (D5, B1): the user ordered "a cheerful baker, not gloomy, a breath of fresh air, doesn't fit the story,
all texts different"; after a few compactions the work log said "rewrite the baker's dialogue" and the model
delivered a gloomy character - and called it done. Summaries keep the task and lose the qualities.

So, at the start of a big task, the harness (no model call: zero tokens, and the words stay the user's) splits the
request into requirement units - bullets, then sentences - and keeps them word for word, numbered. The card is a
system message the context manager never folds, compacts or masks (context.py keeps `pinned` messages as they are),
it is saved with the session, re-shown with every check failure at the final gate and given to the reviewer.
When the request is too long for the card, sentences with requirement words (must, never, every, only, different,
ALL-CAPS emphasis, quoted phrases ...) win; the request itself stays in the head verbatim anyway."""

from __future__ import annotations

import re

BRIEF_MARKER = "BRIEF CARD"
BRIEF_MAX_CHARS = 3500             # ~1.1k tokens: the card rides in every prompt of the task
BRIEF_MIN_TASK_CHARS = 200         # shorter requests stay visible verbatim in the head anyway
MAX_CARDS = 3                      # a session with several big tasks keeps the newest cards
MIN_UNIT_CHARS = 12

_MARKERS = re.compile(
    r"(?i)\b(must|mustn't|should|shall|never|always|only|every|each|all|any|no|not|don't|do not|doesn't|does not|without|keep|make sure|"
    r"ensure|need|needs|has to|have to|required?|exactly|different|same|unique|instead|avoid|forbidden|allowed|"
    r"muss|müssen|soll|sollen|nie|niemals|immer|nur|jede[rsn]?|alle|kein\w*|nicht|ohne|genau|anders|verschieden\w*)\b")
_CAPS = re.compile(r"\b[A-ZÄÖÜ]{4,}\b")
_QUOTE = re.compile(r"[\"“”„«»]|(?<!\w)'(?=\w)")


def _units(text: str) -> list[str]:
    """Bullets and sentences of the request, each with its whitespace collapsed (hard wraps vanish, words stay exact)."""
    units: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            units.extend(_sentences(" ".join(para)))
            para.clear()

    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if re.match(r"^(?:[-*•]|\d+[.)])\s+\S", line):
            flush()
            para.append(re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line))
            continue
        para.append(line)
    flush()
    return [u for u in (" ".join(x.split()) for x in units) if len(u) >= MIN_UNIT_CHARS]


def _sentences(text: str) -> list[str]:
    """Sentence split that never cuts inside a quotation ("I-it's not like I made this for you!" stays whole)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"“„(])", text)
    out: list[str] = []
    for part in parts:
        if out and (out[-1].count('"') % 2 or out[-1].count("“") > out[-1].count("”") or out[-1].count("(") > out[-1].count(")")):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _weight(unit: str) -> int:
    return len(_MARKERS.findall(unit)) + 2 * len(_CAPS.findall(unit)) + (2 if _QUOTE.search(unit) else 0)


def extract(text: str, max_chars: int = BRIEF_MAX_CHARS) -> list[str]:
    """The requirement units to pin, in the order the user wrote them. All of them when they fit; else the ones with
    requirement words and emphasis first (each still verbatim), the rest while room is left."""
    units = _units(text)
    if sum(len(u) + 8 for u in units) <= max_chars:
        return units
    ranked = sorted(range(len(units)), key=lambda i: (-_weight(units[i]), i))
    keep: set[int] = set()
    used = 0
    for i in ranked:
        cost = len(units[i]) + 8
        if used + cost <= max_chars:
            keep.add(i)
            used += cost
    return [units[i] for i in sorted(keep)]


CLARIFY_HEADER = "Clarifications from the user:"       # intake.py appends this block to a task whose questions the user answered


def wanted(text: str) -> bool:
    """Is this request a task that gets a card? Big enough (or carrying the user's answers from the intake), not a slash command, not a
    supervisor directive."""
    t = str(text or "").strip()
    big = len(t) >= BRIEF_MIN_TASK_CHARS or ("\n" + CLARIFY_HEADER + "\n") in t
    return big and not t.startswith(("/", "AUTOPILOT", "[harness]")) and len(_units(t)) >= 2


def render(items: list[str]) -> str:
    lines = [f'{n}. "{u}"' for n, u in enumerate(items, 1)]
    return (f"{BRIEF_MARKER} (the user's requirements, quoted verbatim from the request - pinned by the harness, it is never summarised; "
            "every item must hold in the finished work, qualities and tone included):\n" + "\n".join(lines))


def is_card(message: dict) -> bool:
    return message.get("role") == "system" and str(message.get("content", "")).startswith(BRIEF_MARKER)


def make(text: str) -> str | None:
    """The card text for a request, or None when it gets none."""
    if not wanted(text):
        return None
    items = extract(text)
    return render(items) if len(items) >= 2 else None
