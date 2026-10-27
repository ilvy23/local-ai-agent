"""The big NIGHTAGE heading for the home screen: a static block-letter wordmark, no animation.

Wide enough terminals get the 5-row block heading; narrower ones a letter-spaced one-liner, so the
screen never wraps. Plain/ASCII mode draws the blocks with '#'. The word comes from agent/brand.py,
so a rename only needs letters this font knows (see GLYPHS); unknown letters fall back to the one-liner."""

from __future__ import annotations

from agent import brand

BLOCK, GAP = "█", 2

GLYPHS: dict[str, tuple[str, ...]] = {
    "A": (" ███ ", "█   █", "█████", "█   █", "█   █"),
    "B": ("████ ", "█   █", "████ ", "█   █", "████ "),
    "C": (" ████", "█    ", "█    ", "█    ", " ████"),
    "D": ("████ ", "█   █", "█   █", "█   █", "████ "),
    "E": ("█████", "█    ", "████ ", "█    ", "█████"),
    "F": ("█████", "█    ", "████ ", "█    ", "█    "),
    "G": (" ███ ", "█    ", "█  ██", "█   █", " ███ "),
    "H": ("█   █", "█   █", "█████", "█   █", "█   █"),
    "I": ("███", " █ ", " █ ", " █ ", "███"),
    "L": ("█    ", "█    ", "█    ", "█    ", "█████"),
    "N": ("█   █", "██  █", "█ █ █", "█  ██", "█   █"),
    "O": (" ███ ", "█   █", "█   █", "█   █", " ███ "),
    "R": ("████ ", "█   █", "████ ", "█  █ ", "█   █"),
    "S": (" ████", "█    ", " ███ ", "    █", "████ "),
    "T": ("█████", "  █  ", "  █  ", "  █  ", "  █  "),
    "U": ("█   █", "█   █", "█   █", "█   █", " ███ "),
    "V": ("█   █", "█   █", "█   █", " █ █ ", "  █  "),
    "W": ("█   █", "█   █", "█ █ █", "██ ██", "█   █"),
    "Y": ("█   █", " █ █ ", "  █  ", "  █  ", "  █  "),
}


def block_lines(word: str | None = None, plain: bool = False) -> list[str] | None:
    """The 5 rows of the block heading, or None when a letter has no glyph."""
    word = (word or brand.NAME).upper()
    if any(ch not in GLYPHS for ch in word):
        return None
    rows = [(" " * GAP).join(GLYPHS[ch][r] for ch in word) for r in range(5)]
    return [r.replace(BLOCK, "#") for r in rows] if plain else rows


def banner_width(word: str | None = None) -> int:
    rows = block_lines(word)
    return len(rows[0]) if rows else 0


def spaced(word: str | None = None) -> str:
    """Narrow fallback: 'N I G H T A G E'."""
    return " ".join((word or brand.NAME).upper())


def heading(width: int, plain: bool = False) -> list[str]:
    """Lines to print above the status block for a terminal `width` columns wide."""
    rows = block_lines(plain=plain)
    if rows and width >= len(rows[0]) + 4:
        return [*rows, ""]
    return [spaced()]
