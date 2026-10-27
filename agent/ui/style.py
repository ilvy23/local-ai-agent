"""Palette, layout classes, glyphs and small formatters. Independent of agent/theme.py."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import IO

from rich.console import Console

from agent import brand

WIDE_MIN, NORMAL_MIN = 100, 60


def plain_mode(env: dict[str, str] | None = None, stream: IO[str] | None = None) -> bool:
    """True when colour/unicode decoration should be off."""
    e = os.environ if env is None else env
    if e.get("NO_COLOR") or e.get("NIGHTAGE_PLAIN") or e.get("AGENT_PLAIN"):
        return True
    if e.get("TERM") == "dumb":
        return True
    s = stream if stream is not None else sys.stdout
    enc = (getattr(s, "encoding", None) or "").lower()
    if sys.platform == "win32" and enc and "utf" not in enc:      # legacy code page (cp1252/cp437): box glyphs would crash or turn to '?'
        return True
    try:
        return not s.isatty()
    except (AttributeError, ValueError):
        return True


def layout(width: int) -> str:
    return "wide" if width >= WIDE_MIN else "normal" if width >= NORMAL_MIN else "narrow"


@dataclass(frozen=True)
class Layout:
    width: int

    @property
    def name(self) -> str:
        return layout(self.width)

    @property
    def wide(self) -> bool:
        return self.name == "wide"

    @property
    def narrow(self) -> bool:
        return self.name == "narrow"


@dataclass(frozen=True)
class Glyphs:
    bullet: str
    arrow: str
    check: str
    cross: str
    ellipsis: str
    fill: str
    empty: str
    expand: str
    collapse: str
    rule: str = "─"                                  # horizontal divider under headings


UNICODE = Glyphs("•", "→", "✓", "✗", "…", "█", "░", "▸", "▾")
ASCII = Glyphs("*", "->", "+", "x", "...", "#", "-", ">", "v", "-")


def glyphs(plain: bool | None = None) -> Glyphs:
    return ASCII if (plain_mode() if plain is None else plain) else UNICODE


def bar(pct: float, width: int = 10, g: Glyphs | None = None) -> str:
    g = g or UNICODE
    width = max(1, width)
    pct = min(100.0, max(0.0, float(pct)))
    n = round(pct / 100 * width)
    return g.fill * n + g.empty * (width - n)


def trunc(text: str, width: int, g: Glyphs | None = None) -> str:
    g = g or UNICODE
    text = " ".join(str(text).split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    e = g.ellipsis
    return text[: max(0, width - len(e))] + e if width > len(e) else text[:width]


def fmt_tokens(n: float) -> str:
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        v = n / 1000
        return f"{v:.1f}k" if v < 100 else f"{round(v)}k"
    return f"{n / 1_000_000:.1f}M"


def fmt_duration(s: float) -> str:
    s = max(0.0, float(s))
    if s < 10:
        return f"{s:.1f}s"
    if s < 60:
        return f"{int(s)}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


_RISK = {"safe": "green", "caution": "yellow", "dangerous": "red", "blocked": "bold red"}


def risk_style(tier: object) -> str:
    """rich style for a risk tier (name string or agent.safety.RiskLevel)."""
    name = getattr(tier, "name", tier)
    return _RISK.get(str(name).lower(), "dim")


def _mix(hexcol: str, other: tuple[int, int, int], t: float) -> str:
    h = hexcol.lstrip("#")
    if len(h) != 6:
        return hexcol
    rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    out = tuple(round(a + (b - a) * t) for a, b in zip(rgb, other))
    return "#%02x%02x%02x" % out


@dataclass(frozen=True)
class Palette:
    accent: str
    dim: str
    ok: str = "green"
    warn: str = "yellow"
    error: str = "red"
    text: str = "default"

    @classmethod
    def from_accent(cls, accent: str) -> "Palette":
        return cls(accent=accent, dim=_mix(accent, (128, 128, 128), 0.6) if accent.startswith("#") else "dim")


def palette() -> Palette:
    return Palette.from_accent(brand.ACCENT)


def make_console(width: int | None = None, plain: bool | None = None, file: IO[str] | None = None) -> Console:
    p = plain_mode(stream=file) if plain is None else plain
    kw: dict = {"width": width, "file": file, "highlight": False}
    if p:
        kw.update(no_color=True, color_system=None, emoji=False, legacy_windows=False)
    return Console(**kw)
