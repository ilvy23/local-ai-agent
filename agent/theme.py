"""Neon palette shared by the Rich-based screens that still use it (`agent settings`, the prompt studio).

The logo, skyline and boot animation of the old menu are gone; the new terminal UI takes its colours from
`agent.ui.style`. Only the constants and `blend` are kept here so the older screens keep their look.
"""

from __future__ import annotations

CYAN = "#00e5ff"      # borders, headings
MAGENTA = "#ff2bd6"   # prompts, accents
YELLOW = "#f8f32b"    # numbers, highlights
GREEN = "#39ff14"     # ok / success
PURPLE = "#9d4dff"    # secondary accent
TEXT = "#c9d6ff"      # body text, labels
DIM = "#5b6a8f"       # hints
BG = "#07060d"        # background


def blend(a: str, b: str, t: float) -> str:
    """Interpolate two #rrggbb colours (t=0 -> a, t=1 -> b)."""
    t = max(0.0, min(1.0, t))
    ar = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    br = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(ar[i] + (br[i] - ar[i]) * t):02x}" for i in range(3))
