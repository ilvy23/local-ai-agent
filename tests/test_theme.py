"""The small palette module the Rich-based screens (settings, prompt studio) still import."""
from agent import theme as T


def test_palette_constants_are_hex_colours():
    for name in ("CYAN", "MAGENTA", "YELLOW", "GREEN", "PURPLE", "TEXT", "DIM", "BG"):
        v = getattr(T, name)
        assert len(v) == 7 and v[0] == "#" and int(v[1:], 16) >= 0


def test_blend_endpoints_and_clamping():
    assert T.blend("#000000", "#ffffff", 0) == "#000000"
    assert T.blend("#000000", "#ffffff", 1) == "#ffffff"
    assert T.blend("#000000", "#ffffff", 0.5) == "#7f7f7f"
    assert T.blend("#000000", "#ffffff", 9) == "#ffffff"
