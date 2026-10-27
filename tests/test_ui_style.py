import io

from agent import brand
from agent.ui import style


def test_plain_mode():
    class Tty(io.StringIO):
        def isatty(self):
            return True
    t = Tty()
    assert style.plain_mode({}, t) is False
    assert style.plain_mode({"NO_COLOR": "1"}, t)
    assert style.plain_mode({"NIGHTAGE_PLAIN": "1"}, t)
    assert style.plain_mode({"AGENT_PLAIN": "1"}, t)
    assert style.plain_mode({"TERM": "dumb"}, t)
    assert style.plain_mode({}, io.StringIO())      # not a tty


def test_layout_classes():
    assert style.Layout(40).name == "narrow" and style.Layout(40).narrow
    assert style.Layout(59).name == "narrow"
    assert style.Layout(60).name == "normal" and style.Layout(99).name == "normal"
    assert style.Layout(100).name == "wide" and style.Layout(100).wide


def test_glyph_sets():
    assert style.glyphs(True) is style.ASCII and style.glyphs(False) is style.UNICODE
    for f in style.ASCII.__dict__.values():
        assert f.isascii()
    assert not style.UNICODE.bullet.isascii()


def test_bar_trunc_fmt():
    assert style.bar(50, 10, style.ASCII) == "#####-----"
    assert style.bar(-5, 4, style.ASCII) == "----" and style.bar(500, 4, style.ASCII) == "####"
    assert style.trunc("hello world", 8, style.ASCII) == "hello..."
    assert style.trunc("hello world", 8) == "hello w…"
    assert style.trunc("hi", 8) == "hi" and style.trunc("x", 0) == ""
    assert style.fmt_tokens(999) == "999" and style.fmt_tokens(1234) == "1.2k"
    assert style.fmt_tokens(250000) == "250k" and style.fmt_tokens(2_500_000) == "2.5M"
    assert style.fmt_duration(3.21) == "3.2s" and style.fmt_duration(42) == "42s"
    assert style.fmt_duration(125) == "2m 05s" and style.fmt_duration(3700) == "1h 01m"


def test_risk_style_and_palette():
    from agent.safety import RiskLevel
    assert style.risk_style("safe") == "green" and style.risk_style(RiskLevel.DANGEROUS) == "red"
    assert style.risk_style("???") == "dim"
    p = style.palette()
    assert p.accent == brand.ACCENT and p.dim.startswith("#") and p.dim != p.accent


def test_make_console():
    buf = io.StringIO()
    c = style.make_console(width=40, file=buf)
    c.print("[bold red]hi[/bold red]")
    assert c.width == 40 and buf.getvalue() == "hi\n"   # plain: no escape codes
    buf2 = io.StringIO()
    c2 = style.make_console(width=50, plain=False, file=buf2)
    c2.print("[bold]x[/bold]")
    assert "x" in buf2.getvalue()
