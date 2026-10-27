import io

import pytest

from agent.ui import help as H
from agent.ui import style
from agent.ui.commands import REGISTRY


def render(topic=None, width=100, plain=True):
    buf = io.StringIO()
    con = style.make_console(width=width, plain=plain, file=buf)
    ok = H.render_help(con, REGISTRY, topic, plain=plain, width=width)
    return ok, buf.getvalue()


@pytest.mark.parametrize("width", [40, 60, 100])
def test_every_command_appears_exactly_once(width):
    _, out = render(width=width)
    for c in REGISTRY.commands():
        label = "/" + c.name + ("-<dir>" if c.pattern else "")
        lines = [ln for ln in out.splitlines() if ln.strip().startswith(label + " ") or ln.strip() == label]
        assert len(lines) == 1, (label, lines)


def test_groups_cover_registry_and_have_headings():
    groups = H.help_groups(REGISTRY)
    flat = [c.name for _, cs in groups for c in cs]
    assert sorted(flat) == sorted(c.name for c in REGISTRY.commands())
    assert len(flat) == len(set(flat))
    _, out = render()
    for g, _ in groups:
        assert g in out


@pytest.mark.parametrize("width", [40, 60, 100])
def test_lines_fit_width(width):
    _, out = render(width=width)
    assert max(len(ln) for ln in out.splitlines()) <= width


def test_detail_view_has_usage_aliases_examples_and_timing():
    ok, out = render("queue")
    assert ok and "/queue [rm N | clear | now N]" in out and "/queue rm 2" in out and "immediately" in out
    ok, out = render("/ctx")           # alias, with slash
    assert ok and "Aliases" not in out.split("Usage")[0] and "context" in out
    ok, out = render("undo")
    assert "'list'" in out


def test_unknown_topic_suggests():
    ok, out = render("statu")          # prefix of status -> still resolved? no: unknown, suggestion
    assert not ok and "/status" in out
    ok, out = render("qzx")
    assert not ok and "No command" in out


def test_all_commands_have_examples_or_usage():
    for c in REGISTRY.commands():
        ok, out = render(c.name)
        assert ok and ("Usage" in out)


def test_keys_overlay_reuses_dock_keys():
    from agent.ui.dock import KEYS_HELP
    txt = H.keys_help_text(100)
    for k, what in KEYS_HELP:
        assert k in txt and what in txt
    narrow = H.keys_help_text(40)
    assert max(len(ln) for ln in narrow.splitlines()) <= 40


def test_plain_mode_is_ascii():
    _, out = render(plain=True)
    assert out.isascii()
