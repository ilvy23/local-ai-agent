import io
from agent.ui.home import VERSION
import time

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent.ui import doctor as D
from agent.ui import home as H
from agent.ui import style

CFG = {"models": {"chat": "qwen3.5:9b", "coding": "qwen3-coder:30b", "embed": "bge-m3"}}


class G:
    name, total_mb = "RX 7900", 24576


class FakeProbes(D.Probes):
    def __init__(self, ollama=None, session=None, gpus=()):
        super().__init__()
        self._o, self._s, self._g = ollama or D.OllamaInfo(), session, list(gpus)

    def ollama(self): return self._o
    def gpus(self): return self._g
    def gpu_mode(self, gpus): return "single" if gpus else None
    def ctx_budget(self, model, size, gpus, config): return {"override": None, "final": 32768, "source": "vram"}
    def last_session(self, config): return self._s
    def data_paths(self, config):
        from pathlib import Path
        return Path("/data"), Path("/data/agent.db")


UP = D.OllamaInfo(True, "0.6.1", {"qwen3.5:9b": 1, "bge-m3:latest": 1})
SESSION = {"id": 7, "title": "Refactor the parser", "last_active_at": "2026-10-26T10:00:00+00:00"}


def screen(probes, width, plain):
    buf = io.StringIO()
    con = style.make_console(width=width, plain=plain, file=buf)
    st = H.collect_status(CFG, probes)
    H.render_home(con, st, plain=plain, width=width)
    return buf.getvalue(), st


@pytest.mark.parametrize("width", [40, 60, 100])
@pytest.mark.parametrize("plain", [True, False])
def test_home_renders_status_and_fits(width, plain):
    out, st = screen(FakeProbes(UP, SESSION, [G()]), width, plain)
    from agent.ui import banner

    assert banner.heading(width, plain)[0] in out and f"v{VERSION}" in out          # block heading when wide enough, spaced word when narrow
    assert "Ollama" in out and "qwen3.5:9b" in out and "RX 7900" in out and "Refactor the parser" in out
    assert "ollama pull qwen3-coder:30b" in out or width < 60 or "missing" in out
    assert max(len(l) for l in out.splitlines()) <= width
    if plain:
        assert out.isascii()


def test_heading_is_a_block_wordmark_when_wide_and_a_spaced_word_when_narrow():
    from agent.ui import banner

    wide = banner.heading(100)
    assert len(wide) == 6 and all(len(r) <= 56 for r in wide) and "█" in wide[0]
    assert banner.heading(100, plain=True)[0].isascii() and "#" in banner.heading(100, plain=True)[0]
    assert banner.heading(40) == ["N I G H T A G E"]
    assert banner.heading(banner.banner_width() + 3) == ["N I G H T A G E"]     # never wider than the terminal


def test_no_ollama_state_shows_fix_and_is_fast():
    t = time.time()
    out, st = screen(FakeProbes(), 80, True)
    assert "ollama serve" in out and "not running" in out
    assert time.time() - t < 2
    assert st.chat_present is None
    assert "no sessions yet" in out


def test_probe_exceptions_never_escape():
    class Bad(FakeProbes):
        def ollama(self): raise RuntimeError("x")
        def gpus(self): raise RuntimeError("x")
        def ctx_budget(self, *a): raise RuntimeError("x")
        def last_session(self, c): raise RuntimeError("x")
    st = H.collect_status(CFG, Bad())
    assert not st.ollama.reachable and st.last_session is None


def test_items_continue_only_when_session_exists():
    keys = [i.key for i in H.build_items(H.collect_status(CFG, FakeProbes(UP, SESSION)))]
    assert keys[:3] == ["continue", "chat", "code"] and keys[-4:] == ["settings", "doctor", "help", "quit"]
    assert {"research", "memory", "apps", "status", "codetools", "commands"} <= set(keys)
    keys = [i.key for i in H.build_items(H.collect_status(CFG, FakeProbes(UP)))]
    assert keys[0] == "chat" and "continue" not in keys


def test_filtering():
    items = H.build_items(H.collect_status(CFG, FakeProbes(UP, SESSION)))
    assert [i.key for i in H.filter_items(items, "")] == [i.key for i in items]
    assert [i.key for i in H.filter_items(items, "3")] == ["code"]
    assert H.filter_items(items, "99") == []
    assert H.filter_items(items, "res")[0].key == "research"
    assert H.filter_items(items, "doc")[0].key == "doctor"
    assert H.filter_items(items, "zzz") == []


@pytest.mark.parametrize("width", [40, 60, 100])
@pytest.mark.parametrize("plain", [True, False])
def test_action_rows_fit_and_mark_selection(width, plain):
    items = H.build_items(H.collect_status(CFG, FakeProbes(UP, SESSION)))
    rows = H.action_rows(items, items, 1, width, plain)
    assert len(rows) == len(items)
    assert all(len(t) <= width - 1 for _, t in rows)
    assert rows[1][0] == "class:sel" and rows[1][1].lstrip().startswith((">", "›"))
    if plain:
        assert all(t.isascii() for _, t in rows)


def test_footer_adapts():
    assert len(H.footer_text(100)) > len(H.footer_text(45)) > len(H.footer_text(30))
    assert all(len(H.footer_text(w)) <= w for w in (30, 45, 100))


def test_age_text():
    from datetime import datetime, timezone
    now = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    assert H.age_text("2026-10-26T11:59:30+00:00", now) == "just now"
    assert H.age_text("2026-10-26T11:30:00+00:00", now) == "30 min ago"
    assert H.age_text("2026-10-26T07:00:00+00:00", now) == "5 h ago"
    assert H.age_text("2026-10-20T12:00:00", now) == "6 days ago"
    assert H.age_text("garbage", now) == ""


@pytest.fixture(autouse=True)
def _no_whatsapp_plugin(monkeypatch):
    monkeypatch.setattr("agent.ui.launcher.whatsapp_installed", lambda: False)


def drive(keys, probes=None, plain=True, width=80):
    """Run run_home against pipe input in a thread so a hang fails the test instead of freezing the run."""
    import threading
    probes = probes or FakeProbes(UP, SESSION, [G()])
    box = {}

    def target():
        with create_pipe_input() as inp:
            inp.send_text(keys)
            with create_app_session(input=inp, output=DummyOutput()):
                box["r"] = H.run_home(CFG, input=inp, output=DummyOutput(), plain=plain, probes=probes, width=width,
                                      console=style.make_console(width=width, plain=plain, file=io.StringIO()))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(15)
    assert "r" in box, f"run_home did not return for input {keys!r}"
    return box["r"]


def test_enter_picks_first_item():
    assert drive("\r").action == "continue"
    assert drive("\r").args == {"session_id": 7}


def test_arrow_keys_move_selection():
    assert drive("\x1b[B\r").action == "chat"          # down
    assert drive("\x1b[B\x1b[B\r").action == "code"
    assert drive("\x1b[B\x1b[B\x1b[A\r").action == "chat"
    assert drive("\x1b[A\r").action == "continue"      # clamps at top
    assert drive("\x1b[B" * 20 + "\r").action == "quit"  # clamps at bottom


def _pos(key):
    """1-based number of a home item with a session present (derived, so new items never break the tests)."""
    keys = [i.key for i in H.build_items(H.collect_status(CFG, FakeProbes(UP, SESSION)))]
    return keys.index(key) + 1


def test_number_and_filter_selection():
    assert drive("2\r").action == "chat"
    assert drive(f"{_pos('doctor')}\r").action == "doctor"
    assert drive("sett\r").action == "settings"
    assert drive("99\r\x7f\x7f\r").action == "continue"      # "9": no match, Enter is ignored; Backspace clears it


def test_code_returns_cwd_and_research_asks_question():
    c = drive("3\r")
    import os
    assert c.action == "code" and c.args["cwd"] == os.getcwd()
    r = drive("4\rhow do bees fly\r")
    assert r.action == "research" and r.args["question"] == "how do bees fly"
    assert drive("4\r\r").args["question"] == ""


def test_quit_keys():
    assert drive("\x04").action == "quit"
    assert drive("\x03").action == "quit"
    assert drive(f"{_pos('quit')}\r").action == "quit"


def test_slash_opens_command_palette():
    c = drive("/stat\r")
    assert c.action == "chat" and c.args == {"command": "/status"}
    c = drive("/\x1b[B\r")
    assert c.action == "chat" and c.args["command"].startswith("/")


def test_question_mark_toggles_keys_only_on_empty_line():
    assert drive("?\r").action == "continue"            # keys overlay shown, Enter still picks
    assert drive("res?\x7f\x7f\x7f\x7f\r").action == "continue"   # '?' inside text is just a character


def test_works_without_ollama_and_in_ansi_mode():
    c = drive("\r", probes=FakeProbes(), plain=False)
    assert c.action == "chat"                            # no session -> first item is New chat


def test_non_tty_fallback_reads_a_line(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("doctor\n"))
    buf = io.StringIO()
    c = H.run_home(CFG, plain=True, probes=FakeProbes(UP), console=style.make_console(width=60, plain=True, file=buf))
    assert c.action == "doctor" and "Help" in buf.getvalue()
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert H.run_home(CFG, plain=True, probes=FakeProbes(UP), console=style.make_console(file=io.StringIO())).action == "quit"


def test_topic_items_open_launcher_topics(monkeypatch):
    assert drive("memory\r").action == "topic" and drive("memory\r").args == {"topic": "memory"}
    assert drive("all comm\r").args == {"topic": "commands"}


def test_whatsapp_item_only_when_plugin_installed(monkeypatch):
    st = H.collect_status(CFG, FakeProbes(UP))
    assert "whatsapp" not in [i.key for i in H.build_items(st)]
    monkeypatch.setattr("agent.ui.launcher.whatsapp_installed", lambda: True)
    assert "whatsapp" in [i.key for i in H.build_items(st)]
    assert drive("whatsapp\r").args == {"topic": "whatsapp"}
