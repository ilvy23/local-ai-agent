"""Home topics (agent/ui/launcher.py): the capabilities of the old nested menu, now reachable from the home screen.

Absorbs the still-valid intent of the old tests/test_menu.py (placeholder filling, number-or-typed picker) and of the
WhatsApp split (private module absent => public entries still there, notice entry instead)."""

import io

from rich.console import Console

from agent.ui import launcher as L


class StubConsole:
    def __init__(self, answers=()):
        self.answers = list(answers)
        self.out: list[str] = []

    def input(self, _prompt=""):
        return self.answers.pop(0)

    def print(self, *a, **_k):
        self.out.append(" ".join(str(x) for x in a))


def _labels(sections):
    return {e.label for _, entries in sections for e in entries}


def test_fill_prompts_for_placeholder():
    assert L.fill(["memory", "search", "{query}"], StubConsole(["bees"])) == ["memory", "search", "bees"]


def test_fill_blank_placeholder_cancels():
    assert L.fill(["memory", "forget", "{id}"], StubConsole([""])) is None


def test_fill_passes_literals_through():
    assert L.fill(["worker", "start"], StubConsole()) == ["worker", "start"]


def test_pick_from_number_and_freetext():
    rows = [("First option", "alpha"), ("Second option", "beta")]
    assert L.pick_from(StubConsole(["2"]), rows, "choice") == "beta"
    assert L.pick_from(StubConsole(["typed"]), rows, "choice") == "typed"
    assert L.pick_from(StubConsole([""]), rows, "choice") is None


def test_every_topic_builds_and_has_entries():
    for key, _label, _hint in L.TOPICS:
        if key == "commands":
            continue
        title, sections = L.topic_sections(key)
        assert title and sections and all(entries for _, entries in sections)


def test_whatsapp_hides_private_entries_when_unavailable():
    labels = _labels(L.whatsapp_sections(private=False))
    assert "Chat analysis (not available in this build)" in labels
    for fragment in ("friendship", "Sum up a chat", "Talk to a clone", "Build a clone"):
        assert not any(fragment in label for label in labels)
    assert any("Write a reply" in label for label in labels)       # public entries stay
    assert any("auto-reply is doing" in label for label in labels)


def test_whatsapp_shows_private_entries_when_available():
    labels = _labels(L.whatsapp_sections(private=True))
    assert any("friendship" in label for label in labels)
    assert any("Talk to a clone" in label for label in labels)
    assert "Chat analysis (not available in this build)" not in labels


def test_unavailable_notice_runs_through_execute():
    c = StubConsole()
    entry = next(e for _, es in L.whatsapp_sections(private=False) for e in es if e.call)
    assert L.execute(entry, c) is True
    assert any("not available" in line for line in c.out)


def test_execute_confirm_declined_runs_nothing():
    ran = []
    entry = L.Entry("x", ["wa", "summarize", "--all"], confirm="sure?")
    c = StubConsole(["n"])
    assert L.execute(entry, c, app=lambda argv, **k: ran.append(argv)) is False
    assert ran == []


def test_execute_runs_argv_in_process():
    ran = []
    assert L.execute(L.Entry("x", ["memory", "list"]), StubConsole(), app=lambda argv, **k: ran.append(argv)) is True
    assert ran == [["memory", "list"]]


def test_all_commands_generated_from_the_typer_app():
    from agent.main import app

    secs = dict(L.all_commands_sections(app))
    top = {e.label: e for e in secs["Commands"]}
    for name in ("chat", "code", "doctor", "research", "panel", "demo"):
        assert name in top and top[name].hint
    assert any(title.startswith("wa") for title in secs) and any(title.startswith("memory") for title in secs)
    assert "menu" in top   # kept as an alias of home


def test_all_commands_entry_accepts_extra_arguments():
    ran = []
    secs = L.all_commands_sections()
    entry = next(e for e in secs[0][1] if e.label == "research")
    L.execute(entry, StubConsole(['"why is the sky blue"']), app=lambda argv, **k: ran.append(argv))
    assert ran == [["research", "why is the sky blue"]]


def test_run_topic_loop_back_and_invalid():
    c = StubConsole(["abc", "99", "q"])
    L.run_topic("memory", c)
    assert sum("not a valid number" in line for line in c.out) == 2


def test_run_topic_runs_chosen_entry():
    ran = []
    L.run_topic("status", StubConsole(["1", ""]), app=lambda argv, **k: ran.append(argv))
    assert ran == [["panel"]]


def test_whatsapp_installed_follows_plugin_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path))
    assert L.whatsapp_installed() is False
    (tmp_path / "whatsapp").mkdir()
    (tmp_path / "whatsapp" / "__init__.py").write_text("")
    assert L.whatsapp_installed() is True
