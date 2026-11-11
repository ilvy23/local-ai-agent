"""Templates, the prompt splitter and the dry-run plan of `nightage night`."""
from __future__ import annotations

from agent.night import plan, preflight, queue, splitter, templates


def test_init_writes_rules_and_templates_that_pass_their_own_preflight_and_never_overwrites(tmp_path):
    files = templates.init(tmp_path / "p")
    names = sorted(f.name for f in files)
    assert names == ["01-feature.md", "02-bug.md", "03-coverage.md", "04-refactor.md", "05-docs.md", "rules.md"]
    (tmp_path / "p" / "02-bug.md").write_text("mine")
    assert templates.init(tmp_path / "p") == [] and (tmp_path / "p" / "02-bug.md").read_text() == "mine"
    only = templates.init(tmp_path / "q", ["bug"])
    assert sorted(f.name for f in only) == ["02-bug.md", "rules.md"]
    (tmp_path / "p" / "02-bug.md").unlink()
    templates.init(tmp_path / "p", ["bug"])
    items, rules = queue.load([str(tmp_path / "p")])
    assert [i.name for i in items][:1] == ["feature"] and "Never delete, skip or weaken a test" in rules and items[1].kind == "bug" and items[0].weight == 2.0
    assert not [w for w in preflight.check(items, tmp_path) if "nothing in it can be checked" in w]       # every template carries something checkable


BIG = """Context for all parts: a terminal game in Python.

Stage 1: Map the project
Read the code and write docs/MAP.md. Tests must still pass.

Stage 2: One new character
Add a character and a test; it must load.

Stage 3: Dialogue
Make lines depend on inventory.
"""


def test_split_by_stage_headers_with_shared_preamble():
    pre, parts = splitter.split(BIG)
    assert pre.startswith("Context for all parts") and [t for t, _ in parts] == ["Stage 1: Map the project", "Stage 2: One new character", "Stage 3: Dialogue"]
    assert len(parts) == 3 and parts[1][1].startswith("Stage 2: One new character") and "Add a character" in parts[1][1]


def test_split_falls_back_to_markdown_headings_and_refuses_a_single_part():
    pre, parts = splitter.split("intro\n\n## Parser\nwrite it\n\n## Printer\nwrite that\n")
    assert [t for t, _ in parts] == ["Parser", "Printer"] and pre == "intro"
    assert splitter.split("just one paragraph, no headers at all") == ("just one paragraph, no headers at all", [])
    assert splitter.split("Stage 1: only one\nbody") [1] == []


def test_write_chains_the_parts_and_the_queue_reads_them_back(tmp_path):
    files = splitter.write(BIG, tmp_path / "out")
    assert [f.name for f in files] == ["01-stage-1-map-the-project.md", "02-stage-2-one-new-character.md", "03-stage-3-dialogue.md"]
    items, _ = queue.load([str(tmp_path / "out")])
    assert items[0].depends == [] and items[1].depends == [items[0].name] and items[2].depends == [items[1].name]
    assert all(i.text.startswith("Context for all parts") for i in items)                       # the preamble is in every part
    loose = splitter.write(BIG, tmp_path / "loose", chain=False)
    assert all("depends" not in f.read_text() for f in loose) and splitter.write("no parts", tmp_path / "none") == []


def test_plan_shows_slices_weights_deliverables_and_warnings(tmp_path):
    items = [queue.Item("a", "Create `new_a.py` and make the tests pass, it must work.", weight=3.0), queue.Item("b", "tiny", depends=["a"], verify="make check")]
    s = dict(plan.slices(items, 8))
    assert s["a"] > s["b"] and abs(sum(s.values()) - (8 * 60 - 15)) < 1 or s["b"] >= 20                 # weights respected, never below the minimum slice
    text = plan.render(items, tmp_path, 8, "allow  pytest *", "tests: `pytest -q` takes 3 s", "a game is running (proton)")
    assert "1. a" in text and "weight 3" in text and "must create: new_a.py" in text and "after a" in text and "verify `make check`" in text
    assert "too vague" in text and "Charter:" in text and "takes 3 s" in text and "Machine: a game is running" in text
    assert plan.slices([], 8) == []
