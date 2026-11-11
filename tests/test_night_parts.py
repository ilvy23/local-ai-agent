"""The measuring parts of `nightage night`: test-output parsing, handoff, failure analysis, lessons, guards, health, prompt check, front matter, map.
Synthetic data only; git repos live in tmp_path."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent.night import analyzer, frontmatter, gitbox, guard, handoff, health, lessons, mapper, preflight, queue, testwatch
from agent.night.runner import Outcome
from agent.night.testwatch import TestState

PYTEST_RED = """\
FAILED tests/test_a.py::test_one - assert 1 == 2
FAILED tests/test_b.py::TestX::test_two - KeyError
E   assert 1 == 2
E   KeyError: 'x'
2 failed, 38 passed in 1.2s
"""


# ----------------------------------------------------------------------------- testwatch
def test_parse_pytest_failures_counts_and_digest():
    st = testwatch.parse(PYTEST_RED, 1)
    assert not st.ok and st.passed == 38 and st.failed == ["tests/test_a.py::test_one", "tests/test_b.py::TestX::test_two"]
    assert "2 failing" in st.digest and "assert 1 == 2" in st.digest and len(st.digest) <= 1400
    ok, tail = st                                                       # the older (ok, tail) unpacking still works
    assert ok is False and "2 failed" in tail


@pytest.mark.parametrize("out,names", [
    ("--- FAIL: TestFoo (0.00s)\nFAIL", ["TestFoo"]),
    ("test parser::bad ... FAILED\ntest parser::good ... ok", ["parser::bad"]),
    ("FAIL: test_x (pkg.tests.CaseA)\n", ["pkg.tests.CaseA.test_x"]),
    ("  ✕ adds numbers (3 ms)\n  ✓ fine", ["adds numbers"]),
])
def test_parse_other_runners(out, names):
    assert testwatch.parse(out, 1).failed == names


def test_regressions_fixed_and_score():
    before = TestState(ok=False, passed=10, failed=["a", "b"])
    after = TestState(ok=False, passed=11, failed=["b", "c"])
    assert testwatch.regressions(before, after) == ["c"] and testwatch.fixed(before, after) == ["a"]
    assert testwatch.regressions(None, after) == ["b", "c"]            # no baseline: everything that fails counts
    assert testwatch.score(TestState(ok=True, passed=5)) > testwatch.score(TestState(ok=False, passed=50))


def test_settle_flaky_trusts_a_green_rerun_and_names_the_flaky_test():
    first = TestState(ok=False, failed=["t1"])
    state, flaky = testwatch.settle_flaky(first, lambda: TestState(ok=True))
    assert state.ok and flaky == ["t1"]
    state, flaky = testwatch.settle_flaky(first, lambda: TestState(ok=False, failed=["t1"]))
    assert not state.ok and flaky == []                                 # a stable failure is a real failure
    assert testwatch.settle_flaky(TestState(ok=True), lambda: 1 / 0)[1] == []


# ----------------------------------------------------------------------------- handoff / analyzer / lessons
def _repo(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    gitbox.ensure_repo(tmp_path)
    return tmp_path


def _events(tmp_path, rows):
    f = tmp_path / "events.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    return f


def test_event_summary_counts_only_the_window_and_the_session(tmp_path):
    rows = [{"ts": 5, "session": "proj-1", "kind": "step"}, {"ts": 20, "session": "proj-1", "kind": "step"},
            {"ts": 21, "session": "proj-1", "kind": "tool", "name": "read_file", "ok": True}, {"ts": 22, "session": "proj-1", "kind": "tool", "name": "edit_file", "ok": True},
            {"ts": 23, "session": "proj-1", "kind": "loop"}, {"ts": 24, "session": "other-9", "kind": "loop"}]
    es = handoff.summarize_events(_events(tmp_path, rows), since=10, prefix="proj")
    assert (es.steps, es.reads, es.edits, es.loops) == (1, 1, 1, 1) and es.last_tools == ["read_file", "edit_file"]
    assert handoff.summarize_events(tmp_path / "missing.jsonl", 0).steps == 0


def test_handoff_states_facts_only(tmp_path):
    root = _repo(tmp_path)
    tag = gitbox.last_green(root)
    (root / "a.py").write_text("x = 1\ny = 2\n")
    (root / "new.py").write_text("z = 3\n")
    es = handoff.EventSummary(steps=30, tools={"read_file": 25, "edit_file": 2}, loops=2, edits=2, last_tools=["read_file"] * 3)
    h = handoff.build(root, tag, TestState(ok=True, passed=5), TestState(ok=False, passed=4, failed=["t_x"], digest="1 failing: t_x"), es, 3)
    assert "a.py (+1/-0)" in h.text and "new.py" in h.text and "BROKE" in h.text and "t_x" in h.text and "looped 2" in h.text and "do not redo it" in h.text
    empty = handoff.build(root, gitbox.checkpoint(root, "x"), None, TestState(ok=True), handoff.EventSummary(), 0)
    assert "changed NOTHING" in empty.text


def _v(code=3, why="", before=None, after=None, files=1, **ev):
    d = handoff.DiffStat([("a.py", 1, 0)] * files)
    return analyzer.classify(Outcome(code, why), before, after or TestState(ok=False, failed=["t"]), d, handoff.EventSummary(**ev), ev.pop("log", "") if False else "")


@pytest.mark.parametrize("kwargs,kind", [
    (dict(why="timeout"), "timeout"), (dict(why="stalled"), "stalled"), (dict(why="crash", code=-9), "crash"),
    (dict(code=0, after=TestState(ok=True)), "green"), (dict(files=0), "no_edits"), (dict(loops=3), "loop"),
    (dict(model_errors=4), "server_down"),
    (dict(before=TestState(ok=True), after=TestState(ok=False, failed=["t"])), "broke_tests"),
    (dict(before=None, after=TestState(ok=False, failed=["t"])), "tests_red"),
    (dict(code=3, after=TestState(ok=True)), "partial"),
])
def test_classify_kinds(kwargs, kind):
    assert _v(**kwargs).kind == kind


def test_next_strategy_escalates_on_repeats_and_stays_on_the_last_rung():
    assert analyzer.next_strategy("no_edits", []) == "small"
    assert analyzer.next_strategy("no_edits", ["no_edits"]) == "fresh"
    assert analyzer.next_strategy("no_edits", ["no_edits"] * 9) == "rollback"
    assert analyzer.next_strategy("no_edits", ["loop", "no_edits"]) == "fresh"      # only CONSECUTIVE repeats escalate
    assert analyzer.next_strategy("broke_tests", []) == "rollback"
    assert analyzer.next_strategy("???", []) == "resume"                              # unknown kinds behave like tests_red


def test_lessons_are_deduplicated_scoped_and_budgeted(tmp_path):
    ls = lessons.Lessons(tmp_path / "L.md")
    assert ls.add("no_edits", "", "p1") and not ls.add("no_edits", "", "p1") and not ls.add("nonsense", "", "p1")
    ls.add("loop", "read story.py x8", "p2")
    ls.add("timeout", "", "any")
    assert "read story.py" in ls.recent("p2") and "read story.py" not in ls.recent("p1")
    assert "[p1/no_edits]" in ls.recent("p1") and "[any/timeout]" in ls.recent("p1")
    assert len(ls.recent("p1", budget=60)) <= 140 and ls.recent("zzz").count("\n") == 0


# ----------------------------------------------------------------------------- guard
def _git_commit_all(root, msg):
    return gitbox.checkpoint(root, msg)


def test_guard_reverts_weakened_tests_but_allows_new_ones(tmp_path):
    root = tmp_path
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n    assert 2\n")
    (root / "m.py").write_text("def f():\n    return 1\n")
    gitbox.ensure_repo(root)
    tag = gitbox.last_green(root)
    (root / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")                 # weakened: 2 -> 1 assertions
    (root / "tests" / "test_new.py").write_text("def test_new():\n    assert True\n")          # a new test is welcome
    rep = guard.check(root, tag)
    kinds = {(f.kind, f.path) for f in rep.fatal}
    assert kinds == {("weakened", "tests/test_a.py")}
    assert guard.revert(root, tag, rep) == ["tests/test_a.py"]
    assert (root / "tests" / "test_a.py").read_text().count("assert") == 2 and (root / "tests" / "test_new.py").exists()
    assert not guard.check(root, tag).fatal


def test_guard_catches_skips_deleted_tests_protected_files_syntax_and_stubs(tmp_path):
    root = tmp_path
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")
    (root / "uv.lock").write_text("lock\n")
    (root / "m.py").write_text("x = 1\n")
    gitbox.ensure_repo(root)
    tag = gitbox.last_green(root)
    (root / "tests" / "test_a.py").write_text("import pytest\n" "@pytest.mark.skip\ndef test_a():\n    assert 1\n")
    (root / "uv.lock").write_text("changed\n")
    (root / "m.py").write_text("def broken(:\n")
    (root / "stubs.py").write_text("def a():\n    raise NotImplementedError\ndef b():\n    pass\ndef c():\n    pass  # TODO\n")
    rep = guard.check(root, tag)
    got = {(f.kind, f.path) for f in rep.findings}
    assert {("weakened", "tests/test_a.py"), ("protected", "uv.lock"), ("syntax", "m.py")} <= got and any(f.kind == "stubs" and not f.fatal for f in rep.findings)
    (root / "tests" / "test_a.py").unlink()
    assert ("deleted", "tests/test_a.py") in {(f.kind, f.path) for f in guard.check(root, tag).findings}
    assert guard.revert(root, tag, guard.check(root, tag)) and (root / "tests" / "test_a.py").exists() and (root / "uv.lock").read_text() == "lock\n"


def test_guard_scope_budget_is_a_warning(tmp_path):
    root = tmp_path
    (root / "a.py").write_text("x=1\n")
    gitbox.ensure_repo(root)
    for i in range(5):
        (root / f"f{i}.py").write_text("y = 1\n")
    rep = guard.check(root, gitbox.last_green(root), guard.Budget(max_files=3))
    assert [f.kind for f in rep.findings] == ["scope"] and not rep.fatal


# ----------------------------------------------------------------------------- health
def test_health_gate_reports_each_reason_and_revives_a_dead_server(tmp_path):
    alive = {"up": False}
    ran = []

    def runner(cmd, **kw):
        ran.append(cmd)
        alive["up"] = True

    clock = {"t": 1000.0}
    h = health.Health(tmp_path, urls=["http://x"], restart_cmds=[["restart-it"]], clock=lambda: clock["t"], probe=lambda u: alive["up"], run=runner,
                      game=lambda: None, temps=lambda: [], free_gb=lambda p: 50.0)
    assert h.gate() is None and ran == [["restart-it"]] and h.restarts == 1            # down -> restarted -> fine
    alive["up"] = False
    runner_called = len(ran)
    assert h.gate().startswith("model server down") and len(ran) == runner_called       # a second restart is rate limited
    h2 = health.Health(tmp_path, probe=lambda u: True, game=lambda: "proton", temps=lambda: [95.0], free_gb=lambda p: 0.5)
    assert "disk" in h2.gate()
    h2.free_gb = lambda p: 50.0
    assert "game" in h2.gate()
    h2.pause_on_game = False
    assert "hot" in h2.gate()
    h2.temps = lambda: [60.0]
    assert h2.gate() is None


# ----------------------------------------------------------------------------- front matter / queue / preflight
def test_frontmatter_parse_and_graceful_fallbacks():
    meta, body = frontmatter.split("---\nname: chars\nweight: 3\nverify: pytest -q\ndepends: a, b\nkind: bug\nplan: true\nunknown: x\n---\nDo the thing.\n")
    assert meta == {"name": "chars", "weight": "3", "verify": "pytest -q", "depends": "a, b", "kind": "bug", "plan": "true"} and body == "Do the thing.\n"
    assert frontmatter.weight(meta) == 3.0 and frontmatter.depends(meta) == ["a", "b"]
    text = "---\njust some words\n---\nbody"
    assert frontmatter.split(text) == ({}, text)                                       # not key: value -> left alone
    assert frontmatter.weight({"weight": "abc"}) == 1.0 and frontmatter.weight({"weight": "999"}) == 20.0


def test_queue_applies_front_matter(tmp_path):
    (tmp_path / "p1.md").write_text("---\nname: first\nweight: 2\nverify: make check\n---\nWrite the parser and make `pytest` pass.")
    items, _ = queue.load([str(tmp_path)])
    assert items[0].name == "first" and items[0].weight == 2.0 and items[0].verify == "make check" and items[0].text.startswith("Write the parser")


def test_preflight_flags_unverifiable_vague_and_asking_prompts(tmp_path):
    (tmp_path / "real.py").write_text("")
    items = [queue.Item("vague", "make it nicer"),
             queue.Item("ok", "Add `real.py` support and a test; it must pass. Also create `new_file.py`.", depends=["later"]),
             queue.Item("asks", "Add logging, and let me know which format you prefer, then the tests should pass." + " x" * 40),
             queue.Item("vague", "duplicate name with enough text to pass the length check, tests should pass" + " y" * 20),
             queue.Item("later", "another prompt that is long enough and that must work correctly in the end" + " z" * 10)]
    out = "\n".join(preflight.check(items, tmp_path))
    assert "too vague" in out and "nothing in it can be checked" in out and "asks you something" in out and "two prompts are called 'vague'" in out
    assert "new_file.py" in out and "real.py" not in out.replace("new_file", "") and "depends on 'later'" in out
    assert not any("[ok]" in w and "vague" in w for w in preflight.check(items[1:2], tmp_path))


# ----------------------------------------------------------------------------- mapper
def test_map_lists_defs_docstrings_importers_and_respects_the_budget(tmp_path):
    (tmp_path / "core.py").write_text('"""core"""\nDEFAULT = 1\n\ndef run(a, b):\n    """Run the thing.\n\n    more"""\n\nclass Box:\n    """A box."""\n    def open(self): ...\n    def _hidden(self): ...\n')
    (tmp_path / "app.py").write_text("import core\nfrom core import run\n\ndef main():\n    return run(1, 2)\n")
    (tmp_path / "web.js").write_text("export function render() {}\nconst go = async () => 1;\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.js").write_text("function nope() {}")
    files = mapper.scan(tmp_path)
    assert {f.path for f in files} == {"core.py", "app.py", "web.js"}
    text = mapper.render(files)
    assert "run(a, b) - Run the thing." in text and "class Box [open] - A box." in text and "used by app" in text and "DEFAULT" in text and "render" in text and "_hidden" not in text
    small = mapper.render(files, budget=230)
    assert len(small) <= 230 and "core.py" in small


def test_map_refresh_reports_changes_and_keeps_the_model_notes(tmp_path):
    (tmp_path / "a.py").write_text("def one(): pass\n")
    assert mapper.refresh(tmp_path) == ["a.py"] and mapper.refresh(tmp_path) == []
    f = tmp_path / ".night" / "MAP.md"
    f.write_text(f.read_text() + "\n## Notes\nthe tricky part is X\n")
    (tmp_path / "b.py").write_text("def two(): pass\n")
    assert mapper.refresh(tmp_path) == ["b.py"]
    text = f.read_text()
    assert "two()" in text and "the tricky part is X" in text and text.count("## Notes") == 1
    (tmp_path / "a.py").unlink()
    assert mapper.refresh(tmp_path) == ["a.py"] and "one()" not in f.read_text()


# ----------------------------------------------------------------------------- probe / deliverables / clock / notify / speed
def test_probe_detects_an_unusable_test_command():
    from agent.night import probe

    bad = probe.probe("pytest", lambda: testwatch.parse("sh: pytest: command not found", 127))
    assert not bad.usable and "cannot run" in bad.text()
    none = probe.probe(None, lambda: 1 / 0)
    assert none.usable and "none detected" in none.text()
    fine = probe.probe("pytest -q", lambda: testwatch.parse("38 passed in 3s", 0))
    assert fine.usable and "38 passing" in fine.text() and "all green" in fine.text()
    nothing = probe.probe("pytest -q", lambda: testwatch.parse("no tests ran in 0.01s", 5))
    assert not nothing.usable


def test_deliverables_only_count_files_introduced_by_a_creating_verb(tmp_path):
    from agent.night import deliverables

    (tmp_path / "old.py").write_text("")
    text = "Create `tests/test_wren.py` and add `wren.py`. The existing `old.py` stays; compare with `example.py` for style."
    assert deliverables.expected(text, tmp_path) == ["tests/test_wren.py", "wren.py"]
    (tmp_path / "wren.py").write_text("")
    assert deliverables.missing(["tests/test_wren.py", "wren.py"], tmp_path) == ["tests/test_wren.py"]


def test_clock_helpers():
    from datetime import datetime

    from agent.night import clock

    now = datetime(2026, 11, 6, 23, 0)
    assert clock.hours_until("07:30", now) == 8.5 and clock.hours_until("23:30", now) == 0.5
    assert clock.seconds_to_start(None, now) == 0.0 and clock.seconds_to_start("01:00", now) == 7200.0
    assert clock.next_occurrence("23:00", now).day == 7                                   # the same minute is already past: tomorrow
    for bad in ("25:00", "7", "ab:cd", ""):
        with pytest.raises(ValueError):
            clock.next_occurrence(bad, now)


def test_notify_always_leaves_a_marker_and_only_posts_when_asked(tmp_path, monkeypatch):
    from agent.night import notify

    posted = []
    monkeypatch.delenv("NIGHT_NTFY_URL", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert notify.notify(tmp_path, "done", post=lambda *a, **k: posted.append(a)) == ["marker", "bell"] and not posted
    assert (tmp_path / ".night" / "DONE").read_text().strip() == "done"
    monkeypatch.setenv("NIGHT_NTFY_URL", "https://ntfy.example/topic")
    assert "ntfy" in notify.notify(tmp_path, "done", post=lambda *a, **k: posted.append(a)) and posted
    monkeypatch.setenv("NIGHT_NTFY_URL", "file:///etc/passwd")
    assert "ntfy" not in notify.notify(tmp_path, "done", post=lambda *a, **k: posted.append(a))      # only http(s)
    assert notify.headline(["a"], ["b", "c"], False) == "nightage night finished: 1 done, 2 not finished, tests RED"


def test_event_summary_speed(tmp_path):
    rows = [{"ts": 50, "session": "p-1", "kind": "step", "gen": 100, "decode_s": 2.0, "prefill_s": 1.0, "prompt": 5000},
            {"ts": 51, "session": "p-1", "kind": "step", "gen": 100, "decode_s": 2.0, "prefill_s": 3.0, "prompt": 8000}]
    es = handoff.summarize_events(_events(tmp_path, rows), 0, "p")
    assert es.tps == 50.0 and es.peak_prompt == 8000 and es.prefill_share == pytest.approx(4 / 8)
    assert handoff.EventSummary().tps == 0.0


def test_lint_problems_reads_ruff_output_and_is_silent_without_ruff(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("print(undefined_name)\n")

    class P:
        stdout = "a.py:1:7: F821 Undefined name `undefined_name`\nFound 1 error.\n"

    monkeypatch.setattr("shutil.which", lambda name: "/bin/ruff")
    assert guard.lint_problems(tmp_path, ["a.py", "notes.md"], run=lambda *a, **k: P()) == ["a.py:1:7: F821 Undefined name `undefined_name`"]
    assert guard.lint_problems(tmp_path, ["notes.md"], run=lambda *a, **k: 1 / 0) == []
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert guard.lint_problems(tmp_path, ["a.py"]) == []


def test_watch_render_shows_clock_queue_control_and_recent_events():
    from agent.night import watch

    state = {"left_s": 3 * 3600 + 5 * 60, "items": [{"name": "map", "status": "done", "attempts": 1, "seconds": 600, "kinds": ["green"]},
                                                    {"name": "characters", "status": "active", "attempts": 3, "seconds": 5400, "kinds": ["no_edits", "tests_red", "tests_red"]}],
             "events": [f"event {i}" for i in range(20)], "flaky": ["t1"]}
    text = watch.render(state, "qwen3.8:27b-text 20 GB 100% GPU", [61.0, 70.4], control="pause")
    assert "3 h 05 min left" in text and "[PAUSE requested]" in text and "prompts: 1/2 done" in text and "✓ map" in text and "▶ characters" in text and "tests_red" in text
    assert "flaky tests: 1" in text and "model: qwen3.8:27b-text" in text and "61 C, 70 C" in text and "event 19" in text and "event 5" not in text
    assert watch.render({}) .startswith("night: 0 h 00 min left")


def test_preflight_does_not_flag_the_harnesss_own_files(tmp_path):
    out = preflight.check([queue.Item("a", "Use `.night/MAP.md` and create `new_file.py`; the tests must pass and it should work.")], tmp_path)
    assert any("new_file.py" in w for w in out) and not any(".night/MAP.md" in w for w in out)
