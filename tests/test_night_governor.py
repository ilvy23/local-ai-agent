"""The governor of `nightage night`: reveal order, adaptive escalation, guards, tournament, review, deliverables, dependencies, control file, health gate.
All on fakes: no model, a fake clock, throw-away git repos."""
from __future__ import annotations

import json

import pytest

from agent.night import gitbox, queue
from agent.night.governor import Night, as_state
from agent.night.runner import Attempt, Outcome
from agent.night.testwatch import TestState


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _night(tmp_path, items, script, tests, hours=8.0, **kw):
    """`script(attempt, n) -> (Outcome, seconds spent)` (it may edit files in tmp_path); `tests()` -> TestState or (ok, tail)."""
    (tmp_path / "a.txt").write_text("base")
    gitbox.ensure_repo(tmp_path)
    clock, seen = Clock(), []

    def runner(a: Attempt) -> Outcome:
        out, spent = script(a, len(seen))
        seen.append(a)
        clock.t += spent
        return out

    kw.setdefault("tournament", 0)
    kw.setdefault("map_mode", "off")
    n = Night(tmp_path, items, "RULES", hours, runner=runner, tests=tests, clock=clock, sleep=lambda s: setattr(clock, "t", clock.t + s), log=lambda s: None, **kw)
    return n, seen


def green():
    return TestState(ok=True, passed=3)


def red(*names):
    return TestState(ok=False, passed=1, failed=list(names), digest="red: " + ", ".join(names), tail="red")


def test_as_state_accepts_both_shapes():
    assert as_state((True, "x")).ok and as_state(green()) is not None and as_state((False, "boom")).tail == "boom"


def test_prompts_are_revealed_one_at_a_time_and_done_needs_tests_green(tmp_path):
    items = [queue.Item("p1", "FIRST TASK"), queue.Item("p2", "SECOND TASK")]

    def script(a, i):
        (tmp_path / f"work{i}.txt").write_text("x")
        return Outcome(0), 600

    n, seen = _night(tmp_path, items, script, green, map_mode="auto")
    rep = n.run()
    assert rep.done == ["p1", "p2"] and not rep.partial and rep.tests_ok
    assert "FIRST TASK" in seen[0].task and "SECOND TASK" not in seen[0].task
    assert "SECOND TASK" in seen[1].task and ".night/KNOWLEDGE.md" in seen[1].task and ".night/MAP.md" not in seen[1].task
    assert "RULES" in seen[0].task and (tmp_path / ".night" / "NIGHT_REPORT.md").exists() and "Never delete, skip or weaken a test" in seen[0].task


def test_a_failing_attempt_picks_its_next_strategy_from_the_failure_kind(tmp_path):
    items = [queue.Item("p1", "HARD TASK")]
    n, seen = _night(tmp_path, items, lambda a, i: (Outcome(3), 1500), lambda: red("t_x"), hours=3)      # never edits anything -> kind no_edits
    rep = n.run()
    assert rep.partial == ["p1"]
    assert "HARD TASK" in seen[0].task                                                  # start
    assert "ONE most important part" in seen[1].task                                    # no_edits -> small
    assert "HANDOFF" in seen[2].task and "changed NOTHING" in seen[2].task              # again no_edits -> fresh, with the measured handoff
    assert "different way" in seen[3].task                                              # again -> rollback
    assert n.items[0].kinds[:4] == ["no_edits"] * 4 and "LESSONS" in seen[2].task


def test_a_tests_red_attempt_resumes_without_new_text(tmp_path):
    def script(a, i):
        (tmp_path / "wip.txt").write_text(str(i))
        return Outcome(3), 900

    n, seen = _night(tmp_path, [queue.Item("p1", "TASK")], script, lambda: red("t1"), hours=2)
    n.run()
    assert seen[1].task == "" and seen[1].new is False                                  # tests_red with edits on disk: continue the same session


def test_the_end_state_is_the_last_green_checkpoint_and_red_work_is_kept(tmp_path):
    state = {"i": 0}

    def script(a, i):
        (tmp_path / "work.txt").write_text(f"attempt {i}")
        return Outcome(0 if i == 0 else 3), 1200

    def tests():
        state["i"] += 1
        return green() if state["i"] <= 2 else red("t")                                 # baseline green, attempt 0 green, everything after red

    n, seen = _night(tmp_path, [queue.Item("p1", "T")], script, tests, hours=2)
    rep = n.run()
    assert (tmp_path / "work.txt").read_text() == "attempt 0" and gitbox.git(tmp_path, "tag", "-l", "night/red-*").stdout.strip() and "Night report" in rep.text


def test_a_down_server_does_not_spin(tmp_path):
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: (Outcome(-1, "crash"), 5), lambda: red(), hours=1)
    n.run()
    assert len(seen) < 80


def test_weakened_tests_are_reverted_and_the_lesson_is_recorded(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n    assert 2\n")

    def script(a, i):
        (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")
        (tmp_path / "feature.py").write_text("x = 1\n")
        return Outcome(0), 600

    n, seen = _night(tmp_path, [queue.Item("p1", "T")], script, green, hours=1)
    n.run()
    assert (tmp_path / "tests" / "test_a.py").read_text().count("assert") == 2          # reverted to the checkpoint
    assert "weakened" in (tmp_path / ".night" / "LESSONS.md").read_text() or "broke" in (tmp_path / ".night" / "LESSONS.md").read_text()
    assert any("guard:" in e for e in n.events)


def test_a_missing_deliverable_keeps_the_prompt_open_until_it_exists(tmp_path):
    def script(a, i):
        if i >= 1:
            (tmp_path / "wren.py").write_text("x = 1\n")
        else:
            (tmp_path / "other.py").write_text("y = 1\n")
        return Outcome(0), 600

    n, seen = _night(tmp_path, [queue.Item("p1", "Please create `wren.py` with a class and make the tests pass.")], script, green, hours=3)
    rep = n.run()
    assert rep.done == ["p1"] and n.items[0].kinds == ["incomplete", "green"] and len(seen) == 2


def test_the_verify_command_of_a_prompt_must_pass(tmp_path):
    calls = []

    def verify(cmd):
        calls.append(cmd)
        return len(calls) >= 2                                                           # fails the first time, passes the second

    n, seen = _night(tmp_path, [queue.Item("p1", "T", verify="make check")], lambda a, i: ((tmp_path / f"f{i}").write_text("x"), (Outcome(0), 600))[1], green, hours=2,
                     verify=verify)
    rep = n.run()
    assert n.items[0].kinds == ["tests_red", "green"] and rep.done == ["p1"] and calls == ["make check", "make check"]


def test_a_prerequisite_that_ran_out_of_time_does_not_block_the_next_prompt(tmp_path):
    items = [queue.Item("p1", "T1"), queue.Item("p2", "T2", depends=["p1"])]
    n, seen = _night(tmp_path, items, lambda a, i: (Outcome(3), 1800), lambda: red("x"), hours=2.5)
    n.run()
    assert n.items[0].attempts > 0 and any("T2" in a.task for a in seen)                # p1 spent its slice (partial) - p2 still got its turn


def test_a_prerequisite_that_never_ran_still_blocks(tmp_path):
    items = [queue.Item("p1", "T1"), queue.Item("p2", "T2", depends=["p1"])]
    n, _ = _night(tmp_path, items, lambda a, i: (Outcome(0), 60), green, hours=1)
    assert n._ready(items[1]) is False                                                 # p1 is pending: p2 waits
    items[0].status = "done"
    assert n._ready(items[1]) is True
    items[0].status, items[0].attempts = "partial", 0
    assert n._ready(items[1]) is False                                                 # partial without a single attempt (it was skipped) blocks


def test_tournament_keeps_the_best_candidate_from_the_same_checkpoint(tmp_path):
    cands = {"n": 0}

    def script(a, i):
        cands["n"] += 1
        (tmp_path / "out.txt").write_text(f"candidate {cands['n']}")
        (tmp_path / f"extra{cands['n']}.txt").write_text("x")
        return Outcome(3), 600

    results = {}

    def tests():
        txt = (tmp_path / "out.txt").read_text() if (tmp_path / "out.txt").exists() else ""
        results[txt] = results.get(txt, 0) + 1
        # candidate 4 is the only good one; candidates before it are red with fewer passing tests
        if txt == "candidate 5":
            return TestState(ok=True, passed=9)
        return TestState(ok=False, passed=1 if txt != "candidate 4" else 5, failed=["t"], digest="red")

    n, seen = _night(tmp_path, [queue.Item("p1", "TASK")], script, tests, hours=6, tournament=3)
    n.run()
    assert any("tournament" in e for e in n.events) and any("winner" in e for e in n.events)
    assert (tmp_path / "out.txt").read_text() in ("candidate 4", "candidate 5", "candidate 3", "candidate 6")   # one winner, not a mix
    assert len(list(tmp_path.glob("extra*.txt"))) <= 3                                    # the other candidates' files did not pile up


def test_review_is_kept_when_green_and_discarded_when_it_breaks_things(tmp_path):
    marks = []

    def script(a, i):
        if "git diff" in a.task:
            marks.append("review")
            (tmp_path / "review.txt").write_text("reviewed")
        else:
            (tmp_path / "feature.txt").write_text("done")
        return Outcome(0), 600

    state = {"broken": False}

    def tests():
        return red("t") if state["broken"] and (tmp_path / "review.txt").exists() else green()

    n, seen = _night(tmp_path, [queue.Item("p1", "T")], script, tests, hours=4, review=True)
    n.run()
    assert marks == ["review"] and (tmp_path / "review.txt").exists() and any("review kept" in e for e in n.events)
    (tmp_path / "x").mkdir(exist_ok=True)

    tmp2 = tmp_path.parent / (tmp_path.name + "_b")
    tmp2.mkdir()
    state["broken"] = True

    def script2(a, i):
        if "git diff" in a.task:
            (tmp2 / "review.txt").write_text("bad")
        else:
            (tmp2 / "feature.txt").write_text("done")
        return Outcome(0), 600

    n2, _ = _night(tmp2, [queue.Item("p1", "T")], script2, lambda: red("t") if (tmp2 / "review.txt").exists() else green(), hours=4, review=True)
    n2.run()
    assert not (tmp2 / "review.txt").exists() and (tmp2 / "feature.txt").exists() and any("review discarded" in e for e in n2.events)


def test_control_file_stops_and_pauses(tmp_path):
    n, seen = _night(tmp_path, [queue.Item("p1", "T"), queue.Item("p2", "T2")], lambda a, i: (Outcome(3), 600), lambda: red("t"), hours=6)
    (tmp_path / ".night").mkdir()
    (tmp_path / ".night" / "CONTROL").write_text("stop\n")
    rep = n.run()
    assert seen == [] and "Night report" in rep.text and n.control() == "stop"


def test_health_gate_waits_for_the_machine_and_the_wait_costs_night_time(tmp_path):
    reasons = iter(["a game is running (proton)", "a game is running (proton)", None])

    class Gate:
        def gate(self):
            return next(reasons, None)

    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 100))[1], green, hours=2, health=Gate())
    n.run()
    assert any("waiting: a game is running" in e for e in n.events) and any("fit again" in e for e in n.events)
    assert n.clock() >= 120                                                              # two polls of 60 s were spent waiting


def test_a_slow_model_is_called_out(tmp_path):
    ev = tmp_path / "events.jsonl"
    rows = [{"ts": 1000 + i, "session": tmp_path.name + "-1", "kind": "step", "gen": 40, "decode_s": 10.0, "prefill_s": 1.0, "prompt": 9000} for i in range(12)]
    ev.write_text("\n".join(json.dumps(r) for r in rows))
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 100))[1], green, hours=1, events=ev, wall=lambda: 900.0)
    n.run()
    assert any("was slow: 4.0 tok/s" in e for e in n.events)


def test_state_is_written_atomically_and_status_data_is_complete(tmp_path):
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: ((tmp_path / "f").write_text("x"), (Outcome(0), 100))[1], green, hours=1)
    n.run()
    state = json.loads((tmp_path / ".night" / "state.json").read_text())
    assert state["items"][0]["status"] == "done" and state["left_s"] > 0 and not (tmp_path / ".night" / "state.json.tmp").exists()


def test_bug_flow_proves_the_bug_first_then_fixes_it_and_never_rolls_back_past_the_proof(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_old.py").write_text("def test_old():\n    assert 1\n")
    steps = []

    def script(a, i):
        steps.append(a.task)
        if "ONLY a test" in a.task:
            (tmp_path / "tests" / "test_bug.py").write_text("def test_bug():\n    assert fixed()\n")
        else:
            (tmp_path / "fix.txt").write_text("fixed")
        return Outcome(0), 600

    def tests():
        fixed = (tmp_path / "fix.txt").exists()
        has = (tmp_path / "tests" / "test_bug.py").exists()
        return TestState(ok=not has or fixed, passed=1, failed=[] if (not has or fixed) else ["tests/test_bug.py::test_bug"], digest="red", tail="red")

    n, seen = _night(tmp_path, [queue.Item("bugfix", "Crash when the list is empty.", kind="bug")], script, tests, hours=3)
    rep = n.run()
    assert rep.done == ["bugfix"] and "ONLY a test" in steps[0] and "tests/test_bug.py::test_bug" in steps[1] and "Do NOT fix" not in steps[1]
    assert n.items[0].repro == ["tests/test_bug.py::test_bug"] and any("bug is proven" in e for e in n.events)
    assert gitbox.git(tmp_path, "tag", "-l", "night/repro-*").stdout.strip()


def test_a_repro_attempt_that_touches_code_is_undone_and_after_three_tries_the_prompt_runs_plain(tmp_path):
    def script(a, i):
        (tmp_path / "src.py").write_text(f"x = {i}\n")                                  # changes code, never writes a test
        return Outcome(0), 300

    n, seen = _night(tmp_path, [queue.Item("bugfix", "Fix the crash on empty input, it must work.", kind="bug")], script, lambda: TestState(ok=True, passed=1), hours=3)
    n.run()
    assert sum("repro attempt" in e for e in n.events) == 3 and any("plain task" in e for e in n.events) and n.items[0].kind == ""


def test_closing_round_repairs_a_nearly_finished_prompt_instead_of_rolling_it_back(tmp_path):
    box = {"fixed": False}

    def script(a, i):
        if "almost finished" in a.task:
            box["fixed"] = True
            return Outcome(0), 300
        (tmp_path / "feature.py").write_text("x = 1\n")
        return Outcome(3), 1000                                                    # the slice ends while one test is red

    def tests():
        return green() if box["fixed"] or not (tmp_path / "feature.py").exists() else red("tests/test_a.py::test_one")

    n, seen = _night(tmp_path, [queue.Item("p1", "Build the feature, it must work.")], script, tests, hours=1.0)
    rep = n.run()
    assert any("closing round" in e for e in n.events) and any("almost finished" in a.task and "test_one" in a.task for a in seen)
    assert rep.tests_ok and rep.done == ["p1"] and (tmp_path / "feature.py").exists()                           # the work stayed
    assert gitbox.tags(tmp_path, "night/done-p1")                                                               # and the prompt was tagged as finished


def test_closing_round_is_skipped_without_time_or_when_stopped_and_a_failing_repair_still_rolls_back(tmp_path):
    def script(a, i):
        (tmp_path / "feature.py").write_text(f"x = {i}\n")
        return Outcome(3), 600

    n, seen = _night(tmp_path, [queue.Item("p1", "Build it, it must work.")], script, lambda: red("t") if (tmp_path / "feature.py").exists() else green(), hours=0.6)
    rep = n.run()
    assert any("closing round" in e for e in n.events) and not rep.done and rep.tests_ok and not (tmp_path / "feature.py").exists()   # repair failed: normal rollback, work kept under a tag
    assert gitbox.tags(tmp_path, "night/red-")
    quiet, _ = _night(tmp_path / "q", [queue.Item("p1", "T")], lambda a, i: (Outcome(3), 3000), lambda: red("t"), hours=0.3) if (tmp_path / "q").mkdir() is None else (None, None)
    quiet.run()
    assert not any("closing round" in e for e in quiet.events)                                                  # no time left: no extra attempt


def test_a_productive_prompt_that_ran_out_of_its_slice_gets_one_grace_stretch(tmp_path):
    box = {"fixed": False}

    def script(a, i):
        if "almost finished" in a.task:
            box["fixed"] = True
            return Outcome(0), 400
        (tmp_path / f"part{i}.py").write_text("x = 1\n" * 60)
        return Outcome(-1, "timeout"), 2400                                             # the slice ends in a timeout, one test still red

    def tests():
        return green() if box["fixed"] else red("tests/test_x.py::test_one")

    n, seen = _night(tmp_path, [queue.Item("p1", "Build it, it must work."), queue.Item("p2", "Second one, it must work.")], script, tests, hours=4)
    n.run()
    assert any("grace period" in e and "repair" in e for e in n.events) and n.items[0].grace_used
    assert n.items[0].status == "done" and sum("grace period" in e for e in n.events) == 1           # once per prompt, and it finished the work


def test_no_grace_when_the_prompt_made_no_attempt_progress_or_time_is_short(tmp_path):
    n, seen = _night(tmp_path, [queue.Item("p1", "T")], lambda a, i: (Outcome(0), 600), green, hours=4)
    n.run()
    assert not any("grace period" in e for e in n.events)                                              # done in time: nothing to extend
    short, _ = _night(tmp_path / "s", [queue.Item("p1", "T")], lambda a, i: (Outcome(-1, "timeout"), 1500), lambda: red("t"), hours=0.6) if (tmp_path / "s").mkdir() is None else (None, None)
    short.run()
    assert not any("grace period" in e for e in short.events)                                          # not enough night left for it


def test_an_unfinished_red_prompt_is_shelved_as_a_patch_and_the_next_prompt_starts_green(tmp_path):
    def script(a, i):
        if "SECOND" in a.task:
            (tmp_path / "second.py").write_text("y = 2\n")
            return Outcome(0), 600
        (tmp_path / "first.py").write_text("x = 1\n" * 40)
        return Outcome(3), 900                                                       # never gets the tests green, not even in the grace stretch

    def tests():
        return red("tests/test_first.py::test_one") if (tmp_path / "first.py").exists() else green()

    n, seen = _night(tmp_path, [queue.Item("p1", "FIRST thing, it must work."), queue.Item("p2", "SECOND thing, it must work.")], script, tests, hours=4)
    n.run()
    patch = tmp_path / ".night" / "patches" / "p1.patch"
    assert patch.exists() and "first.py" in patch.read_text() and any("saved as p1.patch" in e for e in n.events)
    assert n.items[0].status == "partial" and n.items[0].attempts > 1                    # (later revisit attempts overwrite the note, the patch file and the event stay)
    assert n.items[1].status == "done" and (tmp_path / "second.py").exists()           # the second prompt got a green base and finished


def test_a_revisit_of_a_shelved_prompt_is_told_about_its_patch(tmp_path):
    n, _ = _night(tmp_path, [queue.Item("p1", "FIRST thing.")], lambda a, i: (Outcome(0), 60), green, hours=1)
    (tmp_path / ".night" / "patches").mkdir(parents=True)
    (tmp_path / ".night" / "patches" / "p1.patch").write_text("diff --git a/x b/x\n")
    task = n._task(n.items[0], "fresh", "")
    assert ".night/patches/p1.patch" in task and "git apply" in task
    assert ".night/patches" not in n._task(n.items[0], "resume", "")                    # a resume carries no new text
