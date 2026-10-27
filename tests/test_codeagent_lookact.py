"""D2: the look-vs-act logic as one module (lookact.py) - classification, the read pause, notes - and the D1 deadlock:
with reading paused the model can still get the lines it needs by acting (replace_lines answers with the current lines)."""

from __future__ import annotations

import pytest

from agent.codeagent import lookact as L

from tests.test_codeagent_engine import FakeLLM, engine  # noqa: F401


@pytest.mark.parametrize("name,args,result,kind", [
    ("read_file", {"path": "a.py"}, "x", "look"),
    ("show_symbol", {"name": "f"}, "x", "look"),
    ("run", {"command": "sed -n '1,80p' a.py"}, "exit code 0", "look"),
    ("run", {"command": "python3 a.py"}, "exit code 0", "act"),
    ("replace_lines", {"path": "a.py"}, "NOT CHANGED: you have not seen ...", "look"),
    ("replace_lines", {"path": "a.py"}, "edited a.py: lines 1-2 ...", "act"),
    ("edit_file", {"path": "a.py"}, "edit_file failed, nothing changed", "act"),
    ("mark_done", {}, "Item 1 confirmed", "neutral"),
    ("project_note", {}, "noted", "neutral"),
    ("web_search", {"query": "x"}, "...", "neutral"),
])
def test_classify(name, args, result, kind):
    assert L.classify(name, args, result) == kind


def test_pause_blocks_looks_but_never_writes_and_an_act_resumes_reading():
    la = L.LookAct()
    for _ in range(L.LOOK_BLOCK_AT):
        la.observe("read_file", {"path": "a.py"}, "text")
    assert la.blocked("read_file", {"path": "a.py"}) and la.blocked("run", {"command": "cat a.py"})
    assert la.blocked("replace_lines", {"path": "a.py"}) is None and la.blocked("run", {"command": "python3 a.py"}) is None
    la.observe("replace_lines", {"path": "a.py"}, "NOT CHANGED: lines ...")          # it got lines, but nothing changed yet
    assert la.blocked("read_file", {"path": "a.py"})
    la.observe("replace_lines", {"path": "a.py"}, "edited a.py: lines 3-4 replaced")
    assert la.blocked("read_file", {"path": "a.py"}) is None and la.look_streak == 0


def test_the_first_steps_of_a_stage_are_free_then_the_pause_comes_at_the_limit():
    la = L.LookAct()
    la.begin_stage()
    for _ in range(L.FREE_STAGE_STEPS):
        la.observe("read_file", {"path": "a.py"}, "text")                    # initial orientation: not counted
    assert la.look_streak == 0 and la.blocked("read_file", {"path": "a.py"}) is None
    for _ in range(L.LOOK_BLOCK_AT - 1):
        la.observe("read_file", {"path": "a.py"}, "text")
    assert la.blocked("read_file", {"path": "a.py"}) is None
    la.observe("read_file", {"path": "a.py"}, "text")
    assert la.look_streak == L.LOOK_BLOCK_AT and la.blocked("read_file", {"path": "a.py"})
    plain = L.LookAct()                                                      # outside a stage there is no allowance
    for _ in range(L.LOOK_BLOCK_AT):
        plain.observe("read_file", {"path": "a.py"}, "text")
    assert plain.blocked("read_file", {"path": "a.py"})


def test_an_act_does_not_refill_the_stage_allowance():
    la = L.LookAct()
    la.begin_stage()
    la.observe("write_file", {"path": "a.py"}, "wrote a.py (3 lines)")      # step 1 of 6 was an act
    for _ in range(L.FREE_STAGE_STEPS - 1):
        la.observe("read_file", {"path": "a.py"}, "text")
    assert la.look_streak == 0
    la.observe("read_file", {"path": "a.py"}, "text")
    assert la.look_streak == 1


def test_pause_can_be_set_up_front_for_a_restart():
    la = L.LookAct()
    la.pause()
    assert la.blocked("outline", {"path": "a.py"})


def test_wander_notes_every_ten_looks_and_edit_gap_after_probing():
    lines = {"n": 0}
    la = L.LookAct(read_lines=lambda: lines["n"], reset_lines=lambda: lines.update(n=0))
    notes = [r for _ in range(20) for r in la.observe("search", {"pattern": "x"}, "hits")]
    assert [r.event for r in notes] == ["wander", "wander"] and not any(r.think for r in notes)
    lines["n"] = 50
    la.observe("write_file", {"path": "a.py"}, "wrote a.py (3 lines)")
    assert la.look_streak == 0 and lines["n"] == 0
    gaps = [r for i in range(L.TEST_STREAK_NUDGE) for r in la.observe("run", {"command": f"python3 a.py {i}"}, "exit code 0")]
    assert [r.event for r in gaps] == ["edit_gap"] and gaps[0].think
    la.observe("write_file", {"path": ".scratch/t.py"}, "wrote .scratch/t.py")          # scratch is not progress
    assert la.edit_gap == L.TEST_STREAK_NUDGE + 1


def test_d1_deadlock_is_impossible_in_the_engine(tmp_path):
    """Reading paused + no verbatim old text: replace_lines shows the current lines, the next replace_lines edits."""
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "a.py").write_text("def f():\n    return 1\n")
    eng._look_streak = L.LOOK_BLOCK_AT
    assert eng.execute("read_file", {"path": "a.py"}).startswith("NOT RUN")
    eng._run_call({"name": "replace_lines", "arguments": {"path": "a.py", "start": 2, "end": 2, "new_text": "    return 2"}})
    shown = [m for m in eng.ctx.messages if m["role"] == "tool"][-1]["content"]
    assert shown.startswith("NOT CHANGED") and "2|     return 1" in shown and eng._look_streak == L.LOOK_BLOCK_AT + 1
    eng._run_call({"name": "replace_lines", "arguments": {"path": "a.py", "start": 2, "end": 2, "new_text": "    return 2"}})
    assert [m for m in eng.ctx.messages if m["role"] == "tool"][-1]["content"].startswith("edited a.py") and eng._look_streak == 0
    assert (repo / "a.py").read_text() == "def f():\n    return 2\n"


def test_lookact_state_round_trips_and_tolerates_junk():
    la = L.LookAct()
    la.look_streak, la.wander_nudges, la.edit_gap, la.edit_nudges, la.probing = 9, 2, 5, 1, True
    other = L.LookAct()
    other.load(la.state())
    assert other.state() == la.state() == {"look_streak": 9, "wander_nudges": 2, "edit_gap": 5, "edit_nudges": 1, "probing": True}
    other.load({"look_streak": "x", "probing": None})
    assert other.state() == {"look_streak": 0, "wander_nudges": 0, "edit_gap": 0, "edit_nudges": 0, "probing": False}
    other.load(None)
    assert other.look_streak == 0
