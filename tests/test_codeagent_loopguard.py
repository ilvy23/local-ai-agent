"""Loop guard, second generation: near-duplicate calls with an equal outcome, the escalation ladder (note at 2, nudge + one reasoning
pass at 3, segment ends at 4), the DEAD ENDS ledger (kept in the engine, saved, shown to a fresh context), error-shape hints and
the interactive-program hint. FakeLLM only, no model."""

from __future__ import annotations

import json

import pytest

from agent.codeagent import loopguard as LG, sessionstore as SS, session as S, supervisor as SV, verified
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_hardening import engine
from tests.test_codeagent_sessions import _resume, _saved, new_engine as new_session_engine
from tests.test_codeagent_supervisor import Clock


def play(eng, command, result, tool="run", **extra):
    """One finished tool call as the engine stores it, then the guard (the guard appends its notes to the last tool message)."""
    args = {"command": command, **extra} if tool == "run" else {"pattern": command}
    eng.ctx.add({"role": "assistant", "content": json.dumps({"tool": tool, "arguments": args})})
    eng.ctx.add({"role": "tool", "content": result})
    eng._loop_guard(tool, args, result)


def fresh(tmp_path, sub, llm=None):
    (tmp_path / sub).mkdir(exist_ok=True)
    return engine(tmp_path / sub, llm or FakeLLM([]))[0]


def systems(eng):
    return [m["content"] for m in eng.ctx.messages if m["role"] == "system"]


def last_tool(eng):
    return [m["content"] for m in eng.ctx.messages if m["role"] == "tool"][-1]


# ── near duplicates ─────────────────────────────────────────────────────────

def test_edited_python_c_variants_with_the_same_output_count_as_one_attempt(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    out = "exit code 0\nscore 0\nthe room is dark"
    play(eng, 'python3 -c "import game; g = game.Game(); print(g.score, g.room(1))"', out)
    play(eng, "python3 -c 'import game; g = game.Game(); print(g.score, g.room(2))'", out)
    assert "Same result" in last_tool(eng) and not systems(eng)                      # repeat 2: only the note
    play(eng, 'python3 -c "import game;  g = game.Game(); print( g.score, g.room(3) )"', out)
    assert len(systems(eng)) == 1 and "3 times" in systems(eng)[0]                   # repeat 3: the nudge


def test_the_same_command_with_new_output_is_not_a_repeat(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    words = ["the bell rings", "a door creaks open", "snow falls on the square", "the ferryman waits", "a dog barks twice", "smoke over the hills",
             "nothing moves", "the clock stops"]
    for w in words:
        play(eng, "python3 main.py", f"exit code 0\n{w} and the story goes on in its own way")
    assert not systems(eng) and not eng._abort_loop and "Same result" not in last_tool(eng)


def test_pytest_with_a_different_k_and_different_outputs_is_not_a_loop(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    for name in ("test_alpha", "test_beta", "test_gamma", "test_delta", "test_eps"):
        play(eng, f"python3 -m pytest -q tests -k {name}", f"exit code 1\nFAILED tests/test_x.py::{name} - assert 1 == 2\n1 failed in 0.1s")
    assert not systems(eng) and not eng._abort_loop and "Same result" not in last_tool(eng)


def test_the_same_error_from_edited_commands_counts_when_the_first_error_line_is_equal(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    tb = "exit code 1\n{} rooms loaded\nTraceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nKeyError: 'room'"
    play(eng, "python3 -c \"import m; m.go('a')\"", tb.format("4"))
    play(eng, "python3 -c \"import m; m.go('b')\"", tb.format("a few"))                  # other text before it, same error
    assert "Same result" in last_tool(eng)


def test_other_tools_and_distant_calls_do_not_join_a_near_match():
    assert LG.near_calls("run", "python3 -c 'print(1)'", "python3 -c 'print(2)'")
    assert not LG.near_calls("read_file", "a.py", "a.py")
    assert not LG.near_calls("run", "pytest -q", "cat README.md")
    assert LG.tokens("echo 'A1' \"B22\" 7") == {"echo", "a1", "b22"} and LG.tokens("f(1, 2)") == LG.tokens("f(3,4)")
    assert not LG.near_calls("search", "p1", "p2")                                # an identifier with a digit is a different word


# ── the ladder ──────────────────────────────────────────────────────────────

def test_ladder_note_at_2_nudge_and_one_reasoning_pass_at_3_abort_at_4(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    cmd = "python3 sim.py --seed 1"
    play(eng, cmd, "exit code 0\nsame screen")
    assert last_tool(eng) == "exit code 0\nsame screen" and eng.think_steps == 0
    play(eng, cmd, "exit code 0\nsame screen")
    assert last_tool(eng).endswith(LG.SAME_RESULT_NOTE) and not systems(eng) and eng.think_steps == 0     # no nag, no thinking
    play(eng, cmd, "exit code 0\nsame screen")
    nudge = systems(eng)[-1]
    assert eng.think_steps == 1 and not eng._abort_loop
    assert "Your last 3 attempts" in nudge and nudge.count("`python3 sim.py --seed 1`") == 3 and "same screen" in nudge
    assert "read the code at the failing line" in nudge and "change your hypothesis" in nudge
    eng.think_steps = 0
    play(eng, cmd, "exit code 0\nsame screen")
    assert eng._abort_loop == 4 and eng.think_steps == 0
    assert eng.events.of("loop_abort") and eng.events.of("loop")


def test_the_ladder_limits_come_from_the_config(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "loop_note": 3, "loop_abort": 5}}
    eng.ctx.step = 20
    for _ in range(2):
        play(eng, "python3 a.py", "exit code 0\nsame")
    assert "Same result" not in last_tool(eng)
    for _ in range(2):
        eng.ctx.step += 10
        play(eng, "python3 a.py", "exit code 0\nsame")
    assert not eng._abort_loop                                                  # 4 repeats: below the configured 5
    play(eng, "python3 a.py", "exit code 0\nsame")
    assert eng._abort_loop == 5


def test_a_loop_ends_the_segment_at_the_4th_repeat_without_another_model_call(tmp_path):
    steps = [call("run", command="python3 -c 'import nothere'")] * 4 + [say("never asked")]
    llm = FakeLLM(steps)
    eng, repo, _ = engine(tmp_path, llm)
    eng.config = {**eng.config, "coding": {**(eng.config.get("coding") or {}), "forced_summary": False}}      # the bare stop (the summary: test_codeagent_finalize.py)
    res = eng.turn("make it run")
    assert res.stopped == "loop" and len(llm.calls) == 4 and len(llm.script) == 1
    assert eng.dead_ends.items and eng.dead_ends.items[-1]["tool"] == "run"


def test_only_the_third_repeat_pass_may_think_and_only_for_loops(tmp_path):
    llm = FakeLLM([call("list_dir", path="."), call("list_dir", path="d1"), call("list_dir", path="d2"), say("done")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.turn("look")
    assert all(c["think"] is False for c in llm.calls)                           # distinct reads: no guard, no thinking
    llm2 = FakeLLM([call("run", command="echo hi")] * 2 + [say("done")])
    eng2 = fresh(tmp_path, "b", llm2)
    eng2.turn("say hi")
    assert all(c["think"] is False for c in llm2.calls)                          # the repeat-2 note alone never forces a pass
    llm3 = FakeLLM([call("run", command="echo hi")] * 3 + [say("done")])
    eng3 = fresh(tmp_path, "c", llm3)
    eng3.turn("say hi")
    assert [c["think"] for c in llm3.calls] == [False, False, False, True]


# ── error-shape hints ───────────────────────────────────────────────────────

def _traceback(repo):
    return (f"exit code 1\nTraceback (most recent call last):\n  File \"{repo}/mod.py\", line 7, in f\n    x = y\nValueError: bad value")


def test_the_traceback_hint_comes_on_the_second_identical_error_only(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "mod.py").write_text("\n" * 10)
    eng.ctx.step = 20
    play(eng, "python3 mod.py", _traceback(repo))
    assert "Traceback ends" not in last_tool(eng)
    play(eng, "python3 mod.py", _traceback(repo))
    assert "Traceback ends at mod.py:7 - open it with read_file/show_symbol instead of re-running." in last_tool(eng)
    assert last_tool(eng).count("Traceback ends") == 1


def test_no_traceback_hint_for_a_frame_outside_the_repo(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    tb = "exit code 1\nTraceback (most recent call last):\n  File \"/usr/lib/python3.13/json/decoder.py\", line 5, in raw_decode\nValueError: x"
    play(eng, "python3 mod.py", tb)
    play(eng, "python3 mod.py", tb)
    assert "Traceback ends" not in last_tool(eng) and "Same result" in last_tool(eng)


def test_the_pytest_hint_names_the_test_id_on_the_second_failure(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    out = "exit code 1\nFAILED tests/test_a.py::test_one - assert 1 == 2\n1 failed in 0.2s"
    play(eng, "python3 -m pytest -q tests", out)
    assert "Failing test" not in last_tool(eng)
    play(eng, "python3 -m pytest -q tests", out)
    assert "Failing test: tests/test_a.py::test_one - open it with read_file/show_symbol instead of re-running." in last_tool(eng)


# ── interactive programs ────────────────────────────────────────────────────

def test_piped_interactive_runs_get_the_script_hint_once_per_session(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 20
    screen = "exit code 0\nWelcome to the harbour\n1) look\n2) leave\n> "
    for word in ("look", "leave", "look"):
        play(eng, f"printf '{word}\\nquit\\n' | python3 main.py", screen)
    assert LG.INTERACTIVE_HINT in last_tool(eng)
    eng.ctx.step += 20
    play(eng, "printf 'look\\n' | python3 main.py", screen)
    assert LG.INTERACTIVE_HINT not in last_tool(eng)                             # once per session
    other = fresh(tmp_path, "z", FakeLLM([]))
    other.ctx.step = 20
    for i in range(3):
        play(other, "python3 main.py", "exit code 0\nWelcome to the harbour\n> ")
    assert LG.INTERACTIVE_HINT not in last_tool(other)                           # no piping: no hint
    third = fresh(tmp_path, "y", FakeLLM([]))
    third.ctx.step = 20
    for i in range(3):
        play(third, f"printf 'a{i}\\n' | python3 main.py", "exit code 0\nvalue is 3\nvalue is 4")
    assert LG.INTERACTIVE_HINT not in last_tool(third)                           # the output is no prompt echo


# ── dead ends ───────────────────────────────────────────────────────────────

def looped(eng, cmd="python3 sim.py", out="exit code 1\nKeyError: 'room'"):
    eng.ctx.step = 20
    for _ in range(3):
        play(eng, cmd, out)


def test_a_loop_is_recorded_deduped_and_capped(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    looped(eng)
    (e,) = eng.dead_ends.items
    assert e["tool"] == "run" and e["cmd"] == "python3 sim.py" and e["out"] == "KeyError: 'room'" and e["times"] == 3
    eng.ctx.step += 10
    eng._sigs.clear()
    looped(eng, "python3 sim.py")                                                 # the same attempt again: one entry
    assert len(eng.dead_ends.items) == 1
    for i in range(20):
        eng.dead_ends.record("run", f"cmd{'x' * i} unique{'abcdefghijklmnopqrstuvwxyz'[i]}", "boom", 3, error=True, files=[], rsha="r")
    assert len(eng.dead_ends.items) == 12 and eng.dead_ends.items[-1]["cmd"].startswith("cmd")
    long = LG.DeadEnds()
    long.record("run", "x" * 400, "y" * 400, 3, error=True, files=[], rsha="r")
    assert len(long.items[0]["cmd"]) <= 120 and len(long.items[0]["out"]) <= 120


def test_dead_ends_are_saved_and_restored_and_an_old_session_loads(tmp_path):
    eng, repo = _saved(tmp_path)
    looped(eng)
    eng.dead_ends.items[-1]["files"] = ["sim.py"]
    eng2 = _resume(tmp_path, eng, repo)
    assert eng2.dead_ends.items == eng.dead_ends.items
    assert "DEAD ENDS (already tried, same result - do not repeat)" in eng2.resume_note and "python3 sim.py" in eng2.resume_note
    meta = SS.latest(repo)
    data = json.loads(meta.path.read_text())
    data["engine_state"].pop("dead_ends", None)
    meta.path.write_text(json.dumps(data))
    eng3, _r, _o = new_session_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng3, meta) and eng3.dead_ends.items == [] and "DEAD ENDS" not in (eng3.resume_note or "")
    data["engine_state"]["dead_ends"] = ["junk", {"tool": "run"}, {"tool": "run", "sig": "run:a b", "times": "x"}]
    meta.path.write_text(json.dumps(data))
    eng4, _r, _o = new_session_engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng4, meta) and eng4.dead_ends.items == []


def test_dead_ends_show_up_in_the_verified_state_after_the_open_items_and_in_a_fresh_stage(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    looped(eng)
    st = verified.build(eng, run_tests=False)
    assert "DEAD ENDS (already tried, same result - do not repeat)" in st.text and "python3 sim.py" in st.text
    assert st.text.index("OPEN ACCEPTANCE ITEMS") < st.text.index("DEAD ENDS") < st.text.index("FILES")
    assert len(st.sections["DEAD ENDS"]["text"]) <= 600
    eng.begin_stage("do the thing", st.text)
    first = [m["content"] for m in eng.ctx.messages if m["role"] == "user"][0]
    assert "DEAD ENDS" in first and sum("DEAD ENDS" in str(m["content"]) for m in eng.ctx.messages) == 1
    eng.begin_stage("do the thing", "VERIFIED STATE (could not be measured)")
    assert sum("DEAD ENDS" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "system") == 1      # the stage start adds it itself
    assert "DEAD ENDS" not in verified.build(fresh(tmp_path, "q", FakeLLM([])), run_tests=False).text


def test_dead_ends_are_dropped_late_under_the_size_cap(tmp_path):
    sections = [("TESTS", "t" * 100), ("OPEN ITEMS", "o" * 100), ("DEAD ENDS", "d" * 100), ("FILES", "f" * 100), ("SYMBOLS", "s" * 100)]
    kept, dropped = verified.fit(sections, 330)
    assert [n for n, _t in kept] == ["TESTS", "OPEN ITEMS", "DEAD ENDS"] and dropped == ["SYMBOLS", "FILES"]


def test_an_edit_of_the_related_file_clears_the_dead_end(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "sim.py").write_text("print(1)\n")
    looped(eng)
    eng.dead_ends.items[-1]["files"] = ["sim.py"]
    eng._loop_guard("edit_file", {"path": "other.py"}, "edited other.py: 1 edit(s) applied (exact). Syntax OK.")
    assert eng.dead_ends.items                                                     # another file: the ledger stays
    eng._loop_guard("edit_file", {"path": "./sim.py"}, "edited sim.py: 1 edit(s) applied (exact). Syntax OK.")
    assert eng.dead_ends.items == []


def test_a_success_of_the_same_attempt_clears_the_error_dead_end_but_a_repeat_does_not(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    looped(eng)
    eng._loop_guard("run", {"command": "python3 sim.py"}, "exit code 1\nKeyError: 'room'")
    assert eng.dead_ends.items                                                     # failing again is not a success
    eng._loop_guard("run", {"command": "python3 sim.py"}, "exit code 0\nall rooms fine")
    assert eng.dead_ends.items == []
    same = LG.DeadEnds()
    same.record("run", "python3 sim.py", "banner", 3, error=False, files=[], rsha="r1")
    assert same.attempt("run", "python3 sim.py", False, "r1") == 0                # same output: still a dead end
    assert same.attempt("run", "python3 sim.py", False, "r2") == 1                # new output: not any more


# ── the supervisor restart ──────────────────────────────────────────────────

def test_the_restarted_segment_after_a_loop_abort_starts_with_the_dead_end(tmp_path):
    bad = call("run", command="python3 -c 'import nothere_module'")
    other = call("run", command="python3 -c 'import still_missing_module'")
    llm = FakeLLM([bad] * 4 + [other] * 4 + [say("unused")])
    eng, repo, _o = new_session_engine(tmp_path, llm)
    eng.session_id = "sess"
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "autopilot_inhibit": False, "autopilot_max_idle": 2}}
    eng.pol.max_steps, eng.pol.steps_set = 12, True
    sup = SV.Supervisor(eng, clock=Clock(), save=lambda: S.save_session(eng))
    res = sup.run("one small task")
    assert [d["reason"] for d in sup.decisions][:1] == ["loop"] and res.reason == "idle"
    first_of_second = llm.calls[4]["snapshot"]
    text = "\n".join(str(m["content"]) for m in first_of_second)
    assert "DEAD ENDS (already tried, same result - do not repeat)" in text and "import nothere_module" in text


def test_fresh_stage_and_resume_pause_keep_the_ledger(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    looped(eng)
    eng.lookact.pause()
    eng._sigs.clear()                                                             # what the supervisor does at u=1
    eng.begin_stage("goal", "VERIFIED STATE", reset_plan=True)
    assert eng.dead_ends.items and "DEAD ENDS" in "\n".join(systems(eng))




# ── command families and read-only run spam ─────────────────────────────────

def playt(eng, tool, args, result):
    """Like play() for any tool with exact arguments; runs the whole guard like _run_call does."""
    eng.ctx.add({"role": "assistant", "content": json.dumps({"tool": tool, "arguments": args})})
    eng.ctx.add({"role": "tool", "content": result})
    eng._loop_guard(tool, args, result)


def orient(eng, n=6):
    for i in range(n):
        playt(eng, "read_file", {"path": f"orient{i}.py"}, f"text {i}")


def pyc(i):
    return {"command": f"python3 -c \"import re; t = open('tests/test_scripted_playthroughs.py').read(); print(re.findall(r'def test_{i}\\w+', t))\""}


def edit(eng, i=0):
    playt(eng, "edit_file", {"path": f"a{i}.py", "edits": []}, "edited a.py (1 edit)")


def test_family_key_shapes():
    k = LG.family_key("run", pyc(1))
    assert k == LG.family_key("run", pyc(99)) and "tests/test_scripted_playthroughs.py" in k and "import re" in k
    assert LG.family_key("run", {"command": "sed -n '10,40p' src/engine.py"}) == LG.family_key("run", {"command": "sed -n '200,260p' ./src/engine.py"}) == "sed src/engine.py"
    assert LG.family_key("run", {"command": "sed -n 1,5p a.py"}) != LG.family_key("run", {"command": "sed -n 1,5p b.py"})
    assert LG.family_key("run", {"command": "ls"}) is None and LG.family_key("run", {"command": "python3 -c 'print(1)'"}) is None
    assert LG.family_key("search", {"pattern": "x", "path": "src/engine.py"}) == LG.family_key("search", {"pattern": "y", "path": "src/engine.py"})
    assert LG.family_key("search", {"pattern": "x"}) is None and LG.family_key("read_file", {"path": "a.py"}) is None


def test_thirty_python_c_reads_with_changing_output_note_at_5_abort_at_8_and_a_dead_end(tmp_path):
    eng = fresh(tmp_path, "fam")
    orient(eng)
    for i in range(1, 5):
        playt(eng, "run", pyc(i), f"exit code 0\nmatch {i} " + "x" * i)
    assert "Same kind of command" not in last_tool(eng)
    playt(eng, "run", pyc(5), "exit code 0\nmatch 5 xxxxx")
    tail = last_tool(eng)
    assert tail.count("Same kind of command as your last 4 (python3 tests/test_scripted_playthroughs.py") == 1 and "Make an edit now, or try a different approach." in tail
    for i in range(6, 8):
        playt(eng, "run", pyc(i), f"exit code 0\nmatch {i} " + "y" * i)
        assert "Same kind of command" not in last_tool(eng) and not eng._abort_loop           # one note only
    playt(eng, "run", pyc(8), "exit code 0\nmatch 8 zzzzzzzz")
    assert eng._abort_loop == 8 and eng.think_steps == 0
    ev = eng.events.of("loop_abort")[-1]
    assert ev["shape"] == "family" and ev["tool"] == "run"
    assert any("tests/test_scripted_playthroughs.py" in e["cmd"] and e["out"] == "investigated without editing" and e["times"] == 8 for e in eng.dead_ends.items)
    assert not eng.events.of("loop")                                                       # the old ladder never saw equal outcomes


def test_an_edit_resets_the_family_counter(tmp_path):
    eng = fresh(tmp_path, "reset")
    orient(eng)
    for i in range(1, 8):
        playt(eng, "run", pyc(i), f"exit code 0\nm{i} " + "q" * i)
    assert not eng._abort_loop
    edit(eng)
    for i in range(8, 14):                                   # 6 more calls since the edit: well below 8, and the note may come again
        playt(eng, "run", pyc(i), f"exit code 0\nm{i} " + "q" * i)
    assert not eng._abort_loop
    assert "Same kind of command" in "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")


def test_other_reads_do_not_reset_the_family_counter(tmp_path):
    eng = fresh(tmp_path, "noreset")
    orient(eng)
    for i in range(1, 9):
        playt(eng, "run", pyc(i), f"exit code 0\nm{i} " + "q" * i)
        if i < 8:
            playt(eng, "read_file", {"path": f"other{i}.py"}, f"text of other {i}")       # 8 of 15 calls: window of 12 still holds >= 5, then 8
    assert eng._abort_loop in (0, 8)
    # 12-call window with an interleaved read: 6 family calls in the window, so only the note appears, no abort yet
    assert "Same kind of command" in "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")


def test_edit_and_test_cycles_never_trigger(tmp_path):
    eng = fresh(tmp_path, "cycle")
    orient(eng)
    for i in range(30):
        edit(eng, i)
        playt(eng, "run", {"command": "python3 -m pytest -q tests/test_x.py"}, "exit code 0\n3 passed in 0.1s")
        playt(eng, "run", {"command": "sed -n 1,20p tests/test_x.py"}, f"line {i}")
    assert not eng._abort_loop and "Same kind" not in "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")
    assert not eng.events.of("loop_abort")


def test_edit_every_sixth_call_never_counts(tmp_path):
    eng = fresh(tmp_path, "six")
    orient(eng)
    edit(eng, 99)
    for r in range(6):
        for i in range(5):
            playt(eng, "run", pyc(r * 5 + i), f"exit code 0\nout {r} {i} " + "w" * (r * 5 + i))
        edit(eng, r)
    assert not eng._abort_loop and "Same kind" not in "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")


def test_reading_twenty_different_files_is_no_family_but_the_look_guard_still_applies(tmp_path):
    eng = fresh(tmp_path, "look")
    for i in range(20):
        playt(eng, "read_file", {"path": f"src/mod{i}.py"}, f"contents of module {i}")
        eng.lookact.observe("read_file", {"path": f"src/mod{i}.py"}, f"contents of module {i}")
    assert not eng._abort_loop and "Same kind" not in "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")
    assert eng.lookact.blocked("read_file", {"path": "src/mod20.py"}) is not None          # the existing pause is untouched


def test_run_spam_fourteen_sed_reads_of_different_files_without_edits(tmp_path):
    eng = fresh(tmp_path, "spam")
    orient(eng)
    for i in range(1, 10):
        playt(eng, "run", {"command": f"sed -n 1,20p mod{i}.py"}, f"exit code 0\nbody {i}")
    assert not eng._abort_loop and "changed nothing" not in last_tool(eng)
    playt(eng, "run", {"command": "sed -n 1,20p mod10.py"}, "exit code 0\nbody 10")
    assert "10 of your last 10 calls" in last_tool(eng) or "changed nothing" in last_tool(eng)
    for i in range(11, 14):
        playt(eng, "run", {"command": f"sed -n 1,20p mod{i}.py"}, f"exit code 0\nbody {i}")
    assert not eng._abort_loop
    playt(eng, "run", {"command": "sed -n 1,20p mod14.py"}, "exit code 0\nbody 14")
    assert eng._abort_loop == 14 and eng.events.of("loop_abort")[-1]["shape"] == "run_spam"
    assert any(e["out"] == "investigated without editing" for e in eng.dead_ends.items)


def test_run_spam_with_edits_interleaved_does_not_trigger(tmp_path):
    eng = fresh(tmp_path, "spamedit")
    orient(eng)
    for i in range(1, 40):
        playt(eng, "run", {"command": f"sed -n 1,20p mod{i}.py"}, f"exit code 0\nbody {i}")
        if i % 5 == 0:
            edit(eng, i)
    assert not eng._abort_loop and "changed nothing" not in "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")


def test_test_runs_with_a_changing_summary_are_excluded_but_identical_ones_count(tmp_path):
    eng = fresh(tmp_path, "tests")
    orient(eng)
    for i in range(30):
        playt(eng, "run", {"command": "python3 -m pytest -q tests"}, f"exit code 1\nFAILED tests/t.py::t{i} - x\n{i % 2 + 1} failed, {10 + i} passed in 0.{i}s" if i % 2 else f"exit code 0\n{10 + i} passed in 0.{i}s")
    assert not eng._abort_loop                                                              # the verdict changes every time
    eng2 = fresh(tmp_path, "tests2")
    orient(eng2)
    for i in range(14):
        playt(eng2, "run", {"command": "python3 -m pytest -q tests"}, f"exit code 0\n5 passed in 0.{i}s")
    assert eng2._abort_loop in (8, 14) and eng2.events.of("loop_abort")


def test_first_six_steps_of_a_stage_are_lenient(tmp_path):
    eng = fresh(tmp_path, "lenient")
    for i in range(1, 7):
        playt(eng, "run", pyc(i), f"exit code 0\nm{i} " + "r" * i)
    assert not eng._abort_loop and "Same kind" not in "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")
    playt(eng, "run", pyc(7), "exit code 0\nm7 rrrrrrr")                                    # from the 7th call on it counts
    assert "Same kind of command" in last_tool(eng)
    eng.family.reset_stage()                                                                # a new stage starts lenient again
    assert eng.family.stage_calls == 0


def test_family_limits_come_from_the_config(tmp_path):
    eng = fresh(tmp_path, "cfg")
    eng.config = {**eng.config, "coding": {**(eng.config.get("coding") or {}), "family_note": 3, "family_abort": 4}}
    orient(eng)
    for i in range(1, 4):
        playt(eng, "run", pyc(i), f"exit code 0\nm{i} " + "t" * i)
    assert "Same kind of command as your last 2" in last_tool(eng) and not eng._abort_loop
    playt(eng, "run", pyc(4), "exit code 0\nm4 tttt")
    assert eng._abort_loop == 4
    eng3 = fresh(tmp_path, "cfg2")
    eng3.config = {**eng3.config, "coding": {"run_spam_note": 3, "run_spam_abort": 5}}
    orient(eng3)
    for i in range(1, 6):
        playt(eng3, "run", {"command": f"cat f{i}.txt"}, f"exit code 0\nc{i}")
    assert eng3._abort_loop == 5 and eng3.events.of("loop_abort")[-1]["shape"] == "run_spam"


def test_the_new_guards_never_force_thinking(tmp_path):
    eng = fresh(tmp_path, "think")
    orient(eng)
    for i in range(1, 9):
        playt(eng, "run", pyc(i), f"exit code 0\nm{i} " + "u" * i)
    assert eng._abort_loop == 8 and eng.think_steps == 0 and not systems(eng)



def test_rewriting_a_scratch_script_between_identical_runs_is_not_progress(tmp_path):
    """Seen live: write .scratch/scan.py, run it, rewrite it, run it ... 14 times; every scratch write cleared the loop counters."""
    eng = fresh(tmp_path, "scratch")
    for _ in range(6):
        eng.ctx.add({"role": "assistant", "content": "{}"})
        eng.ctx.add({"role": "tool", "content": "wrote .scratch/scan.py (7 lines). Syntax OK."})
        eng._loop_guard("write_file", {"path": ".scratch/scan.py", "content": "x"}, "wrote .scratch/scan.py (7 lines). Syntax OK.")
        play(eng, "python3 .scratch/scan.py", "same output every time")
    assert eng._abort_loop >= 4, "the scratch rewrites must not reset the counters"


def test_a_real_project_edit_still_resets_the_counters(tmp_path):
    eng = fresh(tmp_path, "real")
    for _ in range(3):
        play(eng, "python3 check.py", "same output every time")
    eng.ctx.add({"role": "assistant", "content": "{}"})
    eng.ctx.add({"role": "tool", "content": "edited game.py: 1 edit(s) applied (exact). Syntax OK."})
    eng._loop_guard("edit_file", {"path": "game.py"}, "edited game.py: 1 edit(s) applied (exact). Syntax OK.")
    play(eng, "python3 check.py", "same output every time")
    assert not eng._abort_loop
