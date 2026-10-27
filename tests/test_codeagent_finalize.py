"""Phase-aware finalization: the verification-spiral phase (files written, nothing failing: the guard notes ask for the final answer and a post-edit
budget ends the tool use) and the honest forced ending (a stop without a model answer gets ONE tool-less summary turn behind a factual harness line).
Replay of the measured shape: one write, then a dozen near-identical smoke-test heredocs. FakeLLM only, no model."""

from __future__ import annotations

from agent.codeagent import loopguard as LG, phase as PH

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_hardening import engine
from tests.test_codeagent_loopguard import edit, last_tool, orient, playt, systems
from tests.test_codeagent_supervisor import make as make_supervisor


class SummaryLLM(FakeLLM):
    """A tool-less call (tools == []) is answered from `summary` (or raises when that entry is an exception); every call's tools are recorded."""

    def __init__(self, script, summary=None):
        super().__init__(script)
        self.summary, self.summaries = list(summary if summary is not None else []), []

    def chat_with_tools(self, messages, model, tools, num_ctx=None, think=None, extra_options=None):
        if tools == []:
            self.summaries.append({"messages": [dict(m) for m in messages], "think": think, "opts": extra_options})
            out = self.summary.pop(0)
            if isinstance(out, Exception):
                raise out
            return dict(out) if isinstance(out, dict) else say(out)
        return super().chat_with_tools(messages, model, tools, num_ctx=num_ctx, think=think, extra_options=extra_options)


WORDS = ["alpha frame drawn", "bravo snake moved", "charlie food eaten", "delta wall hit", "echo score shown", "foxtrot input read", "golf timer fired",
         "hotel border drawn", "india tail grown", "juliet menu shown", "kilo level up", "lima game over"]
SNAKE = "import curses\n\n\ndef main(scr):\n    scr.addstr(0, 0, 'snake')\n\n\nif __name__ == '__main__':\n    curses.wrapper(main)\n"


def smoke(i, same=False):
    """One of the near-identical heredoc smoke tests; with `same` the output is identical every time."""
    word = "all frames drawn, nothing to see" if same else WORDS[i % len(WORDS)]
    return call("run", command=f"python3 - <<'EOF'\nimport sys, curses  # fake screen {i}\nprint('{word}')\nEOF")


def shape(n, same=False):
    """The measured shape: one write, then n runs (whose OUTPUT differs unless `same`, so only the new guards can see them)."""
    return [call("write_file", path="snake.py", content=SNAKE), *[smoke(i, same) for i in range(n)]]


def cfg(eng, **keys):
    eng.config = {**eng.config, "coding": {**(eng.config.get("coding") or {}), **keys}}


def stored_assistant(eng):
    return [m["content"] for m in eng.ctx.messages if m["role"] == "assistant" and not m["content"].lstrip().startswith("{")]


PREFIX = "[stopped: verify_budget - the post-edit verification budget was used up - summary below was written after the stop; verified facts:"


# ── the verification phase ──────────────────────────────────────────────────

def test_replay_the_budget_ends_the_tool_use_after_six_and_one_toolless_turn_writes_the_summary(tmp_path):
    llm = SummaryLLM(shape(12), summary=["Built snake.py (curses snake game). I could not run the curses screen; only smoke tests of the logic ran."])
    eng, repo, out = engine(tmp_path, llm)
    res = eng.turn("Make a snake game.")
    assert res.stopped == "verify_budget" and res.steps == 1 + 6 and len(llm.calls) == 7 and len(llm.script) == 6      # six runs after the edit, then no more tool use
    assert len(llm.summaries) == 1                                                        # exactly one summary turn
    s = llm.summaries[0]
    assert s["think"] is False and s["opts"]["num_predict"] <= 400                       # tools disabled (== [] in the fake), thinking off, <= 400 tokens
    last = s["messages"][-1]
    assert last["role"] == "user" and last["content"].endswith(PH.INSTRUCTION) and "snake.py" in last["content"]
    assert res.text.startswith(PREFIX) and "Built snake.py" in res.text and res.files == ["snake.py"]
    assert "snake.py" in res.text.split("\n")[0] and "NOT VERIFIED" in res.text.split("\n")[0]          # measured facts: the file, and what no command proved
    assert "summary below was written after the stop" in out.getvalue() and "ending the tool use" in out.getvalue()
    assert eng.events.of("verify_budget") and eng.events.of("forced_summary")[-1]["ok"] is True


def test_the_stored_assistant_text_has_no_harness_line(tmp_path):
    llm = SummaryLLM(shape(12), summary=["Built snake.py. Not run on a real terminal."])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("Make a snake game.")
    assert stored_assistant(eng)[-1] == "Built snake.py. Not run on a real terminal." and res.text != stored_assistant(eng)[-1]
    assert not any("[stopped:" in str(m.get("content")) for m in eng.ctx.messages)


def test_phase_note_replaces_try_a_different_approach_and_forces_no_reasoning(tmp_path):
    eng = engine(tmp_path, FakeLLM([]))[0]
    orient(eng)
    edit(eng)                                                                          # a real edit
    out = "exit code 0\nall frames drawn, nothing to see"
    for i in range(1, 3):
        playt(eng, "run", {"command": f"python3 -c \"import snake; print({i})\""}, out)
    assert not systems(eng) and LG.SAME_RESULT_NOTE in last_tool(eng)                   # 2 clean runs: not a phase yet, the plain repeat note
    playt(eng, "run", {"command": "python3 -c \"import snake; print(3)\""}, out)       # the 3rd: phase + repeat nudge
    assert systems(eng) == [LG.PHASE_NOTE] and "different approach" not in systems(eng)[0] and "different" not in systems(eng)[0]
    assert "final answer now" in systems(eng)[0] and eng.think_steps == 0 and not eng._abort_loop
    assert eng.events.of("loop")[-1]["phase"] is True


def test_family_and_pattern_notes_use_the_phase_text_too(tmp_path):
    eng = engine(tmp_path, FakeLLM([]))[0]
    cfg(eng, verify_phase_budget=0)                                                    # no budget: only the notes are looked at
    orient(eng)
    edit(eng)
    for i in range(1, 9):
        playt(eng, "search", {"pattern": f"symbol{i}"}, f"hits for {i}: " + "m" * i)
    tail = last_tool(eng)
    assert "`search` 8 times in a row" not in tail and tail.count(LG.PHASE_NOTE) == 1      # the same-tool note was replaced
    assert "different approach" not in " ".join(m["content"] for m in eng.ctx.messages if m["role"] == "tool")
    assert eng.think_steps == 0


def test_replay_with_identical_output_loop_abort_ends_with_the_summary_in_one_shot(tmp_path):
    llm = SummaryLLM(shape(12, same=True), summary=["Built snake.py. Not verified: the curses screen."])
    eng, _repo, out = engine(tmp_path, llm)
    res = eng.turn("Make a snake game.")
    assert res.stopped == "loop" and len(llm.summaries) == 1 and res.text is not None
    assert res.text.startswith("[stopped: loop - the same kind of call repeated without progress - summary below") and "Built snake.py" in res.text
    assert any(m["role"] == "system" and m["content"] == LG.PHASE_NOTE for m in llm.calls[-1]["snapshot"])      # the model was told to answer before the abort
    assert "different approach" not in " ".join(str(m["content"]) for m in llm.calls[-1]["snapshot"] if m["role"] == "system")


def test_a_spiral_without_a_prior_edit_is_not_the_verification_phase(tmp_path):
    eng = engine(tmp_path, FakeLLM([]))[0]
    orient(eng)
    for i in range(1, 4):
        playt(eng, "run", {"command": f"python3 -c \"import snake; print({i})\""}, "exit code 0\nall frames drawn, nothing to see")
    assert not eng.phase.edited and not eng._phase_active()
    assert len(systems(eng)) == 1 and "Stop re-sending it" in systems(eng)[0] and LG.PHASE_NOTE not in systems(eng)[0]      # the ordinary loop nudge
    assert eng.think_steps == 1                                                           # and its one reasoning pass, as before
    for i in range(10):
        playt(eng, "run", {"command": f"echo {i}"}, f"exit code 0\nline {i}")
    assert not eng.phase.ended


def test_a_spiral_without_an_edit_has_no_budget_end(tmp_path):
    llm = SummaryLLM([smoke(i) for i in range(6)] + [say("Looked around, changed nothing.")])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("what does this repo do")
    assert res.stopped == "" and res.text == "Looked around, changed nothing." and not llm.summaries


def test_a_failure_signal_resets_the_phase(tmp_path):
    eng = engine(tmp_path, FakeLLM([]))[0]
    orient(eng)
    edit(eng)
    for i in range(1, 4):
        playt(eng, "run", {"command": f"python3 -c 'import a; print({i})'"}, f"exit code 0\nline {i} " + "z" * 10 * i)
    assert eng._phase_active() and eng.phase.clean == 3
    playt(eng, "run", {"command": "python3 snake.py"}, "exit code 1\nTraceback (most recent call last):\n  File \"snake.py\", line 1\nNameError: x")
    assert not eng._phase_active() and eng.phase.clean == 0 and eng.phase.edited                  # error shape / non-zero exit
    for i in range(3):
        playt(eng, "run", {"command": f"python3 -c 'import a; print({i + 10})'"}, f"exit code 0\nrow {i} " + "q" * 10 * (i + 1))
    assert eng._phase_active()
    playt(eng, "run", {"command": "python3 -m pytest -q"}, "exit code 0\n1 failed, 3 passed in 0.2s")
    assert eng.phase.clean == 0                                                              # a red test, even behind exit 0
    playt(eng, "write_file", {"path": "x.py", "content": "def f(:\n"}, "wrote x.py (1 lines)\nSYNTAX GATE FAILED line 1")
    assert eng.phase.clean == 0 and not eng.phase.ended


def test_budget_counts_from_the_last_failure_and_from_the_last_edit(tmp_path):
    eng = engine(tmp_path, FakeLLM([]))[0]
    orient(eng)
    edit(eng)
    n = [0]

    def clean():
        n[0] += 1
        playt(eng, "run", {"command": f"python3 -c 'import a; print({n[0]})'"}, f"exit code 0\nstep {n[0]} " + "w" * 7 * n[0])

    for _ in range(5):
        clean()
    playt(eng, "run", {"command": "python3 snake.py"}, "exit code 1\nNameError: name 'x' is not defined")            # reset at 5 of 6
    for _ in range(5):
        clean()
    assert not eng.phase.ended and eng.phase.clean == 5
    edit(eng, 1)                                                                          # a new real edit resets as well
    assert eng.phase.clean == 0 and eng.phase.edited
    for _ in range(5):
        clean()
    assert not eng.phase.ended
    clean()
    assert eng.phase.ended and eng.phase.clean == 6


def test_scratch_writes_and_changed_files_are_activity_not_clean_calls(tmp_path):
    eng = engine(tmp_path, FakeLLM([]))[0]
    orient(eng)
    playt(eng, "write_file", {"path": ".scratch/t.py", "content": "x = 1\n"}, "wrote .scratch/t.py (1 lines)")
    assert not eng.phase.edited                                                           # a scratch write is no real edit: no phase
    edit(eng)
    for i in range(1, 3):
        playt(eng, "run", {"command": f"python3 -c 'print({i})'"}, f"exit code 0\nline {i} " + "k" * 9 * i)
    playt(eng, "write_file", {"path": ".scratch/t.py", "content": "x = 2\n"}, "wrote .scratch/t.py (1 lines)")
    assert eng.phase.clean == 0


def test_budget_zero_and_min_are_configurable(tmp_path):
    llm = SummaryLLM(shape(12), summary=["done"])
    eng, _repo, _ = engine(tmp_path, llm)
    cfg(eng, verify_phase_budget=3)
    res = eng.turn("Make a snake game.")
    assert res.stopped == "verify_budget" and res.steps == 1 + 3
    llm2 = SummaryLLM(shape(7) + [say("Done: snake.py.")])
    (tmp_path / "b").mkdir()
    eng2 = engine(tmp_path / "b", llm2)[0]
    cfg(eng2, verify_phase_budget=0)
    res2 = eng2.turn("Make a snake game.")
    assert res2.stopped == "" and res2.text == "Done: snake.py." and not llm2.summaries


# ── the honest forced ending ────────────────────────────────────────────────

def test_fallback_mechanical_summary_when_the_summary_call_raises(tmp_path):
    llm = SummaryLLM(shape(12), summary=[RuntimeError("model gone")])
    eng, _repo, out = engine(tmp_path, llm)
    res = eng.turn("Make a snake game.")
    assert res.stopped == "verify_budget" and len(llm.summaries) == 1
    body = res.text.split("\n\n", 1)[1]
    assert res.text.startswith(PREFIX) and "Files changed: snake.py." in body and "Not verified:" in body and "no test run since the last edit" in body
    assert stored_assistant(eng)[-1] == body and eng.events.of("forced_summary")[-1]["ok"] is False
    assert "RuntimeError" in eng.events.of("forced_summary")[-1]["error"]


def test_an_unusable_summary_answer_falls_back_too(tmp_path):
    tool_call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run", "arguments": {"command": "ls"}}}]}
    llm = SummaryLLM(shape(12), summary=[tool_call])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("Make a snake game.")
    assert "Work stopped before a final answer" in res.text and len(llm.summaries) == 1


def test_forced_summary_can_be_switched_off(tmp_path):
    llm = SummaryLLM(shape(12), summary=["never asked"])
    eng, _repo, _ = engine(tmp_path, llm)
    cfg(eng, forced_summary=False)
    res = eng.turn("Make a snake game.")
    assert res.stopped == "verify_budget" and res.text is None and not llm.summaries


def test_a_normal_answer_gets_no_summary_turn_and_no_prefix(tmp_path):
    llm = SummaryLLM([call("write_file", path="snake.py", content=SNAKE), smoke(0), say("Built snake.py and smoke-tested it.")])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("Make a snake game.")
    assert res.stopped == "" and res.text == "Built snake.py and smoke-tested it." and not llm.summaries


def test_the_step_budget_in_a_repl_prompt_ends_with_a_summary_and_unchanged_stop(tmp_path):
    llm = SummaryLLM([call("read_file", path=f"f{i}.txt") for i in range(5)], summary=["Only looked at the files; nothing changed."])
    eng, repo, out = engine(tmp_path, llm, max_steps=5)
    for i in range(5):
        (repo / f"f{i}.txt").write_text(f"file number {i} " * (i + 1))
    res = eng.turn("what is here")
    assert res.stopped == "budget" and res.text.startswith("[stopped: budget - the step budget was used up") and len(llm.summaries) == 1
    assert "Files changed: none" not in res.text.split("\n\n", 1)[1] and "files changed: none" in res.text.split("\n")[0]


def test_malformed_abort_gets_the_summary_too(tmp_path):
    llm = SummaryLLM([say('{"tool": "run", "arguments": ') for _ in range(8)], summary=["Nothing was done."])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("do it")
    assert res.stopped == "error" and res.reason == "malformed" and res.text.startswith("[stopped: error/malformed") and len(llm.summaries) == 1


def test_an_interrupted_or_dead_server_prompt_gets_no_summary(tmp_path):
    def boom(messages):
        raise KeyboardInterrupt

    llm = SummaryLLM([boom], summary=["never"])
    eng, _repo, _ = engine(tmp_path, llm)
    res = eng.turn("x")
    assert res.stopped == "interrupted" and res.text is None and not llm.summaries


def test_the_summary_call_is_the_only_extra_call_and_the_conversation_is_reused(tmp_path):
    llm = SummaryLLM(shape(12), summary=["ok"])
    eng, _repo, _ = engine(tmp_path, llm)
    eng.turn("Make a snake game.")
    s = llm.summaries[0]["messages"]
    assert s[0]["role"] == "system" and any(m["role"] == "tool" for m in s) and len(llm.calls) == 7             # the history rides along; no other extra call
    assert all(c["think"] is False for c in llm.calls)                                                         # no reasoning pass from the guards


# ── autopilot / stage runs stay as they were ────────────────────────────────

def test_a_stage_run_gets_no_summary_and_stops_as_verify_budget(tmp_path):
    llm = SummaryLLM(shape(12), summary=["must not be asked"])
    eng, _repo, out = engine(tmp_path, llm)
    eng.stage_goal = "stage 1 goal"                                                      # what begin_stage sets: the supervisor / runner owns the restart
    res = eng.turn("stage 1 goal")
    assert res.stopped == "verify_budget" and res.text is None and not llm.summaries and "[stopped:" not in out.getvalue()
    assert len(llm.calls) == 7


def test_the_supervisor_treats_verify_budget_like_a_loop_stop(tmp_path):
    sup, sess, runner, _c = make_supervisor(tmp_path, {1: ["verify_budget", "verify_budget", "ok"]}, stages=1)
    res = sup.run("t")
    assert res.verdict == "done" and [d["reason"] for d in res.decisions] == ["verify_budget", "verify_budget"]
    assert [d["action"] for d in res.decisions] == ["resume", "resume_pause"] and sup._loop_stops[1] == 2


def test_phase_note_in_a_stage_run_replaces_the_text_too(tmp_path):
    eng = engine(tmp_path, FakeLLM([]))[0]
    eng.stage_goal = "g"
    orient(eng)
    edit(eng)
    for i in range(1, 4):
        playt(eng, "run", {"command": f"python3 -c \"import snake; print({i})\""}, "exit code 0\nall frames drawn, nothing to see")
    assert systems(eng) == [LG.PHASE_NOTE]
