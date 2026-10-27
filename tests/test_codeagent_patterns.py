"""Loop detection, third generation (ideas of qwen-code's loopDetectionService, own implementation): the same tool with changing arguments,
alternating call patterns, re-reads inside a window, and repetition inside ONE streamed generation (content and thinking). Every
scenario below was invisible to the older guards (exact / near-duplicate with an equal outcome, kind of command, 16 looks in a row,
a block repeated over 8000 chars). FakeLLM only, no model."""

from __future__ import annotations

import hashlib
import json

from agent.codeagent import loopguard as LG
from agent.codeagent.engine import repeating
from agent.llm import GenerationAborted

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_hardening import StreamLLM, engine
from tests.test_codeagent_loopguard import edit, fresh, last_tool, orient, playt, pyc, systems


def cfg(eng, **keys):
    eng.config = {**eng.config, "coding": {**(eng.config.get("coding") or {}), **keys}}


def srch(eng, i):
    playt(eng, "search", {"pattern": f"symbol{i}"}, f"hits for {i}: " + "m" * i)


def events(eng):
    return eng.events.of("loop_abort")


# ── same tool, changing arguments ───────────────────────────────────────────

def test_twelve_searches_with_changing_patterns_note_at_8_abort_at_12_and_a_dead_end(tmp_path):
    eng = fresh(tmp_path, "same")
    orient(eng)
    for i in range(1, 8):
        srch(eng, i)
        assert not eng._abort_loop and "in a row" not in last_tool(eng)
    srch(eng, 8)
    assert last_tool(eng).count("`search` 8 times in a row") == 1 and "Make an edit now" in last_tool(eng)
    for i in range(9, 12):
        srch(eng, i)
        assert "in a row" not in last_tool(eng) and not eng._abort_loop            # one note only
    srch(eng, 12)
    assert eng._abort_loop == 12 and eng.think_steps == 0
    ev = events(eng)[-1]
    assert ev["shape"] == "same_tool" and ev["tool"] == "search"
    assert any(e["tool"] == "search" and e["out"] == "investigated without editing" and e["times"] == 12 for e in eng.dead_ends.items)
    assert not eng.events.of("loop")                                              # the older ladder never saw an equal outcome


def test_run_with_changing_commands_and_no_family_is_the_same_tool(tmp_path):
    eng = fresh(tmp_path, "runs")
    orient(eng)
    for i in range(1, 12):
        playt(eng, "run", {"command": f"curl -s localhost:80{i}/ping"}, f"exit code 0\npong {i}")
        assert not eng._abort_loop
    playt(eng, "run", {"command": "curl -s localhost:8012/ping"}, "exit code 0\npong 12")
    assert eng._abort_loop == 12 and events(eng)[-1]["shape"] == "same_tool"


def test_twelve_reads_of_different_files_never_count(tmp_path):
    eng = fresh(tmp_path, "distinct")
    orient(eng)
    for i in range(1, 30):
        playt(eng, "read_file", {"path": f"src/mod{i}.py"}, f"text of module {i}")
    assert not eng._abort_loop and "in a row" not in last_tool(eng) and not events(eng)


def test_paging_through_one_file_with_distinct_ranges_never_counts():
    g = LG.PatternGuard()
    for i in range(6):
        g.observe("read_file", {"path": f"w{i}.py"}, "x", changed=False)
    out = []
    for i in range(30):
        out += g.observe("read_file", {"path": "big.py", "start": 1 + 100 * i, "end": 100 * (i + 1)}, f"part {i}", changed=False)
    assert not out


def test_rereading_the_same_file_with_overlapping_ranges_counts_per_tool_and_path():
    g = LG.PatternGuard()
    for i in range(6):
        g.observe("read_file", {"path": f"w{i}.py"}, "x", changed=False)
    kinds = []
    for i in range(14):
        kinds += [(a, s) for a, s, *_ in g.observe("read_file", {"path": "big.py", "start": 1, "end": 100 + i}, f"part {i}", changed=False,
                                                 read_note=99, read_abort=99)]
    assert ("note", "same_tool") in kinds and ("abort", "same_tool") in kinds


def test_a_failing_write_streak_counts_but_successful_edits_never_do(tmp_path):
    eng = fresh(tmp_path, "writes")
    orient(eng)
    for i in range(12):
        playt(eng, "edit_file", {"path": f"m{i}.py", "edits": [{"old": f"x{i}", "new": "y"}]}, f"edit_file: old text {i} not found")
    assert eng._abort_loop == 12 and events(eng)[-1]["shape"] == "same_tool"
    eng2 = fresh(tmp_path, "writes_ok")
    orient(eng2)
    for i in range(20):
        edit(eng2, i)
    for i in range(20):                                                           # scratch writes succeed too and are no progress, but no streak either
        playt(eng2, "write_file", {"path": f".scratch/t{i}.py", "content": "x"}, f"wrote .scratch/t{i}.py (1 line). Syntax OK.")
    assert not eng2._abort_loop and "in a row" not in last_tool(eng2)


def test_a_changing_test_summary_is_information_not_circling():
    g = LG.PatternGuard()
    for i in range(6):
        g.observe("read_file", {"path": f"w{i}.py"}, "x", changed=False)
    out = []
    for i in range(30):
        out += g.observe("run", {"command": f"pytest -k case{i}"}, f"{30 - i} failed, {i} passed in 1.0s", changed=False)
    assert not out


def test_an_edit_resets_the_same_tool_count(tmp_path):
    eng = fresh(tmp_path, "reset")
    orient(eng)
    for i in range(1, 8):
        srch(eng, i)
    edit(eng)
    for i in range(20, 27):
        srch(eng, i)
    assert not eng._abort_loop and "in a row" not in last_tool(eng)


def test_the_first_six_steps_of_a_stage_are_lenient_for_the_new_guards(tmp_path):
    eng = fresh(tmp_path, "lenient")
    cfg(eng, same_tool_note=3, same_tool_abort=50)
    for i in range(1, 6):
        srch(eng, i)
    assert not eng._abort_loop and "in a row" not in last_tool(eng)
    srch(eng, 6)
    assert not eng._abort_loop
    srch(eng, 7)
    assert "in a row" in last_tool(eng)                                           # stage call 7: the guard is live


def test_same_tool_limits_come_from_the_config(tmp_path):
    eng = fresh(tmp_path, "cfg")
    cfg(eng, same_tool_note=3, same_tool_abort=5)
    orient(eng)
    for i in range(1, 4):
        srch(eng, i)
    assert "`search` 3 times in a row" in last_tool(eng) and not eng._abort_loop
    srch(eng, 4)
    srch(eng, 5)
    assert eng._abort_loop == 5 and events(eng)[-1]["shape"] == "same_tool"


def test_the_new_guards_never_force_thinking(tmp_path):
    eng = fresh(tmp_path, "think")
    orient(eng)
    for i in range(1, 13):
        srch(eng, i)
    assert eng._abort_loop == 12 and eng.think_steps == 0
    assert not [m for m in systems(eng) if "in a row" in m]                      # notes are appended to the result, never a system message


# ── the real shape: python3 -c "import lena, engine, context, dialogue" ────────

def imp(i):
    return {"command": f"python3 -c \"import lena, engine, context, dialogue; print(lena.run({i}), dialogue.line({i * 3}))\""}


def test_replay_fourteen_import_probes_note_once_by_the_family_guard_and_stop_at_8(tmp_path):
    eng = fresh(tmp_path, "replay")
    orient(eng)
    for i in range(1, 8):
        playt(eng, "run", imp(i), f"exit code 0\nline {i} " + "o" * i)
    assert last_tool(eng).count("Same kind of command") == 0
    assert not any("times in a row" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "tool")
    playt(eng, "run", imp(8), "exit code 0\nline 8 oooooooo")
    assert eng._abort_loop == 8 and events(eng)[-1]["shape"] == "family"
    assert sum("Same kind of command" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "tool") == 1
    assert not any("times in a row" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "tool")        # one voice, not two


def test_replay_with_the_family_abort_raised_the_same_tool_stays_silent(tmp_path):
    """The probes are read-only shell commands: the family guard owns them, so the same-tool streak does not count them at all."""
    eng = fresh(tmp_path, "replay2")
    cfg(eng, family_abort=40, run_spam_note=90, run_spam_abort=99)
    orient(eng)
    for i in range(1, 15):
        playt(eng, "run", imp(i), f"exit code 0\nline {i} " + "o" * i)
    tools = [str(m["content"]) for m in eng.ctx.messages if m["role"] == "tool"]
    assert sum("Same kind of command" in t for t in tools) == 1
    assert sum("times in a row" in t for t in tools) == 0 and not eng._abort_loop and not events(eng)


# ── alternating patterns ────────────────────────────────────────────────────

def ab(eng, i, third=False):
    playt(eng, "where_used", {"name": "State"}, f"uses {i} " + "a" * i)
    playt(eng, "list_dir", {"path": "."}, f"entries {i} " + "b" * i)
    if third:
        playt(eng, "outline", {"path": "game.py"}, f"defs {i} " + "c" * i)


def test_a_b_a_b_changing_outputs_notes_at_three_cycles_and_stops_at_four(tmp_path):
    eng = fresh(tmp_path, "abab")
    orient(eng)
    for i in range(1, 3):
        ab(eng, i)
    assert "alternating" not in last_tool(eng)
    ab(eng, 3)
    assert last_tool(eng).count("alternating between 2 calls") == 1 and not eng._abort_loop
    playt(eng, "where_used", {"name": "State"}, "uses 4 aaaa")
    assert "alternating" not in last_tool(eng)
    playt(eng, "list_dir", {"path": "."}, "entries 4 bbbb")                      # the 8th call of the pattern: four rounds
    assert eng._abort_loop == 4 and events(eng)[-1]["shape"] == "alternating" and eng.think_steps == 0
    assert any(e["out"] == "investigated without editing" and "where_used" in e["cmd"] and "list_dir" in e["cmd"] for e in eng.dead_ends.items)


def test_a_b_c_a_b_c_is_found_over_nine_calls(tmp_path):
    eng = fresh(tmp_path, "abcabc")
    orient(eng)
    for i in range(1, 3):
        ab(eng, i, third=True)
    assert "alternating" not in last_tool(eng)
    ab(eng, 3, third=True)
    assert "alternating between 3 calls" in last_tool(eng)
    ab(eng, 4, third=True)
    assert eng._abort_loop and events(eng)[-1]["shape"] == "alternating"


def test_alternating_between_a_test_run_and_an_edit_never_triggers(tmp_path):
    eng = fresh(tmp_path, "cycle")
    orient(eng)
    for i in range(15):
        playt(eng, "run_tests", {}, f"exit code 1\nFAILED tests/test_a.py::t{i} - assert 0\n1 failed in 1.0s")
        edit(eng, i)
    assert not eng._abort_loop and "alternating" not in last_tool(eng)


def test_two_cycles_and_a_third_call_in_between_do_not_count(tmp_path):
    eng = fresh(tmp_path, "two")
    orient(eng)
    for i in range(1, 3):
        ab(eng, i)
    playt(eng, "repo_map", {}, "map")
    for i in range(3, 5):
        ab(eng, i)
    assert not eng._abort_loop and "alternating" not in last_tool(eng)


def test_alternating_limits_come_from_the_config(tmp_path):
    eng = fresh(tmp_path, "altcfg")
    cfg(eng, alt_note=2, alt_abort=3)
    orient(eng)
    ab(eng, 1)
    ab(eng, 2)
    assert "alternating between 2 calls" in last_tool(eng)
    ab(eng, 3)
    assert eng._abort_loop and events(eng)[-1]["shape"] == "alternating"


# ── re-reads inside a window ────────────────────────────────────────────────

def uniq(i, tag=""):
    """A result that shares nothing with the others (the ladder's fuzzy same-result check must not match)."""
    return tag + hashlib.sha256(str(i).encode()).hexdigest() * 3


def rr(eng, i):
    playt(eng, "read_file", {"path": "game.py", "start": 1, "end": 100 + i}, uniq(i, "r"))
    playt(eng, "run", {"command": f"python3 main.py {i}"}, "exit code 0\n" + uniq(i, "s"))


def test_rereads_between_runs_are_a_read_window_note_then_stop(tmp_path):
    """Seen in measured runs: 229 reads against 32 edits. The 16-looks-in-a-row block never sees it: every run resets that streak."""
    eng = fresh(tmp_path, "window")
    orient(eng)
    rr(eng, 0)
    for i in range(1, 4):
        rr(eng, i)
    tools = [str(m["content"]) for m in eng.ctx.messages if m["role"] == "tool"]
    assert sum("already have" in t for t in tools) == 0
    rr(eng, 4)
    tools = [str(m["content"]) for m in eng.ctx.messages if m["role"] == "tool"]
    assert sum("already have" in t for t in tools) == 1 and not eng._abort_loop
    for i in range(5, 10):
        rr(eng, i)
    assert eng._abort_loop and events(eng)[-1]["shape"] == "read_window" and eng.think_steps == 0
    assert sum("already have" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "tool") == 1


def test_exploring_many_distinct_files_between_runs_is_never_a_read_window(tmp_path):
    eng = fresh(tmp_path, "explore")
    orient(eng)
    for i in range(1, 25):
        playt(eng, "read_file", {"path": f"src/m{i}.py", "start": 1, "end": 200}, uniq(i, "r"))
        playt(eng, "run", {"command": f"python3 main.py {i}"}, "exit code 0\n" + uniq(i, "s"))
        if i % 6 == 0:
            edit(eng, i)                                                          # real work in between
    assert not eng._abort_loop and not events(eng)


def test_a_reread_after_an_edit_is_legitimate(tmp_path):
    eng = fresh(tmp_path, "afteredit")
    orient(eng)
    for i in range(1, 5):
        rr(eng, i)
        edit(eng, i)
    assert not eng._abort_loop and not events(eng)


def test_read_window_limits_come_from_the_config(tmp_path):
    eng = fresh(tmp_path, "readcfg")
    cfg(eng, read_repeat_note=2, read_repeat_abort=3)
    orient(eng)
    for i in range(0, 5):
        rr(eng, i)
        if eng._abort_loop:
            break
    assert eng._abort_loop and events(eng)[-1]["shape"] == "read_window"


# ── streaming repetition ────────────────────────────────────────────────────

def feed_all(det, text, size=7):
    """Feed `text` piece by piece; the number of chars fed when the detector fired, else None."""
    fed = 0
    for i in range(0, len(text), size):
        fed += len(text[i:i + size])
        if det.feed(text[i:i + size]):
            return fed
    return None


def test_thinking_that_chants_a_short_sentence_is_cut_long_before_8000_chars():
    text = "Hmm. " + "Wait, let me reconsider the approach. " * 200
    fed = feed_all(LG.ChantDetector("thinking"), text)
    assert fed is not None and fed < 900
    assert not repeating(text[:fed])                                              # the old detector needs 8000 chars


def test_a_long_unit_repeated_five_times_is_periodic_and_cut_early():
    unit = " ".join(f"step{(i * 7) % 53}={i * 13 % 91}" for i in range(40))[:300] + "\n"
    text = "plan:\n" + unit * 30
    det = LG.ChantDetector("thinking")
    fed = feed_all(det, text)
    assert fed is not None and fed < 2200 and not repeating(text[:fed])


def test_a_900_char_paragraph_three_times_in_thinking_but_not_twice():
    para = " ".join(f"consider option {i} because reason {(i * 31) % 17} matters" for i in range(40))[:900] + "\n"
    assert feed_all(LG.ChantDetector("thinking"), "start\n" + para * 2 + "then something else entirely follows here. " * 3) is None
    assert feed_all(LG.ChantDetector("thinking"), "start\n" + para * 3) is not None


def test_content_is_judged_more_leniently_than_thinking():
    line = "        self.assertTrue(result.ok)\n"
    assert feed_all(LG.ChantDetector("content"), "def t():\n" + line * 11) is None
    assert feed_all(LG.ChantDetector("content"), "def t():\n" + line * 30) is not None
    assert feed_all(LG.ChantDetector("thinking"), "def t():\n" + line * 12) is not None


def test_legitimate_long_generations_never_fire():
    prose = " ".join(f"Step {i}: the handler for room {i} returns state {i * 7} after {i % 5} moves." for i in range(120))
    table = "".join(f'    ("room_{i}", "a cold room", {i}, {i % 7}),\n' for i in range(300))
    code = "".join(f"def handler_{i}(state):\n    return state.step({i}, '{i * 7}')\n\n" for i in range(200))
    dividers = ("# " + "=" * 70 + "\n# section\n") * 12
    zeros = "grid = [\n" + "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n" * 40 + "]\n"
    for kind in ("content", "thinking"):
        for text in (prose, table, code, dividers) + ((zeros,) if kind == "content" else ()):      # twelve identical zero rows in a THOUGHT is a chant
            assert feed_all(LG.ChantDetector(kind), text) is None, (kind, text[:30])


def test_hits_can_be_overridden():
    line = "        self.assertTrue(result.ok)\n"
    assert feed_all(LG.ChantDetector("content", hits=5), "def t():\n" + line * 8) is not None


class ThinkLLM(FakeLLM):
    """Streams the thinking text of a scripted step piece by piece, then the reply."""

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, think=None, extra_options=None):
        self.calls.append({"n": len(messages), "think": think, "opts": extra_options, "snapshot": [dict(m) for m in messages]})
        step = dict(self.script.pop(0))
        text, got = step.get("thinking", ""), []
        try:
            for i in range(0, len(text), 40):
                got.append(text[i:i + 40])
                on_thinking(text[i:i + 40])
        except GenerationAborted as stop:
            return {"role": "assistant", "content": "", "thinking": "".join(got), "_aborted": str(stop)}
        return {k: v for k, v in step.items() if k != "thinking"}


def test_engine_stops_a_looping_thought_early_and_rerolls(tmp_path):
    loop = "Wait, let me reconsider the approach. " * 120                          # ~4500 chars: under the old 8000-char span
    llm_ = ThinkLLM([{**say("never used"), "thinking": loop}, call("list_dir", path="."), say("Listed the folder.")])
    eng, repo, out = engine(tmp_path, llm_)
    res = eng.turn("look")
    ev = [r for r in eng.events.recent if r["kind"] == "degenerate"]
    assert res.text == "Listed the folder." and len(ev) == 1 and ev[0]["reason"] == "thinking repeats itself" and ev[0]["pieces"] < 40


def test_engine_stops_a_looping_reply_far_earlier_than_before(tmp_path):
    loop = '{"tool": "write_file", "arguments": {"path": "a.py", "content": "' + "x = compute(a, b)\\n" * 3000
    llm_ = StreamLLM([say(loop), call("list_dir", path="."), say("Listed the folder.")])
    eng, repo, out = engine(tmp_path, llm_)
    res = eng.turn("look")
    ev = [r for r in eng.events.recent if r["kind"] == "degenerate"]
    assert res.text == "Listed the folder." and len(ev) == 1 and ev[0]["pieces"] < 60       # the 8000-char detector needed ~200 pieces here


def test_chant_detection_can_be_switched_off(tmp_path):
    loop = "Wait, let me reconsider the approach. " * 120
    llm_ = ThinkLLM([{**say("fine"), "thinking": loop}])
    eng, repo, out = engine(tmp_path, llm_)
    cfg(eng, chant_detect=False)
    eng.turn("look")
    assert not [r for r in eng.events.recent if r["kind"] == "degenerate"]


def test_read_only_shell_commands_stay_with_run_spam_and_are_neutral_for_same_tool(tmp_path):
    eng = fresh(tmp_path, "sed")
    orient(eng)
    for i in range(1, 12):
        playt(eng, "run", {"command": f"sed -n 1,20p mod{i}.py"}, f"exit code 0\nbody {i} " + uniq(i))
    assert not eng._abort_loop and not events(eng) and "in a row" not in last_tool(eng)
