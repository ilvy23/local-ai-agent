from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from agent.codeagent import policy as P
from agent.codeagent.engine import CodeEngine
from agent.codeagent.eventlog import EventLog


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}],
            "_prompt_eval_count": 1200, "_eval_count": 30}


def say(text):
    return {"role": "assistant", "content": text, "_prompt_eval_count": 1300, "_eval_count": 40}


class FakeLLM:
    def __init__(self, script, background=None):
        self.script, self.background, self.calls = list(script), list(background or []), []

    def chat_with_tools(self, messages, model, tools, num_ctx=None, think=None, extra_options=None):
        self.calls.append({"n": len(messages), "think": think, "opts": extra_options, "snapshot": [dict(m) for m in messages]})
        step = self.script.pop(0)
        return step(messages) if callable(step) else dict(step)

    def chat(self, messages, model=None, stream=False, background=False, extra_options=None, **kw):
        text = self.background.pop(0) if self.background else "NONE"
        yield text


def no_summary(eng):
    """These tests pin the bare stop (call counts, `res.text is None`): the one tool-less summary turn after a stop is switched off here and
    covered in test_codeagent_finalize.py."""
    eng.config = {**eng.config, "coding": {**(eng.config.get("coding") or {}), "forced_summary": False}}
    return eng


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


CFG = {"models": {"chat": "m", "background": "m", "chat_num_ctx": 32768}, "coding": {"claims_check": False}, "autonomy": {}}


def engine(tmp_path, llm, mode="auto", approve=None, **pol):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    P.save_repo(repo, mode=mode, **pol)
    out = io.StringIO()
    console = Console(file=out, width=120, force_terminal=False)
    eng = CodeEngine(repo, CFG, llm, console, approve_fn=approve, interactive=False, events=EventLog("t", tmp_path / "ev.jsonl"))
    return eng, repo, out


def test_syntax_gate_then_fix_then_verified_answer(tmp_path):
    llm = FakeLLM([
        call("write_file", path="game.py", content="def start(:\n    pass\n"),
        call("edit_file", path="game.py", edits=[{"search": "def start(:", "replace": "def start():"}]),
        say("Created game.py with start()."),
    ])
    eng, repo, out = engine(tmp_path, llm)
    res = eng.turn("make a tiny game module")
    assert res.text == "Created game.py with start()." and res.files == ["game.py"] and res.verified is True
    assert (repo / "game.py").read_text() == "def start():\n    pass\n"
    first_result = [m for m in eng.ctx.messages if m["role"] == "tool"][0]["content"]
    assert "SYNTAX GATE FAILED" in first_result and "line 1" in first_result
    stored_call = json.loads([m for m in eng.ctx.messages if m["role"] == "assistant"][0]["content"])
    assert stored_call["arguments"]["content"] == "def start(:\n    pass\n"       # a recent write keeps its real text (a placeholder got copied 94x);
    # the bulk mask shrinks old ones - see test_recent_writes_keep_their_real_text_old_ones_are_masked
    assert all(c["think"] is False for c in llm.calls)                         # tool steps run without a reasoning pass
    assert all(c["opts"]["num_predict"] == 8192 and c["opts"]["temperature"] == 0.25 for c in llm.calls)
    assert "1 file changed" in out.getvalue()
    assert [r["kind"] for r in eng.events.recent if r["kind"] == "step"]


def test_big_writes_are_always_accepted_with_advice_afterwards(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "big.py").write_text("x = 1\n" * 1000)
    msg = eng.execute("write_file", {"path": "big.py", "content": "y = 2\n"})
    assert msg.startswith("wrote big.py") and "replaced an existing" in msg and "edit_file is cheaper" in msg
    assert (repo / "big.py").read_text() == "y = 2\n"                                  # the generated work is never thrown away
    big = "a = 1\n" * 4000                                                           # 24k chars in ONE call: accepted
    msg = eng.execute("write_file", {"path": "huge.py", "content": big})
    assert msg.startswith("wrote huge.py (4000 lines)") and "large single write accepted" in msg
    assert (repo / "huge.py").read_text() == big
    assert eng.execute("write_file", {"path": "gigantic.py", "content": "z = 1\n" * 30000}).startswith("error:")      # absurd sizes only


def test_policy_ask_decline_and_approve(tmp_path):
    answers = iter([False, True])
    asked = []
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="ask", approve=lambda d, r: asked.append(d) or next(answers))
    assert eng.execute("write_file", {"path": "a.py", "content": "a = 1\n"}).startswith("The user declined")
    assert not (repo / "a.py").exists()
    assert eng.execute("write_file", {"path": "a.py", "content": "a = 1\n"}).startswith("wrote a.py")
    assert len(asked) == 2 and "a.py" in asked[0]
    # reads never ask; paths outside the repo always do
    n = len(asked)
    answers2 = iter([False])
    eng.approve_fn = lambda d, r: asked.append(d) or next(answers2)
    assert "a = 1" in eng.execute("read_file", {"path": "a.py"})
    assert eng.execute("read_file", {"path": "/etc/hostname"}).startswith("The user declined") and len(asked) == n + 1


def test_blocked_and_deny_patterns_never_run(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]), deny=["rm *"])
    assert eng.execute("run", {"command": "rm -rf /"}).startswith("BLOCKED")
    (repo / "keep.txt").write_text("x")
    assert eng.execute("run", {"command": "rm keep.txt"}).startswith("BLOCKED") and (repo / "keep.txt").exists()
    assert "exit code 0" in eng.execute("run", {"command": "echo hi"})


def test_failed_verification_sends_the_model_back_to_work(tmp_path):
    llm = FakeLLM([
        call("write_file", path="app.py", content="VALUE = 1\n"),
        say("done, VALUE is right"),                                           # verification (check.py) fails -> nudge
        call("edit_file", path="app.py", edits=[{"search": "VALUE = 1", "replace": "VALUE = 2"}]),
        say("Fixed: VALUE = 2 and check.py passes."),
    ])
    eng, repo, out = engine(tmp_path, llm, test_cmd="python3 check.py")
    (repo / "check.py").write_text("import app\nassert app.VALUE == 2, 'VALUE must be 2'\n")
    res = eng.turn("make app.py with VALUE equal to 2 and make check.py pass")
    assert res.text.startswith("Fixed") and res.verified is True and "verify" in out.getvalue()
    nudge = [m for m in llm.calls[2]["snapshot"] if m["role"] == "system" and "check" in m["content"].lower()]
    assert nudge and "AssertionError" in nudge[-1]["content"] or "VALUE must be 2" in nudge[-1]["content"]


def test_claims_check_refuses_unproven_statements(tmp_path):
    llm = FakeLLM([
        call("write_file", path="m.py", content="print('hi')\n"),
        say("All endings are reachable and tested."),
        call("run", command="python3 m.py"),
        say("m.py runs and prints hi (ran it)."),
    ], background=["- all endings are reachable and tested (no command exercised them)", "NONE"])
    eng, repo, _ = engine(tmp_path, llm)
    eng.config = {**CFG, "coding": {"claims_check": True}}
    res = eng.turn("write m.py")
    assert res.text.startswith("m.py runs")
    sys_msgs = [m["content"] for m in eng.ctx.messages if m["role"] == "system"]
    assert any("claims things the evidence does not support" in s for s in sys_msgs)


def test_step_budget_and_empty_reply_reroll(tmp_path):
    llm = FakeLLM([say("")] + [call("list_dir", path=f"d{i}") for i in range(10)])
    eng, repo, out = engine(tmp_path, llm, max_steps=5)
    no_summary(eng)
    res = eng.turn("look around")
    assert res.stopped == "budget" and res.steps == 5 and "stopped after 5 steps" in out.getvalue()
    assert llm.script[0:1] and len(llm.calls) == 5                             # the blank reply was re-rolled, not accepted


def test_run_feeds_stdin_and_times_out(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "ask.py").write_text("name = input('who? ')\nprint('hello', name)\n")
    out = eng.execute("run", {"command": "python3 ask.py", "stdin": "alice\n"})
    assert "hello alice" in out and "exit code 0" in out
    (repo / "wait.py").write_text("input()\n")
    assert "exit code 1" in eng.execute("run", {"command": "python3 wait.py"})     # EOF on stdin instead of hanging
    (repo / "slow.py").write_text("import time; time.sleep(5)\n")
    assert "timed out" in eng.execute("run", {"command": "python3 slow.py", "timeout": 1})


def test_tiny_appends_are_cut_off_and_rewrite_loops_get_advice(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 1
    assert eng.execute("append_file", {"path": "c.py", "content": "a = 1"}).startswith("appended")
    assert eng.execute("append_file", {"path": "c.py", "content": "b = 2"}).startswith("appended")
    msg = eng.execute("append_file", {"path": "c.py", "content": "c = 3"})
    assert msg.startswith("refused") and "tiny append" in msg and "ONE bigger chunk" in msg
    assert eng.execute("append_file", {"path": "c.py", "content": "x = 1\n" * 80}).startswith("appended")      # a real chunk resets it
    assert eng.execute("append_file", {"path": "c.py", "content": "d = 4"}).startswith("appended")
    # the same file rewritten again and again
    for i in range(2):
        eng.ctx.step += 1
        assert eng.execute("write_file", {"path": "s.py", "content": f"v = {i}\n"}).startswith("wrote")
    eng.ctx.step += 1
    again = eng.execute("write_file", {"path": "s.py", "content": "v = 9\n"})
    assert again.startswith("wrote s.py") and "rewritten 3 times" in again and "edit_file" in again       # advice, not a refusal


def test_wandering_without_writing_is_called_out(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    for _ in range(10):
        eng._wander_guard("read_file")
    assert [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "spent 10 steps reading" in m["content"]]
    for _ in range(15):
        eng._wander_guard("search")
    nudges = [m for m in eng.ctx.messages if m["role"] == "system" and "steps reading" in m["content"]]
    assert len(nudges) == 2                                                    # at 10 and at 20 steps (25 so far) ...
    for _ in range(10):
        eng._wander_guard("search")
    assert any("Stop re-reading" in m["content"] for m in eng.ctx.messages if m["role"] == "system") and eng.think_steps == 0   # ... then every 10 more, never a forced reasoning pass (4.7 min live)
    eng._wander_guard("run")
    assert eng._look_streak == 0 and eng._wander_nudges == 0


def test_one_session_per_folder(tmp_path, monkeypatch):
    import subprocess
    import sys
    from agent.codeagent import session as S

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "proj"
    repo.mkdir()
    ok, _ = S.acquire_lock(repo)
    assert ok
    code = ("import sys; from pathlib import Path; from agent.codeagent import session as S\n"
            f"ok, owner = S.acquire_lock(Path({str(repo)!r})); print('LOCKED' if not ok else 'FREE', owner)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "XDG_STATE_HOME": str(tmp_path / "state")})
    assert out.stdout.startswith("LOCKED") and str(__import__("os").getpid()) in out.stdout


def test_verification_is_not_repeated_when_the_model_already_ran_it(tmp_path):
    llm = FakeLLM([
        call("write_file", path="app.py", content="VALUE = 2\n"),
        call("run", command="python3 check.py"),
        say("Done: check.py passes."),
    ])
    eng, repo, out = engine(tmp_path, llm, test_cmd="python3 check.py")
    (repo / "check.py").write_text("import app\nassert app.VALUE == 2\n")
    res = eng.turn("make app.py with VALUE = 2")
    assert res.text.startswith("Done") and res.verified is True
    assert out.getvalue().count("verify  python3 check.py") == 0           # no second run of the same command


def test_claims_check_only_runs_when_the_answer_claims_something(tmp_path):
    llm = FakeLLM([call("write_file", path="m.py", content="print('hi')\n"), say("Created m.py.")], background=["- must not be asked"])
    eng, repo, _ = engine(tmp_path, llm)
    eng.config = {**CFG, "coding": {"claims_check": True}}
    res = eng.turn("write m.py")
    assert res.text == "Created m.py." and llm.background == ["- must not be asked"]      # the claims model call was skipped


def test_failed_verification_switches_thinking_on_for_the_fix(tmp_path):
    llm = FakeLLM([
        call("write_file", path="app.py", content="VALUE = 1\n"),
        say("done"),
        call("edit_file", path="app.py", edits=[{"search": "VALUE = 1", "replace": "VALUE = 2"}]),
        say("Fixed."),
    ])
    eng, repo, _ = engine(tmp_path, llm, test_cmd="python3 check.py")
    (repo / "check.py").write_text("import app\nassert app.VALUE == 2\n")
    eng.turn("make app.py satisfy check.py")
    assert [c["think"] for c in llm.calls] == [False, False, True, False]            # exactly the step after the failure reasons


def test_working_status_estimates_tokens_and_verified_commands_become_notes(tmp_path):
    class Notes:
        def __init__(self):
            self.rows = []

        def add(self, key, note, kind="note"):
            self.rows.append(note)
            return True

        def system_message(self, key):
            return None

    llm = FakeLLM([call("write_file", path="game.py", content="print('hi')\n"), call("run", command="python3 game.py"), say("Created game.py.")])
    eng, repo, out = engine(tmp_path, llm)
    eng.notes, eng.note_key = Notes(), "k"
    eng.events.emit("step", n=1, prompt=1000, gen=400, prefill_s=0.5, decode_s=10.0)
    assert 38 < eng.avg_tps() < 42
    from agent.codeagent.engine import _Working

    rendered = "".join(seg.text for seg in Console(width=120, force_terminal=False).render(_Working(eng, "working")))
    assert "working…" in rendered and "tok/s" in rendered and "context 0% of 32k" in rendered
    eng.turn("write game.py and run it")
    assert any("run: python3 game.py" in n for n in eng.notes.rows)


def test_mark_done_accepts_several_items_at_once(tmp_path):
    from agent import plan as plan_mod

    holder = plan_mod.Holder()
    holder.plan = plan_mod.Plan("t", [plan_mod.Item(1, "a"), plan_mod.Item(2, "b"), plan_mod.Item(3, "c")])
    tool = plan_mod.make_tool(holder, Console(file=io.StringIO()))
    res = tool.handler(ids=[1, 2], evidence="python3 check.py printed OK for both")
    assert holder.plan.confirmed() == 2 and "Still open: 3" in res
    assert tool.handler(id=3, evidence="python3 check.py printed OK").startswith("Item 3 confirmed")
    assert tool.handler(evidence="no id given here for sure").startswith("Error")


def test_failed_run_shows_the_source_at_the_traceback(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "main.py").write_text("def run():\n    items = []\n    return items[3]\n\nrun()\n")
    out = eng.execute("run", {"command": "python3 main.py"})
    assert "exit code 1" in out and "IndexError" in out
    assert "SOURCE AT THE TRACEBACK" in out and "main.py:3 in run" in out and "return items[3]" in out


def test_xref_warnings_ride_on_the_write_result(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.execute("write_file", {"path": "engine.py", "content": "class State:\n    def take(self, item):\n        pass\n"})
    out = eng.execute("write_file", {"path": "main.py", "content": "from engine import State\ns = State()\ns.gain_item(1)\n"})
    assert "Syntax OK" in out and "XREF" in out and "gain_item" in out and "take" in out
    good = eng.execute("write_file", {"path": "main2.py", "content": "from engine import State\nState().take(1)\n"})
    assert "XREF" not in good


def test_garbled_tool_calls_are_rescued_not_taken_as_the_final_answer(tmp_path):
    from agent.codeagent.engine import salvage_call

    names = {"read_file", "write_file", "run"}
    mixed = '{"tool": "read_file",\n<parameter=arguments>\n{"path": "locations.py", "start": 380, "end": 471}}\n</parameter>\n<'
    assert salvage_call(mixed, names) == {"name": "read_file", "arguments": {"path": "locations.py", "start": 380, "end": 471}}
    xml = "<function=write_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\nprint(1)\n</parameter>\n</function>"
    assert salvage_call(xml, names) == {"name": "write_file", "arguments": {"path": "a.py", "content": "print(1)"}}
    assert salvage_call('{"tool": "nonexistent", "arguments": {}}', names) is None and salvage_call("just words", names) is None
    llm = FakeLLM([say(mixed), say("Looked at it, nothing to change.")])
    eng, repo, _ = engine(tmp_path, llm)
    (repo / "locations.py").write_text("x = 1\n" * 500)
    res = eng.turn("look at locations.py")
    tool_msgs = [m["content"] for m in eng.ctx.messages if m["role"] == "tool"]
    assert any("locations.py" in t and "lines 380-471" in t for t in tool_msgs)        # the call was executed
    assert res.text == "Looked at it, nothing to change." and [r for r in eng.events.recent if r["kind"] == "salvaged_call"]


def test_unreadable_call_and_blank_loops_are_bounded(tmp_path):
    garbled = "<function=read_file>\n<parameter=path"
    llm = FakeLLM([say(garbled)] * 3 + [say("Final.")])
    eng, repo, out = engine(tmp_path, llm)
    res = eng.turn("anything")
    assert res.text == "Final." and len([r for r in eng.events.recent if r["kind"] == "malformed_call"]) == 3
    notes = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "could not be parsed" in m["content"]]
    assert len(notes) == 1 and not any(garbled in m["content"] for m in eng.ctx.messages if m["role"] == "assistant")   # garbage never enters the history; one reminder per streak
    blank = FakeLLM([say("")] * 12)
    (tmp_path / "b").mkdir()
    eng2, _repo2, out2 = engine(tmp_path / "b", blank)
    no_summary(eng2)
    res2 = eng2.turn("anything")
    assert res2.stopped == "error" and res2.reason == "malformed" and len(blank.calls) == 5 and "unusable replies" in out2.getvalue()
    assert len([m for m in eng2.ctx.messages if m["role"] == "system" and "reply was empty" in m["content"]]) == 1


def test_repeated_errors_and_cycles_trigger_the_loop_guard(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 10
    for _ in range(3):
        eng._loop_guard("edit_file", {"path": "a.py"}, "Error: invalid arguments for tool 'edit_file': argument 'edits' must be array, got string.")
    notes = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "repeating yourself" in m["content"]]
    assert len(notes) == 1 and "real JSON array" in notes[0] and eng.think_steps == 1       # one bounded reasoning step, nothing more
    eng.ctx.step += 1
    for _ in range(3):                                                        # not again right away
        eng._loop_guard("edit_file", {"path": "a.py"}, "Error: invalid arguments for tool 'edit_file': argument 'edits' must be array, got string.")
    assert len([m for m in eng.ctx.messages if "repeating yourself" in str(m["content"])]) == 1
    eng2, _r, _o = engine(tmp_path / "c", FakeLLM([])) if (tmp_path / "c").mkdir() is None else (None, None, None)
    eng2.ctx.step = 30
    for _ in range(3):                                                        # a cycle with changing arguments
        eng2._loop_guard("read_file", {"path": "d.py", "start": 1}, "dialogue.py (579 lines) ...")
        eng2._loop_guard("edit_file", {"path": "d.py"}, "Error: invalid arguments for tool 'edit_file': argument 'edits' must be array")
    assert any("repeating yourself" in str(m["content"]) for m in eng2.ctx.messages) and [r for r in eng2.events.recent if r["kind"] == "loop"]
    eng3, _r3, _o3 = engine(tmp_path / "d", FakeLLM([])) if (tmp_path / "d").mkdir() is None else (None, None, None)
    for _ in range(5):                                                        # successful repeats (reading) are not an error loop
        eng3._loop_guard("read_file", {"path": f"f{_}.py"}, "file contents here")
    assert not any("repeating yourself" in str(m["content"]) for m in eng3.ctx.messages)


def test_reading_a_lot_without_writing_gets_a_nudge_once(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "a.py").write_text("x = 1\n" * 300)
    for i in range(1, 4):
        eng.execute("read_file", {"path": "a.py", "start": 1 + i, "end": 300})
        eng._wander_guard("read_file")
    nudges = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "without writing or running" in m["content"]]
    assert len(nudges) == 1 and "write the first piece now" in nudges[0]
    eng.execute("write_file", {"path": "b.py", "content": "y = 2\n"})
    eng._wander_guard("write_file")
    assert eng.tools.read_lines == 0


def test_without_a_terminal_asks_are_decided_at_once_and_never_repeated(tmp_path, monkeypatch):
    import sys as _sys
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False, raising=False)
    eng, repo, out = engine(tmp_path, FakeLLM([]), mode="auto")
    eng._explicit_approve = False                                              # the real CLI: nobody to ask
    cmd = {"command": "cd /srv/luke/repos/Projects && python3 -c 'print(1)'"}
    first = eng.execute("run", cmd)
    assert first.startswith("NOT RUN") and "no one" not in first and str(repo) in first and "relative paths" in first
    again = eng.execute("run", cmd)
    assert again.startswith("NOT RUN (asked before)") and "Do not send it again" in again
    for _ in range(3):
        eng._loop_guard("run", {"command": "x"}, again)
    assert any("repeating yourself" in str(m["content"]) for m in eng.ctx.messages)        # and the loop guard sees it as an error loop
    assert "exit code 0" in eng.execute("run", {"command": "echo ok"})                   # normal commands are unaffected


def test_the_project_block_states_the_repo_root(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.start()
    sysmsg = eng.ctx.messages[0]["content"]
    assert f"root: {repo.resolve()}" in sysmsg and "never `cd` somewhere else" in sysmsg and "never repeat it" in sysmsg


def test_identical_successful_calls_are_a_loop_too_and_end_the_turn(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 5
    same = {"pattern": "thin|low|base"}
    for _ in range(3):
        eng._loop_guard("search", same, "1:# locations.py - HARBOUR: the six places")
    notes = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "identical `search` call" in m["content"]]
    assert len(notes) == 1 and "already in your context" in notes[0]
    for _ in range(5):
        eng._loop_guard("search", same, "1:# locations.py - HARBOUR: the six places")
    assert getattr(eng, "_abort_loop", False) and [r for r in eng.events.recent if r["kind"] == "loop_abort"]
    eng2, _r, _o = engine(tmp_path / "x", FakeLLM([])) if (tmp_path / "x").mkdir() is None else (None, None, None)
    for i in range(10):                                                         # different arguments each time: never a loop
        eng2._loop_guard("search", {"pattern": f"p{i}"}, "1:# same result text")
    assert not getattr(eng2, "_abort_loop", False)


def test_a_looping_model_stops_the_turn_after_four_identical_calls(tmp_path):
    llm = FakeLLM([call("list_dir", path=".")] * 14)
    eng, repo, out = engine(tmp_path, llm)
    no_summary(eng)
    (repo / "a.py").write_text("x = 1\n")
    res = eng.turn("look around")
    assert res.stopped == "loop" and "repeated the same call 4 times" in out.getvalue() and len(llm.calls) == 4


def test_the_history_keeps_edit_calls_in_a_shape_the_model_can_safely_imitate(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="x = 1\n"),
                   call("edit_file", path="a.py", edits=[{"search": "x = 1", "replace": "x = 2"}]),
                   call("edit_file", path="a.py", search="x = 2", replace="x = 3"), say("Done: a.py has x = 3.")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.turn("make x equal 3")
    assert (repo / "a.py").read_text() == "x = 3\n"
    stored = [json.loads(m["content"]) for m in eng.ctx.messages if m["role"] == "assistant" and m["content"].startswith("{")]
    list_edit = next(c for c in stored if c["tool"] == "edit_file" and "edits" in c["arguments"])
    assert isinstance(list_edit["arguments"]["edits"], list) and list_edit["arguments"]["edits"][0] == {"search": "x = 1", "replace": "x = 2"}
    flat_edit = next(c for c in stored if c["tool"] == "edit_file" and "search" in c["arguments"])
    assert flat_edit["arguments"]["replace"] == "x = 3"


def test_apply_patch_accepts_a_path_and_codex_style_blocks(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "m.py").write_text("def f():\n    return 1\n")
    assert eng.execute("apply_patch", {"patch": "@@\n def f():\n-    return 1\n+    return 2\n", "path": "m.py"}).startswith("patched 1 file")
    assert "return 2" in (repo / "m.py").read_text()
    codex = "*** Begin Patch\n*** Update File: m.py\n@@\n def f():\n-    return 2\n+    return 3\n*** End Patch"
    assert eng.execute("apply_patch", {"patch": codex}).startswith("patched 1 file") and "return 3" in (repo / "m.py").read_text()
    err = eng.execute("apply_patch", {"patch": "[12 lines of diff applied]"})
    assert "no files found" in err and "edit_file is simpler" in err


def test_thinking_steps_have_a_hard_budget_and_a_wasted_one_switches_thinking_off(tmp_path):
    from agent.codeagent.engine import NUM_PREDICT, THINK_PREDICT

    llm = FakeLLM([say(""), call("list_dir", path="."), say("Done.")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.think_steps = 2                                                       # e.g. after a loop warning
    eng.turn("look")
    assert [c["think"] for c in llm.calls] == [True, False, False]            # the empty thinking step cost one step, then no more thinking
    assert llm.calls[0]["opts"]["num_predict"] == THINK_PREDICT and THINK_PREDICT < NUM_PREDICT
    assert llm.calls[1]["opts"]["num_predict"] == NUM_PREDICT


def test_identical_successful_run_gets_a_run_specific_hint(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 10
    for _ in range(3):
        eng._loop_guard("run", {"command": "printf x | python main.py"}, "exit code 0\nsame output")
    last = [m["content"] for m in eng.ctx.messages if m["role"] == "system"][-1]
    assert "not doing what you expect" in last and "small python script" in last and eng.think_steps == 1


def test_loop_guard_collapses_the_repeated_pairs_it_flags(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 10
    call = {"role": "assistant", "content": '{"tool": "run", "arguments": {"command": "x"}}'}
    for _ in range(3):
        eng.ctx.add(dict(call))
        eng.ctx.add({"role": "tool", "content": "exit code 0\nsame"})
        eng._loop_guard("run", {"command": "x"}, "exit code 0\nsame")
    calls = [m for m in eng.ctx.messages if m["role"] == "assistant" and "run" in m["content"]]
    assert len(calls) == 1 and sum(m["role"] == "tool" for m in eng.ctx.messages) == 1
    assert eng.ctx.messages[-1]["role"] == "system"


def test_on_step_hook_runs_every_step(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([{"role": "assistant", "content": "done"}]))
    calls = []
    eng.on_step = lambda: calls.append(eng.ctx.step)
    eng.turn("say hi")
    assert calls and calls[0] == 1


def test_search_resent_with_a_counting_argument_is_a_loop(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 10
    for n in range(9, 19):
        args = {"pattern": "a|b|c", "path": "locations.py", "context": n}
        eng.ctx.add({"role": "assistant", "content": json.dumps({"tool": "search", "arguments": args})})
        eng.ctx.add({"role": "tool", "content": "23- same long result\n26: x"})
        eng._loop_guard("search", args, "23- same long result\n26: x")
        if n == 11:
            assert [r for r in eng.events.recent if r["kind"] == "loop"]
            assert sum(m["role"] == "tool" for m in eng.ctx.messages) == 1
    assert eng._abort_loop


def test_long_run_of_tests_without_an_edit_gets_one_note_and_a_thinking_step(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 5
    for i in range(14):
        eng._edit_gap_guard("run", {"command": f"python3 main.py {i}"}, "exit code 0")
    assert not [m for m in eng.ctx.messages if "without editing" in str(m["content"])] and eng.think_steps == 0
    eng._edit_gap_guard("run", {"command": "python3 main.py x"}, "exit code 0")
    assert len([m for m in eng.ctx.messages if "without editing" in str(m["content"])]) == 1 and eng.think_steps == 1
    eng._edit_gap_guard("write_file", {"path": ".scratch/drive.py"}, "wrote .scratch/drive.py")      # a scratch write is not progress
    assert eng._edit_gap == 16
    eng._edit_gap_guard("edit_file", {"path": "main.py"}, "edited main.py")                           # a real edit is
    assert eng._edit_gap == 0 and eng._edit_nudges == 0


def test_reading_before_the_first_run_is_not_an_edit_gap(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    for i in range(20):
        eng._edit_gap_guard("read_file", {"path": f"m{i}.py"}, "x = 1")             # exploring: the wander guard's business
    assert eng._edit_gap == 0 and not [m for m in eng.ctx.messages if "without editing" in str(m["content"])]
    eng._edit_gap_guard("run", {"command": "python3 main.py"}, "exit code 0")
    eng._edit_gap_guard("read_file", {"path": "m1.py"}, "x = 1")                    # probing now: reads between runs count
    assert eng._edit_gap == 2


def test_writes_that_fail_the_syntax_gate_do_not_reset_the_loop_count(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 10
    args = {"path": "d.py", "content": "[1 lines, 33 chars - now on disk]"}
    for _ in range(10):
        eng.ctx.add({"role": "assistant", "content": json.dumps({"tool": "append_file", "arguments": args})})
        eng.ctx.add({"role": "tool", "content": "appended to d.py (162 lines total)  ✗ syntax"})
        eng._loop_guard("append_file", args, "appended to d.py (162 lines total)  ✗ syntax")
        if eng._abort_loop:
            break
    assert eng._abort_loop


def test_reading_is_paused_after_a_long_read_streak_until_the_model_changes_something(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    (repo / "a.py").write_text("x = 1\n")
    assert "x = 1" in eng.execute("read_file", {"path": "a.py"})
    eng._look_streak = 25
    res = eng.execute("read_file", {"path": "a.py"})
    assert res.startswith("NOT RUN") and "reading is paused" in res and [r for r in eng.events.recent if r["kind"] == "look_blocked"]
    assert eng.execute("search", {"pattern": "x"}).startswith("NOT RUN")
    assert "wrote" in eng.execute("write_file", {"path": "b.py", "content": "y = 2\n"})
    eng._wander_guard("write_file")                                              # a change resets the streak
    assert not eng.execute("read_file", {"path": "b.py"}).startswith("NOT RUN")


def test_read_only_shell_commands_count_as_looking_not_as_acting(tmp_path):
    from agent.codeagent.engine import readonly_shell

    assert readonly_shell("sed -n '60,252p' locations.py") and readonly_shell("cat a.py | head -40") and readonly_shell("grep -n foo *.py | wc -l")
    assert not readonly_shell("python3 main.py") and not readonly_shell("sed -i 's/a/b/' x.py") and not readonly_shell("cat a > b")
    assert not readonly_shell("printf 'x' | python3 main.py | grep y") and not readonly_shell("")
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    (repo / "a.py").write_text("x = 1\n")
    eng._look_streak = 25
    assert eng.execute("run", {"command": "sed -n '1,5p' a.py"}).startswith("NOT RUN")
    assert not eng.execute("run", {"command": "python3 a.py"}).startswith("NOT RUN")


# ── shapes a Qwen3.x reply can take when Ollama's parser does not turn it into message.tool_calls ──
_SALVAGE_NAMES = {"read_file", "run", "write_file", "edit_file"}
_SALVAGE_SCHEMA = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "start": {"type": "integer"}, "outline": {"type": "boolean"}, "paths": {"type": "array", "items": {"type": "string"}}}}}}]


@pytest.mark.parametrize("text, want", [
    ('<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py", "start": 3}}\n</tool_call>', {"path": "a.py", "start": 3}),
    ('```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```', {"path": "a.py"}),
    ('{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\", \\"start\\": 3}"}', {"path": "a.py", "start": 3}),
    ('{"name": "read_file", "parameters": {"path": "a.py"}}', {"path": "a.py"}),
    ('{"name": "read_file"}', {}),
])
def test_salvage_call_reads_the_hermes_name_arguments_shape(text, want):
    from agent.codeagent.engine import salvage_call

    assert salvage_call(text, _SALVAGE_NAMES) == {"name": "read_file", "arguments": want}


def test_salvage_call_ignores_json_that_is_not_a_known_tool():
    from agent.codeagent.engine import salvage_call

    assert salvage_call('{"name": "Alice", "arguments": {"x": 1}}', _SALVAGE_NAMES) is None
    assert salvage_call('Use it like: {"name": "run", "arguments": {"command": "ls"}} - that is the syntax.', {"read_file"}) is None


def test_salvage_call_types_xml_parameters_by_schema():
    from agent.codeagent.engine import salvage_call

    xml = "<function=read_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=start>\n3\n</parameter>\n<parameter=outline>\ntrue\n</parameter>\n" \
          "<parameter=paths>\n[\"a.py\", \"b.py\"]\n</parameter>\n</function>"
    assert salvage_call(xml, _SALVAGE_NAMES, _SALVAGE_SCHEMA)["arguments"] == {"path": "a.py", "start": 3, "outline": True, "paths": ["a.py", "b.py"]}
    assert salvage_call(xml, _SALVAGE_NAMES)["arguments"]["start"] == "3"                      # without a schema nothing is guessed
