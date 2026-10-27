"""Long-horizon behaviour: announced-but-not-done next steps, the self-extending step budget, compaction written in place
(cache-friendly) that carries the live acceptance list and session notes, and masks that are only paid when worth it."""

from __future__ import annotations

import json

import pytest

from agent import plan as plan_mod
from agent.codeagent import context as C
from agent.codeagent.engine import announced_step
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_context import Summarizer, fill
from tests.test_codeagent_engine import FakeLLM, call, engine, say  # noqa: F401  (engine fixture helpers)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


@pytest.mark.parametrize("text", [
    "Wrote models.py and storage.py.\n\nNow update main.py to use the new storage layer ...",
    "The parser works. Next, I'll add the CLI tests:",
    "Done with stage 2. Let me now wire the commands into cli.py.",
    "**Now I will implement the report module.**",
    "Jetzt passe ich main.py an.",
    "- Moving on to stage 3: the exporter.",
])
def test_announced_own_next_steps_are_recognised(text):
    assert announced_step(text)


@pytest.mark.parametrize("text", [
    "All done: tests pass (12 passed).",
    "Next, you could add a config file if you want.",
    "Should I now add the exporter?",
    "Now everything is in place and verified with pytest.",
    "I cannot run the tests: the command needs your approval.",
    "Let me know if you want more features.",
    "Next steps (optional): a GUI.",
])
def test_reports_suggestions_and_questions_are_not_announcements(text):
    assert announced_step(text) is None


def plan_of(*texts):
    return plan_mod.Plan(task="t", items=[plan_mod.Item(i, t) for i, t in enumerate(texts, 1)])


def test_an_announced_step_with_open_plan_items_continues_instead_of_ending(tmp_path):
    llm = FakeLLM([call("write_file", path="models.py", content="X = 1\n"),
                   say("Wrote models.py.\nNow update main.py to import it ..."),
                   call("write_file", path="main.py", content="import models\nprint(models.X)\n"),
                   call("mark_done", ids=[1, 2], evidence="python3 main.py -> prints 1 (exit 0)"),
                   say("models.py and main.py are written; main.py prints 1.")])
    eng, repo, out = engine(tmp_path, llm)
    eng.plan.plan = plan_of("models.py defines X", "main.py prints X")
    res = eng.turn("build it")
    assert res.text.startswith("models.py and main.py") and (repo / "main.py").exists()
    notes = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "and stopped without doing it" in m["content"]]
    assert len(notes) == 1 and "Next open acceptance item: 1." in notes[0]
    assert [e for e in eng.events.recent if e["kind"] == "announce_continue"]


def test_announcements_are_bounded_and_never_back_to_back(tmp_path):
    llm = FakeLLM([say("Now update main.py ..."), say("Now update main.py ..."), say("Now update main.py ...")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.turn("x")
    assert sum("and stopped without doing it" in m["content"] for m in eng.ctx.messages) == 1    # the 2nd came without a tool step
    eng2, _r, _o = engine(tmp_path, FakeLLM([say("Here is the summary. Next, I'll keep it as is")]))
    eng2.turn("y")
    assert not [m for m in eng2.ctx.messages if "and stopped without doing it" in m["content"]]   # no plan and no trailing ":" / "..."


def test_the_step_budget_extends_itself_while_the_run_makes_progress(tmp_path):
    writes = [call("write_file", path=f"m{i}.py", content=f"X{i} = {i}\n") for i in range(8)]
    eng, repo, out = engine(tmp_path, FakeLLM([*writes, say("eight modules written")]), max_steps=5)
    res = eng.turn("write eight modules")
    assert res.stopped == "" and res.steps == 9 and "still making progress" in out.getvalue()
    assert [e for e in eng.events.recent if e["kind"] == "budget_extend"]


def test_without_progress_the_budget_stops_and_says_how_to_continue(tmp_path):
    eng, repo, out = engine(tmp_path, FakeLLM([call("list_dir", path=f"d{i}") for i in range(12)]), max_steps=5)
    res = eng.turn("look")
    text = " ".join(out.getvalue().split())
    assert res.stopped == "budget" and res.steps == 5 and "without progress" in text and "nightage code --resume continue" in text


def test_scratch_writes_do_not_count_as_progress(tmp_path):
    writes = [call("write_file", path=f".scratch/t{i}.py", content="x = 1\n") for i in range(8)]
    eng, repo, _ = engine(tmp_path, FakeLLM(writes), max_steps=5)
    assert eng.turn("probe").stopped == "budget"


# ── compaction ──────────────────────────────────────────────────────────────

def cm(tmp_path, ctx=10000, llm=None):
    return C.ContextManager(llm or Summarizer(), {"models": {"background": "m"}}, ctx, EventLog("t", tmp_path / "e.jsonl"), tmp_path)


def test_compaction_written_in_place_skips_the_separate_summary_call_and_keeps_the_plan(tmp_path):
    bg = Summarizer()
    m = cm(tmp_path, llm=bg)
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 12)
    seen = []
    m.summarize_in_place = lambda instruction: seen.append(instruction) or "DONE: a\nSTATE: b\nOPEN: c\nPITFALLS: none"
    m.extras = lambda: "ACCEPTANCE LIST NOW (1 of 2 confirmed):\n1. [x] a\n2. [ ] b"
    assert m.compact()
    assert bg.calls == [] and "Do NOT call a tool" in seen[0] and "2. [ ] b" in seen[0]
    log = next(x for x in m.messages if x["content"].startswith("WORK LOG"))["content"]
    assert "DONE: a" in log and "2. [ ] b" in log and "LEDGER" in log
    assert [e for e in m.events.recent if e["kind"] == "compact"][0]["via"] == "in_place"


def test_in_place_compaction_falls_back_to_the_summary_call(tmp_path):
    bg = Summarizer()
    m = cm(tmp_path, llm=bg)
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 12)
    m.summarize_in_place = lambda instruction: None
    m.extras = lambda: "PROJECT NOTES SAVED IN THIS SESSION:\n- run: python3 main.py"
    assert m.compact() and len(bg.calls) == 1 and "python3 main.py" in bg.calls[0][1]["content"]
    assert [e for e in m.events.recent if e["kind"] == "compact"][0]["via"] == "background"


def test_engine_writes_the_log_on_its_own_conversation(tmp_path):
    log = "DONE: wrote a.py\nSTATE: ok\nDECISIONS: none\nOPEN: tests\nPITFALLS: none"
    llm = FakeLLM([{"role": "assistant", "content": log, "_prompt_eval_count": 9000, "_eval_count": 200, "_prompt_eval_ns": 3e8},
                   call("read_file", path="a.py")])
    eng, repo, _ = engine(tmp_path, llm)
    eng.start()
    eng.ctx.add_user("task")
    before = [dict(m) for m in eng.ctx.messages]
    assert eng._summarize_in_place("[harness] write the log") == log
    sent = llm.calls[0]["snapshot"]
    assert sent[:-1] == before and sent[-1]["content"] == "[harness] write the log"            # same prefix as the last step
    assert llm.calls[0]["opts"]["num_predict"] == 3072 and llm.calls[0]["think"] is False
    assert eng._summarize_in_place("again") is None                                            # a tool call is not a log
    assert [e["ok"] for e in eng.events.recent if e["kind"] == "compact_call"] == [True, False]


def test_compaction_extras_carry_the_live_plan_and_session_notes(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.plan.plan = plan_of("a works", "b works")
    eng.plan.plan.mark_done(1, "python3 a.py -> exit 0")
    eng.session_notes.append("tests: pytest -q  (verified working)")
    x = eng._compaction_extras()
    assert "1 of 2 confirmed" in x and "1. [x] a works" in x and "2. [ ] b works" in x and "pytest -q" in x


def test_a_small_mask_is_skipped_until_compaction(tmp_path):
    m = cm(tmp_path, ctx=100000)
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 14, size=1500)                                   # ~20k chars of old output: ~6k tokens = 6% of the window
    m.last_prompt, m._since = 74000, 0
    before = [dict(x) for x in m.messages]
    m.maintain()
    assert m.messages == before and m.mask_floor == C.COMPACT_AT * 100000      # prompt untouched (server cache kept)
    assert [e for e in m.events.recent if e["kind"] == "mask_skipped"] and not [e for e in m.events.recent if e["kind"] == "mask"]


def test_a_failed_compaction_is_not_retried_every_step(tmp_path):
    class Broken:
        calls = 0

        def chat(self, *a, **k):
            Broken.calls += 1
            raise OSError("down")

    m = cm(tmp_path, ctx=100000, llm=Broken())
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 12, size=600)
    m.mask_floor = 10 ** 9
    for _ in range(3):
        m.last_prompt, m._since = 82000, 0
        m.maintain()
    assert Broken.calls == 1
    m.last_prompt = 88000                                    # 6% more: try again
    m.maintain()
    assert Broken.calls == 2


def test_plan_and_notes_survive_a_resume_and_a_compaction_afterwards(tmp_path, monkeypatch):
    from agent.codeagent import session as S, sessionstore as SS

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    eng, repo, _ = engine(tmp_path, FakeLLM([say("ok")]))
    eng.plan.plan = plan_of("a works", "b works")
    eng.plan.plan.mark_done(1, "python3 a.py -> exit 0")
    eng.session_notes.append("run: python3 a.py  (verified working)")
    eng.session_id = SS.new_id()
    eng.turn("start")
    S.save_session(eng)
    eng2, _r, _o = engine(tmp_path, FakeLLM([]))
    S.resume_into(eng2, SS.latest(eng.root))
    assert eng2._compaction_extras() == eng._compaction_extras()
    assert "open: 2. b works" in eng2.resume_note
    assert json.loads(json.dumps(eng2.ctx.messages)) == eng.ctx.messages


def test_a_new_big_task_after_a_finished_plan_gets_its_own_list(tmp_path):
    big = "Extend the package with stages: " + "; ".join(f"stage {i}: write module m{i}.py with tests" for i in range(6))
    eng, repo, _ = engine(tmp_path, FakeLLM([say("ok"), say("ok")], background=['["m0 exists", "m1 exists"]', '["x", "y"]']))
    eng.plan.plan = plan_of("old a", "old b")
    eng.plan.plan.mark_done(1, "python3 a.py -> exit 0")
    eng.turn(big)
    assert [i.text for i in eng.plan.plan.items] == ["old a", "old b"]          # still open items: kept
    eng.plan.plan.mark_done(2, "python3 b.py -> exit 0")
    eng.turn(big)
    assert [i.text for i in eng.plan.plan.items] == ["m0 exists", "m1 exists"]


def test_a_log_stuck_in_the_head_is_merged_by_the_next_compaction(tmp_path):
    m = cm(tmp_path)
    m.add({"role": "system", "content": "sys"})
    m.add({"role": "system", "content": "WORK LOG (auto-compacted from 9 earlier steps):\nDONE: stage one\n\nLEDGER (x):\n- a.py"})
    m.add_user("second task")
    fill(m, 12)
    seen = []
    m.summarize_in_place = lambda instruction: seen.append(1) or "DONE: stage one, stage two\nSTATE: ok\nOPEN: three\nPITFALLS: -"
    assert m.compact()
    logs = [x for x in m.messages if x["content"].startswith("WORK LOG")]
    assert len(logs) == 1 and "stage two" in logs[0]["content"]
    assert [x["role"] for x in m.messages[:3]] == ["system", "user", "system"] and m.messages[1]["content"] == "second task"
    assert "EARLIER REQUESTS" not in logs[0]["content"]                       # the request is still in the head verbatim


def test_small_emergency_trims_wait_for_the_edge(tmp_path):
    m = cm(tmp_path, ctx=100000, llm=Summarizer(""))       # compaction returns nothing
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 8, size=600)                                     # little to trim
    m.mask_floor = 10 ** 9
    m.last_prompt, m._since = 88000, 0
    before = [dict(x) for x in m.messages]
    m.maintain()
    assert m.messages == before
    m.last_prompt, m._since = 96000, 0
    m._compact_retry_at = 0
    m.maintain()
    assert m.messages != before and [e for e in m.events.recent if e["kind"] == "emergency_mask"]


def test_in_place_log_is_not_tried_without_room_to_write_it(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.start()
    eng.ctx.last_prompt = eng.num_ctx - 600
    assert eng._summarize_in_place("[harness] write the log") is None and eng._in_place_failures == 0
