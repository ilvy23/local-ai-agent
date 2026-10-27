"""CodeRunner: worker thread, event flow, controls, approval bridge, plan-first and steer re-queue on a fake-LLM engine (no real model)."""

from __future__ import annotations

import threading
import time

from agent.ui.activity import ActivityRegistry
from agent.ui.code_runner import CodeRunner
from agent.ui.events import EventBus
from tests._screen_code import render
from tests.test_codeagent_engine import FakeLLM, _isolated, call, engine, say  # noqa: F401 - _isolated is autouse
from tests.test_codeagent_stages import TASK3, new_engine, three_stage_script


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def runner(eng, **kw):
    reg, bus = ActivityRegistry(), EventBus()
    kw.setdefault("autosave", False)
    return CodeRunner(eng, registry=reg, bus=bus, **kw), reg, bus


def test_events_flow_activity_and_snapshot(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="x = 1\n"), say("Wrote a.py.")])
    eng, repo, _ = engine(tmp_path, llm)
    r, reg, bus = runner(eng)
    r.start("make a.py", staged=False)
    assert r.activity is not None and r.activity.kind == "code" and r.activity.stop and r.activity.pause and r.activity.resume
    assert r.join(5)
    kinds = [e.kind for e in bus.recent(n=500)]
    assert {"tool_start", "tool_result", "edit", "done"} <= set(kinds)
    assert all(e.source in ("code", "code") for e in r.events)
    assert r.result["text"] == "Wrote a.py." and r.result["files"] == ["a.py"] and r.result["exit_code"] == 0
    act = reg.get(r.activity.id)
    assert act.state == "done" and act.ended is not None and not act.active
    snap = r.snapshot()
    assert snap["state"] == "idle" and snap["model"] and snap["step"] >= 1
    assert r.panel.file_rows() == [{"path": "a.py", "added": 1, "removed": 0, "created": True}]
    # the panel followed along
    out = render(r.panel.render(width=80), 80)
    assert "DONE" in out and "a.py" in out
    r.detach()


def test_busy_rejects_second_start_and_followup_prompt(tmp_path):
    gate = threading.Event()

    def slow(messages):
        gate.wait(5)
        return say("one")

    eng, _, _ = engine(tmp_path, FakeLLM([slow, say("two")]))
    r, _, _ = runner(eng)
    r.start("first", staged=False)
    assert wait_for(lambda: r.busy)
    try:
        r.start("second", staged=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    gate.set()
    assert r.join(5) and r.result["text"] == "one"
    r.start("follow up", staged=False)
    assert r.join(5) and r.result["text"] == "two"


def test_say_reaches_next_model_call(tmp_path):
    holder = {}

    def first(messages):
        holder["r"].say("also add a README")
        return call("list_dir", path=".")

    llm = FakeLLM([first, say("done")])
    eng, _, _ = engine(tmp_path, llm)
    r, _, _ = runner(eng)
    holder["r"] = r
    r.start("do it", staged=False)
    assert r.join(5)
    assert any("README" in str(m["content"]) for m in llm.calls[1]["snapshot"])
    assert r.requeued == [] and "steer" in [e.kind for e in r.events]


def test_pending_steer_is_requeued_when_the_run_ends(tmp_path):
    holder = {}

    def last(messages):
        holder["r"].say("one more thing")        # arrives after the final model call: the engine never sees it
        return say("finished")

    got = []
    eng, _, _ = engine(tmp_path, FakeLLM([last]))
    r, _, bus = runner(eng, on_requeue=got.append)
    holder["r"] = r
    r.start("go", staged=False)
    assert r.join(5)
    assert got == ["one more thing"] == r.requeued and r.result["requeued"] == "one more thing"
    assert "steer_requeued" in [e.kind for e in bus.recent(n=500)] and eng.live.queued() == 0


def test_compact_request_fires(tmp_path):
    steps = [call("write_file", path=f"f{i}.txt", content=f"file {i}\n") for i in range(12)]
    holder = {}

    def ask(messages):
        holder["r"].compact()
        return call("write_file", path="ask.txt", content="x\n")

    llm = FakeLLM([*steps[:10], ask, *steps[10:], say("done")], background=["DONE\nSTATE\nDECISIONS\nOPEN\nPITFALLS"])
    eng, _, _ = engine(tmp_path, llm)
    eng.ctx.summarize_in_place = None
    r, _, _ = runner(eng)
    holder["r"] = r
    r.start("explore", staged=False)
    assert r.join(10)
    assert eng.events.of("compact_requested") and "compact" in [e.kind for e in r.events]


def test_pause_resume_updates_activity_and_snapshot(tmp_path):
    holder = {}

    def pause_now(messages):
        holder["r"].pause()
        return call("list_dir", path=".")

    llm = FakeLLM([pause_now, say("finished")])
    eng, _, _ = engine(tmp_path, llm)
    r, reg, _ = runner(eng)
    holder["r"] = r
    r.start("go", staged=False)
    assert wait_for(lambda: eng.now == "paused")
    snap = r.snapshot()
    assert snap["state"] == "paused" and reg.get(r.activity.id).state == "paused" and r.busy
    out = render(r.panel.header(80), 80)
    assert "paused" in out
    r.activity.resume()                            # the registry callback is wired to the runner
    assert r.join(5) and r.result["text"] == "finished" and len(llm.calls) == 2


def test_stop_via_activity_callback_keeps_work(tmp_path):
    holder = {}

    def stop_now(messages):
        holder["r"].activity.stop()
        return call("list_dir", path=".")

    llm = FakeLLM([call("write_file", path="keep.py", content="a = 1\n"), stop_now, say("never")])
    eng, repo, _ = engine(tmp_path, llm)
    r, reg, _ = runner(eng)
    holder["r"] = r
    r.start("write then stop", staged=False)
    assert r.join(5)
    assert r.result["stopped"] == "interrupted" and (repo / "keep.py").exists()
    assert reg.get(r.activity.id).state == "stopped"


def test_stop_wakes_a_paused_run(tmp_path):
    holder = {}

    def pause_now(messages):
        holder["r"].pause()
        return call("list_dir", path=".")

    eng, _, _ = engine(tmp_path, FakeLLM([pause_now, say("never")]))
    r, reg, _ = runner(eng)
    holder["r"] = r
    r.start("go", staged=False)
    assert wait_for(lambda: eng.now == "paused")
    r.stop()
    assert r.join(5) and r.result["stopped"] == "interrupted"


def pump(r, allow=True, limit=200):
    """Answer every distinct approval request until the run ends (the engine may ask again for the same edit)."""
    n, last = 0, 0
    while r.busy and n < limit:
        req = r.pending_approval()
        if req is not None and req.id != last:
            last = req.id
            r.answer_approval(allow, approval_id=req.id)
            n += 1
        time.sleep(0.02)
    return n


def _ask_engine(tmp_path, script):
    return engine(tmp_path, FakeLLM(script), mode="ask")


def test_approval_bridge_allow(tmp_path):
    eng, repo, _ = _ask_engine(tmp_path, [call("write_file", path="w.py", content="a = 1\n"), say("ok")])
    seen = []
    r, reg, bus = runner(eng, on_approval=seen.append)
    r.start("write w.py", staged=False)
    assert wait_for(lambda: r.pending_approval() is not None)
    req = r.pending_approval()
    assert seen == [req] and "w.py" in req.description and not req.dangerous
    assert r.snapshot()["state"] == "waiting" and reg.get(r.activity.id).state == "waiting"
    assert r.answer_approval(True, approval_id=req.id + 99) is False          # wrong id: ignored
    assert r.answer_approval(True)
    pump(r)
    assert r.join(5) and (repo / "w.py").exists() and r.pending_approval() is None
    assert "awaiting_approval" in [e.kind for e in bus.recent(n=500)] and reg.get(r.activity.id).state == "done"


def test_approval_bridge_deny(tmp_path):
    eng, repo, _ = _ask_engine(tmp_path, [call("write_file", path="w.py", content="a = 1\n"), say("could not")])
    r, _, _ = runner(eng)
    r.start("write w.py", staged=False)
    assert wait_for(lambda: r.pending_approval() is not None)
    r.answer_approval(False)
    pump(r, allow=False)
    assert r.join(5) and not (repo / "w.py").exists()


def test_dangerous_needs_typed_yes(tmp_path):
    eng, _, _ = engine(tmp_path, FakeLLM([]))
    r, _, _ = runner(eng)
    results = []

    def ask(typed):
        t = threading.Thread(target=lambda: results.append(r._approve("run  rm -rf build", "dangerous command")))
        t.start()
        assert wait_for(lambda: r.pending_approval() is not None)
        assert r.pending_approval().dangerous
        r.answer_approval(True, typed=typed)
        t.join(5)

    ask("y")
    ask("")
    ask("yes")
    assert results == [False, False, True]


def test_approval_timeout_denies_and_stop_denies(tmp_path):
    eng, _, _ = engine(tmp_path, FakeLLM([]))
    r, _, _ = runner(eng, approval_timeout=0.1)
    assert r._approve("write x", "") is False and r.pending_approval() is None
    r2, _, _ = runner(eng, approval_timeout=30)
    out = []
    t = threading.Thread(target=lambda: out.append(r2._approve("write x", "")))
    t.start()
    assert wait_for(lambda: r2.pending_approval() is not None)
    r2.stop()
    t.join(5)
    assert out == [False]


def test_plan_first_decline_runs_nothing(tmp_path):
    llm = FakeLLM([])
    eng, _, _ = new_engine(tmp_path, llm)
    shown = []
    r, reg, bus = runner(eng)
    r.start(TASK3, plan_first=True, confirm=lambda renderable: shown.append(render(renderable, 80)) or False)
    assert r.join(5)
    assert r.declined_plan and r.result["stopped"] == "declined" and llm.calls == [] and eng.stage_plan is None
    assert "PLAN 0/3" in shown[0] and "stage" in shown[0].lower() or "a.py" in shown[0]
    plan_events = [e for e in bus.recent(n=500) if e.kind == "plan"]
    assert plan_events and plan_events[0].data["awaiting_confirm"]


def test_plan_first_confirm_via_api_runs_the_staged_plan(tmp_path):
    llm = FakeLLM(three_stage_script())
    eng, _, _ = new_engine(tmp_path, llm)
    r, _, _ = runner(eng)
    r.start(TASK3, plan_first=True)
    assert wait_for(lambda: any(e.kind == "plan" for e in r.events))
    assert r.busy and llm.calls == []
    r.confirm_plan(True)
    assert r.join(20)
    assert r.result["exit_code"] == 0 and [s.status for s in eng.stage_plan.stages] == ["done"] * 3
    assert r.snapshot()["plan"]["stages"]


def test_engine_error_is_reported_not_raised(tmp_path):
    def boom(messages):
        raise RuntimeError("model server vanished")

    eng, _, _ = engine(tmp_path, FakeLLM([boom]))
    r, reg, _ = runner(eng)
    r.start("go", staged=False)
    assert r.join(5)
    assert r.result["stopped"] == "error" and reg.get(r.activity.id).state in ("failed", "done")


def test_detach_stops_forwarding(tmp_path):
    eng, _, _ = engine(tmp_path, FakeLLM([say("hi")]))
    r, _, bus = runner(eng)
    r.detach()
    eng.turn("hello")
    assert bus.recent(n=10) == [] or all(e.kind != "tool_start" for e in bus.recent(n=10))
    assert len(r.events) == 0
