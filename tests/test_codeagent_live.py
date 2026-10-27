"""Live-UI hooks of the coding engine: event subscription, live-only events, step hooks, steering, forced compaction, pause / stop, snapshot."""

from __future__ import annotations

import json
import threading
import time

from tests.test_codeagent_engine import FakeLLM, _isolated, call, engine, say  # noqa: F401 - _isolated is an autouse fixture

from agent.codeagent import livestate
from agent.codeagent.eventlog import EventLog


def test_subscribers_get_events_and_a_broken_one_breaks_nothing(tmp_path):
    log = EventLog("s", tmp_path / "e.jsonl")
    seen, bad = [], []

    def broken(rec):
        bad.append(rec)
        raise RuntimeError("ui exploded")

    off_broken, off = log.subscribe(broken), log.subscribe(seen.append)
    log.emit("a", x=1)
    log.emit_live("b", y=2)
    assert [r["kind"] for r in seen] == ["a", "b"] and len(bad) == 2
    off()
    log.emit("c")
    assert [r["kind"] for r in seen] == ["a", "b"]
    off_broken()
    log.emit("d")
    assert len(bad) == 3


def test_emit_live_is_not_persisted(tmp_path):
    path = tmp_path / "e.jsonl"
    log = EventLog("s", path)
    log.emit("kept", n=1)
    rec = log.emit_live("secret", payload="x" * 100)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in lines] == ["kept"] and rec["payload"] == "x" * 100
    assert log.of("secret") == []


def test_subscribe_is_thread_safe(tmp_path):
    log = EventLog("s", tmp_path / "e.jsonl")
    got = []
    log.subscribe(got.append)
    threads = [threading.Thread(target=lambda: [log.emit_live("t") for _ in range(50)]) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(got) == 200


def test_several_step_hooks_and_the_legacy_slot_all_fire(tmp_path):
    eng, _, _ = engine(tmp_path, FakeLLM([call("list_dir", path="."), say("done")]))
    hits = []
    eng.on_step = lambda: hits.append("legacy")
    h1, h2 = (lambda: hits.append("h1")), (lambda: (_ for _ in ()).throw(RuntimeError("bad hook")))
    eng.add_step_hook(h1)
    eng.add_step_hook(h2)
    eng.turn("look around")
    assert hits.count("legacy") == 2 and hits.count("h1") == 2
    eng.remove_step_hook(h1)
    eng.llm.script = [say("again")]
    eng.turn("once more")
    assert hits.count("h1") == 2 and hits.count("legacy") == 3


def test_tool_events_edit_diff_and_now(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="x = 1\ny = 2\n"),
                   call("edit_file", path="a.py", edits=[{"search": "y = 2", "replace": "y = 3\nz = 4"}]),
                   say("Edited a.py.")])
    eng, repo, _ = engine(tmp_path, llm)
    events = []
    eng.events.subscribe(events.append)
    eng.turn("make a.py")
    kinds = [e["kind"] for e in events]
    assert "tool_start" in kinds and "tool_result" in kinds and "tool" in kinds
    start = next(e for e in events if e["kind"] == "tool_start")
    assert start["name"] == "write_file" and "x = 1" in start["args"] and start["live"]
    res = next(e for e in events if e["kind"] == "tool_result")
    assert res["name"] == "write_file" and res["ok"] is True and res["result"]
    edits = [e for e in events if e["kind"] == "edit"]
    assert edits[0]["path"] == "a.py" and edits[0]["created"] and edits[0]["added"] == 2 and edits[0]["removed"] == 0
    assert edits[1]["added"] == 2 and edits[1]["removed"] == 1
    assert "-y = 2" in edits[1]["diff"] and "+y = 3" in edits[1]["diff"] and "+z = 4" in edits[1]["diff"]
    persisted = [json.loads(line)["kind"] for line in (tmp_path / "ev.jsonl").read_text().splitlines()]
    assert "edit" not in persisted and "tool_start" not in persisted and "tool" in persisted
    nows = [e["text"] for e in events if e["kind"] == "now"]
    assert any("waiting for the model" in n for n in nows) and nows[-1] == "idle" and eng.now == "idle"


def test_thinking_is_forwarded_only_when_enabled(tmp_path):
    def run(flag):
        msg = {**say("ok"), "thinking": "let me think"}
        eng, _, _ = engine(tmp_path, FakeLLM([msg]))
        eng.capture_thinking = flag
        got = []
        eng.events.subscribe(got.append)
        eng.turn("hi")
        return [e for e in got if e["kind"] == "thinking"]

    assert run(False) == []
    assert run(True)[0]["text"] == "let me think"


def test_steer_message_reaches_the_next_model_call(tmp_path):
    holder = {}

    def first(messages):
        holder["eng"].queue_user_message("also add a README")
        assert holder["eng"].live.queued() == 1
        return call("list_dir", path=".")

    llm = FakeLLM([first, say("done")])
    eng, _, _ = engine(tmp_path, llm)
    holder["eng"] = eng
    got = []
    eng.events.subscribe(got.append)
    eng.turn("do the thing")
    assert not any("README" in str(m["content"]) for m in llm.calls[0]["snapshot"])
    steer = [m for m in llm.calls[1]["snapshot"] if "README" in str(m["content"])]
    assert len(steer) == 1 and steer[0]["role"] == "user" and steer[0]["content"].startswith("[user message during the run]")
    assert [e["kind"] for e in got].count("steer") == 2        # persisted summary + live text
    assert eng.live.queued() == 0


def test_request_compact_runs_exactly_one_pass(tmp_path):
    steps = [call("write_file", path=f"f{i}.txt", content=f"file {i}\n") for i in range(12)]
    holder = {}

    def ask(messages):
        holder["eng"].request_compact()
        return call("write_file", path="ask.txt", content="x\n")

    llm = FakeLLM([*steps[:10], ask, *steps[10:], say("done")], background=["DONE\nSTATE\nDECISIONS\nOPEN\nPITFALLS"])
    eng, _, _ = engine(tmp_path, llm)
    holder["eng"] = eng
    eng.ctx.summarize_in_place = None
    eng.turn("explore")
    assert len(eng.events.of("compact")) == 1 and eng.events.of("compact_requested")[0]["done"] is True
    assert not eng.live.take_compact()


def test_pause_blocks_and_resume_continues(tmp_path):
    holder = {}

    def pause_now(messages):
        holder["eng"].request_pause()
        return call("list_dir", path=".")

    llm = FakeLLM([pause_now, say("finished")])
    eng, _, _ = engine(tmp_path, llm)
    holder["eng"] = eng
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("res", eng.turn("go")))
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline and eng.now != "paused":
        time.sleep(0.02)
    assert eng.now == "paused" and t.is_alive() and len(llm.calls) == 1
    snap = livestate.snapshot(eng)
    assert snap["state"] == "paused" and snap["paused"] is True
    eng.request_resume()
    t.join(5)
    assert not t.is_alive() and out["res"].text == "finished" and len(llm.calls) == 2


def test_stop_ends_cleanly_and_keeps_the_work(tmp_path):
    holder = {}

    def stop_now(messages):
        holder["eng"].request_stop()
        return call("list_dir", path=".")

    llm = FakeLLM([call("write_file", path="keep.py", content="a = 1\n"), stop_now, say("never reached")])
    eng, repo, out = engine(tmp_path, llm)
    holder["eng"] = eng
    res = eng.turn("write then stop")
    assert res.stopped == "interrupted" and (repo / "keep.py").read_text() == "a = 1\n" and "work so far is kept" in out.getvalue()
    assert res.files == ["keep.py"] and len(llm.calls) == 2
    eng.request_stop()                                     # idle: ignored, the next turn is not killed
    eng.llm.script = [say("fine")]
    assert eng.turn("again").text == "fine"


def test_stop_while_paused_wakes_up(tmp_path):
    holder = {}

    def pause_now(messages):
        holder["eng"].request_pause()
        return call("list_dir", path=".")

    eng, _, _ = engine(tmp_path, FakeLLM([pause_now, say("x")]))
    holder["eng"] = eng
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("res", eng.turn("go")))
    t.start()
    while eng.now != "paused":
        time.sleep(0.02)
    eng.request_stop()
    t.join(5)
    assert not t.is_alive() and out["res"].stopped == "interrupted"


def test_snapshot_matches_engine_state(tmp_path):
    llm = FakeLLM([call("write_file", path="s.py", content="a = 1\nb = 2\n"), say("ok")])
    eng, _, _ = engine(tmp_path, llm)
    idle = livestate.snapshot(eng)
    assert idle["state"] == "idle" and idle["step"] == 0 and idle["model"] == eng.model and idle["mode"] == "auto"
    seen = []
    eng.add_step_hook(lambda: seen.append(livestate.snapshot(eng)))
    eng.turn("make s.py")
    assert seen[0]["state"] == "running" and seen[0]["step"] == 1 and seen[0]["budget_now"] == eng.budget_now and seen[0]["files_changed"] == 0
    assert seen[1]["step"] == 2 and seen[1]["files_changed"] == 1 and seen[1]["lines_added"] == 2 and seen[1]["files"][0]["path"].endswith("s.py")
    snap = livestate.snapshot(eng)
    assert snap["step"] == eng.ctx.step == 2 and snap["state"] == "idle" and snap["now"] == "idle"
    assert snap["ctx_limit"] == eng.num_ctx and snap["ctx_used"] == eng.ctx.estimate()
    assert snap["step_ceiling"] == eng.step_ceiling and snap["budget_source"] == eng.budget_source
    assert snap["dead_ends"] == 0 and snap["queued"] == 0 and snap["plan"] is None
    eng.queue_user_message("hello")
    assert livestate.snapshot(eng)["queued"] == 1


def test_snapshot_never_raises():
    assert livestate.snapshot(None)["state"] == "idle"
    assert livestate.snapshot(object())["state"] == "idle"


def test_request_compact_with_nothing_to_fold_is_harmless(tmp_path):
    holder = {}

    def ask(messages):
        holder["eng"].request_compact()
        return call("list_dir", path=".")

    eng, _, out = engine(tmp_path, FakeLLM([ask, say("done")]))
    holder["eng"] = eng
    eng.turn("tiny")
    assert eng.events.of("compact_requested")[0]["done"] is False and "nothing to compact yet" in out.getvalue()
