"""Stage-aware masking: inside a stage the context decisions are made against min(window, stage cap), old tool results are stubbed in ONE
batch (rarely: every change of the prefix costs the server's prompt cache) and the cap only ends a segment when that batch cannot help.
FakeLLM only - the prompt sizes are scripted."""

from __future__ import annotations

import json

import pytest

from agent.codeagent import context as C, sessionstore as SS
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_context import cm, fill
from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import cfg, new_engine

CAP = 10000


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def staged(tmp_path, cap=CAP, ctx=96000, turns=14, **coding):
    m = cm(tmp_path, ctx=ctx)
    if coding:
        m.config = {**m.config, "coding": coding}
    m.stage_cap_fn = lambda: cap
    m.add({"role": "system", "content": "sys"})
    m.add({"role": "system", "content": "BRIEF CARD\n- keep this"})
    m.add({"role": "system", "content": "STAGE GOAL (pinned by the harness; work on this stage only):\nbuild it"})
    m.add({"role": "user", "content": "VERIFIED STATE\n" + "state line\n" * 400})
    fill(m, turns)
    return m


def stubbed(m):
    return [x for x in m.messages if x["role"] == "tool" and x["content"].startswith("[old ")]


def test_the_batch_fires_at_70_percent_of_the_cap_and_shrinks_the_prompt(tmp_path):
    m = staged(tmp_path)
    m.last_prompt, m._since = 6900, 0                                           # 69% of the cap: nothing yet (the old line would be 69k tokens)
    assert m.maintain() == [] and not stubbed(m)
    m.last_prompt, m._since = 7100, 0
    cleared = []
    m.on_clear_cache = lambda: cleared.append(1)
    notes = m.maintain()
    assert notes and "stage context trimmed" in notes[0] and cleared
    ev = [r for r in m.events.recent if r["kind"] == "stage_mask"]
    assert len(ev) == 1 and ev[0]["before"] == 7100 and ev[0]["after"] <= 5000 and ev[0]["masked_results"] == len(stubbed(m)) == 3
    assert m.estimate() == ev[0]["after"]


def test_the_newest_turns_the_brief_the_goal_and_edit_results_stay_untouched(tmp_path):
    # ADJUSTED: the keep is now a token budget (newest 2000 tokens = 2 results here) with a floor of 2 turns, not 6 turns; target -1 = mask all that may be
    m = staged(tmp_path, turns=0, stage_mask_target=-1.0)
    edit = "wrote a.py (3 lines)\n" + "diff line\n" * 300
    m.add({"role": "assistant", "content": json.dumps({"tool": "replace_lines", "arguments": {"path": "a.py", "start": 1, "end": 1, "new_text": "x"}})})
    m.add({"role": "tool", "content": edit})
    fill(m, 12)
    keep_before = [dict(x) for x in m.messages[:4]]
    tail_before = [dict(x) for x in m.messages[-4:]]               # the newest 2 results (working set) = 4 messages
    m.last_prompt, m._since = 7200, 0
    m.maintain()
    assert m.messages[:4] == keep_before                           # system prompt, brief card, stage goal, verified state
    assert m.messages[-4:] == tail_before
    assert any(x["content"] == edit for x in m.messages)           # an edit result is never stubbed
    assert len(stubbed(m)) == 10
    assert sum(x["role"] == "assistant" for x in m.messages) == 13


def test_mask_keep_turns_is_a_floor_and_configurable(tmp_path):
    # ADJUSTED: the working set is 0 tokens here (only the newest result is always kept), so the turn floor alone decides
    m = staged(tmp_path, mask_keep_turns=2, stage_mask_target=-1.0, mask_keep_tokens=0.0)
    m.last_prompt, m._since = 7200, 0
    m.maintain()
    assert len(stubbed(m)) == 12                                    # all but the results inside the newest 2 turns
    m6 = staged(tmp_path, mask_keep_turns=6, stage_mask_target=-1.0, mask_keep_tokens=0.0)
    m6.last_prompt, m6._since = 7200, 0
    m6.maintain()
    assert len(stubbed(m6)) == 14 - 6 and C.MASK_KEEP_TURNS == 2


def test_no_second_batch_before_the_growth_gap_and_one_after_it(tmp_path):
    m = staged(tmp_path, stage_mask_gap=1.0)
    m.last_prompt, m._since = 7200, 0
    m.maintain()
    floor = m.stage_mask_floor
    after = m.estimate()
    assert floor == pytest.approx(after + 1.0 * CAP)
    fill(m, 6)
    m.step += 10                                                   # (ADJUSTED: past the 6-step thrash guard)
    snap = [dict(x) for x in m.messages]
    m.last_prompt, m._since = int(floor) - 50, 0                   # grown, above the 70% line, but not by the gap yet
    assert m.maintain() == [] and m.messages == snap
    m.last_prompt, m._since = int(floor) + 50, 0
    assert m.maintain() and m.messages != snap
    assert len([r for r in m.events.recent if r["kind"] == "stage_mask"]) == 2


def test_a_batch_that_frees_too_little_is_skipped_not_applied(tmp_path):
    m = staged(tmp_path, cap=28000, turns=7)                       # one result is old: ~1k tokens of a 28k budget is not worth a cache miss
    m.last_prompt, m._since = 20000, 0
    snap = [dict(x) for x in m.messages]
    assert m.maintain() == [] and m.messages == snap
    assert [r for r in m.events.recent if r["kind"] == "stage_mask_skipped"] and m.stage_mask_floor > 20000


def test_without_a_stage_cap_the_old_thresholds_apply(tmp_path):
    m = cm(tmp_path, ctx=10000)
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 14)
    m.last_prompt, m._since = 7100, 0                                           # 71% of the window: below MASK_AT, a stage would have masked here
    assert m.maintain() == [] and not stubbed(m) and not [r for r in m.events.recent if r["kind"].startswith("stage_mask")]
    m.last_prompt, m._since = 7300, 0
    assert any("trimmed" in n for n in m.maintain()) and [r for r in m.events.recent if r["kind"] == "mask"]
    assert not [r for r in m.events.recent if r["kind"] == "stage_mask"]


def test_the_budget_is_the_smaller_of_window_and_cap(tmp_path):
    assert staged(tmp_path, cap=28000, ctx=96000).stage_budget() == 28000
    assert staged(tmp_path, cap=28000, ctx=16000).stage_budget() == 16000
    assert cm(tmp_path).stage_budget() is None


def test_the_stub_names_the_exact_range_and_the_ranges_seen(tmp_path):
    (tmp_path / "story.py").write_text("class Story:\n    def start(self):\n        pass\n" + "# filler\n" * 300)
    m = cm(tmp_path)
    for a, b in ((1, 100), (150, 250)):
        m.add({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": "story.py", "start": a, "end": b}})})
        m.add({"role": "tool", "content": f"story.py (303 lines)\n--- lines {a}-{b} ---\n" + "x = 1\n" * 300})
    m.add({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": "story.py"}})})
    m.add({"role": "tool", "content": "story.py (303 lines)\n--- lines 1-250 ---\n" + "x = 1\n" * 300})
    masked, _ = C.mask_observations(m.messages, tmp_path, keep_last=0)
    stubs = [x["content"] for x in masked if x["role"] == "tool"]
    assert 'read_file(path="story.py", start=1, end=100)' in stubs[0] and 'read_file(path="story.py", start=150, end=250)' in stubs[1]
    assert 'start=1, end=250' in stubs[2]                                         # a range-less read: the range the harness returned
    assert "Ranges of this file you have read: 1-100, 150-250, 1-250" in stubs[0] and "class Story" in stubs[0]


def test_stubs_are_harness_markers_the_model_cannot_write_into_a_file(tmp_path):
    m = staged(tmp_path, stage_mask_target=-1.0, mask_keep_tokens=0.0)     # ADJUSTED: mask everything allowed so every stub kind is present
    m.add({"role": "assistant", "content": json.dumps({"tool": "search", "arguments": {"pattern": "x"}})})
    m.add({"role": "tool", "content": "\n".join(f"f.py:{i}: x = 1" for i in range(300))})
    m.add({"role": "assistant", "content": json.dumps({"tool": "run", "arguments": {"command": "pytest"}})})
    m.add({"role": "tool", "content": "\n".join(f"out {i}" for i in range(300))})
    fill(m, 6)
    m.last_prompt, m._since = 7500, 0
    m.maintain()
    stubs = stubbed(m)
    assert len(stubs) >= 10
    for s in (x["content"] for x in stubs):
        assert C.new_markers(None, s), s[:60]                                      # every stub trips the write guard (tools._commit_write)
        assert C.new_markers(None, s + "\n" + s)


def test_the_marker_guard_refuses_a_stub_in_a_write(tmp_path):
    from tests.test_codeagent_engine import engine

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    stub = C._stub_for("read_file", {"path": "a.py", "start": 1, "end": 9}, "x" * 3000, repo, ["1-9"])
    res = eng.execute("write_file", {"path": "n.py", "content": stub, "overwrite": True})
    assert "harness marker" in res and not (repo / "n.py").exists()


# ── the engine loop: the cap comes after the batch ───────────────────────────

def reads(repo, n):
    for i in range(n):
        (repo / f"f{i}.py").write_text(f"def fn{i}():\n    return {i}\n" + f"# line of file {i}\n" * 330)         # ~6k chars each


def sized(step, tokens):
    return {**step, "_prompt_eval_count": tokens}


def engine_in_stage(tmp_path, script, n_files=16, **coding):
    llm = FakeLLM(script)
    eng, repo, out = new_engine(tmp_path, llm, cfg(**coding))
    reads(repo, n_files)
    eng.pol.max_steps, eng.pol.steps_set = 40, True
    eng.stage_cap = 28000
    eng.begin_stage("read the files", "VERIFIED STATE\nnothing yet", "")
    return eng, llm, repo, out


def test_a_stage_that_would_have_died_at_the_cap_survives_when_the_batch_frees_enough(tmp_path):
    # the trigger level is parked at 99% so that only the cap check can fire; 29000 > 28000, the old results hold far more than 1000 tokens
    script = [sized(call("read_file", path=f"f{i}.py"), 2000 + 1000 * i) for i in range(10)]
    script += [sized(call("read_file", path="f10.py"), 29000), sized(call("read_file", path="f11.py"), 15000), sized(say("Finished reading."), 16000)]
    eng, llm, _repo, out = engine_in_stage(tmp_path, script, stage_mask_at=0.99)
    res = eng.turn("read the files")
    assert res.stopped != "stage_cap" and res.text == "Finished reading."
    ev = eng.events.of("stage_mask")
    assert len(ev) == 1 and ev[0]["before"] >= 29000 and ev[0]["after"] <= 28000 and ev[0]["masked_results"] >= 3
    seen = llm.calls[-1]["snapshot"]
    assert any(m["content"].startswith("[old read of f0.py") for m in seen if m["role"] == "tool")
    assert "stage context trimmed" in out.getvalue()


def test_the_cap_still_ends_the_segment_when_the_batch_cannot_free_enough(tmp_path):
    script = [sized(call("run", command="echo hi"), 40000)]                          # nothing old to mask
    eng, llm, _r, _o = engine_in_stage(tmp_path, script)
    res = eng.turn("go")
    assert res.stopped == "stage_cap" and not eng.events.of("stage_mask")
    # something to mask, but far from enough: 60000 -> ~50000 is still over the cap
    script = [sized(call("read_file", path=f"f{i}.py"), 3000) for i in range(9)] + [sized(call("read_file", path="f9.py"), 60000)]
    eng2, _l, _r2, _o2 = engine_in_stage(tmp_path, script)
    res2 = eng2.turn("go")
    assert res2.stopped == "stage_cap" and eng2.events.of("stage_mask")


def test_the_batch_fires_inside_a_stage_before_the_cap_is_reached(tmp_path):
    script = [sized(call("read_file", path=f"f{i}.py"), 2000 * (i + 1)) for i in range(10)] + [sized(say("ok"), 12000)]
    eng, llm, _repo, _o = engine_in_stage(tmp_path, script)
    res = eng.turn("go")
    assert res.stopped != "stage_cap"
    ev = eng.events.of("stage_mask")
    assert len(ev) == 1 and ev[0]["before"] >= 0.7 * 28000 > 19000                    # at 20000 of 28000, not at 69k
    assert ev[0]["after"] < ev[0]["before"]


def test_runs_without_a_stage_cap_do_not_use_the_stage_batch(tmp_path):
    llm = FakeLLM([sized(call("read_file", path=f"f{i}.py"), 4000 * (i + 1)) for i in range(7)] + [sized(say("ok"), 29000)])
    eng, repo, _o = new_engine(tmp_path, llm)
    reads(repo, 8)
    eng.pol.max_steps, eng.pol.steps_set = 20, True
    res = eng.turn("read")
    assert res.stopped != "stage_cap" and not eng.events.of("stage_mask")


def test_begin_stage_rearms_the_batch(tmp_path):
    eng, _l, _r, _o = engine_in_stage(tmp_path, [])
    eng.ctx.stage_mask_floor = 99999.0
    eng.begin_stage("again", "VERIFIED STATE\nx", "")
    assert eng.ctx.stage_mask_floor == 0.0


# ── identical re-reads ──────────────────────────────────────────────────────

def test_an_identical_reread_is_a_one_liner_while_unmasked_and_unchanged(tmp_path):
    from agent.codeagent import profile as PR
    from agent.codeagent.tools import CodeTools

    (tmp_path / "w.py").write_text("def build():\n    pass\n" + "# pad\n" * 100)
    ctx = cm(tmp_path)
    tools = CodeTools(tmp_path, PR.detect(tmp_path), ctx)
    ctx.step = 3
    first = tools.read_file("w.py", start=1, end=40)
    assert "def build" in first
    lines = tools.read_lines
    again = tools.read_file("w.py", start=1, end=40)
    assert "unchanged since step 3" in again and "already in your context" in again and "\n" not in again and len(again) < 200
    assert tools.read_counts["w.py"] == 2                                              # still counted as a read (lookact), never as an edit
    assert "# pad" in tools.read_file("w.py", start=1, end=41)                         # another range: a real read
    (tmp_path / "w.py").write_text("def build():\n    return 1\n" + "# pad\n" * 100)  # changed on disk: a real read
    assert "return 1" in tools.read_file("w.py", start=1, end=40)
    assert tools.read_lines > lines
    tools.read_file("w.py", start=1, end=40)
    ctx._cache_reset()                                                                 # a mask drops the old results from the context: real read again
    assert "return 1" in tools.read_file("w.py", start=1, end=40)


def test_a_stage_batch_makes_the_next_identical_read_a_real_one(tmp_path):
    from agent.codeagent import profile as PR
    from agent.codeagent.tools import CodeTools

    (tmp_path / "w.py").write_text("def build():\n    pass\n" + "# pad\n" * 100)
    m = staged(tmp_path)
    tools = CodeTools(tmp_path, PR.detect(tmp_path), m)
    tools.read_file("w.py", start=1, end=40)
    assert "unchanged since" in tools.read_file("w.py", start=1, end=40)
    m.last_prompt, m._since = 7200, 0
    m.maintain()
    assert "def build" in tools.read_file("w.py", start=1, end=40)


# ── persistence ─────────────────────────────────────────────────────────────

def test_the_masked_since_marker_is_saved_and_a_missing_one_means_not_masked(tmp_path):
    eng, _l, _r, _o = engine_in_stage(tmp_path, [])
    eng.ctx.stage_mask_floor = 12345.0
    data = SS.snapshot(eng)
    assert data["ctx_state"]["stage_mask_floor"] == 12345.0
    eng.ctx.stage_mask_floor = 0.0
    SS.restore(eng, data)
    assert eng.ctx.stage_mask_floor == 12345.0
    data["ctx_state"].pop("stage_mask_floor")                                          # an older save: "not masked since" = allowed to mask
    SS.restore(eng, data)
    assert eng.ctx.stage_mask_floor == 0.0


# ── the measured thrash: token-budgeted keep, worth-it rule, thrash guard ────

def synth(tmp_path, before, old, recent, cap=28000, **coding):
    """A staged context of `before` tokens: head + `old` results (token sizes, oldest first) + `recent` ones (the newest 6 turns)."""
    m = cm(tmp_path, ctx=96000)
    if coding:
        m.config = {**m.config, "coding": coding}
    m.stage_cap_fn = lambda: cap
    m.add({"role": "system", "content": "sys"})
    m.add({"role": "system", "content": "BRIEF CARD\n- keep"})
    sizes = [*old, *recent]
    head = before - sum(sizes)
    assert head > 0
    m.add({"role": "user", "content": "VERIFIED STATE\n" + "s" * int(head * C.CHARS_PER_TOKEN)})
    for i, t in enumerate(sizes):
        m.add({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": f"g{i}.py"}})})
        m.add({"role": "tool", "content": f"g{i}.py\n" + "x = 1\n" * int(t * C.CHARS_PER_TOKEN / 6)})
    m.last_prompt, m._since = before, 0
    return m


# (before, old results, six recent results, measured after): the prompts of the live run; the old rule (keep 6 turns) masked 2/3/2/7 results
REAL = [(29198, [1600] * 2, [4000] * 6, 26008), (34315, [2800] * 3, [3800] * 6, 26725), (33758, [3000] * 2, [4300] * 6, 27223),
        (27465, [2000] * 7, [1500] * 6, 13437)]


@pytest.mark.parametrize("before,old,recent,measured_after", REAL)
def test_the_measured_batches_now_free_down_to_the_target(tmp_path, before, old, recent, measured_after):
    m = synth(tmp_path, before, old, recent)
    # the old rule (keep 6 turns) reproduces the measurement: it frees only what the old results hold
    _, saved, n = C._mask(m.messages, tmp_path, 0, 700, 6, C.STAGE_MASK_TOOLS)
    assert n == len(old) and abs((before - saved / C.CHARS_PER_TOKEN) - measured_after) < 0.12 * measured_after
    assert m.stage_mask(force=True)
    ev = [r for r in m.events.recent if r["kind"] == "stage_mask"][-1]
    assert ev["before"] == before and ev["after"] <= 0.5 * 28000 and m.estimate() <= 0.5 * 28000
    assert ev["masked_results"] > n or measured_after <= 14000


def test_the_newest_result_is_kept_even_when_it_alone_exceeds_the_keep_budget(tmp_path):
    m = synth(tmp_path, 30000, [2000] * 4, [9000, 9000], mask_keep_turns=0)     # budget 5600: the newest (9000) is the only one kept (no turn floor)
    assert m.stage_mask(force=True)
    tools = [x for x in m.messages if x["role"] == "tool"]
    assert not tools[-1]["content"].startswith("[old ") and tools[-2]["content"].startswith("[old ")
    assert len(stubbed(m)) == 5


def test_the_working_set_keeps_the_newest_results_up_to_the_keep_budget(tmp_path):
    m = synth(tmp_path, 30000, [3000] * 4, [2500] * 4, stage_mask_target=-1.0)      # budget 5600: the newest two (2 x 2500) stay, the third would not fit
    assert m.stage_mask(force=True)
    tools = [x for x in m.messages if x["role"] == "tool"]
    assert [x["content"].startswith("[old ") for x in tools[-4:]] == [True, True, False, False]


def test_the_batch_masks_a_contiguous_old_prefix_oldest_first(tmp_path):
    m = synth(tmp_path, 30000, [2500] * 6, [2000] * 2)                 # target 14000: needs 16000 -> masks oldest first, stops when enough
    assert m.stage_mask(force=True)
    flags = [x["content"].startswith("[old ") for x in m.messages if x["role"] == "tool"]
    k = flags.index(False)
    assert k >= 5 and not any(flags[k:])                                # stubs form one prefix; nothing newer than the first unmasked one


def test_edit_results_are_never_masked_by_the_budgeted_batch(tmp_path):
    m = synth(tmp_path, 30000, [2500] * 2, [])
    edit = "wrote a.py\n" + "diff line\n" * 400
    m.add({"role": "assistant", "content": json.dumps({"tool": "edit_file", "arguments": {"path": "a.py"}})})
    m.add({"role": "tool", "content": edit})
    for i in range(3):
        m.add({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": f"h{i}.py"}})})
        m.add({"role": "tool", "content": "x = 1\n" * 3000})
    m.last_prompt = 40000
    assert m.stage_mask(force=True)
    assert any(x["content"] == edit for x in m.messages)


def test_a_trigger_batch_is_skipped_when_it_would_free_less_than_a_quarter_of_the_prompt(tmp_path):
    # 20000 of 28000 (past the 70 % line); the only old results are 2 x 1000 tokens = 10 % of the prompt: skipped, nothing changes
    m = synth(tmp_path, 20000, [1000] * 2, [2000] * 2)       # 14000 tokens of the prompt are the head (not maskable)
    snap = [dict(x) for x in m.messages]
    assert m.maintain() == [] and m.messages == snap
    sk = [r for r in m.events.recent if r["kind"] == "stage_mask_skipped"]
    assert sk and sk[0]["reason"] == "low_gain"
    assert C.STAGE_MASK_MIN_GAIN == 0.25
    # the same prompt with plenty of old results: applied
    m2 = synth(tmp_path, 20000, [2500] * 4, [1500] * 2)
    assert m2.maintain() and m2.estimate() <= 14000


def test_the_forced_batch_keeps_the_rule_any_gain(tmp_path):
    m = synth(tmp_path, 29000, [1700], [1500] * 3)                      # frees ~1k of 29k (3 %): fine when the cap forces it
    assert m.stage_mask(force=True) and len(stubbed(m)) == 1


def test_a_batch_needed_again_within_six_steps_ends_the_segment(tmp_path):
    m = synth(tmp_path, 29000, [2500] * 6, [1000] * 3)
    m.step = 10
    assert m.stage_mask(force=True)
    m.step = 13
    m.stage_mask_floor = 0.0                                            # (the growth gap is not what this test is about)
    m.last_prompt, m._since = 29000, 0
    assert m.stage_mask(force=True) is None and m.stage_mask_thrash
    assert [r for r in m.events.recent if r["kind"] == "stage_mask_skipped"][-1]["reason"] == "again_within_steps"


def test_two_fruitless_batches_within_the_window_end_the_segment(tmp_path):
    # nothing is old enough to free much: every batch frees < 20 % of the prompt
    m = synth(tmp_path, 29000, [1200], [1500] * 3)
    m.step = 1
    assert m.stage_mask(force=True) and not m.stage_mask_thrash
    m.step = 8                                                           # >= 6 steps later, within the 8-step window of the first
    m.stage_mask_floor = 0.0
    for name in ("z", "y"):
        m.add({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": f"{name}.py"}})})
        m.add({"role": "tool", "content": "x = 1\n" * 1000})
    m.last_prompt, m._since = 29000, 0
    assert m.stage_mask(force=True)
    assert m.stage_mask_thrash
    assert [r for r in m.events.recent if r["kind"] == "stage_mask_skipped"][-1]["reason"] == "two_fruitless_batches"
    assert m.stage_mask(force=True) is None                              # no more batches in this segment


def test_the_engine_ends_the_segment_when_the_guard_fires(tmp_path):
    script = [sized(call("read_file", path=f"f{i}.py"), 2000 + 1000 * i) for i in range(4)] + [sized(say("never reached"), 5000)]
    eng, llm, _r, _o = engine_in_stage(tmp_path, script)
    eng.ctx.stage_mask_thrash = True
    res = eng.turn("read")
    assert res.stopped == "stage_cap"
    eng.begin_stage("again", "VERIFIED STATE\nx", "")
    assert eng.ctx.stage_mask_thrash is False and eng.ctx.stage_mask_log == []


def test_the_new_defaults_and_config_overrides(tmp_path):
    assert (C.STAGE_MASK_GAP, C.STAGE_MASK_TARGET, C.MASK_KEEP_TOKENS, C.STAGE_MASK_WINDOW, C.MASK_KEEP_TURNS) == (0.40, 0.50, 0.20, 8, 2)
    m = synth(tmp_path, 30000, [2500] * 6, [2000] * 2, stage_mask_target=0.8, stage_mask_gap=0.1)
    assert m.stage_mask(force=True)
    assert 0.7 * 28000 < m.estimate() <= 0.8 * 28000                     # stopped at the configured target, not at 50 %
    big = synth(tmp_path, 30000, [2500] * 6, [2000] * 2, mask_keep_tokens=0.9)       # a bigger working set protects more of the old results
    big.stage_mask(force=True)
    assert big.estimate() > 14000
    assert m.stage_mask_floor == pytest.approx(m.estimate() + 0.1 * 28000)
    d = synth(tmp_path, 30000, [2500] * 6, [2000] * 2)
    d.stage_mask(force=True)
    assert d.stage_mask_floor == pytest.approx(d.estimate() + 0.40 * 28000)
