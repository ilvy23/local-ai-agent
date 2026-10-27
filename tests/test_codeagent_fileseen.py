"""FILES SO FAR (fileseen.py): a per-file record of what a stage has already read and edited, built mechanically from the tool calls (never from
model text) and shown when the stage restarts in a fresh context, so the model does not read the same ranges again. Idea from pi's compaction
(`readFiles` / `modifiedFiles` taken from tool calls). FakeLLM only."""

from __future__ import annotations

import pytest

from agent.codeagent import fileseen as FS, stages as ST, verified

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import new_engine, write_lines


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def stage(**kw):
    return ST.Stage(1, "Texts", "Rewrite the texts of story.py", files=["story.py"], **kw)


# ── recording ───────────────────────────────────────────────────────────────

def test_reads_are_merged_and_counted_per_file():
    seen: dict = {}
    FS.note_read(seen, "story.py", 1, 120)
    FS.note_read(seen, "story.py", 100, 240)
    FS.note_read(seen, "story.py", 300, 380)
    FS.note_read(seen, "other.py", 1, 5)
    assert seen["story.py"]["read"] == [[1, 240], [300, 380]] and seen["story.py"]["reads"] == 3 and seen["story.py"]["last"] == "read"
    assert seen["other.py"]["reads"] == 1


def test_a_placeholder_read_counts_but_adds_no_range():
    seen: dict = {}
    FS.note_read(seen, "story.py", 1, 50)
    FS.note_read(seen, "story.py")
    assert seen["story.py"]["read"] == [[1, 50]] and seen["story.py"]["reads"] == 2


def test_an_edit_sets_the_last_action_and_a_later_read_sets_it_back():
    seen: dict = {}
    FS.note_edit(seen, "story.py")
    assert seen["story.py"]["last"] == "edited" and seen["story.py"]["read"] == [] and seen["story.py"]["reads"] == 0
    FS.note_read(seen, "story.py", 1, 10)
    assert seen["story.py"]["last"] == "read"


# ── the text ────────────────────────────────────────────────────────────────

def test_a_row_names_ranges_count_and_edited_lines_and_edited_files_come_first():
    seen: dict = {}
    for a, b in ((1, 240), (300, 380)):
        FS.note_read(seen, "story.py", a, b)
    FS.note_read(seen, "story.py", 1, 10)
    FS.note_read(seen, "zeta.py", 1, 40)
    FS.note_read(seen, "alpha.py", 1, 40)
    FS.note_read(seen, "alpha.py", 1, 40)
    FS.note_edit(seen, "story.py")
    rows = FS.rows(seen, {"story.py": [[1, 120]], "new.py": [[1, 30]]})
    assert rows == ["new.py  edited 1-30", "story.py  read 1-240,300-380 (3x), edited 1-120", "alpha.py  read 1-40 (2x)", "zeta.py  read 1-40 (1x)"]


def test_the_block_has_one_factual_line_and_only_when_something_was_read():
    seen: dict = {}
    assert FS.block(seen, {}) == []
    FS.note_read(seen, "story.py", 1, 40)
    out = FS.block(seen, {})
    assert out[0].startswith("FILES SO FAR") and out[1] == "  story.py  read 1-40 (1x)"
    assert out[-1] == "Files above were already read in this stage; read only the ranges you still need."
    assert sum("already read" in ln for ln in out) == 1
    assert FS.block({}, {"a.py": [[1, 5]]}) == []                      # edits alone are the cursor's business, and nothing was read


def test_the_block_is_capped_at_twelve_files_and_seven_hundred_characters():
    seen: dict = {}
    for i in range(30):
        FS.note_read(seen, f"some/long/folder/name/file_{i:02d}.py", 1, 100)
        FS.note_read(seen, f"some/long/folder/name/file_{i:02d}.py", 300, 399)
    out = FS.block(seen, {})
    text = "\n".join(out)
    assert len(text) <= FS.MAX_CHARS == 700 and len([ln for ln in out if ln.startswith("  ")]) <= FS.MAX_FILES == 12
    assert "more files" in text and out[-1].startswith("Files above")
    few: dict = {}
    for i in range(20):
        FS.note_read(few, f"f{i}.py", 1, 9)
    assert len([ln for ln in FS.block(few, {}) if ln.startswith("  f")]) == 12


# ── persistence ─────────────────────────────────────────────────────────────

def test_seen_is_saved_with_the_stage_and_junk_or_an_old_session_loads_empty():
    st = stage()
    FS.note_read(st.seen, "story.py", 1, 80)
    FS.note_edit(st.seen, "story.py")
    plan = ST.StagePlan([st], 0, "model", "t")
    back = ST.StagePlan.from_json(plan.to_json()).stages[0]
    assert back.seen == {"story.py": {"read": [[1, 80]], "reads": 1, "last": "edited"}}
    data = plan.to_json()
    del data["stages"][0]["seen"]
    assert ST.StagePlan.from_json(data).stages[0].seen == {}
    for junk in ("x", [1], {"a.py": 5}, {"a.py": {"read": "no", "reads": "many", "last": 7}}):
        data["stages"][0]["seen"] = junk
        got = ST.StagePlan.from_json(data).stages[0].seen
        assert isinstance(got, dict) and all(isinstance(v["read"], list) and isinstance(v["reads"], int) for v in got.values())


# ── wiring ──────────────────────────────────────────────────────────────────

def test_real_reads_and_writes_of_a_running_stage_are_recorded(tmp_path):
    steps = [call("read_file", path="story.py", start=1, end=12),
             call("read_file", path="story.py", start=1, end=12),                  # the placeholder: counted, no new range
             call("read_file", path="story.py", start=30, end=40),
             call("replace_lines", path="story.py", start=3, end=4, new_text="three\n"),
             say("done")]
    eng, repo, _out = new_engine(tmp_path, FakeLLM(steps))
    write_lines(repo, "story.py", 60)
    st = stage()
    ST.run_stages(eng, ST.StagePlan([st], 0, "model", "t"))
    row = st.seen["story.py"]
    assert row["read"] == [[1, 12], [30, 40]] and row["reads"] == 3 and row["last"] == "edited"
    assert eng.stage_seen is None


def test_a_read_cut_by_the_cap_records_only_the_lines_that_were_shown(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    write_lines(repo, "story.py", 400)
    eng.tools.read_max_chars = 1000
    eng.stage_seen, eng.stage_goal = {}, "g"
    eng.tools.read_file("story.py")
    (a, b), = eng.stage_seen["story.py"]["read"]
    assert a == 1 and 20 <= b < 40


def test_reads_outside_a_stage_are_not_recorded(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    write_lines(repo, "story.py", 20)
    eng.tools.read_file("story.py")
    assert eng.stage_seen is None


def test_the_second_segment_of_a_stage_sees_the_block_in_verified_state_and_notes(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    write_lines(repo, "story.py", 406)
    st = stage(edited={"story.py": [[1, 120]]})
    eng.stage_obj = st
    first = verified.build(eng, run_tests=False)
    assert "FILES SO FAR" not in first.text and ST._Runner(eng, ST.StagePlan([st], 0, "model", "t"), None).notes_for(st).count("FILES SO FAR") == 0
    FS.note_read(st.seen, "story.py", 1, 240)
    FS.note_read(st.seen, "story.py", 300, 380)
    FS.note_edit(st.seen, "story.py")
    state = verified.build(eng, run_tests=False)
    assert "story.py  read 1-240,300-380 (2x), edited 1-120" in state.text and "Files above were already read in this stage" in state.text
    assert state.text.index("STAGE CURSOR") < state.text.index("FILES SO FAR") < state.text.index("\nFILES", state.text.index("FILES SO FAR") + 1)
    assert len(state.sections["FILES SO FAR"]["text"]) <= verified.SECTION_CAP["FILES SO FAR"] == 700
    notes = ST._Runner(eng, ST.StagePlan([st], 0, "model", "t"), None).notes_for(st)
    assert "story.py  read 1-240,300-380 (2x), edited 1-120" in notes and notes.index("Continue at:") < notes.index("FILES SO FAR")
    eng.stage_obj = None
    assert "FILES SO FAR" not in verified.build(eng, run_tests=False).text
