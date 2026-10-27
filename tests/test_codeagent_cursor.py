"""The stage cursor and the chunk protocol (cursor.py): a rewrite-type stage over big files gets a working-method note, a per-file record of
the line ranges it edited (restored after a restart), a STAGE CURSOR section and an earlier reading pause that names the next chunk.
Measured: four 400-700 line files, 69 steps, 45 reads, zero edits, three restarts that each surveyed everything again. FakeLLM only."""

from __future__ import annotations

from agent.codeagent import cursor as CU, lookact as L, stages as ST, verified

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import cfg, new_engine, request_text, write_lines

REWRITE = "Rewrite the cryptic texts of these files so they are easy to understand"


def stage(goal=REWRITE, files=("story.py", "dialogue.py"), title="Texts", **kw):
    return ST.Stage(1, title, goal, files=list(files), **kw)


def lines(n, prefix="l"):
    return [f"{prefix}{i}" for i in range(1, n + 1)]


# ── ranges ──────────────────────────────────────────────────────────────────

def test_merge_joins_overlapping_and_adjacent_ranges():
    assert CU.merge([[5, 9], [1, 3], [4, 4], [8, 12], [20, 20], ["x", 1]]) == [[1, 12], [20, 20]]
    assert CU.fmt([[1, 120], [150, 200]]) == "1-120,150-200" and CU.fmt([[7, 7]]) == "7"


def test_a_replaced_block_is_marked_and_a_new_file_is_marked_whole():
    old = lines(10)
    new = old[:3] + ["a", "b"] + old[5:]                       # lines 4-5 replaced by two lines
    assert CU.apply_change([], old, new) == [[4, 5]]
    assert CU.apply_change(None, None, lines(6)) == [[1, 6]]


def test_later_ranges_move_when_an_edit_changes_the_line_count():
    v0 = lines(20)
    v1 = v0[:1] + ["x", "y", "z"] + v0[2:]                     # line 2 became 3 lines: lines 2-4 edited, the rest moved by +2
    r1 = CU.apply_change([], v0, v1)
    assert r1 == [[2, 4]]
    v2 = v1[:13] + ["new"] + v1[14:]                           # line 14 replaced
    r2 = CU.apply_change(r1, v1, v2)
    assert r2 == [[2, 4], [14, 14]]
    v3 = ["top"] + v2                                          # a new first line: every mark moves down by one
    assert CU.apply_change(r2, v2, v3) == [[1, 1], [3, 5], [15, 15]]


def test_deleted_lines_leave_a_mark_where_they_were():
    old = lines(10)
    assert CU.apply_change([], old, old[:4] + old[6:]) == [[5, 5]]


def test_next_is_the_end_of_the_block_edited_from_line_one():
    assert CU.next_line([[1, 200], [250, 260]]) == 201
    assert CU.next_line([[10, 20]]) == 1 and CU.next_line([]) == 1


# ── the stage objects ───────────────────────────────────────────────────────

def test_stage_edited_is_saved_and_an_old_session_without_it_loads(tmp_path):
    st = stage()
    st.edited = {"story.py": [[1, 80], [90, 95]]}
    plan = ST.StagePlan([st], 0, "model", "t")
    back = ST.StagePlan.from_json(plan.to_json()).stages[0]
    assert back.edited == {"story.py": [[1, 80], [90, 95]]}
    data = plan.to_json()
    del data["stages"][0]["edited"]
    assert ST.StagePlan.from_json(data).stages[0].edited == {}
    data["stages"][0]["edited"] = "junk"
    assert ST.StagePlan.from_json(data).stages[0].edited == {}


def test_cursor_reports_edited_next_total_and_not_started_files(tmp_path):
    write_lines(tmp_path, "story.py", 406)
    st = stage(edited={"story.py": [[1, 120], [121, 200]]})
    cur = CU.cursor(st, tmp_path)
    assert cur["story.py"] == {"edited": "1-200", "next": 201, "total": 406}
    assert cur["dialogue.py"]["edited"] == "" and cur["dialogue.py"]["next"] == 1
    assert CU.note(cur) == "Continue at: story.py line 201 (edited 1-200 of 406); dialogue.py not started"
    assert CU.next_chunk(cur) == "Rewrite lines 201-280 of story.py now (read them first if needed)."


def test_next_chunk_skips_finished_files_and_stops_at_the_end_of_the_file(tmp_path):
    write_lines(tmp_path, "story.py", 100)
    write_lines(tmp_path, "dialogue.py", 50)
    st = stage(edited={"story.py": [[1, 100]], "dialogue.py": [[1, 20]]})
    assert CU.next_chunk(CU.cursor(st, tmp_path)) == "Rewrite lines 21-50 of dialogue.py now (read them first if needed)."
    st.edited["dialogue.py"] = [[1, 50]]
    assert CU.next_chunk(CU.cursor(st, tmp_path)) == ""


# ── the working-method note ─────────────────────────────────────────────────

def test_the_method_note_is_for_rewrite_goals_that_name_files():
    assert CU.is_chunk_stage(stage()) and CU.is_chunk_stage(stage("Überarbeite die Texte")) and CU.is_chunk_stage(stage("POLISH the dialogue"))
    assert not CU.is_chunk_stage(stage("Add a feature that counts words")) and not CU.is_chunk_stage(stage(files=()))
    assert CU.is_chunk_stage(ST.Stage(1, "t", REWRITE, scope=["a.md"]))
    assert len(CU.METHOD.split()) <= 60


def notes(st, tmp_path):
    eng, _repo, _out = new_engine(tmp_path, FakeLLM([]))
    runner = ST._Runner(eng, ST.StagePlan([st], 0, "model", "t"), None)
    return runner.notes_for(st)


def test_notes_carry_the_method_once_and_the_cursor(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    write_lines(repo, "story.py", 406)
    st = stage(edited={"story.py": [[1, 200]]})
    runner = ST._Runner(eng, ST.StagePlan([st], 0, "model", "t"), None)
    text = runner.notes_for(st)
    assert text.count("WORKING METHOD") == 1 and "Continue at: story.py line 201 (edited 1-200 of 406); dialogue.py not started" in text


def test_a_feature_stage_gets_no_method_note(tmp_path):
    text = notes(stage("Add a feature that counts words"), tmp_path)
    assert "WORKING METHOD" not in text
    assert notes(ST.Stage(1, "t", REWRITE), tmp_path) == ""            # no files: nothing changes


# ── recording writes ────────────────────────────────────────────────────────

def test_every_file_tool_write_of_a_running_stage_is_recorded(tmp_path):
    steps = [call("read_file", path="story.py", start=1, end=12),
             call("replace_lines", path="story.py", start=3, end=4, new_text="three\nfour\nfive\n"),
             call("edit_file", path="story.py", search="line number 0009 of the text", replace="nine"),
             call("write_file", path="dialogue.py", content="a\nb\n"),
             say("done")]
    eng, repo, _out = new_engine(tmp_path, FakeLLM(steps))
    write_lines(repo, "story.py", 20)
    st = stage(goal="Rewrite story.py and dialogue.py", files=("story.py", "dialogue.py"))
    ST.run_stages(eng, ST.StagePlan([st], 0, "model", "t"))
    assert st.edited["story.py"] == [[3, 5], [11, 11]]          # lines 3-4 became 3 lines (+1), so file line 10 is now line 11
    assert st.edited["dialogue.py"] == [[1, 2]]
    assert eng.stage_edited is None and eng.stage_obj is None and not eng.stage_chunk


def test_writes_outside_a_stage_are_not_recorded(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([call("write_file", path="a.py", content="x = 1\n"), say("ok")]))
    eng.turn("write a.py")
    assert eng.stage_edited is None and (repo / "a.py").exists()


# ── VERIFIED STATE ──────────────────────────────────────────────────────────

def test_verified_state_has_a_stage_cursor_section_only_inside_a_stage_with_files(tmp_path):
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]))
    write_lines(repo, "story.py", 406)
    assert "STAGE CURSOR" not in verified.build(eng, run_tests=False).text
    eng.stage_obj = stage(edited={"story.py": [[1, 200]]})
    state = verified.build(eng, run_tests=False)
    assert "STAGE CURSOR" in state.text and "story.py line 201 (edited 1-200 of 406)" in state.text and "dialogue.py not started" in state.text
    text = state.sections["STAGE CURSOR"]["text"]
    assert len(text) <= verified.SECTION_CAP["STAGE CURSOR"]
    assert state.text.index("OPEN ACCEPTANCE ITEMS") < state.text.index("STAGE CURSOR") < state.text.index("FILES")
    eng.stage_obj = stage(files=())
    assert "STAGE CURSOR" not in verified.build(eng, run_tests=False).text


def test_the_cursor_section_respects_its_size_cap(tmp_path):
    eng, _repo, _out = new_engine(tmp_path, FakeLLM([]))
    files = [f"some/deep/folder/with/a/long/name/file_number_{i}.py" for i in range(30)]
    eng.stage_obj = stage(files=files)
    state = verified.build(eng, run_tests=False)
    assert len(state.sections["STAGE CURSOR"]["text"]) <= 500 and len(state.text) <= verified.MAX_CHARS


def test_the_cursor_is_dropped_before_the_open_items_under_pressure():
    secs = [("TESTS", "t" * 100), ("OPEN ITEMS", "o" * 100), ("STAGE CURSOR", "c" * 100), ("DEAD ENDS", "d" * 100), ("FILES", "f" * 100)]
    kept, dropped = verified.fit(secs, 320)
    assert [n for n, _t in kept] == ["TESTS", "OPEN ITEMS", "STAGE CURSOR"] and dropped == ["FILES", "DEAD ENDS"]
    kept, dropped = verified.fit(secs, 220)
    assert "OPEN ITEMS" in [n for n, _t in kept] and "STAGE CURSOR" in dropped
    assert list(verified.SECTION_CAP).index("OPEN ITEMS") < list(verified.SECTION_CAP).index("STAGE CURSOR") < list(verified.SECTION_CAP).index("DEAD ENDS")


# ── edit pressure ───────────────────────────────────────────────────────────

def test_a_chunk_stage_pauses_reading_at_ten_and_names_the_next_chunk(tmp_path):
    reads = [call("read_file", path="story.py", start=1 + 10 * i, end=10 + 10 * i) for i in range(12)]
    eng, repo, _out = new_engine(tmp_path, FakeLLM([]), cfg())
    write_lines(repo, "story.py", 406)
    eng.stage_obj, eng.stage_goal, eng.stage_chunk = stage(edited={"story.py": [[1, 200]]}), "g", True
    la = eng.lookact
    assert la.block_at == 10
    la.look_streak = 9
    assert la.blocked("read_file", {"path": "story.py"}) is None
    la.look_streak = 10
    msg = la.blocked("run", {"command": "sed -n '1,80p' story.py"})
    assert msg and "Rewrite lines 201-280 of story.py now (read them first if needed)." in msg
    assert la.blocked("replace_lines", {"path": "story.py"}) is None
    assert len(reads) == 12


def test_chunk_look_block_is_configurable_and_other_stages_keep_the_general_limit(tmp_path):
    eng, _repo, _out = new_engine(tmp_path, FakeLLM([]), cfg(chunk_look_block=6))
    eng.stage_goal, eng.stage_chunk, eng.stage_obj = "g", True, stage()
    assert eng.lookact.block_at == 6
    eng.stage_chunk = False
    assert eng.lookact.block_at == L.LOOK_BLOCK_AT
    (tmp_path / "x").mkdir()
    plain, _r, _o = new_engine(tmp_path / "x", FakeLLM([]))
    assert plain.lookact.block_at == L.LOOK_BLOCK_AT
    plain.lookact.look_streak = L.LOOK_BLOCK_AT
    assert "Rewrite lines" not in plain.lookact.blocked("read_file", {"path": "a.py"})


def test_the_runner_sets_the_chunk_mode_and_shows_the_not_started_files(tmp_path):
    llm = FakeLLM([call("write_file", path="story.py", content="s\n"), say("ok")])
    eng, repo, _out = new_engine(tmp_path, llm)
    write_lines(repo, "dialogue.py", 100)
    st = stage(goal="Rewrite story.py and dialogue.py", files=("story.py", "dialogue.py"))
    ST.run_stages(eng, ST.StagePlan([st], 0, "model", "t"))
    first = request_text(llm.calls[0]["snapshot"])
    assert "WORKING METHOD" in first and "Continue at: story.py not started; dialogue.py not started (100 lines)" in first
    assert "STAGE CURSOR" in first
