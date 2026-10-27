"""Measured live (a copy of a game project, 15 minutes): 'In dialogue.py, rewrite the spoken lines ... Work through the whole file section by
section.' fired the broad rule and the re-split produced 'Part 1/6: main.py', 'Part 2/6: PLAN-v2.md' ... - it split the REPOSITORY instead of
the 687-line file the task names, the model read unrelated files and dialogue.py had zero changed lines. Now: the files a stage's goal NAMES are
its scope (1-3 files), a named file above coding.range_split_lines is split deterministically (no model call) into consecutive line-range parts,
the scope gate / cursor / read notes speak in line ranges, and later parts follow the line shifts of earlier ones. FakeLLM only."""

from __future__ import annotations

import json

from agent.codeagent import cursor as cursor_mod, stages as ST

from tests.test_codeagent_engine import FakeLLM, call, say
from tests.test_codeagent_stages import cfg, new_engine, request_text, subs_json

TASK = ("In dialogue.py, rewrite the spoken lines so they are plain and easy to read, keep every speaker and every choice. "
        "Work through the whole file section by section.")
NOT_USED = "NOT USED: no model call is allowed for a split that names a file"


def prose(n, blank_every=7):
    """A valid python file of n lines: a list of string lines with a blank line every few lines (section boundaries)."""
    rows = ["LINES = ["]
    while len(rows) < n - 1:
        rows.append("" if (len(rows) % blank_every) == 0 else f'    "line {len(rows):04d}: the old man says something about the weather",')
    rows.append("]")
    return "\n".join(rows[:n]) + "\n"


class Recorder(FakeLLM):
    """FakeLLM that keeps the background (split) prompts."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.asked = []

    def chat(self, messages, model=None, stream=False, background=False, extra_options=None, **kw):
        self.asked.append(messages)
        return super().chat(messages, model, stream, background, extra_options, **kw)


def game(tmp_path, steps=(), background=(NOT_USED,), config=None, dialogue=687, extra=None, task=TASK, llm=None):
    llm = llm or FakeLLM(list(steps), background=list(background))
    eng, repo, out = new_engine(tmp_path, llm, config)
    (repo / "dialogue.py").write_text(prose(dialogue))
    (repo / "main.py").write_text("import dialogue\n\n\ndef main():\n    print(dialogue.LINES[0])\n" + "x = 1\n" * 300)
    (repo / "PLAN.md").write_text("# plan\n" + "- a step\n" * 400)
    (repo / "PLAN-v2.md").write_text("# plan v2\n" + "- another step\n" * 400)
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_game.py").write_text("def test_it():\n    assert True\n")
    for name, n in (extra or {}).items():
        (repo / name).write_text(prose(n) if name.endswith(".py") else "# t\n" + "word word word\n" * (n - 1))
    plan = ST.StagePlan([ST.Stage(1, "Rewrite the dialogue", task)], 0, "single", task)
    return llm, eng, repo, plan


def split_now(eng, plan, source="broad"):
    runner = ST.make_runner(eng, plan)
    assert runner.resplit(plan.stages[0], source)
    return runner


def spans(plan, name="dialogue.py"):
    return [s.lines for s in plan.stages if s.scope == [name] and s.lines]


# ── 1. the measured failure ─────────────────────────────────────────────────

def test_the_failing_task_is_split_into_line_ranges_of_dialogue_py_without_a_model_call(tmp_path):
    llm, eng, repo, plan = game(tmp_path)
    runner = ST.make_runner(eng, plan)
    assert runner.broad_applies(plan.stages[0])                           # 'whole' + 'rewrite' still fires the broad rule
    assert runner.resplit(plan.stages[0], "broad")
    assert llm.background == [NOT_USED] and llm.calls == []                # no model call at all
    parts = plan.stages
    assert 4 <= len(parts) <= 6 and all(s.scope == ["dialogue.py"] and s.depth == 1 and s.parent == 1 for s in parts)
    ranges = [s.lines for s in parts]
    assert ranges[0][0] == 1 and ranges[-1][1] == 687                       # complete ...
    assert all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:]))        # ... and contiguous
    assert all(100 <= b - a + 1 <= 175 for a, b in ranges)
    n = len(parts)
    for i, s in enumerate(parts, 1):
        a, b = s.lines
        assert s.title == f"Part {i}/{n}: dialogue.py lines {a}-{b}"
        assert f"Work ONLY on dialogue.py lines {a}-{b} (the rest of the file is handled by other parts). In dialogue.py, rewrite the spoken lines" in s.goal
        assert s.goal.startswith("Overall task:") and s.lines_total == 687
    everything = " ".join(s.title + s.goal + " ".join(s.scope) for s in parts)
    assert "main.py" not in everything and "PLAN" not in everything
    ev = eng.events.of("stage_resplit")
    assert len(ev) == 1 and ev[0]["source"] == "broad" and ev[0]["to"] == list(range(1, n + 1))


def test_part_boundaries_follow_blank_lines(tmp_path):
    _llm, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    rows = (repo / "dialogue.py").read_text().split("\n")
    for s in plan.stages[1:]:
        assert rows[s.lines[0] - 2] == ""                                  # a part starts right after a blank line


def test_python_parts_prefer_top_level_definitions(tmp_path):
    body = "".join(f"def f{i}(x):\n    return x + {i}\n\n\n" for i in range(110))        # 440 lines
    _llm, eng, repo, plan = game(tmp_path)
    (repo / "dialogue.py").write_text(body)
    split_now(eng, plan)
    rows = body.split("\n")
    assert len(plan.stages) >= 2 and all(rows[s.lines[0] - 1].startswith("def ") for s in plan.stages[1:])


def test_a_small_named_file_is_not_split_and_the_broad_rule_leaves_it_alone(tmp_path):
    llm, eng, repo, plan = game(tmp_path, dialogue=150)
    runner = ST.make_runner(eng, plan)
    assert not runner.broad_applies(plan.stages[0])
    assert runner.resplit(plan.stages[0]) is False and len(plan.stages) == 1           # a cap re-split finds nothing to divide either
    assert eng.events.of("stage_resplit") == []


def test_the_threshold_is_configurable(tmp_path):
    _l, eng, repo, plan = game(tmp_path, config=cfg(range_split_lines=100), dialogue=150)
    runner = ST.make_runner(eng, plan)
    assert runner.broad_applies(plan.stages[0]) and runner.resplit(plan.stages[0], "broad")
    assert len(plan.stages) == 2 and plan.stages[0].lines[0] == 1 and plan.stages[-1].lines[1] == 150 and plan.stages[1].lines[0] == plan.stages[0].lines[1] + 1
    (tmp_path / "off").mkdir()
    _l, eng, repo, plan = game(tmp_path / "off", config=cfg(range_split_lines=0))
    assert not ST.make_runner(eng, plan).broad_applies(plan.stages[0])               # 0 = no range split (and a named file is no broad sweep either)


def test_a_huge_file_gets_at_most_eight_parts(tmp_path):
    _l, eng, repo, plan = game(tmp_path, dialogue=5000)
    split_now(eng, plan)
    assert len(plan.stages) == 8 and plan.stages[0].lines[0] == 1 and plan.stages[-1].lines[1] == 5000
    assert all(b.lines[0] == a.lines[1] + 1 for a, b in zip(plan.stages, plan.stages[1:]))


def test_two_named_files_get_parts_across_both_in_file_order(tmp_path):
    task = "Rewrite the whole of dialogue.py and then story.py so every line is plain, section by section."
    _l, eng, repo, plan = game(tmp_path, task=task, extra={"story.py": 400})
    split_now(eng, plan)
    names = [s.scope[0] for s in plan.stages]
    assert set(names) == {"dialogue.py", "story.py"} and names == sorted(names, key=["dialogue.py", "story.py"].index)
    for f, total in (("dialogue.py", 687), ("story.py", 400)):
        r = spans(plan, f)
        assert r[0][0] == 1 and r[-1][1] == total and all(b[0] == a[1] + 1 for a, b in zip(r, r[1:])), f
    assert len(plan.stages) <= 8 and [s.title.split(":")[0] for s in plan.stages] == [f"Part {i}/{len(plan.stages)}" for i in range(1, len(plan.stages) + 1)]
    assert "main.py" not in " ".join(s.goal for s in plan.stages)


def test_a_small_file_beside_a_big_one_is_one_whole_file_part(tmp_path):
    task = "Rewrite the whole of story.py and dialogue.py so every line is plain."
    _l, eng, repo, plan = game(tmp_path, task=task, extra={"story.py": 120})
    split_now(eng, plan)
    first = plan.stages[0]
    assert first.scope == ["story.py"] and first.lines is None and first.title == f"Part 1/{len(plan.stages)}: story.py"
    assert all(s.scope == ["dialogue.py"] and s.lines for s in plan.stages[1:])


def test_a_cap_resplit_of_a_started_stage_skips_what_is_finished(tmp_path):
    _l, eng, repo, plan = game(tmp_path)
    st = plan.stages[0]
    st.edited = {"dialogue.py": [[1, 300]]}
    split_now(eng, plan, "model")
    assert plan.stages[0].lines[0] == 301 and plan.stages[-1].lines[1] == 687


# ── 2. which files a goal names ─────────────────────────────────────────────

def repo_of(tmp_path, names):
    for n in names:
        (tmp_path / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / n).write_text("x = 1\n" * 30)
    return ST._repo_files(tmp_path)


def test_named_files(tmp_path):
    repo = repo_of(tmp_path, ["dialogue.py", "main.py", "PLAN.md", "PLAN-v2.md", "pkg/util.py", "mydialogue.py.bak.txt", "world.py"])
    named = lambda text, title="t": ST.named_files(ST.Stage(1, title, text), repo)
    assert named("In dialogue.py, rewrite it") == ["dialogue.py"]
    assert named("Overall task: fix main.py\n\nrewrite dialogue.py, please.") == ["dialogue.py"]            # the Overall head is not the stage's goal
    assert named("Edit PLAN-v2.md") == ["PLAN-v2.md"] and named("Edit PLAN.md.") == ["PLAN.md"]
    assert named("Edit pkg/util.py and util.py") == ["pkg/util.py"] and named("Edit util.py") == ["pkg/util.py"]
    assert named("Clean up the dialogue module") == ["dialogue.py"] and named("Clean up the world module and the harbour module") == ["world.py"]
    assert named("Polish", title="Part 1/2: dialogue.py lines 1-10") == ["dialogue.py"]                  # the title counts
    assert named("Rewrite mydialogue.py and redialogue.py and x/dialogue.pyc") == []                     # no partial matches
    assert named("Rewrite all texts") == []
    assert named("Rewrite dialogue.py, main.py, world.py and PLAN.md") == []                             # more than 3 files: not a named-file goal


# ── 3. a goal that names no file keeps the old behaviour ────────────────────

def test_a_goal_without_a_file_name_still_uses_the_model_and_the_whole_repo(tmp_path):
    answer = subs_json(("Part A", "Clear the texts of the first files in the repository", ["main.py"]),
                       ("Part B", "Clear the texts of the planning files in the repository", ["PLAN.md", "PLAN-v2.md"]))
    llm = Recorder([], background=[answer])
    _l, eng, repo, plan = game(tmp_path, task="Rewrite all texts of the game so they are plain.", llm=llm)
    runner = ST.make_runner(eng, plan)
    assert runner.broad_applies(plan.stages[0]) and runner.resplit(plan.stages[0], "broad")
    assert [s.scope for s in plan.stages] == [["main.py"], ["PLAN.md", "PLAN-v2.md"]] and all(s.lines is None for s in plan.stages)
    assert "main.py (" in llm.asked[0][1]["content"] and "dialogue.py (" in llm.asked[0][1]["content"]


def test_with_named_files_the_model_sees_only_those_and_may_not_use_others(tmp_path):
    task = "Polish a.md and b.md so the wording is clear."
    bad = subs_json(("Part A", "Clear the texts in a.md and the planning files", ["a.md", "PLAN.md"]), ("Part B", "Clear the texts in b.md and more", ["b.md"]))
    good = subs_json(("Part A", "Clear the texts in a.md and keep them connected", ["a.md"]), ("Part B", "Clear the texts in b.md and keep them connected", ["b.md"]))
    llm = Recorder([], background=[bad, good])
    _l, eng, repo, plan = game(tmp_path, task=task, llm=llm, extra={"a.md": 350, "b.md": 350}, config=cfg(range_split_lines=0))
    runner = ST.make_runner(eng, plan)
    assert runner.resplit(plan.stages[0], "broad")                          # the first answer names PLAN.md: rejected, the deterministic fallback takes over
    assert len(llm.asked) == 1 and "a.md (" in llm.asked[0][1]["content"] and "b.md (" in llm.asked[0][1]["content"]
    prompt = llm.asked[0][1]["content"]
    assert "main.py" not in prompt and "PLAN" not in prompt and "dialogue.py" not in prompt.split("REPO FILES")[1]
    assert sorted(f for s in plan.stages for f in s.scope) == ["a.md", "b.md"]


def test_the_fallback_split_stays_inside_the_named_files(tmp_path):
    task = "Polish a.md, b.md and c.md so the wording is clear."
    _l, eng, repo, plan = game(tmp_path, task=task, background=["not json"], extra={"a.md": 250, "b.md": 250, "c.md": 250})
    runner = ST.make_runner(eng, plan)
    runner.reads[1] = {"main.py": 9, "PLAN.md": 7}                           # the model read other files a lot: irrelevant now
    assert runner.resplit(plan.stages[0], "broad")
    scopes = sorted(f for s in plan.stages for f in s.scope)
    assert scopes == ["a.md", "b.md", "c.md"] and all(s.lines is None for s in plan.stages)
    assert "main.py" not in " ".join(s.goal + s.title for s in plan.stages)


# ── 4. persistence ──────────────────────────────────────────────────────────

def test_lines_survive_a_json_round_trip_and_an_old_session_loads_without_them(tmp_path):
    _l, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    back = ST.StagePlan.from_json(json.loads(json.dumps(plan.to_json())))
    assert [(s.lines, s.lines_total) for s in back.stages] == [(s.lines, s.lines_total) for s in plan.stages] and back.stages[0].lines[0] == 1
    old = plan.to_json()
    for d in old["stages"]:
        del d["lines"], d["lines_total"]
    legacy = ST.StagePlan.from_json(old)
    assert all(s.lines is None and s.lines_total == 0 for s in legacy.stages)
    junk = plan.to_json()
    junk["stages"][0]["lines"], junk["stages"][1]["lines"], junk["stages"][2]["lines"] = "x", [5, 2], [3]
    assert [s.lines for s in ST.StagePlan.from_json(junk).stages[:3]] == [None, None, None]


# ── 5. the cursor speaks in ranges, later parts follow line shifts ──────────

def ranged(lines, edited=None, total=687, scope="dialogue.py"):
    return ST.Stage(2, f"Part 1/5: {scope} lines {lines[0]}-{lines[1]}", f"Overall task: x\n\nWork ONLY on {scope} lines {lines[0]}-{lines[1]} (the rest of the file is handled by other parts). Rewrite it.",
                    scope=[scope], parent=1, depth=1, lines=tuple(lines), lines_total=total, edited=edited or {})


def test_cursor_texts_name_the_range(tmp_path):
    (tmp_path / "dialogue.py").write_text(prose(687))
    st = ranged((1, 150))
    assert cursor_mod.lines(cursor_mod.cursor(st, tmp_path)) == ["dialogue.py lines 1-150: next unedited line 1"]
    st.edited = {"dialogue.py": [[1, 36]]}
    cur = cursor_mod.cursor(st, tmp_path)
    assert cursor_mod.lines(cur) == ["dialogue.py lines 1-150: next unedited line 37"]
    assert cursor_mod.note(cur) == "Continue at: dialogue.py lines 1-150: next unedited line 37"
    assert cursor_mod.next_chunk(cur) == "Rewrite lines 37-116 of dialogue.py now (read them first if needed)."
    st.edited = {"dialogue.py": [[1, 150]]}
    assert cursor_mod.lines(cursor_mod.cursor(st, tmp_path)) == ["dialogue.py lines 1-150: done"]
    assert cursor_mod.next_chunk(cursor_mod.cursor(st, tmp_path)) == ""
    st2 = ranged((151, 300), {"dialogue.py": [[1, 40], [151, 160], [200, 210]]})            # edits outside the range do not count; a gap stops the cursor
    assert cursor_mod.lines(cursor_mod.cursor(st2, tmp_path)) == ["dialogue.py lines 151-300: next unedited line 161"]
    st3 = ranged((151, 300), {"dialogue.py": [[100, 220]]})                                  # an edit that starts before the range and runs into it
    assert cursor_mod.lines(cursor_mod.cursor(st3, tmp_path)) == ["dialogue.py lines 151-300: next unedited line 221"]


def test_overlap_rule():
    assert cursor_mod.overlaps([[5, 9]], (1, 5)) and cursor_mod.overlaps([[1, 1]], (1, 5)) and cursor_mod.overlaps([[0 + 3, 400]], (10, 20))
    assert not cursor_mod.overlaps([[6, 9]], (1, 5)) and not cursor_mod.overlaps([], (1, 5)) and not cursor_mod.overlaps(None, (1, 5))


def test_a_later_part_follows_the_lines_an_earlier_part_added(tmp_path):
    """Approach: a ranged stage keeps its range and the file length (lines_total) the range was valid for. When the file length differs, the
    difference is added to the start AND end of a part that has written nothing yet (earlier parts changed the file before it), and to the
    END only of a part that already wrote (its own edits)."""
    _l, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    p1, p2, p3 = plan.stages[:3]
    p1_end = p1.lines[1]
    f = repo / "dialogue.py"
    rows = f.read_text().split("\n")
    second_first = rows[p2.lines[0] - 1]
    third_first = rows[p3.lines[0] - 1]
    p2_old = p2.lines
    grown = rows[:20] + [f'    "extra line {i}",' for i in range(30)] + rows[20:]            # part 1 grows by 30 lines
    f.write_text("\n".join(grown))
    p1.edited = {"dialogue.py": [[1, 60]]}
    cursor_mod.sync_range(p1, repo)
    assert p1.lines == (1, p1_end + 30) and p1_end == p2_old[0] - 1 and p1.lines_total == 717
    cursor_mod.sync_range(p2, repo)                                                          # part 2 has written nothing: it moves as a whole
    assert p2.lines == (p2_old[0] + 30, p2_old[1] + 30) and grown[p2.lines[0] - 1] == second_first
    cursor_mod.sync_range(p3, repo)
    assert grown[p3.lines[0] - 1] == third_first
    assert f"dialogue.py lines {p2.lines[0]}-{p2.lines[1]}" in p2.title and f"dialogue.py lines {p2.lines[0]}-{p2.lines[1]} (" in p2.goal
    assert f"lines {p2_old[0]}-{p2_old[1]}" not in p2.goal
    cursor_mod.sync_range(p2, repo)                                                          # idempotent
    assert p2.lines == (p2_old[0] + 30, p2_old[1] + 30)
    p2.edited = {"dialogue.py": [[p2.lines[0], p2.lines[0] + 10]]}                           # part 2 writes and shrinks its own region by 5 lines
    f.write_text("\n".join(grown[:p2.lines[0] + 20] + grown[p2.lines[0] + 25:]))
    cursor_mod.sync_range(p2, repo)
    assert p2.lines == (p2_old[0] + 30, p2_old[1] + 25)


def test_the_start_notes_tell_the_read_window_and_the_cursor(tmp_path):
    _l, eng, repo, plan = game(tmp_path)
    runner = split_now(eng, plan)
    p2 = plan.stages[1]
    a, b = p2.lines
    notes = runner.notes_for(p2)
    assert f"read dialogue.py lines {a}-{b} only (start/end)" in notes
    assert f"Continue at: dialogue.py lines {a}-{b}: next unedited line {a}" in notes
    assert "later stages, NOT yours now" in notes


def test_the_stage_cursor_section_speaks_in_ranges(tmp_path):
    from agent.codeagent import verified

    _l, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    p1 = plan.stages[0]
    p1.edited = {"dialogue.py": [[1, 36]]}
    eng.stage_obj = p1
    text = "\n".join(verified._stage_cursor(eng)[0])
    assert f"dialogue.py lines 1-{p1.lines[1]}: next unedited line 37" in text


# ── 6. the scope gate asks for a change INSIDE the range ────────────────────

def test_the_gate_needs_a_change_that_overlaps_the_range(tmp_path):
    _l, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    p1 = plan.stages[0]
    a, b = p1.lines
    eng.stage_obj, eng.stage_edited, eng.stage_scope, eng.stage_goal, eng.stage_waived = p1, p1.edited, ["dialogue.py"], "goal", set()
    eng.stage_changed0, eng.stage_marks = set(), {}
    label = f"dialogue.py lines {a}-{b}"
    assert eng.scope_missing() == [label]
    eng._record_write("dialogue.py", "\n".join(["x"] * 300), "\n".join(["x"] * 299 + ["y"]))      # a change at line 300: outside part 1
    assert eng.scope_missing() == [label]
    rows = (repo / "dialogue.py").read_text().split("\n")
    eng._record_write("dialogue.py", "\n".join(rows), "\n".join(rows[:4] + ["changed"] + rows[5:]))      # line 5: inside
    assert eng.scope_missing() == []
    p1.edited.clear()
    eng.stage_waived = {"dialogue.py"}                                                          # an explicit waiver naming the file
    assert eng.scope_missing() == []


def test_a_change_outside_the_range_is_nudged_and_the_gate_names_the_range(tmp_path):
    steps = [call("replace_lines", path="dialogue.py", start=600, end=600, new_text='    "plain line at 600",'), say("Done."),
             call("replace_lines", path="dialogue.py", start=5, end=5, new_text='    "plain line at 5",'), say("Now done.")]
    llm, eng, repo, plan = game(tmp_path, steps, background=[NOT_USED])
    runner = split_now(eng, plan)
    p1 = plan.stages[0]
    assert ST.run_stages(eng, ST.StagePlan([p1], 0, "headings", "t"), max_stages=1).verdict == "done"
    nudge = [m["content"] for m in llm.calls[2]["snapshot"] if m["role"] == "system" and str(m["content"]).startswith("You are not done")]
    assert nudge and f"dialogue.py lines 1-{p1.lines[1]}" in nudge[0] and "lines 1-" in nudge[0]
    assert (repo / "dialogue.py").read_text().split("\n")[4].strip() == '"plain line at 5",' and p1.edited["dialogue.py"]


def test_the_harness_close_rule_needs_the_range_not_just_the_file(tmp_path):
    from agent.codeagent import supervisor

    _l, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    p1 = plan.stages[0]
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    p1.files = ["dialogue.py"]
    p1.edited = {"dialogue.py": [[600, 600]]}
    assert not sup._scope_satisfied(p1)
    p1.edited = {"dialogue.py": [[3, 3]]}
    assert sup._scope_satisfied(p1)
    p1.edited, p1.waived = {}, ["dialogue.py"]
    assert sup._scope_satisfied(p1)


# ── 7. look discipline of a ranged part ─────────────────────────────────────

def stage_ready(eng, st):
    eng.stage_obj, eng.stage_edited, eng.stage_scope, eng.stage_goal = st, st.edited, list(st.scope), "goal"
    eng.lookact.free_steps = 0


def note_count(eng):
    return sum("Outside this part" in str(m.get("content", "")) for m in eng.ctx.messages)


def test_reading_another_file_gets_one_note_and_counts_double(tmp_path):
    _l, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    p1 = plan.stages[0]
    stage_ready(eng, p1)
    eng._run_call({"name": "read_file", "arguments": {"path": "main.py", "start": 1, "end": 10}})
    assert eng.lookact.look_streak == 2 and note_count(eng) == 1
    assert any(f"Outside this part: only dialogue.py lines 1-{p1.lines[1]} are yours" in str(m.get("content")) for m in eng.ctx.messages)
    eng._run_call({"name": "read_file", "arguments": {"path": "PLAN.md", "start": 1, "end": 10}})
    assert eng.lookact.look_streak == 4 and note_count(eng) == 1                                  # the note comes once
    eng._run_call({"name": "read_file", "arguments": {"path": "dialogue.py", "start": 1, "end": 10}})
    assert eng.lookact.look_streak == 5                                                           # the part's own file counts once
    eng._run_call({"name": "read_files", "arguments": {"paths": ["dialogue.py", "main.py"]}})
    assert eng.lookact.look_streak == 7 and eng.events.of("out_of_scope_read")


def test_the_weight_is_configurable_and_other_stages_are_unaffected(tmp_path):
    _l, eng, repo, plan = game(tmp_path, config=cfg(out_of_scope_read_weight=3))
    split_now(eng, plan)
    stage_ready(eng, plan.stages[0])
    eng._run_call({"name": "read_file", "arguments": {"path": "main.py", "start": 1, "end": 10}})
    assert eng.lookact.look_streak == 3
    (tmp_path / "plain").mkdir()
    _l, eng2, repo2, plan2 = game(tmp_path / "plain")
    plain = ST.Stage(1, "t", "Look at things", scope=["dialogue.py"])                              # no range: reading main.py is ordinary
    stage_ready(eng2, plain)
    eng2._run_call({"name": "read_file", "arguments": {"path": "main.py", "start": 1, "end": 10}})
    assert eng2.lookact.look_streak == 1 and note_count(eng2) == 0
    eng2.stage_obj = None                                                                         # outside any stage nothing changes either
    eng2._run_call({"name": "read_file", "arguments": {"path": "main.py", "start": 1, "end": 10}})
    assert note_count(eng2) == 0


def test_orientation_reads_of_a_new_segment_stay_free(tmp_path):
    _l, eng, repo, plan = game(tmp_path)
    split_now(eng, plan)
    stage_ready(eng, plan.stages[0])
    eng.lookact.free_steps = 6
    eng._run_call({"name": "read_file", "arguments": {"path": "main.py", "start": 1, "end": 10}})
    assert eng.lookact.look_streak == 0 and note_count(eng) == 1                                  # still told, but not counted
