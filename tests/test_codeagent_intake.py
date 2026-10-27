"""Intake (clarifying questions before a task), the ask_user tool, the supervisor's hint question and the model stage split by default.
FakeLLM with scripted JSON, a fake input function, no model, no real terminal."""

from __future__ import annotations

import json

import pytest

from agent.codeagent import brief as brief_mod, intake as IN, lookact, sessionstore, stages as ST
from agent.codeagent.engine import CodeEngine

from tests.test_codeagent_engine import FakeLLM, call, engine, say
from tests.test_codeagent_stages import new_engine
from tests.test_codeagent_supervisor import _isolated as _sv_isolated, make as sv_make  # noqa: F401 - the autouse fixture


TASK = "make a snake game with a menu, high scores and sound"
LONG_TASK = (TASK + " - it should be playable in the terminal and feel arcade-like, with a short readme at the end. Keep every part in its own"
             " module so that the menu, the scores and the game loop can be tested one by one.")
assert len(LONG_TASK) >= 160
QS = json.dumps({"questions": ["Which language and UI (terminal or window)?", "What should 'sound' cover?"]})


class Recorder(FakeLLM):
    """FakeLLM that also remembers the model and options of every background chat call."""

    def __init__(self, script=(), background=None):
        super().__init__(script, background)
        self.chats = []

    def chat(self, messages, model=None, stream=False, background=False, extra_options=None, **kw):
        self.chats.append({"model": model, "background": background, "opts": extra_options, "messages": messages})
        yield from super().chat(messages, model, stream, background, extra_options, **kw)


def reader(*answers):
    it = iter(answers)
    asked = []

    def read(prompt):
        asked.append(prompt)
        return next(it)

    read.asked = asked
    return read


@pytest.fixture
def human(monkeypatch):
    monkeypatch.setattr(IN, "tty", lambda: True)


# ── clarify ─────────────────────────────────────────────────────────────────

def test_clarify_returns_valid_questions_on_the_main_model(tmp_path):
    llm = Recorder(background=[QS])
    eng, _repo, _ = engine(tmp_path, llm)
    qs = IN.clarify(eng, TASK, cfg=eng.config)
    assert qs == ["Which language and UI (terminal or window)?", "What should 'sound' cover?"]
    c = llm.chats[0]
    assert c["model"] == eng.model == "m" and c["background"] is True and c["opts"]["num_predict"] <= 300
    assert TASK in c["messages"][1]["content"] and "PROJECT" in c["messages"][1]["content"]       # the task and the repo profile


@pytest.mark.parametrize("reply", [
    '{"questions": []}', "not json at all", "", '{"questions": "x"}', '{"nope": 1}',
    json.dumps({"questions": ["a?", "b?", "c?", "d?"]}),                       # max 3
    json.dumps({"questions": ["ok?", ""]}), json.dumps({"questions": ["ok?", 3]}),
    json.dumps({"questions": ["x" * 201]}),
])
def test_clarify_is_empty_for_clear_or_invalid_replies(tmp_path, reply):
    eng, _repo, _ = engine(tmp_path, FakeLLM([], background=[reply]))
    assert IN.clarify(eng, TASK, cfg=eng.config) == []


def test_clarify_accepts_three_questions_and_survives_exceptions(tmp_path):
    three = json.dumps({"questions": ["a?", "b?", "c?"]})
    eng, _repo, _ = engine(tmp_path, FakeLLM([], background=[three]))
    assert IN.clarify(eng, TASK, cfg=eng.config) == ["a?", "b?", "c?"]

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("server down")

    assert IN.clarify(Boom(), TASK, cfg={"models": {"chat": "m"}}) == []


# ── run: asking, skipping, composing ────────────────────────────────────────

def test_run_asks_numbered_questions_and_composes_answers(tmp_path, human):
    llm = FakeLLM([], background=[QS])
    eng, _repo, out = engine(tmp_path, llm)
    read = reader("a terminal curses game", "s")
    text = IN.run(eng, LONG_TASK, read=read)
    assert text == (f"{LONG_TASK}\n\nClarifications from the user:\n1. Which language and UI (terminal or window)? -> a terminal curses game\n"
                    f"2. What should 'sound' cover? -> {IN.YOUR_CALL}")
    shown = out.getvalue()
    assert "1. Which language" in shown and "2. What should" in shown and len(read.asked) == 2
    assert [e for e in eng.events.recent if e["kind"] == "intake"][0]["answered"] == 1


def test_go_stops_asking_and_enter_skips(tmp_path, human):
    three = json.dumps({"questions": ["a?", "b?", "c?"]})
    eng, _repo, _ = engine(tmp_path, FakeLLM([], background=[three]))
    read = reader("", "go", "never read")
    text = IN.run(eng, LONG_TASK, read=read)
    assert len(read.asked) == 2 and text.count(IN.YOUR_CALL) == 3 and "-> never read" not in text


def test_eof_while_answering_stops_asking(tmp_path, human):
    eng, _repo, _ = engine(tmp_path, FakeLLM([], background=[QS]))

    def read(prompt):
        raise EOFError

    assert IN.run(eng, LONG_TASK, read=read).count(IN.YOUR_CALL) == 2


def test_no_questions_leaves_the_task_alone(tmp_path, human):
    eng, _repo, _ = engine(tmp_path, FakeLLM([], background=['{"questions": []}']))
    assert IN.run(eng, LONG_TASK, read=reader()) == LONG_TASK


def test_never_twice_for_the_same_task_text(tmp_path, human):
    llm = Recorder(background=[QS, QS])
    eng, _repo, _ = engine(tmp_path, llm)
    IN.run(eng, LONG_TASK, read=reader("x", "y"))
    assert IN.run(eng, LONG_TASK, read=reader()) == LONG_TASK and len(llm.chats) == 1
    spaced = "  " + LONG_TASK.replace(" ", "  ")
    assert IN.run(eng, spaced, read=reader()) == spaced and len(llm.chats) == 1                 # whitespace does not make it a new task


def test_not_asked_when_it_should_not(tmp_path, monkeypatch):
    llm = Recorder(background=[QS] * 8)
    eng, _repo, _ = engine(tmp_path, llm)
    read = reader()
    monkeypatch.setattr(IN, "tty", lambda: False)
    assert IN.run(eng, LONG_TASK, read=read) == LONG_TASK                                    # no terminal / autopilot headless
    monkeypatch.setattr(IN, "tty", lambda: True)
    assert IN.run(eng, "/status", read=read) == "/status"                                    # slash command
    assert IN.run(eng, "fix typo", read=read) == "fix typo"                                  # short task (below the 12 characters)
    assert IN.run(eng, "AUTOPILOT: continue", read=read) == "AUTOPILOT: continue"
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "intake": "off"}}
    assert IN.run(eng, LONG_TASK, read=read) == LONG_TASK                                    # off
    eng.config = {**eng.config, "coding": {"intake": "auto", "intake_min_chars": 500}}
    assert IN.run(eng, LONG_TASK, read=read) == LONG_TASK                                    # below the configured minimum
    eng.stage_plan = ST.StagePlan([ST.Stage(1, "t", "g")], 0, "model", "x")                  # a resumed, unfinished stage plan
    eng.config = {**eng.config, "coding": {"intake": "auto"}}
    assert IN.run(eng, LONG_TASK, read=read) == LONG_TASK
    assert llm.chats == []


# ── who gets the intake: short vague prompts too, specific ones on an existing code base do not ─────────

def test_default_minimum_is_twelve_chars_and_greenfield_prompt_on_an_empty_repo_asks(tmp_path, human):
    assert IN.DEFAULT_MIN_CHARS == 12 and IN.min_chars({}) == 12
    llm = Recorder(background=[json.dumps({"questions": ["Which language and interface (CLI, GUI, web)?"]})])
    eng, _repo, _ = engine(tmp_path, llm)
    out = IN.run(eng, "make a todo app", read=reader("python cli"))
    assert len(llm.chats) == 1 and out.endswith("1. Which language and interface (CLI, GUI, web)? -> python cli")
    assert IN.wanted("make a todo app", {}) and not IN.wanted("fix typo", {}) and not IN.wanted("/status now please", {})


def _repo_with_code(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "parser.py").write_text("def parse(x):\n    return x[0]\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'p'\n")
    return repo


@pytest.mark.parametrize("task", [
    "fix the IndexError in parser.py line 40", "there is a bug when the list is empty", "parse() raises ValueError on blank input",
    "update the docstring of `parse` please", "Traceback (most recent call last): boom",
])
def test_specific_task_on_a_repo_with_code_skips_the_intake(tmp_path, human, task):
    _repo_with_code(tmp_path)
    llm = Recorder(background=[QS])
    eng, _repo, _ = engine(tmp_path, llm)
    assert IN.looks_specific(task, eng) is True
    assert IN.run(eng, task, read=reader()) == task and llm.chats == []


@pytest.mark.parametrize("task", ["make a todo app", "Please build me a web server", "write me a script that renames photos", "create a new game"])
def test_greenfield_wording_always_runs_the_intake_even_with_code_and_a_file_name(tmp_path, human, task):
    _repo_with_code(tmp_path)
    eng, _repo, _ = engine(tmp_path, Recorder(background=[QS]))
    assert IN.looks_specific(task + " next to parser.py", eng) is False
    assert IN.looks_specific(task, eng) is False


def test_empty_repo_always_runs_the_intake_even_for_a_specific_looking_task(tmp_path, human):
    llm = Recorder(background=['{"questions": []}'])
    eng, _repo, _ = engine(tmp_path, llm)
    task = "fix the IndexError in parser.py line 40"
    assert IN.looks_specific(task, eng) is False
    assert IN.run(eng, task, read=reader()) == task and len(llm.chats) == 1                 # the model was asked; it said [] (nothing to ask)


def test_vague_task_on_a_repo_with_code_asks_the_model(tmp_path, human):
    _repo_with_code(tmp_path)
    llm = Recorder(background=['{"questions": []}'])
    eng, _repo, _ = engine(tmp_path, llm)
    assert IN.looks_specific("improve the project somehow", eng) is False
    IN.run(eng, "improve the project somehow", read=reader())
    assert len(llm.chats) == 1


def test_slash_command_and_no_terminal_never_call_the_model(tmp_path, monkeypatch):
    llm = Recorder(background=[QS] * 3)
    eng, _repo, _ = engine(tmp_path, llm)
    monkeypatch.setattr(IN, "tty", lambda: True)
    assert IN.run(eng, "/status verbose please", read=reader()) == "/status verbose please"
    monkeypatch.setattr(IN, "tty", lambda: False)
    assert IN.run(eng, "make a todo app", read=reader()) == "make a todo app"
    assert llm.chats == []


def test_clarify_prompt_asks_greenfield_tasks_about_the_decisions_that_change_the_result():
    for word in ("greenfield", "language", "CLI", "persist", "scope"):
        assert word.lower() in IN.SYSTEM.lower(), word


def test_always_mode_asks_even_for_short_tasks(tmp_path, human):
    eng, _repo, _ = engine(tmp_path, FakeLLM([], background=['{"questions": ["Which one?"]}']))
    eng.config = {**eng.config, "coding": {**eng.config["coding"], "intake": "always"}}
    assert IN.run(eng, "add a flag", read=reader("--verbose")).endswith("1. Which one? -> --verbose")


# ── the answers reach the model and survive in the brief card ───────────────

def test_answers_are_in_the_first_request_and_the_brief_card(tmp_path, human):
    llm = Recorder([say("Done: snake.py.")], background=[QS])
    eng, _repo, _ = engine(tmp_path, llm)
    text = IN.run(eng, LONG_TASK, read=reader("terminal curses", "beeps only"))
    eng.turn(text)
    first = llm.calls[0]["snapshot"]
    users = [m["content"] for m in first if m["role"] == "user"]
    assert "terminal curses" in users[-1] and "Clarifications from the user:" in users[-1]
    cards = [m["content"] for m in first if brief_mod.is_card(m)]
    assert len(cards) == 1 and "terminal curses" in cards[0] and "beeps only" in cards[0]
    assert eng.briefs and "terminal curses" in eng.briefs[-1]


def test_intake_state_is_saved_with_the_session(tmp_path, human):
    eng, _repo, _ = engine(tmp_path, FakeLLM([say("ok")], background=['{"questions": []}']))
    IN.run(eng, LONG_TASK, read=reader())
    eng.split_cache["abc"] = [["t", "g" * 30]]
    eng.ctx.user_requests.append(LONG_TASK)
    eng.session_id = "s1"
    snap = sessionstore.snapshot(eng)
    eng.intake_seen, eng.split_cache = set(), {}
    sessionstore.restore(eng, json.loads(json.dumps(snap)))
    assert eng.intake_seen == {IN.key_of(LONG_TASK)} and eng.split_cache == {"abc": [["t", "g" * 30]]}


# ── ask_user ────────────────────────────────────────────────────────────────

def test_ask_user_interactive_returns_the_answer_and_counts_as_neutral(tmp_path, human):
    llm = FakeLLM([call("ask_user", question="Use pygame or curses?"), say("Used curses as told.")])
    eng, repo, out = engine(tmp_path, llm)
    eng.ask_fn = lambda q: "curses please"
    res = eng.turn("pick a toolkit and note it")
    tool_msgs = [m["content"] for m in eng.ctx.messages if m["role"] == "tool"]
    assert tool_msgs == ["curses please"] and res.files == [] and not list(repo.iterdir())          # no file edit
    assert "Use pygame or curses?" in out.getvalue()
    assert lookact.classify("ask_user", {"question": "x"}) == "neutral" and "ask_user" in lookact.NEUTRAL_ACTS
    assert eng.lookact.look_streak == 0


def test_ask_user_is_in_the_schema_with_the_short_description(tmp_path):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    tool = eng.registry.get("ask_user")
    assert tool is not None and tool.description.startswith("Ask the human ONE short question when you are blocked")
    assert eng.decide("ask_user", {"question": "q"}).verdict == "allow"
    eng.pol = eng.pol.__class__(**{**eng.pol.__dict__, "mode": "ask"})
    assert eng.decide("ask_user", {"question": "q"}).verdict == "allow"                          # every mode


def test_ask_user_fourth_question_is_refused_and_a_new_prompt_resets(tmp_path, human):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ask_fn = lambda q: "yes"
    got = [eng.execute("ask_user", {"question": f"q{n}?"}) for n in range(4)]
    assert got == ["yes", "yes", "yes", "Question limit reached; decide yourself."]
    eng.llm.script = [say("fine")]
    eng.turn("another prompt")
    assert eng.execute("ask_user", {"question": "again?"}) == "yes"


def test_ask_user_headless_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setattr(IN, "tty", lambda: False)
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))

    def boom(q):
        raise AssertionError("must not read")

    eng.ask_fn = boom
    assert eng.execute("ask_user", {"question": "which colour?"}) == IN.NO_HUMAN
    assert "No human available now. Decide yourself, state the assumption in your final summary." == IN.NO_HUMAN


def test_ask_user_empty_answer_tells_the_model_to_decide(tmp_path, human):
    eng, _repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ask_fn = lambda q: ""
    assert "decide yourself" in eng.execute("ask_user", {"question": "q?"})


# ── supervisor: one hint question on the 2nd loop stop ──────────────────────

def test_supervisor_asks_once_on_the_second_loop_stop_and_passes_the_hint(tmp_path, human):
    asked = []
    sup, sess, runner, _c = sv_make(tmp_path, {1: ["loop", "loop", "loop", "ok"]}, stages=1, ask=lambda q: asked.append(q) or "try the other port")
    sess.dead_ends = type("D", (), {"items": [{"tool": "run", "cmd": "pytest -x tests/test_a.py"}]})()
    sup.run("t")
    assert len(asked) == 1 and asked[0].startswith("I keep failing at run: pytest -x") and asked[0].endswith("Hint? (Enter to let me continue)")
    notes = [c[1] for c in runner.calls]
    assert "try the other port" in notes[2] and not any("try the other port" in n for n in (notes[0], notes[1], notes[3]))


def test_supervisor_enter_means_no_hint_and_no_tty_means_no_question(tmp_path, human, monkeypatch):
    sup, _s, runner, _c = sv_make(tmp_path, {1: ["loop", "loop", "ok"]}, stages=1, ask=lambda q: "")
    sup.run("t")
    assert not any("hint" in c[1].lower() for c in runner.calls)
    monkeypatch.setattr(IN, "tty", lambda: False)
    asked = []
    sup2, _s2, _r2, _c2 = sv_make(tmp_path, {1: ["loop", "loop", "ok"]}, stages=1, ask=lambda q: asked.append(q) or "x")
    sup2.run("t")
    assert asked == []


# ── model split by default ──────────────────────────────────────────────────

def stages_json(*pairs):
    return json.dumps({"stages": [{"title": t, "goal": g} for t, g in pairs]})


G = [("Data model", "Create scores.py with the high score store: load, add, top10, saved to a json file."),
     ("Game loop", "Create game.py with the snake movement, food, collision and the main loop using curses."),
     ("Menu and sound", "Create menu.py with the start menu and add terminal beeps in game.py for eating and dying.")]


def test_plain_prompt_is_split_by_the_model_without_any_config():
    llm = Recorder([], background=[stages_json(*G)])
    plan = ST.split(LONG_TASK, llm=llm, cfg={"models": {"chat": "main", "background": "small"}}, model="main")
    assert plan.source == "model" and [s.title for s in plan.stages] == [g[0] for g in G]
    assert all(g[1] in s.goal for g, s in zip(G, plan.stages)) and all(s.goal.startswith("Overall task:") for s in plan.stages)
    c = llm.chats[0]
    assert c["model"] == "main" and c["background"] is True and c["opts"]["num_predict"] <= 600          # the MAIN model, capped output
    assert "ONE stage if the task is one coherent change" in c["messages"][0]["content"]


@pytest.mark.parametrize("reply", [
    stages_json(G[0]),                                                      # one item = a single stage
    "garbage", json.dumps({"stages": "x"}),
    stages_json(G[0], ("Short", "too short")),                              # goal < 20 chars
    stages_json(G[0], ("T" * 81, G[1][1])),                                 # title > 80
    stages_json(G[0], ("Big", "g" * 1201)),                                 # goal > 1200
    stages_json(G[0], (G[0][0], G[1][1])), stages_json(G[0], ("Other", G[0][1])),      # duplicates
    stages_json(*[(f"S{n}", f"build the part number {n} of the thing") for n in range(9)]),        # more than 8
])
def test_invalid_or_single_stage_answers_fall_back_to_one_stage(reply):
    plan = ST.split(LONG_TASK, llm=FakeLLM([], background=[reply]), cfg={"models": {"chat": "m"}})
    assert plan.source == "single" and len(plan.stages) == 1 and plan.stages[0].goal == LONG_TASK


def test_short_task_never_calls_the_model_and_the_switch_can_turn_it_off():
    llm = Recorder([], background=[stages_json(*G)] * 3)
    cfg = {"models": {"chat": "m"}}
    assert ST.split("make a snake game", llm=llm, cfg=cfg).source == "single"
    assert ST.split(LONG_TASK, llm=llm, cfg={**cfg, "coding": {"stage_split_with_model": False}}).source == "single"
    assert ST.split(LONG_TASK, llm=llm, cfg={**cfg, "coding": {"stage_split_min_chars": 5000}}).source == "single"
    assert ST.split(LONG_TASK, llm=None, cfg=cfg).source == "single" and llm.chats == []
    assert ST.split(LONG_TASK, llm=llm, cfg={**cfg, "coding": {"stage_split_min_chars": 20}}).source == "model" and len(llm.chats) == 1


def test_structure_still_wins_over_the_model():
    llm = Recorder([], background=[stages_json(*G)])
    from tests.test_codeagent_stages import TASK3

    assert ST.split(TASK3, llm=llm, cfg={"models": {"chat": "m"}}).source == "headings" and llm.chats == []


def test_cache_keeps_the_answer_per_task_text_including_single():
    llm = Recorder([], background=[stages_json(*G), stages_json(G[0])])
    cache, cfg = {}, {"models": {"chat": "m"}}
    first = ST.split(LONG_TASK, llm=llm, cfg=cfg, cache=cache)
    again = ST.split(LONG_TASK, llm=llm, cfg=cfg, cache=cache)
    assert len(llm.chats) == 1 and first.source == again.source == "model" and [s.goal for s in first.stages] == [s.goal for s in again.stages]
    other = LONG_TASK + " Also add a pause key."
    assert ST.split(other, llm=llm, cfg=cfg, cache=cache).source == "single" and len(llm.chats) == 2
    assert ST.split(other, llm=llm, cfg=cfg, cache=cache).source == "single" and len(llm.chats) == 2          # a 1-stage answer is cached too
    assert json.loads(json.dumps(cache))                                                                       # saved with the session: plain JSON


def test_a_failed_call_is_not_cached():
    class Down:
        def chat(self, *a, **k):
            raise ConnectionError("down")

    cache = {}
    assert ST.split(LONG_TASK, llm=Down(), cfg={"models": {"chat": "m"}}, cache=cache).source == "single" and cache == {}


def test_stage_titles_are_announced_one_dim_line_each_for_two_or_more(tmp_path):
    import io

    from rich.console import Console

    out = io.StringIO()
    con = Console(file=out, width=100, force_terminal=False)
    ST.announce(con, ST.split(LONG_TASK, llm=FakeLLM([], background=[stages_json(*G)]), cfg={"models": {"chat": "m"}}))
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 3 and "Data model" in lines[0] and "Menu and sound" in lines[2]
    out.truncate(0)
    out.seek(0)
    ST.announce(con, ST.split(LONG_TASK, llm=None, cfg={}))
    assert out.getvalue() == ""


def test_run_task_staged_uses_the_engine_model_and_cache_and_prints_titles(tmp_path):
    from agent.codeagent import session as S

    llm = Recorder([say("Part one done."), say("Part two done."), say("Part three done.")], background=[stages_json(*G)])
    eng, _repo, out = new_engine(tmp_path, llm)
    eng.model = "the-main-model"
    report = S.run_task_staged(eng, LONG_TASK)
    assert report is not None and len(report.stages) == 3 and llm.chats[0]["model"] == "the-main-model"
    assert "Data model" in out.getvalue() and len(eng.split_cache) == 1
