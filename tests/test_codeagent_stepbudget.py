"""Adaptive step budget: sized per goal, a user value is a hard cap, one bounded extension on demonstrable progress. Fake model only."""
from __future__ import annotations

from agent.codeagent import policy as P, stepbudget as SB
from tests.test_codeagent_engine import FakeLLM, _isolated, call, engine, say  # noqa: F401 - _isolated is an autouse fixture


def test_estimate_scales_with_the_task_and_clamps():
    tiny = SB.estimate("simple snake terminal game")
    big = SB.estimate("refactor the whole server and migrate the database, add an api and a gui " * 3, files=400)
    assert SB.LO <= SB.clamp(tiny, 60) < SB.clamp(big, 60) <= 60
    assert SB.clamp(500, 60) == 60 and SB.clamp(1, 60) == SB.LO and SB.clamp(1, 5) == 5
    assert SB.estimate("fix a typo", effort="min") < SB.estimate("fix a typo", effort="max")
    assert SB.estimate("fix x.py y.py z.py") > SB.estimate("fix it")


def test_grow_is_bounded_by_the_ceiling():
    assert SB.grow(12, 60) == 18 and SB.grow(58, 60) == 60 and SB.grow(4, 60) == 8


def test_auto_budget_is_set_per_goal_and_reported(tmp_path):
    eng, _repo, _out = engine(tmp_path, FakeLLM([say("done")]))
    assert eng.budget_source == "auto"
    eng.turn("simple snake terminal game")
    assert eng.budget_source == "auto" and SB.LO <= eng.budget_now < eng.step_ceiling == 60


def test_a_user_value_is_the_hard_cap_and_source(tmp_path):
    eng, _repo, _out = engine(tmp_path, FakeLLM([say("done")]), max_steps=7)
    eng.turn("simple snake terminal game")
    assert (eng.budget_source, eng.budget_now, eng.step_ceiling) == ("user", 7, 7)


def test_policy_marks_an_explicit_value(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    assert P.for_repo(repo).steps_set is False
    assert P.save_repo(repo, max_steps=30).steps_set is True


def test_auto_budget_stops_a_runaway_without_progress(tmp_path):
    eng, _repo, out = engine(tmp_path, FakeLLM([call("list_dir", path=f"d{i}") for i in range(70)]))
    res = eng.turn("fix a typo")
    assert res.steps < 30 and eng.budget_now < 30 and "automatic budget" in out.getvalue()


def test_auto_budget_extends_once_with_progress_and_never_past_the_ceiling(tmp_path):
    writes = [call("write_file", path=f"m{i}.py", content=f"X{i} = {i}\n") for i in range(60)]
    eng, _repo, out = engine(tmp_path, FakeLLM([*writes, say("done")]))
    start = []
    orig = eng._size_budget
    eng._size_budget = lambda: (orig(), start.append(eng.budget_now))
    res = eng.turn("fix a typo")
    assert eng.budget_now == SB.grow(start[0], 60) and res.steps == eng.budget_now and out.getvalue().count("files/tests moved") == 1
