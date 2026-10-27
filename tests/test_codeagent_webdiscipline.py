"""Web discipline: a web lookup is refused until the task asks for one or a command failed twice. FakeLLM, a counting fake web tool."""

from __future__ import annotations

import pytest

from agent import autonomy
from agent.codeagent import engine as E, loopguard
from agent.tools.registry import Tool

from tests.test_codeagent_engine import FakeLLM, engine


REFUSAL = ("Web lookup not needed yet: write the code first; use web only for an unfamiliar library/API or after the same error happened twice.")


def web_engine(tmp_path):
    eng, repo, _out = engine(tmp_path, FakeLLM([]))
    calls = []

    def handler(query="", url="", **_):
        calls.append(query or url)
        return f"results for {query or url}"

    for name in ("web_search", "web_fetch"):
        eng.registry.register(Tool(name=name, description="x", parameters={"type": "object", "properties": {"query": {"type": "string"}, "url": {"type": "string"}}},
                                   handler=handler, risk="safe"))
    return eng, repo, calls


def start(eng, text):
    eng.start()
    eng._task_text, eng._prompt_fails = text, 0


def test_web_search_is_refused_with_one_deterministic_line(tmp_path):
    eng, _repo, calls = web_engine(tmp_path)
    start(eng, "make a todo app")
    assert eng.execute("web_search", {"query": "build a todo app"}) == REFUSAL
    assert eng.execute("web_fetch", {"url": "https://example.com"}) == REFUSAL
    assert calls == []


@pytest.mark.parametrize("word", ["http", "www", "search", "online", "docs", "documentation", "latest", "current version", "look up", "google"])
def test_task_words_allow_the_lookup_case_insensitively(tmp_path, word):
    eng, _repo, calls = web_engine(tmp_path)
    start(eng, f"make a todo app, {word.upper()} please")
    assert eng.execute("web_search", {"query": "q"}) == "results for q"
    assert calls == ["q"]


def test_two_failed_runs_since_the_prompt_allow_the_lookup(tmp_path):
    eng, _repo, calls = web_engine(tmp_path)
    start(eng, "make a todo app")
    eng._run_call({"name": "run", "arguments": {"command": "python3 -c 'import sys; sys.exit(3)'"}})
    assert eng.execute("web_search", {"query": "q"}) == REFUSAL                    # one failure is not enough
    eng._run_call({"name": "run", "arguments": {"command": "python3 -c 'import nope_mod_a'"}})
    assert eng.execute("web_search", {"query": "q"}) == "results for q" and calls == ["q"]
    eng._task_text, eng._prompt_fails = "next task", 0                              # a new user prompt starts the count again
    assert eng.execute("web_search", {"query": "q2"}) == REFUSAL


def test_a_new_prompt_resets_the_failure_count_but_autopilot_continuations_do_not(tmp_path):
    eng, _repo, _calls = web_engine(tmp_path)
    eng._prompt_fails, eng._task_text = 2, "make a todo app, look up flask"
    eng.llm.script[:] = [{"role": "assistant", "content": "ok", "_prompt_eval_count": 1, "_eval_count": 1}] * 6
    eng.turn("AUTOPILOT: continue")
    assert eng._prompt_fails == 2 and eng._task_text == "make a todo app, look up flask"
    eng.turn("write a calculator")
    assert eng._prompt_fails == 0 and eng._task_text == "write a calculator"


def test_the_stuck_lookup_path_bypasses_the_check(tmp_path):
    eng, _repo, calls = web_engine(tmp_path)
    start(eng, "make a todo app")
    assert autonomy.enabled(eng.config)
    eng._react_to_stuck(autonomy.Verdict(kind="lookup", query="ImportError flask", count=2))
    assert calls == ["ImportError flask"]


def test_the_refusal_is_not_an_act_and_a_repeat_feeds_the_loop_guard(tmp_path):
    eng, _repo, calls = web_engine(tmp_path)
    start(eng, "make a todo app")
    before = (eng._edit_gap, eng._look_streak)
    eng._run_call({"name": "web_search", "arguments": {"query": "build a todo app"}})
    eng._run_call({"name": "web_search", "arguments": {"query": "build a todo app"}})
    assert (eng._edit_gap, eng._look_streak) == before and eng._progress == 0 and calls == []
    assert loopguard.SAME_RESULT_NOTE in eng.ctx.messages[-1]["content"]            # the repeat is a repeated call for the guard
    assert eng._prompt_fails == 0                                                    # and a refusal is not a failed run
