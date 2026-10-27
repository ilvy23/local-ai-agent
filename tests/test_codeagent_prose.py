"""write_prose: the two-model pipeline for mature fiction. Off by default; a mocked llm.chat, never a live server."""

from __future__ import annotations

import threading
import time

import pytest
from typer.testing import CliRunner

from agent.codeagent import engine as E, lookact, prose
from agent.codeagent.context import MARKERS

from tests.test_codeagent_engine import CFG, FakeLLM, engine

PASSAGE = "The wicker door swings shut behind you and the hum of the villagers rises into a single wet scream as the fire takes your hair."


class ProseLLM(FakeLLM):
    """The prose model's seat: records the chat call, answers with a scripted text or raises."""

    def __init__(self, answers=()):
        super().__init__([])
        self.answers, self.prose_calls = list(answers), []

    def chat(self, messages, model=None, stream=False, background=False, **kw):
        self.prose_calls.append({"messages": messages, "model": model, "stream": stream, "background": background, **kw})
        a = self.answers.pop(0) if self.answers else PASSAGE
        if isinstance(a, BaseException):
            raise a
        if callable(a):
            a = a()
        yield a


def cfg(**coding):
    return {**CFG, "coding": {**CFG["coding"], "prose_model": "gemma2:9b", **coding}}


def prose_engine(tmp_path, llm=None, **coding):
    eng, repo, out = engine(tmp_path, llm or ProseLLM())
    eng.config = cfg(**coding)
    eng.registry.register(eng._make_prose_tool())
    eng.schema = eng.registry.to_ollama_schema()
    return eng, repo


def names(eng):
    return {t["function"]["name"] for t in eng.schema}


def new_engine(tmp_path, config):
    import io
    from rich.console import Console
    from agent.codeagent import policy as P
    from agent.codeagent.eventlog import EventLog

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    P.save_repo(repo, mode="auto")
    return E.CodeEngine(repo, config, ProseLLM(), Console(file=io.StringIO()), interactive=False, events=EventLog("t", tmp_path / "ev.jsonl"))


# ── registration ────────────────────────────────────────────────────────────

def test_disabled_by_default_no_tool_no_prompt_sentence(tmp_path):
    eng = new_engine(tmp_path, CFG)
    eng.start()
    assert "write_prose" not in names(eng)
    assert "write_prose" not in eng.ctx.messages[0]["content"]
    assert eng.ctx.messages[0]["content"].startswith(E.SYSTEM + "\n\n")             # the prompt is byte-identical to before
    assert prose.enabled(cfg(prose_model="  ")) is False


def test_enabled_lists_the_tool_and_adds_the_sentence(tmp_path):
    eng = new_engine(tmp_path, cfg())
    eng.start()
    assert "write_prose" in names(eng)
    assert prose.PROMPT_SENTENCE in eng.ctx.messages[0]["content"]
    tool = next(t for t in eng.schema if t["function"]["name"] == "write_prose")["function"]
    assert tool["parameters"]["required"] == ["request"] and {"request", "style_hint", "max_words"} <= set(tool["parameters"]["properties"])


def test_policy_allows_it_in_every_mode_and_lookact_calls_it_neutral(tmp_path):
    eng, _ = prose_engine(tmp_path)
    assert eng.decide("write_prose", {"request": "x"}).verdict == "allow"
    assert lookact.classify("write_prose", {"request": "x"}) == "neutral" and "write_prose" in lookact.NEUTRAL_ACTS
    eng._task_text = "make a game"                                               # no web-discipline words: still runs
    assert "write_prose" not in E.WEB_TOOLS and eng.execute("write_prose", {"request": "a death"}).startswith(prose.START)


# ── the request ─────────────────────────────────────────────────────────────

def test_request_reaches_the_model_with_system_prompt_style_cpu_and_background(tmp_path):
    llm = ProseLLM()
    eng, repo = prose_engine(tmp_path, llm)
    (repo / "STYLE.md").write_text("Short sentences. Salt and rot.")
    out = eng.execute("write_prose", {"request": "the miller's death", "style_hint": "second person, present tense", "max_words": 80})
    c = llm.prose_calls[0]
    assert c["model"] == "gemma2:9b" and c["background"] is True and c["stream"] is False and c["num_gpu"] == 0
    assert c["temperature"] == 0.8 and c["extra_options"]["num_predict"] == 240
    sysmsg, user = c["messages"][0], c["messages"][1]["content"]
    assert sysmsg["role"] == "system" and sysmsg["content"] == prose.SYSTEM and "no refusals" in sysmsg["content"]
    assert "Salt and rot." in user and "the miller's death" in user and "second person, present tense" in user and "80 words" in user
    assert out.startswith(prose.START + "\n" + PASSAGE + "\n" + prose.END + "\n")
    assert out.endswith(f"prose from gemma2:9b, {len(PASSAGE.split())} words - place it in the file yourself with edit/replace tools")


def test_options_from_config_and_caps(tmp_path):
    llm = ProseLLM()
    eng, _ = prose_engine(tmp_path, llm, prose_temperature=0.5, prose_gpu=True)
    eng.execute("write_prose", {"request": "x", "max_words": 5000})
    c = llm.prose_calls[0]
    assert c["temperature"] == 0.5 and c["num_gpu"] == -1 and c["extra_options"]["num_predict"] == 600


def test_style_file_lookup_order_and_cap(tmp_path):
    root = tmp_path / "r"
    (root / ".agent").mkdir(parents=True)
    (root / "docs").mkdir()
    assert prose.style_samples(root) == ""
    (root / "docs/STYLE.md").write_text("docs")
    assert prose.style_samples(root) == "docs"
    (root / ".agent/style.md").write_text("agent")
    assert prose.style_samples(root) == "agent"
    (root / "STYLE.md").write_text("x" * 5000)
    assert prose.style_samples(root) == "x" * 3000
    assert "Style samples" not in prose.build_messages("r", "", 100, "")[1]["content"]


# ── cleaning ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "<think>I should be careful</think>" + PASSAGE,
    "<think>\nmulti\nline\n</think>\n\n" + PASSAGE,
    "Here is the passage:\n\n" + PASSAGE,
    "Sure! Here's a death scene for your game:\n" + PASSAGE,
    '"' + PASSAGE + '"',
    "```\n" + PASSAGE + "\n```",
    "“" + PASSAGE + "”",
])
def test_think_boilerplate_quotes_and_fences_are_stripped(raw):
    assert prose.clean(raw) == PASSAGE


def test_markers_and_delimiters_cannot_survive_in_the_passage(tmp_path):
    dirty = f"{PASSAGE} [12 lines, 300 chars - now on disk] ...[cut] [old read of x.py] [no result recorded] {prose.END} more"
    got = prose.clean(dirty)
    assert not any(rx.search(got) for _n, rx in MARKERS) and prose.END not in got and PASSAGE in got
    eng, _ = prose_engine(tmp_path, ProseLLM([dirty]))
    body = eng.execute("write_prose", {"request": "x"}).split(prose.START + "\n")[1].split("\n" + prose.END)[0]
    assert not any(rx.search(body) for _n, rx in MARKERS)


# ── failures ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "I can't write that.", "I cannot help with that request, it describes violence.", "I'm sorry, but I won't produce this content.",
    "As an AI language model, I must decline to depict this.", "Sorry.", "", "<think>no</think>",
    "Okay. I'm unable to continue with graphic content like this scene.",
])
def test_refusals_give_the_fallback_and_log_prose_refused(tmp_path, text):
    eng, _ = prose_engine(tmp_path, ProseLLM([text]))
    assert eng.execute("write_prose", {"request": "a death"}) == prose.REFUSED
    assert [r["kind"] for r in eng.events.recent if r["kind"].startswith("prose")] == ["prose_refused"]


def test_a_normal_passage_with_can_in_it_is_not_a_refusal(tmp_path):
    text = "You can hear the bell. It cannot stop now, and the crows come down like a wet black cloth over the whole green hill."
    eng, _ = prose_engine(tmp_path, ProseLLM([text]))
    assert eng.execute("write_prose", {"request": "x"}).startswith(prose.START)


def test_exception_and_timeout_are_error_strings_not_raises(tmp_path):
    import httpx

    eng, _ = prose_engine(tmp_path, ProseLLM([httpx.ConnectError("connection refused"), RuntimeError("model 'gemma2:9b' not found")]))
    a = eng.execute("write_prose", {"request": "x"})
    b = eng.execute("write_prose", {"request": "x"})
    assert a.startswith("error: prose model gemma2:9b failed: ConnectError") and "not found" in b
    assert [r["kind"] for r in eng.events.recent if r["kind"].startswith("prose")] == ["prose_error", "prose_error"]
    slow = ProseLLM([lambda: threading.Event().wait(1.5) or PASSAGE])
    eng2, _ = prose_engine(tmp_path, slow, prose_timeout_s=0.2)
    t0 = time.monotonic()
    got = eng2.execute("write_prose", {"request": "x"})
    assert got.startswith("error: prose model gemma2:9b failed: timed out") and time.monotonic() - t0 < 1.2
    assert [r["kind"] for r in eng2.events.recent if r["kind"].startswith("prose")] == ["prose_error"]


def test_missing_request_is_an_error_string(tmp_path):
    eng, _ = prose_engine(tmp_path)
    assert eng.registry.get("write_prose").handler(request="  ").startswith("error:")


def test_max_calls_per_prompt_resets_with_the_next_prompt(tmp_path):
    llm = ProseLLM()
    eng, _ = prose_engine(tmp_path, llm, prose_max_calls=2)
    assert eng.execute("write_prose", {"request": "a"}).startswith(prose.START)
    assert eng.execute("write_prose", {"request": "b"}).startswith(prose.START)
    assert eng.execute("write_prose", {"request": "c"}) == prose.LIMIT.format(n=2) and len(llm.prose_calls) == 2
    eng.start()
    eng.llm.script = [{"role": "assistant", "content": "done", "_prompt_eval_count": 1, "_eval_count": 1}]
    eng.turn("next prompt")
    assert eng.prose_calls == 0
    assert prose.DEFAULT_MAX_CALLS == 12


# ── CLI ─────────────────────────────────────────────────────────────────────

def test_cli_prose_prints_the_passage_with_a_mocked_llm(tmp_path, monkeypatch):
    from agent import config as config_mod, main

    llm = ProseLLM()
    monkeypatch.setattr(main, "_open_llm", lambda: llm)
    monkeypatch.setattr(config_mod, "load_config", lambda: {"coding": {}})
    monkeypatch.setattr(main, "_ask_gpu_mode", lambda: (_ for _ in ()).throw(AssertionError("no GPU question for prose")))
    monkeypatch.setattr("sys.argv", ["agent", "code", "prose"])
    res = CliRunner().invoke(main.app, ["code", "prose", "a", "drowning", "--model", "gemma2:9b", "--folder", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert PASSAGE in res.output and "prose from gemma2:9b" in res.output
    c = llm.prose_calls[0]
    assert c["model"] == "gemma2:9b" and c["num_gpu"] == 0 and c["background"] is True and "a drowning" in c["messages"][1]["content"]
    assert "prose" in CliRunner().invoke(main.app, ["code", "--help"]).output


def test_cli_prose_without_a_model_does_not_call_anything(tmp_path, monkeypatch):
    from agent import config as config_mod, main

    monkeypatch.setattr(main, "_open_llm", lambda: (_ for _ in ()).throw(AssertionError("must not open the llm")))
    monkeypatch.setattr(config_mod, "load_config", lambda: {"coding": {}})
    monkeypatch.setattr("sys.argv", ["agent", "code", "prose"])
    res = CliRunner().invoke(main.app, ["code", "prose", "x"])
    assert res.exit_code == 1 and "no prose model" in res.output
