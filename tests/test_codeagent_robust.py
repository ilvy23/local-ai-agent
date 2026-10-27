"""Robustness of the coding engine against a flaky model server, suspend/resume and small windows (no network, no real sleeping)."""
from __future__ import annotations

import json
import time
import types

import httpx
import pytest

from agent import llm
from agent.codeagent import context as C, serverwait, tools as T
from tests.test_codeagent_engine import FakeLLM, call, engine, say


# ── A5b: clocks that survive suspend ────────────────────────────────────────

def test_a_wall_clock_jump_does_not_time_out_a_running_command(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    real, calls = time.time, []

    def jumpy():                                                                # the first reading is normal, every later one is an hour ahead
        calls.append(1)
        return real() + (3600 if len(calls) > 1 else 0)

    monkeypatch.setattr(T, "time", types.SimpleNamespace(time=jumpy, monotonic=time.monotonic, sleep=time.sleep))
    code, out, secs = eng.tools._exec("sleep 0.5; echo done", 30)               # the PC suspends: the wall clock leaps an hour
    assert code == 0 and "done" in out and "timed out" not in out and secs < 10


def test_a_monotonic_clock_past_the_limit_times_out(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    ticks = iter(range(0, 10_000, 100))
    monkeypatch.setattr(T, "time", types.SimpleNamespace(time=time.time, monotonic=lambda: next(ticks), sleep=time.sleep))
    code, out, _s = eng.tools._exec("sleep 2", 600)
    assert code is None and "timed out after 600s" in out


# ── A5c: a long prefill must not hit the read timeout ───────────────────────

def test_chat_calls_wait_long_for_the_first_byte_but_connect_fast():
    client = llm.OllamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    t = client._client.timeout
    assert t.read == 1800 and t.connect == 10
    client.set_read_timeout(2400)                                           # coding.read_timeout_s
    assert client._client.timeout.read == 2400 and client._client.timeout.connect == 10
    assert client._vulkan().timeout.read == 2400 and client._vulkan().timeout.connect == 10


def test_read_timeout_follows_the_coding_config(tmp_path, monkeypatch):
    from agent.codeagent import session

    seen = []
    fake = types.SimpleNamespace(set_read_timeout=seen.append)
    session.apply_client_config(fake, {"coding": {"read_timeout_s": 900}})
    session.apply_client_config(fake, {})
    assert seen == [900, 1800]


# ── A5a: wait for the server instead of giving up in milliseconds ───────────

from tests.test_codeagent_hardening import FailingLLM, _http_error      # noqa: E402


class Clock:
    """A fake clock: sleeping advances it, nothing really sleeps."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class ServerLLM(FailingLLM):
    def server_url(self, model):
        return "http://fake:11434"


@pytest.fixture
def no_real_sleep(monkeypatch):
    def boom(_s):
        raise AssertionError("real time.sleep called")

    monkeypatch.setattr(time, "sleep", boom)


def waiting_engine(tmp_path, script, errors, probes, wait_s=None):
    clock = Clock()
    eng, repo, out = engine(tmp_path, ServerLLM(script, errors))
    eng._sleep, eng._clock = clock.sleep, clock
    queue = iter(probes)
    eng._probe = lambda url: next(queue, False) if probes is not None else False
    if wait_s is not None:
        eng.config = {**eng.config, "coding": {**eng.config["coding"], "server_wait_s": wait_s}}
    return eng, clock, out


def test_wait_for_server_backs_off_1_2_4_8_16_30_30(no_real_sleep):
    clock, seen = Clock(), []
    ok = serverwait.wait_for_server("http://x", max_wait_s=600, sleep=clock.sleep, clock=clock, probe=lambda u: seen.append(u) or len(seen) == 9)
    assert ok is True and clock.sleeps == [1, 2, 4, 8, 16, 30, 30, 30] and seen[0] == "http://x"


def test_wait_for_server_gives_up_after_max_wait_in_fake_time(no_real_sleep):
    clock = Clock()
    assert serverwait.wait_for_server("http://x", max_wait_s=100, sleep=clock.sleep, clock=clock, probe=lambda u: False) is False
    assert sum(clock.sleeps) == 100 and clock.sleeps[-1] <= 30
    assert serverwait.wait_for_server("http://x", max_wait_s=0, sleep=clock.sleep, clock=clock, probe=lambda u: True) is False


def test_the_default_probe_asks_api_version():
    asked = []

    def handler(request):
        asked.append(str(request.url))
        return httpx.Response(200)

    real = httpx.get
    try:
        httpx.get = lambda url, **kw: httpx.Client(transport=httpx.MockTransport(handler)).get(url)
        assert serverwait._http_probe("http://h:11436/") is True
    finally:
        httpx.get = real
    assert asked == ["http://h:11436/api/version"]


def test_a_dead_server_is_waited_for_then_the_same_request_is_retried(tmp_path, no_real_sleep):
    eng, clock, out = waiting_engine(tmp_path, [say("Done.")], [httpx.ConnectError("refused"), httpx.ConnectError("refused")],
                                     [False, False, False, True, True])
    res = eng.turn("say done")
    assert res.text == "Done." and res.stopped == ""
    assert clock.sleeps == [1, 2, 4]                                         # growing delays, all on the fake clock
    waits = [r for r in eng.events.recent if r["kind"] == "server_wait"]
    assert [w["seconds"] for w in waits] == [7, 0] and all(w["ok"] for w in waits)
    assert "server not reachable - waiting up to 600 s" in out.getvalue()
    assert len(eng.llm.calls) == 1 and eng.llm.calls[0]["snapshot"][-1]["content"] == "say done"   # the same request went through


def test_a_server_that_never_returns_stops_after_the_wait_in_fake_time(tmp_path, no_real_sleep):
    eng, clock, out = waiting_engine(tmp_path, [say("never")], [httpx.ConnectError("refused")] * 3, [], wait_s=120)
    res = eng.turn("say done")
    assert res.stopped == "error" and sum(clock.sleeps) == 120
    assert "model call failed" in out.getvalue() and "all edits are kept" in out.getvalue()
    assert [r["kind"] for r in eng.events.recent].count("server_wait") == 1


def test_three_failed_attempts_stop_even_when_the_server_answers_probes(tmp_path, no_real_sleep):
    eng, clock, out = waiting_engine(tmp_path, [say("never")], [httpx.ReadError("reset")] * 3, [True] * 10)
    res = eng.turn("say done")
    assert res.stopped == "error" and "3 attempts" in out.getvalue()
    assert len([r for r in eng.events.recent if r["kind"] == "server_wait"]) == 2          # a wait before attempts 2 and 3


@pytest.mark.parametrize("exc", [KeyError("message"), json.JSONDecodeError("Expecting value", "{\"mes", 0), ValueError("half a body"),
                                 httpx.RemoteProtocolError("peer closed"), httpx.ReadTimeout("slow"),
                                 _http_error(500, '{"error":"boom"}'), _http_error(503, "busy"),
                                 _http_error(200, '{"error":"unexpected EOF"}')])
def test_server_side_failures_are_waited_out_and_retried(tmp_path, no_real_sleep, exc):
    eng, clock, out = waiting_engine(tmp_path, [say("Done.")], [exc], [True])
    res = eng.turn("say done")
    assert res.text == "Done." and res.stopped == "" and "internal error" not in out.getvalue()
    assert [r["kind"] for r in eng.events.recent].count("server_wait") == 1


def test_a_client_error_stops_at_once_without_waiting(tmp_path, no_real_sleep):
    eng, clock, out = waiting_engine(tmp_path, [say("never")], [_http_error(400, '{"error":"bad request"}')], [True])
    res = eng.turn("say done")
    assert res.stopped == "error" and clock.sleeps == [] and len(eng.llm.errors) == 0
    assert not [r for r in eng.events.recent if r["kind"] == "server_wait"] and eng.llm.script        # the model was asked exactly once


def test_waiting_can_be_switched_off(tmp_path, no_real_sleep):
    eng, clock, out = waiting_engine(tmp_path, [say("Done.")], [httpx.ReadError("reset")], [True], wait_s=0)
    assert eng.turn("say done").text == "Done." and "retrying once" in out.getvalue() and clock.sleeps == []


# ── A5d: an out-of-memory retry compacts first ──────────────────────────────

OOM = '{"error":"cudaMalloc failed: out of memory"}'


def _tokens(messages):
    return sum(len(str(m.get("content", ""))) for m in messages) / C.CHARS_PER_TOKEN


def _bulk(n):
    """n assistant/tool pairs of ~2.5k chars of realistic multi-line tool output."""
    out = []
    for i in range(n):
        out.append({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": f"src/mod{i}.py"}})})
        out.append({"role": "tool", "content": "\n".join(f"line {j} of module {i}: value_{i}_{j} = compute({i}, {j})" for j in range(60))})
    return out


class GrowingLLM(FailingLLM):
    """The first reply also grows the conversation by ~72k chars (a long run's progress); errors apply from the second call on."""

    def __init__(self, ctx_holder, script, errors):
        super().__init__(script, errors)
        self.holder = ctx_holder

    def chat_with_tools(self, messages, *a, **k):
        if not self.calls:
            messages.extend(_bulk(24))
            self.calls.append({"snapshot": []})
            return {**call("list_dir", path="."), "_prompt_eval_count": 21000}
        return super().chat_with_tools(messages, *a, **k)


def test_oom_retry_compacts_to_the_smaller_window_before_resending(tmp_path):
    llm_ = GrowingLLM(None, [say("Done.")], [_http_error(500, OOM)])
    eng, repo, out = engine(tmp_path, llm_)
    llm_.background = ["DONE: read the modules\nDECISIONS: none\nPITFALLS: none"]
    eng.ctx.summarize_in_place = None
    res = eng.turn("look at the code")
    assert res.text == "Done." and eng.num_ctx == 24576 and "ran out of memory" in out.getvalue()
    resent = llm_.calls[-1]["snapshot"]
    assert _tokens(resent) <= 0.8 * 24576 and resent[0]["role"] == "system" and "senior software engineer" in resent[0]["content"]
    assert any(m["role"] == "user" and m["content"] == "look at the code" for m in resent)       # the task survives


def test_oom_retry_that_cannot_fit_stops_instead_of_sending_a_truncated_prompt(tmp_path):
    llm_ = FailingLLM([say("never")], [_http_error(500, OOM)])
    eng, repo, out = engine(tmp_path, llm_)
    res = eng.turn("x" * 120_000)                                              # one request alone is bigger than the shrunk window
    assert res.stopped == "error" and "does not fit" in out.getvalue() and eng.num_ctx == 24576
    assert llm_.script and not llm_.calls                                      # nothing was sent after the shrink


# ── A5e: the compaction fallback must not evict the main model ──────────────

class ModelSpy:
    def __init__(self):
        self.models = []

    def chat(self, messages, model=None, stream=False, background=False, **kw):
        self.models.append(model)
        yield "DONE: wrote main.py\nSTATE: ok\nDECISIONS: none\nOPEN: tests\nPITFALLS: none"


def _compactable(tmp_path, config, model_fn=None):
    from agent.codeagent.eventlog import EventLog

    spy = ModelSpy()
    m = C.ContextManager(spy, config, 10000, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    m.model_fn = model_fn
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    for msg in _bulk(8):
        m.add(msg)
    return m, spy


def test_the_compaction_fallback_uses_the_main_model_not_the_background_one(tmp_path):
    m, spy = _compactable(tmp_path, {"models": {"background": "qwen2.5:7b"}}, model_fn=lambda: "qwen3.5:27b")
    assert m.compact() and spy.models == ["qwen3.5:27b"]


def test_a_configured_compact_model_wins(tmp_path):
    cfg = {"models": {"background": "qwen2.5:7b"}, "coding": {"compact_model": "tiny:1b"}}
    m, spy = _compactable(tmp_path, cfg, model_fn=lambda: "qwen3.5:27b")
    assert m.compact() and spy.models == ["tiny:1b"]


def test_without_a_known_main_model_the_background_model_is_the_last_resort(tmp_path):
    m, spy = _compactable(tmp_path, {"models": {"background": "qwen2.5:7b"}})
    assert m.compact() and spy.models == ["qwen2.5:7b"]


def test_the_engine_tells_the_context_its_current_model(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.model = "other:9b"                                                   # /model changes it at run time
    assert eng.ctx.model_fn() == "other:9b"


class UnloadSpy:
    def __init__(self):
        self.unloaded = 0

    def unload_all(self):
        self.unloaded += 1
        return ["m"]


def test_the_model_is_unloaded_on_exit_unless_keep_model_loaded(tmp_path):
    import io
    from rich.console import Console

    from agent.codeagent import session

    con = Console(file=io.StringIO())
    for cfg, expected in (({}, 1), ({"coding": {"keep_model_loaded": False}}, 1), ({"coding": {"keep_model_loaded": True}}, 0)):
        spy = UnloadSpy()
        session.release_gpu_on_exit(con, spy, cfg)
        assert spy.unloaded == expected, cfg
    spy = UnloadSpy()                                                        # the general switch still applies
    session.release_gpu_on_exit(con, spy, {"ollama": {"unload_on_exit": False}})
    assert spy.unloaded == 0


def test_keep_model_flag_sets_the_config_key(monkeypatch):
    from typer.testing import CliRunner

    from agent import main
    from agent.codeagent import session

    seen = []
    monkeypatch.setattr(session, "run_session", lambda *a, **kw: seen.append((a, kw)) or 0)
    CliRunner().invoke(main.app, ["code", "--old-ui", "--keep-model", "fix it"])
    CliRunner().invoke(main.app, ["code", "--old-ui", "fix it"])
    assert seen[0][1] == {"keep_model": True} and seen[1][1] == {}
    cfg = session.with_keep_model({"coding": {"x": 1}}, True)
    assert cfg["coding"] == {"x": 1, "keep_model_loaded": True} and session.with_keep_model({}, False) == {}
