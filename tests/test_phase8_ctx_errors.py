"""Phase 8: context overrides and the ceiling apply to every request, chat and code share one window, the output cap fits the
window, gpt-oss gets the configured reasoning effort, empty replies are retried a bounded number of times, and errors name what
really happened (the server's own message; the Vulkan server by its port)."""

from __future__ import annotations

import httpx
import pytest

from agent import friendly, llm


def test_a_per_model_override_applies_without_a_config_argument(monkeypatch):
    llm.configure({"models": {"overrides": {"big:1": {"num_ctx": 131072}}, "max_ctx_ceiling": 95000}})
    try:
        assert llm.cap_ctx("big:1", 131072) == 131072
        assert llm.cap_ctx("other:1", 131072) == 95000
    finally:
        llm.configure({})


def test_the_chat_window_never_exceeds_the_models_own_maximum(monkeypatch):
    from agent.chat.helpers import _num_ctx

    llm.note_native_ctx("small:7b", 32768)
    try:
        assert _num_ctx({"models": {"chat_num_ctx": 95000}}, "small:7b") == 32768
    finally:
        llm._NATIVE_CTX.pop("small:7b", None)


def test_code_and_chat_use_the_same_window(tmp_path):
    from agent.chat.helpers import _num_ctx
    from tests.test_codeagent_engine import FakeLLM, engine

    eng, _repo, _out = engine(tmp_path, FakeLLM([]))
    assert eng.num_ctx == _num_ctx(eng.config, eng.model)


def test_the_output_cap_fits_the_room_the_prompt_leaves():
    msgs = [{"role": "user", "content": "x" * 3200 * 30}]          # ~30k tokens of prompt in a 32k window
    assert llm.fit_output(8192, 32768, msgs) <= 32768 - 30000
    assert llm.fit_output(8192, 131072, msgs) == 8192             # plenty of room: unchanged


def test_gpt_oss_uses_the_configured_effort_for_chat_turns(monkeypatch):
    seen = {}

    class Http:
        def stream(self, method, path, json=None, **kw):
            seen["payload"] = json

            class R:
                status_code = 200
                extensions = {}

                def __enter__(self): return self
                def __exit__(self, *a): return False
                def raise_for_status(self): pass
                def read(self): pass
                def iter_lines(self): return iter(['{"message": {"content": "hi"}, "done": true}'])
            return R()

    c = llm.OllamaClient.__new__(llm.OllamaClient)
    c._vk_client, c._tools_unsupported = None, set()
    monkeypatch.setattr(llm.OllamaClient, "_client_for", lambda self, model: Http())
    llm.configure({"models": {"reasoning_effort": "high"}})
    try:
        c.chat_with_tools_stream([{"role": "user", "content": "x"}], "gpt-oss:20b", [], num_ctx=8192)
        assert seen["payload"]["think"] == "high"
        llm.configure({"models": {"reasoning_effort": "medium"}})
        c.chat_with_tools_stream([{"role": "user", "content": "x"}], "gpt-oss:20b", [], num_ctx=8192)
        assert "think" not in seen["payload"]                    # medium is the model's own default: nothing sent
    finally:
        llm.configure({})


def test_a_mid_stream_error_is_shown_as_the_servers_own_text():
    exc = llm.OllamaStreamError("cudaMalloc failed: out of memory", "gpt-oss:20b")
    line, hint = friendly.explain(exc)
    assert "(200)" not in line and "graphics memory" in line and "smaller" in hint


def test_a_vulkan_crash_names_that_server_not_the_default_one():
    line, hint = friendly.explain(llm.VulkanServerDown("http://127.0.0.1:11436"))
    assert "11436" in line and "ollama-vk" in hint and "nightage ollama start" not in hint


def test_a_failed_vulkan_check_is_remembered_for_a_while(monkeypatch):
    c = llm.OllamaClient.__new__(llm.OllamaClient)
    c._vulkan_ok, c._vk_client, c._transport, c._timeout = None, None, None, 5
    probes = []
    monkeypatch.setattr(llm.OllamaClient, "_reachable", staticmethod(lambda client, wait_s=0.0: probes.append(1) or False))
    monkeypatch.setattr(llm, "_vulkan_start_command", lambda: None)
    assert c._ensure_vulkan_server() is False and c._ensure_vulkan_server() is False
    assert len(probes) == 1 and llm.vulkan_down()
    llm._VULKAN_STATE["down_since"] = None


def test_empty_replies_are_retried_at_most_twice():
    from agent.chat import turn

    assert turn.MAX_EMPTY_RETRIES == 2 and "empty reply" in turn._EMPTY_REPLY_STOP
