"""Phase 6: the chat / code model is protected from the start of a session - background helpers never load it on the CPU or at
another context size (both make Ollama reload it), even before the first chat message; other helper models only go to the CPU
when another model really holds the GPUs; helper calls report why they stopped and can be cancelled."""

from __future__ import annotations

import json
import threading

import pytest

from agent import llm


class _Resp:
    def __init__(self, lines):
        self.lines = lines
        self.status_code = 200
        self.extensions = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)

    def close(self):
        pass


def _client(monkeypatch, lines=('{"message": {"content": "ok"}, "done": true, "done_reason": "stop"}',)):
    seen = {}

    class FakeHttp:
        def stream(self, method, path, json=None, **kw):
            seen["payload"] = json
            return _Resp(list(lines))

    c = llm.OllamaClient.__new__(llm.OllamaClient)
    c._ps_cache = {}
    monkeypatch.setattr(llm.OllamaClient, "_client_for", lambda self, model: FakeHttp())
    monkeypatch.setenv("AGENT_VULKAN_MODELS", ".*")
    monkeypatch.delenv("AGENT_BG_CPU", raising=False)
    return c, seen


def test_protect_roles_covers_chat_and_code_with_the_chat_window(monkeypatch):
    monkeypatch.setattr("agent.chat.helpers._num_ctx", lambda config, model: 131072)
    llm.protect_roles({"models": {"chat": "gpt-oss:20b"}, "code": {"model": "qwen3-coder:30b"}})
    assert llm.is_protected("gpt-oss:20b") and llm.protected_ctx("gpt-oss:20b") == 131072 and llm.is_protected("qwen3-coder:30b")


def test_a_protected_model_is_never_forced_onto_the_cpu_even_before_the_first_chat(monkeypatch):
    c, seen = _client(monkeypatch)
    monkeypatch.setattr(llm.OllamaClient, "_vram_taken_by_other", lambda self, model: True)
    llm.mark_foreground("gpt-oss:20b", 65536)
    "".join(c.chat([{"role": "user", "content": "facts"}], "gpt-oss:20b", stream=False, background=True, num_ctx=4096))
    opts = seen["payload"]["options"]
    assert "num_gpu" not in opts and opts["num_ctx"] == 65536          # its own window, whatever the caller asked for


def test_another_helper_model_goes_to_the_cpu_only_when_the_gpus_are_taken(monkeypatch):
    c, seen = _client(monkeypatch)
    monkeypatch.setattr(llm.OllamaClient, "_vram_taken_by_other", lambda self, model: False)
    "".join(c.chat([{"role": "user", "content": "x"}], "llama3.2:3b", stream=False, background=True))
    assert "num_gpu" not in seen["payload"].get("options", {})
    monkeypatch.setattr(llm.OllamaClient, "_vram_taken_by_other", lambda self, model: True)
    "".join(c.chat([{"role": "user", "content": "x"}], "llama3.2:3b", stream=False, background=True))
    assert seen["payload"]["options"]["num_gpu"] == 0


def test_an_unanswered_probe_never_forces_the_cpu(monkeypatch):
    c = llm.OllamaClient.__new__(llm.OllamaClient)
    c._ps_cache = {}
    monkeypatch.setattr(llm.OllamaClient, "_ps", lambda self, model, max_age=5.0: None)
    assert c._vram_taken_by_other("llama3.2:3b") is False


def test_background_calls_do_not_overwrite_the_remembered_window(monkeypatch):
    c, seen = _client(monkeypatch)
    monkeypatch.setitem(llm._last_num_ctx, "m:1", 65536)
    "".join(c.chat([{"role": "user", "content": "x"}], "m:1", stream=False, background=True, num_ctx=8192))
    assert llm._last_num_ctx["m:1"] == 65536


def test_gpt_oss_helpers_think_low(monkeypatch):
    c, seen = _client(monkeypatch)
    "".join(c.chat([{"role": "user", "content": "x"}], "gpt-oss:20b", stream=False, background=True))
    assert seen["payload"]["think"] == "low" and seen["payload"]["options"]["num_predict"] == llm.BACKGROUND_NUM_PREDICT_EFFORT


def test_utility_chat_reports_the_done_reason(monkeypatch):
    c, _seen = _client(monkeypatch, lines=('{"message": {"content": "[{\\"fact\\""}, "done": false}',
                                           '{"message": {"content": ""}, "done": true, "done_reason": "length"}'))
    text, why = llm.utility_chat(c, [{"role": "user", "content": "x"}], "m:1")
    assert text == '[{"fact"' and why == "length"


def test_utility_chat_works_with_small_test_doubles():
    class Tiny:
        def chat(self, messages, model, stream=True, **kw):
            assert kw.get("background") is True
            return iter(["fine"])

    assert llm.utility_chat(Tiny(), [], "m") == ("fine", "")


def test_preload_protects_the_model_and_sends_the_chat_window(monkeypatch):
    sent = {}

    class Http:
        def post(self, path, json=None, **kw):
            sent.update(json)

            class R:
                def raise_for_status(self):
                    pass
            return R()

    c = llm.OllamaClient.__new__(llm.OllamaClient)
    c._ps_cache = {}
    monkeypatch.setattr(llm.OllamaClient, "_client_for", lambda self, model: Http())
    monkeypatch.setitem(llm._last_num_ctx, "gpt-oss:20b", 131072)
    assert c.preload("gpt-oss:20b")
    assert llm.is_protected("gpt-oss:20b") and sent["options"]["num_ctx"] == 131072


def test_warm_up_repairs_a_cpu_only_load(monkeypatch):
    from agent import gpumode
    from agent.ui import chat_app

    calls = []

    class C:
        def __init__(self):
            self.where = [{"size": 10, "size_vram": 0}, {"size": 10, "size_vram": 10}]

        def placement(self, model):
            return self.where.pop(0)

        def unload(self, model):
            calls.append("unload")

        def preload(self, model):
            calls.append("preload")

    monkeypatch.setattr(gpumode, "current_mode", lambda: gpumode.DUAL)
    monkeypatch.setattr(llm, "is_remote", lambda *a, **k: False)
    msg = chat_app.repair_cpu_placement(C(), "gpt-oss:20b")
    assert calls == ["unload", "preload"] and "now runs on the GPU" in msg


def test_same_model_tells_sizes_apart():
    assert llm.same_model("gpt-oss:20b", "gpt-oss:20b") and llm.same_model("qwen3", "qwen3:latest")
    assert not llm.same_model("gpt-oss:20b", "gpt-oss:120b")
