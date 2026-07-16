"""Models must be evicted from VRAM when the chat ends.

A resident chat model leaves no room for the embedding model on a smaller GPU,
which is what made fact-embedding fail. Leaving the chat should give the card
back.
"""

from __future__ import annotations

import httpx
from rich.console import Console

from agent.llm import OllamaClient
from agent.tui import _unload_models_on_exit


def _client(loaded, record=None, fail_unload=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n} for n in loaded]})
        if request.url.path == "/api/generate":
            body = httpx.Request("POST", "x", content=request.content).content
            if record is not None:
                record.append(__import__("json").loads(body))
            if fail_unload:
                return httpx.Response(500, json={"error": "nope"})
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    return OllamaClient(transport=httpx.MockTransport(handler))


def test_lists_loaded_models():
    assert _client(["qwen2.5:7b", "bge-m3"]).loaded_models() == ["qwen2.5:7b", "bge-m3"]


def test_unload_sends_keep_alive_zero():
    sent = []
    assert _client(["qwen2.5:7b"], record=sent).unload("qwen2.5:7b") is True
    assert sent == [{"model": "qwen2.5:7b", "keep_alive": 0}]


def test_unload_all_frees_every_resident_model():
    sent = []
    freed = _client(["qwen2.5:7b", "bge-m3"], record=sent).unload_all()
    assert freed == ["qwen2.5:7b", "bge-m3"]
    assert [s["model"] for s in sent] == ["qwen2.5:7b", "bge-m3"]
    assert all(s["keep_alive"] == 0 for s in sent)


def test_nothing_loaded_is_not_an_error():
    assert _client([]).unload_all() == []


def test_a_failed_unload_is_reported_not_raised():
    assert _client(["m"], fail_unload=True).unload_all() == []


def test_unreachable_ollama_does_not_raise():
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = OllamaClient(transport=httpx.MockTransport(dead))
    assert client.loaded_models() == []
    assert client.unload_all() == []


class _Spy:
    def __init__(self):
        self.called = False

    def unload_all(self):
        self.called = True
        return ["qwen2.5:7b"]


def test_exit_hook_frees_by_default():
    spy = _Spy()
    console = Console(record=True, width=80)
    _unload_models_on_exit(console, spy, {})  # no config section -> default on
    assert spy.called
    assert "qwen2.5:7b" in console.export_text()


def test_exit_hook_respects_opt_out():
    spy = _Spy()
    _unload_models_on_exit(Console(), spy, {"ollama": {"unload_on_exit": False}})
    assert not spy.called


def test_exit_hook_never_raises_on_the_way_out():
    class Boom:
        def unload_all(self):
            raise RuntimeError("ollama died")

    _unload_models_on_exit(Console(), Boom(), {})  # must not propagate
