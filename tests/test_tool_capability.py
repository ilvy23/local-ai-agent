"""A chat model without tool support must be called out, not trusted.

Ollama accepts `tools=` for models whose template has no slot for them and just
drops them — the model never sees the tools and invents results instead (a
made-up `ls` listing looks exactly like a real one). These pin the detection.
"""

from __future__ import annotations

import httpx
import pytest
from rich.console import Console

from agent.llm import OllamaClient
from agent.tui import _warn_if_model_cannot_use_tools


def _client_reporting(capabilities):
    """An OllamaClient whose /api/show answers with `capabilities`."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            body = {} if capabilities is None else {"capabilities": capabilities}
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    return OllamaClient(transport=httpx.MockTransport(handler))


def test_detects_a_tools_capable_model():
    assert _client_reporting(["completion", "tools"]).supports_tools("qwen2.5:7b") is True


def test_detects_a_model_that_cannot_use_tools():
    assert _client_reporting(["completion"]).supports_tools("dolphin3:8b") is False


def test_unknown_capabilities_are_not_a_false_alarm():
    # No capabilities field, or an unreachable server: we don't know, so don't
    # claim the model is broken.
    assert _client_reporting(None).supports_tools("mystery") is None

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    assert OllamaClient(transport=httpx.MockTransport(dead)).supports_tools("m") is None


def test_capability_lookup_is_cached(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"capabilities": ["completion"]})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    client.supports_tools("m")
    client.supports_tools("m")
    assert len(calls) == 1  # asked once, remembered after


class _Rec:
    def __init__(self, supported):
        self._s = supported

    def supports_tools(self, model):  # noqa: ARG002
        return self._s


def _warned(supported) -> str:
    console = Console(record=True, width=100)
    _warn_if_model_cannot_use_tools(console, _Rec(supported), "some-model")
    return console.export_text()


def test_warns_when_the_model_cannot_use_tools():
    out = _warned(False)
    assert "make up" in out and "some-model" in out


def test_stays_quiet_when_tools_work_or_are_unknown():
    assert _warned(True).strip() == ""
    assert _warned(None).strip() == ""


def test_a_broken_capability_check_never_breaks_chat():
    class Boom:
        def supports_tools(self, model):  # noqa: ARG002
            raise RuntimeError("ollama exploded")

    console = Console(record=True, width=100)
    _warn_if_model_cannot_use_tools(console, Boom(), "m")  # must not raise
    assert console.export_text().strip() == ""
