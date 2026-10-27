"""Tests for the routing helpers added to agent.llm.OllamaClient."""

from unittest.mock import patch

import httpx

from agent.llm import OllamaClient

_CFG = {
    "models": {
        "main": "big-model:27b",
        "orchestrator": "small-model:9b",
        "fallback": "fallback-model:9b",
    }
}


# ---------------------------------------------------------------------------
# route()
# ---------------------------------------------------------------------------


def test_route_returns_main_model_for_coding_task():
    client = OllamaClient()
    result = client.route("write a coding task", config=_CFG)
    assert result == _CFG["models"]["main"]


def test_route_returns_main_model_when_planning_in_task():
    client = OllamaClient()
    result = client.route("help me with planning", config=_CFG)
    assert result == _CFG["models"]["main"]


def test_route_returns_orchestrator_for_generic_task():
    client = OllamaClient()
    result = client.route("what is the weather today?", config=_CFG)
    assert result == _CFG["models"]["orchestrator"]


def test_route_uses_default_config_when_none_passed():
    client = OllamaClient()
    # Should not raise; default config uses qwen3.5:9b for everything
    result = client.route("some task")
    assert isinstance(result, str)
    assert result  # non-empty


# ---------------------------------------------------------------------------
# with_fallback()
# ---------------------------------------------------------------------------


def test_with_fallback_returns_preferred_when_model_is_loaded():
    client = OllamaClient()
    with patch.object(client, "check_model_loaded", return_value=True):
        result = client.with_fallback("big-model:27b", _CFG)
    assert result == "big-model:27b"


def test_with_fallback_returns_fallback_when_model_not_loaded():
    client = OllamaClient()
    with patch.object(client, "check_model_loaded", return_value=False):
        result = client.with_fallback("big-model:27b", _CFG)
    assert result == _CFG["models"]["fallback"]


def test_with_fallback_never_raises_when_check_raises():
    client = OllamaClient()
    with patch.object(client, "check_model_loaded", side_effect=RuntimeError("boom")):
        result = client.with_fallback("any-model:9b", _CFG)
    assert result == _CFG["models"]["fallback"]


# ---------------------------------------------------------------------------
# check_model_loaded()
# ---------------------------------------------------------------------------


def test_check_model_loaded_returns_true_when_model_in_ps_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3.5:9b", "size": 1, "size_vram": 1}]},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert client.check_model_loaded("qwen3.5:9b") is True


def test_check_model_loaded_returns_true_on_base_name_match():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3.5:latest", "size": 1, "size_vram": 1}]},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert client.check_model_loaded("qwen3.5:9b") is True


def test_check_model_loaded_returns_false_when_model_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "other-model:7b"}]},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert client.check_model_loaded("qwen3.5:9b") is False


def test_check_model_loaded_returns_false_gracefully_when_ollama_not_running(monkeypatch):
    """No mock — if Ollama is actually down, the method must return False."""
    # Point at ports that are virtually guaranteed to be closed - the default server AND the
    # dual-GPU Vulkan server, which check_model_loaded also asks (a real one may be running).
    monkeypatch.setenv("AGENT_VULKAN_OLLAMA_URL", "http://127.0.0.1:19998")
    client = OllamaClient(base_url="http://localhost:19999")
    result = client.check_model_loaded("qwen3.5:9b")
    assert result is False


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


def test_route_does_not_match_coding_inside_another_word():
    """'encoding' contains 'coding' but is not a coding task."""
    client = OllamaClient()
    assert client.route("fix the encoding of this file", config=_CFG) == \
        _CFG["models"]["orchestrator"]


def test_route_survives_config_without_models_block():
    client = OllamaClient()
    assert client.route("write a coding task", config={})  # no KeyError, non-empty


def test_with_fallback_returns_preferred_when_no_fallback_configured():
    """Documented as 'never raises' — a config with no fallback must not KeyError."""
    client = OllamaClient()
    with patch.object(client, "check_model_loaded", return_value=False):
        assert client.with_fallback("big-model:27b", {}) == "big-model:27b"


def test_with_fallback_ignores_blank_fallback_name():
    client = OllamaClient()
    with patch.object(client, "check_model_loaded", return_value=False):
        result = client.with_fallback("big-model:27b", {"models": {"fallback": ""}})
    assert result == "big-model:27b"
