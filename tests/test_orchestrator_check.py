"""Tests for ensure_models_available() in agent.orchestrator."""

from unittest.mock import patch


def test_ensure_models_returns_dict():
    from agent.orchestrator import ensure_models_available

    with patch("agent.llm.OllamaClient.check_model_loaded", return_value=True):
        result = ensure_models_available()
    assert "main" in result
    assert "orchestrator" in result


def test_ensure_models_both_available():
    from agent.orchestrator import ensure_models_available

    with patch("agent.llm.OllamaClient.check_model_loaded", return_value=True):
        result = ensure_models_available()
    assert result["main"] is True
    assert result["orchestrator"] is True


def test_ensure_models_main_missing():
    from agent.orchestrator import ensure_models_available

    def fake_check(self, model):  # noqa: ANN001
        return "orchestrator" in model or "9b" in model

    with patch("agent.llm.OllamaClient.check_model_loaded", fake_check):
        result = ensure_models_available(
            {"models": {"main": "qwen3.8:27b", "orchestrator": "qwen3.5:9b"}}
        )
    assert result["main"] is False
    assert result["orchestrator"] is True


def test_ensure_models_never_raises():
    from agent.orchestrator import ensure_models_available

    with patch(
        "agent.llm.OllamaClient.check_model_loaded",
        side_effect=Exception("connection refused"),
    ):
        result = ensure_models_available()
    assert isinstance(result, dict)
    assert "main" in result


def test_ensure_models_returns_warnings_key():
    from agent.orchestrator import ensure_models_available

    with patch("agent.llm.OllamaClient.check_model_loaded", return_value=True):
        result = ensure_models_available()
    assert "warnings" in result
    assert isinstance(result["warnings"], list)


def test_ensure_models_warning_when_main_missing():
    from agent.orchestrator import ensure_models_available

    with patch("agent.llm.OllamaClient.check_model_loaded", return_value=False):
        result = ensure_models_available()
    assert result["main"] is False
    assert len(result["warnings"]) >= 1


def test_ensure_models_no_warnings_when_both_present():
    from agent.orchestrator import ensure_models_available

    with patch("agent.llm.OllamaClient.check_model_loaded", return_value=True):
        result = ensure_models_available()
    assert result["warnings"] == []


def test_ensure_models_uses_config_model_names():
    from agent.orchestrator import ensure_models_available

    seen: list[str] = []

    def record_check(self, model):  # noqa: ANN001
        seen.append(model)
        return True

    with patch("agent.llm.OllamaClient.check_model_loaded", record_check):
        ensure_models_available(
            {"models": {"main": "big-model:27b", "orchestrator": "small-model:9b"}}
        )
    assert "big-model:27b" in seen
    assert "small-model:9b" in seen
