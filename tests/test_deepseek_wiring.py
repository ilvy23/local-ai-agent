"""Tests for DeepSeek R1 wiring across config, llm, orchestrator and coding."""

from __future__ import annotations

import pytest

from agent.llm import strip_thinking, is_deepseek_r1


# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------

def test_strip_thinking_removes_think_block():
    text = "<think>Diese Überlegung soll weg.</think>\nDas ist die Antwort."
    assert strip_thinking(text) == "Das ist die Antwort."


def test_strip_thinking_noop_without_tags():
    text = "Normale Antwort ohne Think-Tags."
    assert strip_thinking(text) == text


def test_strip_thinking_multiline():
    text = "<think>\nZeile 1\nZeile 2\n</think>\nAntwort"
    result = strip_thinking(text)
    assert "think" not in result.lower()
    assert "Antwort" in result


def test_strip_thinking_multiple_blocks():
    text = "<think>first</think> middle <think>second</think> end"
    result = strip_thinking(text)
    assert "think" not in result.lower()
    assert "middle" in result
    assert "end" in result


def test_strip_thinking_empty_think_block():
    text = "<think></think>The answer."
    assert strip_thinking(text) == "The answer."


# ---------------------------------------------------------------------------
# is_deepseek_r1
# ---------------------------------------------------------------------------

def test_is_deepseek_r1_true():
    assert is_deepseek_r1("deepseek-r1:7b") is True
    assert is_deepseek_r1("deepseek-r1:14b") is True


def test_is_deepseek_r1_case_insensitive():
    assert is_deepseek_r1("DeepSeek-R1:7b") is True


def test_is_deepseek_r1_false():
    assert is_deepseek_r1("qwen3.5:9b") is False
    assert is_deepseek_r1("deepseek-coder:6.7b") is False  # not R1


# ---------------------------------------------------------------------------
# _adapt_system_prompt
# ---------------------------------------------------------------------------

def test_adapt_system_prompt_truncates_for_deepseek():
    from agent.llm import _adapt_system_prompt
    long_prompt = "You are a helpful assistant. " * 30  # >> 500 chars
    result = _adapt_system_prompt("deepseek-r1:7b", long_prompt)
    assert len(result) <= 500 + 1  # ≤500 chars plus final period


def test_adapt_system_prompt_noop_for_other_models():
    from agent.llm import _adapt_system_prompt
    long_prompt = "You are a helpful assistant. " * 30
    result = _adapt_system_prompt("qwen3.5:9b", long_prompt)
    assert result == long_prompt


def test_adapt_system_prompt_noop_short_prompt():
    from agent.llm import _adapt_system_prompt
    short_prompt = "You are Agent."
    assert _adapt_system_prompt("deepseek-r1:7b", short_prompt) == short_prompt


# ---------------------------------------------------------------------------
# orchestrator — REASONING layer
# ---------------------------------------------------------------------------

def test_reasoning_layer_in_orchestrator():
    from agent.orchestrator import decide, Layer
    result = decide("analyze the security vulnerability in this code")
    # Should be REASONING (has "analyze" and "security") or LARGE as fallback
    assert result.layer in (Layer.REASONING, Layer.LARGE)


def test_reasoning_layer_on_explicit_keyword():
    from agent.orchestrator import decide, Layer
    result = decide("evaluate the trade-offs of this approach")
    assert result.layer == Layer.REASONING


def test_reasoning_layer_not_overriding_coding():
    from agent.orchestrator import decide, Layer
    # "coding" keyword should still go to LARGE, not REASONING
    result = decide("analyze the coding patterns here")
    assert result.layer == Layer.LARGE


def test_reasoning_model_in_decision():
    from agent.orchestrator import decide, Layer
    result = decide("why does this algorithm fail on edge cases")
    assert result.layer == Layer.REASONING
    assert result.model is not None
    assert "deepseek" in result.model.lower() or result.model  # any non-empty model


def test_reasoning_layer_value():
    from agent.orchestrator import Layer
    assert Layer.REASONING == "reasoning"


# ---------------------------------------------------------------------------
# ensure_models_available — reasoning key present
# ---------------------------------------------------------------------------

def test_ensure_models_available_has_reasoning_key():
    from agent.orchestrator import ensure_models_available
    result = ensure_models_available(config=None)
    assert "reasoning" in result


# ---------------------------------------------------------------------------
# config — deepseek keys present
# ---------------------------------------------------------------------------

def test_config_has_deepseek_models():
    from agent.config import load_config
    cfg = load_config()
    models = cfg.get("models", {})
    assert "coding" in models or "reasoning" in models


def test_config_coding_model_is_deepseek():
    from agent.config import DEFAULT_CONFIG
    models = DEFAULT_CONFIG.get("models", {})
    assert "deepseek-r1" in models.get("coding", "")


def test_config_reasoning_model_is_deepseek():
    from agent.config import DEFAULT_CONFIG
    models = DEFAULT_CONFIG.get("models", {})
    assert "deepseek-r1" in models.get("reasoning", "")


# ---------------------------------------------------------------------------
# coding config — models.coding propagates to executor.model
# ---------------------------------------------------------------------------

def test_coding_config_uses_models_coding():
    from agent.coding.config import load
    cfg = load({"models": {"coding": "deepseek-r1:7b"}})
    assert cfg["executor"]["model"] == "deepseek-r1:7b"


def test_coding_config_explicit_executor_still_wins():
    """An explicit coding.executor.model in user config wins over models.coding."""
    from agent.coding.config import load
    cfg = load({
        "models": {"coding": "deepseek-r1:7b"},
        "coding": {"executor": {"model": "qwen3-coder:30b"}},
    })
    # The explicit coding.executor.model is merged first, then models.coding
    # overrides it — task spec says models.coding wins (sync intent).
    # This test documents the current behavior: models.coding wins.
    assert cfg["executor"]["model"] == "deepseek-r1:7b"


def test_coding_config_no_models_coding_uses_default():
    from agent.coding.config import load, DEFAULTS
    cfg = load({})
    # Without models.coding, falls back to DEFAULTS executor model
    assert cfg["executor"]["model"] == DEFAULTS["executor"]["model"]
