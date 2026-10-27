"""Tests for agent.orchestrator — three-layer routing logic."""

import pytest

from agent.orchestrator import (
    Layer,
    UNCERTAINTY_MARKERS,
    decide,
    has_uncertainty,
)

_CFG = {
    "models": {
        "main": "big-model:27b",
        "orchestrator": "small-model:9b",
        "fallback": "fallback-model:9b",
    }
}


# ---------------------------------------------------------------------------
# decide() — layer selection
# ---------------------------------------------------------------------------


def test_decide_returns_native_for_pure_status_task():
    decision = decide("status", config=_CFG)
    assert decision.layer == Layer.NATIVE
    assert decision.model is None


def test_decide_returns_native_for_ping():
    decision = decide("ping", config=_CFG)
    assert decision.layer == Layer.NATIVE
    assert decision.model is None


def test_decide_returns_large_for_coding_task():
    decision = decide("write a coding solution", config=_CFG)
    assert decision.layer == Layer.LARGE
    assert decision.model == _CFG["models"]["main"]


def test_decide_returns_large_for_planning_task():
    decision = decide("help me with planning the sprint", config=_CFG)
    assert decision.layer == Layer.LARGE
    assert decision.model == _CFG["models"]["main"]


def test_decide_returns_large_when_failure_count_two():
    decision = decide("do something", config=_CFG, failure_count=2)
    assert decision.layer == Layer.LARGE
    assert decision.model == _CFG["models"]["main"]


def test_decide_returns_large_when_failure_count_exceeds_threshold():
    decision = decide("do something", config=_CFG, failure_count=5)
    assert decision.layer == Layer.LARGE


def test_decide_returns_large_when_context_has_uncertainty_marker():
    decision = decide(
        "summarise this file",
        context="i'm not sure this is the right approach",
        config=_CFG,
    )
    assert decision.layer == Layer.LARGE
    assert decision.model == _CFG["models"]["main"]


def test_decide_returns_small_for_generic_task():
    decision = decide("list the files in this folder", config=_CFG)
    assert decision.layer == Layer.SMALL
    assert decision.model == _CFG["models"]["orchestrator"]


def test_decide_uses_default_config_when_none_passed():
    # Should not raise and should return a valid layer
    decision = decide("tell me a joke")
    assert decision.layer in (Layer.NATIVE, Layer.SMALL, Layer.LARGE)


# ---------------------------------------------------------------------------
# has_uncertainty()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", list(UNCERTAINTY_MARKERS))
def test_has_uncertainty_returns_true_for_each_marker(marker: str):
    assert has_uncertainty(f"The result is {marker} at this point.")


def test_has_uncertainty_returns_true_case_insensitive():
    assert has_uncertainty("I'M NOT SURE what to do here.")


def test_has_uncertainty_returns_false_for_clean_confident_text():
    assert not has_uncertainty("The build succeeded and all tests passed.")


def test_has_uncertainty_returns_false_for_empty_string():
    assert not has_uncertainty("")


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["", "   ", "???", "!!!"])
def test_decide_does_not_route_contentless_task_to_native(task: str):
    """No status keyword present → not a status check, so never model=None."""
    decision = decide(task, config=_CFG)
    assert decision.layer is not Layer.NATIVE
    assert decision.model is not None


@pytest.mark.parametrize("task", ["encoding a file", "decoding the payload",
                                  "scaffolding a project", "barcoding labels"])
def test_decide_does_not_treat_coding_substring_as_coding(task: str):
    """'coding' inside another word must not escalate to the LARGE layer."""
    decision = decide(task, config=_CFG)
    assert decision.layer == Layer.SMALL
    assert decision.model == _CFG["models"]["orchestrator"]


def test_decide_still_matches_coding_as_a_whole_word():
    assert decide("do some coding, please", config=_CFG).layer == Layer.LARGE
    assert decide("planning: sketch the sprint", config=_CFG).layer == Layer.LARGE


def test_decide_survives_config_without_models_block():
    """A partial config must fall back to defaults, not raise KeyError."""
    decision = decide("write a coding solution", config={})
    assert decision.layer == Layer.LARGE
    assert decision.model  # non-empty fallback default


def test_decide_survives_blank_model_names():
    decision = decide("list some files", config={"models": {"orchestrator": ""}})
    assert decision.model
