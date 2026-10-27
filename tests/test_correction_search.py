"""Being corrected should send it to the web, not make it apologise and guess.

Told "no, those songs aren't on that album", the agent abandoned a correct,
web-sourced answer and invented an album that doesn't exist. Being told you're
wrong is the strongest evidence your memory is unreliable — and it's exactly
when a small model's reflex is to agree and confabulate.
"""

from __future__ import annotations

from agent.tui import (
    _CORRECTION_CHECK,
    _CORRECTION_NUDGE,
    _is_factual_correction,
    _looks_like_disagreement,
)

CONFIG = {"models": {"background": "qwen2.5:7b"}}


class _Says:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def chat(self, **_kw):
        self.calls += 1
        return iter([self.answer])


def test_cue_fires_on_real_disagreements():
    for text in (
        "oh, no these songs arent included in there haha",
        "so you were wrong in your previous message haha",
        "that's not right",
        "nope, wrong band",
        "actually it came out in 1996",
    ):
        assert _looks_like_disagreement(text), text


def test_cue_stays_quiet_on_ordinary_messages():
    """It gates an LLM call, so it must not fire on every other sentence."""
    for text in (
        "haha thanks!",
        "can you also list the b-sides?",
        "what songs are on the album",
        "cool, tell me more",
        "read /tmp/notes.txt for me",
    ):
        assert not _looks_like_disagreement(text), text


def test_classifier_reads_yes_and_no():
    assert _is_factual_correction(_Says("yes"), CONFIG, "It was 2002.", "no that is wrong") is True
    assert _is_factual_correction(_Says("no"), CONFIG, "It was 2002.", "thanks!") is False
    assert _is_factual_correction(_Says("Yes.\n"), CONFIG, "a", "x") is True  # tolerant


def test_a_broken_classifier_leaves_the_turn_alone():
    class Boom:
        def chat(self, **_kw):
            raise RuntimeError("ollama down")

    # Never raise, and don't claim a correction we couldn't verify.
    assert _is_factual_correction(Boom(), CONFIG, "It was 2002.", "no that is wrong") is False


def test_nudge_forbids_both_failure_modes():
    """It must neither answer from memory nor simply cave — both produce a
    confident wrong answer."""
    assert "web_search" in _CORRECTION_NUDGE
    assert "NOT answer from memory" in _CORRECTION_NUDGE
    assert "NOT simply" in _CORRECTION_NUDGE and "agree" in _CORRECTION_NUDGE


def test_classifier_prompt_asks_a_plain_yes_no():
    """A small model handed a list of reasons and asked to pick a label just
    parrots the salient word — ask the narrow question instead."""
    assert "'yes' or 'no'" in _CORRECTION_CHECK
    assert "their own wording" in _CORRECTION_CHECK  # the benign case stays 'no'
