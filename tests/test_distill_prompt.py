"""The distiller must catch how people actually talk, without letting junk back in.

Two failure modes pull against each other:
  - too loose: it hoovers up file paths and tool output (memory filled with
    `ls` results and, once, the plot of 1984)
  - too strict: it drops real disclosures. "ABBA is my favourite band" and
    "my dog is called Rex" were both extracted as nothing, because their
    grammatical subject isn't "I".

These pin the prompt's contract without needing a live model.
"""

from __future__ import annotations

from agent.memory.distill import _PROMPT


def test_prompt_teaches_indirect_phrasing():
    """The misses were all sentences that aren't grammatically about "I"."""
    assert "favourite band" in _PROMPT
    assert "dog is called Rex" in _PROMPT
    assert "grammatical subject" in _PROMPT


def test_prompt_still_excludes_the_junk_that_polluted_memory():
    for banned in ("files", "paths", "directories", "tool output"):
        assert banned in _PROMPT
    assert "requests" in _PROMPT  # "list my music" is not a fact about you


def test_prompt_asks_for_a_json_array_and_empty_when_nothing():
    assert "JSON array" in _PROMPT
    assert "[]" in _PROMPT


def test_prompt_does_not_hand_the_model_reusable_content():
    """The old prompt's example facts ("prefers tea over coffee", "has a sister
    named Lena") came back verbatim as extracted facts — the model copied the
    sample instead of reading the transcript."""
    assert "prefers tea over coffee" not in _PROMPT
    assert "has a sister named Lena" not in _PROMPT


def test_prompt_leads_with_what_to_include():
    """Ordering matters: the old prompt opened with four DO-NOTs and refused
    real disclosures. Include-guidance should come before the exclusions."""
    assert _PROMPT.index("Include anything lasting") < _PROMPT.index("Leave out")
