"""Item 1 (D5): the brief card - the user's requirements verbatim, numbered, pinned: byte-identical through compaction, a
history fold, masks and a resume."""

from __future__ import annotations

import json

import pytest

from agent.codeagent import brief as B, context as C
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_context import Summarizer, fill
from tests.test_codeagent_engine import FakeLLM, engine, say  # noqa: F401

LENA = ("Rewrite Lena in lena.py. Lena is a CHEERFUL BAKER who talks like a classic SHOPKEEPER and who does NOT BELONG TO THE\n"
        "WINTER CHAPTERS AT ALL - she is a BREATH OF FRESH AIR in this quiet town. She does not fit into the story (that is the joke).\n"
        "Her voice: bossy, flustered (\"I-it's not like I baked this for you! Sit down.\"), secretly kind.\n"
        "- Every text must be different: no sentence reused.\n"
        "- Keep LENA_OPEN, LENA_TREE and place_text(s) with the same shapes.\n")


def test_units_keep_the_users_words_join_hard_wraps_and_never_split_a_quote():
    items = B.extract(LENA)
    assert items[1] == ("Lena is a CHEERFUL BAKER who talks like a classic SHOPKEEPER and who does NOT BELONG TO THE WINTER CHAPTERS AT ALL - "
                        "she is a BREATH OF FRESH AIR in this quiet town.")                 # the hard wrap is gone, every word is the user's
    assert any("\"I-it's not like I baked this for you! Sit down.\"" in i for i in items)   # one unit, the quote not cut at "you!"
    assert "Every text must be different: no sentence reused." in items
    card = B.make(LENA)
    assert card.startswith(B.BRIEF_MARKER) and '1. "Rewrite Lena in lena.py."' in card


def test_a_long_request_keeps_the_requirement_sentences_first():
    filler = " ".join(f"Background sentence number {i} tells a story." for i in range(200))
    text = filler + " The texts MUST all be different. Never mention counting."
    items = B.extract(text, max_chars=400)
    assert "The texts MUST all be different." in items and "Never mention counting." in items
    assert sum(len(i) + 8 for i in items) <= 400


def test_short_requests_and_directives_get_no_card():
    assert B.make("fix the typo in main.py") is None
    assert B.make("AUTOPILOT RESTART: " + "x " * 200) is None


def test_card_is_pinned_before_the_request_and_survives_compaction_byte_identical(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([say("ok")]))
    eng.turn(LENA)
    card = eng.brief()
    msgs = eng.ctx.messages
    idx = next(i for i, m in enumerate(msgs) if B.is_card(m))
    assert msgs[idx + 1]["role"] in ("user", "system") and any(m["role"] == "user" for m in msgs[idx + 1:])
    eng.ctx.llm = Summarizer()
    eng.ctx.summarize_in_place = None
    eng.ctx.add_user("next request")                  # the card is now in the middle of the conversation
    fill(eng.ctx, 12)
    assert eng.ctx.compact()                          # a forced compaction
    cards = [m["content"] for m in eng.ctx.messages if B.is_card(m)]
    assert cards == [card]
    C_mask, _saved = C.mask_observations(eng.ctx.messages, repo, keep_last=0, min_chars=10)
    assert [m["content"] for m in C_mask if B.is_card(m)] == [card]


def test_card_survives_a_history_fold(tmp_path):
    m = C.ContextManager(Summarizer(), {"models": {"background": "m"}}, 10000, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    m.add({"role": "system", "content": "sys"})
    card = B.make(LENA)
    m.add({"role": "system", "content": card})
    m.add_user(LENA)
    fill(m, 12)
    m.add_user("continue")
    assert m.fold_history()
    assert [x["content"] for x in m.messages if B.is_card(x)] == [card]


def test_card_survives_a_resume_and_newest_cards_win(tmp_path, monkeypatch):
    from agent.codeagent import session as S, sessionstore as SS

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    eng, repo, _ = engine(tmp_path, FakeLLM([say("ok")]))
    eng.session_id = SS.new_id()
    eng.turn(LENA)
    S.save_session(eng)
    eng2, _r, _o = engine(tmp_path, FakeLLM([]))
    S.resume_into(eng2, SS.latest(eng.root))
    assert eng2.brief() == eng.brief() and [m for m in eng2.ctx.messages if B.is_card(m)][0]["content"] == eng.brief()
    for k in range(3):
        eng2._pin_brief(B.render([f"task {k} requirement one", f"task {k} requirement two"]))
    cards = [m["content"] for m in eng2.ctx.messages if B.is_card(m)]
    assert len(cards) == B.MAX_CARDS and "task 2" in cards[-1] and eng.brief() not in cards
    assert json.dumps(eng2.briefs)


@pytest.mark.parametrize("text", ["- a bullet that is long enough\n- another bullet item here", "One sentence here. Another sentence here."])
def test_units_split_bullets_and_sentences(text):
    assert len(B._units(text)) == 2
