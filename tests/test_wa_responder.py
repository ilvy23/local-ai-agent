"""whatsapp/responder.py::_user_examples - blending in same-caliber examples from other
chats when a chat has too little history of its own. Synthetic fixtures only."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PLUGINS = Path.home() / ".config" / "agent" / "plugins"
pytestmark = pytest.mark.skipif(not (PLUGINS / "whatsapp" / "responder.py").exists(), reason="WhatsApp plugin not installed")
if PLUGINS.exists():
    sys.path.insert(0, str(PLUGINS))

SPARSE_JID = "new_contact@s.whatsapp.net"
TIGHT_JID = "bestie@s.whatsapp.net"
FORMAL_JID = "boss@s.whatsapp.net"

TIGHT_TEXTS = ["jo", "ne lol", "digga wtf", "k bin da", "joa bock", "lmao nice", "ne alles gut",
               "jo passt", "k", "bro ernsthaft", "digga", "haha ja", "ne ernsthaft", "k passt", "joa"]
FORMAL_TEXTS = [
    "Guten Tag, ich wollte nachfragen, ob das Treffen morgen noch stattfindet.",
    "Vielen Dank für die schnelle Rückmeldung, das passt gut in meinen Zeitplan.",
    "Ich melde mich, sobald ich mehr Informationen von meiner Seite habe.",
    "Entschuldigung für die späte Antwort, ich war die letzten Tage stark beschäftigt.",
] * 4


class Store:
    def __init__(self):
        from whatsapp.schema import ensure_schema

        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)
        self.n = 0

    def msgs(self, chat, texts):
        for i, text in enumerate(texts):
            self.n += 1
            self.conn.execute(
                "INSERT INTO wa_messages (id, chat_jid, sender_jid, from_me, timestamp, text, type) VALUES (?,?,NULL,1,?,?,'text')",
                (f"m{self.n}", chat, 1_700_000_000 + self.n, text),
            )
        self.conn.commit()


def test_a_sparse_tight_caliber_chat_borrows_tight_examples_not_formal_ones():
    from whatsapp.responder import _user_examples

    s = Store()
    s.msgs(SPARSE_JID, ["yo", "k"])  # only 2 real messages of its own - well under the sparse threshold
    s.msgs(TIGHT_JID, TIGHT_TEXTS)
    s.msgs(FORMAL_JID, FORMAL_TEXTS)

    examples = _user_examples(s, SPARSE_JID)
    lines = examples.splitlines()
    assert "yo" in lines and "k" in lines  # its own real lines are always kept
    assert any(t in lines for t in TIGHT_TEXTS)  # topped up from the same (tight) caliber
    assert not any(t in lines for t in FORMAL_TEXTS)  # never from the opposite register
    # the chat's own lines are last (most recent/most specific to this person)
    assert lines[-2:] == ["yo", "k"]


def test_a_chat_with_enough_history_never_borrows_anything():
    from whatsapp.responder import _user_examples, _USER_SAMPLE

    s = Store()
    own = [f"eigene nachricht {i}" for i in range(_USER_SAMPLE)]
    s.msgs(SPARSE_JID, own)
    s.msgs(TIGHT_JID, TIGHT_TEXTS)

    examples = _user_examples(s, SPARSE_JID).splitlines()
    assert examples == own
    assert not any(t in examples for t in TIGHT_TEXTS)


def test_a_brand_new_contact_with_zero_history_gets_nothing_to_borrow_from(monkeypatch):
    """No caliber can be read from zero messages, so there is nothing to blend in -
    _user_examples must not crash, and draft_reply's own <5-messages gate (unchanged)
    is still what actually withholds a draft for a contact this new."""
    from whatsapp.responder import _user_examples

    s = Store()
    assert _user_examples(s, "total_stranger@s.whatsapp.net") == ""
