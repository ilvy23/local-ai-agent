"""whatsapp/caliber.py: grouping chats by how casually/formally YOU write in them, so a
sparse chat can borrow real same-register examples instead of nothing. Synthetic fixtures
only - invented phrases, no real message content."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PLUGINS = Path.home() / ".config" / "agent" / "plugins"
pytestmark = pytest.mark.skipif(not (PLUGINS / "whatsapp" / "caliber.py").exists(), reason="WhatsApp plugin not installed")
if PLUGINS.exists():
    sys.path.insert(0, str(PLUGINS))

TIGHT_JID = "bestie@s.whatsapp.net"
TIGHT_JID_2 = "crew@s.whatsapp.net"
FORMAL_JID = "boss@s.whatsapp.net"
SPARSE_JID = "new_acquaintance@s.whatsapp.net"

TIGHT_TEXTS = ["jo", "ne lol", "digga wtf", "k bin da", "joa bock", "lmao nice", "ne alles gut",
               "jo passt", "k", "bro ernsthaft", "digga", "haha ja"]
FORMAL_TEXTS = [
    "Guten Tag, ich wollte nachfragen, ob das Treffen morgen noch stattfindet.",
    "Vielen Dank für die schnelle Rückmeldung, das passt gut in meinen Zeitplan.",
    "Ich melde mich, sobald ich mehr Informationen von meiner Seite habe.",
    "Entschuldigung für die späte Antwort, ich war die letzten Tage stark beschäftigt.",
]


class Store:
    def __init__(self):
        from whatsapp.schema import ensure_schema

        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)
        self.n = 0

    def msgs(self, chat, texts, from_me=1):
        for text in texts:
            self.n += 1
            self.conn.execute(
                "INSERT INTO wa_messages (id, chat_jid, sender_jid, from_me, timestamp, text, type) VALUES (?,?,?,?,?,?,'text')",
                (f"m{self.n}", chat, None, from_me, 1_700_000_000 + self.n, text),
            )
        self.conn.commit()


def test_tight_and_formal_chats_land_in_different_calibers():
    from whatsapp.caliber import caliber_for_chat

    s = Store()
    s.msgs(TIGHT_JID, TIGHT_TEXTS * 2)
    s.msgs(FORMAL_JID, FORMAL_TEXTS * 4)

    tight = caliber_for_chat(s, TIGHT_JID)
    formal = caliber_for_chat(s, FORMAL_JID)
    assert tight == "tight"
    assert formal == "formal"
    assert tight != formal


def test_a_chat_with_no_sent_messages_is_unknown():
    from whatsapp.caliber import caliber_for_chat

    s = Store()
    assert caliber_for_chat(s, "nobody@s.whatsapp.net") == "unknown"


def test_is_sparse_uses_the_own_text_message_count():
    from whatsapp.caliber import MIN_MESSAGES_FOR_OWN_VOICE, is_sparse

    s = Store()
    s.msgs(SPARSE_JID, ["hey"] * 3)
    assert is_sparse(s, SPARSE_JID)
    s.msgs(SPARSE_JID, ["hey"] * MIN_MESSAGES_FOR_OWN_VOICE)
    assert not is_sparse(s, SPARSE_JID)


def test_borrowed_examples_only_pulls_from_the_same_caliber_and_never_from_itself():
    from whatsapp.caliber import borrowed_examples

    s = Store()
    s.msgs(TIGHT_JID, TIGHT_TEXTS * 2)
    s.msgs(TIGHT_JID_2, TIGHT_TEXTS * 2)
    s.msgs(FORMAL_JID, FORMAL_TEXTS * 4)

    borrowed = borrowed_examples(s, TIGHT_JID, "tight", limit=50)
    assert borrowed  # found something
    assert all(t in TIGHT_TEXTS for t in borrowed)  # only tight-caliber text, never formal
    assert not set(borrowed).intersection(set(FORMAL_TEXTS))


def test_borrowed_examples_respects_the_limit():
    from whatsapp.caliber import borrowed_examples

    s = Store()
    s.msgs(TIGHT_JID, TIGHT_TEXTS * 10)
    s.msgs(TIGHT_JID_2, TIGHT_TEXTS * 10)
    borrowed = borrowed_examples(s, "someone_else@s.whatsapp.net", "tight", limit=7)
    assert len(borrowed) <= 7
