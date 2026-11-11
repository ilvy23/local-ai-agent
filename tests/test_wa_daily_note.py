"""whatsapp/daily_note.py: building today's availability note from the "message
yourself" self-chat, instead of only the CLI brief. Synthetic fixtures only."""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

PLUGINS = Path.home() / ".config" / "agent" / "plugins"
pytestmark = pytest.mark.skipif(not (PLUGINS / "whatsapp" / "daily_note.py").exists(), reason="WhatsApp plugin not installed")
if PLUGINS.exists():
    sys.path.insert(0, str(PLUGINS))

ME = "15551234567"
SELF_JID = f"{ME}@s.whatsapp.net"


@pytest.fixture(autouse=True)
def _utc(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


class Store:
    def __init__(self):
        from whatsapp.schema import ensure_schema

        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)
        self.n = 0

    def msg(self, chat, text, ts, kind="text", from_me=1):
        self.n += 1
        self.conn.execute(
            "INSERT INTO wa_messages (id, chat_jid, sender_jid, from_me, timestamp, text, type) VALUES (?,?,?,?,?,?,?)",
            (f"m{self.n}", chat, None, from_me, ts, text, kind),
        )
        self.conn.commit()


class FakeBridge:
    def __init__(self, me=ME, raises=False):
        self.me = me
        self.raises = raises

    def status(self):
        if self.raises:
            raise RuntimeError("bridge down")
        return {"me": self.me}


TODAY_NOON = int(time.mktime(time.strptime(f"{__import__('datetime').date.today()} 12:00:00", "%Y-%m-%d %H:%M:%S")))


def test_own_jid_from_bridge_status():
    from whatsapp.daily_note import own_jid

    assert own_jid(FakeBridge()) == SELF_JID
    assert own_jid(FakeBridge(me=None)) is None
    assert own_jid(FakeBridge(raises=True)) is None


def test_ingest_builds_a_note_from_todays_self_messages():
    from whatsapp.daily_note import get_note, ingest_from_self_chat

    s = Store()
    s.msg(SELF_JID, "treffe tom um 18 uhr", TODAY_NOON)
    s.msg(SELF_JID, "danach nichts mehr geplant", TODAY_NOON + 60)

    note = ingest_from_self_chat(s, FakeBridge())
    assert note is not None
    assert "treffe tom um 18 uhr" in note and "danach nichts mehr geplant" in note
    # oldest first - a running brain-dump reads in the order it was written
    assert note.index("treffe tom") < note.index("danach nichts")
    assert get_note(s) == note


def test_media_and_other_chats_are_never_mistaken_for_the_brief():
    from whatsapp.daily_note import ingest_from_self_chat

    s = Store()
    s.msg(SELF_JID, "[image]", TODAY_NOON, kind="image")
    s.msg("someoneelse@s.whatsapp.net", "treffe tom um 18 uhr", TODAY_NOON)
    s.msg(SELF_JID, "ok hier text", TODAY_NOON, from_me=0)  # incoming to yourself (edge case) - not from_me
    assert ingest_from_self_chat(s, FakeBridge()) is None


def test_an_existing_note_survives_a_quiet_day(monkeypatch):
    """No self-chat message today must never wipe a note set another way (CLI brief,
    or an earlier pass today)."""
    from whatsapp.daily_note import get_note, ingest_from_self_chat, set_note

    s = Store()
    set_note(s, "Manuell gesetzt über die CLI")
    assert ingest_from_self_chat(s, FakeBridge()) is None
    assert get_note(s) == "Manuell gesetzt über die CLI"


def test_bridge_unreachable_is_not_fatal():
    from whatsapp.daily_note import ingest_from_self_chat

    s = Store()
    assert ingest_from_self_chat(s, FakeBridge(raises=True)) is None
