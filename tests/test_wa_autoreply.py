"""Auto-reply queueing (whatsapp/autoreply.py): the "never reply to media" guard.
Plugin lives in ~/.config/agent; synthetic fixtures only, no real message content."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PLUGINS = Path.home() / ".config" / "agent" / "plugins"
pytestmark = pytest.mark.skipif(not (PLUGINS / "whatsapp" / "autoreply.py").exists(), reason="WhatsApp plugin not installed")
if PLUGINS.exists():
    sys.path.insert(0, str(PLUGINS))

JID = "friend@s.whatsapp.net"


class Store:
    def __init__(self):
        from whatsapp.schema import ensure_schema

        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)
        self.conn.execute("INSERT INTO wa_chats (jid, name, is_group) VALUES (?, 'Friend', 0)", (JID,))
        self.n = 0

    def msg(self, chat, text, me, ts, kind="text"):
        self.n += 1
        self.conn.execute(
            "INSERT INTO wa_messages (id, chat_jid, sender_jid, from_me, timestamp, text, type) VALUES (?,?,?,?,?,?,?)",
            (f"m{self.n}", chat, None if me else chat, 1 if me else 0, ts, text, kind),
        )
        self.conn.commit()

    def get_active_facts(self):
        return []


def make_store():
    """5 answered exchanges, then one more incoming text the user has NOT replied to yet -
    that's the one `handle_inbound` should pick up (see `_user_replied_since`)."""
    s = Store()
    for i in range(5):
        s.msg(JID, f"hey nachricht {i}", False, 1_700_000_000 + i * 10)
        s.msg(JID, f"antwort {i}", True, 1_700_000_005 + i * 10)
    s.msg(JID, "letzte echte frage", False, 1_700_000_090)
    return s


class FakeLLM:
    def chat(self, messages, model, stream=True, **kw):
        return iter(["ok klingt gut"])


def _config(mode="draft"):
    return {"whatsapp": {"auto_reply_contacts": ["Friend"], "auto_reply": {"mode": mode}}}


class _Resolved:
    def __init__(self, jid):
        self.jid = jid
        self.ambiguous = False


def _patch_resolve(monkeypatch):
    import whatsapp.tools as T

    monkeypatch.setattr(T, "resolve_contact", lambda store, name: _Resolved(JID))


def _patch_draft(monkeypatch, result="hey klar"):
    """draft_reply is imported *inside* handle_inbound, so patch where it's looked up: the module object."""
    import whatsapp.responder as R

    calls = []
    monkeypatch.setattr(R, "draft_reply", lambda *a, **k: (calls.append((a, k)), result)[1])
    return calls


def test_an_image_message_is_never_the_one_replied_to(monkeypatch):
    from whatsapp import autoreply as A

    _patch_resolve(monkeypatch)
    draft_calls = _patch_draft(monkeypatch)
    s = make_store()
    s.msg(JID, "[image]", False, 1_700_000_100, kind="image")

    n = A.handle_inbound(s, FakeLLM(), _config())
    assert n == 1  # still replies - to the last real TEXT message, not the image
    assert draft_calls[0][1]["incoming"] != "[image]" and draft_calls[0][1]["incoming"] == "letzte echte frage"


def test_an_image_only_burst_queues_nothing_but_moves_the_cursor_past_it(monkeypatch):
    from whatsapp import autoreply as A

    _patch_resolve(monkeypatch)
    draft_calls = _patch_draft(monkeypatch)
    s = make_store()
    # Drain the text backlog first, as a real run would.
    A.handle_inbound(s, FakeLLM(), _config())
    draft_calls.clear()

    s.msg(JID, "[image]", False, 1_700_000_200, kind="image")
    s.msg(JID, "[sticker]", False, 1_700_000_201, kind="sticker")
    n = A.handle_inbound(s, FakeLLM(), _config())
    assert n == 0 and not draft_calls

    # The cursor moved past both media rows - a later real text message is the next one seen.
    s.msg(JID, "echte frage", False, 1_700_000_300, kind="text")
    n2 = A.handle_inbound(s, FakeLLM(), _config())
    assert n2 == 1
    assert draft_calls[0][1]["incoming"] == "echte frage"


def test_a_captioned_image_still_does_not_count_as_the_text_to_answer(monkeypatch):
    """normalize.js stores a caption as e.g. '[image] nice pic' with type='image' - the type,
    not the text, decides; a caption must not be mistaken for a real message to answer."""
    from whatsapp import autoreply as A

    _patch_resolve(monkeypatch)
    draft_calls = _patch_draft(monkeypatch)
    s = make_store()
    A.handle_inbound(s, FakeLLM(), _config())
    draft_calls.clear()

    s.msg(JID, "[image] schau mal", False, 1_700_000_400, kind="image")
    n = A.handle_inbound(s, FakeLLM(), _config())
    assert n == 0 and not draft_calls


# --------------------------------------------------------------------------------- kill switch


@pytest.fixture(autouse=True)
def _isolated_halt_file(tmp_path, monkeypatch):
    """Every test gets its own HALT_FILE path - never the real plugin's file."""
    import whatsapp.autoreply as A

    monkeypatch.setattr(A, "HALT_FILE", str(tmp_path / "AUTOREPLY_HALT"))
    yield
    A.resume()  # leave no halt file behind even if a test forgot


def test_halt_stops_handle_inbound_and_resume_lifts_it(monkeypatch):
    from whatsapp import autoreply as A

    _patch_resolve(monkeypatch)
    draft_calls = _patch_draft(monkeypatch)
    s = make_store()

    assert not A.halted()
    A.halt("manual test stop")
    assert A.halted()
    n = A.handle_inbound(s, FakeLLM(), _config())
    assert n == 0 and not draft_calls  # nothing drafted at all, even though a real question is pending

    assert A.resume() is True
    assert not A.halted()
    n2 = A.handle_inbound(s, FakeLLM(), _config())
    assert n2 == 1  # normal operation resumes exactly where it left off


def test_resume_without_a_prior_halt_is_a_harmless_noop():
    from whatsapp import autoreply as A

    assert A.resume() is False


def test_halt_also_stops_flush_due(monkeypatch):
    from whatsapp import autoreply as A

    s = make_store()
    s.conn.execute(
        "INSERT INTO wa_autoreply (contact_jid, incoming_id, incoming_text, draft, status, reason, send_at, created_at) "
        "VALUES (?, 'm1', 'frage', 'entwurf', 'scheduled', NULL, 0, '2026-01-01T00:00:00Z')",
        (JID,),
    )
    s.conn.commit()
    A.halt()

    class FakeBridge:
        def send(self, jid, text):
            raise AssertionError("must never be called while halted")

    sent = A.flush_due(s, _config(mode="auto"), FakeBridge())
    assert sent == 0


# --------------------------------------------------------------------------------- allowlist ceiling


def test_check_allowlist_size_allows_up_to_the_ceiling():
    from whatsapp import autoreply as A

    A.check_allowlist_size(["a"] * A.MAX_ALLOWLIST)  # must not raise


def test_check_allowlist_size_rejects_one_over():
    from whatsapp import autoreply as A

    with pytest.raises(ValueError, match="ceiling"):
        A.check_allowlist_size(["a"] * (A.MAX_ALLOWLIST + 1))


# --------------------------------------------------------------------------------- full audit trail


def test_audit_shows_every_status_not_just_pending():
    from whatsapp import autoreply as A

    s = make_store()
    rows = [
        (JID, "held", "confidence gate"),
        (JID, "scheduled", None),
        (JID, "cancelled", "superseded"),
        (JID, "sent", None),
    ]
    for jid, status, reason in rows:
        s.conn.execute(
            "INSERT INTO wa_autoreply (contact_jid, incoming_id, incoming_text, draft, status, reason, send_at, created_at) "
            "VALUES (?, 'mX', 'frage', 'entwurf', ?, ?, 0, '2026-01-01T00:00:00Z')",
            (jid, status, reason),
        )
    s.conn.commit()

    out = A.audit(s)
    assert {r["status"] for r in out} == {"held", "scheduled", "cancelled", "sent"}
    assert len(out) == 4  # pending() would have shown only 2 of these (held+scheduled)


def test_audit_can_filter_to_one_contact():
    from whatsapp import autoreply as A

    s = make_store()
    other = "someone_else@s.whatsapp.net"
    s.conn.execute(
        "INSERT INTO wa_autoreply (contact_jid, incoming_id, incoming_text, draft, status, reason, send_at, created_at) "
        "VALUES (?, 'm1', 'a', 'b', 'sent', NULL, 0, '2026-01-01T00:00:00Z')", (JID,))
    s.conn.execute(
        "INSERT INTO wa_autoreply (contact_jid, incoming_id, incoming_text, draft, status, reason, send_at, created_at) "
        "VALUES (?, 'm2', 'a', 'b', 'sent', NULL, 0, '2026-01-01T00:00:00Z')", (other,))
    s.conn.commit()

    assert len(A.audit(s)) == 2
    assert len(A.audit(s, contact_jid=JID)) == 1
