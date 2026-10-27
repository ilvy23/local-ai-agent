"""Public/private WhatsApp CLI boundary.

All fakes: no real ~/.config/agent/plugins/whatsapp plugin, no real data. These tests
simulate the FUTURE public export (agent/wa_private.py absent) by making its import
fail, and check that:

- the public surface (agent/wa_public.py, composed via agent/wa.py) still has every
  public command and nothing crashes;
- private commands never show up on the composed app when the private module can't
  import;
- the private module still attaches correctly when it *can* import (today's normal
  state in this repo);
- the home WhatsApp topic (agent/ui/launcher.py) hides the private WhatsApp entries instead of crashing, in both states;
- auto-reply/opener/reply never send without the explicit env gate - there is no path
  around `_send_allowed()`.
"""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest
from typer.main import get_command


def _command_names(wa_app) -> set[str]:
    return set(get_command(wa_app).commands.keys())


def _fresh_agent_wa(monkeypatch, *, block_private: bool):
    """Reimport agent.wa (and its two halves) fresh, optionally making
    `agent.wa_private` fail to import - the way it would if a public export simply
    dropped that file."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if block_private and (name == "agent.wa_private" or name.startswith("agent.wa_private.")):
            raise ImportError("simulated: agent/wa_private.py absent (public export)")
        return real_import(name, *args, **kwargs)

    if block_private:
        monkeypatch.setattr(builtins, "__import__", fake_import)

    for mod in ("agent.wa", "agent.wa_private", "agent.wa_public"):
        sys.modules.pop(mod, None)
    return importlib.import_module("agent.wa")


# --------------------------------------------------------------------------------- the boundary itself


def test_public_surface_survives_private_module_absent(monkeypatch):
    wa = _fresh_agent_wa(monkeypatch, block_private=True)
    assert wa.WA_PRIVATE_AVAILABLE is False

    names = _command_names(wa.wa_app)
    # Public: link/sync, search, auto-reply drafting (mode + allowlist), aliases.
    for must_have in ("status", "sync", "pair", "bridge", "chats", "recent",
                       "search", "merge-lids", "reply", "auto", "alias", "brief", "plans"):
        assert must_have in names, f"public command {must_have!r} missing when private module absent"

    # Private: profiles, deep analysis, digest/nudges, persona, self-reflection must NOT appear.
    for must_not_have in ("profile", "profiles", "summarize", "summaries", "digest", "nudges",
                          "health", "vibe", "redflags", "predict", "persona", "clone", "clones",
                          "clone-start", "clone-rename", "queue", "me", "repair", "speech-profile",
                          "watch", "refresh", "day-summary", "weekly-summary", "day-log", "moments",
                          "forget-summary", "forget-profile"):
        assert must_not_have not in names, f"private command {must_not_have!r} leaked without the private module"


def _needs_private():
    """The public export ships without agent/wa_private.py; these tests only make sense where it exists."""
    pytest.importorskip("agent.wa_private")


def test_private_surface_attaches_when_module_present(monkeypatch):
    _needs_private()
    wa = _fresh_agent_wa(monkeypatch, block_private=False)
    assert wa.WA_PRIVATE_AVAILABLE is True

    names = _command_names(wa.wa_app)
    # Everything public is still there...
    assert {"status", "sync", "reply", "auto", "search"} <= names
    # ...and now the private commands/sub-apps are reachable through the same wa_app.
    for must_have in ("profile", "digest", "health", "vibe", "redflags", "predict",
                      "persona", "clone", "queue", "me", "watch", "refresh"):
        assert must_have in names, f"private command {must_have!r} missing although the module imported fine"


def test_wa_public_module_has_no_dependency_on_private():
    """agent/wa_public.py must import and build its Typer app with zero reference to
    agent.wa_private - that's what lets a real export ship one file and drop the other."""
    import agent.wa_public as wa_public

    assert "wa_private" not in wa_public.__dict__
    assert not any("wa_private" in (getattr(v, "__module__", "") or "") for v in wa_public.__dict__.values())
    # No "import agent.wa_private" / "from agent.wa_private import ..." statement anywhere in the file.
    for line in open(wa_public.__file__, encoding="utf-8"):
        stripped = line.strip()
        assert not (stripped.startswith("import agent.wa_private") or stripped.startswith("from agent.wa_private"))


def test_registering_private_twice_is_independent_of_order(monkeypatch):
    _needs_private()
    """register_private(app) only decorates the Typer instance handed to it - calling
    it on a fresh public app (as agent/wa.py does at import time) must not require or
    assume any prior state."""
    wa = _fresh_agent_wa(monkeypatch, block_private=False)
    from agent.wa_private import register_private
    import agent.wa_public as wa_public

    fresh = importlib.reload(wa_public).wa_app
    before = _command_names(fresh)
    register_private(fresh)
    after = _command_names(fresh)
    assert after - before  # private commands were actually added
    assert before <= after  # nothing public was removed/replaced


# --------------------------------------------------------------------------------- home topic degrades, never crashes


def test_whatsapp_topic_hides_private_entries_when_unavailable(monkeypatch):
    from agent.ui import launcher

    monkeypatch.setattr(launcher, "wa_private_available", lambda: False)
    _title, sections = launcher.topic_sections("whatsapp")
    labels = {e.label for _, items in sections for e in items}
    assert "Chat analysis (not available in this build)" in labels
    for private_label_fragment in ("friendship", "Sum up a chat", "Talk to a clone", "Build a clone"):
        assert not any(private_label_fragment in label for label in labels)
    # The always-public entries are still there.
    assert any("Write a reply for me" in label for label in labels)
    assert any("auto-reply is doing" in label for label in labels)


def test_whatsapp_topic_shows_private_entries_when_available(monkeypatch):
    from agent.ui import launcher

    monkeypatch.setattr(launcher, "wa_private_available", lambda: True)
    _title, sections = launcher.topic_sections("whatsapp")
    labels = {e.label for _, items in sections for e in items}
    assert any("friendship" in label for label in labels)
    assert any("Talk to a clone" in label for label in labels)


def test_unavailable_notice_does_not_raise():
    import io

    from rich.console import Console

    from agent.ui import launcher

    console = Console(file=io.StringIO(), width=100)
    launcher.wa_private_unavailable_notice(console)  # must not raise
    assert "not available" in console.file.getvalue()


def test_execute_of_notice_entry_does_not_crash(monkeypatch):
    """The 'call' entry kind used for the disabled notice must run cleanly through the same execute() path as every
    other entry."""
    import io

    from rich.console import Console

    from agent.ui import launcher

    console = Console(file=io.StringIO(), width=100)
    entry = launcher.Entry("notice", call=launcher.wa_private_unavailable_notice)
    assert launcher.execute(entry, console, app=None) is True


# --------------------------------------------------------------------------------- auto-reply never has a bypass send path


def test_send_is_off_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_WA_ALLOW_SEND", raising=False)
    from agent.wa_shared import _send_allowed

    assert _send_allowed() is False


def test_send_requires_the_exact_env_value(monkeypatch):
    from agent.wa_shared import _send_allowed

    monkeypatch.setenv("AGENT_WA_ALLOW_SEND", "true")   # anything but the literal "1" must stay blocked
    assert _send_allowed() is False
    monkeypatch.setenv("AGENT_WA_ALLOW_SEND", "1")
    assert _send_allowed() is True


def test_wa_app_commands_that_can_send_all_check_the_gate():
    """Every public command whose source can reach a bridge .send(...) call must also
    reference the gate (_send_allowed/SEND_OFF) somewhere in the same function - a
    cheap static check that a future edit can't quietly add an unguarded send path."""
    import inspect

    import agent.wa_public as wa_public

    checked_any = False
    for name in ("wa_reply", "wa_auto_on", "wa_auto_flush", "wa_auto_open"):
        fn = getattr(wa_public, name)
        src = inspect.getsource(fn)
        if "bridge" in src.lower() and ".send(" in src:
            checked_any = True
            assert "_send_allowed" in src or "SEND_OFF" in src, f"{name} can send but doesn't check the gate"
    assert checked_any, "test didn't actually find any send-capable command to check"
