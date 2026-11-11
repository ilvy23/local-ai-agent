"""Permanent guard: this plugin must never gain a way to POST a WhatsApp Status/Story or
broadcast anything, even by accident later. A past incident involved exactly this - an
autonomous post to the user's own Status, on top of mass-messaging every past contact.
Static text scan only; reads source code, never message content."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PLUGIN_DIR = Path.home() / ".config" / "agent" / "plugins" / "whatsapp"
pytestmark = pytest.mark.skipif(not PLUGIN_DIR.exists(), reason="WhatsApp plugin not installed")

# Baileys/WhatsApp-Web API shapes for POSTING a status/broadcast. Reading/filtering
# "@broadcast" (status messages the user receives) is fine and common; these are the
# write-side signatures that would have to exist for the plugin to ever POST one.
_FORBIDDEN = re.compile(
    r"sendStatus|updateProfileStatus|statusJidList|postStatus|"
    r"""send\w*\(\s*['"]status@broadcast['"]""",
)

_SOURCE_GLOBS = ("*.py", "bridge/*.js")


def _plugin_source_files():
    for pattern in _SOURCE_GLOBS:
        for f in PLUGIN_DIR.glob(pattern):
            if f.is_file() and "node_modules" not in f.parts:
                yield f


def test_no_status_or_broadcast_posting_capability_exists_anywhere():
    hits = []
    for f in _plugin_source_files():
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if _FORBIDDEN.search(line):
                hits.append(f"{f.relative_to(PLUGIN_DIR)}:{i}: {line.strip()}")
    assert not hits, "A Status/broadcast-posting capability was found - this must never exist:\n" + "\n".join(hits)


def test_the_guard_pattern_actually_catches_something(tmp_path):
    """Sanity check on the regex itself, so a typo in it can't silently make the guard above
    pass vacuously forever."""
    from tests.test_wa_no_status_posting import _FORBIDDEN

    assert _FORBIDDEN.search('await sock.sendMessage("status@broadcast", { image: buf })')
    assert _FORBIDDEN.search("sock.updateProfileStatus('hi')")
    assert not _FORBIDDEN.search("jid.endsWith('@broadcast')")  # filtering, not posting
