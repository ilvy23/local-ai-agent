"""`agent wa` — composes the public and private WhatsApp CLI surfaces into one Typer app.

This file is the seam the rest of the codebase (`agent/main.py`'s `from agent.wa import
wa_app`) was already written against, so it stays as a stable import path while the
actual commands live in:

- `agent/wa_public.py`  — link/sync, search, auto-reply drafting (public; ships in an
  open-source export).
- `agent/wa_private.py` — optional extra analysis commands in a separate module that is
  not part of the open-source build.

`agent/wa_private.py` is attached defensively: if it fails to import (as it will once a
public export drops the file, and as tests can simulate), `wa_app` still has every public
command and `WA_PRIVATE_AVAILABLE` is False so `agent/ui/launcher.py` and anything else that cares
can hide the private entries instead of crashing.
"""

from __future__ import annotations

from agent.wa_public import auto_pull, wa_app  # noqa: F401 - re-exported for callers

try:
    from agent.wa_private import register_private

    register_private(wa_app)
    WA_PRIVATE_AVAILABLE = True
except Exception:  # noqa: BLE001 - a broken/missing private module must never break public WhatsApp commands
    WA_PRIVATE_AVAILABLE = False
