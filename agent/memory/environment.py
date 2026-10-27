"""System environment facts injected into every chat context.

Gives the model persistent awareness of *where it's running* — the OS, hostname,
the user's home/Documents/Downloads paths, and the agent project dir — so it
can reference real paths without guessing. Computed live each session (paths/OS
don't change often, and live means never stale). Custom user facts still live in
the editable `facts` memory alongside this.
"""

from __future__ import annotations

import platform
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any


def _exists(p: Path) -> str | None:
    return str(p) if p.exists() else None


@lru_cache(maxsize=1)
def _osname() -> str:
    """Pretty OS name via lsb_release, cached — this is called on every chat
    turn (build_context) and the OS doesn't change mid-session."""
    fallback = f"{platform.system()} {platform.release()}"
    try:
        import subprocess

        pretty = subprocess.run(
            ["lsb_release", "-ds"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return pretty or fallback
    except Exception:  # noqa: BLE001 - lsb_release missing is fine
        return fallback


def system_facts(config: dict[str, Any]) -> list[str]:
    """Human-readable environment lines for the system prompt."""
    from agent.config import PROJECT_ROOT

    home = Path.home()
    lines: list[str] = []

    lines.append(f"OS: {_osname()} ({platform.machine()}), hostname: {platform.node()}")
    lines.append(f"Today's date: {date.today().isoformat()}")
    lines.append(f"Home directory: {home}")

    for label, sub in (("Documents", "Documents"), ("Downloads", "Downloads"),
                       ("Desktop", "Desktop")):
        path = _exists(home / sub)
        if path:
            lines.append(f"{label} folder: {path}")

    lines.append(f"Agent project directory: {PROJECT_ROOT}")

    return lines
