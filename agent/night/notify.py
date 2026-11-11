"""Tell the user the night is over. Always: a marker file in the project (.night/DONE) and the terminal bell. If a desktop notifier exists: one notification.
Only when the user sets NIGHT_NTFY_URL (their own ntfy topic): one short POST with the headline - the one place data may leave the machine, opt-in."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def headline(done: list[str], partial: list[str], tests_ok: bool) -> str:
    return f"nightage night finished: {len(done)} done, {len(partial)} not finished, tests {'green' if tests_ok else 'RED'}"


def notify(root: Path, text: str, *, run=subprocess.run, post=None, marker: bool = True) -> list[str]:
    """Returns what was used: marker, bell, desktop, ntfy."""
    used = []
    (root / ".night").mkdir(exist_ok=True)
    if marker:                                                 # .night/DONE ends the systemd unit: only written when nothing more is to come
        (root / ".night" / "DONE").write_text(text + "\n")
        used.append("marker")
    print("\a", end="", flush=True)
    used.append("bell")
    if shutil.which("notify-send"):
        try:
            run(["notify-send", "nightage", text], timeout=10)
            used.append("desktop")
        except (OSError, subprocess.SubprocessError):
            pass
    url = os.environ.get("NIGHT_NTFY_URL", "").strip()
    if url.startswith(("http://", "https://")):
        try:
            import httpx

            (post or httpx.post)(url, content=text.encode(), timeout=10)
            used.append("ntfy")
        except Exception:  # noqa: BLE001 - a notification must never break the end of a night
            pass
    return used
