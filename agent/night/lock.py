"""One night per folder: `.night/LOCK` holds the pid of the running governor. A lock whose process is gone (a crash, a power cut) is stale and taken over."""
from __future__ import annotations

import os
from pathlib import Path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner(folder: Path) -> int | None:
    """The pid of the night running in `folder`, or None."""
    try:
        pid = int((folder / ".night" / "LOCK").read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _alive(pid) and pid != os.getpid() else None


def acquire(folder: Path) -> bool:
    """True when this process now holds the lock; False when another live night owns it."""
    if owner(folder) is not None:
        return False
    (folder / ".night").mkdir(exist_ok=True)
    (folder / ".night" / "LOCK").write_text(str(os.getpid()))
    return True


def release(folder: Path) -> None:
    try:
        if int((folder / ".night" / "LOCK").read_text().strip()) == os.getpid():
            (folder / ".night" / "LOCK").unlink()
    except (OSError, ValueError):
        pass
