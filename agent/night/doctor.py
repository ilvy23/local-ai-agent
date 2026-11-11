"""`nightage night doctor`: can this machine run a night RIGHT NOW? Each check is a (name, ok, detail) row; nothing is changed.
Checks: git, a coding model that is installed, the model server answers, the test command works, enough disk, no game running, GPUs not hot,
`systemd-inhibit` available (the PC must not sleep), no other night running in this folder."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.night import health as health_mod, lock, vramhogs


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = True                 # False: a warning (the night can still run)


def run(root: Path, *, coding_model: str | None, installed: list[str] | None, urls: list[str], test_facts, probe: Callable[[str], bool] = health_mod.ping,
        game: Callable[[], str | None] | None = None, temps: Callable[[], list[float]] = health_mod.gpu_temps, which: Callable[[str], str | None] = shutil.which,
        free_gb: float | None = None, hogs: list[tuple[int, str, int]] | None = None) -> list[Check]:
    out = [Check("git", bool(which("git")), "found" if which("git") else "git is not installed: no checkpoints are possible")]
    if coding_model:
        have = installed is None or any(coding_model == m or coding_model.split(":")[0] == m.split(":")[0] and ":" not in coding_model for m in installed)
        out.append(Check("coding model", have, coding_model if have else f"{coding_model} is not installed on the model server"))
    for u in urls:
        up = probe(u)
        out.append(Check(f"model server {u}", up, "answers" if up else "does not answer (it will be restarted if it is the Vulkan service)", fatal=False))
    out.append(Check("test command", test_facts.usable, test_facts.text(), fatal=test_facts.state is not None))
    free = free_gb if free_gb is not None else shutil.disk_usage(root).free / 2**30
    out.append(Check("disk", free >= health_mod.MIN_FREE_GB * 2, f"{free:.0f} GB free"))
    g = game() if game else health_mod._game()
    out.append(Check("no game running", g is None, "none" if g is None else f"{g} is running: the night would wait until it ends", fatal=False))
    hot = [t for t in temps() if t >= health_mod.HOT_C - 10]
    out.append(Check("gpu temperature", not hot, "ok" if not hot else f"{max(hot):.0f} C already", fatal=False))
    out.append(Check("keeps the pc awake", bool(which("systemd-inhibit")), "systemd-inhibit found" if which("systemd-inhibit") else "no systemd-inhibit: the PC may suspend at night", fatal=False))
    held = vramhogs.hogs() if hogs is None else hogs
    out.append(Check("graphics memory is free", sum(r[2] for r in held) < 2500, vramhogs.summary(held) + " (a big model gets fewer layers on the GPU and runs slower)", fatal=False))
    pid = lock.owner(root)
    out.append(Check("no other night here", pid is None, "free" if pid is None else f"night pid {pid} is already running in this folder"))
    return out


def text(checks: list[Check]) -> str:
    rows = [f"{'ok  ' if c.ok else ('FAIL' if c.fatal else 'warn')}  {c.name}: {c.detail}" for c in checks]
    bad = [c for c in checks if not c.ok and c.fatal]
    return "\n".join(rows) + ("\n\nnot ready: " + ", ".join(c.name for c in bad) if bad else "\n\nready")
