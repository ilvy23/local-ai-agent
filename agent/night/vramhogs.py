"""Who holds graphics memory right now? The speed of a night depends on it: the same 128k run was 15-21 tok/s with a game or browser holding VRAM and 36 tok/s
without (measured), because the model server places fewer layers on the GPU when less is free. Reads the kernel's per-process accounting (amdgpu / i915 write
`drm-memory-vram` or `drm-total-vram0` into /proc/<pid>/fdinfo/*) and, for NVIDIA, `nvidia-smi`. Several descriptors of one DRM client are counted once."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

_VRAM = re.compile(r"^drm-(?:memory-vram|total-vram\d*):\s+(\d+)\s*(KiB|MiB|GiB)?", re.M)
_CLIENT = re.compile(r"^drm-client-id:\s+(\d+)", re.M)
_UNIT = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, None: 1 / 1024}


def _nvidia() -> list[tuple[int, int]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        a, _, b = line.partition(",")
        if a.strip().isdigit() and b.strip().isdigit():
            rows.append((int(a), int(b)))
    return rows


def hogs(proc: Path = Path("/proc"), nvidia: Callable[[], list[tuple[int, int]]] = _nvidia, min_mb: int = 100) -> list[tuple[int, str, int]]:
    """[(pid, name, MB)] of the processes holding at least `min_mb`, biggest first."""
    use: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        seen: set[str] = set()
        total = 0.0
        try:
            files = list((entry / "fdinfo").iterdir())
        except OSError:
            continue
        for f in files:
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            m, cid = _VRAM.search(text), _CLIENT.search(text)
            if not m or (cid and cid.group(1) in seen):
                continue
            if cid:
                seen.add(cid.group(1))
            total += int(m.group(1)) * _UNIT[m.group(2)]
        if total >= 1:
            use[int(entry.name)] = int(total)
    for pid, mb in nvidia():
        use[pid] = use.get(pid, 0) + mb
    out = []
    for pid, mb in use.items():
        if mb >= min_mb:
            try:
                name = (proc / str(pid) / "comm").read_text().strip()
            except OSError:
                name = "?"
            out.append((pid, name, mb))
    return sorted(out, key=lambda r: -r[2])


def summary(rows: list[tuple[int, str, int]], top: int = 4) -> str:
    if not rows:
        return "no other program holds graphics memory"
    return f"{sum(r[2] for r in rows)} MB held by other programs: " + ", ".join(f"{n} {mb} MB" for _, n, mb in rows[:top])
