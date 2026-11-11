"""What the night ran on, for the top of the report: versions, models, the context window, GPUs, free disk. Everything is read, nothing is changed;
every part is optional (a missing tool just leaves its line out)."""
from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect(root: Path, config: dict | None = None, overlay: dict | None = None) -> list[str]:
    cfg = config or {}
    lines = [f"host: {platform.node()} / {platform.system()} {platform.release()} / python {platform.python_version()}"]
    lines.append(f"git: {_run(['git', '--version'])}" if shutil.which("git") else "git: missing")
    models = cfg.get("models") or {}
    code = (cfg.get("code") or {}).get("model") or models.get("coding") or models.get("chat")
    if code:
        lines.append(f"coding model: {code}")
    over = (overlay or {}).get("coding") or {}
    cap = over.get("stage_prompt_cap") or (cfg.get("coding") or {}).get("stage_prompt_cap") or "default (28000)"
    lines.append(f"stage prompt cap: {cap}")
    if shutil.which("nvidia-smi"):
        gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if gpu:
            lines.append("nvidia: " + gpu.replace("\n", "; "))
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]/device/mem_info_vram_total")):
        try:
            lines.append(f"{card.parent.parent.name} vram: {int(card.read_text()) // 2**20} MB")
        except (OSError, ValueError):
            pass
    free = shutil.disk_usage(root).free / 2**30
    lines.append(f"disk free: {free:.0f} GB")
    return lines
