"""`nightage night tune`: the two settings of the Vulkan Ollama service that the measurements said matter for a long run at 128k, as a SEPARATE systemd drop-in
(`ollama-vk.service.d/zz-night.conf`, sorted last so it wins; your own tune.conf is never touched, `--revert` deletes only this file).

Measured on this machine (two GPUs, 16 GB + 8 GB, qwen3.8:27b, 128k context, single runs, so read it as a tendency):
  LLAMA_ARG_FIT_TARGET  256 -> 64 : the server keeps less headroom per GPU and places more layers on the GPUs: 94.8% -> 100% on the GPU, 32.9 -> 35.8 tok/s.
  CPUAffinity (one thread per P-core): only helps when part of the model already spilled onto the CPU (15-21 -> 26-28 tok/s then); harmless otherwise.
The risk of a lower target is an out-of-memory error when something else grabs VRAM later: the model server then dies and the night's health gate restarts it.
The real lever is free VRAM (a game or browser holding 1-3 GB cost more than any setting): see `nightage night doctor`."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

UNIT = "ollama-vk"
DROPIN = "zz-night.conf"


def dropin_path(config_home: Path | None = None) -> Path:
    return (config_home or Path.home() / ".config") / "systemd" / "user" / f"{UNIT}.service.d" / DROPIN


def p_core_threads(cpu_root: Path = Path("/sys/devices/system/cpu")) -> list[int]:
    """One thread (the first sibling) of every performance core of a hybrid CPU, found by the highest max frequency; [] when the CPU is not hybrid or unknown."""
    freqs: dict[int, int] = {}
    first: dict[int, int] = {}
    for d in cpu_root.glob("cpu[0-9]*"):
        try:
            n = int(d.name[3:])
            freqs[n] = int((d / "cpufreq" / "cpuinfo_max_freq").read_text())
            sib = (d / "topology" / "thread_siblings_list").read_text().strip()
        except (OSError, ValueError):
            continue
        group = min(int(x) for x in sib.replace("-", ",").split(",") if x.isdigit()) if sib else n
        first[group] = min(first.get(group, n), n)
    if not freqs or len(set(freqs.values())) < 2:
        return []
    top = max(freqs.values())
    return sorted({first[min(int(x) for x in (cpu_root / f"cpu{n}" / "topology" / "thread_siblings_list").read_text().replace("-", ",").split(",") if x.strip().isdigit())]
                   for n, f in freqs.items() if f == top})


@dataclass
class Plan:
    fit_target: int | None
    cpus: list[int]

    def text(self) -> str:
        lines = ["# written by `nightage night tune`; remove with `nightage night tune --revert`"]
        if self.fit_target is not None:
            lines += ["[Service]", f"Environment=LLAMA_ARG_FIT_TARGET={self.fit_target}"]
        if self.cpus:
            lines += ["[Service]", "CPUAffinity=" + " ".join(map(str, self.cpus))]
        return "\n".join(lines) + "\n"


def current(config_home: Path | None = None) -> str | None:
    p = dropin_path(config_home)
    return p.read_text() if p.exists() else None


def apply(plan: Plan, config_home: Path | None = None, run: Callable[..., object] = subprocess.run, restart: bool = False) -> Path:
    """Write the drop-in, reload systemd's user manager, and restart the service only when asked (a restart unloads the model)."""
    p = dropin_path(config_home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plan.text())
    run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=30)
    if restart:
        run(["systemctl", "--user", "restart", UNIT], capture_output=True, timeout=60)
    return p


def revert(config_home: Path | None = None, run: Callable[..., object] = subprocess.run, restart: bool = False) -> bool:
    p = dropin_path(config_home)
    if not p.exists():
        return False
    p.unlink()
    run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=30)
    if restart:
        run(["systemctl", "--user", "restart", UNIT], capture_output=True, timeout=60)
    return True
