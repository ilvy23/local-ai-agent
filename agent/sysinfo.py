"""What kind of machine is this? One cheap, never-raising probe the rest of the code asks instead of guessing.

`detect()` returns a `SysInfo`: OS family (linux / android-termux / macos / windows), distro id, init system, whether
this is Termux, tty + terminal width, RAM, CPU count, GPUs and which helper binaries exist. Every input (environment,
files, `shutil.which`, GPU probe) can be replaced, so tests run it against a fake /proc and a fake environment.

Termux / Android specifics here are UNVERIFIED on a real phone (no Android device was available when this was written):
  * Termux sets TERMUX_VERSION and PREFIX=/data/data/com.termux/files/usr (documented Termux behaviour, widely used by
    scripts for detection).
  * /proc/meminfo is normally readable inside Termux; /proc/stat is blocked for apps from Android 8 on, which is why
    callers use `read_meminfo()` / the helpers below instead of assuming /proc exists everywhere.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LINUX, ANDROID, MACOS, WINDOWS = "linux", "android-termux", "macos", "windows"
TERMUX_PREFIX_MARK = "com.termux"


@dataclass
class SysInfo:
    os_family: str = LINUX
    distro: str = ""                    # /etc/os-release ID ("nixos", "debian", ...); "" when unknown
    distro_name: str = ""               # PRETTY_NAME
    init_system: str = "none"           # "systemd" | "none"
    is_termux: bool = False
    is_tty: bool = False
    term_width: int = 80
    ram_total_mb: int = 0               # 0 = unknown
    ram_free_mb: int = 0                # MemAvailable (falls back to MemFree); 0 = unknown
    cpu_count: int = 1
    gpus: list[Any] = field(default_factory=list)
    has_nvidia_smi: bool = False
    has_rg: bool = False
    has_git: bool = False
    has_uv: bool = False
    has_systemctl: bool = False
    unicode_ok: bool = True

    @property
    def is_android(self) -> bool:
        return self.os_family == ANDROID

    @property
    def low_ram(self) -> bool:
        """Under 6 GB (a known value only): the threshold at which `doctor` suggests the phone profile."""
        return 0 < self.ram_total_mb < LOW_RAM_MB

    def summary(self) -> str:
        where = {LINUX: "Linux", ANDROID: "Android (Termux)", MACOS: "macOS", WINDOWS: "Windows"}.get(self.os_family, self.os_family)
        if self.distro and self.os_family == LINUX:
            where += f" ({self.distro})"
        init = f", init {self.init_system}" if self.os_family == LINUX else ""
        return f"{where}{init}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gpus"] = [getattr(g, "name", str(g)) for g in self.gpus]
        return d


LOW_RAM_MB = 6 * 1024


def parse_os_release(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("\"'")
    return out


def read_meminfo(path: str | Path = "/proc/meminfo") -> dict[str, int]:
    """/proc/meminfo as {key: kB}; {} when unreadable (never raises)."""
    info: dict[str, int] = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                info[key.strip()] = int(parts[0])
    except (OSError, ValueError):
        return {}
    return info


def ram_mb(meminfo: Mapping[str, int] | None = None) -> tuple[int, int]:
    """(total_mb, available_mb); (0, 0) when unknown."""
    mi = read_meminfo() if meminfo is None else meminfo
    total = int(mi.get("MemTotal", 0)) // 1024
    free = int(mi.get("MemAvailable", mi.get("MemFree", 0))) // 1024
    return total, free


def is_termux(env: Mapping[str, str] | None = None) -> bool:
    e = os.environ if env is None else env
    return bool(e.get("TERMUX_VERSION")) or TERMUX_PREFIX_MARK in (e.get("PREFIX") or "")


def _os_family(env: Mapping[str, str], platform: str) -> str:
    if is_termux(env):
        return ANDROID
    if platform.startswith("win"):
        return WINDOWS
    if platform == "darwin":
        return MACOS
    return LINUX


def _init_system(platform: str, exists: Callable[[str], bool]) -> str:
    # /run/systemd/system exists only while systemd is PID 1 (the documented sd_booted() check).
    return "systemd" if platform.startswith("linux") and exists("/run/systemd/system") else "none"


def detect(*, env: Mapping[str, str] | None = None, platform: str | None = None, which: Callable[[str], str | None] | None = None,
           meminfo_path: str | Path = "/proc/meminfo", os_release_path: str | Path = "/etc/os-release",
           exists: Callable[[str], bool] | None = None, cpu_count: int | None = None, is_tty: bool | None = None,
           width: int | None = None, gpus: list[Any] | None = None, encoding: str | None = None) -> SysInfo:
    """Probe this machine. Never raises: every field has a safe default."""
    try:
        e = os.environ if env is None else env
        plat = sys.platform if platform is None else platform
        w = shutil.which if which is None else which
        ex = os.path.exists if exists is None else exists
        info = SysInfo(os_family=_os_family(e, plat))
        info.is_termux = info.os_family == ANDROID
        info.init_system = _init_system(plat, ex)
        try:
            osr = parse_os_release(Path(os_release_path).read_text())
            info.distro, info.distro_name = osr.get("ID", ""), osr.get("PRETTY_NAME", "")
        except OSError:
            pass
        info.ram_total_mb, info.ram_free_mb = ram_mb(read_meminfo(meminfo_path))
        if not info.ram_total_mb and info.os_family in (WINDOWS, MACOS):      # no /proc/meminfo there: ask the OS
            from agent import osport

            info.ram_total_mb, info.ram_free_mb = osport.native_ram_mb()
        try:
            info.cpu_count = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
        except (TypeError, ValueError):
            info.cpu_count = 1
        try:
            info.is_tty = bool(sys.stdout.isatty()) if is_tty is None else bool(is_tty)
        except (AttributeError, ValueError):
            info.is_tty = False
        try:
            info.term_width = int(width) if width else shutil.get_terminal_size((80, 24)).columns
        except (OSError, ValueError):
            info.term_width = 80
        enc = (encoding if encoding is not None else (getattr(sys.stdout, "encoding", None) or "")).lower()
        lang = (e.get("LC_ALL") or e.get("LC_CTYPE") or e.get("LANG") or "").lower()
        info.unicode_ok = "utf" in enc or ("utf" in lang and not enc)
        info.has_nvidia_smi = bool(w("nvidia-smi"))
        info.has_rg, info.has_git, info.has_uv = bool(w("rg")), bool(w("git")), bool(w("uv"))
        info.has_systemctl = bool(w("systemctl")) and info.init_system == "systemd"
        if gpus is not None:
            info.gpus = list(gpus)
        elif info.os_family in (LINUX,):          # sysfs/nvidia-smi probing only makes sense on a desktop Linux
            try:
                from agent import gpumode

                info.gpus = list(gpumode.detect_gpus())
            except Exception:  # noqa: BLE001 - detection must never break the caller
                info.gpus = []
        return info
    except Exception:  # noqa: BLE001 - last resort: a SysInfo with defaults
        return SysInfo()


_cached: SysInfo | None = None


def current() -> SysInfo:
    """Process-wide cached `detect()` (the GPU probe may spawn nvidia-smi once)."""
    global _cached
    if _cached is None:
        _cached = detect()
    return _cached


def reset_cache() -> None:
    global _cached
    _cached = None
