"""agent.sysinfo against fake /proc, fake environment and fake `which`: no real hardware is read."""
from pathlib import Path

from agent import sysinfo

MEMINFO = "MemTotal:        3984000 kB\nMemFree:          100000 kB\nMemAvailable:    1536000 kB\nSwapTotal: 0 kB\n"
OSREL = 'NAME="NixOS"\nID=nixos\nPRETTY_NAME="NixOS 25.05 (Warbler)"\n# comment\n'


def fake(tmp_path: Path, **kw):
    mem = tmp_path / "meminfo"
    mem.write_text(kw.pop("meminfo", MEMINFO))
    osr = tmp_path / "os-release"
    osr.write_text(kw.pop("osrel", OSREL))
    args = dict(env={}, platform="linux", which=lambda n: None, meminfo_path=mem, os_release_path=osr, exists=lambda p: False,
                cpu_count=4, is_tty=True, width=72, gpus=[], encoding="utf-8")
    args.update(kw)
    return sysinfo.detect(**args)


def test_desktop_linux_with_systemd(tmp_path):
    info = fake(tmp_path, exists=lambda p: p == "/run/systemd/system", which=lambda n: "/bin/" + n if n in ("rg", "git", "systemctl") else None)
    assert info.os_family == sysinfo.LINUX and info.distro == "nixos" and info.distro_name.startswith("NixOS")
    assert info.init_system == "systemd" and info.has_systemctl and info.has_rg and info.has_git and not info.has_uv
    assert (info.ram_total_mb, info.ram_free_mb) == (3984000 // 1024, 1536000 // 1024)
    assert info.term_width == 72 and info.is_tty and info.cpu_count == 4 and not info.is_termux
    assert "nixos" in info.summary() and "systemd" in info.summary()


def test_termux_by_env_and_by_prefix(tmp_path):
    a = fake(tmp_path, env={"TERMUX_VERSION": "0.118"})
    b = fake(tmp_path, env={"PREFIX": "/data/data/com.termux/files/usr"})
    for info in (a, b):
        assert info.is_termux and info.is_android and info.os_family == sysinfo.ANDROID and info.init_system == "none"
        assert "Termux" in info.summary()
    assert not fake(tmp_path, env={"PREFIX": "/usr/local"}).is_termux


def test_other_os_families(tmp_path):
    assert fake(tmp_path, platform="darwin").os_family == sysinfo.MACOS
    win = fake(tmp_path, platform="win32")
    assert win.os_family == sysinfo.WINDOWS and win.init_system == "none"


def test_missing_files_fall_back_never_raise(tmp_path):
    info = sysinfo.detect(env={}, platform="linux", which=lambda n: None, meminfo_path=tmp_path / "nope", os_release_path=tmp_path / "nope2",
                          exists=lambda p: False, gpus=[])
    assert info.ram_total_mb == 0 and info.ram_free_mb == 0 and info.distro == "" and not info.low_ram


def test_total_failure_returns_defaults():
    def boom(_n):
        raise RuntimeError("which exploded")
    info = sysinfo.detect(which=boom, gpus=[])
    assert isinstance(info, sysinfo.SysInfo) and info.ram_total_mb == 0


def test_memavailable_falls_back_to_memfree(tmp_path):
    info = fake(tmp_path, meminfo="MemTotal: 2048000 kB\nMemFree: 512000 kB\n")
    assert info.ram_free_mb == 500


def test_low_ram_threshold(tmp_path):
    assert fake(tmp_path).low_ram                                              # ~3.9 GB
    assert not fake(tmp_path, meminfo="MemTotal: 8388608 kB\nMemFree: 1 kB\n").low_ram


def test_unicode_detection(tmp_path):
    assert not fake(tmp_path, encoding="ascii").unicode_ok
    assert fake(tmp_path, encoding="UTF-8").unicode_ok


def test_gpu_probe_skipped_on_termux_and_injectable(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("agent.gpumode.detect_gpus", lambda *a, **k: called.append(1) or [])
    sysinfo.detect(env={"TERMUX_VERSION": "1"}, platform="linux", which=lambda n: None, meminfo_path=tmp_path / "x", os_release_path=tmp_path / "y")
    assert not called
    sysinfo.detect(env={}, platform="linux", which=lambda n: None, meminfo_path=tmp_path / "x", os_release_path=tmp_path / "y")
    assert called


def test_to_dict_is_json_friendly(tmp_path):
    import json
    json.dumps(fake(tmp_path).to_dict())
