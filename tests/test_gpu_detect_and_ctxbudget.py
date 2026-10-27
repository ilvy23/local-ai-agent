"""Multi-vendor GPU detection (AMD/Intel/NVIDIA/CPU-only) and the per-model context budget.

No real hardware anywhere here: sysfs is a fake tree under tmp_path, `nvidia-smi` is a fake
subprocess, and ctxbudget is fed fake `HardwareSummary` snapshots directly.
"""

from __future__ import annotations

import subprocess

import pytest

from agent import ctxbudget, gpumode
from agent.ctxbudget import HardwareSummary
from agent.gpumode import Gpu


# --------------------------------------------------------------------------- fake sysfs tree


def _write_amd_or_intel_card(root, card_name, vendor_id, total_mb, used_mb, product_name=None):
    dev = root / card_name / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor_id)
    (dev / "mem_info_vram_total").write_text(str(total_mb * 1024 * 1024))
    (dev / "mem_info_vram_used").write_text(str(used_mb * 1024 * 1024))
    if product_name:
        (dev / "product_name").write_text(product_name)


# --------------------------------------------------------------------------- detect_gpus()


def test_no_gpu_anywhere_gives_the_cpu_only_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: None)  # no nvidia-smi on PATH
    assert gpumode.detect_gpus(drm_root=tmp_path) == []


def test_one_amd_gpu_is_detected_from_fake_sysfs(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: None)
    _write_amd_or_intel_card(tmp_path, "card0", "0x1002", 16000, 1300, "Radeon RX 7800 XT")
    gpus = gpumode.detect_gpus(drm_root=tmp_path)
    assert len(gpus) == 1 and gpus[0].vendor == "amd" and gpus[0].total_mb == 16000
    assert gpus[0].name == "Radeon RX 7800 XT" and gpus[0].vulkan_capable


def test_one_intel_gpu_is_detected_from_fake_sysfs(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: None)
    _write_amd_or_intel_card(tmp_path, "card0", "0x8086", 8000, 500, "Intel Arc A770")
    gpus = gpumode.detect_gpus(drm_root=tmp_path)
    assert len(gpus) == 1 and gpus[0].vendor == "intel" and gpus[0].total_mb == 8000
    assert gpus[0].vulkan_capable


def test_an_intel_igpu_with_no_vram_counters_is_skipped_not_crashed(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: None)
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x8086")
    # no mem_info_vram_total/used - typical for an integrated GPU
    assert gpumode.detect_gpus(drm_root=tmp_path) == []


def test_unknown_vendor_card_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: None)
    _write_amd_or_intel_card(tmp_path, "card0", "0x10de", 8000, 0)  # nvidia's PCI id, no sysfs VRAM path for it
    assert gpumode.detect_gpus(drm_root=tmp_path) == []


def test_one_nvidia_gpu_from_a_fake_nvidia_smi(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="RTX 4060 Ti, 8192, 20\n", stderr="")

    monkeypatch.setattr(gpumode.subprocess, "run", fake_run)
    gpus = gpumode.detect_gpus(drm_root=tmp_path)
    assert len(gpus) == 1 and gpus[0].vendor == "nvidia" and gpus[0].total_mb == 8192
    assert gpus[0].vulkan_capable


def test_two_mixed_gpus_amd_plus_nvidia(tmp_path, monkeypatch):
    _write_amd_or_intel_card(tmp_path, "card0", "0x1002", 16000, 1300, "Radeon RX 7800 XT")
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(gpumode.subprocess, "run",
                         lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="RTX 4060 Ti, 8192, 20\n", stderr=""))
    gpus = gpumode.detect_gpus(drm_root=tmp_path)
    assert {g.vendor for g in gpus} == {"amd", "nvidia"}
    assert gpumode.gpu_mode_available(gpus)


def test_nvidia_smi_erroring_degrades_to_absent_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(gpumode.subprocess, "run", boom)
    assert gpumode.detect_gpus(drm_root=tmp_path) == []


def test_a_broken_sysfs_entry_is_skipped_not_crashed(tmp_path, monkeypatch):
    monkeypatch.setattr(gpumode.shutil, "which", lambda name: None)
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002")
    (dev / "mem_info_vram_total").write_text("not-a-number")   # malformed
    (dev / "mem_info_vram_used").write_text("0")
    assert gpumode.detect_gpus(drm_root=tmp_path) == []


def test_gpu_mode_available_is_false_with_no_gpus():
    assert gpumode.gpu_mode_available([]) is False


def test_gpu_mode_available_is_true_with_any_vulkan_capable_gpu():
    assert gpumode.gpu_mode_available([Gpu("x", "intel", 8000, 0)]) is True


# --------------------------------------------------------------------------- auto_mode() / current_mode()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    gpumode.reset()
    monkeypatch.delenv("AGENT_GPU_MODE", raising=False)
    yield
    gpumode.reset()


def test_auto_mode_is_none_with_no_gpus():
    assert gpumode.auto_mode([]) is None


def test_auto_mode_is_single_with_exactly_one_vulkan_capable_gpu():
    assert gpumode.auto_mode([Gpu("x", "intel", 8000, 0)]) == gpumode.SINGLE


def test_auto_mode_is_dual_with_two_or_more_vulkan_capable_gpus():
    assert gpumode.auto_mode([Gpu("a", "amd", 16000, 0), Gpu("b", "nvidia", 8000, 0)]) == gpumode.DUAL


def test_current_mode_falls_back_to_auto_mode_when_nothing_else_decided(monkeypatch):
    monkeypatch.setattr(gpumode, "detect_gpus", lambda: [Gpu("a", "amd", 16000, 0), Gpu("b", "nvidia", 8000, 0)])
    assert gpumode.current_mode() == gpumode.DUAL


def test_env_still_overrides_auto_mode(monkeypatch):
    monkeypatch.setattr(gpumode, "detect_gpus", lambda: [Gpu("a", "amd", 16000, 0), Gpu("b", "nvidia", 8000, 0)])
    monkeypatch.setenv("AGENT_GPU_MODE", "single")
    assert gpumode.current_mode() == gpumode.SINGLE


# --------------------------------------------------------------------------- ctxbudget


ZERO_GPU = HardwareSummary(gpus=[], total_ram_mb=16000)
ONE_AMD = HardwareSummary(gpus=[Gpu("amd", "amd", 16000, 0)])
ONE_NVIDIA = HardwareSummary(gpus=[Gpu("nv", "nvidia", 8000, 0)])
ONE_INTEL = HardwareSummary(gpus=[Gpu("intel", "intel", 8000, 0)])
TWO_MIXED = HardwareSummary(gpus=[Gpu("amd", "amd", 16000, 0), Gpu("nv", "nvidia", 8000, 0)])


def test_hardware_summary_cpu_only_flag():
    assert ZERO_GPU.cpu_only is True
    assert ONE_AMD.cpu_only is False


@pytest.mark.parametrize("hw", [ZERO_GPU, ONE_AMD, ONE_NVIDIA, ONE_INTEL, TWO_MIXED])
def test_budget_never_exceeds_the_ceiling(hw):
    assert ctxbudget.budget_for("some-model", None, hw) <= ctxbudget.DEFAULT_CEILING


def test_budget_is_at_least_the_floor_even_with_nothing_detected():
    nothing = HardwareSummary(gpus=[], total_ram_mb=0)
    assert ctxbudget.budget_for("m", None, nothing) >= ctxbudget._CTX_FLOOR


def test_more_free_vram_never_gives_a_smaller_budget():
    small = ctxbudget.budget_for("m", None, ONE_NVIDIA)
    big = ctxbudget.budget_for("m", None, TWO_MIXED)
    assert big >= small


def test_requested_value_is_never_exceeded():
    assert ctxbudget.budget_for("m", None, TWO_MIXED, requested=4096) == 4096


def test_explicit_override_wins_even_over_a_tiny_requested_value():
    config = {"models": {"overrides": {"m": {"num_ctx": 200000}}}}
    assert ctxbudget.budget_for("m", None, ZERO_GPU, requested=1024, config=config) == 200000


def test_no_override_present_falls_through_to_the_computed_budget():
    config = {"models": {"overrides": {"other-model": {"num_ctx": 200000}}}}
    assert ctxbudget.budget_for("m", None, TWO_MIXED, requested=4096, config=config) == 4096


def test_custom_ceiling_from_config_is_respected():
    config = {"models": {"max_ctx_ceiling": 12345}}
    assert ctxbudget.budget_for("m", None, TWO_MIXED, config=config) == 12345


def test_env_ceiling_overrides_config_ceiling(monkeypatch):
    monkeypatch.setenv("AGENT_CTX_CEILING", "5000")
    config = {"models": {"max_ctx_ceiling": 12345}}
    assert ctxbudget.budget_for("m", None, TWO_MIXED, config=config) == 5000


def test_legacy_env_alias_still_works(monkeypatch):
    monkeypatch.delenv("AGENT_CTX_CEILING", raising=False)
    monkeypatch.setenv("AGENT_MAX_QWEN_CTX", "7000")
    assert ctxbudget.budget_for("m", None, TWO_MIXED) == 7000


def test_known_model_facts_give_a_tighter_estimate_than_the_unknown_guess():
    facts = {"size_mb": 5000, "kv_elems_per_token": 36 * 8 * 256, "kv_type": "f16"}
    with_facts = ctxbudget.budget_for("small:9b", None, ONE_AMD, facts=facts)
    without_facts = ctxbudget.budget_for("small:9b", None, ONE_AMD)
    assert with_facts != without_facts


def test_explain_reports_the_full_decision():
    config = {"models": {"overrides": {"m": {"num_ctx": 42000}}}}
    info = ctxbudget.explain("m", None, TWO_MIXED, requested=1024, config=config)
    assert info["override"] == 42000 and info["final"] == 42000
    assert info["gpu_count"] == 2 and info["cpu_only"] is False
    assert info["source"] == "vram"


def test_explain_without_override_shows_computed_equals_final_when_nothing_requested():
    info = ctxbudget.explain("m", None, ONE_AMD)
    assert info["override"] is None
    assert info["computed"] == info["final"]
    assert info["cpu_only"] is False


def test_explain_cpu_only_reports_ram_source():
    info = ctxbudget.explain("m", None, ZERO_GPU)
    assert info["cpu_only"] is True and info["source"] == "ram"


def test_read_total_ram_mb_never_raises_even_if_meminfo_is_missing(monkeypatch):
    monkeypatch.setattr(ctxbudget, "Path", ctxbudget.Path)  # sanity: Path still importable

    def boom(*a, **kw):
        raise OSError("no /proc here")

    monkeypatch.setattr(ctxbudget.Path, "read_text", boom, raising=False)
    assert ctxbudget.read_total_ram_mb() == 0
