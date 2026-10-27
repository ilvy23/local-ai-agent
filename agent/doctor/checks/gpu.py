"""GPU: what is there (any vendor), driver state, and the inference backend that goes with it. CPU-only gets advice."""

from __future__ import annotations

import glob
import re

from agent.doctor.core import Context, check, fail, info, ok, skip, warn


def _vendors(ctx: Context) -> set[str]:
    return {getattr(g, "vendor", "") for g in ctx.gpus}


@check("gpu", "gpu", "GPU", core=True, timeout=10)
def gpu(ctx: Context):
    gpus = ctx.gpus
    mode = ctx.p.gpu_mode(gpus)
    if gpus:
        desc = ", ".join(f"{g.name} {g.total_mb // 1024} GB" for g in gpus)
        return ok(f"{desc}; mode: {mode or 'default'}")
    si = ctx.sysinfo
    if si.os_family == "macos":
        return info("no discrete GPU found; Apple silicon uses its built-in GPU through Metal (see the next row)")
    ram = f" With {si.ram_total_mb // 1024} GB RAM, " + _cpu_advice(si.ram_total_mb) if si.ram_total_mb else ""
    return warn("none detected; models run on the CPU",
                "Fine for small models. For speed use a GPU with enough VRAM, or a smaller model." + ram)


def _cpu_advice(ram_mb: int) -> str:
    from agent import profiles

    return f"`{profiles.model_tier(ram_mb)}` is a good size."


@check("gpu.nvidia", "gpu", "NVIDIA driver", systems=("linux", "windows", "wsl"), timeout=10)
def nvidia(ctx: Context):
    smi = ctx.p.which("nvidia-smi")
    has = "nvidia" in _vendors(ctx)
    if not smi:
        return fail("an NVIDIA GPU is present but `nvidia-smi` is missing (no driver?)",
                    "Install the proprietary NVIDIA driver for your distribution.") if has else None
    rc, out = ctx.p.run([smi, "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"], timeout=6)
    if rc != 0:
        return warn(f"nvidia-smi failed: {(out.strip().splitlines() or ['no output'])[0][:100]}",
                    "The driver may not match the kernel module; reboot after a driver or kernel update.")
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append(f"{parts[0]}: driver {parts[1]}, {parts[3]}/{parts[2]} MiB used")
    cuda = ""
    rc2, hdr = ctx.p.run([smi], timeout=6)
    m = re.search(r"CUDA Version:\s*([\d.]+)", hdr) if rc2 == 0 else None
    if m:
        cuda = f"; CUDA {m.group(1)}"
    return ok("; ".join(rows) + cuda if rows else "nvidia-smi works, no GPU listed")


@check("gpu.amd", "gpu", "AMD driver and ROCm", systems=("linux",))
def amd(ctx: Context):
    if "amd" not in _vendors(ctx) and not ctx.p.exists("/sys/module/amdgpu"):
        return None
    bits = ["amdgpu driver loaded" if ctx.p.exists("/sys/module/amdgpu") else "amdgpu module not loaded"]
    rocm = ctx.p.which("rocminfo") or ctx.p.exists("/opt/rocm")
    bits.append("ROCm found" if rocm else "ROCm not found (Vulkan or the CPU is used instead)")
    kfd = ctx.p.exists("/dev/kfd")
    if rocm and kfd and not ctx.p.can_access("/dev/kfd", 6):
        return warn("; ".join(bits) + "; no access to /dev/kfd", "Add your user to the `render` and `video` groups, then log in again.")
    if not ctx.p.exists("/sys/module/amdgpu"):
        return warn("; ".join(bits), "An AMD card is present but the amdgpu driver is not loaded.")
    return ok("; ".join(bits))


@check("gpu.intel", "gpu", "Intel graphics", systems=("linux",))
def intel(ctx: Context):
    if "intel" not in _vendors(ctx) and not glob.glob("/sys/module/i915") and not ctx.p.exists("/sys/module/xe"):
        return None
    drv = "xe" if ctx.p.exists("/sys/module/xe") else "i915" if ctx.p.exists("/sys/module/i915") else "unknown driver"
    return info(f"Intel graphics ({drv}); Ollama can use it only through its (experimental) Vulkan backend; otherwise models run on the CPU")


@check("gpu.vulkan", "gpu", "Vulkan", systems=("linux", "windows", "wsl"), timeout=10)
def vulkan(ctx: Context):
    present = _vendors(ctx) & {"amd", "intel", "nvidia"}
    tool = ctx.p.which("vulkaninfo")
    if not tool:
        return info("`vulkaninfo` is not installed (optional; it lists the Vulkan devices)") if present else None
    rc, out = ctx.p.run([tool, "--summary"], timeout=8)
    if rc != 0:
        return warn("vulkaninfo failed: " + (out.strip().splitlines() or ["no output"])[0][:100],
                    "No working Vulkan driver (ICD) found; install your GPU vendor's Vulkan package (mesa-vulkan-drivers / vulkan-radeon / nvidia).")
    devs = [m.group(1).strip() for m in re.finditer(r"deviceName\s*=\s*(.+)", out)]
    cpu_only = devs and all("llvmpipe" in d.lower() or "lavapipe" in d.lower() for d in devs)
    if cpu_only:
        return warn("only a software renderer (" + ", ".join(devs) + ")", "Install the Vulkan driver for your GPU; software Vulkan is very slow.")
    return ok(f"{len(devs)} device(s): " + ", ".join(devs[:4]) if devs else "vulkaninfo works")


@check("gpu.metal", "gpu", "Apple GPU (Metal)", systems=("macos",), timeout=10)
def metal(ctx: Context):
    rc, out = ctx.p.run(["system_profiler", "SPDisplaysDataType"], timeout=8)
    if rc != 0:
        return skip("system_profiler is not available")
    chip = re.search(r"Chipset Model:\s*(.+)", out)
    metal = re.search(r"Metal(?: Support| Family)?:\s*(.+)", out)
    si = ctx.sysinfo
    unified = f"; unified memory {si.ram_total_mb // 1024} GB shared with the GPU" if si.ram_total_mb else ""
    return ok(f"{chip.group(1).strip() if chip else 'GPU'}; Metal: {metal.group(1).strip() if metal else 'supported'}{unified}")
