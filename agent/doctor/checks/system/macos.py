"""macOS."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, warn


@check("macos.version", "system", "macOS", systems=("macos",), timeout=6)
def version(ctx: Context):
    rc, out = ctx.p.run(["sw_vers", "-productVersion"], timeout=4)
    return info(f"macOS {out.strip()}" if rc == 0 and out.strip() else "macOS (version unknown)")


@check("macos.arch", "system", "Apple silicon or Intel", systems=("macos",), timeout=6)
def arch(ctx: Context):
    machine = ctx.p.platform_info().get("machine", "")
    rc, out = ctx.p.run(["sysctl", "-n", "sysctl.proc_translated"], timeout=3)
    if machine == "arm64":
        return ok("Apple silicon (arm64): models use the GPU through Metal")
    if rc == 0 and out.strip() == "1":
        return warn("this Python runs under Rosetta (translated x86_64)", "Install an arm64 Python; translated programs are much slower.")
    return info(f"{machine or 'unknown'} (Intel Mac)")


@check("macos.ollama_app", "system", "Ollama app", systems=("macos",))
def app(ctx: Context):
    if ctx.p.exists("/Applications/Ollama.app"):
        return ok("Ollama.app is installed")
    return info("Ollama.app not in /Applications (a Homebrew install is fine)")
