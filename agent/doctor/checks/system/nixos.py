"""NixOS: things that differ there (binary wheels, the system Ollama service)."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, warn


@check("nixos.version", "system", "NixOS", systems=("nixos",), timeout=6)
def version(ctx: Context):
    rc, out = ctx.p.run(["nixos-version"], timeout=4)
    return info(f"NixOS {out.strip().splitlines()[0][:60] if rc == 0 and out.strip() else '(version unknown)'}")


@check("nixos.nixld", "system", "Running pip-installed programs", systems=("nixos",))
def nix_ld(ctx: Context):
    if ctx.env.get("NIX_LD") or ctx.p.exists("/run/current-system/sw/share/nix-ld"):
        return ok("nix-ld is available, so binary Python wheels can run")
    return warn("nix-ld is not enabled; binary wheels (sqlite-vec, ruff, ...) may fail with 'No such file or directory'",
                "Use the project's nix shell / `uv` from nixpkgs, or enable `programs.nix-ld.enable = true;` in your system configuration.")


@check("nixos.ollama_service", "system", "Ollama as a NixOS service", systems=("nixos",), timeout=8)
def ollama_service(ctx: Context):
    if not ctx.p.which("systemctl"):
        return None
    _rc, out = ctx.p.run(["systemctl", "is-active", "ollama.service"], timeout=4)
    state = (out.strip().splitlines() or ["unknown"])[0]
    if state == "active":
        return ok("services.ollama is active")
    if ctx.ollama.reachable:
        return info("Ollama answers, but not as a system service (started by hand or per user)")
    return info(f"ollama.service is {state}; enable it with `services.ollama.enable = true;` in your configuration")
