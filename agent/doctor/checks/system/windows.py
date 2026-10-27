"""Windows and WSL."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, warn


@check("windows.version", "system", "Windows", systems=("windows",))
def version(ctx: Context):
    pi = ctx.p.platform_info()
    return info(f"Windows {pi.get('release', '?')} ({pi.get('version', '?')}), {pi.get('machine', '?')}")


@check("windows.terminal", "system", "Windows console", systems=("windows",))
def console(ctx: Context):
    env = ctx.env
    if env.get("WT_SESSION"):
        return ok("Windows Terminal (colours and unicode work)")
    if env.get("TERM_PROGRAM") or env.get("ConEmuANSI") == "ON":
        return ok(f"{env.get('TERM_PROGRAM') or 'ConEmu'}")
    return warn("classic console host: colours and box drawing may be limited", "Use Windows Terminal (from the Microsoft Store) for the best result.")


@check("windows.wsl", "system", "Windows Subsystem for Linux", systems=("wsl",))
def wsl(ctx: Context):
    from agent.doctor.checks.filesystem import config_dir

    data_dir, _ = ctx.p.data_paths(ctx.config)
    if str(data_dir).startswith("/mnt/") or str(config_dir(ctx)).startswith("/mnt/"):
        return warn("your data or config lives on the Windows drive (/mnt/...): the database will be very slow there",
                    "Keep data.db_path inside the Linux filesystem (under ~).")
    gpu_hint = "GPU passthrough: nvidia-smi works inside WSL" if ctx.p.exists("/usr/lib/wsl/lib/nvidia-smi") else "no GPU passthrough library found (/usr/lib/wsl/lib)"
    return ok(f"WSL ({ctx.env.get('WSL_DISTRO_NAME') or 'distro'}); {gpu_hint}")
