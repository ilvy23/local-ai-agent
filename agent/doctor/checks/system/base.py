"""Every system: what the machine is."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok


@check("platform", "system", "Platform", core=True)
def platform(ctx: Context):
    si = ctx.sysinfo
    ram = (f", {si.ram_total_mb // 1024 if si.ram_total_mb >= 1024 else si.ram_total_mb} {'GB' if si.ram_total_mb >= 1024 else 'MB'} RAM"
           if si.ram_total_mb else ", RAM unknown")
    free = f" ({si.ram_free_mb} MB free)" if si.ram_free_mb else ""
    extra = " in WSL" if "wsl" in ctx.tags else ""
    return ok(f"{si.summary()}{extra}{ram}{free}, {si.cpu_count} CPU(s)")


@check("system.arch", "system", "Architecture")
def arch(ctx: Context):
    pi = ctx.p.platform_info()
    return info(f"{pi.get('system', '?')} {pi.get('release', '')} on {pi.get('machine', '?')}".strip())
