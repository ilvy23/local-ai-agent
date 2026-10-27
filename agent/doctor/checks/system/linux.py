"""Linux: distribution and kernel, the user systemd units nightage sets up, memory limits from cgroups."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, skip, warn


@check("linux.distro", "system", "Linux distribution", systems=("linux",))
def distro(ctx: Context):
    si, pi = ctx.sysinfo, ctx.p.platform_info()
    return info(f"{si.distro_name or si.distro or 'unknown distribution'}, kernel {pi.get('release', '?')}, init {si.init_system}")


@check("linux.units", "system", "systemd user services", systems=("systemd",), timeout=10)
def units(ctx: Context):
    if not ctx.p.which("systemctl"):
        return skip("systemctl is not available")
    names = ("ollama-vk", "ollama", "nightage-worker")
    rows, active = [], []
    for n in names:
        rc, out = ctx.p.run(["systemctl", "--user", "is-active", f"{n}.service"], timeout=4)
        state = (out.strip().splitlines() or ["unknown"])[0]
        if state in ("active", "activating", "failed", "inactive", "deactivating") and rc != -1:
            rows.append(f"{n}: {state}")
            if state == "active":
                active.append(n)
            if state == "failed":
                return warn("; ".join(rows), f"Look at why: `journalctl --user -u {n} -n 30` then `systemctl --user restart {n}`.")
    if not rows:
        return info("no nightage-related user services installed")
    return ok("; ".join(rows))


@check("linux.cgroup", "system", "Memory limit (cgroup)", systems=("linux",))
def cgroup(ctx: Context):
    raw = (ctx.p.read_text("/sys/fs/cgroup/memory.max", 100) or "").strip()
    if not raw:
        return None
    if raw == "max":
        return ok("no memory limit on this process group")
    try:
        lim_mb = int(raw) // (1024 * 1024)
    except ValueError:
        return None
    total = ctx.sysinfo.ram_total_mb
    if total and lim_mb < total * 0.9:
        return warn(f"this process is limited to {lim_mb // 1024} GB of the machine's {total // 1024} GB",
                    "A container or service limit: models that need more than the limit are killed. Raise the limit (docker --memory / MemoryMax=).")
    return ok(f"limit {lim_mb // 1024} GB")
