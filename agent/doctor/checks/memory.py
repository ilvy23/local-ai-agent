"""Memory: RAM, swap, pressure and the headroom a model leaves."""

from __future__ import annotations

from agent.doctor.core import Context, check, fail, info, ok, skip, warn
from agent.doctor.probes import model_installed


def _gb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB"


@check("memory.ram", "memory", "RAM")
def ram(ctx: Context):
    si = ctx.sysinfo
    if not si.ram_total_mb:
        return info("RAM size could not be read on this system")
    pct = si.ram_free_mb * 100 // si.ram_total_mb if si.ram_free_mb else 0
    text = f"{_gb(si.ram_total_mb)} total, {_gb(si.ram_free_mb)} available ({pct}%)"
    if si.ram_free_mb and (si.ram_free_mb < 1024 or pct < 8):
        return warn(text, "Very little memory is free: close programs before loading a model.")
    return ok(text)


def _meminfo(ctx: Context) -> dict[str, int]:
    txt = ctx.p.read_text("/proc/meminfo", 20000) or ""
    out = {}
    for line in txt.splitlines():
        k, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[k.strip()] = int(parts[0])
    return out


@check("memory.swap", "memory", "Swap", systems=("linux", "termux", "wsl"))
def swap(ctx: Context):
    mi = _meminfo(ctx)
    total, free = mi.get("SwapTotal", 0) // 1024, mi.get("SwapFree", 0) // 1024
    if not mi:
        return skip("swap information is not readable here")
    if not total:
        ram_mb = ctx.sysinfo.ram_total_mb
        if 0 < ram_mb < 16 * 1024:
            return info("no swap; with this little RAM a large model that does not fit will be killed instead of slowing down",
                        "Add a swap file or zram if you want a safety net.")
        return info("no swap configured")
    used = total - free
    if used * 100 // total > 50 and used > 1024:
        return warn(f"{_gb(used)} of {_gb(total)} swap in use", "The machine is already swapping: models will be slow. Free memory or use a smaller model.")
    return ok(f"{_gb(used)} of {_gb(total)} in use")


@check("memory.pressure", "memory", "Memory pressure", systems=("linux",))
def pressure(ctx: Context):
    txt = ctx.p.read_text("/proc/pressure/memory", 2000)
    if not txt:
        return skip("the kernel does not expose pressure stall information (/proc/pressure/memory)")
    import re

    m = re.search(r"some avg10=([\d.]+)", txt)
    if not m:
        return skip("unreadable pressure data")
    v = float(m.group(1))
    if v >= 10:
        return warn(f"{v:.1f}% of the last 10 s spent waiting for memory", "The system is short of memory right now; close programs first.")
    return ok(f"{v:.1f}% stalled over the last 10 s")


@check("memory.headroom", "memory", "RAM left after the chat model", needs_ollama=True)
def headroom(ctx: Context):
    from agent.doctor.checks.models import size_of
    from agent.doctor.checks.ollama import is_remote

    oll, si = ctx.ollama, ctx.sysinfo
    name = ctx.chat_model()
    if oll.url and is_remote(oll.url):
        return skip("the chat model runs on the remote server")
    if not name or not model_installed(name, oll.models) or not si.ram_total_mb:
        return skip("needs an installed chat model and a known RAM size")
    size = size_of(oll, name)
    if not size:
        return skip("model size unknown")
    gpu_mb = sum(g.total_mb for g in ctx.gpus)
    need = size * 1.15 / 1024 ** 2
    in_ram = max(0.0, need - gpu_mb) if gpu_mb else need          # whatever the GPU cannot hold lives in RAM
    left = si.ram_free_mb - in_ram
    if in_ram == 0:
        return ok(f"{name} lives in GPU memory; {_gb(si.ram_free_mb)} RAM stays free for everything else")
    if left < 512:
        return fail(f"{name} needs about {_gb(in_ram)} of RAM but only {_gb(si.ram_free_mb)} is free",
                    "Use a smaller chat model (models.chat) or free memory.") if left < 0 else warn(
            f"{name} leaves only {_gb(left)} of RAM free", "Tight: expect swapping when other programs run.")
    return ok(f"{_gb(left)} of RAM left after loading {name} (about {_gb(in_ram)})")


@check("memory.roundtrip", "memory", "Saving and recalling a fact", timeout=40)
def roundtrip(ctx: Context):
    """Writes one fact to a throwaway database and recalls it with the configured memory model when it is reachable
    (by meaning), else by keyword. Says exactly why when the meaning path is off. Reads and writes none of your data."""
    from agent.memory.check import roundtrip as run

    model = (ctx.section("models").get("embed") or "").strip()
    oll = ctx.ollama
    if not model:
        res = run(ctx.config, None, unavailable="not configured (models.embed)")
    elif not oll.reachable:
        res = run(ctx.config, None, unavailable="not reachable because Ollama is not running")
    elif not model_installed(model, oll.models):
        res = run(ctx.config, None, unavailable="not installed (`ollama pull %s`, or choose another with `nightage embed list`)" % model)
    else:
        res = run(ctx.config, lambda texts: ctx.p.ollama_embed_vectors(model, texts))
    r = ok(res.detail) if res.status == "ok" else fail(res.detail) if res.status == "fail" else warn(res.detail)
    if res.status == "warn":
        r.hint = "Memory still works through keywords. `nightage embed list` shows memory models; `nightage memory check` repeats this test."
    return r
