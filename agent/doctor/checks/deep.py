"""Deep checks: they USE the model (a few tokens each), so they only run with `nightage doctor --deep`."""

from __future__ import annotations

from agent.doctor.core import Context, Result, check, fail, info, ok, warn
from agent.doctor.probes import model_installed

TIERS = (  # (upper bound of the memory budget in GB, model, what it is good for)
    (3, "qwen3.5:0.8b", "tiny: quick answers on phones and old laptops"),
    (8, "qwen3.5:4b", "small: everyday chat and light coding"),
    (20, "qwen3.5:9b", "standard: the project's default; good tool use"),
    (10_000, "qwen3.5:27b", "large: best quality, needs a big GPU (or two)"),
)


def memory_budget_gb(ctx: Context) -> tuple[float, str]:
    """(GB a model can use, where from): free VRAM of the biggest GPU pool, else half the free RAM."""
    gpus = ctx.gpus
    if gpus:
        total = sum(g.total_mb for g in gpus)
        return total / 1024 * 0.9, f"{total // 1024} GB of GPU memory"
    si = ctx.sysinfo
    if si.ram_total_mb:
        return si.ram_total_mb / 1024 * 0.5, f"{si.ram_total_mb // 1024} GB of RAM (CPU only, so half of it)"
    return 4.0, "unknown memory (assuming 4 GB)"


@check("deep.generate", "deep", "Speed test (generation)", deep=True, needs_ollama=True, timeout=120)
def generate(ctx: Context):
    name = ctx.chat_model()
    if not name or not model_installed(name, ctx.ollama.models):
        return Result("skip", "no installed chat model to test")
    r = ctx.p.ollama_generate(name, "Count from one to ten in words.", timeout=110)
    if r.get("error") or "ttft_ms" not in r:
        return fail(f"{name}: generation failed ({r.get('error') or 'no output'})", "Try `ollama run " + name + "` by hand to see the error.")
    tps = r.get("tps")
    text = f"{name}: first token after {r['ttft_ms'] / 1000:.1f} s" + (f", {tps:.1f} tokens/s" if tps else "")
    if r.get("load_ms", 0) > 3000:
        text += f" (model load took {r['load_ms'] / 1000:.1f} s; later turns skip it)"
    if tps and tps < 3:
        return warn(text, "Very slow: use a smaller model or a GPU. See the recommendation below.")
    return ok(text)


@check("deep.embed", "deep", "Embedding call", deep=True, needs_ollama=True, timeout=90)
def embed(ctx: Context):
    name = str(ctx.section("models").get("embed") or "")
    if not name or not model_installed(name, ctx.ollama.models):
        return Result("skip", "no installed embedding model to test")
    r = ctx.p.ollama_embed(name)
    if r.get("error"):
        return fail(f"{name}: {r['error']}", f"Run `ollama pull {name}` and try again.")
    return ok(f"{name}: {r['dim']} dimensions in {r['ms']:.0f} ms")


@check("deep.toolcall", "deep", "Tool-call test", deep=True, needs_ollama=True, timeout=120)
def toolcall(ctx: Context):
    name = ctx.chat_model()
    if not name or not model_installed(name, ctx.ollama.models):
        return Result("skip", "no installed chat model to test")
    r = ctx.p.ollama_tool_call(name)
    if r.get("error"):
        return fail(f"{name}: {r['error']}")
    if r.get("called") and r.get("args_ok"):
        return ok(f"{name} returned a valid tool call ({r['name']}) in {r['ms'] / 1000:.1f} s")
    if r.get("called"):
        return warn(f"{name} called a tool but with malformed arguments", "Small models sometimes do; nightage repairs simple cases, but a larger model is safer.")
    return warn(f"{name} answered in prose instead of calling the tool" if r.get("prose") else f"{name} returned nothing",
                "The agent needs tool calls. Use a model whose `ollama show` lists \"tools\" (or rely on nightage's text-parsing fallback for small models).")


@check("deep.recommend", "deep", "Recommended model for this machine", deep=True)
def recommend(ctx: Context):
    budget, source = memory_budget_gb(ctx)
    pick = next(t for t in TIERS if budget < t[0])
    rows = []
    for ub, model, why in TIERS:
        mark = ">" if model == pick[1] else " "
        lo = {3: "< 3 GB", 8: "3-8 GB", 20: "8-20 GB"}.get(ub, "20 GB +")
        rows.append(f"{mark} {lo:<8} {model:<14} {why}")
        rows.append(f"    ollama pull {model}")
    r = info(f"{pick[1]} fits your {source}", f"ollama pull {pick[1]}")
    r.lines = rows + ["Model names and tiers are the project's defaults; sizes are approximate."]
    try:
        from agent.doctor import fixes

        if not (ctx.ollama.reachable and model_installed(pick[1], ctx.ollama.models)):
            r.command, r.fix = fixes.pull_model(pick[1])
    except Exception:  # noqa: BLE001
        pass
    return r
