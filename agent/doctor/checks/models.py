"""Models: configured vs installed, tool-call capability, and whether they fit in memory."""

from __future__ import annotations

from agent import brand
from agent.doctor import fixes
from agent.doctor.core import Context, check, info, ok, skip, warn
from agent.doctor.probes import model_installed

GB = 1024 ** 3
_ROLES = (("chat", "Chat model"), ("coding", "Coding model"), ("embed", "Embedding model"))


def _pull_fix(name: str) -> dict:
    try:
        cmd, fn = fixes.pull_model(name)
        return {"command": cmd, "fix": fn}
    except fixes.FixRefused:
        return {}


@check("models", "models", "Configured models", core=True, timeout=10)
def configured(ctx: Context):
    from agent.doctor.core import Result

    m = ctx.section("models")
    oll = ctx.ollama
    out: list[Result] = []
    for key, title in _ROLES:
        name = m.get(key) or ""
        cid = f"model_{key}"
        if not name:
            out.append(Result("warn", "not configured", f"Set models.{key} in config.yaml (`{brand.CMD} settings`).", id=cid, title=title))
        elif not oll.reachable:
            out.append(Result("skip", f"{name} (cannot verify: Ollama is not running)", "Start Ollama, then run the doctor again.", id=cid, title=title))
        elif model_installed(name, oll.models):
            detail = name
            if key == "embed":
                from agent import embedmodels

                detail = f"{name} ({embedmodels.describe(name).label})"
            out.append(Result("ok", detail, id=cid, title=title))
        else:
            sev = "warn" if key == "coding" else "fail"
            out.append(Result(sev, f"{name} is not installed", f"Run `ollama pull {name}`.", id=cid, title=title, **_pull_fix(name)))
    return out


def size_of(oll, name: str) -> int:
    return int(oll.models.get(name) or oll.models.get(f"{name}:latest") or 0)


@check("models.installed", "models", "Installed models", needs_ollama=True)
def installed(ctx: Context):
    ms = ctx.ollama.models
    if not ms:
        return warn("no models installed", "Install one, e.g. `ollama pull qwen3.5:4b` (see `nightage doctor --deep` for a recommendation).")
    total = sum(ms.values())
    configured_names = {str(v) for v in ctx.section("models").values() if isinstance(v, str)}
    unused = [n for n in ms if not any(model_installed(c, {n}) for c in configured_names)]
    detail = f"{len(ms)} model(s), {total / GB:.1f} GB on disk"
    if unused:
        detail += f"; {len(unused)} not used by the config"
    r = info(detail)
    if ctx.verbose:
        r.lines = [f"{n}  {s / GB:.1f} GB" + ("  (unused)" if n in unused else "") for n, s in sorted(ms.items())]
    return r


def fit_verdict(size: int, gpus: list, ram_free_mb: int, ram_total_mb: int) -> tuple[str, str]:
    """(status, sentence) for a model of `size` bytes. Weights need roughly size + 15% for buffers and context."""
    need_mb = size * 1.15 / (1024 ** 2)
    vram_total = sum(g.total_mb for g in gpus)
    vram_free = sum(getattr(g, "free_mb", g.total_mb) for g in gpus)
    if gpus and need_mb <= vram_free:
        return "ok", f"fits in GPU memory ({need_mb / 1024:.1f} of {vram_free / 1024:.1f} GB free VRAM)"
    if gpus and need_mb <= vram_total:
        return "warn", f"fits the GPU ({vram_total / 1024:.1f} GB) only when it is empty; {vram_free / 1024:.1f} GB is free now"
    if ram_free_mb and need_mb <= ram_free_mb * 0.9:
        where = f"too big for the GPU ({vram_total / 1024:.1f} GB), so part runs on the CPU and is slower" if gpus else "runs on the CPU"
        return ("warn" if gpus else "ok"), f"{where}; needs about {need_mb / 1024:.1f} GB of {ram_free_mb / 1024:.1f} GB free RAM"
    if ram_total_mb and need_mb <= ram_total_mb * 0.9:
        return "warn", f"needs about {need_mb / 1024:.1f} GB but only {ram_free_mb / 1024:.1f} GB RAM is free: close other programs or expect swapping"
    if not ram_total_mb and not gpus:
        return "info", f"needs about {need_mb / 1024:.1f} GB; memory size unknown"
    return "fail", f"needs about {need_mb / 1024:.1f} GB, more than this machine has ({max(ram_total_mb, vram_total) / 1024:.1f} GB)"


def _embed_advice(name: str, st: str, text: str, hint: str, si) -> tuple[str, str, str]:
    """The memory model runs on the CPU: say what it is for, warn when it is big next to the free RAM, point at the
    recommended one (agent/embedmodels.py; `nightage embed list` shows all)."""
    import os

    from agent import embedmodels

    cur = embedmodels.describe(name)
    text = f"{text}; {cur.label}"
    big = embedmodels.warn_if_large(cur, si.ram_free_mb, si.ram_total_mb)
    rec = embedmodels.recommend(si.ram_total_mb, None, os.environ.get("LC_ALL") or os.environ.get("LANG"))
    same = embedmodels._canon(rec.tag) == embedmodels._canon(name)
    if big:
        st = "warn" if st in ("ok", "info") else st
        text = f"{text}. {big}"
    if (big or st in ("warn", "fail")) and not same:
        hint = f"Recommended for this computer: {rec.tag} ({rec.label}). Switch with `{brand.CMD} embed use {rec.tag}`; the memory index is rebuilt."
    elif not same and cur.multilingual and not rec.multilingual and cur.ram_mb and cur.ram_mb > 4 * (rec.ram_mb or 1):
        hint = f"If you only chat in English, {rec.tag} ({rec.label}) uses far less memory: `{brand.CMD} embed use {rec.tag}`."
    return st, text, hint


@check("models.fit", "models", "Will the models fit?", needs_ollama=True)
def fit(ctx: Context):
    from agent.doctor.checks.ollama import is_remote
    from agent.doctor.core import Result

    oll, si = ctx.ollama, ctx.sysinfo
    if oll.url and is_remote(oll.url):
        return skip("the server is remote: its memory decides, not this machine's")
    m = ctx.section("models")
    out = []
    seen = set()
    for key, title in _ROLES:
        name = m.get(key) or ""
        if not name or name in seen or not model_installed(name, oll.models):
            continue
        seen.add(name)
        size = size_of(oll, name)
        if not size:
            continue
        gpus = [] if key == "embed" else ctx.gpus
        st, text = fit_verdict(size, gpus, si.ram_free_mb, si.ram_total_mb)
        hint = ""
        if st in ("warn", "fail"):
            hint = "Pick a smaller model (`nightage doctor --deep` lists a recommendation) or free memory."
        if key == "embed":
            st, text, hint = _embed_advice(name, st, text, hint, si)
        out.append(Result(st, f"{name}: {text}", hint, id=f"models.fit.{key}", title=f"{title} fit"))
    return out or skip("no installed model with a known size")


@check("models.capabilities", "models", "Tool-call support", needs_ollama=True, timeout=12)
def capabilities(ctx: Context):
    name = ctx.chat_model()
    if not name or not model_installed(name, ctx.ollama.models):
        return skip("no installed chat model to inspect")
    d = ctx.p.ollama_show(name)
    caps = (d or {}).get("capabilities")
    if not isinstance(caps, list):
        return info(f"{name}: this Ollama does not report capabilities (run `ollama show {name}` and look for \"tools\")")
    if "tools" in caps:
        return ok(f"{name} supports tool calls ({', '.join(map(str, caps))})")
    return warn(f"{name} does not list tool support ({', '.join(map(str, caps)) or 'no capabilities'})",
                "The agent depends on tool calls; choose a model whose `ollama show` lists \"tools\" (qwen3.5 does).")


@check("context", "models", "Context budget", core=True, timeout=10)
def context_budget(ctx: Context):
    chat = ctx.chat_model()
    if not chat:
        return warn("no chat model configured", "Set models.chat in config.yaml.")
    oll = ctx.ollama
    info_ = ctx.p.ctx_budget(chat, oll.models.get(chat) or oll.models.get(f"{chat}:latest") or 0, ctx.gpus, ctx.config)
    if not info_:
        return warn("could not be computed", "Not critical; the configured ceiling is used.")
    ov, comp, fin = info_.get("override"), info_.get("computed"), info_.get("final")
    detail = f"{chat}: {fin} tokens" + (f" (override {ov}, computed {comp})" if ov else f" (computed from {info_.get('source', '?')}, ceiling {info_.get('ceiling')})")
    if isinstance(fin, int) and fin < 4096:
        return warn(detail, "Very small: use a smaller model or free memory.")
    return ok(detail)
