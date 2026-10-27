"""config.yaml: valid YAML, no typos in keys, sensible models, safety lists that only extend the built-in rules."""

from __future__ import annotations

import difflib
import re

from agent.doctor import fixes
from agent.doctor.core import Context, check, fail, ok, warn

DANGEROUS_WORDS = frozenset({"rm", "sudo", "doas", "su", "dd", "mkfs", "chmod", "chown", "kill", "pkill", "killall", "systemctl",
                             "reboot", "shutdown", "sh", "bash", "zsh", "eval", "curl", "wget", "ssh", "scp", "nc", "git", "pip",
                             "npm", "python", "python3", "node"})
_EMBED_HINTS = ("embed", "bge", "nomic", "minilm", "e5", "gte-")


def safety_problems(safety):
    if safety is None:
        return []
    if not isinstance(safety, dict):
        return ["not a mapping"]
    out = []
    for k in ("safe_commands", "blocked_patterns"):
        v = safety.get(k, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            out.append(f"{k} is not a list of strings")
    t = safety.get("max_timeout_s", 300)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0:
        out.append("max_timeout_s must be a positive number")
    return out


@check("config", "config", "config.yaml", core=True)
def config_file(ctx: Context):
    path, parsed, err = ctx.raw_config
    if err == "missing":
        return warn(f"{path} not found; defaults are used", "It is created on the first normal start.")
    if err:
        return fail(err, f"Fix the syntax in {path} or delete it to get the defaults back.")
    bad = safety_problems((parsed or {}).get("safety"))
    if bad:
        return warn("safety: " + "; ".join(bad), "safety.safe_commands / blocked_patterns must be lists of strings; they only extend the built-in lists.")
    return ok(str(path))


def _example_keys() -> set[str]:
    """Top-level keys that config.example.yaml documents (some are valid without being in DEFAULT_CONFIG)."""
    try:
        import yaml

        from agent.config import PROJECT_ROOT

        data = yaml.safe_load((PROJECT_ROOT / "config.example.yaml").read_text(encoding="utf-8")) or {}
        return {str(k) for k in data} if isinstance(data, dict) else set()
    except Exception:  # noqa: BLE001
        return set()


@check("config.keys", "config", "Unknown config keys", requires=())
def keys(ctx: Context):
    _path, parsed, err = ctx.raw_config
    if err or not parsed:
        return None
    from agent.config import DEFAULT_CONFIG

    known = set(DEFAULT_CONFIG) | {"profile", "update", "plugins", "doctor"} | _example_keys()
    unknown = [k for k in parsed if k not in known]
    if not unknown:
        return ok(f"{len(parsed)} top-level keys, all known")
    notes = []
    for k in unknown:
        near = difflib.get_close_matches(str(k), sorted(known), n=1, cutoff=0.7)
        notes.append(f"{k}" + (f" (did you mean {near[0]}?)" if near else ""))
    return warn("not recognised: " + ", ".join(notes), "Unknown keys are ignored. Check the spelling against config.example.yaml.")


@check("config.models", "config", "Model settings")
def models(ctx: Context):
    m = ctx.section("models")
    problems = []
    chat, embed = str(m.get("chat") or ""), str(m.get("embed") or "")
    for k in ("chat", "coding", "embed"):
        v = m.get(k)
        if v and (not isinstance(v, str) or re.search(r"\s", v)):
            problems.append(f"models.{k} is not a plain model name")
    if chat and chat == embed:
        problems.append("models.chat and models.embed are the same model")
    if embed and not any(h in embed.lower() for h in _EMBED_HINTS):
        problems.append(f"models.embed `{embed}` does not look like an embedding model")
    if chat and any(h in chat.lower() for h in _EMBED_HINTS):
        problems.append(f"models.chat `{chat}` looks like an embedding model")
    if problems:
        return warn("; ".join(problems), "Set models.chat to a chat model with tool support and models.embed to an embedding model (bge-m3).")
    return ok("chat and embedding model look sensible")


@check("config.safety", "config", "Safety settings")
def safety(ctx: Context):
    _p, parsed, err = ctx.raw_config
    s = (parsed or {}).get("safety") if not err else None
    if not isinstance(s, dict):
        return ok("nothing overridden (built-in rules apply)") if not err else None
    risky = []
    for entry in s.get("safe_commands") or []:
        if not isinstance(entry, str):
            continue
        first = entry.strip().split(" ")[0] if entry.strip() else ""
        if first in DANGEROUS_WORDS or entry.strip() in ("*", ".*", ""):
            risky.append(entry)
    notes = []
    if risky:
        notes.append("safe_commands lets these run without asking: " + ", ".join(repr(r) for r in risky[:5]))
    t = s.get("max_timeout_s", 300)
    if isinstance(t, (int, float)) and not isinstance(t, bool) and t > 3600:
        notes.append(f"max_timeout_s is {t:g} s (over an hour)")
    if notes:
        return warn("; ".join(notes), "These lists extend the built-in classifier and can never remove its blocks, but a broad safe_commands entry removes the confirmation step for it.")
    return ok("safe_commands / blocked_patterns only extend the built-in rules")


@check("profile", "config", "Profile", core=True)
def profile(ctx: Context):
    from agent import profiles

    si = ctx.sysinfo
    raw = ctx.raw_config[1] or {}
    name = profiles.resolve_name(ctx.config)
    declared = str(raw.get("profile") or "").strip().lower()
    if declared and declared not in profiles.NAMES:
        return warn(f"profile {declared!r} is unknown, so `default` is used", f"Valid profiles: {', '.join(profiles.NAMES)}.")
    hint = profiles.suggest(si, raw)
    if hint:
        kw = {}
        if si.low_ram:
            try:
                cmd, fn = fixes.set_config_key(ctx.raw_config[0], "profile", "phone")
                kw = {"command": cmd, "fix": fn}
            except fixes.FixRefused:
                pass
        return warn("default (not tuned for this device)", hint, **kw)
    return ok(name + (" (experimental)" if name == profiles.PHONE else ""))




@check("config.roles", "config", "Models per role", requires=())
def models_per_role(ctx: Context):
    """Which config.yaml is loaded, and for every role: from the file, a default, or chosen automatically from the installed models."""
    from agent import config as cfgmod
    from agent import modelpick
    from agent.doctor.core import Result

    path, parsed, err = ctx.raw_config
    if err and err != "missing":
        return None
    dm = cfgmod.DEFAULT_CONFIG["models"]
    file_models = ((parsed or {}).get("models") if isinstance(parsed, dict) else None) or {}
    user = {k for k, v in file_models.items() if v and str(v).strip().lower() != cfgmod.AUTO and v != dm.get(k)}
    merged = {**dm, **{k: v for k, v in file_models.items() if v and str(v).strip().lower() != cfgmod.AUTO}}
    oll = ctx.ollama
    inst = dict(oll.models) if oll.reachable else None
    res = modelpick.resolve(merged, user, inst, ram_mb=ctx.sysinfo.ram_total_mb, url=None)
    lines = []
    for role in modelpick.ALL_ROLES:
        p = res.roles.get(role)
        if p and p.model:
            lines.append(f"{role}: {p.model} ({'in the file' if p.source == 'config' else 'chosen from the installed models' if p.source == 'auto' else 'default'}"
                         f"{'' if p.installed in (True, None) else ', NOT installed'})")
    out = f"{path} - " + ("file exists" if err != "missing" else "file does not exist yet")
    r = Result("info", out, id="config.roles", title="Models per role")
    r.lines = lines
    if res.ask and inst:
        role, missing = res.ask[0]
        others = [n for n in res.chat_names if n != missing]
        if others:
            r = Result("warn", f"{role}: {missing} is set in {path} but is not installed, and {others[0]} is",
                       f"Run `nightage models use {others[0]}` (or `nightage pull {missing}`). Editing a different config.yaml than the one above has no effect.",
                       id="config.roles", title="Models per role")
            r.lines = lines
    return r
