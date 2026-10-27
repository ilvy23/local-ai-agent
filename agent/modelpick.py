"""Automatic model assignment: every model role gets a model that is really installed.

The shipped defaults (`models.chat: qwen2.5:7b`, `coding: deepseek-r1:7b`, `embed: bge-m3` ...) are guesses. On a machine that has only
`qwen3.5:0.8b` and one embedder, every role the user did not edit used to fail with a 404. `resolve()` fixes that at the moment a run
starts (chat / code / research / web, see `ollama_local.begin`), never at import time and never in `load_config` (so the config stays a
pure function of the files and tests stay offline):

  * the installed models come from /api/tags (cached `CACHE_SECONDS`); Ollama down or slow -> nothing changes, nothing blocks;
  * a role whose model is installed keeps it;
  * a role whose model is NOT installed and which the user's config.yaml did not choose (absent, or equal to the shipped default, or
    only set by a profile) falls back: `chat` to the best installed chat model, the other chat roles to the resolved chat model,
    `embed` to an installed embedding model;
  * a role the user chose explicitly but which is missing is KEPT and reported in `Resolution.ask` - the caller asks once, it is never
    swapped silently;
  * the embedder is never switched when a memory index already exists with a vector size the new one does not have (vectors of different
    models must not be mixed): the notice points to `nightage embed use`.

Embedding models are told apart by the catalogue (agent/embedmodels.py), common name parts and, for the rest, /api/show `capabilities`."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

CACHE_SECONDS = 30.0
CHAT_ROLES = ("chat", "background", "coding", "reasoning", "unrestricted")
ALL_ROLES = CHAT_ROLES + ("weekly", "embed")
_EMBED_WORDS = re.compile(r"(embed|minilm|bge-|e5-|gte-|arctic-embed|nomic|snowflake|granite-embedding|paraphrase)", re.I)
_FAMILY_RANK = ("qwen3.5", "qwen3", "qwen2.5", "llama3", "gemma", "mistral", "phi", "deepseek", "llama")

_cache: dict[str, tuple[float, dict[str, int] | None]] = {}
_caps: dict[str, bool] = {}


def installed(url: str, *, timeout: float = 1.5, now: float | None = None, use_cache: bool = True) -> dict[str, int] | None:
    """{name: size in bytes} of the installed models, or None when the server does not answer."""
    t = time.monotonic() if now is None else now
    hit = _cache.get(url)
    if use_cache and hit and t - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        data: dict[str, int] | None = {m["name"]: int(m.get("size", 0)) for m in r.json().get("models", []) if m.get("name")}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError):
        data = None
    _cache[url] = (t, data)
    return data


def known_missing(url: str, model: str) -> bool:
    """True only when a recent /api/tags answer (cached or re-fetched right now) shows `model` is NOT installed. Ollama down, nothing
    cached yet, or a cache entry older than `CACHE_SECONDS` -> False: this is a guard against a request that is certain to 404, never a
    reason to block a run on a guess. A model that looks missing is re-checked once with a fresh answer (it may have been pulled meanwhile)."""
    key = url.rstrip("/")
    hit = next((v for k, v in _cache.items() if k.rstrip("/") == key), None)
    if not model or hit is None or hit[1] is None or time.monotonic() - hit[0] >= CACHE_SECONDS:
        return False
    if has(model, hit[1]):
        return False
    fresh = installed(next(k for k in _cache if k.rstrip("/") == key), use_cache=False)
    return fresh is not None and not has(model, fresh)


def forget_cache() -> None:
    _cache.clear()
    _caps.clear()


def has(name: str, models: Any) -> bool:
    from agent.doctor.probes import model_installed

    return model_installed(name, models)


def is_embedding(name: str, url: str | None = None) -> bool:
    from agent import embedmodels

    if name in _caps:
        return _caps[name]
    d = embedmodels.describe(name)
    if d.exact and d.languages != embedmodels.UNKNOWN or _EMBED_WORDS.search(name):
        _caps[name] = True
        return True
    if d.languages != embedmodels.UNKNOWN and not d.exact:
        _caps[name] = True
        return True
    verdict = False
    if url:
        try:
            r = httpx.post(url.rstrip("/") + "/api/show", json={"model": name}, timeout=2.0)
            verdict = "embedding" in (r.json().get("capabilities") or [])
        except (httpx.HTTPError, ValueError, OSError):
            verdict = False
    _caps[name] = verdict
    return verdict


def split(models: dict[str, int], url: str | None = None) -> tuple[list[str], list[str]]:
    """(chat-capable names, embedding names) of the installed models."""
    chat, emb = [], []
    for n in sorted(models):
        (emb if is_embedding(n, url) else chat).append(n)
    return chat, emb


def best_chat(models: dict[str, int], chat_names: list[str], ram_mb: int = 0) -> str:
    """The installed chat model to use: the phone/RAM tier model when installed, else the best family that fits ~60 % of the RAM,
    else the smallest."""
    if not chat_names:
        return ""
    from agent import profiles

    tier = profiles.model_tier(ram_mb)
    for n in chat_names:
        if n == tier or n.split(":")[0] + ":" + n.split(":")[-1] == tier:
            return n

    def rank(n: str) -> int:
        for i, f in enumerate(_FAMILY_RANK):
            if n.lower().startswith(f):
                return i
        return len(_FAMILY_RANK)

    fits = [n for n in chat_names if not ram_mb or models.get(n, 0) * 1.15 / 1048576 <= ram_mb * 0.6] or [min(chat_names, key=lambda n: models.get(n, 0))]
    return sorted(fits, key=lambda n: (rank(n), -models.get(n, 0)))[0]


@dataclass
class Pick:
    model: str
    source: str            # config (chosen in config.yaml) | default | profile | auto
    installed: bool | None  # None = unknown (Ollama down)


@dataclass
class Resolution:
    roles: dict[str, Pick] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    ask: list[tuple[str, str]] = field(default_factory=list)       # (role, configured model that is missing) - explicit choices
    reachable: bool = True
    chat_names: list[str] = field(default_factory=list)
    embed_names: list[str] = field(default_factory=list)
    nothing_for_chat: bool = False


def resolve(models_cfg: dict, user_roles: set[str] | list[str], models: dict[str, int] | None, *, ram_mb: int = 0, index_dim: int | None = None,
            url: str | None = None, profile_roles: set[str] | None = None) -> Resolution:
    """Decide the model of every role. Pure given its inputs (`models` = installed, None = Ollama down). Does not change `models_cfg`."""
    from agent import embedmodels

    user = set(user_roles)
    res = Resolution()
    configured = {r: str(models_cfg.get(r) or "") for r in ALL_ROLES}

    def src(role: str) -> str:
        return "config" if role in user else "profile" if role in (profile_roles or set()) else "default"

    if models is None:
        res.reachable = False
        for r, m in configured.items():
            res.roles[r] = Pick(m, src(r), None)
        return res
    chat_names, emb_names = split(models, url)
    res.chat_names, res.embed_names = chat_names, emb_names
    chosen_chat = ""
    # chat first: every other chat role follows it
    for role in CHAT_ROLES:
        want = configured[role]
        if not want:
            res.roles[role] = Pick("", src(role), None)
            continue
        if has(want, models):
            res.roles[role] = Pick(want, src(role), True)
            if role == "chat":
                chosen_chat = want
            continue
        if role in user:
            res.roles[role] = Pick(want, "config", False)
            res.ask.append((role, want))
            if role == "chat":
                chosen_chat = best_chat(models, chat_names, ram_mb)      # only used as the fallback for the other roles
            continue
        fallback = chosen_chat or best_chat(models, chat_names, ram_mb)
        if role == "chat":
            chosen_chat = fallback
        if fallback:
            res.roles[role] = Pick(fallback, "auto", True)
        else:
            res.roles[role] = Pick(want, src(role), False)
    if not chat_names:
        res.nothing_for_chat = True
    weekly = configured["weekly"]
    res.roles["weekly"] = Pick(weekly, src("weekly"), (has(weekly, models) if weekly else None))
    # the memory model
    want = configured["embed"]
    if has(want, models):
        res.roles["embed"] = Pick(want, src("embed"), True)
    elif "embed" in user:
        res.roles["embed"] = Pick(want, "config", False)
        res.ask.append(("embed", want))
    elif emb_names:
        cand = embedmodels.recommend(ram_mb, None, None).tag
        pick = next((n for n in emb_names if embedmodels._canon(n) == embedmodels._canon(cand)), emb_names[0])
        d = embedmodels.describe(pick)
        if index_dim and d.dim != index_dim:
            res.roles["embed"] = Pick(want, src("embed"), False)
            res.notices.append(f"Memory model {want} is not installed, and the memory index was built with another vector size; "
                               f"{pick} is installed but would not match it. Run: nightage embed use {pick} (rebuilds the index).")
        else:
            res.roles["embed"] = Pick(pick, "auto", True)
    else:
        res.roles["embed"] = Pick(want, src("embed"), False)
    picks = {r: p for r, p in res.roles.items() if p.source == "auto"}
    if picks:
        used = sorted({p.model for r, p in picks.items() if r != "embed"})
        parts = []
        if used:
            parts.append(f"{used[0]} for everything" if len(used) == 1 and len(picks) - ("embed" in picks) >= 4 else
                         "chat: " + res.roles["chat"].model)
        if "embed" in picks:
            parts.append(f"{picks['embed'].model} for memory")
        note = "Using " + " and ".join(parts)
        only = len(chat_names) == 1 and len(emb_names) == 1
        res.notices.append(note + (" (only models installed). " if only else ". ") + "Change: nightage models")
    return res


def apply(config: dict, res: Resolution) -> None:
    """Write the resolved models into `config["models"]` IN PLACE (so copies sharing the dict see them)."""
    m = config.setdefault("models", {})
    for role, pick in res.roles.items():
        if pick.source == "auto" and pick.model:
            m[role] = pick.model


def index_dim(config: dict) -> int | None:
    """The vector size recorded in the memory database (None = no index yet / unreadable). Read-only."""
    import sqlite3
    from pathlib import Path

    try:
        from agent.config import PROJECT_ROOT

        p = Path(config["data"]["db_path"])
        p = p if p.is_absolute() else PROJECT_ROOT / p
        if not p.exists():
            return None
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=1)
        try:
            row = con.execute("SELECT v FROM app_state WHERE k='embed_dim'").fetchone()
        finally:
            con.close()
        return int(row[0]) if row else None
    except Exception:  # noqa: BLE001 - advice only
        return None


def resolve_config(config: dict, url: str, *, models: dict[str, int] | None = None, ram_mb: int | None = None) -> Resolution:
    """Resolve for a loaded config (uses `agent.config.last_load` for the roles the file chose) and apply the automatic picks."""
    from agent import config as cfgmod
    from agent import sysinfo

    info = cfgmod.last_load()
    if models is None:
        models = installed(url)
    ram = sysinfo.current().ram_total_mb if ram_mb is None else ram_mb
    prof = {"chat", "background", "coding", "reasoning", "unrestricted", "embed"} if info.get("profile") == "phone" else set()
    res = resolve(config.get("models") or {}, set(info.get("user_models") or []), models, ram_mb=ram, index_dim=index_dim(config), url=url,
                  profile_roles=prof)
    apply(config, res)
    return res


# ------------------------------------------------------------------------------------------------------ recommendation


CHAT_SIZES_MB = {"qwen3.5:0.8b": 1000, "qwen3.5:2b": 2700, "qwen3.5:4b": 3400, "qwen3.5:9b": 6600, "qwen3.5:27b": 17000}   # ollama.com tag pages, approximate


def recommend_chat(ram_mb: int, vram_mb: int = 0, phone: bool = False) -> str:
    """The chat model to offer when NOTHING is installed. Same tiers as install.sh (`pick_model`): 90 % of the graphics memory, else half of
    the RAM; the phone profile uses agent/profiles.model_tier."""
    if phone:
        from agent import profiles

        return profiles.model_tier(ram_mb)
    budget = vram_mb * 9 // 10 if vram_mb > 0 else (ram_mb // 2 if ram_mb > 0 else 4096)
    return "qwen3.5:0.8b" if budget < 3072 else "qwen3.5:4b" if budget < 8192 else "qwen3.5:9b" if budget < 20480 else "qwen3.5:27b"


def size_text(model: str) -> str:
    mb = CHAT_SIZES_MB.get(model)
    return f"about {mb / 1024:.1f} GB" if mb else "size unknown"


# ------------------------------------------------------------------------------------------------------ writing config.yaml


def set_models(path: Any, values: dict[str, str]) -> str:
    """Write `models.<role>: value` pairs into config.yaml, editing the text in place so comments survive; falls back to a YAML dump when
    the layout is unusual. Returns "text" or "dump". The result is always re-parsed and checked."""
    from pathlib import Path

    import yaml

    path = Path(path)
    text = path.read_text() if path.exists() else ""
    lines = text.splitlines()
    values = {k: (v if str(v) else "''") for k, v in values.items()}
    want = dict(values)
    idx = next((i for i, ln in enumerate(lines) if re.match(r"^models:\s*(#.*)?$", ln)), None)
    new_lines = None
    if idx is not None:
        end = idx + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t", "#"))):
            end += 1
        body = lines[idx + 1:end]
        indent = next((re.match(r"^(\s+)\S", ln).group(1) for ln in body if ln.strip() and not ln.lstrip().startswith("#")), "  ")
        out = []
        for ln in body:
            m = re.match(rf"^{indent}([A-Za-z_]+):\s*(.*?)(\s+#.*)?$", ln)
            if m and m.group(1) in want:
                out.append(f"{indent}{m.group(1)}: {want.pop(m.group(1))}{m.group(3) or ''}")
            else:
                out.append(ln)
        # keep trailing blank lines after the block, add missing keys right under `models:`
        added = [f"{indent}{k}: {v}" for k, v in want.items()]
        new_lines = lines[:idx + 1] + added + out + lines[end:]
    elif not text.strip() or "models:" not in text:
        new_lines = lines + (["models:"] + [f"  {k}: {v}" for k, v in want.items()])
    if new_lines is not None:
        candidate = "\n".join(new_lines) + "\n"
        try:
            parsed = yaml.safe_load(candidate) or {}
            if all(str((parsed.get("models") or {}).get(k)) == str(v) for k, v in values.items()):
                path.write_text(candidate)
                return "text"
        except yaml.YAMLError:
            pass
    data = yaml.safe_load(text) if text.strip() else {}
    data = data if isinstance(data, dict) else {}
    data.setdefault("models", {})
    if not isinstance(data["models"], dict):
        data["models"] = {}
    data["models"].update(values)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return "dump"


def use_everywhere(path: Any, chat_model: str, embed: str | None = None) -> dict[str, str]:
    """`nightage models use`: one chat model for all chat roles (weekly too), optionally the embedder."""
    vals = {r: chat_model for r in CHAT_ROLES}
    vals["weekly"] = chat_model
    if embed:
        vals["embed"] = embed
    set_models(path, vals)
    return vals


# ------------------------------------------------------------------------------------------------------ the start-up flow


class NoModelAvailable(SystemExit):
    """No usable chat model and nothing could be arranged: the run stops with this message (exit code 1) instead of sending a request that
    would only come back as a 404."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _declined_path() -> Any:
    from agent import ollama_local

    return ollama_local.state_dir() / "declined.txt"


def _declined() -> set[str]:
    try:
        return set(_declined_path().read_text().split())
    except OSError:
        return set()


def _decline(tag: str) -> None:
    try:
        p = _declined_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(sorted(_declined() | {tag})) + "\n")
    except OSError:
        pass


def ensure_models(config: dict, url: str, *, say: Any, confirm: Any, pull: Any, require_chat: bool = True, config_path: Any = None,
                  models: dict[str, int] | None = None, ram_mb: int | None = None, vram_mb: int = 0, locale: str | None = None) -> Resolution:
    """Resolve the roles, tell the user what was chosen, and fix what is missing - one continuous path, no file editing:

      * an explicit model that is not installed: offer the best installed chat model instead (written to config.yaml only after the yes), or
        the download of the missing one;
      * nothing installed: offer the recommended model for this machine (size shown) and pull it with the percent bar;
      * no memory model installed: offer the recommended embedder once (a decline is remembered);
      * never returns with a chat model that is not installed when `require_chat` (raises `NoModelAvailable`).

    `confirm(question, default) -> bool | None` (None = cannot ask: no keyboard), `pull(model) -> bool`, `say(text)`."""
    from agent import config as cfgmod
    from agent import embedmodels, sysinfo

    path = config_path or cfgmod.last_load().get("path") or cfgmod.DEFAULT_CONFIG_PATH
    si = sysinfo.current()
    ram = si.ram_total_mb if ram_mb is None else ram_mb
    res = resolve_config(config, url, models=models, ram_mb=ram)
    if not res.reachable:
        return res
    for n in res.notices:
        say(n)
    m = config.setdefault("models", {})

    def refresh() -> Resolution:
        forget_cache()
        return resolve_config(config, url, models=None, ram_mb=ram)

    # 1. models the user chose that are not installed
    seen: set[str] = set()
    for role, missing in res.ask:
        if missing in seen or role == "embed":
            continue
        seen.add(missing)
        alt = best_chat({n: s for n, s in (models or installed(url) or {}).items()}, res.chat_names, ram)
        if alt:
            ans = confirm(f"{missing} is not installed. Use {alt} instead?", True)
            if ans:
                roles = [r for r in CHAT_ROLES + ("weekly",) if str(m.get(r) or "") == missing]
                set_models(path, {r: alt for r in roles})
                for r in roles:
                    m[r] = alt
                say(f"Now using {alt} for {', '.join(roles)} (saved in {path}).")
                continue
            if ans is None:
                raise NoModelAvailable(f"{missing} is not installed. Use {alt} with:  nightage models use {alt}   or download it:  nightage pull {missing}")
        if confirm(f"Download {missing}?", not alt):
            if pull(missing):
                continue
        raise NoModelAvailable(f"{missing} is not installed. Download it:  nightage pull {missing}   or pick another:  nightage models")
    # 2. nothing to chat with at all
    res = refresh() if seen else res
    chat = (config.get("models") or {}).get("chat", "")
    inst = installed(url) or {}
    if require_chat and not has(chat, inst):
        if res.chat_names:
            pass                                   # resolved by the automatic assignment above
        else:
            rec = recommend_chat(ram, sum(getattr(g, "total_mb", 0) for g in si.gpus), phone=cfgmod.last_load().get("profile") == "phone")
            ans = confirm(f"No chat model is installed. Download {rec} ({size_text(rec)})?", True)
            if ans and pull(rec):
                use_everywhere(path, rec)
                for r in CHAT_ROLES + ("weekly",):
                    m[r] = rec
                say(f"Using {rec} for everything (saved in {path}).")
            else:
                raise NoModelAvailable(f"No chat model is installed. Download one:  nightage pull {rec}")
            res = refresh()
    # 3. a memory model
    emb = (config.get("models") or {}).get("embed", "")
    inst = installed(url) or {}
    if emb and not has(emb, inst) and not res.embed_names and "embed" not in {r for r, _ in res.ask}:
        rec = embedmodels.recommend(ram, None, locale)
        dim = index_dim(config)
        if dim and rec.dim != dim:
            say("No memory model is installed (the existing memory index needs a different vector size). Memory searches by keywords. See: nightage embed list")
        elif rec.tag not in _declined():
            ans = confirm(f"No memory model is installed. Download {rec.tag} ({rec.download_mb} MB) so Nightage can recall by meaning?", True)
            if ans and pull(rec.tag):
                set_models(path, {"embed": rec.tag})
                m["embed"] = rec.tag
                say(f"Memory model: {rec.tag} (saved in {path}).")
            elif ans is False:
                _decline(rec.tag)
                say("Memory searches by keywords. Later:  nightage embed list")
            else:
                say("No memory model installed: memory searches by keywords. Later:  nightage pull " + rec.tag)
    return res
