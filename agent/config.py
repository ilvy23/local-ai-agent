"""Load and persist agent configuration.

On first run, a config.yaml with sane defaults is created next to the
project root (or wherever the caller points it). Later tasks (SQLite
persistence, vector memory) read models/persona from the same file.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from agent import osport
from typing import Any

import yaml

_SOURCE_ROOT = Path(__file__).resolve().parent.parent
# From a source checkout (or an editable install) config.yaml and data/ live next to the code. After a plain
# `pip install .` the package sits inside site-packages, which is no place for a database: use a per-user folder
# instead (NIGHTAGE_HOME overrides it).
if (_SOURCE_ROOT / "pyproject.toml").exists() and not os.environ.get("NIGHTAGE_HOME"):
    PROJECT_ROOT = _SOURCE_ROOT
else:
    PROJECT_ROOT = Path(os.environ.get("NIGHTAGE_HOME") or osport.data_home() / "nightage").expanduser()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_CONFIG: dict[str, Any] = {
    # default | phone. `phone` is an EXPERIMENTAL overlay for small devices (agent/profiles.py, docs/PORTABILITY.md); NIGHTAGE_PROFILE or
    # `nightage --profile phone` selects it for one run. Keys you set in this file always win over the overlay.
    "profile": "default",
    "models": {
        # Chat MUST be a tool-capable model (`ollama show <m>` lists "tools").
        # Ollama silently drops the tools for models without template support,
        # and the model then invents results instead of running anything.
        "chat": "qwen2.5:7b",
        "background": "qwen2.5:7b",  # fact distillation (no tools needed)
        "embed": "bge-m3",           # embeddings (multilingual, 1024-dim)
        "unrestricted": "qwen2.5:7b",  # model for the optional deep WhatsApp analysis (set your own in config.yaml)
        "weekly": "",  # empty = fall back to `background` for the weekly recap
        # Context window (tokens) for chat/tool-calling turns. Bigger lets more
        # conversation + recalled memory fit at once, at the cost of more VRAM
        # for the KV cache and (for most local models) some quality falloff
        # well past their trained context length - 16k is a safe middle
        # ground for an 8B model on an 8GB card, not a hard ceiling.
        "chat_num_ctx": 95000,
        # DeepSeek R1 for coding and analytical reasoning tasks.
        "coding": "deepseek-r1:7b",
        "reasoning": "deepseek-r1:7b",
        # Ceiling for agent.ctxbudget's automatic per-model context calculation (replaces the
        # old "qwen models never exceed 95000" name-based special case in agent.llm.cap_ctx).
        # AGENT_CTX_CEILING (or the legacy AGENT_MAX_QWEN_CTX) overrides this at runtime.
        "max_ctx_ceiling": 95000,
        # Per-model num_ctx overrides that always win over the automatic budget, e.g.:
        #   overrides: {qwen3.8:27b: {num_ctx: 65536}}
        "overrides": {},
    },
    # A partial mirror of agent/coding/config.py's DEFAULTS - only the keys
    # agent/settings.py exposes need to live here too, since that module's
    # _get() reads straight from this dict's deep-merge and never runs
    # coding/config.py's own merge. Keep values in sync with that file.
    "coding": {
        "diagnose_first": True,
        "executor": {"max_repair_attempts": 4},
        "escalation": {
            "tier3_enabled": False,  # unpulled + unproven >5 tok/s; opt in later
            "tier3_model": "qwen3-coder:30b",
            "total_budget_seconds": 300,
        },
        "sandbox": {"timeout_seconds": 30, "memory_mb": 2048},
        "auto_commit": True,
    },
    "whatsapp": {
        "min_messages": 5,
    },
    "persona": {
        "name": "Agent",
        "style": (
            "You are Agent, a warm and friendly agent who chats "
            "like a supportive friend: casual, honest, and curious. "
            "Give thorough, detailed answers — explain your reasoning, add "
            "useful context and examples, and cover the relevant angles rather "
            "than one-liners. Structure longer answers with short paragraphs or "
            "bullet points when it helps. Stay natural, not padded: be detailed "
            "because the content earns it, and match a quick reply to a quick "
            "question."
        ),
    },
    "data": {
        "db_path": "data/agent.db",
    },
    "ollama": {
        # Free the GPU when you leave the chat. Models otherwise stay resident
        # for a while, which is faster next time but leaves less room for the
        # embedding model — on a smaller card that's what makes memory flaky.
        # Turn this off if you have VRAM to spare and want warm starts.
        "unload_on_exit": True,
        # Remote gateway: use the Ollama on another machine (e.g. a PC on your network or VPN: "192.0.2.10:11434"). Empty = this
        # machine. The OLLAMA_HOST environment variable wins over this key. "keep_alive" ("30m" by default) is how long a model
        # stays loaded after a call.
        "host": "",
        # Start `ollama serve` in the background when the chat / code / research / web command begins, nothing answers on this machine
        # and an `ollama` program is installed (never when `host` points at another machine). Off: start it yourself (nightage ollama start).
        "autostart": True,
        # When you leave, stop the server again - only if Nightage started it itself (a server you or the system started is never
        # stopped). Loaded models are freed first (unload_on_exit). A SIGKILL cannot be caught: the next run notices the leftover.
        "stop_on_exit": True,
    },
    "ui": {
        # What the status line says while the model is generating and nothing can be shown
        # yet. Followed by animated dots and the seconds waited, e.g. "waiting...  12s".
        "wait_label": "thinking",
        # Write the reply on screen while the model is still generating it (streamed), instead of
        # waiting for the whole answer.
        "live": True,
        # Let the assistant show model-written interactive HTML widgets (`present` kind=html) in the GUI. They run in a
        # sandboxed frame without network access, but they are still code written by a model: off unless you want them.
        "allow_html_blocks": False,
        # Small kaomoji side-channel next to the normal status text (searching/error/done/...,
        # see agent/mood.py) - purely cosmetic, never a model call. `mood_map` lets you add or
        # override event -> kaomoji entries without touching the code, e.g.: {compacting: "o.O"}.
        "mood_emotes": True,
        "mood_map": {},
    },
    "memory": {
        "recall_k": 6,
        # A recalled memory must be at least this similar (cosine) to the question to be shown.
        # 0 = always show the k nearest, relevant or not.
        "recall_min_similarity": 0.55,
        "context_char_budget": 24000,
        # Facts are saved WHILE you chat (agent/memory/queue.py), never at exit. A turn's messages are taken once this
        # many user messages are waiting and the chat model is idle; `models.background` does the extraction.
        "distill_every_turns": 1,
        "distill_batch_messages": 8,
        # Quitting waits at most this long for an extraction that is already running; the rest is finished at the next
        # start (or by `nightage memory process`). Quitting never starts a model call.
        "exit_wait_seconds": 3.0,
    },
    "tools": {
        "max_iterations": 60,
    },
    "autonomy": {
        # When the agent keeps failing with the same error it searches the web for it (the query is
        # stripped of paths, names and tokens first). `AGENT_AUTO_LOOKUP=0` also turns it off.
        "web_lookup": True,
        "max_lookups_per_turn": 3,
        # Before "done" on code work the project's own check (tests / syntax) must pass; `AGENT_VERIFY=0` off.
        "verify": True,
        "verify_retries": 2,
        # Big build tasks get a harness-owned acceptance list the model must confirm item by item.
        "plan": True,
    },
    "safety": {
        # These lists EXTEND the built-in allow/block lists; they never shrink
        # them. Built-in safe commands and blocked patterns are hard-coded in
        # agent.safety and cannot be removed via config.
        "safe_commands": [],
        "blocked_patterns": [],
        "max_timeout_s": 300,
    },
    "background": {
        # Background bulk jobs pause while a game runs or the GPU is busier
        # than this, so gaming keeps full performance. CPU/RAM caps are
        # best-effort (real compute is in Ollama); the GPU + game signals bite.
        "pause_on_game": True,
        "max_gpu_percent": 40,
        "max_cpu_percent": 50,
        "max_ram_gb": 16,
    },
    "curator": {
        # Skills whose failure/(success+failure) ratio exceeds this are archived.
        "max_failure_ratio": 0.7,
        # Skills not used in this many days are archived as stale.
        "max_age_days": 90,
        # How often (in days) the curator maintenance pass runs.
        "run_every_days": 7,
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge `overrides` onto `base`, recursing into nested dicts."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


AUTO = "auto"
AUTO_ROLES = ("chat", "background", "coding", "reasoning", "unrestricted", "embed")      # "weekly" stays "" (= falls back to background)

LAST_LOAD: dict[str, Any] = {"path": "", "user_models": [], "profile": "default", "existed": False}


def last_load() -> dict[str, Any]:
    """Which file the last `load_config` read and which `models.*` keys that FILE set (as opposed to defaults / the profile).
    Kept outside the config dict on purpose: some commands dump the config back to config.yaml."""
    return dict(LAST_LOAD)


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Load config from `config_path`, creating it with defaults if missing.

    Also ensures `data_dir` exists. Values from `config_path` are deep-merged
    over `DEFAULT_CONFIG` so newly-added default keys are always present, even
    in an older config file. Always returns a fresh dict.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    LAST_LOAD.update(path=str(config_path), existed=config_path.exists(), user_models=[])

    if not config_path.exists():
        fresh = copy.deepcopy(DEFAULT_CONFIG)           # the file gets the plain defaults, never an overlay - except the model roles:
        for role in AUTO_ROLES:                         # "auto" = take what is installed (agent/modelpick.py), do not pin an absent model
            fresh["models"][role] = AUTO
        config_path.write_text(yaml.dump(fresh, sort_keys=False))
        return _finish(copy.deepcopy(DEFAULT_CONFIG), {})

    with config_path.open() as f:
        loaded = yaml.safe_load(f) or {}

    return _finish(copy.deepcopy(DEFAULT_CONFIG), loaded)


def _without_auto(loaded: Any) -> Any:
    m = loaded.get("models") if isinstance(loaded, dict) else None
    if not isinstance(m, dict):
        return loaded
    kept = {k: v for k, v in m.items() if not (k in AUTO_ROLES and str(v or "").strip().lower() in ("", AUTO))}
    return {**loaded, "models": kept}


def _finish(base: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    """defaults < profile overlay < the user's file, then push the runtime-affecting keys (remote host, tool list, loop thresholds)."""
    name = "default"
    try:
        from agent import profiles

        name = profiles.resolve_name(loaded if isinstance(loaded, dict) else {})
        if name != "default":
            host = ((loaded.get("ollama") or {}).get("host") if isinstance(loaded.get("ollama"), dict) else "") or ""
            from agent import llm

            remote = llm.is_remote(llm.normalize_host(host)) if host.strip() else llm.is_remote()
            base = _deep_merge(base, profiles.overlay_for(name, remote=remote))
    except Exception:  # noqa: BLE001 - a profile problem must never stop the config from loading
        name = "default"
    loaded = _without_auto(loaded)                       # "auto" / "" in the file = not chosen: the shipped default stands until modelpick resolves it
    merged = _deep_merge(base, loaded)
    um = loaded.get("models") if isinstance(loaded, dict) else None
    # A value identical to the shipped default does not count as "chosen by the user": the first run writes ALL defaults into config.yaml.
    dm = DEFAULT_CONFIG.get("models", {})
    LAST_LOAD.update(user_models=sorted(k for k, v in um.items() if v and v != dm.get(k)) if isinstance(um, dict) else [], profile=name)
    merged["profile"] = name
    _apply_runtime(merged, name)
    return merged


def _apply_runtime(config: dict[str, Any], name: str) -> None:
    """Process-wide settings that live in other modules; each step is isolated so one failure cannot break loading."""
    steps = []
    try:
        from agent import autonomy, llm, profiles
        from agent.tools import registry

        steps = [lambda: llm.configure(config), lambda: autonomy.configure(config), lambda: registry.configure(config),
                 lambda: profiles.activate(name)]
        if name == "phone":
            steps.append(profiles.apply_terminal_defaults)
    except Exception:  # noqa: BLE001
        return
    for step in steps:
        try:
            step()
        except Exception:  # noqa: BLE001
            pass
