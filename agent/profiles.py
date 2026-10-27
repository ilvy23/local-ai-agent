"""Config profiles: `default` (nothing changes) and `phone` (EXPERIMENTAL - tiny models, small context, fewer tools, stricter loop catching).

A profile is a config OVERLAY applied at load time by `agent.config.load_config`:

    merged = DEFAULT_CONFIG  <  profile overlay  <  the user's config.yaml

so a key the user wrote in config.yaml always wins over the profile, and nothing is ever written back to config.yaml.
Selecting a profile (first hit wins): `--profile` on the command line (it sets NIGHTAGE_PROFILE), env NIGHTAGE_PROFILE,
`profile:` in config.yaml, else `default`. Unknown names fall back to `default`.

The phone overlay is built from the machine (`agent.sysinfo`): RAM picks the model tier and the context ceiling. When Ollama runs
on another machine (`ollama.host` / OLLAMA_HOST: the "remote gateway" setup, UI on the phone and the model on the home PC) the
profile leaves the MODELS alone - the home PC can run its normal ones - and only trims the client side.

Model names (verified against ollama.com/library/qwen3.5/tags and /library/all-minilm on 2026-10-26): qwen3.5:0.8b, qwen3.5:2b,
qwen3.5:4b, all-minilm. Tool support is a family-level label on the library page (not per tag) - UNVERIFIED per tag.
"""

from __future__ import annotations

import os
from typing import Any

from agent import sysinfo

DEFAULT, PHONE = "default", "phone"
NAMES = (DEFAULT, PHONE)
ENV = "NIGHTAGE_PROFILE"

# Tools a phone does not need or cannot run well (names / fnmatch patterns, see agent.tools.registry.configure).
PHONE_DISABLED_TOOLS = ["deep_research", "generate_image", "send_album", "send_file", "get_media_info", "make_diagram", "wa_*"]

_active = DEFAULT


def active_name() -> str:
    """The profile the last `load_config` resolved ('default' before any load)."""
    return _active


def activate(name: str) -> None:
    global _active
    _active = name if name in NAMES else DEFAULT


def resolve_name(raw_config: dict | None = None, env: dict[str, str] | None = None) -> str:
    e = os.environ if env is None else env
    for candidate in (e.get(ENV, ""), str((raw_config or {}).get("profile") or "")):
        c = candidate.strip().lower()
        if c in NAMES:
            return c
    return DEFAULT


# ----------------------------------------------------------------------------------------------------------- tiers


def model_tier(ram_mb: int) -> str:
    """Chat model for this much RAM: <3 GB 0.8b, <6 GB 2b, else 4b. Unknown RAM (0) takes the middle tier."""
    if ram_mb <= 0:
        return "qwen3.5:2b"
    if ram_mb < 3 * 1024:
        return "qwen3.5:0.8b"
    if ram_mb < 6 * 1024:
        return "qwen3.5:2b"
    return "qwen3.5:4b"


def ctx_ceiling(ram_mb: int, remote: bool) -> int:
    """Context ceiling (tokens) for `models.max_ctx_ceiling`, 4k..16k by RAM (agent.ctxbudget caps every model by it)."""
    if remote:
        return 16384            # the server's own memory decides; the phone only has to hold the text
    if ram_mb <= 0:
        return 8192
    for limit_mb, ctx in ((3 * 1024, 4096), (4 * 1024, 6144), (6 * 1024, 8192), (8 * 1024, 12288)):
        if ram_mb < limit_mb:
            return ctx
    return 16384


def phone_overlay(info: sysinfo.SysInfo | None = None, *, remote: bool = False) -> dict[str, Any]:
    """The exact keys the phone profile sets (see docs/PORTABILITY.md for the table)."""
    info = info or sysinfo.current()
    ctx = ctx_ceiling(info.ram_total_mb, remote)
    overlay: dict[str, Any] = {
        "models": {"max_ctx_ceiling": ctx, "chat_num_ctx": ctx},
        "ollama": {"unload_on_exit": True, "keep_alive": "5m", "autostart": True, "stop_on_exit": True},   # frees the phone's RAM when you leave
        "memory": {"recall_k": 3, "context_char_budget": min(24000, ctx * 2)},
        "tools": {"max_iterations": 20, "disabled": list(PHONE_DISABLED_TOOLS)},
        "ui": {"mood_emotes": False, "live": True},
        "autonomy": {                         # chat stuck watcher (agent/autonomy.py): sooner than 3 failures / 3 identical calls
            "max_lookups_per_turn": 1, "fail_streak": 2, "repeat_same_error": 2, "identical_calls": 2,
            "plan": False,                    # the harness-owned acceptance list costs several extra turns a tiny model cannot afford
        },
        "background": {"max_ram_gb": 2},
        "coding": {                           # loop guard + step budget (agent/codeagent/*): every threshold lower than its default
            "loop_note": 1, "loop_repeat": 2, "loop_abort": 3,
            "family_note": 3, "family_abort": 5, "run_spam_note": 6, "run_spam_abort": 9,
            "same_tool_note": 5, "same_tool_abort": 7, "alt_note": 2, "alt_abort": 3,
            "read_repeat_note": 3, "read_repeat_abort": 5, "chant_hits": 8, "chant_think_hits": 6,
            "malformed_abort": 3, "auto_extend_max": 3, "announce_max": 4, "stray_looks": 5, "look_block_at": 10,
            "stage_max_steps": 80, "stage_max_minutes": 10, "num_predict": 2048, "think_predict": 1500,
            "sprawl_tokens": 3000, "sprawl_quiet": 1000,
            "escalation": {"tier3_enabled": False},
        },
    }
    if not remote:                            # a local model has to fit the phone; a remote one is the home PC's business
        chat = model_tier(info.ram_total_mb)
        overlay["models"].update({
            "chat": chat, "background": "qwen3.5:0.8b", "coding": chat, "reasoning": chat, "unrestricted": chat,
            "embed": "all-minilm",            # 46 MB instead of bge-m3's ~1.2 GB (a fresh phone DB has no embedding-size history)
        })
    return overlay


def overlay_for(name: str, *, info: sysinfo.SysInfo | None = None, remote: bool = False) -> dict[str, Any]:
    return phone_overlay(info, remote=remote) if name == PHONE else {}


def apply_terminal_defaults(info: sysinfo.SysInfo | None = None, env: dict[str, str] | None = None) -> bool:
    """Phone profile on a terminal without unicode: ask the UI for its ASCII glyph set (NIGHTAGE_PLAIN), unless the user set it.
    Returns True when it switched plain mode on."""
    e = os.environ if env is None else env
    info = info or sysinfo.current()
    if info.unicode_ok or e.get("NIGHTAGE_PLAIN") or e.get("AGENT_PLAIN"):
        return False
    e["NIGHTAGE_PLAIN"] = "1"
    return True


# ----------------------------------------------------------------------------------------------------------- doctor


def suggest(info: sysinfo.SysInfo | None = None, config: dict | None = None, env: dict[str, str] | None = None) -> str:
    """'' when no suggestion; else a one-line reason to try `--profile phone`. Never applied automatically."""
    e = os.environ if env is None else env
    info = info or sysinfo.current()
    if resolve_name(config, e) != DEFAULT or e.get(ENV):
        return ""
    if (config or {}).get("profile"):          # the user chose one (even an explicit 'default')
        return ""
    if info.is_termux:
        return "Termux detected: try `nightage --profile phone` (tiny models, small context, stricter loop catcher)."
    if info.low_ram:
        return f"Only {info.ram_total_mb // 1024} GB of RAM: try `nightage --profile phone` (smaller models, lower context)."
    return ""
