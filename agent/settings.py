"""View and edit a curated set of config values from config.yaml.

Only exposes the knobs worth touching, by a short name mapped to a dotted
path. Writes back to the project config.yaml, coercing obvious int/bool
values. Each setting carries a `kind` (bool / int / choice / model / string /
long_text / list) so the interactive menu can offer the right input widget
instead of making you type a raw value from memory.
"""

from __future__ import annotations

from agent import brand

from pathlib import Path
from typing import Any

import yaml

from agent.config import DEFAULT_CONFIG_PATH, load_config

# short name -> metadata. `path` is the dotted config path; `kind` drives the
# interactive widget; `range`/`choices` are only present where relevant.
# Categories are listed here in the order they should appear to a first-time
# user: Personality (identity-defining, low-risk) first, Safety (consequential,
# rarely touched) last.
EDITABLE: dict[str, dict[str, Any]] = {
    "persona_name": {
        "path": "persona.name", "kind": "string", "category": "Personality",
        "help": "What it calls itself. Purely cosmetic - doesn't change how "
                "it behaves, just its name.",
    },
    "persona_style": {
        "path": "persona.style", "kind": "long_text", "category": "Personality",
        "help": "The instructions for how it talks and acts - its whole "
                "personality in one block of text. This is long, and "
                "changing it can shift its tone completely.",
    },
    "mood_emotes": {
        "path": "ui.mood_emotes", "kind": "bool", "category": "Interface",
        "help": "Show a tiny kaomoji (^_^, O_o, ...) next to the status line "
                "while it's searching/coding/stuck, alongside the normal "
                "text - purely cosmetic, never replaces the real output.",
    },
    "chat_model": {
        "path": "models.chat", "kind": "model", "category": "Models",
        "help": f"Used for everyday chat and tool use ({brand.CMD} chat/resume). "
                "Bigger/smarter models feel more capable but reply slower.",
    },
    "background_model": {
        "path": "models.background", "kind": "model", "category": "Models",
        "help": "Runs quietly to distill facts, summarize chats, and other "
                "passive work. A small/fast model keeps this cheap.",
    },
    "unrestricted_model": {
        "path": "models.unrestricted", "kind": "model", "category": "Models", "private": True,
        "help": "The big model used for deep WhatsApp analysis (profiles, "
                "predictions). Only used by the optional private analysis module.",
    },
    "coding_model": {
        "path": "coding.executor.model", "kind": "model", "category": "Models",
        "help": f"Writes/edits code for `{brand.CMD} code`. A coder "
                "model (for example qwen2.5-coder) follows code instructions best.",
    },
    "embed_model": {
        "path": "models.embed", "kind": "model", "category": "Models",
        "help": "Turns text into vectors for semantic memory search. "
                "Changing this here does NOT rebuild the index - use the "
                "'Re-embed memory' option instead so old memories stay "
                "searchable.",
    },
    "weekly_model": {
        "path": "models.weekly", "kind": "model", "category": "Models", "private": True,
        "help": "Used once a week for your WhatsApp weekly recap. Leave "
                "blank to just reuse the everyday background model above - "
                "only set this if you want the weekly recap written by a "
                "different model.",
    },
    "chat_context_window": {
        "path": "models.chat_num_ctx", "kind": "int", "range": (2048, 131072),
        "category": "Memory & chat",
        "help": "How much conversation + recalled memory the chat model can "
                "see at once, in tokens (roughly 3/4 of a word each). Bigger "
                "means longer conversations and more memory fit without "
                "getting cut off, but uses more GPU memory and can make "
                "some models slightly less sharp well past what they were "
                "trained on. 16384 is a good default for an 8B model.",
    },
    "recall_k": {
        "path": "memory.recall_k", "kind": "int", "range": (1, 20), "category": "Memory & chat",
        "help": "How many past memories get pulled into context per chat "
                "turn. Higher = more context but slower, more likely to "
                "include irrelevant noise.",
    },
    "context_char_budget": {
        "path": "memory.context_char_budget", "kind": "int", "range": (2000, 100000),
        "category": "Memory & chat",
        "help": "Max characters of recalled memory injected per chat turn. "
                "Higher lets the model see more history, at the cost of its "
                "context window.",
    },
    "unload_on_exit": {
        "path": "ollama.unload_on_exit", "kind": "bool", "category": "Performance",
        "help": "Free the model from VRAM the moment you leave a chat, "
                "instead of keeping it warm for your next command. Turn "
                "off if you chat often and don't mind the GPU staying busy "
                "between sessions.",
    },
    "pause_on_game": {
        "path": "background.pause_on_game", "kind": "bool", "category": "Performance",
        "help": "Pause all background AI work (summaries, digests, watch) "
                "while a game or other GPU-heavy app is detected running.",
    },
    "max_gpu_percent": {
        "path": "background.max_gpu_percent", "kind": "int", "range": (0, 100), "category": "Performance",
        "help": "Background jobs pause themselves if GPU usage is already "
                "above this percent, so they don't compete with whatever "
                "you're doing.",
    },
    "max_cpu_percent": {
        "path": "background.max_cpu_percent", "kind": "int", "range": (0, 100), "category": "Performance",
        "help": "Same idea as the GPU cap, but for CPU usage.",
    },
    "max_ram_gb": {
        "path": "background.max_ram_gb", "kind": "int", "range": (1, 128), "category": "Performance",
        "help": "Background jobs pause if free system RAM drops below this "
                "many GB.",
    },
    "coding_effort": {
        "path": "coding.build.effort", "kind": "choice", "choices": ["min", "mid", "max"],
        "category": "Coding",
        "help": "How much the autonomous coding loop writes per pass - "
                "'min' is fast and minimal, 'max' is thorough but slower.",
    },
    "tools_max_iterations": {
        "path": "tools.max_iterations", "kind": "int", "range": (1, 500), "category": "Coding",
        "help": "How many tool calls (shell, file edits, etc.) the agent "
                "can chain in one turn before it has to stop and respond.",
    },
    "diagnose_first": {
        "path": "coding.diagnose_first", "kind": "bool", "category": "Coding",
        "help": "Before trying again after something breaks, have it "
                "explain what went wrong first. Usually fixes things "
                "faster; turn off to just retry blind.",
    },
    "max_repair_attempts": {
        "path": "coding.executor.max_repair_attempts", "kind": "int", "range": (1, 20),
        "category": "Coding",
        "help": "How many times it retries fixing broken code before "
                "giving up and telling you it's stuck.",
    },
    "sandbox_timeout_s": {
        "path": "coding.sandbox.timeout_seconds", "kind": "int", "range": (5, 600),
        "category": "Coding",
        "help": "How many seconds one test run gets before it's killed "
                "for taking too long.",
    },
    "sandbox_memory_mb": {
        "path": "coding.sandbox.memory_mb", "kind": "int", "range": (256, 16384),
        "category": "Coding",
        "help": "How much memory a test run can use before it's shut "
                "down. Raise this if your code legitimately needs more.",
    },
    "coding_auto_commit": {
        "path": "coding.auto_commit", "kind": "bool", "category": "Coding",
        "help": "Automatically save a snapshot whenever it successfully "
                "fixes or builds something, so you can look back at what "
                "changed. Turn off to review changes yourself first.",
    },
    "tier3_escalation": {
        "path": "coding.escalation.tier3_enabled", "kind": "bool", "category": "Coding",
        "help": "If it keeps failing to fix something the same way, let "
                "it switch to a much bigger, slower model as a last "
                "resort. Off by default because that bigger model "
                "probably isn't downloaded on your machine yet and is "
                "still unproven here - turn on only after you've pulled "
                "it.",
    },
    "tier3_model": {
        "path": "coding.escalation.tier3_model", "kind": "model", "category": "Coding",
        "help": "Which bigger model to switch to when the option above "
                "is on.",
    },
    "tier3_budget_s": {
        "path": "coding.escalation.total_budget_seconds", "kind": "int", "range": (30, 3600),
        "category": "Coding",
        "help": "How many seconds it's allowed to keep trying before "
                "giving up entirely, once escalation kicks in.",
    },
    "whatsapp_min_messages": {
        "path": "whatsapp.min_messages", "kind": "int", "range": (1, 100),
        "category": "WhatsApp", "private": True,
        "help": "The fewest messages a chat needs before it'll try to "
                "sum it up, learn about the person, or judge how a "
                "friendship is going. Raise this if short chats produce "
                "summaries that feel like a stretch.",
    },
    "safety_max_timeout_s": {
        "path": "safety.max_timeout_s", "kind": "int", "range": (5, 3600), "category": "Safety",
        "help": "Hard ceiling, in seconds, on how long any single shell "
                "command the agent runs is allowed to take before being "
                "killed.",
    },
    "safe_commands": {
        "path": "safety.safe_commands", "kind": "list", "category": "Safety",
        "help": "Extra shell commands you personally trust it to run "
                "without asking, on top of the built-in safe list.",
    },
    "blocked_patterns": {
        "path": "safety.blocked_patterns", "kind": "list", "category": "Safety",
        "help": "Extra text patterns that make it refuse to run a shell "
                "command, on top of the built-in blocked list.",
    },
}


def _get(config: dict, dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _private_module_available() -> bool:
    """A setting marked `"private": True` only does anything in a build that has the
    private WhatsApp analysis module (agent/wa_private.py).
    A public export that drops that file should not show a dead knob for it here."""
    try:
        from agent.wa import WA_PRIVATE_AVAILABLE
        return WA_PRIVATE_AVAILABLE
    except Exception:  # noqa: BLE001 - settings must stay usable either way
        return False


def _visible(meta: dict[str, Any]) -> bool:
    return not meta.get("private") or _private_module_available()


def current() -> list[tuple[str, str, str, Any]]:
    """(short_name, category, help, current_value) for every editable setting
    actually usable in this build (hides private-module-only settings when that
    module isn't installed)."""
    config = load_config()
    return [
        (name, meta["category"], meta["help"], _get(config, meta["path"]))
        for name, meta in EDITABLE.items() if _visible(meta)
    ]


def audit() -> list[str]:
    """Sanity-check every EDITABLE entry against what load_config() actually
    resolves. Catches the exact bug class this file has hit twice already
    (models.unrestricted, coding.diagnose_first): a setting referenced here
    whose real default only lives in some OTHER file's defaults dict (or
    nowhere at all), so it silently resolves to None until a user happens
    to set it once by hand. Returns a list of human-readable problems -
    empty means everything resolves cleanly."""
    config = load_config()
    problems = []
    for name, meta in EDITABLE.items():
        value = _get(config, meta["path"])
        if value is None:
            problems.append(
                f"'{name}' ({meta['path']}) resolves to None - add a default "
                f"for it in agent/config.py's DEFAULT_CONFIG."
            )
        elif meta["kind"] == "list" and not isinstance(value, list):
            problems.append(f"'{name}' ({meta['path']}) is kind='list' but resolves to {value!r}, not a list.")
    return problems


def categories() -> list[str]:
    seen: list[str] = []
    for meta in EDITABLE.values():
        if _visible(meta) and meta["category"] not in seen:
            seen.append(meta["category"])
    return seen


def in_category(category: str) -> list[tuple[str, dict[str, Any]]]:
    return [(name, meta) for name, meta in EDITABLE.items() if meta["category"] == category and _visible(meta)]


def _coerce(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def validate(name: str, value: str) -> tuple[Any, str | None]:
    """Coerce + validate against the setting's kind. Returns (coerced, error_or_None)."""
    meta = EDITABLE[name]
    coerced = _coerce(value)
    kind = meta["kind"]
    if kind == "bool" and not isinstance(coerced, bool):
        return None, "expected true/false"
    if kind in ("int", "percent") and not isinstance(coerced, int):
        return None, "expected a whole number"
    if kind in ("int", "percent") and "range" in meta:
        lo, hi = meta["range"]
        if not (lo <= coerced <= hi):
            return None, f"must be between {lo} and {hi}"
    if kind == "choice" and coerced not in meta["choices"]:
        return None, f"must be one of: {', '.join(meta['choices'])}"
    return coerced, None


def set_value(name: str, value: Any) -> tuple[str, Any]:
    """Set an editable setting in config.yaml. Returns (dotted_path, coerced).

    `value` is a `str` for every scalar kind (bool/int/choice/model/string/
    long_text) and goes through `validate()`'s string coercion as before.
    For `kind == "list"` it's already a real `list[str]` built by the list
    widget - coercing a list through `_coerce()` (which assumes a string)
    would be wrong, so that case is written through unvalidated.

    Raises ValueError (not KeyError) even for an unknown name: KeyError's
    __str__ wraps the message in an extra repr() layer of quotes when it
    carries a single string arg, which prints literal quote characters
    around otherwise-clean error text at every call site that does
    `console.print(f"...{exc}...")`.
    """
    if name not in EDITABLE:
        raise ValueError(f"unknown setting '{name}' (see: {brand.CMD} settings show)")
    meta = EDITABLE[name]
    if meta["kind"] == "list":
        if not isinstance(value, list):
            raise ValueError(f"'{name}' expects a list of items")
        coerced = value
    else:
        coerced, error = validate(name, value)
        if error:
            raise ValueError(f"invalid value for '{name}': {error}")
    dotted = meta["path"]
    path = Path(DEFAULT_CONFIG_PATH)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    node = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = coerced
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    return dotted, coerced


# Shared neon palette (agent/theme.py is dependency-free, so `agent settings` stays usable on its own).
from agent.theme import CYAN as _GOLD, DIM as _DIM, MAGENTA as _EMBER, YELLOW as _GLOW  # noqa: E402

_KIND_HINT = {
    "bool": "on/off", "choice": "pick one", "model": "pick a model",
    "long_text": "long text", "list": "list of items",
}


def _hint_for(meta: dict[str, Any]) -> str:
    if meta["kind"] in ("int", "percent") and "range" in meta:
        lo, hi = meta["range"]
        return f"{lo}-{hi}"
    return _KIND_HINT.get(meta["kind"], "text")


def _safe_input(console, prompt: str) -> str | None:
    """None means 'cancel/back' - covers an explicit blank/'back' AND a
    Ctrl+C or Ctrl+D landing mid-prompt, which previously propagated as an
    uncaught EOFError/KeyboardInterrupt and crashed out of the settings
    browser entirely instead of just backing up one level."""
    try:
        return console.input(prompt)
    except (EOFError, KeyboardInterrupt):
        console.print()  # land the cursor on its own line before the next prompt
        return None


def interactive(console) -> None:
    """Rich, menu-driven settings browser: pick a category, pick a setting,
    get the right input widget for its kind (toggle / range-checked number /
    a picker over installed Ollama models / choice list) instead of having
    to already know a setting's name and valid value."""
    cats = categories()
    while True:
        console.print(f"\n[bold {_GOLD}]Settings[/bold {_GOLD}]  [{_DIM}](pick a category)[/{_DIM}]\n")
        for i, cat in enumerate(cats, 1):
            n = len(in_category(cat))
            console.print(f"  [{_GLOW}]{i:2}[/{_GLOW}] {cat}  [{_DIM}]({n})[/{_DIM}]")
        console.print(f"  [{_DIM}] 0  back[/{_DIM}]")
        choice = _safe_input(console, "\n ▸ ")
        if choice is None or choice.strip().lower() in ("0", "", "b", "back"):
            return
        choice = choice.strip().lower()
        if not choice.isdigit() or not (1 <= int(choice) <= len(cats)):
            console.print(f"[{_EMBER}]invalid selection[/{_EMBER}]")
            continue
        _category_menu(console, cats[int(choice) - 1])


def _default_get(dotted: str) -> Any:
    """The shipped default for a dotted path, straight from DEFAULT_CONFIG -
    used by the reset-to-defaults action, independent of whatever's
    currently sitting in config.yaml."""
    from agent.config import DEFAULT_CONFIG
    return _get(DEFAULT_CONFIG, dotted)


def _reset_category(console, category: str, items: list[tuple[str, dict[str, Any]]]) -> None:
    ans = _safe_input(console, f"Reset every setting in [{_GOLD}]{category}[/{_GOLD}] back to its default? [y/N] ")
    if ans is None or ans.strip().lower() not in ("y", "yes"):
        console.print(f"[{_DIM}]cancelled[/{_DIM}]")
        return
    for name, meta in items:
        default = _default_get(meta["path"])
        if default is None:
            continue  # no shipped default to reset to - leave this one alone
        try:
            set_value(name, default if meta["kind"] == "list" else str(default))
        except ValueError as exc:
            console.print(f"[{_EMBER}]couldn't reset '{name}': {exc}[/{_EMBER}]")
    console.print(f"[{_GLOW}]{category} reset to defaults.[/{_GLOW}]")


def _category_menu(console, category: str) -> None:
    while True:
        config = load_config()
        items = in_category(category)
        console.print(f"\n[bold {_GOLD}]Settings ▸ {category}[/bold {_GOLD}]\n")
        for i, (name, meta) in enumerate(items, 1):
            value = _get(config, meta["path"])
            console.print(
                f"  [{_GLOW}]{i:2}[/{_GLOW}] [{_GOLD}]{name}[/{_GOLD}] = [{_EMBER}]{value}[/{_EMBER}]"
                f"  [{_DIM}]({_hint_for(meta)})[/{_DIM}]"
            )
            console.print(f"      [{_DIM}]{meta['help']}[/{_DIM}]")
        console.print(f"  [{_DIM}] r  reset this whole category to defaults[/{_DIM}]")
        console.print(f"  [{_DIM}] 0  back[/{_DIM}]")
        choice = _safe_input(console, "\n ▸ pick a setting to change: ")
        if choice is None or choice.strip().lower() in ("0", "", "b", "back"):
            return
        choice = choice.strip().lower()
        if choice == "r":
            _reset_category(console, category, items)
            continue
        if not choice.isdigit() or not (1 <= int(choice) <= len(items)):
            console.print(f"[{_EMBER}]invalid selection[/{_EMBER}]")
            continue
        name, meta = items[int(choice) - 1]
        _edit_setting(console, name, meta, config)


def _edit_setting(console, name: str, meta: dict[str, Any], config: dict) -> None:
    current_value = _get(config, meta["path"])
    kind = meta["kind"]
    new_value: Any

    if kind == "bool":
        proposed = not current_value
        ans = _safe_input(console, f"Flip [{_GOLD}]{name}[/{_GOLD}] to [{_EMBER}]{proposed}[/{_EMBER}]? [Y/n] ")
        if ans is None or ans.strip().lower() not in ("", "y", "yes"):
            console.print(f"[{_DIM}]cancelled[/{_DIM}]")
            return
        new_value = proposed

    elif kind == "choice":
        console.print(f"\nOptions for [{_GOLD}]{name}[/{_GOLD}]:")
        for i, c in enumerate(meta["choices"], 1):
            marker = f" [{_DIM}](current)[/{_DIM}]" if c == current_value else ""
            console.print(f"  {i}. {c}{marker}")
        raw = _safe_input(console, " ▸ pick a number: ")
        if raw is None or not raw.strip():
            console.print(f"[{_DIM}]cancelled[/{_DIM}]")
            return
        raw = raw.strip()
        if not raw.isdigit() or not (1 <= int(raw) <= len(meta["choices"])):
            console.print(f"[{_EMBER}]invalid[/{_EMBER}]")
            return
        new_value = meta["choices"][int(raw) - 1]

    elif kind == "model":
        clear_hint = ", or '-' to clear it blank" if current_value else ""
        from agent.llm import OllamaClient
        models = OllamaClient().list_models()
        if not models:
            console.print(f"[{_EMBER}]Couldn't reach Ollama to list installed models - type one manually.[/{_EMBER}]")
            raw = _safe_input(console, f" ▸ new value for {name}{clear_hint}: ")
            if raw is None or not raw.strip():
                console.print(f"[{_DIM}]cancelled[/{_DIM}]")
                return
            new_value = "" if raw.strip() == "-" else raw.strip()
        else:
            console.print(f"\nInstalled models for [{_GOLD}]{name}[/{_GOLD}]:")
            for i, m in enumerate(models, 1):
                marker = f" [{_DIM}](current)[/{_DIM}]" if m == current_value else ""
                console.print(f"  {i:2}. {m}{marker}")
            raw = _safe_input(console, f" ▸ pick a number, type a model name{clear_hint}: ")
            if raw is None or not raw.strip():
                console.print(f"[{_DIM}]cancelled[/{_DIM}]")
                return
            raw = raw.strip()
            if raw == "-":
                new_value = ""
            elif raw.isdigit() and 1 <= int(raw) <= len(models):
                new_value = models[int(raw) - 1]
            else:
                new_value = raw

    elif kind == "long_text":
        from agent.textio import read_multiline
        console.print(f"\n[{_DIM}]current value:[/{_DIM}]\n{current_value}\n")
        new_value = read_multiline(console, f" ▸ new value for {name} (paste, blank to cancel): ")
        if not new_value.strip():
            console.print(f"[{_DIM}]cancelled[/{_DIM}]")
            return

    elif kind == "list":
        items = list(current_value or [])
        while True:
            console.print(f"\n[{_GOLD}]{name}[/{_GOLD}] has {len(items)} item(s):")
            for i, it in enumerate(items, 1):
                console.print(f"  {i:2}. {it}")
            if not items:
                console.print(f"  [{_DIM}](empty)[/{_DIM}]")
            console.print(f"\n  [{_DIM}]a) add   r) remove   d) done[/{_DIM}]")
            sub = _safe_input(console, " ▸ ")
            if sub is None or sub.strip().lower() in ("d", "done", ""):
                break
            sub = sub.strip().lower()
            if sub == "a":
                new_item = _safe_input(console, " ▸ new item: ")
                if new_item and new_item.strip():
                    items.append(new_item.strip())
            elif sub == "r":
                idx = _safe_input(console, " ▸ remove which number: ")
                if idx and idx.strip().isdigit() and 1 <= int(idx.strip()) <= len(items):
                    items.pop(int(idx.strip()) - 1)
        new_value = items

    elif kind in ("int", "percent"):
        lo, hi = meta.get("range", (None, None))
        hint = f" ({lo}-{hi})" if lo is not None else ""
        raw = _safe_input(console, f" ▸ new value for {name}{hint} [current: {current_value}]: ")
        if raw is None or not raw.strip():
            console.print(f"[{_DIM}]cancelled[/{_DIM}]")
            return
        raw = raw.strip()
        try:
            candidate = int(raw)
        except ValueError:
            console.print(f"[{_EMBER}]not a whole number[/{_EMBER}]")
            return
        if lo is not None and not (lo <= candidate <= hi):
            console.print(f"[{_EMBER}]must be between {lo} and {hi}[/{_EMBER}]")
            return
        new_value = candidate

    else:
        raw = _safe_input(console, f" ▸ new value for {name} [current: {current_value}]: ")
        if raw is None or not raw.strip():
            console.print(f"[{_DIM}]cancelled[/{_DIM}]")
            return
        new_value = raw.strip()

    try:
        dotted, coerced = set_value(name, new_value if kind == "list" else str(new_value))
    except ValueError as exc:
        # Reachable in principle even after the widget-level checks above
        # (e.g. a future kind added without a matching validate() branch) -
        # this must never crash the whole settings browser.
        console.print(f"[{_EMBER}]{exc}[/{_EMBER}]")
        return
    console.print(f"[{_GLOW}]{dotted}: {current_value!r} → {coerced!r} saved.[/{_GLOW}]")
