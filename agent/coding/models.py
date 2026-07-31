"""List installed Ollama models and swap the coding executor.

The coder model is uncensored + swappable. We surface what's installed plus a
short list of recommended abliterated (uncensored) coders so the menu can offer
them even before they're pulled.
"""

from __future__ import annotations

import httpx

from agent.coding import config as coding_config

OLLAMA_URL = "http://localhost:11434"

# Uncensored (abliterated) coders — resolves "uncensored vs best-coder": both.
RECOMMENDED = [
    ("huihui_ai/qwen2.5-coder-abliterate:7b", "uncensored coder · ~4.7GB · fast · 32K ctx (default)"),
    ("huihui_ai/qwen2.5-coder-abliterate:14b", "uncensored coder · ~9GB · better multi-file"),
    ("huihui_ai/qwen3-coder-abliterated:30b", "uncensored · 256K ctx · agentic · CPU-offload, slow"),
]


def installed() -> list[str]:
    """Model tags currently pulled in Ollama."""
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:  # noqa: BLE001 - Ollama down → empty list, caller reports
        return []


def current(config: dict | None = None) -> str:
    return coding_config.load(config)["executor"]["model"]


def set_model(name: str) -> None:
    """Persist the coding executor model to config.yaml under coding.executor.model."""
    import yaml

    from agent.config import DEFAULT_CONFIG_PATH, load_config

    cfg = load_config()
    coding = cfg.setdefault("coding", {})
    coding.setdefault("executor", {})["model"] = name
    DEFAULT_CONFIG_PATH.write_text(yaml.dump(cfg, sort_keys=False))


_EFFORTS = [
    ("min", "smallest single-file solution, minimal test"),
    ("mid", "balanced: program file + a thorough test"),
    ("max", "thorough: may split into a few files, edge-case tests"),
]


def set_effort(level: str) -> None:
    """Persist coding.build.effort (min|mid|max) to config.yaml."""
    import yaml

    from agent.config import DEFAULT_CONFIG_PATH, load_config

    cfg = load_config()
    cfg.setdefault("coding", {}).setdefault("build", {})["effort"] = level
    DEFAULT_CONFIG_PATH.write_text(yaml.dump(cfg, sort_keys=False))


def choose_effort(console) -> None:
    """Menu hook: pick how much the agent writes (min / mid / max)."""
    cur = coding_config.load(_user_config()).get("build", {}).get("effort", "mid")
    console.print(f"[#5a5a8a]current effort:[/] [#05d9e8]{cur}[/]")
    for i, (level, note) in enumerate(_EFFORTS, 1):
        mark = " [#39ff14]◀ current[/]" if level == cur else ""
        console.print(f"   [#39ff14]{i}[/] [#05d9e8]{level}[/]  [#5a5a8a]{note}[/]{mark}")
    raw = console.input("  [#ff2a6d]▸[/] pick min/mid/max (or 1-3): ").strip().lower()
    pick = {"1": "min", "2": "mid", "3": "max"}.get(raw, raw if raw in {"min", "mid", "max"} else "")
    if pick:
        set_effort(pick)
        console.print(f"[#39ff14]effort set to[/] [#05d9e8]{pick}[/]")


def _user_config():
    from agent.config import load_config
    return load_config()


def menu_choose(console) -> None:
    """Menu hook: show installed + recommended models, let the user pick one."""
    have = set(installed())
    cur = current()
    rows: list[str] = []
    console.print(f"[#5a5a8a]current coding model:[/] [#05d9e8]{cur}[/]")
    console.print("[#5a5a8a]recommended (uncensored coders):[/]")
    for i, (tag, note) in enumerate(RECOMMENDED, 1):
        mark = "[#39ff14]✓ installed[/]" if tag in have else "[#5a5a8a]not pulled — `ollama pull " + tag + "`[/]"
        rows.append(tag)
        console.print(f"   [#39ff14]{i}[/] [#05d9e8]{tag}[/]  [#5a5a8a]{note}[/]  {mark}")
    extra = [m for m in sorted(have) if m not in {t for t, _ in RECOMMENDED}]
    for j, tag in enumerate(extra, len(RECOMMENDED) + 1):
        rows.append(tag)
        console.print(f"   [#39ff14]{j}[/] [#05d9e8]{tag}[/]  [#5a5a8a]installed[/]")
    raw = console.input("  [#ff2a6d]▸[/] pick a number (or type a tag): ").strip()
    if not raw:
        return
    tag = rows[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(rows) else raw
    set_model(tag)
    console.print(f"[#39ff14]coding model set to[/] [#05d9e8]{tag}[/]")
    # Free the GPU: unload every model currently resident so the newly-picked one
    # loads alone next run (only one model in VRAM at a time).
    try:
        from agent.llm import OllamaClient
        freed = OllamaClient().unload_all()
        if freed:
            console.print(f"[#5a5a8a]unloaded from memory:[/] {', '.join(freed)}")
    except Exception:  # noqa: BLE001 - swap must not fail if Ollama is momentarily down
        pass
    if tag not in have:
        console.print(f"[#5a5a8a]not pulled yet — run:[/] ollama pull {tag}")
