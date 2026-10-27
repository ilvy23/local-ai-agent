"""One GPU or both? Asked once per run, with the context size each option allows.

Some work (summarising several pages, long documents) needs a big context window,
and a big context is mostly KV cache - VRAM. With two cards the Vulkan Ollama server
(see agent.llm) can use both, which leaves room for a much longer context than one
card does. So when you start a command that talks to a model, you are asked which
option you want and told what context each option roughly allows for your models.

  AGENT_GPU_MODE   ask (default) | single | dual   - skip the question
  AGENT_CHAT_CTX   context tokens for the chat to use with AGENT_GPU_MODE set
  AGENT_SINGLE_KV  KV cache type of the default server (default f16)
  AGENT_DUAL_KV    KV cache type of the Vulkan server (default q8_0)

All numbers are *estimates* from the model's architecture and the free VRAM; Ollama's
own placement has the last word.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent import osport
from typing import Any

SINGLE = "single"
DUAL = "dual"

# bytes per KV-cache element (f16, q8_0 = 8.5 bit, q4_0 = 4.5 bit)
KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
_RESERVE_MB = 700          # compute buffers etc. per GPU
_DUAL_EFFICIENCY = 0.92    # a layer split across two cards never fills both perfectly
_CTX_STEP = 1024
_CTX_CAP = 262144

# process-wide state, set by ensure_choice()
_state: dict[str, Any] = {"mode": None, "ctx": None}
_facts_cache: dict[str, dict[str, Any]] = {}   # model -> facts seen while building the table
_auto_mode_cache: dict[str, Any] = {"done": False, "value": None}   # memoised auto_mode() result

# the models of these config roles are listed in the table (embedding models never are)
_ROLES = ("chat", "main", "background", "coding", "reasoning", "unrestricted", "orchestrator", "fallback")


def reset() -> None:
    _state["mode"] = None
    _state["ctx"] = None
    _facts_cache.clear()
    _auto_mode_cache["done"] = False
    _auto_mode_cache["value"] = None


def auto_mode(gpus: list["Gpu"] | None = None) -> str | None:
    """The GPU mode hardware detection alone would pick, with nothing else (env/history/the
    interactive question) having decided it: DUAL when ≥2 Vulkan-capable GPUs were detected
    (so the dual-GPU Vulkan server in agent.llm gets used automatically), SINGLE when exactly
    one was, or None on a CPU-only / no-GPU machine (the caller's own default applies - for
    agent.llm.wants_vulkan that's the model-name pattern, unchanged).

    Memoised per process (like a real detection round costs a subprocess call); pass `gpus`
    explicitly, or call `reset()`, to bypass the cache (tests do both)."""
    use_cache = gpus is None
    if use_cache and _auto_mode_cache["done"]:
        return _auto_mode_cache["value"]
    if gpus is None:
        gpus = detect_gpus()
    capable = [g for g in gpus if g.vulkan_capable]
    value = DUAL if len(capable) >= 2 else (SINGLE if len(capable) == 1 else None)
    if use_cache:
        _auto_mode_cache["done"] = True
        _auto_mode_cache["value"] = value
    return value


def current_mode() -> str | None:
    """The GPU mode in effect: this run's choice, else AGENT_GPU_MODE, else what hardware
    detection alone would pick (auto_mode), else None (nothing to go on)."""
    if _state["mode"]:
        return _state["mode"]
    env = os.environ.get("AGENT_GPU_MODE", "").strip().lower()
    if env in (SINGLE, DUAL):
        return env
    return auto_mode()


def chat_ctx_override() -> int | None:
    """Context tokens chosen for the chat (None = keep the configured default)."""
    if _state["ctx"]:
        return int(_state["ctx"])
    raw = os.environ.get("AGENT_CHAT_CTX", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def effective_ctx(model: str | None, default: int) -> int:
    """Context tokens for a chat turn with `model`: the run's choice, but never more than
    that model's own maximum (a small 8B model and a 27B have different limits). With no
    choice made, `default` (the configured models.chat_num_ctx) is returned unchanged."""
    chosen = chat_ctx_override()
    if not chosen:
        return default
    facts = _facts_cache.get(model or "")
    if facts and facts.get("ctx_max"):
        return max(2048, min(chosen, int(facts["ctx_max"])))
    return chosen


# ---------------------------------------------------------------------------- GPUs


#   0x1002 AMD, 0x8086 Intel - the PCI vendor IDs sysfs reports under /sys/class/drm/*/device/vendor
_VENDOR_IDS = {"0x1002": "amd", "0x8086": "intel"}
# Vulkan-capable inference backends exist for all three; CPU-only has none.
VULKAN_CAPABLE_VENDORS = frozenset({"amd", "nvidia", "intel"})


@dataclass
class Gpu:
    name: str
    vendor: str          # "amd" | "nvidia" | "intel"
    total_mb: int
    used_mb: int

    @property
    def free_mb(self) -> int:
        return max(self.total_mb - self.used_mb, 0)

    @property
    def vulkan_capable(self) -> bool:
        return self.vendor in VULKAN_CAPABLE_VENDORS


def _sysfs_vram_mb(dev: Path) -> tuple[int, int] | None:
    """(total_mb, used_mb) from the AMD-style sysfs VRAM counters, or None if unavailable.

    Intel's discrete cards (Arc, via the xe/i915 driver) expose the same two files on recent
    kernels; integrated Intel GPUs normally do not, since they share system RAM instead of
    having their own VRAM pool - that's a cheap, resilient way to skip iGPUs without special-
    casing them by name."""
    try:
        total = int((dev / "mem_info_vram_total").read_text()) // (1024 * 1024)
        used = int((dev / "mem_info_vram_used").read_text()) // (1024 * 1024)
        return total, used
    except (OSError, ValueError):
        return None


def detect_gpus(drm_root: Path | str = "/sys/class/drm") -> list[Gpu]:
    """Discrete GPUs with their VRAM: AMD/Intel via sysfs, NVIDIA via nvidia-smi.

    Resilient by design: any single detection step that errors, times out, or reports a
    non-discrete/too-small device is skipped rather than raising - worst case this returns []
    (the CPU-only / no-GPU case), never a crash. `drm_root` lets tests point this at a fake
    sysfs tree instead of the real one."""
    gpus: list[Gpu] = []
    try:
        cards = sorted(Path(drm_root).glob("card[0-9]*"))
    except OSError:
        cards = []
    for card in cards:
        if not re.fullmatch(r"card\d+", card.name):
            continue
        dev = card / "device"
        try:
            vendor_id = (dev / "vendor").read_text().strip()
        except OSError:
            continue
        vendor = _VENDOR_IDS.get(vendor_id)
        if vendor is None:
            continue
        vram = _sysfs_vram_mb(dev)
        if vram is None:
            continue  # e.g. an Intel iGPU with no dedicated VRAM counters - not a usable pool
        total, used = vram
        if total < 2048:  # carve-outs / tiny pools are not worth offering
            continue
        label = "AMD GPU" if vendor == "amd" else "Intel GPU"
        name = f"{label} {round(total / 1024)} GB"
        try:
            name = (dev / "product_name").read_text().strip() or name
        except OSError:
            pass
        gpus.append(Gpu(name, vendor, total, used))
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            for line in out.strip().splitlines():
                name, total, used = [x.strip() for x in line.split(",")]
                short = name.replace("NVIDIA GeForce ", "")
                gpus.append(Gpu(f"{short} {round(int(total) / 1024)} GB", "nvidia", int(total), int(used)))
        except (OSError, subprocess.SubprocessError, ValueError, TimeoutError):
            pass
    return gpus


def gpu_mode_available(gpus: list[Gpu] | None = None) -> bool:
    """True if at least one detected GPU can run a Vulkan-capable inference backend here.
    False (CPU-only / no GPU, or a vendor Vulkan can't help with) means the caller should
    stick to whatever default/CPU backend it already has - never crash for lack of a GPU."""
    gpus = detect_gpus() if gpus is None else gpus
    return any(g.vulkan_capable for g in gpus)


# ---------------------------------------------------------------------------- estimate


@dataclass
class Estimate:
    ctx: int             # tokens that fit next to the weights (0 = does not fit)
    fits: bool           # model weights fit completely
    over_mb: int         # how much of the weights would not fit (when fits is False)
    limited_by_model: bool  # ctx was capped by the model's own maximum
    approx: bool = False    # KV size had to be guessed (conservative upper bound)


def estimate_context(facts: dict[str, Any], usable_mb: float, kv_type: str) -> Estimate:
    """How many tokens of context fit next to the weights in `usable_mb` of VRAM."""
    weights = float(facts["size_mb"])
    approx = bool(facts.get("approx"))
    left = usable_mb - weights
    if left <= 0:
        return Estimate(0, False, int(-left), False, approx)
    kv_mb_per_token = facts["kv_elems_per_token"] * KV_BYTES.get(kv_type, 2.0) / (1024 * 1024)
    if kv_mb_per_token <= 0:
        return Estimate(0, True, 0, False, approx)
    raw = int(left / kv_mb_per_token)
    cap = int(facts.get("ctx_max") or _CTX_CAP)
    ctx = min(raw, cap, _CTX_CAP)
    ctx = (ctx // _CTX_STEP) * _CTX_STEP
    return Estimate(ctx, True, 0, raw >= cap, approx)


@dataclass
class Option:
    mode: str
    label: str
    gpus: list[Gpu]
    usable_mb: float
    kv_type: str


def build_options(gpus: list[Gpu], resident_mb: int = 0) -> list[Option]:
    """The offered choices. `resident_mb` is VRAM held by Ollama models right now: they
    are unloaded before the chosen option runs, so it counts as free."""
    amd = [g for g in gpus if g.vendor == "amd"]
    first = amd[0] if amd else (gpus[0] if gpus else None)
    if first is None:
        return []
    other_use = max(first.used_mb - resident_mb, 0)
    single_free = max(first.total_mb - other_use - _RESERVE_MB, 0)
    options = [Option(SINGLE, first.name, [first], single_free,
                      os.environ.get("AGENT_SINGLE_KV", "f16"))]
    if len(gpus) > 1:
        # resident models may sit on any card; spread what we know over all of them
        extra = resident_mb - (first.used_mb - other_use)
        free = 0.0
        for g in gpus:
            used = g.used_mb - (resident_mb if g is first else 0)
            if g is not first and extra > 0:
                used = max(g.used_mb - extra, 0)
            free += max(g.total_mb - max(used, 0) - _RESERVE_MB, 0)
        names = " + ".join(g.name for g in gpus)
        options.append(Option(DUAL, names, list(gpus), free * _DUAL_EFFICIENCY,
                              os.environ.get("AGENT_DUAL_KV", "q8_0")))
    return options


def _state_file() -> Path:
    base = str(osport.state_home())
    return Path(base) / "agent" / "gpu_mode"


def last_mode() -> str | None:
    try:
        value = _state_file().read_text().strip()
    except OSError:
        return None
    return value if value in (SINGLE, DUAL) else None


def _remember(mode: str) -> None:
    try:
        path = _state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mode + "\n")
    except OSError:
        pass


def _fmt_ctx(est: Estimate) -> str:
    if not est.fits:
        return f"passt nicht ganz rein (~{est.over_mb / 1024:.1f} GB zu viel -> teils CPU, langsam)"
    if est.ctx < 2048:
        return "kaum Platz für Kontext"
    suffix = " (Maximum des Modells)" if est.limited_by_model else ""
    if est.approx:
        suffix += " (vorsichtig geschätzt)"
    return f"~{est.ctx:,} Tokens{suffix}".replace(",", ".")


def models_to_show(client: Any, config: dict[str, Any]) -> list[str]:
    """Every model the agent is configured to use (chat first), plus the biggest installed
    model if that is another one - so the table covers 8B models as well as a 27B."""
    models = (config or {}).get("models") or {}
    sizes = client.installed_sizes() if hasattr(client, "installed_sizes") else {}
    skip = {models.get("embed")} if models.get("embed") else set()
    sizes = {k: v for k, v in sizes.items() if k not in skip and "embed" not in k and "bge" not in k}
    chosen: list[str] = []
    for role in _ROLES:
        name = models.get(role)
        if name and name not in chosen and name not in skip and (not sizes or name in sizes):
            chosen.append(name)
    if sizes:
        biggest = max(sizes, key=sizes.get)
        if biggest not in chosen:
            chosen.append(biggest)
    return chosen


def table_rows(client: Any, config: dict[str, Any], options: list[Option]) -> list[tuple[str, list[Estimate | None]]]:
    rows = []
    configured = {m for m in ((config or {}).get("models") or {}).values() if isinstance(m, str)}
    for model in models_to_show(client, config):
        facts = client.model_facts(model) if hasattr(client, "model_facts") else None
        if facts:
            _facts_cache[model] = facts
        ests = [estimate_context(facts, o.usable_mb, o.kv_type) if facts else None for o in options]
        # configured models are always listed; an extra biggest-installed model only if it fits somewhere
        if model not in configured and not any(e and e.fits for e in ests):
            continue
        rows.append((model, ests))
    return rows


# ---------------------------------------------------------------------------- choice


def _apply(mode: str, ctx: int | None) -> None:
    _state["mode"] = mode
    _state["ctx"] = ctx


def ensure_choice(
    client: Any,
    config: dict[str, Any] | None = None,
    console: Any = None,
    ask: Callable[[str, list[str], str], str] | None = None,
    ask_int: Callable[[str, int], int] | None = None,
    interactive: bool | None = None,
    gpus: list[Gpu] | None = None,
) -> str | None:
    """Decide the GPU mode for this run (once). Returns the mode or None if undecided."""
    if _state["mode"]:
        return _state["mode"]
    config = config or {}
    env = os.environ.get("AGENT_GPU_MODE", "").strip().lower()
    if env in (SINGLE, DUAL):
        _apply(env, chat_ctx_override())
        return env
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    gpus = detect_gpus() if gpus is None else gpus
    if len(gpus) < 2:                      # nothing to choose between
        return None
    if not interactive:
        previous = last_mode()
        if previous:
            _apply(previous, None)
        return previous

    resident = client.vram_in_use_mb() if hasattr(client, "vram_in_use_mb") else 0
    options = build_options(gpus, resident)
    rows = table_rows(client, config, options)

    if console is None:
        from rich.console import Console

        console = Console()
    from rich.table import Table

    table = Table(title="Wie sollen die Modelle laufen?", title_justify="left", show_lines=False)
    table.add_column("")
    for i, o in enumerate(options, 1):
        table.add_column(f"[{i}] {'nur ' if o.mode == SINGLE else 'beide: '}{o.label}")
    table.add_row("Frei (geschätzt)", *[f"{o.usable_mb / 1024:.1f} GB" for o in options])
    for model, ests in rows:
        table.add_row(model, *[_fmt_ctx(e) if e else "unbekannt" for e in ests])
    console.print(table)
    console.print("[dim]Schätzung aus Modellgröße, freiem VRAM und KV-Cache-Typ (Single: "
                  f"{options[0].kv_type}" + (f", beide: {options[1].kv_type}" if len(options) > 1 else "") + ").[/dim]")

    previous = last_mode()
    biggest_fits_single = True
    if rows:
        est = rows[-1][1][0]
        biggest_fits_single = bool(est and est.fits and est.ctx >= 16384)
    default = previous or (SINGLE if biggest_fits_single else DUAL)
    if ask is None:
        from rich.prompt import Prompt

        def ask(q: str, choices: list[str], dflt: str) -> str:  # type: ignore[misc]
            return Prompt.ask(q, choices=choices, default=dflt, console=console)
    answer = ask("Option", [str(i) for i in range(1, len(options) + 1)],
                 str([o.mode for o in options].index(default) + 1))
    picked = options[int(answer) - 1]

    ctx = None
    chat_model = rows[0] if rows else None
    if chat_model and chat_model[1][int(answer) - 1] and chat_model[1][int(answer) - 1].fits:
        est = chat_model[1][int(answer) - 1]
        recommended = max(est.ctx, 4096)
        current = int((config.get("models") or {}).get("chat_num_ctx", 98304))
        if ask_int is None:
            from rich.prompt import IntPrompt

            def ask_int(q: str, dflt: int) -> int:  # type: ignore[misc]
                return IntPrompt.ask(q, default=dflt, console=console)
        ctx = ask_int(f"Kontext für den Chat in Tokens (Empfehlung {recommended:,}, bisher {current:,})".replace(",", "."),
                      recommended)
        ctx = max(2048, min(int(ctx), _CTX_CAP))
        if ctx > recommended * 1.1:
            # Past what fits, layers spill onto the CPU and decode speed is bound by RAM
            # bandwidth. Measured on a 27B model split across two consumer GPUs (q8_0 KV): decode speed
            # falls noticeably once the context no longer fits in VRAM.
            console.print(f"[yellow]Achtung:[/yellow] {ctx:,} liegt über der Schätzung ({recommended:,}) - "
                          "die Schätzung ist vorsichtig; wenn das Modell nicht mehr komplett in den VRAM passt, "
                          "wandern Layer auf die CPU und es wird deutlich langsamer.".replace(",", "."))
    _apply(picked.mode, ctx)
    _remember(picked.mode)
    console.print(f"[green]{'Beide GPUs' if picked.mode == DUAL else 'Eine GPU'}[/green]"
                  + (f", Kontext {ctx:,}".replace(",", ".") if ctx else "") + ".")
    return picked.mode
