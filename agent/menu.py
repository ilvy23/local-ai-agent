"""Interactive nested menu — a friendly launcher over the CLI.

Reuses the Typer app in-process (no subprocess, no duplicated logic): each leaf
choice is an argv the app already knows how to run. Submenus nest; `{...}`
placeholders prompt for a value first. Lantern theme — warm candlelight on
black, because a local tool you live in should feel like a cosy little light.
"""

from __future__ import annotations

import random
import time

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ── Lantern palette · warm candlelight on black ─────────────────────────────
_FLAME = "#ffb454"   # amber gold   · lantern body, headings
_GLOW  = "#ffe1a8"   # soft glow    · numbers, highlights
_EMBER = "#ff8c42"   # deep ember   · prompts, accents
_WARM  = "#f0c489"   # candlelight  · labels, body text
_DIMW  = "#8a6b45"   # dim amber    · hints
_BG    = "#0b0805"   # warm near-black background
# Older lines still use the neon names — alias them onto the warm palette so the
# whole menu re-themes for free.
_CYAN, _PINK, _GREEN, _PURPLE, _DIM = _WARM, _EMBER, _GLOW, _FLAME, _DIMW


def cmd(argv: list[str]):
    return ("cmd", argv)


def sub(menu):
    return ("menu", menu)


def call(fn):
    return ("call", fn)


def _pick_from(console: Console, rows, prompt: str):
    """Show a numbered list of (label, value) and return the chosen value, a
    typed free-text value, or None to cancel."""
    if not rows:
        return console.input(f"  [{_PINK}]▸[/] {prompt}: ").strip() or None
    for i, (label, _v) in enumerate(rows, 1):
        console.print(f"   [{_GREEN}]{i:2}[/] [{_CYAN}]{label}[/]")
    raw = console.input(f"  [{_PINK}]▸[/] {prompt} (number, or type): ").strip()
    if not raw:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(rows):
        return rows[int(raw) - 1][1]
    return raw


def _about(console: Console) -> None:
    """A short, friendly 'what is this' screen — keeps the menu from feeling hollow."""
    from agent import main

    body = Text()
    body.append("✦ lantern", style=f"bold {_FLAME}")
    body.append(f"  v{main.__version__}\n", style=_DIM)
    body.append("A fully-local AI agent. Everything runs against your own\n", style=_CYAN)
    body.append("Ollama — no account, no cloud, nothing leaves your machine.\n\n", style=_CYAN)
    for line in (
        ("remembers", "past chats + facts about you, recalled semantically"),
        ("uses your PC", "shell, files, system stats — behind a safety gate"),
        ("searches the web", "end a chat message with /web for live, cited answers"),
        ("stays yours", "one local SQLite file; delete it and it's gone"),
    ):
        body.append(f"  ● {line[0]}", style=f"bold {_GREEN}")
        body.append(f" — {line[1]}\n", style=_DIM)
    body.append("\nTip: ", style=f"bold {_PINK}")
    body.append("type /web after a question in chat to search the internet.", style=_DIM)
    console.print(Panel(body, border_style=_PURPLE, title=f"[{_GREEN}]ABOUT[/]"))


# --- submenus ---
_CHAT = [("◢ CHAT", [
    ("Start a new chat", cmd(["chat"])),
    ("Resume last chat", cmd(["resume"])),
    ("Past chat sessions", cmd(["sessions"])),
])]

_MEMORY = [("◢ MEMORY", [
    ("Facts it remembers about you", cmd(["memory", "list"])),
    ("Search facts  ▸ query", cmd(["memory", "search", "{query}"])),
    ("Add a fact  ▸ text", cmd(["memory", "add", "{text}"])),
    ("Forget a fact  ▸ id", cmd(["memory", "forget", "{id}"])),
    ("Prune junk facts (paths, tool output)", cmd(["memory", "prune"])),
])]

_SETTINGS = [("◢ SETTINGS", [
    ("View all settings", cmd(["settings", "show"])),
    ("Change a setting  ▸ name, value", cmd(["settings", "set", "{setting}", "{value}"])),
    ("Re-embed memory with a new model  ▸ model", cmd(["reembed", "{model}"])),
])]

def _from_menu_models(console: Console) -> None:
    from agent.coding import models
    models.menu_choose(console)


def _from_menu_effort(console: Console) -> None:
    from agent.coding import models
    models.choose_effort(console)


_CODING = [("◢ CODING  ·  run agent from inside your project's git repo", [
    ("Autonomous build/fix  ▸ goal (it plans, writes, tests, debugs)", cmd(["code", "{goal}"])),
    ("Fix code against tests  ▸ task, file", cmd(["code", "{task}", "-f", "{codefile}"])),
    ("Fix (specific test target)  ▸ task, file, test",
     cmd(["code", "{task}", "-f", "{codefile}", "-t", "{codetest}"])),
    ("Swap the coding model (uncensored coders)", call(_from_menu_models)),
    ("How much to write  ▸ min / mid / max", call(_from_menu_effort)),
    ("Benchmark the fix loop (eval)  ▸ seeds", cmd(["code-eval", "--seeds", "{seeds}"])),
    ("Benchmark the autonomous loop (multi-file, slow)  ▸ trials",
     cmd(["code-eval", "--agentic", "--trials", "{trials}"])),
])]

_SYSTEM = [("◢ SYSTEM", [
    ("Live status panel (machine + Ollama)", cmd(["panel"])),
    ("Is background work paused? (governor)", cmd(["governor"])),
    ("Audit trail — what the agent ran", cmd(["audit"])),
    ("Background worker · status", cmd(["worker", "status"])),
    ("Background worker · start", cmd(["worker", "start"])),
    ("Background worker · stop", cmd(["worker", "stop"])),
    ("Settings  ▸", sub(_SETTINGS)),
])]

_MAIN = [
    ("◢ MENU  ·  pick a number", [
        ("Chat  ▸", sub(_CHAT)),
        ("Coding  ▸  (fix code against your tests)", sub(_CODING)),
        ("Memory  ▸", sub(_MEMORY)),
        ("System  ▸", sub(_SYSTEM)),
        ("Settings  ▸", sub(_SETTINGS)),
        ("About & help", call(_about)),
    ]),
]

# A symmetric hanging lantern: ring, domed cap, glass with a radial glow core
# (░▒▓█), a footed base, a finial. Every row mirrors around the centre column.
_LANTERN_ART = [
    "        ⊙        ",
    "       ╱ ╲       ",
    "      ╭───╮      ",
    "     ╭┴───┴╮     ",
    "    ╭─┴───┴─╮    ",
    "   ╭─┴─────┴─╮   ",
    "   │ ╭─────╮ │   ",
    "   │ │░▒▓▒░│ │   ",
    "  ·│ │▒▓█▓▒│ │·  ",
    "   │ │▓███▓│ │   ",
    "  ·│ │▒▓█▓▒│ │·  ",
    "   │ │░▒▓▒░│ │   ",
    "   │ ╰─────╯ │   ",
    "   ╰─┬─────┬─╯   ",
    "     ╰──┬──╯     ",
    "        ╨        ",
]

# glow chars → (dim fg, hot fg, halo-strength) — the halo is a warm background
# that makes the flame's light bleed into the black around it.
_GLOW_CHARS = {
    "░": ("#2a1a08", "#d0965a", 0.16),
    "▒": ("#341f08", "#ffa863", 0.19),
    "▓": ("#3e260a", "#ffc06a", 0.22),
    "█": ("#4a2e0c", "#fff0d2", 0.26),
}
_HALO_HOT, _FRAME_DIM = "#2c1a06", "#6a4a1e"


def _blend(a: str, b: str, t: float) -> str:
    """Interpolate two #rrggbb colours (t=0 → a, t=1 → b)."""
    t = max(0.0, min(1.0, t))
    ar = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    br = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(ar[i] + (br[i] - ar[i]) * t):02x}" for i in range(3))


def _lantern(t: float = 1.0) -> Text:
    """The lantern at glow intensity t∈[0,1]. Glow cells get a warm background
    halo so the light appears to shine out into the dark. NOTE: no per-line
    justify — the rows are fixed-width and left-aligned so the box stays true;
    the whole block is centred by the caller (Align.center)."""
    out = Text()
    for row in _LANTERN_ART:
        for ch in row:
            if ch in _GLOW_CHARS:
                dim, hot, halo = _GLOW_CHARS[ch]
                out.append(ch, style=f"bold {_blend(dim, hot, t)} on {_blend(_BG, _HALO_HOT, halo * t)}")
            elif ch == "·":
                out.append(ch, style=_blend("#241a0c", _EMBER, t))
            elif ch == " ":
                out.append(" ")
            elif ch in ("⊙", "╨"):
                out.append(ch, style=f"bold {_blend(_FRAME_DIM, _GLOW, t)}")
            else:
                out.append(ch, style=_blend(_FRAME_DIM, _FLAME, 0.4 + 0.6 * t))
        out.append("\n")
    return out


def _wordmark(t: float = 1.0) -> Text:
    """'l a n t e r n' with a per-letter gradient on a warm glow bar."""
    grad = [_EMBER, _FLAME, _GLOW, _GLOW, _GLOW, _FLAME, _EMBER]
    bg = _blend(_BG, "#2a1806", 0.7 * t)
    wm = Text(justify="center")
    wm.append("\n")
    for ch, col in zip("lantern", grad):
        wm.append(" " + ch, style=f"bold {_blend(_FRAME_DIM, col, t)} on {bg}")
    wm.append(" \n", style=f"on {bg}")
    if t > 0.6:
        wm.append("·  carry your own light  ·\n", style=f"italic {_DIMW}")
    return wm


def _banner_group(lt: float = 1.0, wt: float = 1.0):
    """Lit lantern + wordmark, each centred independently so they line up."""
    return Group(Align.center(_lantern(lt)), _wordmark(wt))


def _light_the_lantern(console: Console) -> None:
    """Intro: the flame catches (glow ramps up), breathes and flickers as if
    alive, then the wordmark glows on. Plays once when the menu opens."""
    with Live(console=console, refresh_per_second=30, transient=True) as live:
        for i in range(14):                       # the flame catches — smooth ramp
            live.update(Align.center(_lantern((i / 13) ** 1.5)))
            time.sleep(0.032)
        for k in range(13):                       # breathes/flickers, easing calm
            damp = 1 - k / 12                      # jitter shrinks toward the end
            t = 0.88 + 0.12 * random.uniform(-1, 1) * damp
            live.update(Align.center(_lantern(max(0.68, min(1.0, t)))))
            time.sleep(0.05)
        for i in range(8):                        # wordmark glows on over a STEADY flame
            live.update(_banner_group(1.0, i / 7))
            time.sleep(0.05)
        live.update(_banner_group(1.0, 1.0))       # settle on the clean, full banner
        time.sleep(0.3)                            # brief hold — no jitter on hand-off

_PROMPTS = {
    "{setting}": "setting name",
    "{value}": "new value",
    "{query}": "search for",
    "{text}": "fact to remember",
    "{id}": "fact id",
    "{model}": "embedding model (e.g. bge-m3)",
    "{note}": "note",
    "{goal}": "the goal (paste multi-line if you want; e.g. build a CLI todo app with tests)",
    "{task}": "what to do (multi-line ok; e.g. fix the failing test in parser.py)",
    "{codefile}": "file the model may edit (relative path)",
    "{codetest}": "pytest target (optional, e.g. tests/test_parser.py)",
    "{seeds}": "seeds (3+ for real numbers)",
    "{trials}": "trials per task (agentic runs are slow; 1-2)",
}


# Free-text prose fields where a pasted multi-line prompt should be kept whole.
_MULTILINE_TOKENS = {"{goal}", "{task}", "{text}", "{note}", "{query}"}


def _read_prose(console: Console, label: str) -> str:
    """Read a value that may span several lines. A pasted block hits stdin all at
    once, so after the first line we drain any lines already buffered — that keeps
    a copied multi-line prompt intact instead of leaking its tail into the next
    menu prompt. A normally typed single line has nothing buffered and returns as-is."""
    import select
    import sys

    first = console.input(f"  [{_PINK}]▸[/] {label}: ")
    lines = [first]
    if sys.stdin.isatty():
        try:
            while select.select([sys.stdin], [], [], 0.1)[0]:
                more = sys.stdin.readline()
                if not more:
                    break
                lines.append(more.rstrip("\n"))
        except (OSError, ValueError):
            pass  # stdin not selectable → just use what we have
    return "\n".join(lines).strip()


def _fill(template: list[str], console: Console) -> list[str] | None:
    argv: list[str] = []
    for tok in template:
        if tok in _PROMPTS:
            if tok in _MULTILINE_TOKENS:
                val = _read_prose(console, _PROMPTS[tok])
            else:
                val = console.input(f"  [{_PINK}]▸[/] {_PROMPTS[tok]}: ").strip()
            if not val:
                return None
            argv.append(val)
        else:
            argv.append(tok)
    return argv


def _flat(sections) -> list:
    return [item for _, items in sections for item in items]


def _render(sections, console: Console, root: bool) -> None:
    from agent import main

    rows: list = []
    if root:
        rows.append(_banner_group())
        rows.append(Align.center(Text("· local · offline · yours ·", style=f"italic {_DIMW}")))
    rows.append(Text(""))
    n = 0
    for title, items in sections:
        rows.append(Text(f" {title}", style=f"bold {_PINK}"))
        for label, _action in items:
            n += 1
            line = Text()
            line.append(f"   [{n:02}] ", style=f"bold {_GREEN}")
            line.append(label, style=_CYAN)
            rows.append(line)
        rows.append(Text(""))
    rows.append(Text("   [ q] exit" if root else "   [ b] back", style=_DIM))
    console.print(Panel(
        Group(*rows), border_style=_FLAME, style=f"on {_BG}", padding=(0, 1),
        title=f"[bold {_GLOW}]✦ lantern[/] [{_DIM}]v{main.__version__}[/]",
        subtitle=f"[{_DIM}]select ▸ number  ·  q to quit[/]",
    ))


def _run(sections, console: Console, app, root: bool) -> None:
    items = _flat(sections)
    while True:
        _render(sections, console, root)
        choice = console.input(f"\n [{_GREEN}]╺╸[/] ").strip().lower()
        if root and choice in ("q", "quit", "exit", "0", ""):
            console.print(f"[{_EMBER}]  ·  the flame goes out. see you soon.  ·[/]")
            return
        if not root and choice in ("b", "back", "0", ""):
            return
        if not choice.isdigit() or not (1 <= int(choice) <= len(items)):
            console.print(f"[{_PINK}]!! invalid selection[/]")
            continue
        kind, payload = items[int(choice) - 1][1]
        if kind == "menu":
            _run(payload, console, app, root=False)
            continue
        if kind == "call":
            try:
                payload(console)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[{_PINK}]!! {exc}[/]")
            continue
        argv = _fill(payload, console)
        if argv is None:
            continue
        console.print(f"[{_DIM}]── running ──[/]")
        try:
            app(argv, standalone_mode=False)
        except SystemExit:
            pass
        except Exception as exc:  # noqa: BLE001 - one failed action must not kill the menu
            console.print(f"[{_PINK}]!! {exc}[/]")


def run_menu() -> None:
    from agent.main import app  # imported here to avoid a circular import

    console = Console()
    try:
        _light_the_lantern(console)   # the flame catches on open
    except Exception:  # noqa: BLE001 - a dumb terminal shouldn't block the menu
        pass
    _run(_MAIN, console, app, root=True)
