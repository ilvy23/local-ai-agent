"""Interactive nested menu — a friendly launcher over the CLI.

Reuses the Typer app in-process (no subprocess, no duplicated logic): each leaf
choice is an argv the app already knows how to run. Submenus nest; `{...}`
placeholders prompt for a value first. Neon theme, because a local tool you live
in should have a little character.
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

_CYAN, _PINK, _GREEN, _PURPLE, _DIM = "#05d9e8", "#ff2a6d", "#39ff14", "#b967ff", "#5a5a8a"


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
    body.append("agent", style=f"bold {_CYAN}")
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
        ("Memory  ▸", sub(_MEMORY)),
        ("System  ▸", sub(_SYSTEM)),
        ("Settings  ▸", sub(_SETTINGS)),
        ("About & help", call(_about)),
    ]),
]

_BANNER = r"""
 ▄▄▄       ▄████ ▓█████ ███▄    █ ▄▄▄█████▓
▒████▄    ██▒ ▀█▒▓█   ▀ ██ ▀█   █ ▓  ██▒ ▓▒
▒██  ▀█▄ ▒██░▄▄▄░▒███  ▓██  ▀█ ██▒▒ ▓██░ ▒░
░██▄▄▄▄██░▓█  ██▓▒▓█  ▄▓██▒  ▐▌██▒░ ▓██▓ ░
 ▓█   ▓██░▒▓███▀▒░▒████▒██░   ▓██░  ▒██▒ ░
 ▒▒   ▓▒█░░▒   ▒ ░░ ▒░ ░ ▒░   ▒ ▒   ▒ ░░
"""

_PROMPTS = {
    "{setting}": "setting name",
    "{value}": "new value",
    "{query}": "search for",
    "{text}": "fact to remember",
    "{id}": "fact id",
    "{model}": "embedding model (e.g. bge-m3)",
}


def _fill(template: list[str], console: Console) -> list[str] | None:
    argv: list[str] = []
    for tok in template:
        if tok in _PROMPTS:
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
        rows.append(Align.center(Text(_BANNER.strip("\n"), style=f"bold {_CYAN}")))
        rows.append(Align.center(Text("// local · offline · yours //", style=f"italic {_PURPLE}")))
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
        Group(*rows), border_style=_PINK,
        title=f"[{_GREEN}]AGENT[/] [{_DIM}]v{main.__version__}[/]",
        subtitle=f"[{_DIM}]select ▸ number  ·  q to quit[/]",
    ))


def _run(sections, console: Console, app, root: bool) -> None:
    items = _flat(sections)
    while True:
        _render(sections, console, root)
        choice = console.input(f"\n [{_GREEN}]╺╸[/] ").strip().lower()
        if root and choice in ("q", "quit", "exit", "0", ""):
            console.print(f"[{_PURPLE}]// see you around //[/]")
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
    _run(_MAIN, console, app, root=True)
