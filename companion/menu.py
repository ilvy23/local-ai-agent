"""Interactive nested menu — a friendly launcher over the existing commands.

Reuses the Typer app in-process (no subprocess, no duplicated logic): each leaf
choice is an argv the app already knows how to run. Submenus nest. Placeholders
prompt for contact/depth/draft/effort/setting when needed. Neon cyberpunk theme.
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


def _open():
    """(config, store-context-manager) — lazy import to avoid a cycle."""
    from companion.config import load_config
    from companion.main import _open_store

    config = load_config()
    return config, _open_store(config)


def _pick_from(console: Console, rows, prompt: str):
    """Show a numbered list of (label, value) and return the chosen value, or a
    typed free-text value, or None to cancel."""
    if not rows:
        return console.input(f"  [{_PINK}]▸[/] {prompt}: ").strip() or None
    for i, (label, _v) in enumerate(rows, 1):
        console.print(f"   [{_GREEN}]{i:2}[/] [{_CYAN}]{label}[/]")
    raw = console.input(f"  [{_PINK}]▸[/] {prompt} (number, or type a name): ").strip()
    if not raw:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(rows):
        return rows[int(raw) - 1][1]
    return raw  # typed free text (resolve_contact will handle a name)




# --- submenus ---
_SETTINGS = [("◢ SETTINGS", [
    ("View all settings", cmd(["settings", "show"])),
    ("Change a setting", cmd(["settings", "set", "{setting}", "{value}"])),
])]

# Each top-level category is its own submenu: pick its number, drill in, see all
# of its options. Keeps the home screen to a handful of choices instead of ~40.
_CHAT = [("◢ CHAT", [
    ("Chat with your companion", cmd(["chat"])),
    ("Resume last chat", cmd(["resume"])),
    ("Past chat sessions", cmd(["sessions"])),
])]


_SYSTEM = [("◢ SYSTEM", [
    ("Live panel (this machine)", cmd(["panel"])),
    ("Background worker · status", cmd(["worker", "status"])),
    ("Background worker · start", cmd(["worker", "start"])),
    ("Background worker · stop", cmd(["worker", "stop"])),
    ("Memory · facts about you", cmd(["memory", "list"])),
    ("Audit trail", cmd(["audit"])),
    ("Settings  ▸", sub(_SETTINGS)),
])]

_MAIN = [
    ("◢ MENU  ·  pick a number", [
        ("Chat  ▸", sub(_CHAT)),
        ("System  ▸", sub(_SYSTEM)),
    ]),
]

_BANNER = r"""
 ▄▄· ▄▄▄  • ▌ ▄ ·. ▄▄▄· ▄▄▄· ▐ ▄ ▪  ▄▄▄  ▐ ▄
▐█ ▌▪▀▄ █··██ ▐███▪▐█ ▄█▐█ ▀█•█▌▐███ ▀▄ █·•█▌▐█
██ ▄▄▐▀▀▄ ▐█ ▌▐▌▐█·██ ▀·▄█▀▀█▐█▐▐▌▐█·▐▀▀▄ ▐█▐▐▌
▐███▌▐█•█▌██ ██▌▐█▌▐█▄▄▌▐█▪ ▐▌██▐█▌▐█▌▐█•█▌██▐█▌
·▀▀▀ .▀  ▀▀▀  █▪▀▀▀ ·▀▀▀  ▀  ▀ ▀▀ █▪▀▀▀.▀  ▀▀▀ █▪
"""

_PROMPTS = {
    "{draft}": "draft message",
    "{setting}": "setting name",
    "{value}": "new value",
    "{query}": "search for",
    "{note}": "your plans/availability today",
}


def _fill(template: list[str], console: Console) -> list[str] | None:
    argv: list[str] = []
    skip_next = False
    for tok in template:
        if skip_next:
            skip_next = False
            continue
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
        title=f"[{_GREEN}]COMPANION[/] [{_DIM}]v1[/]",
        subtitle=f"[{_DIM}]select ▸ number[/]",
    ))


def _run(sections, console: Console, app, root: bool) -> None:
    items = _flat(sections)
    while True:
        _render(sections, console, root)
        choice = console.input(f"\n [{_GREEN}]╺╸[/] ").strip().lower()
        if root and choice in ("q", "quit", "exit", "0", ""):
            console.print(f"[{_PURPLE}]// disconnected //[/]")
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
            console.print(f"[{_DIM}]── running ──[/]")
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
    from companion.main import app  # imported here to avoid a circular import

    console = Console()
    _run(_MAIN, console, app, root=True)
