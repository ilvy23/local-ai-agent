"""Grouped help (by task, not alphabet), `/help <command>` detail view and the key overlay.

Built from the command REGISTRY, so a new command shows up here automatically. Rendered with rich so it works in
plain mode and at any width (40 columns and up)."""

from __future__ import annotations

import textwrap
from typing import Any

from rich.text import Text

from agent import brand
from agent.ui import style
from agent.ui.commands import DEFERRED, INSTANT, STEP, REGISTRY, Command, Registry

# Short, concrete examples per command; commands without an entry just show their usage line.
EXAMPLES: dict[str, tuple[str, ...]] = {
    "web": ("/web latest stable python version", "/search rust async runtime comparison"),
    "say": ("/say also keep the public API unchanged",),
    "thinking": ("/thinking on", "/thinking off"),
    "model": ("/model",),
    "steps": ("/steps 40",),
    "mode": ("/mode", "/mode careful"),
    "plan": ("/plan", "/plan clear"),
    "verify": ("/verify",),
    "undo": ("/undo list", "/undo"),
    "context": ("/context",),
    "compact": ("/compact",),
    "notes": ("/notes", "/notes clear"),
    "project": ("/project", "/project clear"),
    "access": ("/access",),
    "auto": ("/auto-~/projects/demo",),
    "medium": ("/medium-~/projects/demo",),
    "ask": ("/ask-~/projects/demo",),
    "greenlist": ("/greenlist-~/projects/demo",),
    "ungreenlist": ("/ungreenlist-~/projects/demo",),
    "status": ("/status",),
    "tasks": ("/tasks",),
    "queue": ("/queue", "/queue rm 2", "/queue now 1", "/queue clear"),
    "log": ("/log", "/log error"),
    "events": ("/events tool_call",),
    "usage": ("/usage",),
    "transcript": ("/transcript",),
    "why-slow": ("/why-slow",),
    "doctor": ("/doctor",),
    "export": ("/export md", "/export json"),
    "help": ("/help", "/help queue"),
    "stop": ("/stop",),
    "pause": ("/pause",),
    "resume": ("/resume",),
    "clear": ("/clear",),
    "new": ("/new",),
    "quit": ("/quit",),
}

_WHEN_TEXT = {
    INSTANT: "Runs immediately, even while the agent is working.",
    STEP: "While the agent is working it takes effect at the next step.",
    DEFERRED: "While the agent is working it waits and runs after the current turn.",
}

# Non-slash ways in, shown once under the groups (they are CLI commands, not registry entries).
START_LINES = (
    (f"{brand.CMD}", "home screen"),
    (f"{brand.CMD} chat", "start a chat"),
    (f"{brand.CMD} code", "code in this folder"),
    (f"{brand.CMD} doctor", "check your setup"),
)


def help_groups(registry: Registry | None = None) -> list[tuple[str, list[Command]]]:
    return (registry or REGISTRY).help_groups()


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    return textwrap.wrap(text, max(12, width), initial_indent=indent, subsequent_indent=indent) or [indent]


def render_help(console: Any = None, registry: Registry | None = None, topic: str | None = None, *,
                plain: bool | None = None, width: int | None = None) -> bool:
    """Print the grouped overview, or the detail view when `topic` names a command. Returns False for an unknown
    topic (after printing suggestions)."""
    plain = style.plain_mode() if plain is None else plain
    console = console or style.make_console(width=width, plain=plain)
    reg = registry or REGISTRY
    w = max(30, console.width or width or 80)
    pal = style.palette()
    acc = "" if plain else f"bold {pal.accent}"
    dim = "" if plain else pal.dim

    if topic and topic.strip():
        return _render_detail(console, reg, topic.strip().lstrip("/"), w, acc, dim)

    console.print(Text(f"{brand.NAME} help", style=acc))
    console.print(Text("Type a message to chat, or a /command. Details: /help <command>.", style=dim))
    name_w = min(22, max(len(_label(c)) for _, cmds in help_groups(reg) for c in cmds) + 2)
    for group, cmds in help_groups(reg):
        console.print()
        console.print(Text(group, style=acc))
        for c in cmds:
            label = _label(c)
            if w >= 70:
                body = _wrap(c.summary, w - name_w - 3)
                console.print(Text("  " + label.ljust(name_w) + body[0]))
                for extra in body[1:]:
                    console.print(Text("  " + " " * name_w + extra))
            else:
                console.print(Text("  " + label))
                for line in _wrap(c.summary, w - 6, "    "):
                    console.print(Text(line, style=dim))
    console.print()
    console.print(Text("Start from the shell", style=acc))
    for cmd, what in START_LINES:
        console.print(Text(f"  {cmd.ljust(14)} {what}" if w >= 50 else f"  {cmd}"))
    console.print()
    console.print(Text("More, from the home screen", style=acc))
    for label, what in more_topics():
        console.print(Text(f"  {label.ljust(28)} {what}" if w >= 70 else f"  {label}"))
    console.print()
    console.print(Text("Keys: press ? on an empty line.", style=dim))
    return True


def more_topics() -> list[tuple[str, str]]:
    """Home-screen topics that are not slash commands (WhatsApp only when the plugin is installed; its private
    commands only when this build has the private module)."""
    from agent.ui import launcher

    out = []
    for key, label, hint in launcher.TOPICS:
        if key == "whatsapp":
            if not launcher.whatsapp_installed():
                continue
            hint += "" if launcher.wa_private_available() else " (analysis not in this build)"
        out.append((label, hint))
    return out


def _label(c: Command) -> str:
    return "/" + c.name + ("-<dir>" if c.pattern else "")


def _render_detail(console: Any, reg: Registry, topic: str, w: int, acc: str, dim: str) -> bool:
    hit = reg.get(topic)
    if hit is None:
        from agent.ui.palette import pick_command
        near = pick_command(reg, topic, limit=3)
        console.print(Text(f"No command called /{topic}."))
        if near:
            console.print(Text("Did you mean: " + ", ".join(m.name for m in near) + "?"))
        console.print(Text("See all commands with /help.", style=dim))
        return False
    console.print(Text(_label(hit), style=acc))
    for line in _wrap(hit.summary, w - 2):
        console.print(Text(line))
    console.print()
    console.print(Text("Usage    " + (hit.usage or _label(hit))))
    if hit.aliases:
        console.print(Text("Aliases  " + ", ".join("/" + a for a in hit.aliases)))
    console.print(Text("Group    " + hit.group))
    for i, line in enumerate(_wrap(_WHEN_TEXT[hit.when], w - 9)):
        console.print(Text(("Timing   " if i == 0 else "         ") + line))
    for sub, when in hit.sub_when.items():
        console.print(Text(f"         '{sub}': {_WHEN_TEXT[when].split(',')[0].rstrip('.').lower()}", style=dim))
    ex = EXAMPLES.get(hit.name)
    if ex:
        console.print()
        console.print(Text("Examples", style=acc))
        for e in ex:
            console.print(Text("  " + e))
    return True


# ----------------------------------------------------------------------------- key overlay


def keys_help_text(width: int = 80) -> str:
    """The key list as plain text (reuses the dock's KEYS_HELP)."""
    from agent.ui.dock import KEYS_HELP
    kw = max(len(k) for k, _ in KEYS_HELP)
    lines = ["Keys"]
    for k, what in KEYS_HELP:
        if width >= kw + 4 + max(len(w) for _, w in KEYS_HELP):
            lines.append(f"  {k.ljust(kw)}  {what}")
        else:
            lines.append(f"  {k}")
            lines.extend(textwrap.wrap(what, max(12, width - 6), initial_indent="    ", subsequent_indent="    "))
    return "\n".join(lines)


def render_keys(console: Any = None, *, plain: bool | None = None, width: int | None = None) -> None:
    console = console or style.make_console(width=width, plain=plain)
    console.print(Text(keys_help_text(max(30, console.width or width or 80))))
