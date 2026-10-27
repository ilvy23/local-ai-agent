"""Home-screen topic lists: every capability of the old nested menu that has no home action of its own.

The home screen (agent/ui/home.py) offers topics - WhatsApp, Memory & skills, Apps, Background helper, More code
tools and All commands. Picking one opens a short numbered list here; choosing a row runs the matching CLI command
in-process (`app(argv)`), after asking for any `{placeholder}` value. `All commands` is generated from the typer app,
so a new command is discoverable without touching this file.

Entry points:
  topic_sections(topic, ...) -> (title, [(section title, [Entry])])   pure, testable
  all_commands_sections(app) -> [(group title, [Entry])]               generated from the typer app
  run_topic(topic, console, app=None)                                  interactive numbered loop
  whatsapp_installed() / wa_private_available()                        availability checks used by home/help
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:    # optional extra apps of the private edition; the public export does not ship the module
    from agent import webapps as _private_apps
    _APPS_LABEL, _APPS_HINT = _private_apps.TOPIC_LABEL, _private_apps.TOPIC_HINT
except ImportError:
    _private_apps = None
    _APPS_LABEL, _APPS_HINT = "Web GUI", "the browser version"

TOPICS: tuple[tuple[str, str, str], ...] = (
    # key, label, hint (home.py builds its Items from this)
    ("whatsapp", "WhatsApp", "connect, sync, replies, auto-reply"),
    ("memory", "Memory & skills", "what it remembers about you"),
    ("apps", _APPS_LABEL, _APPS_HINT),
    ("status", "Background helper", "panel, worker, governor"),
    ("codetools", "More code tools", "saved sessions, repo rules, prompt studio"),
    ("plugins", "Plugins", "extensions: list, add, write your own"),
    ("commands", "All commands", "every command with its help"),
)


@dataclass
class Entry:
    label: str
    argv: list[str] = field(default_factory=list)   # CLI argv template; "{x}" tokens are asked for
    confirm: str | None = None
    call: Callable[[Any], None] | None = None       # alternative to argv: called with the console
    hint: str = ""


PROMPTS = {
    "{query}": "what to look for",
    "{id}": "which one (see the number next to it)",
    "{depth}": "how many messages to look at (e.g. 200)",
    "{promptext}": "your question",
    "{repo}": "project folder (absolute path)",
    "{pname}": "name for the new plugin (letters, digits, -)",
}


# ----------------------------------------------------------------------------- availability


def whatsapp_installed() -> bool:
    """The WhatsApp bridge plugin directory is present in the plugin folder."""
    try:
        from agent import plugins
        return any(p.name == "whatsapp" or p.stem == "whatsapp" for p in plugins.discover())
    except Exception:  # noqa: BLE001 - never break the home screen
        return False


def wa_private_available() -> bool:
    """Whether the private analysis module (profiles, digests, clones) is part of this build."""
    try:
        from agent.wa import WA_PRIVATE_AVAILABLE
        return bool(WA_PRIVATE_AVAILABLE)
    except Exception:  # noqa: BLE001
        return False


def wa_private_unavailable_notice(console: Any) -> None:
    console.print("Chat analysis and clones are not available in this build (the private WhatsApp module is not installed).")


# ----------------------------------------------------------------------------- apps


def _app_entries() -> list[Entry]:
    from agent import profiles
    entries = [
        Entry("Web GUI (browser version, needs the access token)", ["web"]),
        Entry("Show the web access token", ["web-token"]),
    ]
    if profiles.active_name() == profiles.PHONE:          # phone profile: no browser GUI (needs the `web` dependency group, heavy)
        entries = []
    if _private_apps is not None:                          # private edition: its extra apps come first
        entries = _private_apps.menu_entries(Entry) + entries
    return entries or [Entry("The browser version is not part of the phone profile", call=lambda console: console.print(
        "The browser version needs the `web` dependency group, which the phone profile leaves out."))]


# ----------------------------------------------------------------------------- code sessions picker


def code_sessions(console: Any, pick: Callable[..., Any] | None = None) -> None:
    """Saved coding sessions of every folder, newest first; the chosen one resumes in its own folder."""
    import os
    import time
    from pathlib import Path

    from rich.markup import escape

    from agent.codeagent import sessionstore

    warn: list[str] = []
    metas = sessionstore.list_sessions(None, warn)[:30]
    for w in warn:
        console.print(w)
    if not metas:
        console.print("no saved coding sessions yet")
        return
    home = str(Path.home())
    rows = [(f"{m.id}  {escape(m.title[:44]):44}  {escape(m.root.replace(home, '~', 1)[-30:])} - "
             f"{sessionstore.age_text(time.time() - m.updated)} - {m.steps} steps - {m.shown_state()}", m) for m in metas]
    picked = (pick or pick_from)(console, rows, "which session")
    meta = picked if isinstance(picked, sessionstore.Meta) else (sessionstore.find(picked)[0] if picked else None)
    if meta is None:
        return
    if not Path(meta.root).is_dir():
        console.print(f"{meta.root} no longer exists - the session stays saved")
        return
    os.chdir(meta.root)
    run_argv(["code", "--resume", meta.id, "--folder", meta.root], console)


# ----------------------------------------------------------------------------- guided / autonomous builds


def build_guided(console: Any) -> None:
    """Build mode A: guided, milestone checkpoints. Completing a reviewed milestone unlocks mode B."""
    from agent.coding_cli import run_goal_cli
    from agent.modes import MilestoneSession, mark_milestone_mode_used

    goal = console.input("  What should it build? ").strip()
    if not goal:
        console.print("cancelled")
        return
    session = MilestoneSession(console)
    run_goal_cli(goal, mode=session)
    # Any reviewed milestone counts, even if the build itself then failed.
    if session.milestone_count:
        try:
            from agent.config import load_config
            from agent.main import _open_store
            with _open_store(load_config()) as store:
                mark_milestone_mode_used(store)
        except Exception as exc:  # noqa: BLE001 - never fail a finished build on bookkeeping
            console.print(f"couldn't record the guided run: {exc}")


def build_auto(console: Any) -> None:
    """Build mode B: autonomous, checkpoints auto-approved. Only after one completed guided build."""
    from agent.coding_cli import run_goal_cli
    from agent.config import load_config
    from agent.main import _open_store          # resolves a relative data.db_path against the project root, not the cwd
    from agent.modes import AutoSession, auto_mode_unlocked

    with _open_store(load_config()) as store:
        unlocked = auto_mode_unlocked(store)
    if not unlocked:
        console.print("Auto mode unlocks after you complete one guided (step-by-step) build.")
        return
    goal = console.input("  What should it build? ").strip()
    if not goal:
        console.print("cancelled")
        return
    run_goal_cli(goal, mode=AutoSession(console))


# ----------------------------------------------------------------------------- topics


def whatsapp_sections(private: bool | None = None) -> list[tuple[str, list[Entry]]]:
    """Public entries always; the private ones only when the private module is present (else one notice entry)."""
    private = wa_private_available() if private is None else private
    sections: list[tuple[str, list[Entry]]] = [
        ("Connect", [
            Entry("Connect WhatsApp (scan a code with your phone)", ["wa", "bridge"]),
            Entry("Get new messages", ["wa", "sync"]),
            Entry("Is it connected?", ["wa", "status"]),
        ]),
    ]
    if private:
        sections.append(("People (private build)", [
            Entry("How is this friendship going?", ["wa", "health", "{contact}", "--depth", "{depth}"]),
            Entry("Sum up a chat", ["wa", "summarize", "{contact}", "--depth", "{depth}"]),
            Entry("What does their last message really mean?", ["wa", "vibe", "{contact}"]),
            Entry("Learn about someone", ["wa", "profile", "{contact}", "--depth", "{depth}"]),
            Entry("Sum up ALL chats (first run takes hours)", ["wa", "summarize", "--all"],
                  confirm="The first run rebuilds every chat summary - 2 to 3 hours. Keep going?"),
        ]))
    else:
        sections.append(("People", [Entry("Chat analysis (not available in this build)", call=wa_private_unavailable_notice)]))
    replies = [Entry("Write a reply for me", ["wa", "reply", "{contact}"])]
    if private:
        replies += [
            Entry("Talk to a clone (27B + trained adapter)", ["wa", "persona", "chat", "{contact}"]),
            Entry("Build a clone from their chats", ["wa", "persona", "build", "{contact}"],
                  confirm="Reads the whole chat with the 27B - takes several minutes. Keep going?"),
            Entry("How good is the clone?", ["wa", "persona", "eval", "{contact}"],
                  confirm="Replays real exchanges against the clone - takes a while. Keep going?"),
        ]
    sections.append(("Replies (nothing is ever sent)", replies))
    sections.append(("Auto-reply", [
        Entry("See what auto-reply is doing", ["wa", "auto", "status"]),
        Entry("Only write replies, never send them", ["wa", "auto", "draft"]),
        Entry("Turn auto-reply off completely", ["wa", "auto", "off"]),
    ]))
    return sections


def topic_sections(topic: str, private: bool | None = None) -> tuple[str, list[tuple[str, list[Entry]]]]:
    if topic == "whatsapp":
        return "WhatsApp", whatsapp_sections(private)
    if topic == "memory":
        return "Memory & skills", [
            ("What it remembers", [
                Entry("See what it remembers about you", ["memory", "list"]),
                Entry("Find something it remembers", ["memory", "search", "{query}"]),
                Entry("Make it forget something", ["memory", "forget", "{id}"]),
                Entry("Clean up junk memories", ["memory", "prune"]),
            ]),
            ("Skills", [Entry("List saved solution skills", ["skills", "list"])]),
        ]
    if topic == "apps":
        entries = _app_entries()
        return "Apps", [("Open on your phone (same Wi-Fi)" if _private_apps is not None else "Browser version", entries)]
    if topic == "status":
        return "Background helper", [("How it is running", [
            Entry("Live status panel", ["panel"]),
            Entry("Would background work run right now?", ["governor"]),
            Entry("Background worker status", ["worker", "status"]),
            Entry("Start the background worker", ["worker", "start"]),
            Entry("Stop the background worker", ["worker", "stop"]),
        ])]
    if topic == "plugins":
        return "Plugins", plugin_sections()
    if topic == "codetools":
        return "More code tools", [("Code", [
            Entry("Resume a saved coding session (all folders)", call=code_sessions),
            Entry("Code in another folder", ["code", "--folder", "{repo}"]),
            Entry("Build a project step by step (you approve each milestone)", call=build_guided),
            Entry("Build a project autonomously (unlocks after one guided build)", call=build_auto),
            Entry("Repos and what the agent may do there", ["code-policy", "list"]),
        ]), ("Prompts", [
            Entry("Prompt studio (guided)", ["prompt"]),
            Entry("Improve one question", ["prompt", "improve", "{promptext}"]),
        ])]
    raise KeyError(topic)


def plugin_sections() -> list[tuple[str, list[Entry]]]:
    """Home -> Plugins: items the enabled plugins added (api.home_item), then the management entries. Never raises."""
    sections: list[tuple[str, list[Entry]]] = []
    try:
        from agent import plugin_host

        rt = plugin_host.load_quietly()
        mine = []
        for _plugin, label, hint, action in rt.home_items:
            mine.append(Entry(label, argv=list(action), hint=hint) if isinstance(action, list) else Entry(label, call=action, hint=hint))
        if mine:
            sections.append(("From your plugins", mine))
    except Exception:  # noqa: BLE001 - a plugin must not break the home screen
        pass
    sections.append(("Manage plugins", [
        Entry("List plugins", ["plugin", "list"], hint="installed, enabled, what they add"),
        Entry("Check why a plugin does not load", ["plugin", "doctor"]),
        Entry("Make a new plugin (hello-world scaffold)", ["plugin", "new", "{pname}"], hint="guide: docs/PLUGINS.md"),
    ]))
    return sections


def all_commands_sections(app: Any = None) -> list[tuple[str, list[Entry]]]:
    """Every (non-hidden) typer command with its one-line help, generated from the app. Sub-apps list their commands
    under their own heading."""
    import typer

    if app is None:
        from agent.main import app as main_app
        app = main_app
    root = typer.main.get_command(app)
    top: list[Entry] = []
    groups: list[tuple[str, list[Entry]]] = []

    def one_line(cmd: Any) -> str:
        text = (cmd.help or cmd.short_help or "").strip().split("\n\n")[0]
        return " ".join(text.split())

    for name in sorted(root.commands):
        cmd = root.commands[name]
        if getattr(cmd, "hidden", False):
            continue
        sub = getattr(cmd, "commands", None)
        if sub:
            entries = [Entry(f"{name} {s}", [name, s], hint=one_line(c)) for s, c in sorted(sub.items())
                       if not getattr(c, "hidden", False)]
            groups.append((f"{name} - {one_line(cmd)}", entries))
        else:
            top.append(Entry(name, [name], hint=one_line(cmd)))
    return [("Commands", top)] + groups


# ----------------------------------------------------------------------------- running


def pick_from(console: Any, rows: list[tuple[str, Any]], prompt: str) -> Any:
    """Numbered list of (label, value): returns the value, typed free text, or None."""
    if not rows:
        return console.input(f"  {prompt}: ").strip() or None
    for i, (label, _v) in enumerate(rows, 1):
        console.print(f"  {i:2}  {label}")
    raw = console.input(f"  {prompt} (number, or type): ").strip()
    if not raw:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(rows):
        return rows[int(raw) - 1][1]
    return raw


def pick_contact(console: Any) -> str | None:
    """Numbered chooser over active WhatsApp chats; a typed name is accepted too."""
    try:
        from agent.config import load_config
        from agent.main import _open_store

        with _open_store(load_config()) as store:
            rows = store.conn.execute(
                "SELECT COALESCE(c.name, c.jid) AS label, c.jid AS jid, COUNT(m.id) n "
                "FROM wa_chats c JOIN wa_messages m ON m.chat_jid = c.jid "
                "WHERE c.jid NOT LIKE '%@newsletter' AND c.jid NOT LIKE '%@broadcast' "
                "GROUP BY c.jid HAVING n > 0 ORDER BY c.last_message_at DESC LIMIT 40"
            ).fetchall()
        items = [(f"{r['label'][:40]}  ({r['n']})", r["jid"]) for r in rows]
    except Exception:  # noqa: BLE001 - no DB / not synced yet: let them type a name
        items = []
    return pick_from(console, items, "who")


def fill(template: list[str], console: Any) -> list[str] | None:
    """Replace `{placeholder}` tokens by asking; a blank answer cancels (None)."""
    argv: list[str] = []
    for tok in template:
        if tok == "{contact}":
            val = pick_contact(console)
        elif tok in PROMPTS:
            val = console.input(f"  {PROMPTS[tok]}: ").strip()
        else:
            argv.append(tok)
            continue
        if not val:
            return None
        argv.append(val)
    return argv


def run_argv(argv: list[str], console: Any, app: Any = None) -> None:
    if app is None:
        from agent.main import app as main_app
        app = main_app
    try:
        app(argv, standalone_mode=False)
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 - one failed action must not kill the list
        console.print(f"!! {exc}")


def execute(entry: Entry, console: Any, app: Any = None) -> bool:
    """Run one entry. Returns False when the user cancelled before anything ran."""
    if entry.call is not None:
        try:
            entry.call(console)
        except Exception as exc:  # noqa: BLE001
            console.print(f"!! {exc}")
        return True
    if entry.confirm:
        try:
            ans = console.input(f"  {entry.confirm} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            console.print("cancelled")
            return False
    argv = fill(entry.argv, console)
    if argv is None:
        return False
    # Commands from "All commands" may need arguments: offer a free-form line (blank = none).
    if entry.hint and len(entry.argv) <= 2 and not any("{" in t for t in entry.argv):
        try:
            extra = console.input("  arguments, quote text with spaces (Enter for none): ").strip()
        except (EOFError, KeyboardInterrupt):
            extra = ""
        if extra:
            argv += shlex.split(extra)
    run_argv(argv, console, app)
    return True


def render(title: str, sections: list[tuple[str, list[Entry]]], console: Any) -> list[Entry]:
    console.print()
    console.print(title)
    flat: list[Entry] = []
    for head, entries in sections:
        console.print(f"\n {head}")
        for e in entries:
            flat.append(e)
            room = max(0, (getattr(console, "width", 80) or 80) - len(e.label) - 14)
            hint = f"  - {e.hint[:room - 3].rstrip()}..." if e.hint and len(e.hint) > room and room > 12 else (f"  - {e.hint}" if e.hint and room > 12 else "")
            console.print(f"  {len(flat):3}  {e.label}{hint}")
    console.print("\n  Enter or q to go back")
    return flat


def run_topic(topic: str, console: Any, app: Any = None, sections: list | None = None) -> None:
    """Interactive numbered loop for one topic (blank / q / b returns to the home screen)."""
    if topic == "commands":
        title, secs = "All commands", sections or all_commands_sections(app)
    else:
        title, secs = topic_sections(topic)
        secs = sections or secs
    flat = render(title, secs, console)
    while True:
        try:
            raw = console.input("\n  number: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if raw in ("", "q", "b", "back", "quit", "0"):
            return
        if not raw.isdigit() or not (1 <= int(raw) <= len(flat)):
            console.print("not a valid number")
            continue
        execute(flat[int(raw) - 1], console, app)
