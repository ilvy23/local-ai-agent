"""The home screen: a short status block and a keyboard-driven action list. Replaces the old playful menu.

`run_home(config=None, *, input=None, output=None, plain=None) -> HomeChoice` prints (native scrollback, no
full-screen app, no rich.Live) a wordmark + status block, then lets the user pick an action with arrow keys, a
number, or by typing to filter. Typing `/` filters the command palette instead.

The caller (main.py) dispatches on `HomeChoice.action`:
  continue   args {"session_id": int}   resume the last session
  chat       args {} or {"command": "/status"}   new chat (optionally start by running that slash command)
  code       args {"cwd": str}          coding agent in this folder
  research   args {"question": str}     research a question (question may be empty: ask for it)
  topic      args {"topic": str}        open a topic list from agent.ui.launcher (whatsapp, memory, apps, status,
                                        codetools, commands); the caller runs `launcher.run_topic(topic, console)`
  settings   args {}                    run agent.settings.interactive(console)
  doctor     args {}                    run_doctor(...)
  help       args {}                    render_help(...)
  quit       args {}
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.text import Text

from agent import brand
from agent.ui import style
from agent.ui.doctor import OllamaInfo, Probes, model_installed

VERSION = "0.2.0"
_TOPIC_KEYS = frozenset({"whatsapp", "memory", "apps", "status", "codetools", "commands"})
START_OLLAMA_HINT = "not running - `ollama serve` or `nightage ollama start`"


@dataclass
class HomeChoice:
    action: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Item:
    key: str            # action name
    label: str
    hint: str = ""


@dataclass
class StatusInfo:
    ollama: OllamaInfo = field(default_factory=OllamaInfo)
    ollama_ours: bool | None = None        # True: Nightage started the local server; False: external; None: remote/unknown
    chat_model: str = ""
    coding_model: str = ""
    chat_present: bool | None = None       # None = unknown (Ollama down)
    coding_present: bool | None = None
    gpus: list[Any] = field(default_factory=list)
    gpu_mode: str | None = None
    ctx: dict[str, Any] | None = None
    last_session: dict[str, Any] | None = None
    data_dir: str = ""
    update_notes: list[str] = field(default_factory=list)      # cached update notice lines (agent/update.py); [] when none


def collect_status(config: dict | None = None, probes: Probes | None = None) -> StatusInfo:
    """Everything the status block shows. Defensive: any failing probe just leaves its field empty."""
    p = probes or Probes()
    if config is None:
        from agent.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    m = (config or {}).get("models") or {}
    st = StatusInfo(chat_model=m.get("chat") or "", coding_model=m.get("coding") or "")
    try:
        st.ollama = p.ollama()
    except Exception:  # noqa: BLE001
        st.ollama = OllamaInfo(error="probe failed")
    if st.ollama.reachable:
        try:
            from agent import ollama_local

            if ollama_local.is_local(st.ollama.url or ollama_local.base_url(config)):
                st.ollama_ours = ollama_local.read_record() is not None
        except Exception:  # noqa: BLE001
            pass
    if st.ollama.reachable:
        st.chat_present = model_installed(st.chat_model, st.ollama.models) if st.chat_model else None
        st.coding_present = model_installed(st.coding_model, st.ollama.models) if st.coding_model else None
    try:
        st.gpus = list(p.gpus())
        st.gpu_mode = p.gpu_mode(st.gpus)
    except Exception:  # noqa: BLE001
        pass
    if st.chat_model:
        try:
            size = st.ollama.models.get(st.chat_model) or st.ollama.models.get(f"{st.chat_model}:latest") or 0
            st.ctx = p.ctx_budget(st.chat_model, size, st.gpus, config)
        except Exception:  # noqa: BLE001
            st.ctx = None
    try:
        st.last_session = p.last_session(config)
        st.data_dir = str(p.data_paths(config)[0])
    except Exception:  # noqa: BLE001
        pass
    if probes is None:                       # real start only (tests inject probes): read the cached notice, refresh it in the background
        try:
            from agent import update

            st.update_notes = update.notice_lines(config)
            update.start_background_check(config)
        except Exception:  # noqa: BLE001
            pass
    return st


def age_text(ts: str, now: datetime | None = None) -> str:
    """'3 min ago' for an ISO timestamp; '' when unparseable."""
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    secs = ((now or datetime.now(timezone.utc)) - t).total_seconds()
    if secs < 0:
        return ""
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400 * 2:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} days ago"


# ----------------------------------------------------------------------------- pure rendering


def _origin_note(st: "StatusInfo") -> str:
    """' (started by nightage)' / ' (external)' for a local server; nothing for a remote one or when unknown."""
    if st.ollama_ours is None:
        return ""
    return " (started by nightage)" if st.ollama_ours else " (external)"


def status_lines(st: StatusInfo, width: int, plain: bool = False) -> list[tuple[str, str, str]]:
    """[(label, value, level)] where level is ok | warn | fail | '' - rendered by `render_home`."""
    out: list[tuple[str, str, str]] = []
    o = st.ollama
    phone = False
    try:
        from agent import sysinfo

        phone = sysinfo.current().is_termux
    except Exception:  # noqa: BLE001
        pass
    if phone and not o.reachable:
        out.append(("Ollama on this phone", "not running - start it with `nightage ollama start`", "fail"))
        for role, name in (("Chat", st.chat_model), ("Code", st.coding_model)):
            if name:
                out.append((role, name, ""))
    elif o.reachable:
        out.append(("Ollama on this phone" if phone else "Ollama",
                    f"running {o.version}".strip() + f", {len(o.models)} model(s)" + _origin_note(st), "ok"))
        for role, name, present in (("Chat", st.chat_model, st.chat_present), ("Code", st.coding_model, st.coding_present)):
            if not name:
                continue
            if present:
                out.append((role, name, "ok"))
            else:
                out.append((role, f"{name} missing - run `ollama pull {name}`", "warn"))
    else:
        out.append(("Ollama", START_OLLAMA_HINT, "fail"))
        for role, name in (("Chat", st.chat_model), ("Code", st.coding_model)):
            if name:
                out.append((role, name, ""))
    if st.gpus:
        names = ", ".join(f"{g.name} {g.total_mb // 1024} GB" for g in st.gpus)
        out.append(("GPU", names + (f" ({st.gpu_mode})" if st.gpu_mode else ""), "ok"))
    else:
        out.append(("GPU", "none detected (CPU)", "warn"))
    if st.ctx:
        c = st.ctx
        src = "override" if c.get("override") else str(c.get("source", ""))
        out.append(("Context", f"{style.fmt_tokens(c.get('final') or 0)} tokens ({src})", ""))
    if st.last_session:
        ls = st.last_session
        title = ls.get("title") or "untitled"
        age = age_text(ls.get("last_active_at", ""))
        out.append(("Last", f"{title}" + (f" - {age}" if age else ""), ""))
    else:
        out.append(("Last", "no sessions yet", ""))
    if st.data_dir:
        out.append(("Data", st.data_dir, ""))
    for note in st.update_notes:
        out.append(("Update", note, ""))
    return out


def build_items(st: StatusInfo) -> list[Item]:
    items: list[Item] = []
    if st.last_session:
        t = st.last_session.get("title") or "untitled"
        items.append(Item("continue", "Continue last session", t))
    items += [
        Item("chat", "New chat", "talk to the assistant"),
        Item("code", "Code in this folder", os.path.basename(os.getcwd()) or "/"),
        Item("research", "Research a question", "web research with sources"),
    ]
    from agent.ui import launcher
    for key, label, hint in launcher.TOPICS:
        if key == "whatsapp" and not launcher.whatsapp_installed():
            continue
        items.append(Item(key, label, hint))
    items += [
        Item("settings", "Settings", "models, memory, safety"),
        Item("doctor", "Status & doctor", "check your setup"),
        Item("help", "Help", "commands by task"),
        Item("quit", "Quit", ""),
    ]
    return items


def filter_items(items: list[Item], text: str) -> list[Item]:
    """Digit -> that item (1-based); otherwise substring/word-prefix match on the label, key and hint."""
    t = text.strip().lower()
    if not t:
        return list(items)
    if t.isdigit():
        i = int(t)
        return [items[i - 1]] if 1 <= i <= len(items) else []
    hits = [(0 if it.label.lower().startswith(t) or it.key.startswith(t) else 1, n, it)
            for n, it in enumerate(items)
            if t in it.label.lower() or t in it.key or t in it.hint.lower()
            or any(w.startswith(t) for w in it.label.lower().split())]
    hits.sort(key=lambda x: (x[0], x[1]))
    return [it for _, _, it in hits]


def _cut(text: str, width: int, g: style.Glyphs) -> str:
    """Truncate without collapsing the alignment spaces (style.trunc normalises whitespace)."""
    if len(text) <= width:
        return text
    return text[: max(0, width - len(g.ellipsis))] + g.ellipsis


def action_rows(items: list[Item], shown: list[Item], sel: int, width: int, plain: bool) -> list[tuple[str, str]]:
    """Formatted (prompt_toolkit style, text) rows for the action list; every row fits `width - 1`."""
    g = style.glyphs(plain)
    cur = ">" if plain else "›"
    rows: list[tuple[str, str]] = []
    if not shown:
        return [("class:dim", "  no match - keep typing, Backspace to clear")]
    lw = max(len(i.label) for i in items)
    nw = len(str(len(items)))                     # numbers right-aligned so rows 10+ keep the columns
    for n, it in enumerate(shown):
        num = items.index(it) + 1
        mark = cur if n == sel else " "
        line = f"{mark} {num:>{nw}}  {it.label}"
        if it.hint and width >= 60:
            col = 2 + nw + 2 + lw + 3
            hint = style.trunc(it.hint, width - 1 - col, g) if width - 1 - col > 8 else ""
            if hint:
                line = line.ljust(col) + hint
        rows.append(("class:sel" if n == sel else "", _cut(line, width - 1, g)))
    return rows


FOOTER_WIDE = "Up/Down move  Enter select  1-9 jump  type to filter  / commands  ? keys  Esc quit"
FOOTER_NARROW = "Up/Down  Enter  1-9  type  Esc quit"


def footer_text(width: int) -> str:
    return FOOTER_WIDE if width >= len(FOOTER_WIDE) + 2 else FOOTER_NARROW if width >= len(FOOTER_NARROW) + 2 else "Enter select, Esc quit"


def render_home(console: Any, st: StatusInfo, *, plain: bool, width: int | None = None) -> None:
    """Print wordmark + status block (everything above the interactive list)."""
    w = max(30, width or console.width or 80)
    pal = style.palette()
    g = style.glyphs(plain)
    from agent.ui import banner

    big = banner.heading(w, plain)
    acc = "" if plain else f"bold {pal.accent}"
    dim = "" if plain else pal.dim
    if len(big) > 1:                                  # wide terminal: the block-letter heading, then version and tagline
        for row in big:
            console.print(Text(row, style=acc))
        sub = Text(f"v{VERSION}", style=dim)
        sub.append("  " + g.bullet + "  " + brand.TAGLINE, style=dim)
        console.print(sub)
        console.print(Text(g.rule * min(w - 1, banner.banner_width() + 4), style=dim))
    else:                                             # narrow terminal: one letter-spaced line
        head = Text()
        head.append(big[0], style=acc)
        head.append(f"  v{VERSION}", style=dim)
        console.print(head)
        if w >= 50:
            console.print(Text(brand.TAGLINE, style=dim))
    console.print()
    lines = status_lines(st, w, plain)
    lw = max(len(l) for l, _, _ in lines)
    mark = {"ok": (g.check, pal.ok), "warn": ("!", pal.warn), "fail": (g.cross, pal.error), "": (" ", "")}
    for label, value, level in lines:
        sym, col = mark[level]
        t = Text()
        t.append(f" {sym} ", style="" if plain else col)
        if w >= 50:
            t.append(label.ljust(lw) + "  ", style="" if plain else pal.dim)
            body_w = w - 6 - lw
        else:
            t.append(label + " ", style="" if plain else pal.dim)
            body_w = w - 5 - len(label)
        import textwrap
        parts = textwrap.wrap(value, max(8, body_w)) or [""]
        t.append(parts[0])
        console.print(t)
        indent = " " * (w - body_w if w - body_w > 0 else 4)
        for extra in parts[1:]:
            console.print(Text(indent + extra))
    console.print()


# ----------------------------------------------------------------------------- interactive


def run_home(config: dict | None = None, *, input: Any = None, output: Any = None, plain: bool | None = None,
             console: Any = None, probes: Probes | None = None, registry: Any = None,
             width: int | None = None) -> HomeChoice:
    """Show the home screen and return the user's choice. Never blocks on a missing Ollama (probes time out in
    ~1.5 s). With a non-interactive stdin and no injected `input`, reads one line (number or name) instead."""
    from agent import friendly

    friendly.quiet_logging()
    plain = style.plain_mode() if plain is None else plain
    console = console or style.make_console(width=width, plain=plain)
    st = collect_status(config, probes)
    render_home(console, st, plain=plain, width=width)
    items = build_items(st)

    if input is None and not sys.stdin.isatty():
        return _choose_from_line(items, console, plain)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    from agent.ui.commands import REGISTRY
    from agent.ui.palette import pick_command

    reg = registry or REGISTRY
    state = {"sel": 0}
    show_keys = {"on": False}

    def cols() -> int:
        if width:
            return width
        try:
            return session.app.output.get_size().columns
        except Exception:  # noqa: BLE001
            return 80

    def current(text: str) -> tuple[str, list]:
        """(mode, shown) - mode 'actions' or 'palette'."""
        if text.startswith("/"):
            return "palette", pick_command(reg, text, limit=6)
        return "actions", filter_items(items, text)

    def message() -> FormattedText:
        w = cols()
        text = session.default_buffer.text
        mode, shown = current(text)
        parts: list[tuple[str, str]] = []
        if session.app.is_done:
            return FormattedText([("", "> " if plain else "› ")])
        if show_keys["on"]:
            from agent.ui.help import keys_help_text
            parts.append(("class:dim", keys_help_text(w) + "\n\n"))
        state["sel"] = max(0, min(state["sel"], max(0, len(shown) - 1)))
        if mode == "actions":
            for sty, line in action_rows(items, shown, state["sel"], w, plain):
                parts.append((sty, line + "\n"))
        else:
            if not shown:
                parts.append(("class:dim", "  no such command\n"))
            for n, mt in enumerate(shown):
                line = f"{'>' if plain else '›'} {mt.name}  {mt.summary}"
                parts.append(("class:sel" if n == state["sel"] else "", _cut(line, w - 1, style.glyphs(plain)) + "\n"))
        parts.append(("", "> " if plain else "› "))
        return FormattedText(parts)

    def toolbar() -> FormattedText:
        return FormattedText([("class:dim", footer_text(cols()))])

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:  # noqa: ANN001
        state["sel"] = max(0, state["sel"] - 1)

    @kb.add("down")
    @kb.add("c-n")
    @kb.add("tab")
    def _down(event) -> None:  # noqa: ANN001
        _mode, shown = current(event.app.current_buffer.text)
        state["sel"] = min(max(0, len(shown) - 1), state["sel"] + 1)

    @kb.add("?")
    def _q(event) -> None:  # noqa: ANN001
        buf = event.app.current_buffer
        if buf.text == "":
            show_keys["on"] = not show_keys["on"]
        else:
            buf.insert_text("?")

    @kb.add("enter")
    def _enter(event) -> None:  # noqa: ANN001
        buf = event.app.current_buffer
        mode, shown = current(buf.text)
        if not shown:
            return
        pick = shown[min(state["sel"], len(shown) - 1)]
        event.app.exit(result=("palette", pick) if mode == "palette" else ("action", pick))

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("c-d")
    def _quit(event) -> None:  # noqa: ANN001
        event.app.exit(result=("action", Item("quit", "Quit")))

    @kb.add("c-k")
    def _palette(event) -> None:  # noqa: ANN001
        buf = event.app.current_buffer
        if not buf.text.startswith("/"):
            buf.text = "/"
            buf.cursor_position = 1

    def on_change(_buf: Any) -> None:
        state["sel"] = 0

    session = PromptSession(
        message=message, bottom_toolbar=toolbar, key_bindings=kb, input=input, output=output,
        multiline=False, enable_history_search=False, mouse_support=False, erase_when_done=True,
        style=Style.from_dict({"sel": "bold reverse" if not plain else "reverse", "dim": "" if plain else "ansibrightblack"}),
        reserve_space_for_menu=0,
    )
    session.default_buffer.on_text_changed += on_change
    session.app.ttimeoutlen = 0.05
    try:
        kind, pick = session.prompt()
    except (EOFError, KeyboardInterrupt):
        return HomeChoice("quit")
    if kind == "palette":
        return HomeChoice("chat", {"command": "/" + pick.command.name})
    return _finish(pick.key, session, st, console, plain, input, output)


def _finish(key: str, session: Any, st: StatusInfo, console: Any, plain: bool, input: Any, output: Any) -> HomeChoice:
    if key == "continue":
        return HomeChoice("continue", {"session_id": (st.last_session or {}).get("id")})
    if key == "code":
        return HomeChoice("code", {"cwd": os.getcwd()})
    if key == "research":
        from prompt_toolkit import PromptSession
        try:
            q = PromptSession(input=input, output=output).prompt("Question (Enter to enter it later): ")
        except (EOFError, KeyboardInterrupt):
            q = ""
        return HomeChoice("research", {"question": (q or "").strip()})
    if key in _TOPIC_KEYS:
        return HomeChoice("topic", {"topic": key})
    return HomeChoice(key)


def _choose_from_line(items: list[Item], console: Any, plain: bool) -> HomeChoice:
    """Non-interactive fallback: print the list, read one line from stdin."""
    for sty, line in action_rows(items, items, 0, console.width or 80, plain):
        console.print(Text(line))
    try:
        raw = sys.stdin.readline()
    except (OSError, ValueError):
        raw = ""
    if not raw:
        return HomeChoice("quit")
    hits = filter_items(items, raw)
    if not hits:
        return HomeChoice("quit")
    pick = hits[0]
    if pick.key == "code":
        return HomeChoice("code", {"cwd": os.getcwd()})
    if pick.key == "research":
        return HomeChoice("research", {"question": ""})
    if pick.key in _TOPIC_KEYS:
        return HomeChoice("topic", {"topic": pick.key})
    return HomeChoice(pick.key)
