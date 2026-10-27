"""Slash-command registry: metadata, classification (run now / step / defer) and completions.

Handlers are optional here; later waves bind real ones. `when` says what happens if the command is typed while a
turn is running:
  instant   runs immediately
  step      applied at the next engine step boundary
  deferred  runs after the current turn (queued with a notice)
Idle input always runs immediately."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

INSTANT, STEP, DEFERRED = "instant", "step", "deferred"
WHENS = (INSTANT, STEP, DEFERRED)

NOT_A_COMMAND, RUN_NOW, DEFER, STEP_ACTION, REFUSE = "not_a_command", "run_now", "defer", "step", "refuse"

GROUPS = ("Chat", "Code", "Context & memory", "Safety & access", "Inspect", "Session")

# Same shape as agent.repoaccess.COMMAND_RE (/auto-<dir>, /medium <dir>, ...); kept local so this module stays
# import-light and pure.
PATTERN_RE = re.compile(r"^/(auto|medium|ask|greenlist|ungreenlist)(?:[-\s]\s*(.+))?$", re.IGNORECASE)


@dataclass
class Command:
    name: str
    summary: str = ""
    usage: str = ""
    group: str = "Chat"
    when: str = INSTANT
    aliases: tuple[str, ...] = ()
    handler: Callable[..., Any] | None = None
    pattern: bool = False                      # takes a directory: /auto-<dir>
    sub_when: dict[str, str] = field(default_factory=dict)  # per-subcommand override, e.g. {"clear": "deferred"}

    def when_for(self, args: str) -> str:
        sub = args.split(None, 1)[0].lower() if args.strip() else ""
        return self.sub_when.get(sub, self.when)


@dataclass
class Decision:
    action: str
    command: Command | None = None
    args: str = ""
    notice: str = ""


@dataclass
class Completion:
    name: str      # with leading slash
    summary: str
    usage: str = ""


class Registry:
    def __init__(self) -> None:
        self._cmds: dict[str, Command] = {}
        self._lookup: dict[str, Command] = {}

    def register(self, cmd: Command) -> Command:
        if cmd.when not in WHENS:
            raise ValueError(f"bad when: {cmd.when!r}")
        self._cmds[cmd.name] = cmd
        for n in (cmd.name, *cmd.aliases):
            self._lookup[n.lower()] = cmd
        return cmd

    def bind(self, name: str, handler: Callable[..., Any]) -> None:
        self._lookup[name.lower()].handler = handler

    def unregister(self, name: str) -> None:
        """Remove a command and its aliases (used when a plugin is reloaded)."""
        cmd = self._cmds.pop(name, None)
        if cmd is not None:
            for n in (cmd.name, *cmd.aliases):
                self._lookup.pop(n.lower(), None)

    def commands(self) -> list[Command]:
        return list(self._cmds.values())

    def get(self, name: str) -> Command | None:
        return self._lookup.get(name.lower().lstrip("/"))

    def resolve(self, text: str) -> tuple[Command, str] | None:
        t = text.strip()
        if not t.startswith("/") or len(t) < 2:
            return None
        m = PATTERN_RE.match(t)
        if m and m.group(1).lower() in self._lookup:
            return self._lookup[m.group(1).lower()], (m.group(2) or "").strip()
        parts = t[1:].split(None, 1)
        cmd = self._lookup.get(parts[0].lower())
        if cmd is None:
            return None
        return cmd, (parts[1].strip() if len(parts) > 1 else "")

    def classify(self, text: str, busy: bool) -> Decision:
        t = text.strip()
        if not t.startswith("/"):
            return Decision(NOT_A_COMMAND)
        hit = self.resolve(t)
        if hit is None:
            word = t[1:].split(None, 1)[0] if len(t) > 1 else ""
            if not word or "/" in word:  # "/home/me/x ..." is a path, not a command
                return Decision(NOT_A_COMMAND)
            return Decision(REFUSE, notice=f"unknown command /{word} - try /help")
        cmd, args = hit
        if not busy:
            return Decision(RUN_NOW, cmd, args)
        when = cmd.when_for(args)
        if when == INSTANT:
            return Decision(RUN_NOW, cmd, args)
        if when == STEP:
            return Decision(STEP_ACTION, cmd, args, f"/{cmd.name} will apply at the next step")
        return Decision(DEFER, cmd, args, f"/{cmd.name} will run after the current turn")

    def completions(self, prefix: str) -> list[Completion]:
        q = prefix.strip().lstrip("/").lower()
        scored: list[tuple[int, str, Command]] = []
        for cmd in self._cmds.values():
            names = [cmd.name, *cmd.aliases]
            if not q:
                score = 0
            elif any(n == q for n in names):
                score = 0
            elif any(n.startswith(q) for n in names):
                score = 1
            elif any(q in n for n in names):
                score = 2
            elif _subseq(q, cmd.name):
                score = 3
            elif len(q) >= 3 and q in cmd.summary.lower():
                score = 4
            else:
                continue
            scored.append((score, cmd.name, cmd))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [Completion("/" + c.name + ("-<dir>" if c.pattern else ""), c.summary, c.usage) for _, _, c in scored]

    def help_groups(self) -> list[tuple[str, list[Command]]]:
        by: dict[str, list[Command]] = {}
        for c in self._cmds.values():
            by.setdefault(c.group, []).append(c)
        order = [g for g in GROUPS if g in by] + [g for g in by if g not in GROUPS]
        return [(g, by[g]) for g in order]


def _subseq(q: str, s: str) -> bool:
    it = iter(s)
    return all(ch in it for ch in q)


def _plugins_text(args: str = "", app: Any = None) -> str:
    """/plugins: what is installed, enabled and loaded (agent/plugin_host.py)."""
    from agent import plugin_host

    rows = plugin_host.listing()
    if not rows:
        return "no plugins installed. Make one: nightage plugin new my-plugin   (guide: docs/PLUGINS.md)"
    lines = []
    for r in rows:
        lines.append(f"{r['name']} {r['version']}  [{r['status']}]  {r['detail']}")
    return "\n".join(lines)


def _default() -> Registry:
    r = Registry()
    C = Command
    for c in [
        # Chat
        C("web", "Search the web for this question", "/web <question>", "Chat", DEFERRED,
          aliases=("search", "s", "w", "net", "online")),
        C("say", "Send a message into the running code agent", "/say <text>", "Chat", STEP),
        C("thinking", "Show or hide the model's thinking", "/thinking [on|off]", "Chat", INSTANT, aliases=("think",)),
        C("model", "Switch the chat model", "/model", "Chat", DEFERRED),
        # Code
        C("steps", "Change the step limit of the running code task", "/steps <n>", "Code", STEP),
        C("mode", "Change the code agent's mode", "/mode [name]", "Code", STEP),
        C("plan", "Show the current plan (or clear it)", "/plan [clear]", "Code", INSTANT, sub_when={"clear": DEFERRED}),
        C("verify", "Fact-check the last answer against the web", "/verify", "Code", DEFERRED),
        C("undo", "Undo the last file changes (or list them)", "/undo [list]", "Code", DEFERRED, sub_when={"list": INSTANT}),
        # Context & memory
        C("context", "Show what the prompt is made of and the token budget", "/context", "Context & memory", INSTANT, aliases=("ctx",)),
        C("compact", "Compact the conversation context now", "/compact", "Context & memory", STEP),
        C("notes", "Show this session's notes (or clear them)", "/notes [clear]", "Context & memory", INSTANT,
          sub_when={"clear": DEFERRED}),
        C("project", "Show the active project and its notes (or forget them)", "/project [clear]", "Context & memory", INSTANT,
          sub_when={"clear": DEFERRED}),
        # Safety & access
        C("access", "Show folder access and the active level", "/access", "Safety & access", INSTANT),
        C("auto", "Let the agent act in a folder without asking", "/auto-<dir>", "Safety & access", DEFERRED, pattern=True),
        C("medium", "Allow safe actions in a folder, ask for the rest", "/medium-<dir>", "Safety & access", DEFERRED, pattern=True),
        C("ask", "Ask before every action in a folder", "/ask-<dir>", "Safety & access", DEFERRED, pattern=True),
        C("greenlist", "Always allow a folder", "/greenlist-<dir>", "Safety & access", DEFERRED, pattern=True),
        C("ungreenlist", "Remove a folder from the always-allowed list", "/ungreenlist-<dir>", "Safety & access", DEFERRED, pattern=True),
        # Inspect
        C("status", "Model, backend, GPU, context and what is running", "/status", "Inspect", INSTANT),
        C("tasks", "Everything that is running, with stop buttons", "/tasks", "Inspect", INSTANT),
        C("queue", "Show or edit the queued prompts", "/queue [rm N | clear | now N]", "Inspect", INSTANT),
        C("log", "Recent activity log", "/log [filter]", "Inspect", INSTANT),
        C("events", "Filterable event stream", "/events [kind]", "Inspect", INSTANT),
        C("usage", "Tokens and time per turn", "/usage", "Inspect", INSTANT),
        C("transcript", "Show the full transcript including tool details", "/transcript", "Inspect", INSTANT),
        C("why-slow", "Explain where the time went", "/why-slow", "Inspect", INSTANT),
        C("plugins", "List the plugins that are loaded", "/plugins", "Inspect", INSTANT, handler=_plugins_text),
        C("doctor", "Check Ollama, models, GPU and settings", "/doctor", "Inspect", INSTANT),
        C("export", "Export the transcript as Markdown or JSON", "/export [md|json]", "Inspect", INSTANT),
        # Session
        C("help", "Commands grouped by task", "/help [command]", "Session", INSTANT, aliases=("?", "h")),
        C("stop", "Stop the current generation", "/stop", "Session", INSTANT),
        C("pause", "Pause the running task", "/pause", "Session", INSTANT),
        C("resume", "Resume the paused task", "/resume", "Session", INSTANT),
        C("clear", "Start a fresh conversation", "/clear", "Session", DEFERRED),
        C("new", "Start a new chat", "/new", "Session", DEFERRED),
        C("quit", "Leave", "/quit", "Session", DEFERRED, aliases=("exit",)),
    ]:
        r.register(c)
    return r


REGISTRY = _default()
