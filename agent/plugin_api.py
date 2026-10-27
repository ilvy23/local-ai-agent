"""The plugin API: the ONLY thing a plugin needs to know about.

A plugin's `register(api)` function receives a `PluginAPI`. Everything a plugin can do goes through it:

    api.tool(name, description, parameters, handler, risk="caution")   a tool the model can call
    api.command(name, summary, handler, when="instant", usage="")      a slash command (/name)
    api.cli(typer_app_or_function)                                     `nightage <plugin> ...` commands
    api.home_item(label, hint, action)                                 an entry on the home screen
    api.on(event, fn)                                                  a read-only hook
    api.doctor_check(id, title, fn, group="plugins")                   a check for `nightage doctor` (additive, see docs/DOCTOR.md)
    api.config(key, default=None)                                      your settings (config.yaml: plugins.<name>.<key>)
    api.data_dir()                                                     a folder you may write to
    api.log / api.console                                              logging / a rich Console

This surface is small on purpose and versioned (API_VERSION). It is NOT a sandbox: a plugin is ordinary Python
running in this process with your permissions. What the API does guarantee is that a plugin cannot loosen the
safety rules: it can name a risk level for its own tools, but it has no way to touch the command classifier,
`safety.safe_commands`, greenlists, or another plugin's / a built-in tool. Full guide: docs/PLUGINS.md.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

API_VERSION = 1

RISKS = ("safe", "caution", "dangerous")
DEFAULT_RISK = "caution"
EVENTS = ("turn_start", "turn_end", "tool_result", "session_start")
WHENS = ("instant", "step", "deferred")

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_COMMAND_NAME = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
PLUGIN_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,40}$")

logger = logging.getLogger("nightage.plugin")


class PluginError(Exception):
    """A plugin asked the API for something it does not allow. The message says what to change."""


def nightage_version() -> str:
    try:
        from importlib.metadata import version

        return version("agent")
    except Exception:  # noqa: BLE001 - running from a source checkout
        return "0.1.0"


@dataclass
class Provided:
    """What one plugin registered (shown by `plugin list` and `plugin doctor`)."""

    tools: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    cli: str = ""
    home_items: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    doctor_checks: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.tools:
            parts.append("tools: " + ", ".join(self.tools))
        if self.commands:
            parts.append("commands: " + ", ".join("/" + c for c in self.commands))
        if self.cli:
            parts.append(f"cli: nightage {self.cli}")
        if self.home_items:
            parts.append(f"home: {len(self.home_items)}")
        if self.hooks:
            parts.append("hooks: " + ", ".join(self.hooks))
        if self.doctor_checks:
            parts.append("doctor checks: " + ", ".join(self.doctor_checks))
        return "; ".join(parts) or "nothing"


class Runtime:
    """Process-wide plugin state: hooks, CLI sub-apps, home items, per-plugin results. One global instance (RUNTIME);
    tests build their own so nothing leaks between them."""

    def __init__(self) -> None:
        self.hooks: dict[str, list[tuple[str, Callable[[Any], Any]]]] = {e: [] for e in EVENTS}
        self.cli_apps: dict[str, Any] = {}
        self.home_items: list[tuple[str, str, str, Any]] = []     # (plugin, label, hint, action)
        self.provided: dict[str, Provided] = {}
        self.doctor_checks: list[tuple[str, str, str, Any, str]] = []   # (plugin, id, title, fn, group)
        self.status: dict[str, tuple[str, str]] = {}               # plugin -> (state, detail)
        self.session_started = False
        self._lock = threading.Lock()

    def forget(self, plugin: str, commands: Any = None) -> None:
        """Drop everything `plugin` registered earlier (so loading it again does not duplicate)."""
        with self._lock:
            for event in self.hooks:
                self.hooks[event] = [(p, f) for p, f in self.hooks[event] if p != plugin]
            self.cli_apps.pop(plugin, None)
            self.home_items = [i for i in self.home_items if i[0] != plugin]
            self.doctor_checks = [c for c in self.doctor_checks if c[0] != plugin]
            old = self.provided.pop(plugin, None)
        if old and commands is not None:
            for name in old.commands:
                commands.unregister(name)

    def emit(self, event: str, **data: Any) -> None:
        """Run every hook of `event` with a read-only view of `data`. A hook that raises is logged and skipped."""
        if event not in self.hooks:
            return
        view = MappingProxyType(dict(data))
        for plugin, fn in list(self.hooks[event]):
            try:
                fn(view)
            except Exception as exc:  # noqa: BLE001 - a plugin hook must never break a turn
                logging.getLogger(f"nightage.plugin.{plugin}").warning("hook %s failed: %s", event, exc)


RUNTIME = Runtime()


def emit(event: str, **data: Any) -> None:
    """Called by the app at the hook points. Never raises."""
    try:
        RUNTIME.emit(event, **data)
    except Exception:  # noqa: BLE001
        pass


def clamp_risk(risk: Any) -> str:
    """Whatever a plugin passes becomes one of safe / caution / dangerous. Anything unknown is NOT safe;
    "dynamic" (the shell tool's classifier mode) is not available to plugins and is treated as dangerous."""
    if isinstance(risk, str):
        r = risk.strip().lower()
        if r in RISKS:
            return r
        if r == "dynamic":
            return "dangerous"
    return DEFAULT_RISK


class PluginAPI:
    """What `register(api)` receives. Documented in docs/PLUGINS.md; keep it small and stable."""

    version = API_VERSION

    def __init__(self, name: str, *, path: Path, registry: Any, commands: Any, config: dict[str, Any] | None,
                 console: Any, data_root: Path, runtime: Runtime | None = None, plugin_version: str = "0") -> None:
        self.name = name
        self.plugin_version = plugin_version
        self.path = Path(path)
        self.log = logging.getLogger(f"nightage.plugin.{name}")
        self.console = console
        self._registry = registry
        self._commands = commands
        self._config = config if isinstance(config, dict) else {}
        self._data_root = Path(data_root)
        self._rt = runtime or RUNTIME
        self.provided = Provided()

    # --- tools -------------------------------------------------------------------------------------------
    def tool(self, name: str, description: str, parameters: dict[str, Any] | None, handler: Callable[..., Any],
             risk: str = DEFAULT_RISK) -> None:
        """Give the model a tool. `handler(**arguments)` returns text. `risk` is "safe" (runs without asking),
        "caution" (default: asks) or "dangerous" (asks, and the user must type yes)."""
        from agent.tools.registry import Tool

        if not isinstance(name, str) or not _TOOL_NAME.match(name):
            raise PluginError(f"tool name {name!r}: use lower-case letters, digits and _ (2-64 chars, starting with a letter)")
        if not callable(handler):
            raise PluginError(f"tool {name!r}: handler must be a function")
        if not isinstance(description, str) or not description.strip():
            raise PluginError(f"tool {name!r}: give it a description - the model reads it to decide when to call the tool")
        if self._registry.get(name) is not None:
            raise PluginError(f"tool name {name!r} is already taken (built-in or another plugin); pick another name")
        params = parameters if isinstance(parameters, dict) else {}
        params = {"type": "object", "properties": {}, "required": [], **params}
        if not (isinstance(risk, str) and risk.strip().lower() in RISKS):
            self.log.warning("tool %s: risk %r is not one of %s; using %r", name, risk, RISKS, clamp_risk(risk))
        final = clamp_risk(risk)
        plugin = self.name

        def run(**arguments: Any) -> str:
            try:
                out = handler(**arguments)
            except Exception as exc:  # noqa: BLE001 - the model gets the error as text and can recover
                logging.getLogger(f"nightage.plugin.{plugin}").warning("tool %s failed: %s", name, exc)
                return f"error in plugin tool {name}: {type(exc).__name__}: {exc}"
            return out if isinstance(out, str) else ("" if out is None else str(out))

        self._registry.register(Tool(name=name, description=description.strip(), parameters=params, handler=run, risk=final))
        self.provided.tools.append(name)

    # --- slash commands ----------------------------------------------------------------------------------
    def command(self, name: str, summary: str, handler: Callable[[str], Any], when: str = "instant", usage: str = "") -> None:
        """Add `/name`. `handler(args: str)` returns text to show (or None). `when` is what happens if the user
        types it while a turn is running: "instant" runs now, "step" at the next step, "deferred" after the turn."""
        from agent.ui.commands import Command

        name = (name or "").lstrip("/").lower()
        if not _COMMAND_NAME.match(name):
            raise PluginError(f"command name {name!r}: use lower-case letters, digits and - (starting with a letter)")
        if when not in WHENS:
            raise PluginError(f"command /{name}: `when` must be one of {', '.join(WHENS)}")
        if not callable(handler):
            raise PluginError(f"command /{name}: handler must be a function")
        if self._commands.get(name) is not None:
            raise PluginError(f"/{name} already exists (built-in or another plugin); pick another name")
        plugin = self.name

        def run(args: str = "", app: Any = None) -> Any:
            try:
                return handler(args)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(f"nightage.plugin.{plugin}").warning("command /%s failed: %s", name, exc)
                return f"/{name} failed: {exc}"

        self._commands.register(Command(name=name, summary=summary or "", usage=usage or f"/{name}", group="Plugins",
                                        when=when, handler=run))
        self.provided.commands.append(name)

    # --- terminal commands -------------------------------------------------------------------------------
    def cli(self, app_or_function: Any) -> None:
        """Add `nightage <plugin-name> ...`. Pass a typer.Typer app (many commands) or a plain function (one command,
        its parameters become options)."""
        import typer

        if isinstance(app_or_function, typer.Typer):
            app = app_or_function
        elif callable(app_or_function):
            fn = app_or_function
            app = typer.Typer(add_completion=False, help=(fn.__doc__ or "").strip().split("\n")[0] or None)
            app.callback()(lambda: None)                 # a callback keeps `nightage <plugin> <fn>` from collapsing
            app.command(name=getattr(fn, "__name__", "run").replace("_", "-"))(fn)
        else:
            raise PluginError("api.cli() takes a typer.Typer app or a function")
        if app.registered_callback is None and len(app.registered_commands) == 1:
            app.callback()(lambda: None)                 # otherwise typer drops the command name: `nightage <plugin> <cmd>`
        self._rt.cli_apps[self.name] = app
        self.provided.cli = self.name

    # --- home screen -------------------------------------------------------------------------------------
    def home_item(self, label: str, hint: str, action: Callable[[Any], Any] | list[str]) -> None:
        """Add a line to Home -> Plugins. `action` is a function called with the rich console, or a list of
        `nightage` arguments such as ["my-plugin", "greet"]."""
        if not label or not (callable(action) or (isinstance(action, list) and all(isinstance(a, str) for a in action))):
            raise PluginError("home_item(label, hint, action): action is a function(console) or a list of CLI arguments")
        self._rt.home_items.append((self.name, label, hint or "", action))
        self.provided.home_items.append(label)

    # --- hooks -------------------------------------------------------------------------------------------
    def on(self, event: str, fn: Callable[[Any], Any]) -> None:
        """Run `fn(data)` when `event` happens. `data` is read-only. See docs/PLUGINS.md for each event's keys."""
        if event not in EVENTS:
            raise PluginError(f"unknown event {event!r}; choose one of {', '.join(EVENTS)}")
        if not callable(fn):
            raise PluginError("on(event, fn): fn must be a function")
        self._rt.hooks[event].append((self.name, fn))
        self.provided.hooks.append(event)

    # --- doctor ---------------------------------------------------------------------------------------------
    def doctor_check(self, id: str, title: str, fn: Callable[[], Any], group: str = "plugins") -> None:
        """Add a check to `nightage doctor`. `fn()` returns a string (ok), `(status, detail)` or `(status, detail, hint)` with
        status ok | warn | fail | info | skip. It runs with a time limit; an exception becomes a 'could not run' warning.
        A check can only report: it has no way to run a fix."""
        if not isinstance(id, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,40}", id):
            raise PluginError(f"doctor_check id {id!r}: use lower-case letters, digits, . _ - (starting with a letter)")
        if not callable(fn):
            raise PluginError(f"doctor_check {id!r}: fn must be a function")
        if not isinstance(title, str) or not title.strip():
            raise PluginError(f"doctor_check {id!r}: give it a title")
        self._rt.doctor_checks.append((self.name, id, title.strip(), fn, str(group or "plugins")))
        self.provided.doctor_checks.append(id)

    # --- settings and storage ----------------------------------------------------------------------------
    def config(self, key: str, default: Any = None) -> Any:
        """Your setting `key` from config.yaml (`plugins: {<your-name>: {key: value}}`); `default` if unset.
        Dots reach into nested values: api.config("server.port", 8080)."""
        node: Any = ((self._config.get("plugins") or {}).get(self.name) if isinstance(self._config.get("plugins"), dict) else None)
        for part in str(key).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def data_dir(self) -> Path:
        """A folder just for this plugin (created on first use). Put databases, caches and state here."""
        d = self._data_root / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
