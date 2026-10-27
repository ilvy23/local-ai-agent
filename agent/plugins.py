"""Optional extensions, loaded from outside the project.

Two kinds live in `~/.config/agent/plugins/` (override: NIGHTAGE_PLUGIN_DIR or AGENT_PLUGIN_DIR):

  * contract plugins - a folder with `plugin.toml` + `plugin.py` (or one .py file with a `PLUGIN = {...}` header) whose
    `register(api)` uses the small stable API in agent/plugin_api.py. Only loaded once enabled. This is what outsiders
    write; see docs/PLUGINS.md and agent/plugin_host.py.
  * directory plugins (the older kind, e.g. the WhatsApp bridge) - described below, loaded exactly as before.

A directory plugin is a Python module or package in `~/.config/agent/plugins/` exposing a
`register(ctx)` function. It gets a PluginContext — the tool registry, the store,
the LLM client, the config — and can add tools the agent is then able to call.

They live outside this repo on purpose. A plugin is yours: it may talk to your
accounts or your private data, and it should never be one `git add` away from
being published. The agent knows nothing about what any of them do.

Nothing here is required — with no plugin directory this is a no-op.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent import osport
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PLUGIN_DIR = "~/.config/agent/plugins" if sys.platform != "win32" else str(osport.config_home() / "agent" / "plugins")


def plugin_dir() -> Path:
    """Where plugins live. NIGHTAGE_PLUGIN_DIR or AGENT_PLUGIN_DIR override (the latter is what the tests set)."""
    raw = os.environ.get("AGENT_PLUGIN_DIR") or os.environ.get("NIGHTAGE_PLUGIN_DIR") or DEFAULT_PLUGIN_DIR
    return Path(raw).expanduser()


@dataclass
class PluginContext:
    """Everything a plugin might need, handed over at registration."""

    registry: Any          # ToolRegistry — add tools here
    store: Any             # Store (the SQLite database)
    vectors: Any           # VectorIndex
    llm: Any               # OllamaClient
    config: dict[str, Any]
    console: Any           # rich Console, for anything that narrates


def discover() -> list[Path]:
    """Plugin entry points on disk, in a stable order."""
    directory = plugin_dir()
    if not directory.is_dir():
        return []
    from agent import plugin_host

    found = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if plugin_host.is_contract(entry):      # loaded by agent/plugin_host.py, only once enabled
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            found.append(entry)
        elif entry.suffix == ".py":
            found.append(entry)
    return found


def load_plugins(ctx: PluginContext) -> list[str]:
    """Import each plugin and call its register(ctx). Returns the names loaded.

    A broken plugin is reported and skipped — someone else's extension must not
    stop you from using the agent.
    """
    entries = discover()
    contract = _load_contract(ctx)
    if not entries:
        return contract

    directory = str(plugin_dir())
    if directory not in sys.path:
        sys.path.insert(0, directory)

    loaded: list[str] = []
    for entry in entries:
        name = entry.stem if entry.suffix == ".py" else entry.name
        try:
            module = importlib.import_module(name)
            register = getattr(module, "register", None)
            if not callable(register):
                logger.warning("Plugin %r has no register() function; skipping.", name)
                continue
            register(ctx)
            loaded.append(name)
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not break chat
            logger.warning("Plugin %r failed to load: %s", name, exc)
            ctx.console.print(f"[yellow]plugin {name!r} failed to load:[/yellow] {exc}")
    return loaded + contract


def _load_contract(ctx: PluginContext) -> list[str]:
    """The enabled plugin.toml plugins (agent/plugin_host.py). Isolated: whatever happens, directory plugins still load."""
    try:
        from agent import plugin_host

        return plugin_host.load_contract_plugins(ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plugin system failed: %s", exc)
        return []


# --- built-in optional features ---------------------------------------------
#
# A directory plugin above is someone's own extension living outside this
# repo. A FeaturePlugin is the opposite: it ships IN this repo (a thin
# launcher for a separate sibling project that may or may not be installed on
# this machine) but still wants to show up as a real,
# queryable "option" - for a menu entry, a `list_plugins()` caller, anything
# that wants to know what's available without hardcoding a name. It registers
# itself once at import time via register_feature(); nothing here discovers
# it from disk, so there is no second plugin-loading mechanism to maintain.


@dataclass
class FeaturePlugin:
    """A built-in optional feature, registered in-process (not discovered
    from a plugin directory). `available` is called fresh every time
    list_plugins() runs - it should be cheap (e.g. an os.path check) since
    nothing here caches it."""

    name: str
    description: str
    available: Callable[[], bool]
    detail: Callable[[], str] = lambda: ""


_features: dict[str, FeaturePlugin] = {}


def register_feature(feature: FeaturePlugin) -> None:
    """Called by a module (at import time) to register
    itself as a listable built-in feature - independent of whether it's
    actually available on this machine right now."""
    _features[feature.name] = feature


def list_plugins() -> list[dict[str, Any]]:
    """Every known plugin-shaped option in one list: directory plugins found
    on disk (always "available" - they're sitting right there, loadable) and
    registered built-in features (whose own `available()` check decides). A
    broken `available`/`detail` callable just means "not available" / "" -
    this must never raise for a caller like the menu."""
    out: list[dict[str, Any]] = []
    for entry in discover():
        name = entry.stem if entry.suffix == ".py" else entry.name
        out.append({"name": name, "kind": "directory_plugin", "available": True,
                     "description": "", "detail": str(entry)})
    try:
        from agent import plugin_host

        for row in plugin_host.listing():
            out.append({"name": row["name"], "kind": "plugin", "available": row["status"] in ("loaded", "enabled"),
                        "description": row["description"], "detail": f"{row['status']}: {row['detail']}"})
    except Exception:  # noqa: BLE001
        pass
    for feature in _features.values():
        try:
            available = bool(feature.available())
        except Exception:  # noqa: BLE001 - a broken check must not break the listing
            available = False
        try:
            detail = str(feature.detail())
        except Exception:  # noqa: BLE001
            detail = ""
        out.append({"name": feature.name, "kind": "builtin_feature", "available": available,
                     "description": feature.description, "detail": detail})
    return out
