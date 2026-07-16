"""Optional extensions, loaded from outside the project.

A plugin is a Python module or package in `~/.config/agent/plugins/` exposing a
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PLUGIN_DIR = "~/.config/agent/plugins"


def plugin_dir() -> Path:
    """Where plugins live. AGENT_PLUGIN_DIR overrides, mainly for tests."""
    return Path(os.environ.get("AGENT_PLUGIN_DIR", DEFAULT_PLUGIN_DIR)).expanduser()


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
    found = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith((".", "_")):
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
    if not entries:
        return []

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
    return loaded
