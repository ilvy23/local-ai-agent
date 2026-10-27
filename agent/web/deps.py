"""Shared singletons and the per-request tool registry builder for the web
app - mirrors what agent/tui.py's run_repl() assembles once per CLI
session, but built per chat request here since a web server serves many
concurrent sessions rather than one long-lived process per conversation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from agent.config import load_config
from agent.llm import OllamaClient
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex
from agent.tools.registry import ToolRegistry, default_registry
from agent.web.console_shim import WebConsole

_config: dict[str, Any] | None = None
_client: OllamaClient | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def get_store() -> Store:
    """A fresh Store per call - sqlite3 connections aren't safe to share
    across threads/requests, and FastAPI's Depends() re-runs this per
    request anyway, so this matches that lifecycle rather than fighting it.
    """
    config = get_config()
    return Store(config["data"]["db_path"])


def build_registry(
    store: Store,
    vectors: VectorIndex,
    client: OllamaClient,
    config: dict[str, Any],
    session_id: int,
    emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> ToolRegistry:
    """The same tool set agent/tui.py's run_repl assembles inline
    (default_registry + web search + remember_this + take_note + diagram +
    plugins), rebuilt fresh per request since take_note is bound to one
    session_id at construction time and a web server has many concurrent
    sessions, unlike one CLI process per conversation.
    """
    from agent.tools.present import make_present_tool
    from agent.tools.image_gen import make_generate_image_tool
    from agent.tools.create_file import make_create_file_tool
    from agent.tools.send_album import make_send_album_tool
    from agent.tools.send_file import make_send_file_tool
    from agent.tools.web import make_web_tools
    from agent.chat.helpers import _make_remember_tool, _make_take_note_tool

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    console = WebConsole(emit, loop)

    registry = default_registry(config)
    for web_tool in make_web_tools(console):
        registry.register(web_tool)
    registry.register(_make_remember_tool(store, vectors, client, config, console))
    registry.register(_make_take_note_tool(store, session_id, console))
    from agent.tools.calc import CALC_TOOL

    registry.register(CALC_TOOL)
    registry.register(make_present_tool(console, allow_html=bool((config.get("ui") or {}).get("allow_html_blocks"))))
    registry.register(make_send_file_tool(store, session_id, console))
    registry.register(make_send_album_tool(store, session_id, console))
    registry.register(make_create_file_tool(store, session_id, console))
    registry.register(make_generate_image_tool(store, session_id, console, client))
    from agent.tools.deep_research import make_deep_research_tool

    registry.register(make_deep_research_tool(client, config, console, store, session_id))

    from agent.web.agent_runtime import get_runtime

    runtime = get_runtime(session_id)
    if runtime is not None and runtime.plan_holder is not None:         # the acceptance-list tool of this turn's plan gate
        from agent import plan as planmod

        registry.register(planmod.make_tool(runtime.plan_holder, console))

    from agent.plugins import PluginContext, load_plugins

    load_plugins(
        PluginContext(
            registry=registry, store=store, vectors=vectors,
            llm=client, config=config, console=console,
        )
    )
    return registry
