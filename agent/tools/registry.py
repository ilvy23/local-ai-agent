"""Tool definitions and the registry the agent loop calls into.

Ollama's tool-calling support (especially via Mixtral) is imperfect: models
sometimes emit a plain-text JSON object describing the call instead of using
the structured `tool_calls` field. `parse_tool_call_fallback` recovers that
case using the same depth-scanning JSON extractor distillation uses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent.jsonx import extract_json_value


@dataclass
class Tool:
    """A callable the model can invoke.

    `risk` is not consumed by this task; Task 6 uses it to gate execution
    (e.g. "safe" tools run automatically, riskier ones need approval).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    risk: str = "safe"


_DISABLED: tuple[str, ...] = ()


def configure(config: dict[str, Any] | None) -> None:
    """Apply `tools.disabled`: tool names (or fnmatch patterns such as `wa_*`) that no registry will register. Empty by default, so
    nothing changes unless a config (or the phone profile) lists names. Called by load_config."""
    global _DISABLED
    block = (config or {}).get("tools") if isinstance(config, dict) else None
    raw = block.get("disabled") if isinstance(block, dict) else None
    _DISABLED = tuple(str(x) for x in raw) if isinstance(raw, (list, tuple)) else ()


def is_disabled(name: str) -> bool:
    import fnmatch

    return any(fnmatch.fnmatchcase(name, pat) for pat in _DISABLED)


class ToolRegistry:
    """Holds the set of tools available to the agent loop."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._on_call: Callable[[str, dict[str, Any], str], None] | None = None

    def register(self, tool: Tool) -> None:
        if is_disabled(tool.name):            # `tools.disabled` (config / phone profile)
            return
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def set_on_call(
        self, callback: Callable[[str, dict[str, Any], str], None]
    ) -> None:
        """Register a hook invoked after every successful tool execution.

        `callback(tool_name, arguments, result)` receives the tool name, the
        arguments dict passed to the handler, and the string result.  Exceptions
        inside the callback are suppressed so they never interrupt the agent loop.
        Called from _execute_tool in tui.py after the handler returns — the hook
        is the right place for cross-cutting concerns like trajectory logging
        without modifying individual tools.
        """
        self._on_call = callback

    def fire_on_call(self, tool_name: str, arguments: dict[str, Any], result: str) -> None:
        """Invoke the on_call hook if one is registered. Never raises."""
        try:
            from agent import plugin_api

            plugin_api.emit("tool_result", tool=tool_name, arguments=dict(arguments), result=result)
        except Exception:  # noqa: BLE001
            pass
        if self._on_call is None:
            return
        try:
            self._on_call(tool_name, arguments, result)
        except Exception:  # noqa: BLE001
            pass

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call by name. Returns the handler's string result,
        or an error string if the tool name is unknown."""
        tool = self.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        return tool.handler(**arguments)

    def to_ollama_schema(self) -> list[dict[str, Any]]:
        """Return the `tools` array shape expected by Ollama's /api/chat."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]


def _current_time(**_kwargs: Any) -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


CURRENT_TIME_TOOL = Tool(
    name="current_time",
    description="Get the current local date and time.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_current_time,
    risk="safe",
)


def default_registry(config: dict[str, Any] | None = None) -> ToolRegistry:
    """Build the registry with all built-in tools registered.

    When `config` is provided, the shell tool (`run_command`) is registered too;
    it needs the config to honour the `safety.max_timeout_s` cap. Callers that
    only want the memory-safe built-ins can omit config.
    """
    from agent.tools.files import (
        DISK_USAGE_TOOL,
        FIND_FILES_TOOL,
        LIST_DIR_TOOL,
        LIST_DRIVES_TOOL,
        READ_FILE_TOOL,
        REPLACE_IN_FILE_TOOL,
        SEARCH_FILES_TOOL,
        WRITE_FILE_TOOL,
    )
    from agent.tools.media_info import GET_MEDIA_INFO_TOOL
    from agent.tools.system import SYSTEM_STATS_TOOL

    registry = ToolRegistry()
    registry.register(CURRENT_TIME_TOOL)
    registry.register(READ_FILE_TOOL)
    registry.register(LIST_DIR_TOOL)
    registry.register(LIST_DRIVES_TOOL)
    registry.register(DISK_USAGE_TOOL)
    registry.register(SEARCH_FILES_TOOL)
    registry.register(FIND_FILES_TOOL)
    registry.register(WRITE_FILE_TOOL)
    registry.register(REPLACE_IN_FILE_TOOL)
    registry.register(SYSTEM_STATS_TOOL)
    registry.register(GET_MEDIA_INFO_TOOL)
    if config is not None:
        # Imported here to avoid a circular import (shell -> registry).
        from agent.tools.shell import make_shell_tool

        registry.register(make_shell_tool(config))
    return registry


def parse_tool_call_fallback(text: str) -> dict[str, Any] | None:
    """Recover a `{"tool": name, "arguments": {...}}` call from free-form text.

    Returns None if no such JSON object is present. `arguments` defaults to
    `{}` if omitted.

    Accepted false-positive risk: if the model's plain-text reply merely
    quotes a `{"tool": ...}` example (e.g. explaining tool-call syntax to the
    user) rather than making a real call, this will still parse it as one.
    """
    parsed = extract_json_value(text, "{", "}")
    if not isinstance(parsed, dict) or "tool" not in parsed:
        return None
    return {"tool": parsed["tool"], "arguments": parsed.get("arguments", {})}
