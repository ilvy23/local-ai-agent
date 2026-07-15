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

from companion.jsonx import extract_json_value


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


class ToolRegistry:
    """Holds the set of tools available to the agent loop."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

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
    from companion.tools.files import (
        LIST_DIR_TOOL,
        READ_FILE_TOOL,
        SEARCH_FILES_TOOL,
        WRITE_FILE_TOOL,
    )
    from companion.tools.system import SYSTEM_STATS_TOOL

    registry = ToolRegistry()
    registry.register(CURRENT_TIME_TOOL)
    registry.register(READ_FILE_TOOL)
    registry.register(LIST_DIR_TOOL)
    registry.register(SEARCH_FILES_TOOL)
    registry.register(WRITE_FILE_TOOL)
    registry.register(SYSTEM_STATS_TOOL)
    if config is not None:
        # Imported here to avoid a circular import (shell -> registry).
        from companion.tools.shell import make_shell_tool

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
