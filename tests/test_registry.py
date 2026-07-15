import pytest

from companion.tools.registry import Tool, ToolRegistry, default_registry, parse_tool_call_fallback


def _make_tool(name="echo", risk="safe"):
    return Tool(
        name=name,
        description="Echoes back its input.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text: text,
        risk=risk,
    )


def test_register_and_get_returns_the_same_tool():
    registry = ToolRegistry()
    tool = _make_tool()

    registry.register(tool)

    assert registry.get("echo") is tool


def test_get_unknown_tool_returns_none():
    registry = ToolRegistry()

    assert registry.get("nope") is None


def test_list_returns_all_registered_tools():
    registry = ToolRegistry()
    a, b = _make_tool("a"), _make_tool("b")
    registry.register(a)
    registry.register(b)

    assert {t.name for t in registry.list()} == {"a", "b"}


def test_tool_default_risk_is_safe():
    tool = _make_tool()
    assert tool.risk == "safe"


def test_to_ollama_schema_shape():
    registry = ToolRegistry()
    registry.register(_make_tool())

    schema = registry.to_ollama_schema()

    assert schema == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echoes back its input.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]


def test_default_registry_has_current_time_tool():
    registry = default_registry()

    tool = registry.get("current_time")

    assert tool is not None
    assert tool.risk == "safe"
    result = tool.handler()
    assert isinstance(result, str) and result


def test_default_registry_tool_names_and_risks():
    """Pins the security-relevant risk assignments for built-in tools."""
    config = {"safety": {"max_timeout_s": 300}}
    registry = default_registry(config)

    risks = {t.name: t.risk for t in registry.list()}
    assert risks == {
        "current_time": "safe",
        "read_file": "safe",
        "list_dir": "safe",
        "search_files": "safe",
        "write_file": "caution",
        "system_stats": "safe",
        "run_command": "dynamic",
    }


class TestParseToolCallFallback:
    def test_clean_json(self):
        text = '{"tool": "current_time", "arguments": {}}'
        assert parse_tool_call_fallback(text) == {"tool": "current_time", "arguments": {}}

    def test_fenced_json(self):
        text = '```json\n{"tool": "current_time", "arguments": {}}\n```'
        assert parse_tool_call_fallback(text) == {"tool": "current_time", "arguments": {}}

    def test_junk_wrapped_json(self):
        text = 'Sure! {"tool": "current_time", "arguments": {"x": 1}} there you go'
        assert parse_tool_call_fallback(text) == {
            "tool": "current_time",
            "arguments": {"x": 1},
        }

    def test_no_json_returns_none(self):
        assert parse_tool_call_fallback("just a normal reply, no tools here") is None

    def test_json_without_tool_key_returns_none(self):
        assert parse_tool_call_fallback('{"foo": "bar"}') is None

    def test_missing_arguments_defaults_to_empty_dict(self):
        assert parse_tool_call_fallback('{"tool": "current_time"}') == {
            "tool": "current_time",
            "arguments": {},
        }
