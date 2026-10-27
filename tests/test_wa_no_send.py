import io

from rich.console import Console

from agent import tui
from agent.tools.registry import Tool


def _send_tool(calls):
    return Tool(name="wa_send_message", description="send", parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                handler=lambda text="", **k: calls.append(text) or "sent", risk="dangerous")


def test_wa_send_tools_are_blocked_and_never_run(monkeypatch):
    monkeypatch.delenv("AGENT_WA_ALLOW_SEND", raising=False)
    calls = []
    tool = _send_tool(calls)
    assert tui._classify_call(tool, {"text": "hi"}, {}) is tui.RiskLevel.BLOCKED
    console = Console(file=io.StringIO(), width=100)
    asked = []
    console.input = lambda prompt="", **k: asked.append(prompt) or "yes"          # even a "yes" must not matter
    from agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool)

    class _Store:
        def add_audit_log(self, **k): pass

    result = tui._execute_tool(registry, _Store(), console, {}, "wa_send_message", {"text": "hello there"})
    assert calls == [] and asked == [] and result.startswith("BLOCKED") and "switched off" in result


def test_only_the_environment_can_lift_the_block(monkeypatch):
    tool = _send_tool([])
    monkeypatch.setenv("AGENT_WA_ALLOW_SEND", "1")
    assert tui._classify_call(tool, {}, {}) is tui.RiskLevel.DANGEROUS
    monkeypatch.setenv("AGENT_WA_ALLOW_SEND", "true")
    assert tui._classify_call(tool, {}, {}) is tui.RiskLevel.BLOCKED                # only the exact value "1"
    other = Tool(name="wa_search_messages", description="x", parameters={}, handler=lambda **k: "", risk="safe")
    monkeypatch.delenv("AGENT_WA_ALLOW_SEND", raising=False)
    assert tui._classify_call(other, {}, {}) is tui.RiskLevel.SAFE                  # reading stays untouched
