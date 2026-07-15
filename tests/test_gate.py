"""Tests for the approval gate in the agent loop.

The gate sits between the model's tool call and execution. It classifies the
command (for the dynamic-risk shell tool) or reads the static Tool.risk, then
prompts the user via the scripted console, and audits every decision including
denials and blocks.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from agent.memory.store import Store
from agent.tools.registry import Tool, ToolRegistry
from agent.tools.shell import make_shell_tool
from agent.tui import run_repl

CONFIG = {
    "models": {"chat": "mixtral:8x7b", "background": "llama3.1:8b", "embed": "nomic-embed-text"},
    "persona": {"name": "Agent", "style": "You are Agent."},
    "memory": {"recall_k": 6, "context_char_budget": 24000},
    "tools": {"max_iterations": 8},
    "safety": {"blocked_patterns": [], "safe_commands": [], "max_timeout_s": 300},
}


class _ScriptedConsole(Console):
    """Console whose .input() replays scripted lines then raises EOFError."""

    def __init__(self, lines):
        super().__init__(file=StringIO(), force_terminal=False)
        self._lines = list(lines)
        self.prompts = []

    def input(self, prompt="", **kwargs):
        self.print(prompt, end="")
        self.prompts.append(str(prompt))
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


class ScriptedToolClient:
    def __init__(self, messages):
        self._messages = list(messages)
        self.chat_with_tools_calls = []

    def chat_with_tools(self, messages, model, tools, **kwargs):
        self.chat_with_tools_calls.append(list(messages))
        return self._messages.pop(0)

    def chat(self, messages, model, stream=True, **kwargs):
        yield ""


def _shell_call(command):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "run_command", "arguments": {"command": command}}}
        ],
    }


def _shell_registry(config=CONFIG):
    registry = ToolRegistry()
    registry.register(make_shell_tool(config))
    return registry


def _run(client, console, store, registry=None):
    run_repl(
        client,
        CONFIG,
        console=console,
        store=store,
        tool_registry=registry if registry is not None else _shell_registry(),
    )


def _console(approval_lines):
    """Build a scripted console: a user turn, then the approval answers."""
    return _ScriptedConsole(["do it", *approval_lines])


def _audit_rows(store):
    return store.conn.execute(
        "SELECT kind, detail, approved, result FROM audit_log ORDER BY id"
    ).fetchall()


# --- SAFE: auto-runs, no prompt -------------------------------------------


def test_safe_command_auto_runs(tmp_path):
    store = Store(tmp_path / "c.db")
    client = ScriptedToolClient([_shell_call("echo hi"), {"role": "assistant", "content": "done"}])
    console = _console([])  # no approval line needed
    run_repl(client, CONFIG, console=console, store=store,
             tool_registry=_shell_registry())

    rows = _audit_rows(store)
    shell_rows = [r for r in rows if r["kind"] == "shell"]
    assert len(shell_rows) == 1
    assert shell_rows[0]["approved"] == 1
    assert "hi" in shell_rows[0]["result"]
    # The user was never prompted for approval (no y/N prompt).
    assert not any("Run this" in p for p in console.prompts)
    store.close()


# --- CAUTION: y/N prompt --------------------------------------------------


def test_caution_approved_runs(tmp_path):
    store = Store(tmp_path / "c.db")
    client = ScriptedToolClient([_shell_call("mkdir sub"), {"role": "assistant", "content": "ok"}])
    console = _console(["y"])
    _run(client, console, store)

    shell_rows = [r for r in _audit_rows(store) if r["kind"] == "shell"]
    assert shell_rows[0]["approved"] == 1
    output = console.file.getvalue()
    assert "mkdir sub" in output
    store.close()


def test_caution_denied_does_not_run(tmp_path):
    store = Store(tmp_path / "c.db")
    ran = {"v": False}

    def handler(command, **kw):
        ran["v"] = True
        return "should not run"

    registry = ToolRegistry()
    registry.register(Tool(name="run_command", description="", parameters={"type": "object"},
                           handler=handler, risk="dynamic"))
    client = ScriptedToolClient([_shell_call("mkdir sub"),
                                 {"role": "assistant", "content": "okay, skipped"}])
    console = _console([""])  # default No
    _run(client, console, store, registry=registry)

    assert ran["v"] is False
    shell_rows = [r for r in _audit_rows(store) if r["kind"] == "shell"]
    assert shell_rows[0]["approved"] == 0
    assert "denied" in shell_rows[0]["result"].lower()
    # The tool message fed back to the model reflects denial.
    tool_msgs = [m for m in store.get_messages(store.get_last_session_id())
                 if m["role"] == "tool"]
    assert "denied" in tool_msgs[0]["content"].lower()
    store.close()


# --- DANGEROUS: must type `yes` -------------------------------------------


def test_dangerous_requires_typed_yes(tmp_path):
    store = Store(tmp_path / "c.db")
    ran = {"v": False}

    def handler(command, **kw):
        ran["v"] = True
        return "ran"

    registry = ToolRegistry()
    registry.register(Tool(name="run_command", description="", parameters={"type": "object"},
                           handler=handler, risk="dynamic"))
    # "y" is NOT enough for DANGEROUS -> denied.
    client = ScriptedToolClient([_shell_call("sudo apt update"),
                                 {"role": "assistant", "content": "skipped"}])
    console = _console(["y"])
    _run(client, console, store, registry=registry)

    assert ran["v"] is False
    shell_rows = [r for r in _audit_rows(store) if r["kind"] == "shell"]
    assert shell_rows[0]["approved"] == 0
    store.close()


def test_dangerous_typed_yes_runs(tmp_path):
    store = Store(tmp_path / "c.db")
    ran = {"v": False}

    def handler(command, **kw):
        ran["v"] = True
        return "ran the thing"

    registry = ToolRegistry()
    registry.register(Tool(name="run_command", description="", parameters={"type": "object"},
                           handler=handler, risk="dynamic"))
    client = ScriptedToolClient([_shell_call("sudo apt update"),
                                 {"role": "assistant", "content": "done"}])
    console = _console(["yes"])
    _run(client, console, store, registry=registry)

    assert ran["v"] is True
    shell_rows = [r for r in _audit_rows(store) if r["kind"] == "shell"]
    assert shell_rows[0]["approved"] == 1
    store.close()


# --- BLOCKED: never runs, no prompt ---------------------------------------


def test_blocked_never_runs_and_is_audited(tmp_path):
    store = Store(tmp_path / "c.db")
    ran = {"v": False}

    def handler(command, **kw):
        ran["v"] = True
        return "should never happen"

    registry = ToolRegistry()
    registry.register(Tool(name="run_command", description="", parameters={"type": "object"},
                           handler=handler, risk="dynamic"))
    client = ScriptedToolClient([_shell_call("rm -rf /"),
                                 {"role": "assistant", "content": "I can't do that"}])
    console = _console([])  # no prompt should be requested
    _run(client, console, store, registry=registry)

    assert ran["v"] is False
    shell_rows = [r for r in _audit_rows(store) if r["kind"] == "shell"]
    assert shell_rows[0]["approved"] == 0
    assert "blocked" in shell_rows[0]["result"].lower()
    tool_msgs = [m for m in store.get_messages(store.get_last_session_id())
                 if m["role"] == "tool"]
    assert "BLOCKED by safety policy" in tool_msgs[0]["content"]
    # No approval prompt was ever shown for a blocked command.
    assert not any("yes" in p.lower() or "run this" in p.lower() for p in console.prompts)
    store.close()


# --- Static-risk non-shell tools gate the same way -------------------------


def test_static_caution_tool_gated(tmp_path):
    store = Store(tmp_path / "c.db")
    ran = {"v": False}

    def handler(**kw):
        ran["v"] = True
        return "wrote file"

    registry = ToolRegistry()
    registry.register(Tool(name="write_file", description="", parameters={"type": "object"},
                           handler=handler, risk="caution"))
    client = ScriptedToolClient([
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "write_file", "arguments": {}}}]},
        {"role": "assistant", "content": "skipped"},
    ])
    console = _console([""])  # default No -> denied
    _run(client, console, store, registry=registry)

    assert ran["v"] is False
    store.close()


def test_static_safe_tool_runs_without_prompt(tmp_path):
    store = Store(tmp_path / "c.db")
    registry = ToolRegistry()
    registry.register(Tool(name="ping", description="", parameters={"type": "object"},
                           handler=lambda **kw: "pong", risk="safe"))
    client = ScriptedToolClient([
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "ping", "arguments": {}}}]},
        {"role": "assistant", "content": "got pong"},
    ])
    console = _console([])
    _run(client, console, store, registry=registry)

    tool_msgs = [m for m in store.get_messages(store.get_last_session_id())
                 if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "pong"
    # Only the REPL's own you> prompt should appear; no approval prompt.
    assert not any("Run this" in p or "yes" in p.lower() for p in console.prompts)
    store.close()
