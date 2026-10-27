import json
from pathlib import Path

from agent import tui
from agent.memory.store import Store
from agent.tools.registry import Tool, ToolRegistry


def _write_tool(calls):
    def write(path, content, **k):
        calls.append(path)
        Path(path).write_text(content)
        return f"Wrote {path}"

    return Tool(name="write_file", description="w", handler=write, risk="safe",
                parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["path", "content"]})


def _call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


class PlanClient:
    """ScriptedToolClient whose chat() answers the planning request."""

    def __init__(self, scripted, plan_items):
        from tests.test_tui import ScriptedToolClient

        self.inner = ScriptedToolClient(scripted)
        self.plan_items = plan_items
        self.plan_calls = 0

    def chat_with_tools(self, *a, **k):
        return self.inner.chat_with_tools(*a, **k)

    @property
    def chat_with_tools_calls(self):
        return self.inner.chat_with_tools_calls

    def chat(self, messages, model, stream=True, **kw):
        self.plan_calls += 1
        return iter([json.dumps(self.plan_items)])


BIG_TASK = ("Build a small python package in ~/proj with a calculator module, a command line interface and pytest "
            "tests, make sure the old data file still loads and add a README with usage instructions please.")


def test_invalid_arguments_never_reach_the_tool_or_the_user(tmp_path):
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    calls = []
    registry = ToolRegistry()
    registry.register(_write_tool(calls))
    client = ScriptedToolClient([_call("write_file", path=str(tmp_path / "a.py")),          # content missing
                                 {"role": "assistant", "content": "ok"}])
    store = Store(tmp_path / "a.db")
    console = _ScriptedConsole(["go", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=registry)
    assert calls == []
    assert any("missing required argument content" in str(m.get("content")) for m in client.chat_with_tools_calls[1])
    store.close()


def test_undo_command_restores_what_the_last_prompt_wrote(tmp_path):
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    target = tmp_path / "keep.py"
    target.write_text("original\n")
    new = tmp_path / "new.py"
    registry = ToolRegistry()
    registry.register(_write_tool([]))
    client = ScriptedToolClient([_call("write_file", path=str(target), content="changed\n"),
                                 _call("write_file", path=str(new), content="fresh\n"),
                                 {"role": "assistant", "content": "done"}])
    store = Store(tmp_path / "a.db")
    console = _ScriptedConsole(["edit things", "/undo list", "/undo", "/undo", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=registry)
    out = console.file.getvalue()
    assert target.read_text() == "original\n" and not new.exists()
    assert "2 file(s)" in out and "restored keep.py" in out and "nothing to undo" in out
    store.close()


def test_big_task_gets_a_plan_that_must_be_confirmed_before_done(tmp_path):
    from tests.test_tui import CONFIG, _ScriptedConsole

    registry = ToolRegistry()
    client = PlanClient([
        {"role": "assistant", "content": "All finished!"},                                  # premature
        _call("mark_done", id=1, evidence="ran pytest -q: 3 passed in 0.1s"),
        _call("mark_done", id=2, evidence="README.md exists with a usage section"),
        {"role": "assistant", "content": "Now truly finished."},
    ], ["calculator tests pass", "README has usage"])
    store = Store(tmp_path / "a.db")
    console = _ScriptedConsole([BIG_TASK, "/plan", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=registry)
    out = console.file.getvalue()
    assert client.plan_calls >= 1 and "plan  2 checks to confirm" in out
    assert "plan  0 of 2 confirmed - back to work" in out and "plan  2 of 2 confirmed" in out
    first_call_msgs = client.chat_with_tools_calls[0]
    assert any("ACCEPTANCE LIST" in str(m.get("content")) for m in first_call_msgs if m.get("role") == "system")
    assert any("ACCEPTANCE LIST NOT COMPLETE" in str(m.get("content")) for m in client.chat_with_tools_calls[1])
    assert "Now truly finished." in out
    store.close()


def test_plan_gives_up_after_two_nudges_and_says_so(tmp_path):
    from tests.test_tui import CONFIG, _ScriptedConsole

    client = PlanClient([{"role": "assistant", "content": "done 1"}, {"role": "assistant", "content": "done 2"},
                         {"role": "assistant", "content": "done 3"}], ["thing one works", "thing two works"])
    store = Store(tmp_path / "a.db")
    console = _ScriptedConsole([BIG_TASK, "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=ToolRegistry())
    out = console.file.getvalue()
    assert "plan  2 of 2 items not confirmed" in out and "done 3" in out
    store.close()


def test_project_memory_is_injected_after_working_in_a_repo(tmp_path):
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    proj = tmp_path / "w" / "proj"
    (proj / ".git").mkdir(parents=True)
    store = Store(tmp_path / "a.db")
    from agent import projectmem

    projectmem.ProjectMemory(store.conn).add(str(proj.resolve()), "Run the tests with: make check")
    client = ScriptedToolClient([{"role": "assistant", "content": "ok"}])
    console = _ScriptedConsole([f"work in this repo: {proj}", f"/ask-{proj}", "/project", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=ToolRegistry())
    assert any("PROJECT MEMORY for proj" in str(m.get("content")) and "make check" in str(m.get("content"))
               for m in client.chat_with_tools_calls[0])
    assert "make check" in console.file.getvalue()
    store.close()
