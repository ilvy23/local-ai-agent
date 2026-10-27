"""Plugins load from outside the project, and a broken one can't break chat."""

from __future__ import annotations

import pytest
from rich.console import Console

from agent.plugins import PluginContext, discover, load_plugins, plugin_dir
from agent.tools.registry import Tool, ToolRegistry


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path))
    return tmp_path


def _ctx():
    return PluginContext(
        registry=ToolRegistry(), store=None, vectors=None, llm=None,
        config={}, console=Console(quiet=True),
    )


def _write(path, body):
    path.write_text(body)


GOOD = '''
from agent.tools.registry import Tool

def register(ctx):
    ctx.registry.register(Tool(
        name="hello",
        description="says hello",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda **kw: "hi",
        risk="safe",
    ))
'''


def test_no_plugin_directory_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path / "nope"))
    assert discover() == []
    assert load_plugins(_ctx()) == []


def test_loads_a_single_file_plugin_and_its_tools(plugins):
    _write(plugins / "greeter.py", GOOD)
    ctx = _ctx()
    assert load_plugins(ctx) == ["greeter"]
    assert ctx.registry.get("hello") is not None
    assert ctx.registry.get("hello").handler() == "hi"


def test_loads_a_package_plugin(plugins):
    pkg = plugins / "bigger"
    pkg.mkdir()
    _write(pkg / "__init__.py", GOOD)
    ctx = _ctx()
    assert load_plugins(ctx) == ["bigger"]
    assert ctx.registry.get("hello") is not None


def test_a_plugin_that_explodes_is_skipped_not_fatal(plugins):
    _write(plugins / "boom.py", "def register(ctx):\n    raise RuntimeError('bad plugin')\n")
    _write(plugins / "greeter.py", GOOD)
    ctx = _ctx()
    loaded = load_plugins(ctx)  # must not raise
    assert loaded == ["greeter"]  # the good one still works
    assert ctx.registry.get("hello") is not None


def test_a_module_without_register_is_skipped(plugins):
    _write(plugins / "notaplugin.py", "x = 1\n")
    assert load_plugins(_ctx()) == []


def test_private_and_hidden_entries_are_ignored(plugins):
    _write(plugins / "_scratch.py", GOOD)
    _write(plugins / ".hidden.py", GOOD)
    (plugins / "not_a_package").mkdir()  # a dir with no __init__.py
    assert discover() == []


def test_plugin_dir_is_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path / "custom"))
    assert plugin_dir() == tmp_path / "custom"
    monkeypatch.delenv("AGENT_PLUGIN_DIR")
    assert plugin_dir().name == "plugins"  # ~/.config/agent/plugins
