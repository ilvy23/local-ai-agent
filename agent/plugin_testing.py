"""Test a plugin without running nightage: `load_plugin(folder)` runs its register(api) against throwaway objects
and hands back a FakeAPI that can call what the plugin registered.

    from pathlib import Path
    from agent.plugin_testing import load_plugin

    def test_greet():
        api = load_plugin(Path(__file__).parent, config={"greeting": "Hi"})
        assert api.call_tool("my_greet", name="Ada").startswith("Hi")

It uses the REAL PluginAPI (same validation, same risk rules), only with a private tool registry, command registry,
hook list and data folder, so nothing leaks between tests and nothing touches your real plugin folder. Docs: docs/PLUGINS.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console

from agent import plugin_host
from agent.plugin_api import PluginAPI, PluginError, Runtime
from agent.tools.registry import ToolRegistry
from agent.ui.commands import Registry


class FakeAPI(PluginAPI):
    """The real API plus helpers for tests."""

    @property
    def tools(self) -> dict[str, Any]:
        """name -> Tool (with .risk, .description, .parameters) for every tool the plugin registered."""
        return {t.name: t for t in self._registry.list()}

    @property
    def commands(self) -> dict[str, Any]:
        return {c.name: c for c in self._commands.commands()}

    @property
    def home_items(self) -> list[tuple[str, str, Any]]:
        return [(label, hint, action) for p, label, hint, action in self._rt.home_items if p == self.name]

    def call_tool(self, tool_name: str, /, **arguments: Any) -> str:
        """Run a tool the way the agent does (handler errors come back as text, not exceptions)."""
        return self._registry.execute(tool_name, arguments)

    def run_command(self, name: str, /, args: str = "") -> Any:
        """Run a slash command's handler with `args` and return its result."""
        cmd = self._commands.get(name)
        if cmd is None or cmd.handler is None:
            raise PluginError(f"/{name} is not registered")
        return cmd.handler(args, None)

    def emit(self, event: str, **data: Any) -> None:
        """Fire a hook event (e.g. emit("turn_end", outcome="done")); hook errors are logged, never raised."""
        self._rt.emit(event, **data)

    def run_cli(self, *args: str) -> Any:
        """Invoke the plugin's terminal commands: api.run_cli("hello", "--name", "Ada") -> click Result."""
        from typer.testing import CliRunner

        app = self._rt.cli_apps.get(self.name)
        if app is None:
            raise PluginError("the plugin did not call api.cli()")
        return CliRunner().invoke(app, list(args))


def load_plugin(path: str | Path, config: dict[str, Any] | None = None, data_dir: str | Path | None = None) -> FakeAPI:
    """Run the plugin at `path` (its folder, or a single .py file) through the real loader and return a FakeAPI.
    `config` is the plugin's own settings (what `api.config(key)` sees), not the whole config.yaml."""
    p = Path(path)
    m = plugin_host.read_manifest(p)
    full = {"plugins": {m.id: dict(config or {})}}
    root = Path(data_dir) if data_dir else Path(tempfile.mkdtemp(prefix="plugin-test-"))
    return plugin_host.load_one(m, registry=ToolRegistry(), commands=Registry(), config=full, console=Console(width=100),
                                runtime=Runtime(), root=root, api_class=FakeAPI)  # type: ignore[return-value]
