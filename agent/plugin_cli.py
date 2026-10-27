"""`nightage plugin ...`: create, list, enable, disable, check and install plugins. Guide: docs/PLUGINS.md.

Exposed for agent/main.py: `plugin_app` (add with app.add_typer(plugin_app, name="plugin")) and, after that,
`plugin_host.attach_cli(app)` so enabled plugins can add their own `nightage <plugin> ...` commands.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from agent import plugin_host
from agent.plugin_api import API_VERSION, PluginError

plugin_app = typer.Typer(add_completion=False, help="Create, list, enable, check and install plugins (guide: docs/PLUGINS.md).")

WARNING = ("A plugin is ordinary Python code that runs inside nightage with YOUR permissions: it can read your files, "
           "use your network and run commands. There is no sandbox. Only enable plugins you wrote or trust.")


def _console() -> Console:
    return Console()


def _config() -> dict:
    try:
        from agent.config import load_config

        return load_config()
    except Exception:  # noqa: BLE001
        return {}


@plugin_app.command("new")
def new(name: str = typer.Argument(..., help="Plugin name: lower-case letters, digits, - and _"),
        path: Path | None = typer.Option(None, "--path", help="Create it here instead of the plugin folder."),
        description: str = typer.Option("", "--description", "-d", help="One-line description for plugin.toml."),
        enable: bool = typer.Option(False, "--enable", help="Also enable it (you wrote it, so no question is asked).")) -> None:
    """Create a working hello-world plugin: a tool, a slash command, a terminal command and a test file."""
    con = _console()
    try:
        folder = plugin_host.scaffold(name, path, description)
    except PluginError as exc:
        con.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    con.print(f"Created {folder}")
    for f in sorted(folder.iterdir()):
        con.print(f"  {f.name}")
    if enable and path is None:
        plugin_host.set_enabled(name, True)
        con.print(f"Enabled. Start nightage and type /{name}, or run: nightage {name} hello")
    elif path is None:
        con.print(f"Next: nightage plugin enable {name}   then type /{name} in the chat (or: nightage {name} hello)")
    else:
        con.print(f"It is outside the plugin folder; install it with: nightage plugin install {folder}")


@plugin_app.command("list")
def list_() -> None:
    """Show every plugin: version, enabled or not, status and what it provides."""
    con = _console()
    rows = plugin_host.listing(_config())
    if not rows:
        con.print(f"No plugins in {plugin_host.plugin_dir()}. Create one:  nightage plugin new my-plugin")
        return
    from rich.table import Table

    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for col in ("name", "version", "enabled", "status", "provides / note"):
        t.add_column(col, overflow="fold")
    for r in rows:
        t.add_row(r["name"], r["version"], r["enabled"], r["status"], r["detail"])
    con.print(t)
    con.print("[dim]Plugins show what they provide once loaded (start nightage, or run `nightage plugin doctor`).[/dim]")


@plugin_app.command("enable")
def enable(name: str, yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask (for scripts).")) -> None:
    """Turn a plugin on. Asks first, because enabling runs its code."""
    con = _console()
    m = plugin_host.find(name)
    if m is None:
        con.print(f"[red]No plugin called {escape(name)}.[/red] See `nightage plugin list`.")
        raise typer.Exit(1)
    why = plugin_host.check_compat(m)
    if why:
        con.print(f"[red]{escape(m.id)} cannot be enabled:[/red] {escape(why)}")
        raise typer.Exit(1)
    if m.id in plugin_host._cfg_list(_config(), "disabled"):
        con.print(f"[yellow]{escape(m.id)} is listed under plugins.disabled in config.yaml; remove it there first.[/yellow]")
        raise typer.Exit(1)
    con.print(f"[bold]{escape(m.id)}[/bold] {escape(m.version)}  {escape(m.description)}")
    con.print(f"  author: {escape(m.author or 'unknown')}   location: {m.path}   files fingerprint: {plugin_host.fingerprint(m)}")
    con.print(f"[yellow]{WARNING}[/yellow]  (plugin API version {API_VERSION})")
    if not yes and not typer.confirm(f"Enable {m.id}?", default=False):
        con.print("Not enabled.")
        raise typer.Exit(1)
    plugin_host.set_enabled(m.id, True)
    con.print(f"Enabled {m.id}. It loads the next time nightage starts. Check it now: nightage plugin doctor {m.id}")


@plugin_app.command("disable")
def disable(name: str) -> None:
    """Turn a plugin off (its files stay)."""
    con = _console()
    m = plugin_host.find(name)
    plugin_host.set_enabled(m.id if m else name, False)
    con.print(f"Disabled {m.id if m else name}.")


@plugin_app.command("doctor")
def doctor(name: str | None = typer.Argument(None, help="One plugin (default: all)")) -> None:
    """Explain why a plugin does or does not load: manifest, syntax, and a trial run of register()."""
    con = _console()
    results = plugin_host.doctor(name, _config())
    mark = {"ok": "[green]ok  [/green]", "warn": "[yellow]warn[/yellow]", "fail": "[red]FAIL[/red]"}
    for level, plugin, msg in results:
        con.print(f"{mark[level]} {escape(plugin)}: {escape(msg)}")
    if any(level == "fail" for level, _, _ in results):
        raise typer.Exit(1)


@plugin_app.command("install")
def install(source: str = typer.Argument(..., help="A local plugin folder / .py file, or a git URL."),
            force: bool = typer.Option(False, "--force", help="Replace an installed plugin of the same name."),
            yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before cloning (for scripts)."),
            ref: str = typer.Option("", "--ref", help="Git branch or tag to install (default: the repository's default branch).")) -> None:
    """Copy a plugin folder in, or `git clone --depth 1` it. Installing does not enable it."""
    con = _console()

    def confirm(question: str) -> bool:
        con.print(f"[yellow]{escape(question)}[/yellow]")
        return yes or typer.confirm("Continue?", default=False)

    try:
        m = plugin_host.install(source, confirm=confirm, force=force, ref=ref)
    except PluginError as exc:
        con.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    con.print(f"Installed {m.id} {m.version} into {m.path}. It is NOT enabled yet.")
    con.print(f"[yellow]{WARNING}[/yellow]")
    con.print(f"Look at the code, then: nightage plugin enable {m.id}")


@plugin_app.command("update")
def update(names: list[str] = typer.Argument(None, help="Plugin name(s) to update."),
           all_plugins: bool = typer.Option(False, "--all", help="Update every plugin that has a git source."),
           check: bool = typer.Option(False, "--check", help="Only look: list which plugins have an update."),
           yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before replacing files. The plugin still has to be "
                                    "enabled again by you (updated code is new code).")) -> None:
    """Update plugins installed from a git URL: shows the changelog and a file-level summary, asks, and keeps the plugin
    OFF until you confirm the new code with the normal enable question."""
    from agent import plugin_update

    code = plugin_update.run_plugin_update(list(names or []), all_plugins=all_plugins, check_only=check, yes=yes, console=_console(),
                                           config=_config())
    if code:
        raise typer.Exit(code)
