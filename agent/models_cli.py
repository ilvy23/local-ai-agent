"""`nightage models [show|use <model>|embed <model>|auto]`: which model does every role use, and change them without editing files."""

from __future__ import annotations

import copy

import typer
from rich.console import Console
from rich.table import Table

from agent import brand, modelpick

models_app = typer.Typer(add_completion=False, invoke_without_command=True,
                         help="Which model each role uses (chat, background, coding, reasoning, memory ...) and how to change it.")


def _state():
    from agent import config as cfgmod
    from agent import ollama_local

    config = cfgmod.load_config()
    info = cfgmod.last_load()
    url = ollama_local.base_url(config)
    inst = modelpick.installed(url, use_cache=False)
    work = copy.deepcopy(config)
    res = modelpick.resolve_config(work, url, models=inst)
    return config, work, res, info, url, inst


def _show() -> None:
    console = Console()
    config, work, res, info, url, inst = _state()
    table = Table("Role", "Model", "Installed", "From", box=None, pad_edge=False)
    for role in modelpick.ALL_ROLES:
        p = res.roles.get(role)
        if p is None:
            continue
        model = p.model or "(uses background)"
        status = "?" if p.installed is None else "yes" if p.installed else "NO"
        table.add_row(role, model, status, {"config": "config.yaml", "auto": "auto (installed)", "default": "default", "profile": "phone profile"}[p.source])
    console.print(table)
    console.print(f"[dim]Config file: {info.get('path')}{'' if info.get('existed') else ' (does not exist yet)'}[/dim]")
    if inst is None:
        console.print(f"[yellow]Ollama does not answer at {url}[/yellow] [dim](start it: {brand.CMD} ollama start)[/dim]")
    for n in res.notices:
        console.print(f"[dim]{n}[/dim]")
    for role, missing in res.ask:
        console.print(f"[yellow]{role}: {missing} is set in the config but not installed.[/yellow] [dim]Fix: {brand.CMD} pull {missing}  or  {brand.CMD} models use <installed model>[/dim]")
    console.print(f"[dim]Change: {brand.CMD} models use <model>  |  {brand.CMD} models embed <model>  |  {brand.CMD} models auto[/dim]")


@models_app.callback()
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _show()


@models_app.command("show")
def show_cmd() -> None:
    """Each role, its model, whether it is installed and where the choice came from."""
    _show()


@models_app.command("use")
def use_cmd(model: str = typer.Argument(..., help="An installed chat model."),
            yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask.")) -> None:
    """Use one model for chat, background, coding, reasoning, unrestricted and weekly (saved in config.yaml, comments kept)."""
    console = Console()
    config, work, res, info, url, inst = _state()
    if inst is not None and not modelpick.has(model, inst):
        console.print(f"[red]{model} is not installed.[/red] Download it first: {brand.CMD} pull {model}")
        raise typer.Exit(1)
    if inst is not None and modelpick.is_embedding(model, url):
        console.print(f"[red]{model} is an embedding (memory) model, not a chat model.[/red] For memory use: {brand.CMD} models embed {model}")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Use {model} for chat, background, coding, reasoning, unrestricted and weekly? (writes {info.get('path')})", default=True):
        raise typer.Exit(1)
    modelpick.use_everywhere(info["path"], model)
    console.print(f"[green]Done.[/green] Every chat role now uses {model}. Check: {brand.CMD} models")


@models_app.command("embed")
def embed_cmd(model: str = typer.Argument(..., help="An embedding (memory) model."),
              yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask.")) -> None:
    """Choose the memory model. This is `embed use`: the memory index is rebuilt, never mixed."""
    from agent.embed_cli import use

    use(model, yes=yes, pull=True)


@models_app.command("auto")
def auto_cmd(yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask.")) -> None:
    """Hand every role back to the automatic choice (the models that are installed)."""
    console = Console()
    _config, _work, _res, info, _url, _inst = _state()
    if not yes and not typer.confirm(f"Set every model role to auto in {info.get('path')}?", default=True):
        raise typer.Exit(1)
    modelpick.set_models(info["path"], {r: "auto" for r in ("chat", "background", "coding", "reasoning", "unrestricted", "embed")})
    console.print("[green]Done.[/green] Roles follow the installed models again.")
