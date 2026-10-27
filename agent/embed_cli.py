"""`nightage embed [list|use <tag>]`: choose the memory (embedding) model.

`list` shows the catalogue (agent/embedmodels.py) with size, memory use, languages, what is installed and what is
recommended for this machine. `use <tag>` pulls the model when missing (after asking), probes its vector size, sets
models.embed and rebuilds the semantic index with `reembed_all`: vectors of different models are never mixed, even
when two models happen to have the same size."""

from __future__ import annotations

import os
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from agent import brand, embedmodels

embed_app = typer.Typer(add_completion=False, invoke_without_command=True,
                        help="Choose the memory (embedding) model: sizes, languages, switch and rebuild.")


def _locale_hint() -> str:
    return os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES") or os.environ.get("LANG") or ""


def _machine_ram_mb() -> int:
    try:
        from agent import sysinfo

        return int(sysinfo.current().ram_total_mb or 0)
    except Exception:  # noqa: BLE001 - a probe; unknown RAM just makes the advice careful
        return 0


def _installed(llm: Any) -> set[str]:
    try:
        return set(llm.installed_sizes())
    except Exception:  # noqa: BLE001
        return set()


def _is_installed(tag: str, installed: set[str]) -> bool:
    from agent.doctor.probes import model_installed

    return model_installed(tag, installed)


def _pull(tag: str) -> int:
    from agent.ollama_cli import do_pull

    return do_pull(tag)


def _mb(v: int | None) -> str:
    return f"{v} MB" if v else "?"


@embed_app.callback()
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        list_models(all_models=False)


@embed_app.command("list")
def list_models(all_models: bool = typer.Option(False, "--all", help="Also show models that are known but not recommended.")) -> None:
    """The embedding models with size, languages and what is installed here."""
    from agent.config import load_config
    from agent.main import _open_llm

    console = Console()
    config = load_config()
    current = config["models"]["embed"]
    rec = embedmodels.recommend(_machine_ram_mb(), None, _locale_hint())
    installed = _installed(_open_llm())
    rows = list(embedmodels.curated()) + (list(embedmodels.OTHERS) if all_models else [])
    if embedmodels.describe(current).tag not in {m.tag for m in rows} and not any(embedmodels._canon(m.tag) == embedmodels._canon(current) for m in rows):
        rows.insert(0, embedmodels.describe(current))
    table = Table("", "Model", "Download", "RAM", "Dim", "Here", "Who it is for", box=None, pad_edge=False)
    table.columns[1].no_wrap = True
    for m in rows:
        marks = []
        if embedmodels._canon(m.tag) == embedmodels._canon(current) or (not m.exact and embedmodels._base(m.tag) == embedmodels._base(current)):
            marks.append("in use")
        if _is_installed(m.tag, installed):
            marks.append("installed")
        if m.tag == rec.tag:
            marks.append("recommended here")
        table.add_row("*" if "in use" in marks else "", m.tag, _mb(m.download_mb), _mb(m.ram_mb),
                      str(m.dim) if m.dim else "?", ", ".join(marks), m.label)
    console.print(table)
    console.print(f"[dim]Switch with: {brand.CMD} embed use <model>   (rebuilds the memory index; the model runs on the CPU, not the GPU)[/dim]")
    cur = embedmodels.describe(current)
    note = embedmodels.warn_if_large(cur, None, _machine_ram_mb())
    if note:
        console.print(f"[yellow]{note}.[/yellow] [dim]Smaller choice: {rec.tag}[/dim]")


@embed_app.command("use")
def use(
    tag: str = typer.Argument(..., help="Embedding model, e.g. all-minilm or bge-m3 (see `embed list`)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Pull and rebuild without asking."),
    pull: bool = typer.Option(True, "--pull/--no-pull", help="Pull the model first when it is not installed."),
) -> None:
    """Switch the memory model: pull it if missing, set models.embed and rebuild the index (never mixes sizes)."""
    from agent import foreground
    from agent.config import load_config
    from agent.main import _open_llm, _open_store
    from agent.memory.reembed import reembed_all
    from agent.settings import set_value

    console = Console()
    config = load_config()
    current = config["models"]["embed"]
    info = embedmodels.describe(tag)
    if not info.exact and info.languages == embedmodels.UNKNOWN:
        console.print(f"[yellow]{tag}[/yellow] is not in the list: a {info.label}. Size, languages and memory use are unknown.")
    if embedmodels._canon(tag) == embedmodels._canon(current):
        console.print(f"[green]{tag} is already the memory model.[/green] (To rebuild the index anyway: {brand.CMD} reembed {tag})")
        return
    if foreground.active(config):
        console.print("[red]A chat is open.[/red] Close it first: the index is rebuilt and a running chat would write vectors of the old size.")
        raise typer.Exit(1)

    llm = _open_llm()
    installed = _installed(llm)
    console.print(f"{tag}: {info.label}. Download {_mb(info.download_mb)}, about {_mb(info.ram_mb)} of RAM while loaded.")
    if not _is_installed(tag, installed):
        if not pull:
            console.print(f"[red]{tag} is not installed[/red] and --no-pull was given. Nothing was changed.")
            raise typer.Exit(1)
        if not yes and not typer.confirm(f"Download {tag} ({_mb(info.download_mb)}) now?", default=True):
            raise typer.Exit(1)
        if _pull(tag) != 0:
            console.print(f"[red]Could not pull {tag}.[/red] Nothing was changed.")
            raise typer.Exit(1)
    try:
        dim = len(llm.embed(["dimension probe"], tag)[0])
    except Exception as exc:  # noqa: BLE001 - any failure means the model cannot be used for memory
        console.print(f"[red]{tag} could not embed text ({type(exc).__name__}: {exc}).[/red] Nothing was changed.")
        raise typer.Exit(1) from None
    with _open_store(config) as store:
        items = store.conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
        console.print(f"New vector size {dim}; the index holds {items} item(s) made with {current}. "
                      "Vectors of different models cannot be compared, so everything is re-embedded.")
        if items and not yes and not typer.confirm("Rebuild the memory index now?", default=True):
            console.print("[dim]Nothing was changed.[/dim]")
            raise typer.Exit(1)
        set_value("embed_model", tag)
        config = load_config()
        try:
            result = reembed_all(store, llm, config, tag, log=lambda m: console.print(f"[dim]{m}[/dim]"))
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
            console.print(f"[red]Stopped while rebuilding ({type(exc).__name__}).[/red] models.embed is now {tag}; "
                          f"finish with: {brand.CMD} reembed {tag} --no-pull")
            raise typer.Exit(1) from None
    console.print(f"[green]Memory model is now {tag}[/green]: {result['reembedded']}/{result['total']} re-embedded "
                  f"at {result['dim']}-dim ({result['failed']} failed).")
    if result["failed"]:
        console.print(f"[yellow]Some items failed; run {brand.CMD} memory repair when the model is available.[/yellow]")
