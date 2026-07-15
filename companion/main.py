"""Typer CLI entry point for companion."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from companion import foreground
from companion.config import PROJECT_ROOT, load_config
from companion.llm import OllamaClient
from companion.memory.store import Store
from companion.memory.vectors import VectorIndex
from companion.tui import run_repl

app = typer.Typer(add_completion=False)
memory_app = typer.Typer(add_completion=False, help="Manage stored facts about you.")
app.add_typer(memory_app, name="memory")

__version__ = "0.1.0"


def _open_store(config: dict) -> Store:
    db_path = Path(config["data"]["db_path"])
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Store(db_path)


def _open_llm() -> OllamaClient:
    """Build the LLM client. Separate helper so tests can monkeypatch it.

    Also marks this process as a foreground session, so any interactive model
    command pauses the background watcher (it yields the GPU to you)."""
    try:
        foreground.begin(load_config())
    except Exception:  # noqa: BLE001 - foreground marking must never break a command
        pass
    return OllamaClient()


@app.command()
def version() -> None:
    """Print the companion version."""
    typer.echo(f"companion {__version__}")


@app.command()
def governor() -> None:
    """Show whether background AI work would run or pause right now."""
    from companion.governor import game_running, gpu_utilization, pause_reason

    console = Console()
    config = load_config()
    reason = pause_reason(config)
    game = game_running()
    util = gpu_utilization()
    console.print(f"Game running:    {game or '[green]no[/green]'}")
    console.print(f"GPU utilisation: {util if util is not None else 'n/a'}%")
    if reason:
        console.print(f"[yellow]Background work would PAUSE: {reason}[/yellow]")
    else:
        console.print("[green]Background work would run (machine is clear).[/green]")


@app.command()
def chat() -> None:
    """Start an interactive chat session with the local Ollama model."""
    console = Console()
    config = load_config()
    client = OllamaClient()

    with foreground.session(config), _open_store(config) as store:
        run_repl(client, config, console=console, store=store)


@app.command()
def sessions() -> None:
    """List past chat sessions."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        rows = store.list_sessions()

    if not rows:
        console.print("No sessions yet. Start one with [bold]companion chat[/bold].")
        return

    table = Table("ID", "Title", "Started", "Messages")
    for row in rows:
        table.add_row(
            str(row["id"]),
            row["title"] or "(untitled)",
            row["started_at"],
            str(row["message_count"]),
        )
    console.print(table)


@app.command()
def resume(session_id: int | None = typer.Argument(None)) -> None:
    """Resume a past chat session (most recent if no id is given)."""
    console = Console()
    config = load_config()
    client = OllamaClient()

    with foreground.session(config), _open_store(config) as store:
        if session_id is None:
            session_id = store.get_last_session_id()
            if session_id is None:
                console.print(
                    "[red]There are no sessions yet.[/red] Start one with: companion chat"
                )
                raise typer.Exit(code=1)
        else:
            # No dedicated "session exists" lookup in Store; reuse list_sessions
            # with a high limit since session counts stay small for a local app.
            existing_ids = {row["id"] for row in store.list_sessions(limit=10_000)}
            if session_id not in existing_ids:
                console.print(
                    f"[red]No session with id {session_id}.[/red] Try: companion sessions"
                )
                raise typer.Exit(code=1)

        run_repl(client, config, console=console, store=store, session_id=session_id)


@memory_app.command("list")
def memory_list() -> None:
    """List all remembered facts."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        facts = store.get_active_facts()

    if not facts:
        console.print("No facts yet. They're distilled automatically as you chat.")
        return

    table = Table("ID", "Fact", "Created")
    for fact in facts:
        table.add_row(str(fact["id"]), fact["content"], fact["created_at"])
    console.print(table)


@memory_app.command("search")
def memory_search(query: str) -> None:
    """Search facts semantically (falls back to substring match)."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        vectors = VectorIndex(store)

        facts: list[dict] = []
        if vectors.available:
            try:
                client = OllamaClient()
                embedding = client.embed([query], model=config["models"]["embed"])[0]
                hits = vectors.search(embedding, k=config["memory"]["recall_k"], kinds=["fact"])
                facts = [{"id": ref, "content": text} for text, _kind, ref, _dist in hits]
            except Exception:  # noqa: BLE001 - fall back to LIKE if embedding is unavailable
                facts = []
        if not facts:
            facts = store.search_facts_like(query)

    if not facts:
        console.print(f"No facts matching [bold]{query}[/bold].")
        return

    table = Table("ID", "Fact")
    for fact in facts:
        table.add_row(str(fact["id"]), fact["content"])
    console.print(table)


@memory_app.command("forget")
def memory_forget(fact_id: int) -> None:
    """Forget a fact by id (marks it inactive)."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        store.deactivate_fact(fact_id)
    console.print(f"Forgot fact {fact_id}.")


@memory_app.command("prune")
def memory_prune(
    yes: bool = typer.Option(False, "--yes", help="Delete without confirmation."),
) -> None:
    """Forget junk 'facts' (file paths, tool outputs, timestamps) the distiller
    wrongly captured — they pollute chat context."""
    from companion.memory.distill import JUNK_FACT_RE

    console = Console()
    config = load_config()
    with _open_store(config) as store:
        facts = store.get_active_facts()
        junk = [f for f in facts if JUNK_FACT_RE.search(f["content"] or "")]
        if not junk:
            console.print("[green]No junk facts found.[/green]")
            return
        console.print(f"[yellow]{len(junk)} junk fact(s):[/yellow]")
        for f in junk[:20]:
            console.print(f"  [dim]{f['content'][:70]}[/dim]")
        if not yes and not typer.confirm(f"Forget these {len(junk)} facts?"):
            return
        for f in junk:
            store.deactivate_fact(f["id"])
    console.print(f"[green]Pruned {len(junk)} junk fact(s).[/green]")


@memory_app.command("add")
def memory_add(text: str) -> None:
    """Manually add a fact (embedded for recall)."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        vectors = VectorIndex(store)

        fact_id = store.add_fact(text, source_session_id=None)
        if vectors.available:
            try:
                client = OllamaClient()
                embedding = client.embed([text], model=config["models"]["embed"])[0]
                vectors.add("fact", fact_id, text, embedding)
            except Exception:  # noqa: BLE001 - fact is stored even if embedding fails
                console.print("[dim](stored without an embedding; Ollama unreachable)[/dim]")
    console.print(f"Added fact {fact_id}: {text}")


_APPROVAL_MARK = {1: "✓", 0: "✗"}


def _truncate(text: str | None, length: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    return text if len(text) <= length else text[: length - 1] + "…"


@app.command()
def audit(
    limit: int = typer.Option(30, "--limit", help="How many recent rows to show."),
    kind: str | None = typer.Option(None, "--kind", help="Filter by kind (e.g. shell, tool)."),
) -> None:
    """Show the audit trail of tools and commands the agent has executed."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        rows = store.get_audit_log(limit=limit, kind=kind)

    if not rows:
        console.print("No audit entries yet. They're recorded as the agent runs tools.")
        return

    table = Table("Time", "Kind", "OK", "Detail", "Result")
    for row in rows:
        table.add_row(
            row["ts"],
            row["kind"],
            _APPROVAL_MARK.get(row["approved"], "–"),
            _truncate(row["detail"], 60),
            _truncate(row["result"], 40),
        )
    console.print(table)


settings_app = typer.Typer(add_completion=False, help="View and edit config settings.")
app.add_typer(settings_app, name="settings")


@settings_app.command("show")
def settings_show() -> None:
    """List the editable settings and their current values."""
    from companion.settings import current

    console = Console()
    table = Table("Setting", "Value", "What it does")
    for name, help_, value in current():
        table.add_row(name, str(value), help_)
    console.print(table)
    console.print("[dim]Change with: companion settings set <setting> <value>[/dim]")


@settings_app.command("set")
def settings_set(name: str = typer.Argument(...), value: str = typer.Argument(...)) -> None:
    """Set a setting, e.g. `settings set chat_model dolphin3:8b`."""
    from companion.settings import set_value

    console = Console()
    try:
        dotted, coerced = set_value(name, value)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]{dotted} = {coerced!r}[/green] (saved to config.yaml)")


@app.command()
def reembed(
    model: str = typer.Argument(
        "bge-m3", help="Embedding model to switch to (default bge-m3, multilingual)."
    ),
    pull: bool = typer.Option(True, "--pull/--no-pull", help="Run `ollama pull <model>` first."),
) -> None:
    """Switch the embedding model and rebuild the whole semantic index.

    Use a multilingual model (bge-m3) for better recall on German/mixed chats.
    Re-embeds every stored item — slow on ~18k messages, and governor-paced, so
    it yields while you game. Set `models.embed` and re-embeds in one go.
    """
    import subprocess

    from companion.memory.reembed import reembed_all
    from companion.settings import set_value

    console = Console()
    config = load_config()

    if pull:
        console.print(f"[dim]pulling {model}…[/dim]")
        if subprocess.call(["ollama", "pull", model]) != 0:
            console.print(f"[red]Couldn't pull {model}.[/red] Is Ollama running?")
            raise typer.Exit(1)

    set_value("embed_model", model)  # persist models.embed = <model>
    config = load_config()
    console.print(f"[dim]re-embedding everything with {model} (Ctrl-C safe to resume)…[/dim]")
    with _open_store(config) as store:
        result = reembed_all(
            store, _open_llm(), config, model,
            log=lambda m: console.print(f"[dim]{m}[/dim]"),
        )
    console.print(
        f"[green]Done:[/green] {result['reembedded']}/{result['total']} re-embedded "
        f"at {result['dim']}-dim ({result['failed']} failed)."
    )


WATCH_UNIT = "companion-watch.service"
worker_app = typer.Typer(add_completion=False, help="Control the background worker (watcher).")
app.add_typer(worker_app, name="worker")


def _systemctl_user(*args: str) -> tuple[int, str]:
    import subprocess

    try:
        r = subprocess.run(
            ["systemctl", "--user", *args], capture_output=True, text=True, timeout=8
        )
        return r.returncode, (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return 1, "systemctl unavailable"


@worker_app.command("status")
def worker_status() -> None:
    """Show whether the background worker is running and what it last did."""
    import subprocess

    from companion.panel import _last_refresh
    from companion.worker_status import read_activity

    console = Console()
    config = load_config()
    active = _systemctl_user("is-active", WATCH_UNIT)[1]
    enabled = _systemctl_user("is-enabled", WATCH_UNIT)[1]
    colour = "green" if active == "active" else "yellow"
    console.print(f"Worker: [{colour}]{active}[/{colour}] ({enabled})")

    # What it's doing right now.
    act = read_activity(config)
    if act and active == "active":
        text, ago = act
        fresh = "just now" if ago < 5 else f"{ago}s ago"
        console.print(f"[bold cyan]▸ Currently:[/bold cyan] {text} [dim]({fresh})[/dim]")

    hb = _last_refresh(config)
    console.print(f"[dim]last pass: {hb}[/dim]" if hb else "[dim]No pass recorded yet.[/dim]")

    # Recent activity from the journal — what it's been working on.
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", WATCH_UNIT, "-n", "8", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=8,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    if out:
        console.print("[dim]— recent activity —[/dim]")
        for line in out.splitlines()[-8:]:
            console.print(f"[dim]{line}[/dim]")


@worker_app.command("stop")
def worker_stop() -> None:
    """Stop the background worker (it won't restart until you start it again)."""
    console = Console()
    rc, msg = _systemctl_user("stop", WATCH_UNIT)
    console.print("[green]Worker stopped.[/green]" if rc == 0 else f"[red]{msg}[/red]")


@worker_app.command("start")
def worker_start() -> None:
    """Start the background worker."""
    console = Console()
    rc, msg = _systemctl_user("start", WATCH_UNIT)
    console.print("[green]Worker started.[/green]" if rc == 0 else f"[red]{msg}[/red]")


@app.command()
def panel(
    interval: float = typer.Option(2.0, "--interval", help="Seconds between refreshes."),
    once: bool = typer.Option(False, "--once", help="Print one frame and exit (ssh/cron)."),
    as_json: bool = typer.Option(False, "--json", help="Emit one JSON snapshot and exit."),
) -> None:
    """Live status dashboard: machine, Ollama, governor, data. (Node/laptop screen.)"""
    from companion.panel import run_panel

    run_panel(interval, once=once, as_json=as_json)


@app.command()
def menu() -> None:
    """Interactive numbered menu over all commands."""
    from companion.menu import run_menu

    run_menu()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from companion.menu import run_menu

        run_menu()


if __name__ == "__main__":
    app()
