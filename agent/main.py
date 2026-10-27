"""Typer CLI entry point for agent."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import typer

from agent import brand
from rich.console import Console
from rich.table import Table

from agent import foreground
from agent.config import PROJECT_ROOT, load_config
from agent.llm import OllamaClient
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex
from agent.tui import _unload_models_on_exit, run_repl

from agent.wa import wa_app

app = typer.Typer(add_completion=False, help=brand.TAGLINE)
memory_app = typer.Typer(add_completion=False, help="Manage stored facts about you.")
app.add_typer(memory_app, name="memory")
prompt_app = typer.Typer(add_completion=False, help="Improve or anonymise a prompt, locally.")
app.add_typer(prompt_app, name="prompt")
app.add_typer(wa_app, name="wa")
from agent.embed_cli import embed_app  # noqa: E402

app.add_typer(embed_app, name="embed")

# Optional command groups of the private edition. Absent files (the public export) simply mean no such commands.
try:
    from agent.share import share_app
    from agent.story import story_app
except ImportError:
    PRIVATE_APPS_AVAILABLE = False
else:
    PRIVATE_APPS_AVAILABLE = True
    app.add_typer(story_app, name="story")
    app.add_typer(share_app, name="share")

__version__ = "0.2.0"


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
    """Print the agent version."""
    from agent import update

    typer.echo(update.version_text(__version__))
    typer.echo(f"{brand.COPYRIGHT} - {brand.LICENSE_NAME}")


@app.command()
def governor() -> None:
    """Show whether background AI work would run or pause right now."""
    from agent.governor import game_running, gpu_utilization, pause_reason

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
def code(
    task: str | None = typer.Argument(None, help="A task ('add a /health route and test it'). Leave empty for an interactive session."),
    file: list[str] = typer.Option([], "--file", "-f", help="Focus files for the task (repeatable)."),
    test: str | None = typer.Option(None, "--test", "-t", help="The command that proves the work (default: detected from the project)."),
    folder: str | None = typer.Option(None, "--folder", help="Project folder (default: the git root of the current directory)."),
    new: bool = typer.Option(False, "--new", help="Start a fresh session (the earlier ones stay saved)."),
    resume: bool = typer.Option(False, "--resume", "-r", help="Continue a saved session: this folder's latest, or `--resume ID` (any folder)."),
    here: bool = typer.Option(False, "--here", help="With `sessions`: only this folder's sessions."),
    delete: str | None = typer.Option(None, "--delete", help="With `sessions`: delete the saved session with this id."),
    prune: float | None = typer.Option(None, "--prune", help="With `sessions`: delete sessions untouched for this many days."),
    classic: bool = typer.Option(False, "--classic", help="Use the old sandboxed one-shot builder / test fixer."),
    old_ui: bool = typer.Option(False, "--old-ui", help="Use the previous terminal REPL instead of the new interface."),
    plain: bool = typer.Option(False, "--plain", help="No colours or unicode (also NO_COLOR / TERM=dumb)."),
    keep_model: bool = typer.Option(False, "--keep-model", help="Leave the model loaded in VRAM when the session ends (config: coding.keep_model_loaded)."),
    stages: bool = typer.Option(False, "--stages", help="Run a big task as stages, each in a fresh small context (config: coding.stages = off | auto | on)."),
    plan_first: bool = typer.Option(False, "--plan-first", help="Before running, build and print the stage plan and the acceptance checklist "
                                     "(agent/codeagent/planview.py) and ask to proceed. Implies --stages; declining runs nothing."),
    autopilot: bool = typer.Option(False, "--autopilot", help="Do not stop because the work is long: a supervisor restarts stopped stages with graded recovery "
                                   "(config: coding.autopilot = off | auto). Exit code 0 done, 2 needs a human, 3 partial."),
    steps: int | None = typer.Option(None, "--steps", help="Hard cap of tool steps per prompt (5-500). Without it the agent sizes its own budget per goal."),
    mode: str | None = typer.Option(None, "--mode", help="ask | medium | auto: how much may be done without asking, for this run only (not saved). "
                                    "Autopilot uses medium in a folder that is still on ask."),
    session: str | None = typer.Option(None, "--session", help="With `time`: the event-log session (id or prefix); default the latest one."),
    since: str | None = typer.Option(None, "--since", help="With `time`: only events of the last span, e.g. 30m, 2h, 1d."),
    request: list[str] = typer.Argument(None, help="With `prose`: the passage to write."),
    model: str | None = typer.Option(None, "--model", help="With `prose`: the text model (default: coding.prose_model)."),
    gpu: bool = typer.Option(False, "--gpu", help="With `prose`: let the model use the GPU (default: CPU only, config coding.prose_gpu)."),
) -> None:
    """Autonomous coding agent: explores, edits with exact patches, runs and verifies, in your project folder.

    Sessions are saved after every step and survive a reboot: `nightage code sessions` lists them (all folders),
    `nightage code --resume [ID]` continues one in its folder. `nightage code state [--folder PATH]` prints the verified state of a folder
    (tests, checks, open items, files - measured now, no model). `nightage code time [--session ID] [--since 2h]` prints where
    the time of a run went (prefill, generation, tools, waiting) and the likely causes, from the event log (no model).
    `nightage code prose "request..." [--model M] [--gpu]` tries the two-model prose pipeline without the agent: it asks the small text model
    (coding.prose_model, CPU only unless --gpu) for one passage and prints it (uses no GPU unless asked; the coding model is not started).
    Without a flag, the folder's session of the last 12 h continues. Per-repo freedom (ask / medium / auto, allowed commands, network, installs) is set with /mode,
    /allow ... inside the session or `nightage code-policy`. `--classic` keeps the old sandbox loops."""
    import re

    if task is not None and task.strip() == "sessions":
        from agent.codeagent.session import sessions_cli

        raise typer.Exit(sessions_cli(here, folder, delete, prune))
    if task is not None and task.strip() == "state":
        from agent.codeagent.session import state_cli

        raise typer.Exit(state_cli(folder))
    if task is not None and task.strip() == "prose":
        from agent.codeagent.prose import prose_cli

        raise typer.Exit(prose_cli(" ".join(request or []), model, gpu, folder))
    if task is not None and task.strip() == "time":
        from agent.codeagent.timing import time_cli

        raise typer.Exit(time_cli(session, since))
    if classic:
        from agent.coding_cli import run_code, run_goal_cli

        if not task:
            raise typer.BadParameter("--classic needs a task")
        if file:
            raise typer.Exit(run_code(task, list(file), test))
        raise typer.Exit(run_goal_cli(task))
    from agent.codeagent.session import run_session

    which = None
    if resume:
        which = "latest"
        if task and re.fullmatch(r"[0-9a-f]{3,16}", task.strip()):       # `nightage code --resume 3fa9` - an id, not a task ...
            from agent.codeagent import sessionstore

            if sessionstore.id_exists(task.strip()):                     # ... but only when a saved session has it ("fade", "dead", "cafe" are words)
                which, task = task.strip(), None
    if mode is not None and mode not in ("ask", "medium", "auto"):
        raise typer.BadParameter("--mode must be ask, medium or auto")
    extra = {"keep_model": True} if keep_model else {}
    if stages:
        extra["stages"] = True
    if plan_first:
        extra["plan_first"] = True
    if autopilot:
        extra["autopilot"] = True
    if mode is not None:
        extra["mode"] = mode
    if steps is not None:
        extra["steps"] = max(5, min(steps, 500))
    if old_ui:
        raise typer.Exit(run_session(folder, task, new, list(file) or None, test, which, **extra))
    from agent.ui.code_app import run_code_ui

    # A goal given on the command line normally keeps the dock open for follow-ups; scripts (no TTY) and
    # autopilot runs want the exit code of that one run instead.
    once = bool(task) and (autopilot or not sys.stdin.isatty())
    raise typer.Exit(run_code_ui(
        folder, task, resume=which, stages=stages, autopilot=autopilot, plan_first=plan_first, mode=mode,
        steps=(max(5, min(steps, 500)) if steps is not None else None), new=new, keep_model=keep_model, test=test,
        files=list(file) or None, once=once, config=load_config(), plain=True if plain else None,
    ))


@app.command("code-policy", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def code_policy(ctx: typer.Context) -> None:
    """Show or change what the coding agent may do in a repo: list | show DIR | set DIR mode=auto network=on allow=pytest* | forget DIR."""
    from agent.codeagent.session import policy_cli

    raise typer.Exit(policy_cli(list(ctx.args)))


@app.command("code-eval")
def code_eval(
    seeds: int = typer.Option(3, "--seeds", help="Fix-loop: seeds per task (≥3 for real numbers)."),
    agentic: bool = typer.Option(False, "--agentic", help="Benchmark the autonomous loop instead."),
    trials: int = typer.Option(2, "--trials", help="Agentic: trials per task (slow)."),
) -> None:
    """Benchmark the coding loops against held-out tests.

    Default: the focused fix loop (pass@k, false-success). With --agentic: the
    open-ended autonomous loop on multi-file goals."""
    if agentic:
        from agent.coding.evals.agentic import run_agentic_harness

        run_agentic_harness(trials)
    else:
        from agent.coding.evals.harness import run_harness

        run_harness(seeds)


@app.command("code-model")
def code_model(
    set_to: str | None = typer.Argument(None, help="Model tag to use (omit to pick interactively)."),
) -> None:
    """Show or swap the coding model (a few recommended coders are listed)."""
    from rich.console import Console

    from agent.coding import models

    console = Console()
    if set_to:
        models.set_model(set_to)
        console.print(f"coding model set to {set_to}")
    else:
        models.menu_choose(console)


@prompt_app.callback(invoke_without_command=True)
def prompt_default(ctx: typer.Context) -> None:
    """Prompt studio: improve / plan / anonymise a prompt, guided and local.

    With no sub-command, launches the interactive studio."""
    if ctx.invoked_subcommand is None:
        from agent.promptlab import studio

        studio()


@prompt_app.command("improve")
def prompt_improve(
    text: str = typer.Argument(..., help="Your rough prompt (quote it)."),
) -> None:
    """Quick: sharpen a rough prompt (asks clarifying questions, then you can
    correct it)."""
    from agent.promptlab import cli_improve

    cli_improve(text)


@prompt_app.command("plan")
def prompt_plan(
    text: str = typer.Argument(..., help="Your rough prompt (quote it)."),
) -> None:
    """Quick: turn a rough prompt into a full structured plan prompt."""
    from agent.promptlab import cli_plan

    cli_plan(text)


@prompt_app.command("anon")
def prompt_anon(
    text: str = typer.Argument(..., help="The prompt to anonymise (quote it)."),
    strength: str = typer.Option("heavy", "--strength", "-s",
                                 help="light | heavy | poison | max"),
) -> None:
    """Quick: rewrite a prompt so it reveals nothing about you (light/heavy/poison/max)."""
    from agent.promptlab import cli_anon

    cli_anon(text, strength)


@prompt_app.command("resume")
def prompt_resume() -> None:
    """Resume a saved prompt chat."""
    from agent.promptlab import resume

    resume()


@app.command()
def chat(
    classic: bool = typer.Option(False, "--classic", help="Use the previous terminal chat instead of the new interface."),
    plain: bool = typer.Option(False, "--plain", help="No colours or unicode (also NO_COLOR / TERM=dumb)."),
) -> None:
    """Start an interactive chat session with the local Ollama model."""
    console = Console()
    config = load_config()
    if not classic:
        from agent.ui.chat_app import run_chat_ui

        raise typer.Exit(run_chat_ui(config, plain=True if plain else None, open_store_fn=_open_store))
    client = OllamaClient()

    with foreground.session(config), _open_store(config) as store:
        try:
            run_repl(client, config, console=console, store=store)
        finally:
            # In a finally so /quit, Ctrl-C (even mid-generation) and a crash
            # all free the GPU. run_repl has already distilled by this point,
            # so nothing still needs the models.
            _unload_models_on_exit(console, client, config)


@app.command("web-token")
def web_token() -> None:
    """Print the access token the web GUI asks for (created on first use)."""
    from agent.web.auth import TOKEN_PATH, get_token

    typer.echo(get_token())
    typer.echo(f"(stored in {TOKEN_PATH})", err=True)


def _tailscale_ip() -> str | None:
    """This machine's tailnet IPv4, or None when Tailscale is missing or logged out."""
    import subprocess

    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0] if out else None


@app.command()
def demo(
    delay: float = typer.Option(1.0, help="Speed of the scripted fake agent (smaller is faster)."),
    plain: bool = typer.Option(False, "--plain", help="No colours or unicode (also NO_COLOR / TERM=dumb)."),
) -> None:
    """Preview the new terminal UI with a scripted fake agent. No model or Ollama needed."""
    from agent.ui.demo import run_demo

    raise typer.Exit(run_demo(delay=delay, plain=True if plain else None))


@app.command("web-trust")
def web_trust(
    ip: str = typer.Argument(None, help="Tailnet address of a device that never needs the token."),
    remove: bool = typer.Option(False, "--remove", help="Stop trusting this address."),
) -> None:
    """List, add or remove addresses that skip the web access token (no argument lists them)."""
    from agent.web.auth import set_trusted, trusted_ips

    if ip:
        try:
            ips = set_trusted(ip, not remove)
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from None
    else:
        ips = trusted_ips()
    typer.echo("\n".join(sorted(ips)) or "(none)")


@app.command()
def web(
    host: str = typer.Option(None, help="Address to listen on. Default: your Tailscale address if Tailscale is up, else 127.0.0.1."),
    port: int = typer.Option(8420, help="Port to listen on."),
) -> None:
    """Start the web GUI. Every API call needs the access token (`nightage web-token`)."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("The web GUI needs its extra packages: uv sync --group web", err=True)
        raise typer.Exit(1) from None
    from agent.web.auth import auth_enabled, get_token

    if host is None:
        host = _tailscale_ip() or "127.0.0.1"
    if auth_enabled():
        get_token()                                    # create the token file before the first request
        typer.echo(f"Access token: run `{brand.CMD} web-token`")
    else:
        typer.echo("WARNING: AGENT_WEB_AUTH=off - the API is open to anyone who can reach this address.", err=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.echo(f"Listening on {host}:{port} - reachable from other devices; keep it on your tailnet.", err=True)
    uvicorn.run("agent.web.app:app", host=host, port=port, log_level="info")


@app.command()
def sessions() -> None:
    """List past chat sessions."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        rows = store.list_sessions()

    if not rows:
        console.print(f"No sessions yet. Start one with [bold]{brand.CMD} chat[/bold].")
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
def resume(
    session_id: int | None = typer.Argument(None),
    classic: bool = typer.Option(False, "--classic", help="Use the previous terminal chat instead of the new interface."),
    plain: bool = typer.Option(False, "--plain", help="No colours or unicode (also NO_COLOR / TERM=dumb)."),
) -> None:
    """Resume a past chat session (most recent if no id is given)."""
    console = Console()
    config = load_config()
    if not classic:
        from agent.ui.chat_app import run_chat_ui

        raise typer.Exit(run_chat_ui(config, session_id=session_id, resume=True, plain=True if plain else None, open_store_fn=_open_store))
    client = OllamaClient()

    with foreground.session(config), _open_store(config) as store:
        if session_id is None:
            session_id = store.get_last_session_id()
            if session_id is None:
                console.print(
                    f"[red]There are no sessions yet.[/red] Start one with: {brand.CMD} chat"
                )
                raise typer.Exit(code=1)
        else:
            # No dedicated "session exists" lookup in Store; reuse list_sessions
            # with a high limit since session counts stay small for a local app.
            existing_ids = {row["id"] for row in store.list_sessions(limit=10_000)}
            if session_id not in existing_ids:
                console.print(
                    f"[red]No session with id {session_id}.[/red] Try: {brand.CMD} sessions"
                )
                raise typer.Exit(code=1)

        try:
            run_repl(client, config, console=console, store=store, session_id=session_id)
        finally:
            _unload_models_on_exit(console, client, config)


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


@memory_app.command("check")
def memory_check() -> None:
    """Test that saving and recalling a fact works here (uses a throwaway database, never your own data)."""
    from agent.memory.check import roundtrip

    console = Console()
    config = load_config()
    client = _open_llm()
    model = config["models"]["embed"]
    try:
        client.embed(["probe"], model=model)
        res = roundtrip(config, lambda texts: client.embed(texts, model=model))
    except Exception as exc:  # noqa: BLE001 - the reason is what the user needs
        res = roundtrip(config, None, unavailable=f"not usable ({type(exc).__name__}: {exc})")
    colour = {"ok": "green", "warn": "yellow", "fail": "red"}[res.status]
    console.print(f"[{colour}]{res.status}[/{colour}]  {res.detail}")
    if res.status != "ok":
        console.print(f"[dim]Memory models and their sizes: {brand.CMD} embed list[/dim]")
    raise typer.Exit(0 if res.status != "fail" else 1)


@memory_app.command("process")
def memory_process(
    seconds: float = typer.Option(0.0, "--seconds", help="Stop after about this long (0 = work through everything)."),
) -> None:
    """Finish saving facts from messages and tool results that are still queued (what chats leave when you quit)."""
    from agent.memory import queue

    console = Console()
    config = load_config()
    with _open_store(config) as store:
        before = queue.pending_count(store.conn)
        db_path = store.db_path
    if not before:
        console.print("Nothing is waiting to be remembered.")
        return
    console.print(f"[dim]{before} item(s) waiting; saving facts (Ctrl-C is safe: nothing is lost)...[/dim]")
    queue.drain(db_path, _open_llm(), config, max_seconds=seconds or None)
    with _open_store(config) as store:
        left = queue.pending_count(store.conn)
    console.print(f"[green]Done.[/green] {before - left} handled, {left} left." if left else "[green]Done.[/green] Everything is remembered.")


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
    from agent.memory.distill import JUNK_FACT_RE

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


@memory_app.command("repair")
def memory_repair() -> None:
    """Re-index facts that were stored but never embedded.

    If the embedding model couldn't run when a fact was learned (a full GPU is
    the usual cause), the fact is saved but semantic recall can't see it. This
    finds those and embeds them.
    """
    from agent.memory.distill import _UNEMBEDDED, repair_unembedded

    console = Console()
    config = load_config()
    with _open_store(config) as store:
        vectors = VectorIndex(store)
        if not vectors.available:
            console.print("[red]Vector index unavailable — nothing to repair.[/red]")
            raise typer.Exit(1)
        missing = {
            kind: len(store.conn.execute(sql).fetchall())
            for kind, sql in _UNEMBEDDED.items()
        }
        if not any(missing.values()):
            console.print("[green]Everything is indexed.[/green] Nothing to repair.")
            return
        summary = ", ".join(f"{n} {k}(s)" for k, n in missing.items() if n)
        console.print(f"[yellow]{summary} not searchable.[/yellow] Embedding…")
        fixed = repair_unembedded(store, vectors, OllamaClient(), config)

    if fixed == missing:
        console.print(f"[green]Repaired all {sum(fixed.values())}.[/green]")
    else:
        done = ", ".join(f"{n} {k}(s)" for k, n in fixed.items() if n) or "nothing"
        console.print(
            f"[yellow]Repaired {done}.[/yellow] The rest failed — is Ollama up, and "
            "is there room for the embedding model? Re-run to retry."
        )


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


skills_app = typer.Typer(add_completion=False, help="Manage reusable solution skills.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list() -> None:
    """List all saved skills."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        from agent.memory.skills import list_skills
        all_skills = list_skills(store)

    if not all_skills:
        console.print("No skills saved yet. The agent saves them after completing tasks.")
        return

    table = Table("ID", "Title", "✓", "✗", "Last used", "Tags")
    for s in all_skills:
        table.add_row(
            str(s["id"]),
            s["title"],
            str(s["success_count"]),
            str(s["failure_count"]),
            (s["last_used"] or "never")[:19],
            ", ".join(s["tags"]) or "—",
        )
    console.print(table)


@skills_app.command("show")
def skills_show(skill_id: int) -> None:
    """Show the full details of one skill."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        skill = store.get_skill(skill_id)

    if skill is None:
        console.print(f"[red]No skill with id {skill_id}.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]#{skill['id']} — {skill['title']}[/bold]")
    console.print(f"[dim]created {skill['created_at'][:19]}  ·  "
                  f"✓ {skill['success_count']}  ✗ {skill['failure_count']}[/dim]")
    console.print()
    console.print(f"[bold]Problem:[/bold] {skill['problem_description']}")
    console.print()
    console.print("[bold]Steps:[/bold]")
    for i, step in enumerate(skill["solution_steps"], 1):
        console.print(f"  {i}. {step}")
    if skill["tools_used"]:
        console.print(f"\n[bold]Tools:[/bold] {', '.join(skill['tools_used'])}")
    if skill["tags"]:
        console.print(f"[bold]Tags:[/bold] {', '.join(skill['tags'])}")


@skills_app.command("delete")
def skills_delete(skill_id: int) -> None:
    """Delete a skill by id."""
    console = Console()
    config = load_config()

    with _open_store(config) as store:
        if store.get_skill(skill_id) is None:
            console.print(f"[red]No skill with id {skill_id}.[/red]")
            raise typer.Exit(code=1)
        from agent.memory.skills import delete_skill
        delete_skill(store, skill_id)
    console.print(f"Deleted skill {skill_id}.")


settings_app = typer.Typer(add_completion=False, help="View and edit config settings.")
app.add_typer(settings_app, name="settings")


@settings_app.command("show")
def settings_show() -> None:
    """List the editable settings and their current values."""
    from agent.settings import current

    console = Console()
    table = Table("Setting", "Category", "Value", "What it does")
    for name, category, help_, value in current():
        table.add_row(name, category, str(value), help_)
    console.print(table)
    console.print(
        f"[dim]Change with: {brand.CMD} settings set <setting> <value>  "
        f"— or open the settings page from the home screen (`{brand.CMD}`).[/dim]"
    )


@settings_app.command("set")
def settings_set(name: str = typer.Argument(...), value: str = typer.Argument(...)) -> None:
    """Set a setting, e.g. `settings set chat_model qwen2.5:7b`."""
    from agent.settings import set_value

    console = Console()
    try:
        dotted, coerced = set_value(name, value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]{dotted} = {coerced!r}[/green] (saved to config.yaml)")


@settings_app.command("audit")
def settings_audit() -> None:
    """Check every editable setting actually resolves to a real default (dev check)."""
    from agent.settings import audit

    console = Console()
    problems = audit()
    if not problems:
        console.print("[green]All settings resolve cleanly.[/green]")
        return
    for p in problems:
        console.print(f"[red]{p}[/red]")
    raise typer.Exit(code=1)


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

    from agent.memory.reembed import reembed_all
    from agent.settings import set_value

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


WATCH_UNIT = "agent-watch.service"
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

    from agent.worker_status import read_activity

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

    # "last pass" reuses the same activity marker as "Currently" above, but
    # unconditionally - read_activity() isn't gated on the systemd unit being
    # active, so this still shows something useful even when the worker is
    # currently stopped (e.g. "last pass: syncing whatsapp (14m ago)").
    if act:
        text, ago = act
        when = "just now" if ago < 5 else (f"{ago}s ago" if ago < 120 else f"{ago // 60}m ago")
        console.print(f"[dim]last pass: {text} ({when})[/dim]")
    else:
        console.print("[dim]No pass recorded yet.[/dim]")

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
    from agent.panel import run_panel

    run_panel(interval, once=once, as_json=as_json)


def _home() -> int:
    """The start screen: pick what to do, run it, and come back for the quick ones (doctor, help, settings)."""
    from agent.ui.doctor import needs_onboarding, run_doctor, run_onboarding
    from agent.ui.help import render_help
    from agent.ui.home import run_home

    console = Console()
    config = load_config()
    if needs_onboarding(config):
        run_onboarding(config)
    returning = False
    while True:
        if returning and console.is_terminal:
            console.clear()                               # a fresh start screen, not one printed under the last screen's leftovers
        returning = True
        choice = run_home(config)
        action, args = choice.action, choice.args or {}
        if action == "quit":
            return 0
        if action in ("doctor", "help", "settings"):
            if action == "doctor":
                from agent.doctor.cli import run as run_full_doctor

                run_full_doctor(config)
            elif action == "help":
                render_help(console)
            else:
                from agent import settings as settings_mod

                settings_mod.interactive(console)
            if console.is_terminal:                      # read the report before the screen is cleared
                console.input("[dim]Press Enter to go back to the start screen.[/dim] ")
        elif action == "topic":
            from agent.ui.launcher import run_topic

            run_topic(args.get("topic", "commands"), console, app)
        elif action in ("chat", "continue"):
            _ask_gpu_mode()
            from agent.ui.chat_app import run_chat_ui

            if action == "continue":
                return run_chat_ui(config, session_id=args.get("session_id"), resume=True, open_store_fn=_open_store)
            return run_chat_ui(config, open_store_fn=_open_store)
        elif action == "code":
            _ask_gpu_mode()
            from agent.ui.code_app import run_code_ui

            return run_code_ui(args.get("cwd"), config=config)
        elif action == "research":
            question = args.get("question") or console.input("What should it research? ").strip()
            if question:
                _ask_gpu_mode()
                _run_research(question)
                return 0
        else:
            return 0


@app.command()
def menu() -> None:
    """Open the start screen (same as running the command with no arguments)."""
    raise typer.Exit(_home())


@app.command()
def research(
    question: str = typer.Argument(..., help="The question to research in depth."),
    max_subquestions: int = typer.Option(12, "--max-subquestions", help="Upper bound on sub-questions explored (max 60)."),
    budget_chars: int = typer.Option(20000, "--budget-chars", help="Running-notes character budget before compaction."),
    out_dir: str = typer.Option("deep_research", "--out-dir", help="Folder (relative to cwd) the report is written into."),
) -> None:
    """Deep research: keep a growing todo list of sub-questions, search/read for each, follow what's
    discovered, and write a detailed Markdown report when the list is exhausted or a budget is hit."""
    _run_research(question, max_subquestions, budget_chars, out_dir)


def _run_research(question: str, max_subquestions: int = 12, budget_chars: int = 20000, out_dir: str = "deep_research") -> None:
    from agent.research import deep
    from agent.tools.deep_research import make_complete
    from agent.tools.files import write_file
    from agent.tools.web import make_web_tools

    console = Console()
    config = load_config()
    client = _open_llm()
    model = (config.get("code") or {}).get("model") or config["models"]["chat"]
    search_tool, fetch_tool = make_web_tools(console)

    def progress(event: str, fields: dict) -> None:
        if event == "question_start":
            console.print(f"[bold]{fields['step']}.[/bold] researching: {fields['question']}")
        elif event == "question_done":
            extra = f", +{fields['new_questions']} new" if fields.get("new_questions") else ""
            console.print(f"   [dim]-> {fields['status']}{extra}[/dim]")
        elif event == "compacted":
            console.print(f"   [dim](compacted notes: {fields['before_chars']} -> {fields['after_chars']} chars)[/dim]")
        elif event == "run_done":
            console.print(f"[bold green]done[/bold green]: {fields['findings']} answered, "
                          f"{fields['still_open']} still open ({fields['reason']})")

    out = deep.run_deep_research(
        question, complete=make_complete(client, model), web_search=search_tool.handler, web_fetch=fetch_tool.handler,
        write_file=write_file, dest_dir=out_dir, progress=progress,
        max_steps=max(1, min(max_subquestions, 60)), budget_chars=max(2000, min(budget_chars, 200000)),
    )
    console.print(f"\n[green]{out['write_note']}[/green]")


# Commands that talk to a model: before they start, ask once whether to use one GPU or both
# (and which context size that allows). Everything else - settings, memory, version - never asks.
# Help texts are plain docstrings; keep the command name in sync with brand.CMD.
for _fn in (code, web):
    _fn.__doc__ = (_fn.__doc__ or "").replace("`nightage ", f"`{brand.CMD} ")

_GPU_PROMPT_COMMANDS = {"chat", "resume", "code", "code-eval", "prompt", "audit", "research"}


def _ask_gpu_mode() -> None:
    """Never lets a failed GPU question stop the command itself."""
    try:
        from agent import gpumode

        gpumode.ensure_choice(OllamaClient(), load_config(), Console())
    except Exception:  # noqa: BLE001 - convenience prompt, must not break the command
        pass


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    profile: str = typer.Option(
        "", "--profile", help="Config profile for this run: default | phone (experimental, for small devices; see docs/PORTABILITY.md).",
    ),
) -> None:
    if profile:
        from agent import profiles

        if profile.strip().lower() not in profiles.NAMES:
            raise typer.BadParameter(f"unknown profile {profile!r} (choose: {', '.join(profiles.NAMES)})", param_hint="--profile")
        os.environ[profiles.ENV] = profile.strip().lower()          # read by agent.config.load_config
    if Path(sys.argv[0]).name == "lantern":
        typer.echo(f"note: `lantern` has been renamed to `{brand.CMD}` (this alias keeps working).", err=True)
    if ctx.invoked_subcommand is None:
        raise typer.Exit(_home())
    elif ctx.invoked_subcommand in _GPU_PROMPT_COMMANDS and sys.argv[1:3] not in (["code", "time"], ["code", "prose"]):      # `code time` only reads the event log, `code prose` is CPU-only unless --gpu
        _ask_gpu_mode()


# Plugins: `nightage plugin ...` (new/list/enable/doctor/install) and the CLI commands of enabled plugins.
# Must stay below every command definition so a plugin cannot shadow a built-in one.
from agent import plugin_host  # noqa: E402
from agent.plugin_cli import plugin_app  # noqa: E402

app.add_typer(plugin_app, name="plugin")
from agent.doctor.cli import register as _register_doctor  # noqa: E402
from agent.update_cli import register as _register_update  # noqa: E402

_register_doctor(app)       # `nightage doctor` (modular checks, --deep/--fix/--report/--json)
_register_update(app)       # `nightage update`
plugin_host.attach_cli(app)


if __name__ == "__main__":
    app()
