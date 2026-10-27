"""`agent wa` — the PUBLIC WhatsApp surface: link/sync, keyword+semantic search, and the
auto-reply drafting feature (mode off/draft, per-contact allowlist, draft-first / never
auto-sends unless AGENT_WA_ALLOW_SEND=1 is set in the environment).

This is the half of the old `agent/wa.py` monolith meant to ship in the open-source
export. It still depends on the external, never-committed
plugin at ~/.config/agent/plugins/whatsapp for the bridge/ingest/schema code (that part
is unchanged and out of scope here) - it does NOT depend on anything in this repo's
private module (`agent/wa_private.py`). That's the point: this file must work even when
wa_private.py is absent, because a future public export ships this file and drops that one.

Profiles, deep/unrestricted analysis, digests/nudges, persona clone/training/chat and
self-reflection live in `agent/wa_private.py` and attach themselves to `wa_app` (below)
only when that module imports cleanly - see `agent/wa.py`.
"""

from __future__ import annotations

from agent import brand

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agent.llm import OllamaClient
from agent.memory.vectors import VectorIndex
from agent.wa_shared import (
    SEND_OFF,
    BridgeClient,
    BridgeError,
    _PLUGIN_DIR,
    _send_allowed,
    agent_db_path,  # noqa: F401 - re-exported for agent/wa_private.py
    as_jid,
    bridge_allowlist_path,
    load_config,
    open_bridge_client,
    open_llm,
    open_store,
    render_wa_status_table,
    resolve_or_exit,
    wa_local_counts,
    wa_sync,
    warn_if_bridge_elsewhere,
    write_allowlist,
)

wa_app = typer.Typer(add_completion=False,
                     help="WhatsApp: link, sync, search, auto-reply (draft-first; private analysis is a separate build).")


# ------------------------------------------------------------------------------------------------ link / sync


@wa_app.command("bridge")
def wa_bridge() -> None:
    """Run the Node bridge. First run shows a QR to pair; then it live-syncs. Ctrl+C to stop."""
    import subprocess
    bridge_dir = Path(_PLUGIN_DIR) / "whatsapp" / "bridge"
    if not (bridge_dir / "index.js").exists():
        typer.echo(f"No bridge found at {bridge_dir}")
        raise typer.Exit(1)
    typer.echo("Starting WhatsApp bridge.")
    typer.echo("If it shows a QR, scan it: WhatsApp > Settings > Linked Devices > Link a Device.")
    typer.echo("Keep this running for live sync. Ctrl+C to stop.\n")
    try:
        subprocess.run(["node", "index.js"], cwd=bridge_dir, check=True)
    except KeyboardInterrupt:
        typer.echo("\nstopped.")
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(exc.returncode)


@wa_app.command("status")
def wa_status() -> None:
    """Show bridge health/pairing state plus locally-synced counts."""
    console = Console()
    config = load_config()
    bridge = open_bridge_client(config)

    try:
        health = bridge.health()
    except BridgeError as exc:
        console.print(f"[red]Bridge not running.[/red] {exc}")
        with open_store(config) as store:
            counts = wa_local_counts(store)
        render_wa_status_table(console, health=None, counts=counts)
        return

    with open_store(config) as store:
        counts = wa_local_counts(store)

    if not health.get("paired"):
        console.print(
            "[yellow]Bridge is running but not paired yet.[/yellow] "
            f"Run [bold]{brand.CMD} wa pair[/bold] to link WhatsApp."
        )
    render_wa_status_table(console, health=health, counts=counts)


@wa_app.command("pair")
def wa_pair(
    poll_interval: float = typer.Option(2.0, "--poll-interval", help="Seconds between checks."),
    max_attempts: int = typer.Option(150, "--max-attempts", help="Give up after this many polls."),
) -> None:
    """Guide pairing: the QR code shows in the BRIDGE's terminal, not here."""
    console = Console()
    config = load_config()
    bridge = open_bridge_client(config)

    console.print(
        f"Start the bridge if it isn't running: [bold]{brand.CMD} wa bridge[/bold]\n"
        "Then scan the QR code shown in [bold]that[/bold] terminal with WhatsApp "
        "(Linked Devices)."
    )

    for _ in range(max_attempts):
        try:
            health = bridge.health()
        except BridgeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

        if health.get("paired"):
            console.print("[green]Paired![/green]")
            return

        console.print("Waiting for pairing… (Ctrl+C to quit)")
        time.sleep(poll_interval)

    console.print("[yellow]Gave up waiting for pairing.[/yellow] Run this command again anytime.")
    raise typer.Exit(code=1)


def auto_pull(console: Console | None = None) -> int | None:
    """Menu start: pull new messages from the bridge into the DB without embedding them (that is `wa sync`).
    Returns the number of new messages, or None when the bridge/plugin is not reachable. Never raises."""
    if wa_sync is None:
        return None
    try:
        config = load_config()
        bridge = open_bridge_client(config)
        bridge.health()
        with open_store(config) as store:
            result = wa_sync(store, VectorIndex(store), OllamaClient(), config, bridge, embed=False)
    except Exception:  # noqa: BLE001 - the menu must open even if WhatsApp is off
        return None
    if console is not None and result["new_messages"]:
        console.print(f"[dim]WhatsApp: {result['new_messages']} new message(s) pulled (not embedded yet - run Get new messages).[/dim]")
    return result["new_messages"]


@wa_app.command("sync")
def wa_sync_cmd(
    days: int | None = typer.Option(None, "--days", help="Request a backfill of N days first."),
) -> None:
    """Pull new messages/contacts/chats from the bridge into the local DB."""
    console = Console()
    config = load_config()
    bridge = open_bridge_client(config)

    try:
        if days is not None:
            console.print(f"Requesting a {days}-day backfill…")
            bridge.backfill(days)

        console.print("Syncing…")
        with open_store(config) as store:
            vectors = VectorIndex(store)
            client = OllamaClient()
            result = wa_sync(store, vectors, client, config, bridge)
    except BridgeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(
        f"Synced [bold]{result['new_messages']}[/bold] new message(s), "
        f"{result['chats']} chat(s), {result['contacts']} contact(s)."
    )


@wa_app.command("chats")
def wa_chats(
    all_chats: bool = typer.Option(
        False, "--all", help="Include empty chats (0 messages) and broadcasts."
    ),
) -> None:
    """List synced WhatsApp chats with message counts and last activity."""
    console = Console()
    config = load_config()

    having = "" if all_chats else "HAVING message_count > 0"
    where = "" if all_chats else "WHERE c.jid NOT LIKE '%@broadcast'"
    with open_store(config) as store:
        rows = store.conn.execute(
            f"""
            SELECT c.jid AS jid, c.name AS name, c.last_message_at AS last_message_at,
                   COUNT(m.id) AS message_count
            FROM wa_chats c
            LEFT JOIN wa_messages m ON m.chat_jid = c.jid
            {where}
            GROUP BY c.jid
            {having}
            ORDER BY c.last_message_at DESC
            """
        ).fetchall()

    if not rows:
        console.print(f"No synced chats yet. Run [bold]{brand.CMD} wa sync[/bold] first.")
        return

    table = Table("Name", "JID", "Messages", "Last activity")
    for row in rows:
        ts = row["last_message_at"]
        when = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M") if ts else ""
        table.add_row(
            row["name"] or "(unnamed)",
            row["jid"],
            str(row["message_count"]),
            when,
        )
    console.print(table)


@wa_app.command("search")
def wa_search(
    query: str = typer.Argument(..., help="What to search for across your messages."),
    contact: str | None = typer.Option(None, "--contact", help="Limit to one contact."),
) -> None:
    """Search your WhatsApp messages by meaning (semantic, falls back to text)."""
    from agent.memory.vectors import VectorIndex
    from whatsapp.tools import _make_search_tool

    console = Console()
    config = load_config()
    llm = open_llm()
    with open_store(config) as store:
        tool = _make_search_tool(store, VectorIndex(store), llm, config)
        console.print(tool.handler(query=query, contact=contact))


@wa_app.command("recent")
def wa_recent(
    limit: int = typer.Option(15, "--limit", help="How many recent messages to show."),
) -> None:
    """Show the most recently synced WhatsApp messages (what just came in)."""
    console = Console()
    config = load_config()
    with open_store(config) as store:
        rows = store.conn.execute(
            "SELECT m.timestamp, m.from_me, m.text, m.type, "
            "COALESCE(c.name, m.chat_jid) AS chat "
            "FROM wa_messages m LEFT JOIN wa_chats c ON c.jid = m.chat_jid "
            "ORDER BY m.rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        console.print("No messages synced yet. Run [bold]wa sync[/bold].")
        return
    table = Table("When", "Chat", "From", "Message")
    for r in reversed(rows):
        when = datetime.fromtimestamp(r["timestamp"], UTC).strftime("%m-%d %H:%M")
        who = "me" if r["from_me"] else "them"
        text = (r["text"] or f"[{r['type']}]").replace("\n", " ")[:50]
        table.add_row(when, (r["chat"] or "?")[:22], who, text)
    console.print(table)


@wa_app.command("merge-lids")
def wa_merge_lids() -> None:
    """Fold duplicate @lid chats into their real phone-number contact.

    WhatsApp sometimes addresses a DM by an opaque @lid, creating a second chat
    for the same person. This mirrors the bridge's @lid->phone map and merges
    them. Runs automatically after each sync; use this to force it now.
    """
    from whatsapp.lidmerge import merge_lids, store_lid_map

    console = Console()
    config = load_config()
    with open_store(config) as store:
        try:
            bridge = open_bridge_client(config)
            n = store_lid_map(store, bridge.lid_map().get("lid_map", []))
            console.print(f"[dim]refreshed {n} @lid pairs from the bridge[/dim]")
        except BridgeError:
            console.print("[dim]bridge offline — merging with the pairs already stored[/dim]")
        result = merge_lids(store)
    console.print(
        f"[green]Merged {result['merged']} duplicate chat(s)[/green] "
        f"from {result['pairs']} known @lid pair(s)."
    )


# ------------------------------------------------------------------------------------------------ auto-reply drafting (public): mode off/draft, allowlist, draft-first reply


@wa_app.command("reply")
def wa_reply(
    contact: str = typer.Argument(..., help="Contact to draft a reply to (e.g. alice)."),
    incoming: str = typer.Option(
        None, "--incoming", "--as-if", help="Reply as if they sent THIS (for testing)."
    ),
    candidates: int = typer.Option(3, "--candidates", help="How many drafts to weigh."),
    best: bool = typer.Option(
        False, "--best", help="Use the responder model (GPU-resident) — better voice."
    ),
    send: bool = typer.Option(
        False, "--send", help="After drafting, ask and actually SEND it (draft-for-approval)."
    ),
) -> None:
    """Draft a reply AS YOU to a contact, in your real voice. `--send` asks and sends it."""
    from rich.panel import Panel
    from whatsapp import vibe as V
    from whatsapp.responder import draft_reply

    console = Console()
    if send and not _send_allowed():
        console.print(SEND_OFF)
        send = False
    config = load_config()
    llm = open_llm()
    with open_store(config) as store:
        res = resolve_or_exit(store, contact, console)
        if not res:
            return
        if incoming:
            console.print(f"[dim]Imagining {res.name or res.jid} sent: “{incoming}”[/dim]")
        console.print(f"[dim]Reading the vibe of the chat with {res.name or res.jid} (irony, tone) …[/dim]")
        vibe = V.read_vibe(store, llm, config, res.jid, incoming)
        if vibe:
            console.print(Panel(V.format_vibe(vibe), title="Vibe-Check", border_style="magenta"))
        console.print(
            f"[dim]Drafting a reply to {res.name or res.jid} in your voice"
            f"{' (big model, slower)' if best else ''}…[/dim]"
        )
        text = draft_reply(
            store, llm, config, res.jid, incoming=incoming, candidates=candidates,
            high_quality=best, vibe=vibe or {},
        )
    if not text:
        console.print(f"[yellow]Not enough of your own messages to {res.name or res.jid} yet.[/yellow]")
        return
    console.print(f"\n[bold green]Draft reply:[/bold green]\n{text}\n")
    if not send:
        console.print("[dim]Copy-paste it yourself, or re-run with --send to actually send it.[/dim]")
        return
    if not typer.confirm(f"Send this to {res.name or res.jid}?"):
        console.print("[dim]Not sent.[/dim]")
        return
    try:
        r = open_bridge_client(config).send(res.jid, text)
    except BridgeError as exc:
        console.print(f"[red]Bridge error: {exc}[/red]")
        return
    console.print("[green]Sent.[/green]" if r.get("ok")
                  else f"[red]Not sent: {r.get('error')}[/red] (is {res.name} on the send allowlist?)")


_BRIEF_QUESTIONS = [
    ("Doing", "What are you up to today?"),
    ("Availability", "When are you free vs busy? (e.g. 'frei ab 16 uhr, davor schule')"),
    ("Location", "Where are you / going anywhere?"),
    ("Plans", "Any plans with friends? (who / when)"),
    ("Mood", "How are you feeling / what's your energy today?"),
    ("Up for", "Anything you'd be up for if someone asks?"),
    ("Not up for", "Anything you'd turn down today?"),
    ("On your mind", "Anything happening or on your mind? (drama, news, arrangements)"),
    ("Careful with", "Anything the AI should avoid or be careful about?"),
]


@wa_app.command("brief")
def wa_brief() -> None:
    """Answer a few questions so the auto-reply AI knows your day (used to reply
    with confidence — decline clashes, mention plans, match your mood). Blank = skip.

    This is about YOUR own day, not an analysis of anyone else's chats, which is why
    it's part of the public auto-reply feature rather than the private analysis module."""
    from whatsapp.daily_note import set_note

    console = Console()
    config = load_config()
    console.print("[bold cyan]Daily brief[/bold cyan] [dim](press Enter to skip any)[/dim]\n")
    parts = []
    for label, q in _BRIEF_QUESTIONS:
        ans = console.input(f"  [magenta]▸[/] {q} ").strip()
        if ans:
            parts.append(f"- {label}: {ans}")
    if not parts:
        console.print("[dim]Nothing entered — brief unchanged.[/dim]")
        return
    note = "Today's brief:\n" + "\n".join(parts)
    with open_store(config) as store:
        set_note(store, note)
    console.print(f"\n[green]Saved.[/green] The AI will use this today.\n[dim]{note}[/dim]")


@wa_app.command("plans")
def wa_plans(
    note: str = typer.Argument(None, help="Your availability/plans today (blank = show)."),
) -> None:
    """Set or show today's availability note (used to decline clashing plans)."""
    from whatsapp.daily_note import get_note, set_note

    console = Console()
    config = load_config()
    with open_store(config) as store:
        if note:
            set_note(store, note)
            console.print("[green]Saved today's availability.[/green]")
        else:
            console.print(get_note(store) or "[dim]No availability noted today.[/dim]")


# ------------------------------------------------------------------------------------------------ auto-reply state machine (public): mode off/draft/auto, speed, allowlist, loop, opener


auto_app = typer.Typer(add_completion=False, help="Autonomous auto-reply (allowlisted contacts).")
wa_app.add_typer(auto_app, name="auto")


@auto_app.command("sent")
def wa_auto_sent(
    days: int = typer.Option(0, "--days", help="How many days back (0 = today only)."),
) -> None:
    """What the auto-responder actually sent, and to whom — the AI's side of the chat."""
    from datetime import datetime, timedelta

    console = Console()
    config = load_config()
    start = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    start -= timedelta(days=days)
    with open_store(config) as store:
        rows = store.conn.execute(
            "SELECT contact_jid, draft, reason, send_at FROM wa_autoreply "
            "WHERE status='sent' AND send_at>=? ORDER BY send_at",
            (int(start.timestamp()),),
        ).fetchall()
        names = {}
        for jid in {r["contact_jid"] for r in rows}:
            n = store.conn.execute(
                "SELECT name FROM wa_chats WHERE jid=? UNION SELECT name FROM wa_contacts WHERE jid=?",
                (jid, jid),
            ).fetchone()
            names[jid] = (n["name"] if n else None) or jid
    if not rows:
        console.print(f"[dim]Nothing sent {'today' if not days else f'in the last {days+1} days'}.[/dim]")
        return
    table = Table("when", "to", "message", "why")
    for r in rows:
        when = datetime.fromtimestamp(r["send_at"]).astimezone().strftime(
            "%H:%M" if not days else "%m-%d %H:%M")
        why = {"opener": "started chat", "nudge": "revived chat",
               "urgent": "reply (urgent)", "instant": "reply"}.get(r["reason"], r["reason"] or "reply")
        table.add_row(when, names[r["contact_jid"]], r["draft"] or "", why)
    console.print(table)
    console.print(f"[dim]{len(rows)} message(s).[/dim]")


@auto_app.command("status")
def wa_auto_status() -> None:
    """Show the auto-reply mode + pending/held drafts + today's sends."""
    from whatsapp.autoreply import _sent_today, pending

    console = Console()
    config = load_config()
    ac = config["whatsapp"].get("auto_reply", {})
    contacts = config["whatsapp"].get("auto_reply_contacts") or []
    mode = ac.get("mode", "off")
    colour = {"off": "dim", "draft": "yellow", "auto": "green"}.get(mode, "white")
    speed = ac.get("speed", "normal")
    timing = ("[magenta]instant (no delay)[/magenta]" if speed == "instant"
              else f"delay {ac.get('delay_min_s',600)//60}-{ac.get('delay_max_s',900)//60} min")
    cap = ac.get("daily_cap", 0)
    console.print(f"Auto-reply: [{colour}]{mode}[/{colour}]  contacts: {contacts}  "
                  f"{timing}  cap {f'{cap}/day' if cap else 'unlimited'}")
    with open_store(config) as store:
        console.print(f"[dim]sent today: {_sent_today(store)}[/dim]")
        rows = pending(store)
    if not rows:
        console.print("[dim]No pending/held drafts.[/dim]")
        return
    table = Table("id", "status", "their message", "drafted reply")
    for r in rows:
        table.add_row(str(r["id"]), r["status"], (r["incoming_text"] or "")[:34],
                      (r["draft"] or "")[:44])
    console.print(table)


def _set_auto(key: str, value: str) -> None:
    import yaml

    from agent.config import DEFAULT_CONFIG_PATH

    config = load_config()
    config["whatsapp"].setdefault("auto_reply", {})[key] = value
    DEFAULT_CONFIG_PATH.write_text(yaml.dump(config, sort_keys=False))


def _set_auto_mode(mode: str) -> None:
    _set_auto("mode", mode)


@auto_app.command("speed")
def wa_auto_speed(
    speed: str = typer.Argument(..., help="instant = reply immediately; normal = 10-15 min delay."),
) -> None:
    """Switch reply timing. 'instant' skips the delay (the confidence gate still applies)."""
    console = Console()
    if speed not in ("instant", "normal"):
        console.print("[red]speed must be 'instant' or 'normal'.[/red]")
        raise typer.Exit(1)
    _set_auto("speed", speed)
    if speed == "instant":
        console.print("[magenta]INSTANT mode — replies go out as soon as they're drafted.[/magenta]")
    else:
        console.print("[green]NORMAL mode — 10-15 min delay (urgent messages still jump it).[/green]")


@auto_app.command("on")
def wa_auto_on() -> None:
    """Turn ON autonomous sending (after the delay + confidence gate)."""
    console = Console()
    if not _send_allowed():
        console.print(SEND_OFF)
        raise typer.Exit(1)
    config = load_config()
    if not (config["whatsapp"].get("auto_reply_contacts") or []):
        console.print("[red]No allowlisted contacts.[/red] Set whatsapp.auto_reply_contacts first.")
        raise typer.Exit(1)
    if not typer.confirm("Enable AUTONOMOUS sending to the allowlisted contact(s)?"):
        return
    _set_auto_mode("auto")
    console.print("[green]Auto-reply is ON (auto).[/green] Kill it any time: wa auto off")


@auto_app.command("off")
def wa_auto_off() -> None:
    """Kill switch — stop all auto-reply activity."""
    _set_auto_mode("off")
    Console().print("[green]Auto-reply OFF.[/green]")


@auto_app.command("draft")
def wa_auto_draft() -> None:
    """Draft mode — draft + hold replies for review, never send."""
    _set_auto_mode("draft")
    Console().print("[yellow]Auto-reply in DRAFT mode (holds, never sends).[/yellow]")


@auto_app.command("flush")
def wa_auto_flush() -> None:
    """One full auto-reply cycle: pull new messages from the bridge → draft +
    schedule → send any due. Runs every ~2 min from cron so drafts happen fast and
    sends land on time even when nobody's around.

    It syncs itself rather than relying on the watcher: if the watcher dies, the
    auto-replier must still see incoming messages, or it silently answers nothing."""
    if not _send_allowed():
        Console().print(SEND_OFF + " Kein Auto-Antwort-Zyklus.")
        return
    from agent.memory.vectors import VectorIndex
    from whatsapp.watch import _pick_ollama_host
    from whatsapp.autoreply import flush_due, handle_inbound, nudge_if_quiet
    from whatsapp.ingest import sync as wa_sync_

    config = load_config()
    # Same backend choice as the watcher: the PC's GPU over the reverse tunnel if
    # it's up, else local. NOT a bare OllamaClient() — that pins us to the laptop's
    # CPU, where the 5.4GB responder gets OOM-killed in 7GB of RAM.
    llm = OllamaClient(base_url=_pick_ollama_host())
    with open_store(config) as store:
        bridge = open_bridge_client(config)
        try:
            wa_sync_(store, VectorIndex(store), llm, config, bridge)
        except Exception as exc:  # noqa: BLE001 - a failed sync must not stop a due send
            Console().print(f"[yellow]sync failed: {exc}[/yellow]")
        handle_inbound(store, llm, config)
        n = flush_due(store, config, bridge)
        # Loop mode (off by default): revive a chat that's gone quiet.
        n += nudge_if_quiet(store, llm, config, bridge)
    if n:
        Console().print(f"[green]Sent {n} message(s).[/green]")


allow_app = typer.Typer(add_completion=False, help="Who the auto-responder may text.")
auto_app.add_typer(allow_app, name="allow")


@allow_app.command("list")
def wa_auto_allow_list() -> None:
    """Show everyone the auto-responder is allowed to text."""
    console = Console()
    config = load_config()
    jids = config["whatsapp"].get("auto_reply_contacts") or []
    if not jids:
        console.print("[dim]Nobody. The auto-responder can't text anyone.[/dim]")
        return
    table = Table("name", "jid", "in bridge allowlist?")
    bridge_jids = set()
    path = bridge_allowlist_path(config)
    if path.exists():
        bridge_jids = {line.strip() for line in path.read_text().split() if line.strip()}
    with open_store(config) as store:
        for entry in jids:
            jid = as_jid(store, entry)
            if not jid:
                table.add_row(entry, "[red]unresolved[/red]", "[red]NO[/red]")
                continue
            row = store.conn.execute(
                "SELECT name FROM wa_chats WHERE jid=? UNION SELECT name FROM wa_contacts WHERE jid=?",
                (jid, jid),
            ).fetchone()
            ok = "[green]yes[/green]" if jid in bridge_jids else "[red]NO — send would be blocked[/red]"
            table.add_row((row["name"] if row else None) or "?", jid, ok)
    console.print(table)


@allow_app.command("add")
def wa_auto_allow_add(
    contact: str = typer.Argument(..., help="Name or jid, e.g. 'alice'."),
) -> None:
    """Allow the auto-responder to text this person. Asks for confirmation."""
    from whatsapp.tools import resolve_contact

    console = Console()
    config = load_config()
    with open_store(config) as store:
        res = resolve_contact(store, contact)
        if res.ambiguous:
            console.print(f"[yellow]{contact!r} is ambiguous:[/yellow]")
            for jid, name in res.matches:
                console.print(f"  {name or '?'}  {jid}")
            console.print("[dim]Re-run with the exact jid.[/dim]")
            raise typer.Exit(1)
        if not res.jid:
            console.print(f"[red]No contact matching {contact!r}.[/red]")
            raise typer.Exit(1)
        if res.jid.endswith("@g.us"):
            console.print("[red]Groups are not supported — DMs only.[/red]")
            raise typer.Exit(1)
        n = store.conn.execute(
            "SELECT COUNT(*) FROM wa_messages WHERE chat_jid=? AND from_me=1", (res.jid,)
        ).fetchone()[0]

    jids = list(config["whatsapp"].get("auto_reply_contacts") or [])
    with open_store(config) as store:
        if res.jid in {as_jid(store, e) for e in jids}:
            console.print(f"[dim]{res.name or res.jid} is already allowed.[/dim]")
            return
    ac = config["whatsapp"].get("auto_reply", {})
    console.print(f"\n[bold]{res.name or res.jid}[/bold]  ({res.jid})")
    console.print(f"  {n} messages from you — the AI copies your voice from these.")
    if n < 20:
        console.print("  [yellow]Thin history: replies will sound generic and off.[/yellow]")
    console.print("\n[yellow]This lets the AI text them as you, unprompted.[/yellow]")
    console.print(f"  mode: [bold]{ac.get('mode')}[/bold]   loop: "
                  f"[bold]{'on' if (ac.get('loop') or {}).get('enabled') else 'off'}[/bold]")
    if ac.get("mode") == "auto":
        console.print("  [red]Auto mode is ON — it will send to them without asking you.[/red]")
    if not typer.confirm(f"\nAllow the AI to text {res.name or res.jid} as you?"):
        console.print("[dim]Nothing changed.[/dim]")
        return
    with open_store(config) as store:
        write_allowlist(store, config, [*jids, res.jid])
    console.print(f"[green]Added {res.name or res.jid}.[/green]")
    warn_if_bridge_elsewhere(console, config)


@allow_app.command("remove")
def wa_auto_allow_remove(
    contact: str = typer.Argument(..., help="Name or jid to remove."),
) -> None:
    """Stop the auto-responder from texting this person."""
    from whatsapp.tools import resolve_contact

    console = Console()
    config = load_config()
    jids = list(config["whatsapp"].get("auto_reply_contacts") or [])
    with open_store(config) as store:
        res = resolve_contact(store, contact)
        # Match on the resolved jid, so 'alice' removes the entry whether the config
        # stored a name or a jid.
        keep = [e for e in jids if as_jid(store, e) != res.jid and e != contact]
        target = res.jid if len(keep) < len(jids) else None
        if not target:
            console.print(f"[red]{contact!r} is not in the allowlist.[/red]")
            raise typer.Exit(1)
        write_allowlist(store, config, keep)
        # Cancel anything already queued for them.
        store.conn.execute(
            "UPDATE wa_autoreply SET status='cancelled', reason='removed from allowlist' "
            "WHERE contact_jid=? AND status IN ('held','scheduled')", (target,))
        store.conn.commit()
    console.print(f"[green]Removed {res.name or target}.[/green] Pending replies cancelled.")
    warn_if_bridge_elsewhere(console, config)


alias_app = typer.Typer(add_completion=False, help="Nicknames for contacts without a synced name.")
wa_app.add_typer(alias_app, name="alias")


@alias_app.command("set")
def wa_alias_set(
    alias: str = typer.Argument(..., help="Nickname to use, e.g. 'Alice'."),
    contact: str = typer.Argument(..., help="Phone number, jid, or existing partial name."),
) -> None:
    """Map a nickname to a contact that has no name synced yet."""
    from whatsapp.tools import set_alias

    console = Console()
    config = load_config()
    with open_store(config) as store:
        res = set_alias(store, alias, contact)
        if res.ambiguous:
            console.print(f"[yellow]{contact!r} is ambiguous:[/yellow]")
            for jid, name in res.matches:
                console.print(f"  {name or '?'}  {jid}")
            raise typer.Exit(1)
        if res.jid is None:
            console.print(f"[red]No contact matching {contact!r}.[/red]")
            raise typer.Exit(1)
    console.print(f"[green]'{alias}' now means {res.name or res.jid}.[/green]")


@alias_app.command("list")
def wa_alias_list() -> None:
    """Show all remembered nicknames."""
    console = Console()
    config = load_config()
    with open_store(config) as store:
        rows = store.conn.execute(
            "SELECT alias, jid, created_at FROM wa_aliases ORDER BY alias"
        ).fetchall()
    if not rows:
        console.print("[dim]No nicknames set yet.[/dim]")
        return
    table = Table("nickname", "jid", "created")
    for r in rows:
        table.add_row(r["alias"], r["jid"], r["created_at"])
    console.print(table)


@auto_app.command("loop")
def wa_auto_loop(
    state: str = typer.Argument(..., help="on = text first when they go quiet; off = only reply."),
) -> None:
    """Loop mode: keep a conversation alive by texting first after a silence.

    Bounded: it gives up after `max_nudges` unanswered messages and never texts
    during quiet_hours. Requires auto mode. `wa auto off` stops everything.
    """
    import yaml

    from agent.config import DEFAULT_CONFIG_PATH

    console = Console()
    if state not in ("on", "off"):
        console.print("[red]Use 'on' or 'off'.[/red]")
        raise typer.Exit(1)
    config = load_config()
    ac = config["whatsapp"].setdefault("auto_reply", {})
    lc = ac.setdefault("loop", {})
    if state == "on":
        contacts = config["whatsapp"].get("auto_reply_contacts") or []
        console.print(f"[yellow]Loop mode texts {contacts} unprompted, as you.[/yellow] "
                      f"They should know they're part of this.")
        if not typer.confirm("Enable loop mode?"):
            return
    lc["enabled"] = state == "on"
    DEFAULT_CONFIG_PATH.write_text(yaml.dump(config, sort_keys=False))
    if state == "on":
        console.print(f"[magenta]Loop ON[/magenta] — nudges after "
                      f"{lc.get('quiet_after_s',900)//60} min quiet, max "
                      f"{lc.get('max_nudges',2)} unanswered, silent "
                      f"{lc.get('quiet_hours',[23,8])[0]}:00-{lc.get('quiet_hours',[23,8])[1]}:00. "
                      f"Stop: [bold]wa auto loop off[/bold]")
        if ac.get("mode") != "auto":
            console.print(f"[yellow]Note: mode is '{ac.get('mode')}' — loop only sends in 'auto'.[/yellow]")
    else:
        console.print("[green]Loop OFF — it will only reply, never text first.[/green]")


@auto_app.command("open")
def wa_auto_open(
    contact: str = typer.Argument(..., help="Allowlisted contact to text first (e.g. alice)."),
    text: str = typer.Option(None, "--text", help="Send this exact message instead of drafting one."),
    send: bool = typer.Option(False, "--send", help="Actually send. Without it, just show the draft."),
) -> None:
    """Start a conversation: send ONE opening message in your voice.

    Replies are then handled by the normal auto-reply cycle (`wa auto flush`), with
    all its guards. This sends one message and stops — it deliberately does not
    re-ping someone who hasn't answered.
    """
    from whatsapp.watch import _pick_ollama_host
    from whatsapp.autoreply import _allow_jids, _utcnow
    from whatsapp.responder import draft_reply
    from whatsapp.tools import resolve_contact

    console = Console()
    if send and not _send_allowed():
        console.print(SEND_OFF)
        send = False
    config = load_config()
    with open_store(config) as store:
        jid = resolve_contact(store, contact).jid
        if not jid:
            console.print(f"[red]No contact matching {contact!r}.[/red]")
            raise typer.Exit(1)
        # Same allowlist as auto-reply: this can only ever text an approved person.
        if jid not in _allow_jids(store, config):
            console.print(f"[red]{contact} is not in whatsapp.auto_reply_contacts.[/red]")
            raise typer.Exit(1)

        msg = text or draft_reply(
            store, OllamaClient(base_url=_pick_ollama_host()), config, jid,
            high_quality=True, opener=True,
        )
        if not msg:
            console.print("[red]Could not draft an opener (too little history?).[/red]")
            raise typer.Exit(1)
        console.print(f"[cyan]To {contact}:[/cyan] {msg}")
        if not send:
            console.print("[dim]Not sent. Re-run with --send.[/dim]")
            return
        if not typer.confirm(f"Send this to {contact} as you?"):
            return
        result = open_bridge_client(config).send(jid, msg) or {}
        if not result.get("ok"):
            console.print(f"[red]Send rejected: {result.get('error')}[/red]")
            raise typer.Exit(1)
        store.conn.execute(
            "INSERT INTO wa_autoreply (contact_jid, incoming_id, incoming_text, draft, "
            "status, reason, send_at, created_at, sent_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (jid, None, None, msg, "sent", "opener", int(time.time()), _utcnow(), _utcnow()),
        )
        store.conn.commit()
    console.print(f"[green]Sent.[/green] Replies are handled by the normal auto-reply cycle "
                  f"(mode: {config['whatsapp'].get('auto_reply', {}).get('mode')}). "
                  f"Stop any time: [bold]wa auto off[/bold]")
