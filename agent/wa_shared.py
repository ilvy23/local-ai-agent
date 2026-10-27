"""Shared plumbing for the WhatsApp CLI split (`agent/wa_public.py` + `agent/wa_private.py`).

This module knows nothing about public vs. private commands - it just does the
things both sides need: find/import the external plugin at
~/.config/agent/plugins/whatsapp (black box, code only, never this repo's data),
open the store/llm, and the handful of helpers duplicated across many commands
(resolving a contact, the bridge client, the send-allowlist file).

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent.config import PROJECT_ROOT
from agent.config import load_config as _base_load_config
from agent.llm import OllamaClient
from agent.memory.store import Store

# The WhatsApp code is a private plugin outside this repo; make it importable.
_PLUGIN_DIR = os.path.expanduser("~/.config/agent/plugins")
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

SEND_OFF = ("[yellow]Senden ist in dieser Installation abgeschaltet.[/yellow] Es gibt nur Entwürfe - "
            "es wird niemandem eine Nachricht geschickt.")


def _send_allowed() -> bool:
    """Sending needs AGENT_WA_ALLOW_SEND=1 in the environment (never set by default; the bridge and the
    client refuse as well)."""
    return os.environ.get("AGENT_WA_ALLOW_SEND") == "1"


try:
    from whatsapp.bridge_client import BridgeClient, BridgeError
    from whatsapp.ingest import sync as wa_sync
    from whatsapp.schema import ensure_schema as _ensure_schema
except ImportError:  # external plugin missing — commands will report it
    BridgeClient = BridgeError = wa_sync = None

    def _ensure_schema(conn):  # noqa: D401
        raise RuntimeError("WhatsApp plugin not installed at ~/.config/agent/plugins/whatsapp")


_WA_DEFAULTS = {
    "bridge_url": "http://127.0.0.1:8765",
    "token_file": str(Path(_PLUGIN_DIR) / "whatsapp" / "bridge" / "bridge-token.txt"),
    "summarize_batch": 200,
    "min_messages": 5,
    "profile_max_messages": 400,
}


def _ensure_wa(cfg: dict) -> dict:
    """Inject the whatsapp config the public core no longer carries."""
    wa = cfg.setdefault("whatsapp", {})
    for k, v in _WA_DEFAULTS.items():
        wa.setdefault(k, v)
    wa.setdefault("auto_reply", {})
    models = cfg.setdefault("models", {})
    # Safety net only - the real default now lives centrally in
    # agent/config.py's DEFAULT_CONFIG (models.unrestricted), which is
    # always present via load_config()'s deep-merge. This line just
    # protects an old config.yaml saved before that key existed.
    models.setdefault("unrestricted", models.get("chat", "qwen2.5:7b"))
    return cfg


def load_config() -> dict:
    return _ensure_wa(_base_load_config())


def open_store(config: dict) -> Store:
    from agent.main import _open_store as _f
    store = _f(config)
    _ensure_schema(store.conn)  # guarantee the wa_ tables exist
    return store


def open_llm() -> OllamaClient:
    from agent.main import _open_llm as _f
    return _f()


def open_bridge_client(config: dict) -> "BridgeClient":
    token_file = Path(config["whatsapp"]["token_file"])
    if not token_file.is_absolute():
        token_file = PROJECT_ROOT / token_file
    return BridgeClient(base_url=config["whatsapp"]["bridge_url"], token_file=token_file)


def wa_local_counts(store: Store) -> dict:
    def _count(table: str) -> int:
        return store.conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]

    last_sync_row = store.conn.execute(
        "SELECT v FROM wa_sync_state WHERE k = 'last_sync_at'"
    ).fetchone()
    return {
        "messages": _count("wa_messages"),
        "chats": _count("wa_chats"),
        "contacts": _count("wa_contacts"),
        "last_sync_at": last_sync_row["v"] if last_sync_row else None,
    }


def render_wa_status_table(console: Console, health: dict | None, counts: dict) -> None:
    table = Table("Field", "Value")
    if health is None:
        table.add_row("bridge", "not running")
    else:
        table.add_row("bridge", "ok" if health.get("status") == "ok" else "unknown")
        table.add_row("connected", str(health.get("connected", False)))
        table.add_row("paired", str(health.get("paired", False)))
    table.add_row("messages", str(counts["messages"]))
    table.add_row("chats", str(counts["chats"]))
    table.add_row("contacts", str(counts["contacts"]))
    table.add_row("last_sync_at", counts["last_sync_at"] or "never")
    console.print(table)


def resolve_or_exit(store, contact, console, kind="contact"):
    """Resolve a name/jid to a jid or print a message and return None."""
    from whatsapp.tools import resolve_contact

    res = resolve_contact(store, contact)
    if res.ambiguous:
        names = ", ".join(name or jid for jid, name in res.matches)
        console.print(f"[yellow]Several match '{contact}': {names}.[/yellow]")
        return None
    if res.jid is None:
        console.print(f"No {kind} matching '{contact}'.")
        return None
    return res


def agent_db_path(config: dict) -> str:
    db_path = Path(config["data"]["db_path"])
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return str(db_path)


def bridge_allowlist_path(config: dict) -> Path:
    """The bridge's own hard allowlist file. The bridge re-reads it on every send,
    so no restart is needed — but it must live on the machine running the bridge."""
    return Path(_PLUGIN_DIR) / "whatsapp" / "bridge" / "allowed-send.txt"


def as_jid(store, entry: str) -> str | None:
    """Allowlist entries may historically be names ('Alice'); the bridge only ever
    compares jids. Resolve to a jid or return None."""
    from whatsapp.tools import resolve_contact

    if "@" in entry:
        return entry
    return resolve_contact(store, entry).jid


def write_allowlist(store, config: dict, jids: list[str]) -> None:
    """Write both allowlists together. The config one decides who the AI drafts
    for; the bridge one is the hard stop that makes texting anyone else impossible.
    They must never drift apart — hence one function, never two callers."""
    import yaml

    from agent.config import DEFAULT_CONFIG_PATH

    resolved = [j for j in (as_jid(store, e) for e in jids) if j]
    config["whatsapp"]["auto_reply_contacts"] = resolved
    DEFAULT_CONFIG_PATH.write_text(yaml.dump(config, sort_keys=False))
    bridge_allowlist_path(config).write_text("\n".join(resolved) + ("\n" if resolved else ""))


def warn_if_bridge_elsewhere(console, config: dict) -> None:
    """The bridge enforces the allowlist from its OWN copy of the file. If the
    bridge runs on another machine (the laptop node), this change is only half
    applied — say so loudly rather than leaving a silent gap."""
    import httpx

    try:
        httpx.get(f"{config['whatsapp']['bridge_url']}/health", timeout=2)
    except httpx.HTTPError:
        node = config.get("node", {}).get("host") or "the node"
        console.print(f"\n[yellow]The bridge isn't running on this machine.[/yellow]\n"
                      f"  The bridge enforces its own allowlist file, so run the same\n"
                      f"  command on {node} (or sync wa-bridge/allowed-send.txt) or the\n"
                      f"  send will be blocked there.")
