"""Shared plumbing for the WhatsApp GUI routes (whatsapp*.py). Not a router itself.

The WhatsApp code is a private plugin at <plugin dir>/whatsapp (code only, never in this repo). It is loaded lazily and
only through `mod()`: a missing plugin gives a clean 503 ("not installed") instead of an import error, and tests replace
`mod()` with fakes. Nothing here can send a message: the bridge client is only ever used for health, backfill and reading,
and the auto-reply drivers (handle_inbound, flush_due, nudge_if_quiet, refresh_once) are never called by the GUI.

Long work runs as in-memory jobs (one at a time, in a thread). A job's log holds counters only, never message text.
"""

from __future__ import annotations

from agent import brand

import importlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from agent import plugins as _plugins
from agent.config import PROJECT_ROOT

CLI_HINT = f"{brand.CMD} wa --help"
JID_RE = re.compile(r"^[\w.+:@-]{1,120}$")
_DEFAULTS = {"summarize_batch": 200, "min_messages": 5, "profile_max_messages": 400}


class PluginMissing(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ plugin / config / db


def plugin_root() -> Path:
    return _plugins.plugin_dir() / "whatsapp"


def installed() -> bool:
    return (plugin_root() / "__init__.py").is_file()


def mod(name: str) -> Any:
    """A plugin module (`whatsapp.<name>`). The one seam tests replace."""
    if not installed():
        raise PluginMissing("WhatsApp plugin not installed")
    d = str(_plugins.plugin_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        return importlib.import_module(f"whatsapp.{name}")
    except ImportError as exc:
        raise PluginMissing(f"WhatsApp plugin module '{name}' could not be loaded: {exc}") from exc


def require() -> None:
    """Route guard: 503 with the CLI hint when the plugin is absent."""
    if not installed():
        raise HTTPException(503, f"WhatsApp plugin not installed (see: {CLI_HINT})")


def wa_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in the keys the plugin owns (the same defaults its register() and `agent wa` use). Mutates and returns config."""
    wa = config.setdefault("whatsapp", {})
    for k, v in _DEFAULTS.items():
        wa.setdefault(k, v)
    wa.setdefault("bridge_url", "http://127.0.0.1:8765")
    wa.setdefault("token_file", str(plugin_root() / "bridge" / "bridge-token.txt"))
    wa.setdefault("auto_reply", {})
    models = config.setdefault("models", {})
    models.setdefault("unrestricted", models.get("chat", "qwen2.5:7b"))
    return config


def db_path(config: dict[str, Any]) -> Path:
    p = Path(config["data"]["db_path"]).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def ro_rows(config: dict[str, Any], sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Read-only, parametrised query. A missing database or table gives []: nothing synced yet."""
    try:
        conn = sqlite3.connect(f"file:{db_path(config)}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def ro_one(config: dict[str, Any], sql: str, params: tuple = ()) -> dict[str, Any] | None:
    rows = ro_rows(config, sql, params)
    return rows[0] if rows else None


@contextmanager
def open_store(config: dict[str, Any]):
    """A fresh agent Store (sqlite connections are per thread) with the plugin's tables present."""
    from agent.memory.store import Store

    store = Store(db_path(config))
    try:
        mod("schema").ensure_schema(store.conn)
        yield store
    finally:
        store.close()


def bridge_client(config: dict[str, Any], timeout: float = 3.0) -> Any:
    wa = config["whatsapp"]
    tf = Path(wa["token_file"]).expanduser()
    return mod("bridge_client").BridgeClient(base_url=wa["bridge_url"], token_file=tf if tf.is_absolute() else PROJECT_ROOT / tf, timeout=timeout)


def safe_jid(jid: str) -> str:
    if not JID_RE.match(jid or ""):
        raise HTTPException(400, "Not a valid chat id.")
    return jid


def chat_name(config: dict[str, Any], jid: str) -> str:
    r = ro_one(config, "SELECT name FROM wa_chats WHERE jid=? UNION SELECT name FROM wa_contacts WHERE jid=?", (jid, jid))
    return (r or {}).get("name") or jid


def send_state() -> dict[str, Any]:
    """Sending is blocked unless the process environment says AGENT_WA_ALLOW_SEND=1. The GUI never offers a send control either way."""
    on = os.environ.get("AGENT_WA_ALLOW_SEND") == "1"
    return {"blocked": not on, "env_enabled": on}


def update_config_yaml(config: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> None:
    """Edit config.yaml the way `agent settings` does (load the file as it is, change one key, write it back) and mirror the
    change into the running config so the screen shows it at once."""
    import yaml

    from agent import settings

    path = Path(settings.DEFAULT_CONFIG_PATH)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    mutate(config)


# ------------------------------------------------------------------------------------------------ jobs


class Stopped(BaseException):
    """Raised inside a job when the user pressed Stop (at the next checkpoint). A BaseException on purpose: the plugin's own
    `except Exception` guards (per-chat, per-batch) must not swallow it and carry on."""


class Job:
    def __init__(self, kind: str, label: str, target: str = "") -> None:
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.label, self.target = kind, label, target
        self.state = "running"                     # running | done | failed | stopped
        self.pct: float | None = None
        self.step = ""
        self.log: list[str] = []
        self.started = time.time()
        self.finished: float | None = None
        self.result: dict[str, Any] | None = None
        self.error = ""
        self.stop = threading.Event()

    def check(self) -> None:
        if self.stop.is_set():
            raise Stopped()

    def say(self, msg: str, pct: float | None = None) -> None:
        """Progress line (counters only, never message text). Also the checkpoint where Stop takes effect."""
        self.check()
        self.step = msg[:160]
        if pct is not None:
            self.pct = max(0.0, min(100.0, float(pct)))
        self.log = (self.log + [self.step])[-40:]

    def view(self, full: bool = False) -> dict[str, Any]:
        end = self.finished or time.time()
        el = end - self.started
        eta = None
        if self.state == "running" and self.pct and 2 <= self.pct < 100:
            eta = round(el / self.pct * (100 - self.pct))
        d = {"id": self.id, "kind": self.kind, "label": self.label, "target": self.target, "state": self.state,
             "stopping": self.state == "running" and self.stop.is_set(), "pct": self.pct, "step": self.step, "log": self.log[-12:],
             "started": self.started, "elapsed": round(el), "eta": eta, "error": self.error, "has_result": bool(self.result)}
        if full:
            d["result"] = self.result
        return d


_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_stoppers: dict[str, Callable[[], None]] = {}      # job id -> extra stop action (e.g. end a training subprocess)


def start_job(kind: str, label: str, work: Callable[[Job], dict[str, Any] | None], target: str = "",
              on_stop: Callable[[], None] | None = None) -> Job:
    """Run `work(job)` in a thread. Only one job runs at a time (the models and the GPU are shared): a second start is a 409."""
    with _lock:
        running = [j for j in _jobs.values() if j.state == "running"]
        if running:
            raise HTTPException(409, f"Another WhatsApp job is running: {running[0].label}. Wait for it or stop it first.")
        job = Job(kind, label, target)
        _jobs[job.id] = job
        for old in sorted(_jobs.values(), key=lambda j: j.started)[:-30]:
            if old.state != "running":
                _jobs.pop(old.id, None)
        if on_stop:
            _stoppers[job.id] = on_stop

    def run() -> None:
        try:
            job.result = work(job) or {}
            job.state = "done"
            job.pct = 100.0 if job.pct is not None else job.pct
        except Stopped:
            job.state = "stopped"
        except PluginMissing as exc:
            job.state, job.error = "failed", str(exc)[:300]
        except Exception as exc:  # noqa: BLE001 - reported through the job, never raised into the server
            job.state, job.error = "failed", f"{type(exc).__name__}: {str(exc)[:240]}"
        finally:
            job.finished = time.time()
            _stoppers.pop(job.id, None)

    threading.Thread(target=run, daemon=True, name=f"wa-job-{job.id}").start()
    return job


def get_job(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "No such job (the list is kept in memory and resets when the server restarts).")
    return job


def stop_job(job: Job) -> None:
    job.stop.set()
    fn = _stoppers.get(job.id)
    if fn:
        try:
            fn()
        except Exception:  # noqa: BLE001 - best effort
            pass


def list_jobs() -> list[dict[str, Any]]:
    return [j.view() for j in sorted(_jobs.values(), key=lambda j: j.started, reverse=True)]


def reset_jobs() -> None:       # tests
    _jobs.clear()
    _stoppers.clear()


class CountingLLM:
    """Wraps the model client to report embedding progress to a job and to honour Stop between batches."""

    def __init__(self, inner: Any, job: Job, total: int | None = None) -> None:
        self._inner, self._job, self._total, self.n = inner, job, total, 0

    def embed(self, texts, *a, **kw):
        self._job.check()
        out = self._inner.embed(texts, *a, **kw)
        self.n += len(texts)
        self._job.say(f"embedded {self.n}" + (f" of {self._total}" if self._total else ""), 100 * self.n / self._total if self._total else None)
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def jsonable_card(row: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("fingerprint", "eval", "tuned"):
        try:
            out[k] = json.loads((row or {}).get(k) or "{}")
        except ValueError:
            out[k] = {}
    return out
