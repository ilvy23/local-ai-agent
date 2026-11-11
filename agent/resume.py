"""One view of everything resumable on this machine: saved coding sessions (every folder) and overnight runs (every folder with a .night/state.json).

Night runs are indexed in <state home>/agent/night-runs.jsonl (one line per folder, written when `nightage night run` starts: folder, prompt sources, options,
times). Runs that started before the index existed are found by scanning the known folders (index + folders of coding sessions) for .night/state.json; their prompts
are then rebuilt from the state file. `nightage continue [N|ID]` lists / continues; `nightage night runs` lists the night runs only."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agent import brand, osport
from agent.codeagent import sessionstore

OPTION_KEYS = ("plan_first", "review", "map_mode", "tournament", "pause_on_game", "stage_cap", "ctx", "default_limits", "no_model")


def index_path() -> Path:
    return osport.state_home() / "agent" / "night-runs.jsonl"


def night_id(folder: Path | str) -> str:
    return "n" + hashlib.sha1(str(folder).encode()).hexdigest()[:7]


# ── the night index ─────────────────────────────────────────────────────────

def _index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        lines = index_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for ln in lines:
        try:
            d = json.loads(ln)
        except ValueError:
            continue                                        # a damaged line is skipped, never fatal
        if isinstance(d, dict) and isinstance(d.get("folder"), str):
            out[d["folder"]] = d
    return out


def _write_index(entries: dict[str, dict[str, Any]]) -> None:
    f = index_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries.values()), encoding="utf-8")
    os.replace(tmp, f)                                      # atomic: a kill never leaves half an index


def record_night(root: Path, prompts: list[str], options: dict[str, Any], resumed: bool = False) -> None:
    """Index (or refresh) the night of `root`: one line per folder, so calling it again for the same run changes nothing but the time.
    A resumed run keeps the prompt sources it was started with."""
    try:
        idx = _index()
        old = idx.get(str(root)) or {}
        now = time.time()
        sources = [{"file": str(Path(p).expanduser().resolve())} if Path(p).expanduser().exists() else {"text": p} for p in prompts]
        idx[str(root)] = {"folder": str(root), "sources": old.get("sources") if resumed and old.get("sources") else sources,
                          "options": {k: options[k] for k in OPTION_KEYS if k in options}, "started": old.get("started") or now if resumed else now, "updated": now}
        _write_index(idx)
    except OSError:
        pass                                                # the index is a convenience: a night must never fail because of it


# ── the registry ────────────────────────────────────────────────────────────

@dataclass
class Entry:
    kind: str                       # code | night
    id: str
    folder: str
    when: float                     # last activity
    status: str                     # running | interrupted | finished | partial | folder gone | ...
    desc: str
    how: str                        # the command that continues it
    running: bool = False
    gone: bool = False              # folder (or, for a night, its state) is missing
    resumable: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _night_state(folder: Path) -> dict[str, Any] | None:
    try:
        s = json.loads((folder / ".night" / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return s if isinstance(s, dict) and isinstance(s.get("items"), list) else None


def _night_entry(folder: str, rec: dict[str, Any] | None) -> Entry:
    from agent.night import lock as lock_mod

    f = Path(folder)
    st = _night_state(f) if f.is_dir() else None
    when = (rec or {}).get("updated") or (rec or {}).get("started") or 0
    if st is not None:
        try:
            when = max(when, (f / ".night" / "state.json").stat().st_mtime)
        except OSError:
            pass
    items = [i for i in (st or {}).get("items", []) if isinstance(i, dict)]
    first = next((str(i.get("text", "")) for i in items), "") or next((s.get("text") or Path(s.get("file", "")).name for s in (rec or {}).get("sources", []) if isinstance(s, dict)), "")
    desc = sessionstore.title_of([first]) if first else "(night run)"
    running = f.is_dir() and lock_mod.owner(f) is not None
    left = float((st or {}).get("left_s") or 0)
    done = sum(1 for i in items if i.get("status") == "done")
    over = (f / ".night" / "DONE").exists()
    if not f.is_dir():
        status, resumable = "folder gone", False
    elif st is None:
        status, resumable = "state gone", False
    elif running:
        status, resumable = "running", False
    elif over:
        status, resumable = ("finished" if done == len(items) else "partial (night over)"), False
    else:
        status, resumable = "interrupted", True
    if st is not None:
        desc += f"  [{done}/{len(items)} done, {left / 3600:.1f} h left]"
    return Entry("night", night_id(f), folder, when, status, desc, f"{brand.CMD} continue {night_id(f)}", running=running, gone=not f.is_dir() or st is None,
                 resumable=resumable, extra={"rec": rec, "state": st})


def _code_entry(m: sessionstore.Meta) -> Entry:
    gone = not Path(m.root).is_dir()
    state = "folder gone" if gone else m.shown_state()
    return Entry("code", m.id, m.root, m.updated, state, m.title, f"{brand.CMD} code --resume {m.id}", running=state == "in progress", gone=gone,
                 resumable=not gone and state != "in progress", extra={"meta": m})


def collect(warn: list[str] | None = None) -> list[Entry]:
    """Every coding session and every night run, newest first."""
    metas = sessionstore.list_sessions(None, warn)
    idx = _index()
    folders = list(idx) + [m.root for m in metas if m.root]
    nights: dict[str, Entry] = {}
    for fo in dict.fromkeys(folders):
        if fo in idx or _night_state(Path(fo)) is not None:        # discovered: a state.json in a folder we know
            nights[fo] = _night_entry(fo, idx.get(fo))
    code = [_code_entry(m) for m in metas]
    for e in code:                                                   # a coding session in a folder that has a night is one attempt of that night
        e.extra["night_attempt"] = e.folder in nights and not nights[e.folder].gone
    return sorted(code + list(nights.values()), key=lambda e: e.when, reverse=True)


def find(token: str, entries: list[Entry]) -> tuple[Entry | None, str]:
    token = token.strip().lower()
    if token.isdigit() and 1 <= int(token) <= len(entries):
        return entries[int(token) - 1], ""
    hits = [e for e in entries if e.id.startswith(token)] if token else []
    if len(hits) == 1:
        return hits[0], ""
    return None, (f"'{token}' matches {len(hits)} runs - give more characters" if hits else f"no resumable run matches '{token}' (see `{brand.CMD} continue --list`)")


def table(entries: list[Entry]) -> Table:
    home, now = str(Path.home()), time.time()
    t = Table("#", "kind", "id", "folder", "status", "age", "what", box=None)
    for n, e in enumerate(entries, 1):
        fo = e.folder.replace(home, "~", 1) if e.folder.startswith(home) else e.folder
        fo = fo if len(fo) <= 34 else "…" + fo[-33:]
        t.add_row(str(n), e.kind, e.id, escape(fo), e.status, sessionstore.age_text(now - e.when), escape(e.desc[:70]))
    return t


# ── prune ───────────────────────────────────────────────────────────────────

def prune(entries: list[Entry]) -> list[Entry]:
    """Drop what can never be resumed: night entries whose folder or state is gone (index line removed), coding sessions whose folder is gone (file deleted)."""
    gone = [e for e in entries if e.gone and not e.running]
    idx = _index()
    dropped: list[Entry] = []
    for e in gone:
        if e.kind == "night":
            if idx.pop(e.folder, None) is not None or not Path(e.folder).is_dir():
                dropped.append(e)
        elif sessionstore.delete(e.extra["meta"]):
            dropped.append(e)
    if any(e.kind == "night" for e in dropped):
        try:
            _write_index(idx)
        except OSError:
            pass
    return dropped


# ── continue ────────────────────────────────────────────────────────────────

def _night_prompts(e: Entry) -> list[str] | None:
    """The prompt sources of a night: the saved ones while they still exist, else files rebuilt from the state file (name, text, weight, ...)."""
    srcs = [s for s in ((e.extra.get("rec") or {}).get("sources") or []) if isinstance(s, dict)]
    if srcs and all(Path(s["file"]).exists() for s in srcs if "file" in s):
        return [s.get("file") or s.get("text", "") for s in srcs]
    items = [i for i in (e.extra.get("state") or {}).get("items", []) if isinstance(i, dict) and i.get("text")]
    if not items:
        return None
    d = Path(e.folder) / ".night" / "resume-prompts"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for n, i in enumerate(items, 1):
        head = {"name": i.get("name"), "weight": i.get("weight"), "verify": i.get("verify"), "depends": ", ".join(i.get("depends") or []), "kind": i.get("kind"),
                "plan": "true" if i.get("plan") else ""}
        block = "".join(f"{k}: {' '.join(str(v).split())}\n" for k, v in head.items() if v and v != 1.0)
        f = d / f"{n:02d}.md"
        f.write_text(f"---\n{block}---\n{i['text']}\n", encoding="utf-8")
        out.append(str(f))
    return out


def night_argv(e: Entry, prompts: list[str]) -> list[str]:
    o = (e.extra.get("rec") or {}).get("options") or {}
    argv = ["run", *prompts, "--folder", e.folder, "--resume", "--yes"]
    argv += ["--plan"] * bool(o.get("plan_first")) + ["--no-model"] * bool(o.get("no_model")) + ["--default-limits"] * bool(o.get("default_limits"))
    if o.get("review") is not None:
        argv.append("--review" if o["review"] else "--no-review")
    if o.get("pause_on_game") is not None:
        argv.append("--pause-on-game" if o["pause_on_game"] else "--ignore-games")
    for key, flag in (("map_mode", "--map"), ("tournament", "--tournament"), ("stage_cap", "--stage-cap"), ("ctx", "--ctx")):
        if o.get(key) is not None:
            argv += [flag, str(o[key])]
    return argv


def _run_night(argv: list[str]) -> int:
    import typer.main

    from agent.night.cli import night_app

    rc = typer.main.get_command(night_app).main(args=argv, prog_name=f"{brand.CMD} night", standalone_mode=False)
    return rc if isinstance(rc, int) else 0


def _run_code(e: Entry) -> int:
    from agent.config import load_config
    from agent.ui.code_app import run_code_ui

    return run_code_ui(e.folder, None, resume=e.id, config=load_config())


def resume_entry(e: Entry, console: Console) -> int:
    """Continue `e` in its own folder. 0/exit code of the run, 1 when it cannot be continued (the reason is printed)."""
    if e.running:
        what = f"night (pid {_pid(e)})" if e.kind == "night" else "coding session"
        stop = f"`{brand.CMD} night stop --folder {e.folder}`" if e.kind == "night" else "ending it there (/exit)"
        console.print(f"[yellow]{e.id} is running right now ({what}) in {escape(e.folder)}.[/yellow] Stop it first by {escape(stop)}, then continue.")
        return 1
    if e.gone:
        console.print(f"[red]{e.id}: {escape(e.folder)} " + ("no longer exists" if not Path(e.folder).is_dir() else "has no night state any more") + ".[/red] "
                      f"`{brand.CMD} continue --prune` removes such entries.")
        return 1
    if e.kind == "code":
        from agent.night import lock as lock_mod

        if lock_mod.owner(Path(e.folder)) is not None:
            console.print(f"[yellow]a night (pid {_pid(e)}) is running in {escape(e.folder)}: two agents in one folder overwrite each other.[/yellow] "
                          f"`{brand.CMD} night stop --folder {escape(e.folder)}` first, or continue the night itself.")
            return 1
        return _run_code(e)
    if not e.resumable:
        console.print(f"[yellow]the night in {escape(e.folder)} is over ({e.status}).[/yellow] `{brand.CMD} night report --folder {escape(e.folder)}` shows what it did; "
                      f"start a new one with `{brand.CMD} night run ... --folder {escape(e.folder)}`.")
        return 1
    prompts = _night_prompts(e)
    if not prompts:
        console.print(f"[red]the prompts of the night in {escape(e.folder)} are gone and the state holds none.[/red]")
        return 1
    return _run_night(night_argv(e, prompts))


def _pid(e: Entry) -> str:
    from agent.night import lock as lock_mod

    return str(lock_mod.owner(Path(e.folder)) or "?")


def continue_cli(which: str | None = None, list_only: bool = False, prune_it: bool = False, kind: str = "", ask: Callable[[str], str] | None = None, show_all: bool = False) -> int:
    console = Console()
    warn: list[str] = []
    entries = collect(warn)
    for w in warn:
        console.print(f"[yellow]{escape(w)}[/yellow]")
    if kind:
        entries = [e for e in entries if e.kind == kind]
    if prune_it:
        gone = prune(entries)
        console.print(f"removed {len(gone)} entr{'y' if len(gone) == 1 else 'ies'} whose folder or state is gone")
        return 0
    hidden = 0
    if which is None and not show_all:
        attempts = [e for e in entries if e.extra.get("night_attempt")]
        entries = [e for e in entries if not e.extra.get("night_attempt")]          # the night is the unit; its inner sessions only clutter the list                                       # the list shows what can still be continued; dead entries (a deleted temp folder) stay out of the way
        alive = [e for e in entries if e.status != "folder gone"]
        hidden, entries = len(entries) - len(alive) + len(attempts), alive
    if not entries:
        console.print("[dim]nothing to resume yet[/dim]" + (f" [dim]({hidden} dead entries hidden: `{brand.CMD} continue --all` shows them, `--prune` removes them)[/dim]" if hidden else ""))
        return 0
    if which is None:
        console.print(table(entries))
        if hidden:
            console.print(f"[dim]{hidden} entries whose folder is gone (dead ones and the inner sessions of nights) are hidden: `{brand.CMD} continue --all` shows them, `--prune` removes them[/dim]")
        console.print(f"[dim]continue one in its own folder: {brand.CMD} continue N|ID   ·   {brand.CMD} continue --prune removes dead entries[/dim]")
        if list_only or (ask is None and not sys.stdin.isatty()):
            return 0
        try:
            which = (ask or console.input)("continue which one? [number or id, Enter = none] ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not which:
            return 0
    e, why = find(which, entries)
    if e is None:
        console.print(f"[red]{escape(why)}[/red]")
        return 1
    return resume_entry(e, console)
