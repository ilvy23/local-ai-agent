"""Saved coding sessions: many per project folder, resumable after a crash, a kill or a reboot.

Layout (persistent, under ~/.local/state/agent/code-sessions/ - not /tmp):
    <folder key>/<id>.json         the whole session: the message list verbatim (stable prefix included, so nothing is
                                   re-summarised and the server's prompt cache can be reused), ledger, acceptance plan,
                                   per-session settings and what was in progress
    <folder key>/<id>.meta.json    a few hundred bytes for `agent code sessions` (listing never parses the big files)
    <folder key>.json              the old one-file-per-folder format: migrated on first sight into the layout above and
                                   kept as <folder key>.json.migrated (never deleted)
<folder key> = sha1(folder)[:16], <id> = 8 hex chars (any unique prefix works on the command line). Every write is
atomic (temp file + rename), so a kill in the middle of a save leaves the previous save. No expiry by age: a session
stays until it is deleted (`/delete`, `agent code sessions --delete`, `--prune DAYS`).

A resumed session restores the messages exactly, repairs what a kill can leave half-done (a tool call without its
result), checks the ledger against the disk and tells the model in one short factual note what changed meanwhile.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from agent import osport
from typing import Any

logger = logging.getLogger(__name__)

VERSION = 2
DEFAULT_RESUME_H = 12.0            # a plain `agent code` continues the folder's latest session if it is this recent
# harness marker (context.MARKERS "no_result"): the model must never write it into a file
NO_RESULT = ("[no result recorded: the session was interrupted while this call ran. Check the current state "
             "(read the file / run it) before relying on it.]")


def state_root() -> Path:
    base = str(osport.state_home())
    return Path(base) / "agent" / "code-sessions"


def folder_key(root: Path | str) -> str:
    return hashlib.sha1(str(root).encode()).hexdigest()[:16]


def folder_dir(root: Path | str) -> Path:
    return state_root() / folder_key(root)


def new_id() -> str:
    return secrets.token_hex(4)


def title_of(requests: list[str]) -> str:
    first = next((r for r in requests if str(r).strip() and not str(r).startswith("/")), "")
    line = " ".join(next((ln for ln in str(first).splitlines() if ln.strip()), "").split())
    return (line if len(line) <= 60 else line[:59] + "…") or "(no request yet)"


def age_text(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    if s < 2 * 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


@dataclass
class Meta:
    id: str
    root: str
    title: str
    created: float
    updated: float
    steps: int
    model: str
    state: str                     # "finished" | "interrupted" | "in progress"
    stopped: str = ""              # why the last prompt ended early: budget, loop, error, interrupted, killed
    files: int = 0
    path: Path | None = None

    def shown_state(self) -> str:
        """A session saved as "in progress" whose folder nobody holds the lock of was killed mid-prompt."""
        if self.state == "in progress" and not folder_busy(Path(self.root)):
            return "interrupted (killed)"
        return f"{self.state} ({self.stopped})" if self.state == "interrupted" and self.stopped else self.state


def _atomic(path: Path, text: str) -> None:
    from agent.codeagent.tools import atomic_write

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, text)


def meta_of(data: dict[str, Any], path: Path | None = None) -> Meta:
    return Meta(id=str(data["id"]), root=str(data.get("root", "")), title=str(data.get("title") or title_of(data.get("requests") or [])),
                created=float(data.get("created") or data.get("updated") or 0), updated=float(data.get("updated") or 0),
                steps=int(data.get("step") or 0), model=str(data.get("model") or ""), state=str(data.get("state") or "finished"),
                stopped=str((data.get("pending") or {}).get("stopped") or ""), files=len((data.get("ledger") or {}).get("files") or {}),
                path=path)


def save(data: dict[str, Any]) -> Path:
    """Write a session (and its small meta file). Raises OSError/TypeError/ValueError - the caller decides."""
    d = folder_dir(data["root"])
    f = d / f"{data['id']}.json"
    _atomic(f, json.dumps(data, ensure_ascii=False, default=str))
    m = meta_of(data)
    _atomic(d / f"{data['id']}.meta.json", json.dumps({k: v for k, v in m.__dict__.items() if k != "path"}, ensure_ascii=False))
    return f


def load(path: Path) -> dict[str, Any]:
    """The session at `path`. ValueError when it is truncated, corrupt or not a session."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{path.name} is damaged (truncated or not JSON)") from exc
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list) or "id" not in data:
        raise ValueError(f"{path.name} is not a saved session")
    return data


# ── legacy format ───────────────────────────────────────────────────────────

def migrate_legacy(warn: list[str] | None = None) -> int:
    """Old layout: one <folder key>.json per folder, overwritten by every new session. Each becomes a session of its
    own in the new layout; the old file is renamed to .json.migrated (kept, never deleted). Returns how many moved."""
    base = state_root()
    if not base.is_dir():
        return 0
    moved = 0
    for old in sorted(base.glob("*.json")):
        try:
            raw = json.loads(old.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list) or not raw.get("root"):
                raise ValueError("not a session")
        except (OSError, ValueError):
            if warn is not None:
                warn.append(f"skipped {old.name}: damaged old session file (left as it is)")
            continue
        msgs = raw["messages"]
        last = msgs[-1] if msgs else {}
        finished = last.get("role") == "assistant" and not str(last.get("content", "")).lstrip().startswith("{")
        saved = float(raw.get("saved") or old.stat().st_mtime)
        data = {"version": VERSION, "id": new_id(), "root": raw["root"], "title": title_of(raw.get("requests") or []),
                "created": saved, "updated": saved, "state": "finished" if finished else "interrupted", "model": "", "num_ctx": 0,
                "step": 0, "messages": msgs, "requests": list(raw.get("requests") or []),
                "ledger": {"files": raw.get("files") or {}, "runs": raw.get("runs") or [], "stamps": {}}, "plan": None,
                "pending": {"stopped": "" if finished else "unknown"}, "migrated_from": old.name}
        try:
            save(data)
            old.rename(old.with_name(old.name + ".migrated"))
            moved += 1
        except OSError as exc:
            if warn is not None:
                warn.append(f"could not migrate {old.name}: {exc}")
    return moved


# ── listing / lookup ────────────────────────────────────────────────────────

def list_sessions(root: Path | str | None = None, warn: list[str] | None = None) -> list[Meta]:
    """All saved sessions (or those of one folder), newest first. A damaged file is skipped with a warning."""
    migrate_legacy(warn)
    base = state_root()
    dirs = [folder_dir(root)] if root is not None else ([p for p in base.iterdir() if p.is_dir()] if base.is_dir() else [])
    out: list[Meta] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            if f.name.endswith(".meta.json"):
                continue
            mf = f.with_name(f.stem + ".meta.json")
            try:
                m = Meta(**{**json.loads(mf.read_text(encoding="utf-8")), "path": f})
                if m.id != f.stem:
                    raise ValueError("meta mismatch")
            except (OSError, ValueError, TypeError):
                try:                                            # no / stale meta: read the session itself (and fix the meta)
                    data = load(f)
                    m = meta_of(data, f)
                    try:
                        save(data)
                    except (OSError, TypeError, ValueError):
                        pass
                except ValueError as exc:
                    if warn is not None:
                        warn.append(f"skipped {f.parent.name}/{f.name}: {exc}")
                    continue
            out.append(m)
    return sorted(out, key=lambda m: m.updated, reverse=True)


def find(prefix: str, root: Path | str | None = None) -> tuple[Meta | None, str]:
    """The session whose id starts with `prefix` (this folder first, then all folders). (meta, "") or (None, why)."""
    prefix = prefix.strip().lower()
    if not prefix:
        return None, "give a session id (see `agent code sessions`)"
    for scope in ([root, None] if root is not None else [None]):
        hits = [m for m in list_sessions(scope) if m.id.startswith(prefix)]
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            return None, f"'{prefix}' matches {len(hits)} sessions ({', '.join(m.id for m in hits[:5])}) - give more characters"
    return None, f"no saved session matches '{prefix}' (see `agent code sessions`)"


def id_exists(prefix: str) -> bool:
    """True when some saved session (any folder) has an id starting with `prefix` - what makes a hex-looking word an id and not task text."""
    prefix = prefix.strip().lower()
    try:
        return bool(prefix) and any(m.id.startswith(prefix) for m in list_sessions(None))
    except OSError:
        return False


def latest(root: Path | str, max_age_h: float | None = None) -> Meta | None:
    for m in list_sessions(root):
        if max_age_h is None or time.time() - m.updated <= max_age_h * 3600:
            return m
        return None
    return None


def delete(meta: Meta) -> bool:
    ok = False
    for f in (meta.path, meta.path.with_name(meta.id + ".meta.json") if meta.path else None):
        if f is not None:
            try:
                f.unlink()
                ok = True
            except FileNotFoundError:
                pass
            except OSError:
                return False
    return ok


def prune(days: float, keep: str = "") -> list[Meta]:
    """Delete sessions not touched for `days` days (never the one with id `keep`). Returns the deleted ones."""
    cut = time.time() - days * 86400
    gone = [m for m in list_sessions() if m.updated < cut and m.id != keep and m.state != "in progress"]
    return [m for m in gone if delete(m)]


def folder_busy(root: Path) -> bool:
    """Is a coding session running in this folder right now? The folder lock is a kernel flock: it vanishes with its
    process (kill -9, crash, reboot), so this can never report a dead session as running."""
    from agent import osport
    from agent.codeagent import session as session_mod

    f = session_mod.lock_file(root)
    if not f.exists():
        return False
    if session_mod.holds_lock(root):
        return True
    try:
        with open(f, "a+") as fh:
            if not osport.try_lock(fh):
                return True
            osport.unlock(fh)
    except OSError:
        return False
    return False


# ── repair ──────────────────────────────────────────────────────────────────

def repair(messages: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Make a message list from a killed run consistent: drop what is not a message, and give a tool call that has no
    result (killed while it ran) a result that says so - a call without its result confuses the model and some chat
    templates. A turn that simply ended early (last message a tool result or the user's request) is left as it is:
    the next prompt continues it."""
    from agent.codeagent.context import _parse_call

    out: list[dict[str, Any]] = []
    notes: list[str] = []
    dropped = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
            dropped += 1
            continue
        if m.get("content") is None:
            m = {**m, "content": ""}
        out.append(m)
    if dropped:
        notes.append(f"dropped {dropped} damaged message(s)")
    fixed: list[dict[str, Any]] = []
    added = 0
    for i, m in enumerate(out):
        fixed.append(m)
        if m["role"] != "assistant":
            continue
        nxt = out[i + 1]["role"] if i + 1 < len(out) else None
        if nxt == "tool":
            continue
        native = [tc for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)]
        if native or _parse_call(str(m.get("content", ""))) is not None:
            for _ in (native or [None]):
                fixed.append({"role": "tool", "content": NO_RESULT})
                added += 1
    if added:
        notes.append(f"{added} tool call(s) had no result (killed while running) - marked as interrupted")
    return fixed, notes


# ── engine snapshot / restore ───────────────────────────────────────────────

def _stamps(root: Path, files: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for p in files:
        f = Path(p) if Path(p).is_absolute() else root / p
        try:
            st = f.stat()
            out[p] = [st.st_size, st.st_mtime_ns]
        except OSError:
            out[p] = [-1, 0]
    return out


def snapshot(engine: Any) -> dict[str, Any]:
    """Everything a new process needs to continue exactly here."""
    led, ctx = engine.ctx.ledger, engine.ctx
    plan = engine.plan.plan
    files = {p: {"action": e.action, "step": e.step, "lines": e.lines, "size": e.size, "names": e.names} for p, e in led.files.items()}
    state = "in progress" if engine.in_turn else ("finished" if not engine.last_stopped else "interrupted")
    return {
        "version": VERSION, "id": engine.session_id, "root": str(engine.root), "title": title_of(ctx.user_requests),
        "created": engine.session_created, "updated": time.time(), "state": state, "pid": os.getpid(),
        "model": engine.model, "num_ctx": engine.num_ctx, "step": ctx.step, "note_key": engine.note_key,
        "messages": ctx.messages, "requests": ctx.user_requests,
        "ledger": {"files": files, "runs": [list(r) for r in led.runs], "stamps": _stamps(engine.root, led.files)},
        "plan": None if plan is None else {"task": plan.task, "nudges": plan.nudges,
                                           "items": [{"id": i.id, "text": i.text, "done": i.done, "evidence": i.evidence} for i in plan.items]},
        "ctx_state": {"last_prompt": ctx.last_prompt, "last_eval": ctx.last_eval, "since": ctx._since, "mask_floor": ctx.mask_floor, "stage_mask_floor": ctx.stage_mask_floor,
                      "last_tps": ctx.last_tps},
        "engine_state": {"think_steps": engine.think_steps, "look_streak": engine._look_streak, "noted": [list(x) for x in sorted(engine.__dict__.get("_noted", set()))],
                         "declined": [list(x) for x in sorted(engine._declined)], "session_notes": list(engine.session_notes),
                         "briefs": list(getattr(engine, "briefs", [])),
                         "intake_seen": sorted(getattr(engine, "intake_seen", ())), "split_cache": dict(getattr(engine, "split_cache", {})),
                         "progress": engine._progress, "loop_note_at": engine._loop_note_at, "in_place_failures": engine._in_place_failures,
                         "compact_retry_at": ctx._compact_retry_at, "lookact": engine.lookact.state(), "watcher": engine.watcher.state(),
                         "sigs": [list(h[:5]) for h in engine._sigs], "dead_ends": engine.dead_ends.to_json(),
                         "interactive_hinted": engine._interactive_hinted, "dirty": sorted(engine.tools.dirty), "owned": dict(engine.tools.owned),
                         "checks": engine.checks.to_json() if hasattr(engine, "checks") else []},
        "stages": engine.stage_plan.to_json() if getattr(engine, "stage_plan", None) is not None else None,
        "pending": {"stopped": "killed" if engine.in_turn else engine.last_stopped,
                    "task": ctx.user_requests[-1] if ctx.user_requests else ""},
    }


def restore(engine: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Put a saved session into a freshly built engine. Returns facts for the banner; leaves `engine.resume_note` for
    the model (sent once, right before the next request - a system message after an assistant answer would be lost)."""
    from agent import plan as plan_mod
    from agent.codeagent.context import FileEntry

    ctx = engine.ctx
    msgs, repaired = repair(list(data.get("messages") or []))
    ctx.messages = msgs
    engine.tools.read_cache.clear()                  # the restored history holds none of the reads that armed it
    ctx.user_requests = [str(r) for r in data.get("requests") or []]
    led = data.get("ledger") or {"files": data.get("files") or {}, "runs": data.get("runs") or []}
    ctx.ledger.files.clear()
    for p, e in (led.get("files") or {}).items():
        try:
            ctx.ledger.files[p] = FileEntry(e["action"], int(e["step"]), int(e["lines"]), int(e["size"]), list(e.get("names") or []))
        except (KeyError, TypeError, ValueError):
            continue
    ctx.ledger.runs = [tuple(r) for r in led.get("runs") or [] if isinstance(r, (list, tuple)) and len(r) == 4]
    p = data.get("plan")
    engine.plan.plan = None
    if isinstance(p, dict) and p.get("items"):
        engine.plan.plan = plan_mod.Plan(task=str(p.get("task", "")), nudges=int(p.get("nudges") or 0),
                                         items=[plan_mod.Item(int(i["id"]), str(i["text"]), bool(i.get("done")), str(i.get("evidence") or ""))
                                                for i in p["items"]])
    if data.get("model"):
        engine.model = str(data["model"])
    if data.get("num_ctx"):
        engine.num_ctx = ctx.num_ctx = int(data["num_ctx"])
    ctx.step = int(data.get("step") or 0)
    cs = data.get("ctx_state") or {}
    ctx.last_prompt, ctx.last_eval = int(cs.get("last_prompt") or 0), int(cs.get("last_eval") or 0)
    ctx._since = int(cs["since"]) if "since" in cs else sum(len(str(m.get("content", ""))) for m in msgs)
    ctx.mask_floor, ctx.last_tps = float(cs.get("mask_floor") or 0.0), float(cs.get("last_tps") or 0.0)
    ctx.stage_mask_floor = float(cs.get("stage_mask_floor") or 0.0)         # missing = 0 = "not masked since": at worst one more batch
    es = data.get("engine_state") or {}
    engine.think_steps = int(es.get("think_steps") or 0)
    # A restart must not hand out a fresh allowance of reading (seen live: every resume = 25 more reads before the first edit); a few reads stay allowed
    from agent.codeagent.engine import LOOP_WINDOW
    engine.lookact.load(es.get("lookact"))
    engine._look_streak = min(int(es.get("look_streak") or engine.lookact.look_streak or 0), max(engine.lookact.block_at - 5, 0))
    engine.watcher.load(es.get("watcher"))
    engine._progress = _int(es.get("progress"))
    engine._loop_note_at = _int(es.get("loop_note_at"), -99, lowest=-99)
    engine._in_place_failures = _int(es.get("in_place_failures"))
    ctx._compact_retry_at = float(es.get("compact_retry_at") or 0.0)
    engine._sigs = [(str(h[0]), str(h[1]), str(h[2]), bool(h[3]), str(h[4]) if len(h) > 4 else "", None, None, None)
                    for h in es.get("sigs") or [] if isinstance(h, (list, tuple)) and len(h) >= 4][-LOOP_WINDOW:]
    engine.dead_ends.load(es.get("dead_ends"))                       # absent / damaged in an old session: empty
    engine._interactive_hinted = bool(es.get("interactive_hinted"))
    owned = es.get("owned")                                           # absent / damaged in an old session: nothing is owned (rm asks)
    engine.tools.owned = {k: v for k, v in list(owned.items())[-500:] if isinstance(k, str) and v in ("run", "write")} if isinstance(owned, dict) else {}
    engine._noted = {tuple(x) for x in es.get("noted") or []}
    engine._declined = {tuple(x) for x in es.get("declined") or []}
    engine.session_notes = [str(x) for x in es.get("session_notes") or []]
    engine.briefs = [str(x) for x in es.get("briefs") or []]
    engine.intake_seen = {str(x) for x in es.get("intake_seen") or []}
    cache = es.get("split_cache")
    engine.split_cache = {str(k): v for k, v in cache.items() if isinstance(v, list)} if isinstance(cache, dict) else {}
    if hasattr(engine, "checks"):
        engine.checks.load_json(es.get("checks"))
    from agent.codeagent.stages import StagePlan

    engine.stage_plan = StagePlan.from_json(data.get("stages"))          # absent in sessions saved before stages existed
    engine.stage_goal = None
    engine.session_id = str(data["id"])
    engine.session_created = float(data.get("created") or time.time())
    pending = data.get("pending") or {}
    engine.last_stopped = "" if data.get("state") == "finished" else str(pending.get("stopped") or "interrupted")
    engine.in_turn = False
    engine._started = True

    changes = verify_ledger(engine.root, led)
    _rebuild_problems(engine, led, [str(x) for x in es.get("dirty") or []])
    age = time.time() - float(data.get("updated") or time.time())
    parts = [f"SESSION RESUMED (saved {age_text(age)} ago; the process was restarted, the conversation above is intact)."]
    if changes:
        parts.append("Changed on disk since then: " + "; ".join(changes) + ". Re-read only what you need of these.")
    elif led.get("files"):
        parts.append("Every file in the ledger is unchanged on disk.")
    if data.get("state") == "in progress" or pending.get("stopped") == "killed":
        parts.append("The last prompt was cut off mid-step (killed): the last step's effect may be missing or partial - check before relying on it.")
    elif engine.last_stopped:
        parts.append(f"The last prompt ended early ({engine.last_stopped}).")
    sys0 = str(msgs[0].get("content", "")) if msgs and msgs[0].get("role") == "system" else ""
    prof = engine.profile.text()
    if sys0 and prof not in sys0:                                   # keep the prefix byte-identical; report the difference instead
        parts.append("The PROJECT block changed since the session started; it is now:\n" + prof)
    plan = engine.plan.plan
    if plan is not None and plan.open_items():
        parts.append(f"Acceptance list: {plan.confirmed()} of {len(plan.items)} confirmed; open: "
                     + "; ".join(f"{i.id}. {i.text}" for i in plan.open_items()[:6]))
    if engine.dead_ends:
        parts.append(engine.dead_ends.text())                       # the VERIFIED STATE added on resume carries it too (session.append_verified de-duplicates)
    engine.resume_note = " ".join(parts)
    return {"age": age, "changes": changes, "repaired": repaired, "state": data.get("state"), "stopped": engine.last_stopped,
            "title": str(data.get("title") or title_of(ctx.user_requests)), "steps": ctx.step, "files": len(ctx.ledger.files),
            "task": pending.get("task") or (ctx.user_requests[-1] if ctx.user_requests else "")}


RESUME_CHECK_MAX = 200


def _int(v: Any, default: int = 0, lowest: int = 0) -> int:
    try:
        return max(int(v), lowest)
    except (TypeError, ValueError):
        return default


def _rebuild_problems(engine: Any, led: dict[str, Any], saved_dirty: list[str]) -> None:
    """The dirty set and the syntax / static problems live in memory only. After a restart: the saved dirty set plus every ledger file
    that changed on disk since the save (a killed run may have left it half-written); those (and the saved dirty ones) are checked
    again, so the final gate still sees a file that was left broken. At most RESUME_CHECK_MAX files."""
    from agent.codeagent import edits

    tools, root = engine.tools, Path(engine.root)
    tools.dirty, tools.syntax_bad, tools.static_bad = set(saved_dirty), {}, {}
    stamps = led.get("stamps") or {}
    todo: list[str] = []
    for p in sorted(engine.ctx.ledger.files):
        f = Path(p) if Path(p).is_absolute() else root / p
        try:
            st = f.stat()
        except OSError:
            continue
        old = stamps.get(p)
        moved = old is not None and list(old) != [st.st_size, st.st_mtime_ns]
        if moved:
            tools.dirty.add(p)
        if moved or old is None or p in saved_dirty:
            todo.append(p)
    for p in todo[:RESUME_CHECK_MAX]:
        f = Path(p) if Path(p).is_absolute() else root / p
        try:
            if f.stat().st_size > 2_000_000:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        err = edits.check_syntax(str(f), text, getattr(tools, "check_cmd", None))
        if err:
            tools.syntax_bad[p] = err
        elif f.suffix == ".py":
            tools._static_note(f, text)


def verify_ledger(root: Path, led: dict[str, Any]) -> list[str]:
    """Ledger files changed or deleted since the save (size or mtime differ). Short facts, no file contents."""
    stamps = led.get("stamps") or {}
    out: list[str] = []
    for p, e in sorted((led.get("files") or {}).items()):
        f = Path(p) if Path(p).is_absolute() else Path(root) / p
        old = stamps.get(p)
        if not f.exists():
            if not old or old[0] != -1:
                out.append(f"{p} deleted")
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if old and [st.st_size, st.st_mtime_ns] == list(old):
            continue
        if not old and isinstance(e, dict) and st.st_size == e.get("size"):
            continue                                                # old-format session: size is all there is
        try:
            n = f.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            out.append(f"{p} changed (now {n} lines)")
        except OSError:
            out.append(f"{p} changed")
    return out[:20]
