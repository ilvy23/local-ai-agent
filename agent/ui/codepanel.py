"""Pure rendering of the live code-agent state into rich renderables for the terminal UI.

No engine, no threads, no console: every `render_*` takes plain dicts (livestate.snapshot, planview.to_dict, EventLog records or
UiEvents) and returns a rich renderable whose lines are already clipped to `width`, so the same output holds at 40, 60 and 100+ columns and in
plain/ASCII mode (agent/ui/style.py). `CodePanel` is a thin stateful wrapper that remembers the latest snapshot / events / diffs.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text

from agent.ui import style

DEFAULT_WIDTH = 80
DIFF_MAX_LINES = 40
EXPAND_MAX_LINES = 12

# stage status -> (unicode icon, ascii icon, rich style)
_STAGE = {
    "pending": ("○", "o", "dim"),
    "active": ("▶", ">", "bold cyan"),
    "done": ("✓", "+", "green"),
    "blocked": ("✗", "x", "red"),
}


def _g(plain: bool | None) -> tuple[bool, style.Glyphs]:
    p = style.plain_mode() if plain is None else bool(plain)
    return p, style.glyphs(p)


def _t(text: str, st: str, plain: bool) -> Text:
    return Text(text, style="" if plain else st)


def _line(parts: Iterable[tuple[str, str]], width: int, plain: bool, g: style.Glyphs) -> Text:
    """Join (text, style) parts into one Text clipped to `width` (ellipsis at the end)."""
    out = Text(no_wrap=True, overflow="ellipsis")
    used = 0
    for txt, st in parts:
        if used >= width:
            break
        room = width - used
        if len(txt) > room:
            txt = style.trunc(txt, room, g) if room > 1 else txt[:room]
        out.append(txt, style="" if plain else st)
        used += len(txt)
    return out


def _pack(items: list[str], width: int, sep: str = " · ") -> list[str]:
    """Greedy-pack short items into lines of at most `width` characters."""
    lines: list[str] = []
    cur = ""
    for it in items:
        if not it:
            continue
        cand = it if not cur else cur + sep + it
        if cur and len(cand) > width:
            lines.append(cur)
            cur = it
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _rec(e: Any) -> tuple[str, dict[str, Any]]:
    """(kind, data) of an EventLog record dict or a UiEvent."""
    if isinstance(e, Mapping):
        return str(e.get("kind", "")), dict(e)
    return str(getattr(e, "kind", "")), dict(getattr(e, "data", None) or {})


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── (a) header ────────────────────────────────────────────────────────────

def render_header(snapshot: Mapping[str, Any] | None, *, width: int = DEFAULT_WIDTH, plain: bool | None = None) -> RenderableType:
    """What is happening NOW + elapsed + step n/budget (source) <= ceiling + ctx % + tok/s + model + mode (+ queued steer messages)."""
    p, g = _g(plain)
    s = dict(snapshot or {})
    state = str(s.get("state") or "idle")
    now = str(s.get("now") or "") or state
    dot = {"running": ("●" if not p else "*", "cyan"), "paused": ("‖" if not p else "||", "yellow"),
           "waiting": ("?", "yellow"), "idle": ("○" if not p else "o", "dim")}.get(state, ("•" if not p else "*", "dim"))
    head = _line([(dot[0] + " ", dot[1]), (now, "bold" if state == "running" else "")], width, p, g)

    step, budget, ceil = int(_num(s.get("step"))), int(_num(s.get("budget_now"))), int(_num(s.get("step_ceiling")))
    src = str(s.get("budget_source") or "")
    step_txt = f"step {step}/{budget}" if budget else f"step {step}"
    if budget and src and not style.layout(width) == "narrow":
        step_txt += f" ({src})"
    if ceil and ceil != budget and style.layout(width) != "narrow":
        step_txt += f" max {ceil}"
    items = [step_txt]
    if s.get("elapsed_s"):
        items.append(style.fmt_duration(_num(s["elapsed_s"])))
    if s.get("ctx_limit"):
        items.append(f"ctx {_num(s.get('ctx_percent')):.0f}%")
    if _num(s.get("tps")) > 0:
        items.append(f"{_num(s['tps']):.1f} tok/s")
    if int(_num(s.get("queued"))):
        items.append(f"{int(_num(s['queued']))} queued")
    if style.layout(width) == "narrow":
        who = [x for x in (str(s.get("model") or ""), str(s.get("mode") or "")) if x]
        rows = _pack(items, width)
        if who:
            rows += _pack([" ".join(who)], width)
    else:
        items += [x for x in (str(s.get("model") or ""), str(s.get("mode") or "")) if x]
        rows = _pack(items, width)
    d = "dim"
    return Group(head, *[_line([(r, d)], width, p, g) for r in rows])


# ── (b) plan ──────────────────────────────────────────────────────────────

def _stage_budget(st: Mapping[str, Any], narrow: bool) -> str:
    lim_s, lim_n = _num(st.get("budget_limit_s")), int(_num(st.get("budget_limit_steps")))
    used_s, used_n = _num(st.get("budget_used_s")), int(_num(st.get("budget_used_steps")))
    share = _num(st.get("budget_share"))
    if narrow:
        return f"{share * 100:.0f}%" if (lim_s or lim_n) else (f"{used_n}st" if used_n else "")
    t = f"{used_s:.0f}s" + (f"/{lim_s:.0f}s" if lim_s else "")
    n = f"{used_n}" + (f"/{lim_n}" if lim_n else "") + " steps"
    return f"{t} {n}" + (f" ({share * 100:.0f}%)" if lim_s or lim_n else "")


def render_plan(plan: Mapping[str, Any] | None, *, width: int = DEFAULT_WIDTH, plain: bool | None = None) -> RenderableType:
    """Stages with status icon, per-stage time/steps vs budget, acceptance checklist n/m with evidence, dead-ends count.
    `plan` is planview.to_dict(view) (also what livestate.snapshot()['plan'] carries)."""
    p, g = _g(plain)
    if not plan or not (plan.get("stages") or plan.get("checklist")):
        return _line([("no plan yet", "dim")], width, p, g)
    narrow = style.layout(width) == "narrow"
    rows: list[Text] = []
    stages = list(plan.get("stages") or [])
    if stages:
        done = sum(1 for s in stages if s.get("status") == "done")
        rows.append(_line([(f"PLAN {done}/{len(stages)} stages", "bold")] + ([(f" ({plan.get('source')})", "dim")] if plan.get("source") and not narrow else []),
                          width, p, g))
    for s in stages:
        uni, asc, st = _STAGE.get(str(s.get("status")), _STAGE["pending"])
        icon = asc if p else uni
        bud = _stage_budget(s, narrow)
        title = str(s.get("title") or "")
        fixed = len(f" {icon} {s.get('id')} ") + (len(bud) + 2 if bud else 0)
        title = style.trunc(title, max(4, width - fixed), g)
        pad = max(1, width - len(f" {icon} {s.get('id')} {title}") - len(bud)) if bud else 0
        parts = [(f" {icon} ", st), (f"{s.get('id')} ", "dim"), (title, st if s.get("status") == "active" else "")]
        if bud:
            parts += [(" " * pad, ""), (bud, "dim")]
        rows.append(_line(parts, width, p, g))
        if s.get("status") == "blocked" and s.get("reason"):
            rows.append(_line([("     " + str(s["reason"]), "red")], width, p, g))
    cl = list(plan.get("checklist") or [])
    if cl:
        rows.append(_line([(f"ACCEPTANCE {int(plan.get('checklist_confirmed') or 0)}/{int(plan.get('checklist_total') or len(cl))}", "bold")], width, p, g))
        for it in cl:
            mark = ("[x]" if p else "☑") if it.get("done") else ("[ ]" if p else "☐")
            ev = "" if narrow or not (it.get("done") and it.get("evidence")) else f"  {g.arrow} {it['evidence']}"
            rows.append(_line([(f" {mark} ", "green" if it.get("done") else "dim"), (str(it.get("text") or ""), ""), (ev, "dim")], width, p, g))
    if int(plan.get("dead_ends") or 0):
        rows.append(_line([(f" dead ends: {int(plan['dead_ends'])}", "yellow")], width, p, g))
    return Group(*rows)


# ── (c) timeline ──────────────────────────────────────────────────────────

def _args_text(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except ValueError:
            return raw
    return json.dumps(raw, indent=2, ensure_ascii=False, default=str)


def _block(text: str, prefix: str, width: int, p: bool, g: style.Glyphs, limit: int = EXPAND_MAX_LINES) -> list[Text]:
    lines = str(text).splitlines() or [""]
    out = [_line([(prefix + ln, "dim")], width, p, g) for ln in lines[:limit]]
    if len(lines) > limit:
        out.append(_line([(f"{prefix}{g.ellipsis} {len(lines) - limit} more lines", "dim")], width, p, g))
    return out


def timeline_rows(events: Iterable[Any]) -> list[tuple[str, str, str, str, dict[str, Any]]]:
    """Collapse raw events to display rows: (icon_kind, label, summary, duration_text, data). icon_kind in ok|fail|run|info|warn."""
    evs = [_rec(e) for e in events]
    has_result = any(k == "tool_result" for k, _ in evs)
    pending: list[int] = []
    rows: list[list[Any]] = []
    for kind, d in evs:
        if kind == "tool_start":
            pending.append(len(rows))
            rows.append(["run", str(d.get("name", "tool")), str(d.get("summary", "")), "", d])
        elif kind == "tool_result":
            idx = next((i for i in pending if rows[i][1] == str(d.get("name", ""))), None)
            dur = style.fmt_duration(_num(d.get("seconds"))) if d.get("seconds") is not None else ""
            ok = "ok" if d.get("ok", True) else "fail"
            row = [ok, str(d.get("name", "tool")), str(d.get("summary", "")), dur, {**(rows[idx][4] if idx is not None else {}), **d}]
            if idx is not None:
                pending.remove(idx)
                rows[idx] = row
            else:
                rows.append(row)
        elif kind == "tool" and not has_result:
            dur = style.fmt_duration(_num(d.get("seconds"))) if d.get("seconds") is not None else ""
            rows.append(["ok" if d.get("ok", True) else "fail", str(d.get("name", "tool")), "", dur, d])
        elif kind == "edit":
            sign = "new" if d.get("created") else "edit"
            rows.append(["info", sign, f"{d.get('path', '')} +{int(_num(d.get('added')))} -{int(_num(d.get('removed')))}", "", d])
        elif kind == "check":
            st = str(d.get("status", ""))
            rows.append(["ok" if st in ("pass", "passed", "ok") else "fail" if st in ("fail", "failed", "error") else "warn", "check",
                         f"{d.get('id', '')} {d.get('check_kind', '')} {st}".strip(),
                         style.fmt_duration(_num(d["seconds"])) if d.get("seconds") is not None else "", d])
        elif kind == "compact":
            rows.append(["warn", "compact", f"context {style.fmt_tokens(_num(d.get('before')))} {'->'} {style.fmt_tokens(_num(d.get('after')))}", "", d])
        elif kind in ("mask", "stage_mask"):
            rows.append(["info", "mask", f"context {style.fmt_tokens(_num(d.get('before')))} -> {style.fmt_tokens(_num(d.get('after')))}", "", d])
        elif kind in ("loop", "loop_abort"):
            rows.append(["warn" if kind == "loop" else "fail", "loop guard" if kind == "loop" else "loop abort",
                         f"{d.get('tool', '')} x{d.get('repeats', '')}".strip(), "", d])
        elif kind == "budget_extend":
            rows.append(["info", "budget", f"extended to {d.get('to', d.get('steps', ''))} steps", "", d])
        elif kind == "budget_set":
            rows.append(["info", "budget", f"{d.get('steps', '')} steps (ceiling {d.get('ceiling', '')})", "", d])
        elif kind in ("stage_start", "stage_done", "stage_blocked"):
            word = {"stage_start": "start", "stage_done": "done", "stage_blocked": "blocked"}[kind]
            rows.append(["fail" if kind == "stage_blocked" else "ok" if kind == "stage_done" else "info", f"stage {d.get('id', '')}",
                         f"{word} {d.get('title', d.get('reason', ''))}".strip(), "", d])
        elif kind == "steer":
            rows.append(["info", "you", str(d.get("text", d.get("count", ""))), "", d])
        elif kind == "pause":
            rows.append(["warn", "pause" if d.get("paused") else "resume", "", "", d])
        elif kind in ("autopilot_start", "autopilot_end", "autopilot_decision"):
            rows.append(["info", kind.replace("_", " "), str(d.get("text", d.get("reason", d.get("verdict", "")))), "", d])
    return [tuple(r) for r in rows]  # type: ignore[misc]


def render_timeline(events: Iterable[Any], verbose: bool = False, *, width: int = DEFAULT_WIDTH, plain: bool | None = None,
                    limit: int = 12) -> RenderableType:
    """The last `limit` interesting events, one line each (`verbose` adds full args / result / diff under tool and edit lines)."""
    p, g = _g(plain)
    rows = timeline_rows(events)[-max(1, limit):]
    if not rows:
        return _line([("no activity yet", "dim")], width, p, g)
    icons = {"ok": (g.check, "green"), "fail": (g.cross, "red"), "run": ("…" if not p else "~", "cyan"), "info": (g.bullet, "dim"),
             "warn": ("!", "yellow")}
    out: list[Text] = []
    for kind, label, summary, dur, d in rows:
        icon, st = icons[kind]
        tail = f" {dur}" if dur else ""
        head = f" {icon} {label}"
        room = max(0, width - len(head) - len(tail) - 1)
        body = style.trunc(summary, room, g) if summary and room > 3 else ""
        pad = max(1, width - len(head) - (1 + len(body) if body else 0) - len(tail))
        out.append(_line([(f" {icon} ", st), (label, "bold" if kind in ("fail", "run") else ""), (" " + body if body else "", "dim"),
                          (" " * pad + dur if dur else "", "dim")], width, p, g))
        if verbose:
            if d.get("args") and label != "edit":
                out += _block(_args_text(d["args"]), "     ", width, p, g)
            if d.get("result"):
                out += _block(str(d["result"]), "     = ", width, p, g)
            if d.get("diff"):
                out.append(render_diff(str(d["diff"]), width=width, plain=p, max_lines=8))  # type: ignore[arg-type]
    return Group(*out)


# ── (d) files + diff ──────────────────────────────────────────────────────

def render_diff(diff: str, *, width: int = DEFAULT_WIDTH, plain: bool | None = None, max_lines: int = DIFF_MAX_LINES) -> RenderableType:
    """A unified diff: rich Syntax (diff lexer) normally, uncoloured clipped text in plain mode. Long diffs are cut with a count."""
    p, g = _g(plain)
    lines = str(diff or "").splitlines()
    shown = lines[:max_lines]
    more = len(lines) - len(shown)
    if p:
        out = [_line([("   " + ln, "")], width, p, g) for ln in shown]
        if more > 0:
            out.append(_line([(f"   {g.ellipsis} {more} more lines", "")], width, p, g))
        return Group(*out) if out else _line([("   (empty diff)", "")], width, p, g)
    syn = Syntax("\n".join(shown), "diff", theme="ansi_dark", word_wrap=False, background_color="default", padding=(0, 1))
    parts: list[RenderableType] = [syn]
    if more > 0:
        parts.append(_line([(f"   {g.ellipsis} {more} more lines", "dim")], width, p, g))
    return Group(*parts)


def _path_tail(path: str, room: int, g: style.Glyphs) -> str:
    return path if len(path) <= room else g.ellipsis + path[-(max(1, room - len(g.ellipsis))):]


def render_files(files: Iterable[Mapping[str, Any]] | None, diffs: Mapping[str, str] | None = None, *, expand: Iterable[str] | bool = False,
                 width: int = DEFAULT_WIDTH, plain: bool | None = None) -> RenderableType:
    """FILES CHANGED with +/-; `expand` (True = all, or an iterable of paths) prints the inline unified diff from `diffs` {path: diff}."""
    p, g = _g(plain)
    rows = list(files or [])
    if not rows:
        return _line([("no files changed", "dim")], width, p, g)
    wanted = None if expand is True else set(expand or ())
    tot_a = sum(int(_num(f.get("added"))) for f in rows)
    tot_r = sum(int(_num(f.get("removed"))) for f in rows)
    out: list[RenderableType] = [_line([(f"FILES CHANGED {len(rows)}", "bold"), (f"  +{tot_a} -{tot_r}", "dim")], width, p, g)]
    for f in rows:
        path = str(f.get("path", ""))
        a, r = f"+{int(_num(f.get('added')))}", f"-{int(_num(f.get('removed')))}"
        tag = "A" if f.get("created") else "M"
        tail = f" {a} {r}"
        shown = _path_tail(path, max(6, width - len(tail) - 4), g)
        pad = max(1, width - len(f" {tag} {shown}") - len(tail))
        out.append(_line([(f" {tag} ", "green" if tag == "A" else "yellow"), (shown, ""), (" " * pad, ""), (f" {a}", "green"), (f" {r}", "red")], width, p, g))
        if diffs and path in diffs and (wanted is None or path in wanted):
            out.append(render_diff(diffs[path], width=width, plain=p))
    return Group(*out)


# ── (e) summary ───────────────────────────────────────────────────────────

def suggest_next(result: Mapping[str, Any]) -> str:
    stopped = str(result.get("stopped") or "")
    if result.get("not_verified"):
        return "fix what is not verified, or say 'continue' and I will pick it up"
    return {
        "budget": "say 'continue' to give it more steps",
        "loop": "give a hint with /say, or change the approach and say 'continue'",
        "interrupted": "say 'continue' to resume, or /undo to revert the changes",
        "error": "check the error above, then say 'continue' or /undo",
        "gate": "the acceptance list is not fully confirmed: say 'continue' to finish it",
        "verify_budget": "say 'continue' to verify the remaining items",
    }.get(stopped, "review the changes; /undo reverts them" if result.get("files") else "")


def render_summary(result: Mapping[str, Any] | None, snapshot: Mapping[str, Any] | None = None, files: Iterable[Mapping[str, Any]] | None = None,
                   *, next_cmd: str | None = None, width: int = DEFAULT_WIDTH, plain: bool | None = None) -> RenderableType:
    """End of run: outcome, what changed, the not-verified list and a suggested next step.
    `result` = {text, stopped, verified, not_verified, files, summary?} (a TurnResult as a dict)."""
    p, g = _g(plain)
    r = dict(result or {})
    snap = dict(snapshot or {})
    stopped = str(r.get("stopped") or "")
    ok = not stopped and r.get("text") is not None
    title = "DONE" if ok else f"STOPPED ({stopped or 'no answer'})"
    verdict = "" if r.get("verified") is None else ("  verified" if r.get("verified") else "  not verified")
    out: list[RenderableType] = [_line([(f" {g.check if ok else g.cross} ", "green" if ok else "yellow"), (title, "bold"), (verdict, "dim")], width, p, g)]
    text = str(r.get("text") or r.get("summary") or "").strip()
    if text:
        out += _block(text, "   ", width, p, g, limit=6)
    rows = list(files if files is not None else snap.get("files") or [])
    if rows:
        out.append(render_files(rows, width=width, plain=p))
    elif r.get("files"):
        out.append(_line([(f" changed: {', '.join(map(str, r['files']))}", "")], width, p, g))
    nv = [str(x) for x in (r.get("not_verified") or [])]
    if nv:
        out.append(_line([(" NOT VERIFIED", "bold yellow")], width, p, g))
        out += [_line([(f"   {g.bullet} {x}", "yellow")], width, p, g) for x in nv]
    nxt = next_cmd if next_cmd is not None else suggest_next(r)
    if nxt:
        out.append(_line([(f" next {g.arrow} ", "dim"), (nxt, "")], width, p, g))
    return Group(*out)


# ── stateful wrapper ──────────────────────────────────────────────────────

class CodePanel:
    """Remembers the latest snapshot / recent events / per-file diffs of one code run and renders them. `feed()` takes EventLog records or
    UiEvents (cheap; safe to call from the run's event hook if the caller serialises access)."""

    def __init__(self, *, width: int = DEFAULT_WIDTH, plain: bool | None = None, keep: int = 200) -> None:
        self.width = width
        self.plain = plain
        self.events: deque[Any] = deque(maxlen=keep)
        self.diffs: dict[str, str] = {}
        self.edited: dict[str, dict[str, Any]] = {}      # path -> {path, added, removed, created} accumulated from edit events
        self.snapshot: dict[str, Any] = {}
        self.result: dict[str, Any] | None = None

    def feed(self, event: Any) -> None:
        kind, d = _rec(event)
        if kind in ("now", "thinking", "step", "log"):
            return
        if kind == "edit" and d.get("path") and d.get("diff"):
            prev = self.diffs.get(str(d["path"]), "")
            self.diffs[str(d["path"])] = (prev + "\n" if prev else "") + str(d["diff"])
        if kind == "edit" and d.get("path"):
            cur = self.edited.setdefault(str(d["path"]), {"path": str(d["path"]), "added": 0, "removed": 0, "created": False})
            cur["added"] += int(_num(d.get("added")))
            cur["removed"] += int(_num(d.get("removed")))
            cur["created"] = cur["created"] or bool(d.get("created"))
        self.events.append(event)

    def file_rows(self) -> list[dict[str, Any]]:
        """Changed files: the engine's own totals when it has them, else what the edit events added up to."""
        return list(self.snapshot.get("files") or []) or list(self.edited.values())

    def update(self, snapshot: Mapping[str, Any] | None) -> None:
        self.snapshot = dict(snapshot or {})

    def finish(self, result: Mapping[str, Any]) -> None:
        self.result = dict(result)

    def _kw(self, width: int | None) -> dict[str, Any]:
        return {"width": width or self.width, "plain": self.plain}

    def header(self, width: int | None = None) -> RenderableType:
        return render_header(self.snapshot, **self._kw(width))

    def plan(self, width: int | None = None) -> RenderableType:
        return render_plan(self.snapshot.get("plan"), **self._kw(width))

    def timeline(self, verbose: bool = False, width: int | None = None, limit: int = 12) -> RenderableType:
        return render_timeline(list(self.events), verbose, limit=limit, **self._kw(width))

    def files(self, expand: Iterable[str] | bool = False, width: int | None = None) -> RenderableType:
        return render_files(self.file_rows(), self.diffs, expand=expand, **self._kw(width))

    def diff(self, path: str, width: int | None = None) -> RenderableType:
        return render_diff(self.diffs.get(path, ""), **self._kw(width))

    def summary(self, next_cmd: str | None = None, width: int | None = None) -> RenderableType:
        return render_summary(self.result, self.snapshot, files=self.file_rows(), next_cmd=next_cmd, **self._kw(width))

    def render(self, verbose: bool = False, width: int | None = None) -> RenderableType:
        """Header, plan, timeline and files stacked (the /code view). After `finish()` the summary replaces the timeline."""
        parts = [self.header(width), self.plan(width)]
        parts.append(self.summary(width=width) if self.result is not None else self.timeline(verbose, width))
        if self.result is None:
            parts.append(self.files(width=width))
        return Group(*parts)
