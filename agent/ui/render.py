"""Renderer: UiEvents -> rich cards printed ABOVE the prompt, plus the in-memory Transcript.

Design rules (see the plan, P2/P3):
  * Output goes through a rich Console whose file is the (patched) stdout, so prompt_toolkit's `patch_stdout`
    draws it above the dock. Nothing here uses `rich.Live` or `console.status` - the unfinished tail of the
    streamed answer is exposed through `Renderer.tail()` and shown by the dock in its live-tail line.
  * Assistant text is printed as complete Markdown blocks (`liveview.split_blocks`), never half a paragraph.
  * Thinking text lives only in the in-memory Transcript. It is NOT written to disk by `export_*` unless the
    caller explicitly passes include_thinking=True.
  * Detail levels: quiet (answer + errors + approvals), normal (cards collapsed to one line), verbose (full
    arguments, results and diffs).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from agent.liveview import split_blocks
from agent.ui import style
from agent.ui.events import UiEvent

DETAILS = ("quiet", "normal", "verbose")
DIFF_LINES_NORMAL = 24
RESULT_LINES_VERBOSE = 400


def to_event(item: Any, source: str = "chat") -> UiEvent:
    """Accept a UiEvent or a chat-core dict ({"type": ..., ...}) and return a UiEvent."""
    if isinstance(item, UiEvent):
        return item
    if isinstance(item, dict):
        d = dict(item)
        kind = str(d.pop("type", None) or d.pop("kind", None) or "log")
        ts = d.pop("ts", None)
        ev = UiEvent(kind, d, source=source)
        if isinstance(ts, (int, float)):
            ev.ts = float(ts)
        return ev
    return UiEvent("log", {"text": str(item)}, source=source)


# ----------------------------------------------------------------------------- transcript


@dataclass
class TurnUsage:
    turn: int
    prompt: str = ""
    started: float = field(default_factory=time.time)
    ended: float | None = None
    tokens: int = 0
    tok_s: float | None = None
    tools: int = 0
    outcome: str = "running"  # running | done | interrupted | error

    @property
    def seconds(self) -> float:
        return max(0.0, (self.ended if self.ended is not None else time.time()) - self.started)


class Transcript:
    """Everything that happened, in memory. Source for /transcript, /export, /log and /usage."""

    def __init__(self, maxlen: int = 20000) -> None:
        self._lock = threading.RLock()
        self.events: list[UiEvent] = []
        self.turns: list[TurnUsage] = []
        self.maxlen = maxlen

    def begin_turn(self, prompt: str) -> TurnUsage:
        with self._lock:
            t = TurnUsage(len(self.turns) + 1, prompt)
            self.turns.append(t)
            self.events.append(UiEvent("user", {"text": prompt}, turn=t.turn, source="user"))
            return t

    def current(self) -> TurnUsage | None:
        with self._lock:
            return self.turns[-1] if self.turns else None

    def record(self, ev: UiEvent) -> None:
        with self._lock:
            t = self.turns[-1] if self.turns and self.turns[-1].outcome == "running" else None
            if t is not None and ev.turn is None:
                ev.turn = t.turn
            self.events.append(ev)
            if len(self.events) > self.maxlen:
                del self.events[: len(self.events) - self.maxlen]
            if t is None:
                return
            d = ev.data
            if ev.kind == "context":
                t.tokens = max(t.tokens, int(d.get("used") or 0))
                if d.get("tokens_per_s"):
                    t.tok_s = float(d["tokens_per_s"])
            elif ev.kind in ("tool_call", "tool_start"):
                t.tools += 1
            elif ev.kind == "done":
                t.outcome, t.ended = "done", ev.ts
            elif ev.kind == "error":
                t.outcome, t.ended = "error", ev.ts

    def end_turn(self, outcome: str | None = None) -> None:
        with self._lock:
            if not self.turns:
                return
            t = self.turns[-1]
            if t.ended is None:
                t.ended = time.time()
            if outcome and (t.outcome == "running" or outcome == "interrupted"):
                t.outcome = outcome
            elif t.outcome == "running":
                t.outcome = "done"

    def select(self, flt: str = "", n: int = 40) -> list[UiEvent]:
        """Events whose kind equals or whose text contains `flt` (case-insensitive); the last n."""
        f = flt.strip().lower()
        with self._lock:
            items = list(self.events)
        if f:
            items = [e for e in items if e.kind.lower() == f or f in _one_line(e).lower()]
        return items[-n:] if n > 0 else items

    # ---- export
    def to_json(self, include_thinking: bool = False) -> str:
        with self._lock:
            rows = []
            for e in self.events:
                data = dict(e.data)
                if e.kind == "thinking" and not include_thinking:
                    data.pop("text", None)
                rows.append({"kind": e.kind, "ts": e.ts, "turn": e.turn, "source": e.source, "data": _jsonable(data)})
            turns = [
                {"turn": t.turn, "prompt": t.prompt, "seconds": round(t.seconds, 2), "tokens": t.tokens,
                 "tok_s": t.tok_s, "tools": t.tools, "outcome": t.outcome}
                for t in self.turns
            ]
        return json.dumps({"turns": turns, "events": rows}, indent=2, ensure_ascii=False)

    def to_markdown(self, include_thinking: bool = False, verbose: bool = True) -> str:
        out: list[str] = ["# Transcript", ""]
        text_buf: list[str] = []

        def flush() -> None:
            if text_buf:
                out.extend(["**assistant**", "", "".join(text_buf).strip(), ""])
                text_buf.clear()

        with self._lock:
            events = list(self.events)
        think_tokens = 0
        for e in events:
            d = e.data
            if e.kind == "token":
                text_buf.append(str(d.get("text", "")))
                continue
            if e.kind == "thinking":
                think_tokens = max(think_tokens, int(d.get("tokens") or 0))
                if include_thinking and d.get("text"):
                    flush()
                    out.extend(["> thinking", ">", *["> " + ln for ln in str(d["text"]).splitlines()], ""])
                continue
            flush()
            if e.kind == "user":
                out.extend([f"## Turn {e.turn or '?'}", "", "**you**", "", str(d.get("text", "")), ""])
            elif e.kind == "tool_call":
                args = d.get("arguments")
                body = json.dumps(args, indent=2, ensure_ascii=False) if verbose and isinstance(args, (dict, list)) else _args_line(args)
                out.extend([f"- tool `{d.get('name', '?')}` {body if not verbose else ''}".rstrip()])
                if verbose and isinstance(args, (dict, list)):
                    out.extend(["", "```json", body, "```"])
            elif e.kind == "tool_result":
                res = _result_text(d)
                out.extend(["", f"  result of `{d.get('name', '?')}`:", "", "```", res if verbose else res[:300], "```", ""])
            elif e.kind in ("edit", "changes"):
                if d.get("diff"):
                    out.extend([f"edit `{d.get('path', '?')}`", "", "```diff", str(d["diff"]), "```", ""])
                else:
                    out.append(f"- {_one_line(e)}")
            elif e.kind == "done":
                out.extend(["", f"_done_", ""])
            elif e.kind in ("token", "status", "context"):
                continue
            else:
                out.append(f"- {_one_line(e)}")
        flush()
        if think_tokens and not include_thinking:
            out.extend(["", f"_(thinking omitted, about {style.fmt_tokens(think_tokens)} tokens)_", ""])
        return "\n".join(out).rstrip() + "\n"

    def usage_lines(self, width: int = 100, g: style.Glyphs | None = None) -> list[str]:
        with self._lock:
            turns = list(self.turns)
        lines = []
        for t in turns:
            tok = style.fmt_tokens(t.tokens) if t.tokens else "-"
            rate = f"{t.tok_s:.0f} tok/s" if t.tok_s else "-"
            if width < 70:
                lines.append(f"{t.turn:>2} {t.outcome:<11} {style.fmt_duration(t.seconds):>6} {tok:>6}")
                continue
            lines.append(f"{t.turn:>3}  {t.outcome:<11} {style.fmt_duration(t.seconds):>7}  ctx {tok:>6}  {rate:>10}  tools {t.tools}  {style.trunc(t.prompt, max(8, width - 62), g)}")
        return lines


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return json.loads(json.dumps(v, default=str))


def _args_line(args: Any, width: int = 70, g: style.Glyphs | None = None) -> str:
    if isinstance(args, dict):
        for key in ("command", "pattern", "query", "path", "file", "url", "text"):
            if key in args and args[key] not in (None, ""):
                return style.trunc(str(args[key]), width, g)
        return style.trunc(", ".join(f"{k}={v}" for k, v in args.items()), width, g)
    return style.trunc("" if args is None else str(args), width, g)


def _result_text(d: dict[str, Any]) -> str:
    for k in ("result", "result_preview", "output", "text"):
        if d.get(k) not in (None, ""):
            v = d[k]
            return v if isinstance(v, str) else json.dumps(v, indent=2, ensure_ascii=False, default=str)
    return ""


def _one_line(e: UiEvent, width: int = 110, g: style.Glyphs | None = None) -> str:
    d = e.data
    if e.kind in ("tool_call", "tool_start"):
        s = f"{d.get('name', '?')} {_args_line(d.get('arguments'), 70, g)}"
    elif e.kind == "tool_result":
        s = f"{d.get('name', '?')} -> {style.trunc(_result_text(d), 80, g)}"
    elif e.kind in ("error",):
        s = str(d.get("message", ""))
    elif e.kind == "log":
        s = str(d.get("text") or d.get("message") or "")
    elif e.kind == "user":
        s = str(d.get("text", ""))
    elif e.kind == "token":
        s = str(d.get("text", ""))
    else:
        try:
            s = ", ".join(f"{k}={v}" for k, v in d.items() if k not in ("text", "diff", "result", "items", "arguments"))
        except Exception:  # noqa: BLE001
            s = ""
    return style.trunc(f"{e.kind}: {s}".rstrip(": "), width, g)


# ----------------------------------------------------------------------------- renderer


_TRANSPARENT_KINDS = frozenset({"status", "context"})


class Renderer:
    """Turns UiEvents into rich output. Subscribe `renderer.handle` to the EventBus."""

    def __init__(self, console: Console | None = None, transcript: Transcript | None = None, detail: str = "normal",
                 plain: bool | None = None, show_thinking: bool = False) -> None:
        self.plain = style.plain_mode() if plain is None else plain
        self.console = console or style.make_console(plain=self.plain)
        self.transcript = transcript if transcript is not None else Transcript()
        self.detail = detail if detail in DETAILS else "normal"
        self.show_thinking = show_thinking
        self.g = style.glyphs(self.plain)
        self.pal = style.palette()
        self._lock = threading.RLock()
        self._buf = ""                      # unfinished Markdown tail of the streamed answer
        self._printed_text = False
        self._pending: list[dict[str, Any]] = []   # tool calls waiting for a result
        self._edit_card_shown = False              # an edit card was printed for the tool call that is still running
        self._think_text: list[str] = []
        self._think_tokens = 0
        self._think_explicit = 0                          # summed token counts the events themselves carried
        self._thinking_log: list[tuple[int, str]] = []   # (tokens, text) of finished thinking blocks, memory only
        self._ctx: dict[str, Any] = {}
        self._turn_started: float | None = None
        self._turn_tools = 0

    # ---- public
    def set_detail(self, detail: str) -> None:
        if detail in DETAILS:
            self.detail = detail

    def cycle_verbose(self) -> str:
        self.detail = "normal" if self.detail == "verbose" else "verbose"
        return self.detail

    @property
    def verbose(self) -> bool:
        return self.detail == "verbose"

    def tail(self) -> str:
        """The unfinished last line of the answer being streamed (for the dock's live-tail row)."""
        with self._lock:
            lines = [ln for ln in self._buf.split("\n") if ln.strip()]
        return lines[-1].strip() if lines else ""

    def last_thinking(self) -> list[tuple[int, str]]:
        with self._lock:
            return list(self._thinking_log)

    def begin_turn(self, prompt: str) -> None:
        self.transcript.begin_turn(prompt)
        with self._lock:
            self._buf, self._pending, self._printed_text = "", [], False
            self._turn_started, self._turn_tools = time.time(), 0

    def handle(self, item: Any) -> None:
        """Entry point for every event. Never raises (a broken card must not kill a turn)."""
        ev = to_event(item)
        self.transcript.record(ev)
        try:
            with self._lock:
                self._dispatch(ev)
        except Exception as exc:  # noqa: BLE001
            try:
                self.console.print(Text(f"[render error: {exc}]", style="dim"))
            except Exception:  # noqa: BLE001
                pass

    def info(self, text: str, style_: str = "dim") -> None:
        self.print(Text(text, style=self._st(style_)))

    def notice(self, text: str) -> None:
        self.print(Text(f"{self.g.bullet} {text}", style=self._st("yellow")))

    def user(self, text: str) -> None:
        """Echo of a prompt that starts later (a queued one)."""
        first = " ".join(text.split())
        self.print(Text(f"{self.g.arrow} {style.trunc(first, max(10, self.console.width - 4), self.g)}", style=self._st(self.pal.accent)))

    def print(self, renderable: Any = "") -> None:
        self.console.print(renderable)

    def flush_text(self) -> None:
        with self._lock:
            self._flush_text(final=True)

    # ---- dispatch
    def _st(self, s: str) -> str:
        return "" if self.plain else s

    def _dispatch(self, ev: UiEvent) -> None:
        k, d = ev.kind, ev.data
        if k == "token":
            self._on_token(str(d.get("text", "")))
            return
        if k == "thinking":
            self._on_thinking(d)
            return
        if k in _TRANSPARENT_KINDS and getattr(self, "_on_" + k, None) is None:
            return           # progress pulses between thinking pieces must not cut one thinking block into many cards
        # any other event ends the streamed block of text / thinking so ordering stays right
        self._flush_thinking()
        if k in ("tool_call", "tool_start", "awaiting_approval", "edit", "plan", "stage_start", "stage_done",
                 "stage_blocked", "sources", "verdict", "error", "verify", "stuck", "done", "changes"):
            self._flush_text(final=True)
        handler = getattr(self, "_on_" + k, None)
        if handler is not None:
            handler(ev)

    # ---- text
    def _on_token(self, text: str) -> None:
        self._flush_thinking()
        self._buf += text
        self._flush_text(final=False)

    def _flush_text(self, final: bool) -> None:
        if not self._buf:
            return
        blocks, tail = split_blocks(self._buf)
        for b in blocks:
            self._print_markdown(b)
        self._buf = tail
        if final and self._buf.strip():
            self._print_markdown(self._buf)
            self._buf = ""
        elif final:
            self._buf = ""

    def _print_markdown(self, block: str) -> None:
        if not block.strip():
            return
        self._printed_text = True
        if self.plain:
            self.console.print(Text(block))
            self.console.print()
        else:
            self.console.print(Markdown(block))

    # ---- thinking
    def _on_thinking(self, d: dict[str, Any]) -> None:
        if d.get("text"):
            self._think_text.append(str(d["text"]))
        if d.get("tokens"):
            self._think_explicit = max(self._think_explicit, int(d["tokens"]))   # an event's count is the running total of its block
            self._think_tokens = self._think_explicit
        elif d.get("text") and not self._think_explicit:
            self._think_tokens = max(self._think_tokens, sum(len(t) for t in self._think_text) // 4)
        if not d.get("text") and not d.get("tokens") and not self._think_tokens:
            self._think_tokens = 1

    def _flush_thinking(self) -> None:
        if not self._think_tokens and not self._think_text:
            return
        text = "".join(self._think_text)
        tokens = self._think_tokens or max(1, len(text) // 4)
        self._think_text, self._think_tokens, self._think_explicit = [], 0, 0
        self._thinking_log.append((tokens, text))
        if self.detail == "quiet":
            return
        n = len(self._thinking_log)
        head = Text(f"{self.g.expand} thinking ({style.fmt_tokens(tokens)} tokens)", style=self._st("dim"))
        if self.show_thinking or self.verbose:
            if text.strip():
                self.console.print(Text(f"{self.g.collapse} thinking ({style.fmt_tokens(tokens)} tokens)", style=self._st("dim")))
                self.console.print(Text(_indent(text.strip(), "  "), style=self._st("dim italic")))
                return
        if text.strip():
            head.append(f"  /thinking show {n}", style=self._st("dim"))
        self.console.print(head)

    def render_thinking(self, n: int | None = None) -> bool:
        """Expand a stored thinking block (default: the latest) - /thinking show [n]."""
        with self._lock:
            items = list(self._thinking_log)
        if not items:
            return False
        idx = (n - 1) if n else len(items) - 1
        if not 0 <= idx < len(items):
            return False
        tokens, text = items[idx]
        self.console.print(Text(f"{self.g.collapse} thinking ({style.fmt_tokens(tokens)} tokens)", style=self._st("dim")))
        self.console.print(Text(_indent(text.strip() or "(no text recorded)", "  "), style=self._st("dim italic")))
        return True

    # ---- tools
    def _on_tool_call(self, ev: UiEvent) -> None:
        d = ev.data
        rec = {"id": d.get("id"), "name": str(d.get("name", "?")), "args": d.get("arguments"), "risk": d.get("risk") or d.get("risk_level"),
               "ts": ev.ts}
        self._pending.append(rec)
        self._turn_tools += 1
        if self.verbose:
            self.console.print(self._tool_line(rec, running=True))
            self._print_args_full(rec["args"])

    _on_tool_start = _on_tool_call

    def _take_pending(self, d: dict[str, Any]) -> dict[str, Any] | None:
        for i in range(len(self._pending) - 1, -1, -1):
            p = self._pending[i]
            if (d.get("id") is not None and p["id"] == d.get("id")) or (d.get("id") is None and p["name"] == d.get("name")):
                return self._pending.pop(i)
        return self._pending.pop() if self._pending and d.get("name") is None else None

    def _on_tool_result(self, ev: UiEvent) -> None:
        d = ev.data
        rec = self._take_pending(d) or {"name": str(d.get("name", "?")), "args": d.get("arguments"), "risk": d.get("risk"), "ts": ev.ts}
        result = _result_text(d)
        dur = d.get("duration", d.get("seconds"))
        if dur is None:
            dur = max(0.0, ev.ts - rec.get("ts", ev.ts))
        blocked = result.startswith("BLOCKED")
        ok = d.get("ok")
        if ok is None:
            ok = not (blocked or result.lower().startswith(("error", "denied")))
        rec = {**rec, "risk": d.get("risk") or d.get("risk_level") or rec.get("risk") or ("BLOCKED" if blocked else None)}
        edit_card, self._edit_card_shown = self._edit_card_shown, False
        if self.detail == "quiet" and ok:
            return
        if edit_card and ok and not self.verbose:
            return                    # the edit card above already says what this tool call did; /transcript keeps both
        self.console.print(self._tool_line(rec, dur=float(dur), ok=bool(ok), result=result))
        if self.verbose:
            self._print_result_full(result)

    _on_tool = _on_tool_result  # codeagent's "tool" event: {name, seconds, ok}

    def _tool_line(self, rec: dict[str, Any], dur: float | None = None, ok: bool | None = None, result: str = "",
                   running: bool = False) -> Text:
        w = self.console.width
        g = self.g
        mark = g.arrow if running else (g.check if ok else g.cross)
        mst = "" if running else ("green" if ok else "red")
        t = Text()
        t.append(f"{mark} ", style=self._st(mst or "dim"))
        t.append(str(rec.get("name", "?")), style=self._st("bold"))
        risk = rec.get("risk")
        badge = str(getattr(risk, "name", risk) or "").upper()
        tail = Text()
        if badge and badge != "SAFE":
            tail.append(f" [{badge}]", style=self._st(style.risk_style(badge)))
        elif badge == "SAFE" and w >= style.NORMAL_MIN:
            tail.append(" [SAFE]", style=self._st("green"))
        if dur is not None:
            tail.append(f" {style.fmt_duration(dur)}", style=self._st("dim"))
        room = max(8, w - len(t.plain) - len(tail.plain) - 2)
        summary = _args_line(rec.get("args"), room, g) if rec.get("args") else ""
        if not summary and result:
            summary = style.trunc(result, room, g)
        elif result and not ok:
            summary = style.trunc(result, room, g)
        if summary:
            t.append(" " + style.trunc(summary, room, g), style=self._st("dim"))
        t.append_text(tail)
        return t

    def _print_args_full(self, args: Any) -> None:
        if args in (None, "", {}):
            return
        body = json.dumps(args, indent=2, ensure_ascii=False, default=str) if isinstance(args, (dict, list)) else str(args)
        self.console.print(Text(_indent(body, "    "), style=self._st("dim")))

    def _print_result_full(self, result: str) -> None:
        lines = result.splitlines()
        if len(lines) > RESULT_LINES_VERBOSE:
            lines = lines[:RESULT_LINES_VERBOSE] + [f"... {len(result.splitlines()) - RESULT_LINES_VERBOSE} more lines"]
        if lines:
            self.console.print(Text(_indent("\n".join(lines), "    "), style=self._st("dim")))

    # ---- edits / diffs
    def _on_edit(self, ev: UiEvent) -> None:
        d = ev.data
        path = str(d.get("path") or d.get("file") or "?")
        added, removed = d.get("added"), d.get("removed")
        diff = str(d.get("diff") or d.get("patch") or "")
        if added is None and diff:
            added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
            removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        self._edit_card_shown = True
        head = Text()
        head.append(f"{self.g.check} ", style=self._st("green"))
        head.append("created " if d.get("created") else "edit ", style=self._st("bold"))
        head.append(style.trunc(path, max(10, self.console.width - 24), self.g))
        head.append(f" +{added or 0}", style=self._st("green"))
        head.append(f" -{removed or 0}", style=self._st("red"))
        self.console.print(head)
        if not diff or self.detail == "quiet":
            return
        lines = diff.splitlines()
        limit = len(lines) if self.verbose else DIFF_LINES_NORMAL
        shown = "\n".join(lines[:limit])
        self._print_diff(shown)
        if len(lines) > limit:
            self.console.print(Text(f"  {self.g.ellipsis} {len(lines) - limit} more lines (Ctrl-O or /transcript for all)", style=self._st("dim")))

    def _print_diff(self, diff: str) -> None:
        if self.plain:
            self.console.print(Text(diff))
        else:
            self.console.print(Syntax(diff, "diff", theme="ansi_dark", word_wrap=True, background_color="default"))

    # ---- plan / stages / verify / stuck
    def _on_plan(self, ev: UiEvent) -> None:
        d = ev.data
        items = d.get("items") or []
        if self.detail == "quiet":
            return
        total = d.get("total", len(items))
        done = d.get("confirmed", sum(1 for i in items if isinstance(i, dict) and i.get("status") in ("confirmed", "done")))
        self.console.print(Text(f"{self.g.bullet} plan {done}/{total}", style=self._st("bold")))
        if self.detail == "verbose" or len(items) <= 8:
            for it in items:
                if isinstance(it, dict):
                    mark = "[x]" if it.get("status") in ("confirmed", "done") else "[ ]"
                    txt = str(it.get("text", ""))
                    ev_ = f"  ({style.trunc(str(it['evidence']), 40, self.g)})" if it.get("evidence") and self.verbose else ""
                else:
                    mark, txt, ev_ = "[ ]", str(it), ""
                self.console.print(Text(f"  {mark} {style.trunc(txt, max(10, self.console.width - 10), self.g)}{ev_}", style=self._st("dim")))

    def _stage_line(self, label: str, d: dict[str, Any], st: str) -> None:
        if self.detail == "quiet":
            return
        name = str(d.get("name") or d.get("stage") or d.get("title") or "")
        extra = f" {style.trunc(str(d.get('summary') or d.get('reason') or ''), 60, self.g)}" if (d.get("summary") or d.get("reason")) else ""
        self.console.print(Text(f"{self.g.bullet} {label} {name}{extra}".rstrip(), style=self._st(st)))

    def _on_stage_start(self, ev: UiEvent) -> None:
        self._stage_line("stage", ev.data, "cyan")

    def _on_stage_done(self, ev: UiEvent) -> None:
        self._stage_line("stage done", ev.data, "green")

    def _on_stage_blocked(self, ev: UiEvent) -> None:
        self._stage_line("stage blocked", ev.data, "yellow")

    def _on_verify(self, ev: UiEvent) -> None:
        d = ev.data
        state = str(d.get("state", ""))
        if self.detail == "quiet" and state not in ("failed", "gave_up"):
            return
        st = {"passed": "green", "failed": "red", "gave_up": "red"}.get(state, "yellow")
        cmd = style.trunc(str(d.get("command", "")), 50, self.g)
        att = f" ({d.get('attempt')}/{d.get('max_attempts')})" if d.get("attempt") else ""
        self.console.print(Text(f"{self.g.bullet} verify {state}{att} {cmd}".rstrip(), style=self._st(st)))
        if self.verbose and d.get("tail"):
            self.console.print(Text(_indent(str(d["tail"]), "    "), style=self._st("dim")))

    def _on_check(self, ev: UiEvent) -> None:
        d = ev.data
        if self.detail == "quiet":
            return
        ok = d.get("ok", d.get("passed"))
        self.console.print(Text(f"{self.g.bullet} check {d.get('name', d.get('command', ''))} {'ok' if ok else 'failed'}".rstrip(),
                                style=self._st("green" if ok else "red")))

    def _on_stuck(self, ev: UiEvent) -> None:
        d = ev.data
        what = "repeating the same lookup" if d.get("kind") == "lookup" else "repeating itself"
        q = f": {style.trunc(str(d['query']), 50, self.g)}" if d.get("query") else ""
        self.console.print(Text(f"{self.g.bullet} stuck - {what}{q}. Esc stops it, or send a hint.", style=self._st("yellow")))

    def _on_steer(self, ev: UiEvent) -> None:
        self.console.print(Text(f"{self.g.bullet} message delivered to the agent", style=self._st("dim")))

    # ---- sources / verdict
    def _on_sources(self, ev: UiEvent) -> None:
        d = ev.data
        items = d.get("items") or []
        if self.detail == "quiet":
            return
        self.console.print(Text(f"{self.g.bullet} {len(items)} sources" + (f" for {style.trunc(str(d['query']), 50, self.g)}" if d.get("query") else ""), style=self._st("bold")))
        limit = len(items) if self.verbose else 5
        for it in items[:limit]:
            trust = it.get("trust") or {}
            lvl = trust.get("level") or trust.get("tier") or "?"
            flags = it.get("flags") or []
            line = Text(f"  [{it.get('n', '?')}] ")
            line.append(f"{lvl:<6}", style=self._st({"high": "green", "medium": "yellow", "low": "red"}.get(str(lvl), "dim")))
            dom = str(it.get("domain") or it.get("url") or "")
            line.append(" " + style.trunc(dom, max(10, self.console.width - 24), self.g))
            if flags:
                line.append(" !" + ",".join(map(str, flags[:2])), style=self._st("yellow"))
            self.console.print(line)
        if len(items) > limit:
            self.console.print(Text(f"  {self.g.ellipsis} {len(items) - limit} more (Ctrl-O)", style=self._st("dim")))

    def _on_verdict(self, ev: UiEvent) -> None:
        d = ev.data
        label = str(d.get("label", ""))
        st = {"verified": "green", "likely": "green", "conflicting": "red", "low_trust": "yellow", "single_source": "yellow"}.get(label, "dim")
        self.console.print(Text(f"{self.g.bullet} {label.replace('_', ' ')}: {style.trunc(str(d.get('summary', '')), 80, self.g)}".rstrip(": "), style=self._st(st)))
        if self.detail != "quiet":
            for w in (d.get("warnings") or [])[:3]:
                self.console.print(Text(f"    ! {style.trunc(str(w), 80, self.g)}", style=self._st("yellow")))

    # ---- context / status / misc
    def _on_context(self, ev: UiEvent) -> None:
        self._ctx = dict(ev.data)
        if ev.data.get("compacted") and self.detail != "quiet":
            self.console.print(Text(f"{self.g.bullet} context compacted", style=self._st("dim")))
        elif self.verbose:
            self.console.print(self._usage_line(ev.data))

    def _usage_line(self, d: dict[str, Any]) -> Text:
        used, limit = int(d.get("used") or 0), int(d.get("limit") or 0)
        pct = d.get("percent", round(100 * used / limit) if limit else 0)
        bits = [f"ctx {style.fmt_tokens(used)}/{style.fmt_tokens(limit)} ({pct}%)" if limit else f"ctx {style.fmt_tokens(used)}"]
        if d.get("tokens_per_s"):
            bits.append(f"{float(d['tokens_per_s']):.0f} tok/s")
        return Text("  " + " | ".join(bits) if self.plain else "  " + " · ".join(bits), style=self._st("dim"))

    def _on_compact(self, ev: UiEvent) -> None:
        if self.detail != "quiet":
            self.console.print(Text(f"{self.g.bullet} compacting context", style=self._st("dim")))

    def _on_log(self, ev: UiEvent) -> None:
        d = ev.data
        msg = str(d.get("text") or d.get("message") or "")
        if msg and d.get("notice"):
            self.console.print(Text(f"{self.g.bullet} {msg}", style=self._st("yellow")))
        elif msg and self.detail != "quiet" and not d.get("activity"):
            self.console.print(Text(f"  {style.trunc(msg, max(10, self.console.width - 4), self.g)}", style=self._st("dim")))

    def _on_changes(self, ev: UiEvent) -> None:
        d = ev.data
        files = d.get("files") or []
        if self.detail == "quiet" or not files:
            return
        self.console.print(Text(f"{self.g.bullet} {len(files)} file(s) changed +{d.get('added', 0)} -{d.get('removed', 0)}"
                                + ("  (/undo)" if d.get("undo_available") else ""), style=self._st("bold")))

    def _on_file_ready(self, ev: UiEvent) -> None:
        d = ev.data
        self.console.print(Text(f"{self.g.bullet} file ready: {d.get('filename') or d.get('name') or d.get('kind')}", style=self._st("cyan")))

    def _on_budget_set(self, ev: UiEvent) -> None:
        if self.verbose:
            self.console.print(Text(f"{self.g.bullet} step budget {ev.data.get('steps')} (ceiling {ev.data.get('ceiling')})", style=self._st("dim")))

    def _on_budget_extend(self, ev: UiEvent) -> None:
        if self.detail != "quiet":
            self.console.print(Text(f"{self.g.bullet} step budget extended to {ev.data.get('to', ev.data.get('steps'))}", style=self._st("dim")))

    def _on_loop_abort(self, ev: UiEvent) -> None:
        self.console.print(Text(f"{self.g.bullet} loop guard stopped a repeating action", style=self._st("yellow")))

    # ---- approvals, errors, done
    def approval_card(self, req: dict[str, Any]) -> Any:
        risk = str(getattr(req.get("risk_level") or req.get("risk"), "name", req.get("risk_level") or req.get("risk") or "CAUTION")).upper()
        cmd = req.get("command") or req.get("description") or _args_line(req.get("arguments"), 400, self.g)
        parts: list[Any] = [Text(f"{req.get('tool_name') or req.get('tool') or 'action'}", style=self._st("bold"))]
        if cmd:
            parts.append(Text(str(cmd), style=self._st("cyan")))
        if req.get("diff"):
            diff = str(req["diff"]).splitlines()
            lim = len(diff) if self.verbose else DIFF_LINES_NORMAL
            parts.append(Text("\n".join(diff[:lim])) if self.plain else Syntax("\n".join(diff[:lim]), "diff", theme="ansi_dark", background_color="default"))
        if req.get("reason"):
            parts.append(Text(f"why: {req['reason']}", style=self._st("dim")))
        if risk == "DANGEROUS":
            keys = "type  yes  to allow, n to deny, e to explain"
        else:
            keys = "type y allow once, n deny, a always for this folder, e explain - then Enter"
        parts.append(Text(keys, style=self._st("yellow")))
        title = f"approval needed [{risk}]"
        if self.plain:
            return Group(Text("-- " + title + " --"), *parts)
        return Panel(Group(*parts), title=title, title_align="left", border_style=style.risk_style(risk), padding=(0, 1))

    def _on_awaiting_approval(self, ev: UiEvent) -> None:
        self.console.print(self.approval_card(ev.data))

    def error_card(self, message: str) -> Any:
        step = next_step_for(message)
        t = Text()
        t.append(f"{self.g.cross} {message}", style=self._st("red"))
        if step:
            t.append(f"\n  next: {step}", style=self._st("yellow"))
        return t

    def _on_error(self, ev: UiEvent) -> None:
        self.console.print(self.error_card(str(ev.data.get("message") or ev.data.get("error") or "unknown error")))
        self._turn_started = None

    def _on_done(self, ev: UiEvent) -> None:
        if not self._printed_text and ev.data.get("reply"):
            self._print_markdown(str(ev.data["reply"]))   # nothing was streamed: show the final reply once
        if self.detail != "quiet":
            started = self._turn_started
            bits = []
            if started:
                bits.append(style.fmt_duration(ev.ts - started))
            used = self._ctx.get("used")
            if used:
                bits.append(f"ctx {style.fmt_tokens(used)}" + (f" ({self._ctx.get('percent')}%)" if self._ctx.get("percent") is not None else ""))
            if self._ctx.get("tokens_per_s"):
                bits.append(f"{float(self._ctx['tokens_per_s']):.0f} tok/s")
            if self._turn_tools:
                bits.append(f"{self._turn_tools} tool call{'s' if self._turn_tools != 1 else ''}")
            if bits:
                self.console.print(Text("  " + (" | " if self.plain else " · ").join(bits), style=self._st("dim")))
        self._turn_started = None

    # ---- transcript view
    def print_transcript(self, flt: str = "") -> None:
        """Replay the whole transcript with full detail (/transcript)."""
        prev = self.detail, self.show_thinking
        sub = Renderer(self.console, Transcript(), "verbose", self.plain, show_thinking=True)
        sub.g, sub.pal = self.g, self.pal
        events = self.transcript.select(flt, n=0)
        if not events:
            self.info("(transcript is empty)")
            return
        for e in events:
            if e.kind == "user":
                sub.user(str(e.data.get("text", "")))
            elif e.kind == "awaiting_approval":
                sub.console.print(sub.approval_card(e.data))
            else:
                sub.handle(e)
        sub.flush_text()
        self.detail, self.show_thinking = prev


def _indent(text: str, pad: str) -> str:
    return "\n".join(pad + ln for ln in text.splitlines())


_NEXT_STEPS = (
    (("connection refused", "connecterror", "cannot connect", "failed to connect", "ollama is not", "unreachable"),
     "start Ollama (`ollama serve`) and send the message again, or run /doctor"),
    (("no such model", "model is not", "try pulling", "ollama pull", "model not found", "pull the model"),
     "install the model with `ollama pull <name>` or pick another one with /model"),
    (("context", "num_ctx", "too long", "token limit", "out of memory"),
     "run /compact to shrink the conversation, or /clear to start fresh"),
    (("timeout", "timed out"), "the model is slow or busy - wait, or press Esc and try again"),
    (("too many tool calls",), "say \"continue\" to go on, or rephrase the request"),
    (("denied", "blocked"), "the safety policy refused this; see /access to change folder access"),
)


def next_step_for(message: str) -> str:
    m = message.lower()
    if re.search(r"model\b.*\bnot found|not found.*\bmodel", m):
        return _NEXT_STEPS[1][1]
    for keys, step in _NEXT_STEPS:
        if any(k in m for k in keys):
            return step
    return "see /log for details, then retry or rephrase"
