"""NightageApp: wires Dock + Renderer + TurnWorker + PromptQueue + the command REGISTRY.

Integration points (all injectable, all optional except turn_fn):
  turn_fn(prompt, emit, stop, approve)   the chat adapter / code panel runs here (see worker.py)
  handlers      {"model": fn(args, app) -> str | renderable | None}  for commands the UI does not implement
  control       object with request_compact() / queue_user_message(text) / request_steps(n) / request_mode(name) /
                request_pause() / request_resume() / request_stop(); missing methods are reported, not fatal
  footer_provider() -> dict | FooterState     model, backend, gpu, access, folder (static-ish facts)
  status_provider() -> dict[str, str]         extra lines for /status
  context_provider() -> str | renderable      /context inspector
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from agent import brand, mood
from agent.ui import commands as C
from agent.ui import style
from agent.ui.activity import ActivityRegistry
from agent.ui.commands import REGISTRY, Registry
from agent.ui.dock import KEYS_HELP, Dock, DockHooks, FooterState, state_dir
from agent.ui.events import EventBus, UiEvent
from agent.ui.queue import PromptQueue, QueueItem
from agent.ui.render import Renderer, Transcript
from agent.ui.worker import ApprovalDecision, ApprovalRequest, TurnWorker

Handler = Callable[[str, "NightageApp"], Any]

APPROVAL_WORDS = frozenset({"y", "yes", "n", "no", "a", "always", "e", "explain"})

STEP_METHODS = {
    "compact": ("request_compact", lambda a: ()),
    "say": ("queue_user_message", lambda a: (a,)),
    "steps": ("request_steps", lambda a: (int(a),)),
    "mode": ("request_mode", lambda a: (a,)),
}


class NightageApp:
    def __init__(self, turn_fn: Callable[..., Any], *, console: Console | None = None, plain: bool | None = None,
                 detail: str = "normal", registry: Registry = REGISTRY, handlers: dict[str, Handler] | None = None,
                 control: Any = None, footer_provider: Callable[[], Any] | None = None,
                 status_provider: Callable[[], dict[str, Any]] | None = None,
                 context_provider: Callable[[], Any] | None = None, config: dict[str, Any] | None = None,
                 history_path: str | os.PathLike | None = None, workdir: str | None = None, input: Any = None,  # noqa: A002
                 output: Any = None, queue: PromptQueue | None = None, bus: EventBus | None = None,
                 activities: ActivityRegistry | None = None, approval_timeout: float = 600.0, banner: bool = True,
                 live_guard: str = "suppress", exports_dir: str | os.PathLike | None = None,
                 width: Callable[[], int] | None = None) -> None:
        self.config = config or {}
        self.plain = style.plain_mode() if plain is None else plain
        self.registry = registry
        self.handlers = {k.lower(): v for k, v in (handlers or {}).items()}
        self.control = control
        self.footer_provider, self.status_provider, self.context_provider = footer_provider, status_provider, context_provider
        self.workdir = workdir or os.getcwd()
        self.bus = bus or EventBus()
        self.queue = queue if queue is not None else PromptQueue()
        self.activities = activities or ActivityRegistry(self.bus)
        self.transcript = Transcript()
        if console is None:
            console = style.make_console(plain=self.plain)
            if not self.plain:
                console = Console(force_terminal=True, highlight=False)
        self.console = console
        ui_cfg = (self.config.get("ui") or {}) if isinstance(self.config, dict) else {}
        self.renderer = Renderer(self.console, self.transcript, detail, self.plain, show_thinking=bool(ui_cfg.get("show_thinking", False)))
        self.bus.subscribe(self.renderer.handle)
        self.bus.subscribe(self._watch_mood)
        self.worker = TurnWorker(turn_fn, bus=self.bus, queue=self.queue, activities=self.activities,
                                 dispatch=self._run_queued, approval_timeout=approval_timeout,
                                 on_turn_start=self._turn_started, on_turn_end=self._turn_ended)
        self.history_path = history_path if history_path is not None else state_dir() / "history"
        self.exports_dir = Path(exports_dir) if exports_dir else state_dir() / "exports"
        self._input, self._output, self._live_guard, self._width = input, output, live_guard, width
        self.dock: Dock | None = None
        self._banner = banner
        self._mood_event = ""
        self._direct = False
        self.quit_requested = False
        self._local: dict[str, Handler] = {
            n[5:].replace("_", "-"): getattr(self, n) for n in dir(self) if n.startswith("_cmd_")
        }

    # ------------------------------------------------------------------ dock
    def build_dock(self) -> Dock:
        h = DockHooks(
            on_line=self.handle_line, busy=lambda: self.worker.busy, on_stop=self.stop_turn,
            on_toggle_verbose=self._toggle_verbose, on_clear=lambda: None, on_keys_help=self.show_keys,
            pull_back=self._pull_back, approval_pending=lambda: self.worker.pending_approval is not None,
            approval_dangerous=lambda: bool(self.worker.pending_approval and self.worker.pending_approval.risk == "DANGEROUS"),
            on_approval_key=self.answer_approval, footer=self.footer_state, activity=self.activity_snapshot,
            live_tail=self.renderer.tail, on_notice=self.renderer.notice,
        )
        self.dock = Dock(h, registry=self.registry, plain=self.plain, history_path=self.history_path, input=self._input,
                         output=self._output, cwd=lambda: self.workdir, width=self._width, live_guard=self._live_guard)
        return self.dock

    def run(self) -> int:
        dock = self.build_dock()
        if self._banner:
            self.print_banner()
        try:
            dock.run()
        finally:
            if self.worker.busy:
                self.worker.stop()
                self.worker.wait_idle(3.0)
        return 0

    def print_banner(self) -> None:
        w = self.console.width
        t = Text()
        t.append(brand.wordmark(), style="" if self.plain else f"bold {brand.ACCENT}")
        t.append(f"  {brand.TAGLINE}" if w >= 70 else "", style="" if self.plain else "dim")
        self.console.print(t)
        tips = ("Enter send | Alt+Enter newline | / commands | @ files | ? keys | Esc stop" if self.plain else
                "Enter send · Alt+Enter newline · / commands · @ files · ? keys · Esc stop")
        if w < 70:
            tips = "/ commands | @ files | ? keys | Esc stop" if self.plain else "/ commands · @ files · ? keys · Esc stop"
        self.console.print(Text(style.trunc(tips, max(10, w - 1), self.renderer.g), style="" if self.plain else "dim"))

    # ------------------------------------------------------------------ providers for the dock
    def footer_state(self) -> FooterState:
        base: Any = self.footer_provider() if self.footer_provider else {}
        fs = FooterState(**base) if isinstance(base, dict) else (base if isinstance(base, FooterState) else FooterState())
        snap = self.worker.snapshot()
        if snap.get("ctx_limit"):
            fs.ctx_used, fs.ctx_limit = snap["ctx_used"], snap["ctx_limit"]
        if snap.get("tok_s"):
            fs.tok_s = snap["tok_s"]
        fs.queue, fs.queue_paused = snap["queue"], bool(snap["queue_paused"] and snap["queue"])
        fs.agents = len(self.activities.list(active_only=True))
        fs.state = snap["state"]
        if self.config.get("ui", {}).get("mood_emotes", True) if isinstance(self.config.get("ui", {}), dict) else True:
            fs.mood = mood.mood_for(self._mood_event, self.config) if self._mood_event else ""
        if not fs.folder:
            fs.folder = self.workdir
        return fs

    def activity_snapshot(self) -> dict[str, Any]:
        return self.worker.snapshot()

    def _watch_mood(self, ev: UiEvent) -> None:
        k = ev.kind
        if k in ("error", "stuck", "done"):
            self._mood_event = k
        elif k == "status" and "search" in str(ev.data.get("state", "")):
            self._mood_event = "searching"
        elif k in ("verify",) and ev.data.get("state") == "passed":
            self._mood_event = "test_passed"

    def _turn_started(self, prompt: str) -> None:
        self._mood_event = ""
        if not self._direct:
            self.renderer.user(prompt)
        self._direct = False
        self.renderer.begin_turn(prompt)

    def _turn_ended(self, outcome: str) -> None:
        self.renderer.flush_text()
        if outcome == "interrupted":
            self.renderer.notice("interrupted - partial answer kept")
        self.transcript.end_turn(outcome)

    # ------------------------------------------------------------------ input
    def start_turn(self, prompt: str) -> bool:
        self._direct = True
        ok = self.worker.submit(prompt)
        if not ok:
            self._direct = False
        return ok

    def stop_turn(self) -> bool:
        if not self.worker.busy:
            return False
        self.worker.stop()
        self._control("request_stop")
        return True

    def _pull_back(self) -> str | None:
        item = self.queue.pull_back()
        if item is None:
            return None
        self.renderer.info(f"pulled back from the queue ({len(self.queue)} left)")
        return item.text

    def _toggle_verbose(self) -> None:
        self.renderer.info(f"detail: {self.renderer.cycle_verbose()}")

    def show_keys(self) -> None:
        for k, v in KEYS_HELP:
            self.renderer.print(Text(f"  {k:<20} {v}"))

    def handle_line(self, text: str) -> None:
        text = text.strip("\n").rstrip()
        if not text.strip():
            return
        if self.worker.pending_approval is not None and not text.lstrip().startswith("/"):
            if text.strip().lower() in APPROVAL_WORDS:
                self.answer_approval(text.strip())   # a typed word + Enter; commands (/context, /transcript ...) fall through
            else:
                self._enqueue(text)                  # anything else is the next prompt, not a wrong answer: keep it, say what is open
                self.renderer.notice("still waiting for your answer: y, n, a or e (then Enter) - your message is queued")
            return
        if text.strip() == "?":
            self.show_keys()
            return
        busy = self.worker.busy
        d = self.registry.classify(text, busy)
        if d.action == C.NOT_A_COMMAND:
            if busy:
                self._enqueue(text)
            else:
                self.start_turn(text)
        elif d.action == C.REFUSE:
            self.renderer.notice(d.notice)
        elif d.action == C.RUN_NOW:
            self._run_command(d.command, d.args)
        elif d.action == C.DEFER:
            self._enqueue(text, kind="command", note=d.notice)
        elif d.action == C.STEP_ACTION:
            self._step(d.command.name, d.args, d.notice)

    def _enqueue(self, text: str, kind: str = "prompt", note: str = "") -> QueueItem:
        item = self.queue.enqueue(text, kind, note)
        first = " ".join(text.split())
        extra = f" - {note}" if note else ""
        self.renderer.info(f"queued ({len(self.queue)}): {style.trunc(first, max(10, self.console.width - 20), self.renderer.g)}{extra}")
        return item

    def _run_queued(self, item: QueueItem) -> None:
        d = self.registry.classify(item.text, False)
        if d.action == C.RUN_NOW:
            self._run_command(d.command, d.args)

    def _step(self, name: str, args: str, notice: str) -> None:
        spec = STEP_METHODS.get(name)
        if spec is None:
            self.renderer.notice(f"/{name} is not available while a turn runs")
            return
        method, build = spec
        try:
            call_args = build(args)
        except ValueError:
            self.renderer.notice(f"/{name}: expected {'a number' if name == 'steps' else 'text'}")
            return
        if name in ("say", "mode") and not args.strip():
            self.renderer.notice(f"usage: {self.registry.get(name).usage}")
            return
        fn = getattr(self.control, method, None)
        if not callable(fn):
            self.renderer.notice(f"/{name}: this chat has no step controls (needs the code agent)")
            return
        fn(*call_args)
        self.renderer.info(notice)

    def _control(self, method: str, *args: Any) -> bool:
        fn = getattr(self.control, method, None)
        if callable(fn):
            try:
                fn(*args)
                return True
            except Exception as exc:  # noqa: BLE001
                self.renderer.notice(f"{method} failed: {exc}")
        return False

    def _run_command(self, cmd: C.Command, args: str) -> None:
        name = cmd.name
        fn = self.handlers.get(name) or self._local.get(name) or cmd.handler
        if fn is None and name in STEP_METHODS:
            fn = lambda a, app, _n=name: app._step(_n, a, f"/{_n} sent")  # noqa: E731
        if fn is None:
            self.renderer.notice(f"/{name} is not available in this build yet")
            return
        try:
            out = fn(args, self)
        except Exception as exc:  # noqa: BLE001
            self.renderer.notice(f"/{name} failed: {exc}")
            return
        if isinstance(out, str):
            if self.plain:
                out = out.replace("…", "...")
            for ln in out.splitlines() or [""]:
                self.renderer.print(Text(ln))
        elif out is not None:
            self.renderer.print(out)

    # ------------------------------------------------------------------ approvals
    def answer_approval(self, answer: str) -> bool:
        req = self.worker.pending_approval
        if req is None:
            return False
        a = answer.strip().lower()
        if a in ("e", "explain"):
            self.renderer.info(f"{req.tool}: {req.command or req.arguments}")
            self.renderer.info(f"risk {req.risk}" + (f" - {req.reason}" if req.reason else " - no further reason recorded"))
            return False
        if req.risk == "DANGEROUS" and a in ("y", "a", "always"):
            self.renderer.notice("this is DANGEROUS - type  yes  to allow it, or n to deny")
            return False
        ok = self.worker.answer_approval(a)
        if not ok:
            self.renderer.notice("answer with y, n, a or e (then Enter), or use a /command to look around first" if req.risk != "DANGEROUS" else "type  yes  to allow, n to deny (then Enter), or use a /command to look around first")
            return False
        word = "allowed" if a in ("y", "yes") else "always allowed" if a in ("a", "always") else "denied"
        self.renderer.info(f"{word}: {req.tool}")
        return True

    # ------------------------------------------------------------------ instant commands
    def _cmd_help(self, args: str, app: "NightageApp") -> Any:
        w = self.console.width
        if args.strip():
            cmd = self.registry.get(args.strip())
            if cmd is None:
                return f"no command {args.strip()}"
            lines = [f"{cmd.usage or '/' + cmd.name}", f"  {cmd.summary}", f"  group: {cmd.group} | while busy: {cmd.when}"]
            if cmd.aliases:
                lines.append("  aliases: " + ", ".join("/" + a for a in cmd.aliases))
            return "\n".join(lines)
        from agent.ui.help import render_help         # stacked layout that stays readable at 40 columns

        render_help(self.renderer.console, self.registry, plain=self.plain, width=w)
        self.renderer.print(Text("Typed while the agent is busy: instant commands run now, step commands apply at the next step, "
                                 "the rest wait. Anything else is queued.", style="" if self.plain else "dim"))

    def _cmd_status(self, args: str, app: "NightageApp") -> Any:
        snap = self.worker.snapshot()
        rows: list[tuple[str, str]] = []
        now = f"{snap['state']}"
        if snap["busy"]:
            now += f" {style.fmt_duration(snap['elapsed'])} - {snap['now']}"
        rows.append(("state", now))
        fs = self.footer_state()
        for label, val in (("model", fs.model), ("backend", " ".join(x for x in (fs.backend, fs.gpu) if x)), ("access", fs.access),
                           ("folder", fs.folder)):
            if val:
                rows.append((label, val))
        if fs.ctx_pct is not None:
            rows.append(("context", f"{style.fmt_tokens(fs.ctx_used or 0)}/{style.fmt_tokens(fs.ctx_limit or 0)} ({fs.ctx_pct}%)"))
        if snap["tok_s"]:
            rows.append(("speed", f"{snap['tok_s']:.0f} tok/s"))
        rows.append(("queue", f"{snap['queue']}" + (" (paused)" if snap["queue_paused"] else "")))
        rows.append(("running", str(len(self.activities.list(active_only=True)))))
        rows.append(("detail", self.renderer.detail))
        if self.status_provider:
            try:
                rows += [(str(k), str(v)) for k, v in self.status_provider().items()]
            except Exception as exc:  # noqa: BLE001
                rows.append(("status", f"provider failed: {exc}"))
        return "\n".join(f"{k:<9} {v}" for k, v in rows)

    def _cmd_queue(self, args: str, app: "NightageApp") -> Any:
        parts = args.split()
        sub = parts[0].lower() if parts else ""
        if sub == "clear":
            return f"cleared {self.queue.clear()} queued item(s)"
        if sub in ("rm", "remove", "now") and len(parts) == 2 and parts[1].isdigit():
            n = int(parts[1])
            if sub == "now":
                if not self.queue.move(n, 1):
                    return f"no queued item {n}"
                if self.worker.busy:
                    return f"item {n} moved to the front - it runs when the current turn ends"
                self.queue.resume()
                self.worker.advance()
                return f"running item {n}"
            item = self.queue.remove(n)
            return f"removed: {style.trunc(item.text, 60)}" if item else f"no queued item {n}"
        if sub:
            return "usage: /queue [rm N | clear | now N]"
        lines = self.queue.lines(self.console.width - 2)
        return "\n".join(lines) if lines else "queue is empty"

    def _cmd_tasks(self, args: str, app: "NightageApp") -> Any:
        parts = args.split()
        if len(parts) == 2 and parts[0] in ("stop", "pause", "resume"):
            a = self.activities.get(parts[1])
            fn = getattr(a, parts[0], None) if a else None
            if fn is None:
                return f"{parts[1]}: cannot {parts[0]}" if a else f"no task {parts[1]}"
            fn()
            return f"{parts[0]} sent to {parts[1]}"
        snaps = [s for s in self.activities.snapshot()]
        active = [s for s in snaps if s["state"] in ("running", "paused", "waiting")]
        if not active:
            return "nothing is running"
        lines = []
        for s in active:
            can = ",".join(n for n, f in (("stop", s["can_stop"]), ("pause", s["can_pause"])) if f)
            lines.append(f"{s['id']:<10} {s['state']:<8} {style.fmt_duration(s['elapsed']):>7}  "
                         f"{style.trunc(s['label'], 24, self.renderer.g)}  {style.trunc(s['now'], max(8, self.console.width - 60), self.renderer.g)}" + (f"  [{can}]" if can else ""))
        return "\n".join(lines)

    def _cmd_transcript(self, args: str, app: "NightageApp") -> Any:
        self.renderer.print_transcript(args.strip())

    def _cmd_thinking(self, args: str, app: "NightageApp") -> Any:
        parts = args.lower().split()
        sub = parts[0] if parts else ""
        if sub == "show":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            if not self.renderer.render_thinking(n):
                return "no thinking recorded yet"
            return None
        if sub in ("on", "off"):
            self.renderer.show_thinking = sub == "on"
        else:
            self.renderer.show_thinking = not self.renderer.show_thinking
        return "thinking: " + ("shown" if self.renderer.show_thinking else "collapsed") + "  (kept in memory only, never saved)"

    def _cmd_export(self, args: str, app: "NightageApp") -> Any:
        parts = args.split(None, 1)
        fmt = "md"
        rest = args.strip()
        if parts and parts[0].lower() in ("md", "markdown", "json"):
            fmt = "json" if parts[0].lower() == "json" else "md"
            rest = parts[1].strip() if len(parts) > 1 else ""
        if rest:
            path = Path(os.path.expanduser(rest))
        else:
            path = self.exports_dir / f"nightage-{time.strftime('%Y%m%d-%H%M%S')}.{fmt}"
        body = self.transcript.to_json() if fmt == "json" else self.transcript.to_markdown()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        return f"exported {fmt} to {path} (thinking text is not included)"

    def _cmd_log(self, args: str, app: "NightageApp") -> Any:
        evs = [e for e in self.transcript.select(args, 40) if e.kind not in ("token", "status")]
        if not evs:
            return "no matching events"
        from agent.ui.render import _one_line
        return "\n".join(f"{time.strftime('%H:%M:%S', time.localtime(e.ts))} {_one_line(e, max(20, self.console.width - 10), self.renderer.g)}" for e in evs)

    _cmd_events = _cmd_log

    def _cmd_usage(self, args: str, app: "NightageApp") -> Any:
        lines = self.transcript.usage_lines(self.console.width, self.renderer.g)
        if not lines:
            return "no turns yet"
        if self.console.width < 70:
            return "\n".join([*lines, f"{len(lines)} turn(s)"])
        total = sum(t.seconds for t in self.transcript.turns)
        return "\n".join([" #  outcome       time      context        speed  tools  prompt", *lines,
                          f"{len(lines)} turn(s), {style.fmt_duration(total)} total"])

    def _cmd_context(self, args: str, app: "NightageApp") -> Any:
        if self.context_provider:
            return self.context_provider()
        ctx = self.renderer._ctx
        if not ctx:
            return "no context information yet - send a message first"
        self.renderer.print(self.renderer._usage_line(ctx))
        return None

    def _cmd_stop(self, args: str, app: "NightageApp") -> Any:
        return "stopping the current turn" if self.stop_turn() else "nothing is running"

    def _cmd_pause(self, args: str, app: "NightageApp") -> Any:
        if self.worker.pause():
            self._control("request_pause")
            return None
        return "nothing to pause"

    def _cmd_resume(self, args: str, app: "NightageApp") -> Any:
        did = self.worker.resume()
        if did:
            self._control("request_resume")
        if self.queue.paused:
            self.queue.resume()
            did = True
            if not self.worker.busy:
                self.worker.advance()
        return None if did else "nothing to resume"

    def _cmd_quit(self, args: str, app: "NightageApp") -> Any:
        self.quit_requested = True
        if self.worker.busy:
            self.worker.stop()
            self.worker.wait_idle(3.0)
        if self.dock is not None:
            self.dock.request_exit()
        return None
