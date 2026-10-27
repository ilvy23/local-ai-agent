"""The real code agent on the terminal UI: `run_code_ui(...)` is what `nightage code` starts (the classic REPL stays `--classic`).

Startup/shutdown follow `session.run_session` (folder lock, foreground marker, GPU freeing, resume of a saved session, per-folder policy);
the difference is only the surface: a NightageApp (dock + renderer) whose turn function drives a `CodeRunner` on one `CodeEngine`.

    first prompt (or the CLI goal)   a goal: --plan-first / --autopilot / --stages / --steps / --mode apply to it
    later prompts                    follow-up turns on the same engine, one at a time; typed while busy they queue (/say steers instead)
    the dock                         activity row = step n/budget (source) max ceiling + the engine's "now" line; footer = model, ctx %,
                                     tok/s, mode (+network/installs), folder, queue
    printed as events arrive         tool lines, collapsed edit one-liners (/diff <path> expands one), checks, compaction, stages, the plan card,
                                     and at the end CodePanel.summary: changed files, NOT VERIFIED list, a suggested next step
    approvals                        through the dock (y/n/e; DANGEROUS needs a typed  yes ); the engine's own policy decides whether to ask
    nothing reads stdin              intake questions are skipped in this UI (they need a console prompt); say what you want in the goal

`serve()` takes an already opened engine (tests, embedding); `run_code_ui()` opens a real one.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from agent import brand
from agent.ui import code_handlers as H
from agent.ui import style
from agent.ui.activity import ActivityRegistry
from agent.ui.app import NightageApp
from agent.ui.code_runner import ApprovalRequest as RunnerApproval
from agent.ui.code_runner import CodeRunner
from agent.ui.codepanel import CodePanel
from agent.ui.events import EventBus, UiEvent

CONTINUE_LABEL = "continue the saved stage plan"
SNAPSHOT_TTL_S = 0.4
# engine events that never reach the renderer: "now" feeds the activity row, "step" becomes a context event, the rest are replaced
_DROP = frozenset({"now", "pause", "plan", "awaiting_approval", "done", "steer_requeued"})


# ── the engine's console, routed into the UI ───────────────────────────────

class LineSink(io.TextIOBase):
    """File object for the engine's rich Console: complete lines go to `emit(line)` instead of the terminal. The tool-call echo (`  › ...` and
    the indented result line under it) is dropped: the tool events already render those."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit, self._buf, self._lock = emit, "", threading.Lock()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, s: str) -> int:
        with self._lock:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            self._line(ln)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            rest, self._buf = self._buf, ""
        if rest.strip():
            self._line(rest)

    def _line(self, ln: str) -> None:
        text = ln.rstrip()
        if not text.strip() or text.lstrip().startswith(("›", ">")) and text.startswith("  ") or text.startswith("    "):
            return
        with contextlib.suppress(Exception):
            self._emit(text.strip())


def sink_console(emit: Callable[[str], None], width: int = 100) -> Console:
    return Console(file=LineSink(emit), width=width, force_terminal=False, color_system=None, highlight=False, legacy_windows=False)


def late_console(width: int = 100) -> tuple[Console, Callable[[Callable[[str], None]], None]]:
    """A console for an engine that is built BEFORE the UI exists: lines are kept until `bind(emit)` says where they go."""
    state: dict[str, Any] = {"emit": None, "held": []}

    def forward(text: str) -> None:
        fn = state["emit"]
        if fn is None:
            state["held"].append(text)
        else:
            fn(text)

    def bind(fn: Callable[[str], None]) -> None:
        state["emit"] = fn
        held, state["held"] = state["held"], []
        for t in held:
            fn(t)

    return sink_console(forward, width), bind


# ── app subclass: the dock rows come from the engine ───────────────────────

class CodeApp(NightageApp):
    session: "CodeSession"

    def activity_snapshot(self) -> dict[str, Any]:
        snap = super().activity_snapshot()
        if snap.get("busy") and snap.get("state") == "running":
            snap["now"] = self.session.activity_text(snap.get("now") or "")
        return snap

    def footer_state(self):  # noqa: ANN201
        fs = super().footer_state()
        s = self.session.snapshot()
        if s.get("ctx_limit"):
            fs.ctx_used, fs.ctx_limit = int(s["ctx_used"]), int(s["ctx_limit"])
        return fs

    def _turn_ended(self, outcome: str) -> None:
        super()._turn_ended(outcome)
        if self.session.once and outcome == "done" and not len(self.queue) and self.dock is not None:
            self.dock.request_exit()


class CodeSession:
    """Everything that belongs to one `nightage code` process: engine, runner, app, and the glue between them."""

    def __init__(self, engine: Any, config: dict[str, Any], *, goal: str | None = None, plan_first: bool = False, autopilot: bool = False,
                 mode: str | None = None, resumed: bool = False, once: bool = False, plain: bool | None = None, input: Any = None,  # noqa: A002
                 output: Any = None, console: Console | None = None, history_path: Any = None, approval_timeout: float = 600.0,
                 width: Callable[[], int] | None = None, detail: str = "normal", banner: bool = True,
                 bind_console: Callable[[Callable[[str], None]], None] | None = None, **app_kw: Any) -> None:
        self.engine, self.config = engine, config
        self.goal, self.plan_first, self.autopilot_flag, self.mode = goal, plan_first, autopilot, mode
        self.resumed, self.once = resumed, once
        self.exit_code = 0
        self._goal_done = False
        self._turns = 0
        self._emit: Callable[[Any], None] | None = None
        self._handle: Any = None
        self._snap: dict[str, Any] = {}
        self._snap_at = 0.0
        self._snap_lock = threading.Lock()
        self.bus = EventBus()
        self.activities = ActivityRegistry(self.bus)
        self.app = CodeApp(self._turn, console=console, plain=plain, detail=detail, registry=H.build_registry(), control=H.CodeControl(self),
                           config=config, workdir=str(engine.root), input=input, output=output, bus=self.bus, activities=self.activities,
                           approval_timeout=approval_timeout, banner=False, history_path=history_path, width=width,
                           footer_provider=self.footer_facts, status_provider=lambda: H.status_rows(self),
                           context_provider=lambda: H.context_text(self, self.app.console.width), **app_kw)
        self.app.session = self
        self.app.handlers.update({k.lower(): v for k, v in H.make_handlers(self).items()})
        self.app.worker.activities = None          # the code run registers its own "code" activity (stop/pause reach the engine)
        self.app.worker.label = "code"
        self.print_banner_flag = banner
        if bind_console is not None:                 # the engine was built with a late_console(): its lines come here now
            bind_console(self._engine_line)
        else:
            engine.console = sink_console(self._engine_line, self.app.console.width)
            if hasattr(engine.tools, "console"):
                engine.tools.console = engine.console
        self.runner = CodeRunner(engine, registry=self.activities, bus=None, on_event=self._on_event, on_approval=self._on_approval,
                                 on_requeue=self._on_requeue, panel=CodePanel(width=self.app.console.width, plain=self.app.plain),
                                 label="code")

    # ── facts for the dock / status ─────────────────────────────────────
    def policy_text(self) -> str:
        pol = self.engine.pol
        return f"{pol.mode} mode" + (" | network" if pol.network else "") + (" | installs" if pol.installs else "") + f" | commit {pol.commit}"

    def footer_facts(self) -> dict[str, Any]:
        pol = self.engine.pol
        return {"model": str(self.engine.model), "backend": "ollama", "access": pol.mode + ("+net" if pol.network else ""),
                "folder": str(self.engine.root)}

    def snapshot(self) -> dict[str, Any]:
        """The engine snapshot, at most every SNAPSHOT_TTL_S (the dock asks several times a second)."""
        with self._snap_lock:
            if time.monotonic() - self._snap_at > SNAPSHOT_TTL_S or not self._snap:
                self._snap = self.runner.snapshot()
                self._snap_at = time.monotonic()
            return self._snap

    def activity_text(self, fallback: str = "") -> str:
        s = self.snapshot()
        step = int(s.get("step") or 0)
        budget = int(s.get("budget_now") or 0)
        head = f"step {step}/{budget}" if budget else f"step {step}"
        if budget and s.get("budget_source"):
            head += f" ({s['budget_source']})"
        ceil = int(s.get("step_ceiling") or 0)
        if ceil and ceil != budget:
            head += f" max {ceil}"
        now = str(s.get("now") or "") or fallback
        if s.get("state") == "idle":                      # the engine is between two stages / before its first call
            now = fallback or now
        sep = " | " if self.app.plain else " · "
        return head + (sep + now if now else "")

    def print_banner(self, info: dict[str, Any] | None = None) -> None:
        """ONE plain line: product, folder, model, window, policy; then (resumed) one line about the saved session."""
        eng = self.engine
        sep = " | " if self.app.plain else " · "
        head = f"{brand.wordmark()} code"
        bits = [str(eng.model), f"{eng.num_ctx // 1024}k ctx", self.policy_text(), f"session {eng.session_id}" + ("" if info else " (new)"), str(eng.root)]
        body = style.trunc("  " + sep.join(bits), max(10, self.app.console.width - len(head) - 1), self.app.renderer.g)
        t = Text()
        t.append(head, style="" if self.app.plain else f"bold {brand.ACCENT}")
        t.append(body, style="" if self.app.plain else "dim")
        self.app.console.print(t)
        if info:
            from agent.codeagent import sessionstore

            line = (f"resumed: {style.trunc(str(info.get('title') or ''), 50, self.app.renderer.g)} - {sessionstore.age_text(info['age'])} ago, "
                    f"{info['steps']} steps, {info['files']} files touched")
            if info.get("stopped") or info.get("state") == "in progress":
                line += f"; last prompt {'cut off' if info.get('state') == 'in progress' else 'stopped early'} (say 'continue')"
            self.app.renderer.info(line)
            for c in (info.get("changes") or [])[:4]:
                self.app.renderer.notice(f"changed since: {c}")
        self.app.renderer.info("Enter send | /say steers the running agent | Esc stops | /help" if self.app.plain else
                               "Enter send · /say steers the running agent · Esc stops · /help")

    # ── engine -> UI ────────────────────────────────────────────────────
    def _engine_line(self, text: str) -> None:
        if self.app.renderer.detail != "quiet":
            self.app.renderer.info("  " + text)

    def adapt(self, ev: UiEvent) -> UiEvent | None:
        """EventLog record -> what the renderer understands (None = not shown)."""
        k, d = ev.kind, dict(ev.data)
        if k in _DROP:
            return None
        if k == "step":
            prompt, gen, dec = int(d.get("prompt") or 0), int(d.get("gen") or 0), float(d.get("decode_s") or 0)
            limit = int(getattr(self.engine, "num_ctx", 0) or 0)
            data: dict[str, Any] = {"used": prompt, "limit": limit, "percent": round(100 * prompt / limit) if limit else 0}
            if dec > 0.2 and gen > 10:
                data["tokens_per_s"] = round(gen / dec, 1)
            return UiEvent("context", data, ts=ev.ts, source="code")
        if k == "tool_start":
            raw = d.get("args")
            if isinstance(raw, str):
                with contextlib.suppress(ValueError):
                    d["arguments"] = json.loads(raw)
            elif raw is not None:
                d["arguments"] = raw
            d.pop("args", None)
        elif k == "edit" and not self.app.renderer.verbose:
            d["lines"] = len(str(d.get("diff") or "").splitlines())
            d.pop("diff", None)                          # one-liner; the diff stays in the panel (/diff <path>) and in verbose mode
        elif k == "check":
            st = str(d.get("status", ""))
            d["name"] = f"{d.get('id', '')} {d.get('check_kind', '')}".strip()
            d["ok"] = st in ("pass", "passed", "ok")
        elif k == "compact":
            d["compacted"] = True
        return UiEvent(k, d, ts=ev.ts, source="code", turn=ev.turn)

    def _on_event(self, ev: UiEvent) -> None:
        emit = self._emit
        out = self.adapt(ev)
        if out is not None and emit is not None:
            emit(out)

    def _on_requeue(self, text: str) -> None:
        self.app.queue.enqueue(text)
        self.app.renderer.notice(f"your message arrived after the run ended - queued as the next prompt: {style.trunc(text, 60, self.app.renderer.g)}")

    def _on_approval(self, req: RunnerApproval) -> None:
        """Runs on the engine's thread: block until the dock answered, then hand the answer to the runner (which only *asks*; the engine's own
        policy already decided that this needs a human)."""
        handle = self._handle
        if handle is None:
            self.runner.answer_approval(False, approval_id=req.id)
            return
        word = req.description.split(None, 1)[0] if req.description else "action"
        decision = handle({"tool": word if len(word) <= 20 and word.isidentifier() else "action", "command": req.description,
                           "risk": "DANGEROUS" if req.dangerous else "CAUTION", "reason": req.reason, "id": req.id})
        self.runner.answer_approval(bool(decision.approve), typed="yes" if decision.approve else "", approval_id=req.id)

    def _confirm_plan(self, renderable: Any) -> bool:
        self.app.renderer.print(renderable)
        handle = self._handle
        if handle is None:
            return False
        d = handle({"tool": "plan", "command": "Run this plan?  (nothing has been run yet)", "risk": "CAUTION",
                    "reason": "--plan-first: the plan and its acceptance list are shown above; n runs nothing", "id": 0})
        return bool(d.approve)

    # ── turns ───────────────────────────────────────────────────────────
    def start_prompt(self, text: str) -> None:
        """Start a turn from a handler (`/say` while idle, `/stages run`). The prompt is echoed like a queued one."""
        if not self.app.worker.submit(text):
            self.app.queue.enqueue(text)

    def reset_run_view(self) -> None:
        """A different session is in the engine now (/new, /resume): the old run's panel and cached numbers no longer apply."""
        self.runner.panel = CodePanel(width=self.app.console.width, plain=self.app.plain)
        self._snap_at = 0.0
        self._goal_done = True

    def _is_big(self, task: str) -> bool:
        if not task or task.startswith("/"):
            return False
        from agent.codeagent.supervisor import should_autopilot

        with contextlib.suppress(Exception):
            return bool(should_autopilot(task, self.config, self.engine.root))
        return False

    def _turn(self, prompt: str, emit: Callable[[Any], None], stop: threading.Event, approve: Any) -> None:
        self._emit, self._handle = emit, approve
        first = not self._goal_done and self._turns == 0 and self.goal is not None
        self._turns += 1
        task = "" if prompt == CONTINUE_LABEL else prompt
        want_plan = first and self.plan_first and not self.resumed
        autopilot = (first and self.autopilot_flag) or self._is_big(task)
        run_mode = self.mode if first else self.engine.pol.mode
        runner = self.runner
        runner.panel.width, runner.panel.plain = self.app.console.width, self.app.plain
        self._snap_at = 0.0
        t = runner.start(task, plan_first=want_plan, confirm=self._confirm_plan if want_plan else None, staged=True,
                         autopilot=autopilot, mode=run_mode)
        act = runner.activity
        if act is not None:                               # /tasks stop|pause|resume go through the app, so the queue and the dock state follow
            act.stop, act.pause, act.resume = self.app.stop_turn, self._pause, self._resume
        sent_stop = False
        while t.is_alive():
            if stop.is_set() and not sent_stop:
                runner.stop()
                sent_stop = True
            t.join(0.05)
        result = runner.result or {}
        self._snap_at = 0.0
        self.snapshot()
        if first:
            self._goal_done = True
            self.exit_code = int(result.get("exit_code") or 0)
        if runner.declined_plan:
            self.app.renderer.notice("plan declined - nothing was run")
        else:
            self.app.renderer.print(runner.panel.summary(width=self.app.console.width))
        emit(UiEvent("done", {"exit_code": result.get("exit_code"), "stopped": result.get("stopped", "")}, source="code"))
        self._emit = self._handle = None

    def _pause(self) -> None:
        if self.app.worker.pause():
            self.runner.pause()

    def _resume(self) -> None:
        if self.app.worker.resume():
            self.runner.resume()

    # ── run ─────────────────────────────────────────────────────────────
    def run(self) -> int:
        if self.print_banner_flag:
            self.print_banner(self._resume_info)
        if self.goal is not None:
            self.app.worker.submit(self.goal or CONTINUE_LABEL)
        try:
            rc = self.app.run()
        finally:
            self.runner.detach()
        if self.goal is not None and rc == 0:
            return self.exit_code
        return rc

    _resume_info: dict[str, Any] | None = None


def serve(engine: Any, config: dict[str, Any], *, goal: str | None = None, info: dict[str, Any] | None = None, **kw: Any) -> int:
    """Run the UI on an opened engine (already resumed / started). `info` is the resume info for the banner."""
    sess = CodeSession(engine, config, goal=goal, resumed=info is not None, **kw)
    sess._resume_info = info
    return sess.run()


# ── real startup ───────────────────────────────────────────────────────────

def run_code_ui(folder: str | None = None, goal: str | None = None, *, resume: str | None = None, stages: bool = False, autopilot: bool = False,
                plan_first: bool = False, mode: str | None = None, steps: int | None = None, effort: str | None = None,
                diagnose: bool = False, config: dict[str, Any] | None = None, plain: bool | None = None, input: Any = None,  # noqa: A002
                output: Any = None, new: bool = False, keep_model: bool = False, test: str | None = None,
                files: list[str] | None = None, once: bool = False, console: Console | None = None, **app_kw: Any) -> int:
    """`nightage code`. Returns a process exit code: with a goal, the goal run's (0 done, 1 not finished/declined, 2 needs a human, 3 partial,
    130 stopped); otherwise 0. `once` ends the UI when the goal run (and queued follow-ups) finished. `resume`: None, "latest" or a session
    id prefix (as session.run_session)."""
    from agent import foreground
    from agent.codeagent import policy as policy_mod
    from agent.codeagent import session as S
    from agent.codeagent.supervisor import should_autopilot
    from agent.config import load_config

    err = Console(stderr=True)
    stages = stages or plan_first
    cfg = S.with_stages(S.with_keep_model(config if config is not None else load_config(), keep_model), stages)
    if effort:
        cfg = {**cfg, "build": {**(cfg.get("build") or {}), "effort": effort}}
    if diagnose:
        cfg = {**cfg, "coding": {**(cfg.get("coding") or {}), "diagnose_first": True}}
    if goal is None and autopilot and resume:
        goal = ""                                              # `--autopilot --resume`: continue the saved stage plan
    autopilot = goal is not None and (autopilot or should_autopilot(goal, cfg))
    chosen, root = S.resolve_target(err, folder, resume)
    if root is None:
        return 1
    autopilot = autopilot or (goal is not None and should_autopilot(goal, cfg, root))
    run_cfg = S.with_autopilot(cfg) if autopilot else cfg
    ok, owner = S.acquire_lock(root)
    if not ok:
        err.print(f"[red]Another coding session is already working in {root} (pid {owner}).[/red] Two agents in one folder overwrite "
                  "each other - finish or stop that one first (`kill " + owner + "`).")
        return 1
    if (input is None and goal is None and not mode and str(root) not in policy_mod.all_repos()
            and policy_mod.for_repo(root).mode == "ask"):
        S.ask_access(err, root)                                # before the dock exists: the one place a console prompt is fine
    with foreground.session(cfg):
        holder, bind = late_console()                          # the engine's console: its lines go into the UI once it exists
        engine, llm, store = CodeRunner.open_engine(run_cfg, root, holder, test=test)
        try:
            S.free_gpu(err, llm, engine.model)
            info = S.prepare_engine(err, engine, root, chosen, resume, new, mode=mode, autopilot=autopilot, steps=steps)
            if goal is not None and files and goal:
                goal = f"{goal}\n\nFocus on these files: {', '.join(files)}."
            return serve(engine, cfg, goal=goal, info=info, plan_first=plan_first, autopilot=autopilot, mode=mode, once=once, plain=plain,
                         input=input, output=output, console=console, bind_console=bind, **app_kw)
        finally:
            S.save_session(engine)
            S.release_gpu_on_exit(err, llm, cfg)
            with contextlib.suppress(Exception):
                store.close()
