"""Runs a coding goal (or a follow-up prompt) on a CodeEngine in a worker thread for the terminal UI.

The engine is injected (the integration opens it with `CodeRunner.open_engine`, tests pass a fake-LLM engine), so nothing here talks to a real
model by itself. What the runner adds on top of the engine's own live hooks:

  * EventLog records -> `UiEvent`s on an optional EventBus (plus its own ring for CodePanel and `/code` views),
  * an Activity in the ActivityRegistry whose stop/pause/resume call engine.request_*,
  * `snapshot()`, `say()`, `compact()`, `pause()/resume()/stop()`,
  * an approval bridge: the engine's `approve_fn` blocks the worker thread until the dock calls `answer_approval()` (or timeout = no).
    The engine's own policy decision is untouched: this is only reached when the engine already decided to ask.
    A DANGEROUS request is approved only by `answer_approval(True, typed="yes")`,
  * plan-first (stage plan shown through CodePanel, confirm before anything runs; same semantics as session.plan_first_gate),
  * steer messages the engine never got to (the run ended first) are handed to `on_requeue` / `requeued` as the next prompt.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.ui.activity import Activity, ActivityRegistry
from agent.ui.codepanel import CodePanel
from agent.ui.events import EventBus, UiEvent, from_record

APPROVAL_TIMEOUT_S = 600.0


@dataclass
class ApprovalRequest:
    id: int
    description: str
    reason: str
    dangerous: bool
    asked: float = field(default_factory=time.time)
    allowed: bool | None = None            # None = still open


class CodeRunner:
    def __init__(self, engine: Any, *, registry: ActivityRegistry | None = None, bus: EventBus | None = None,
                 on_event: Callable[[UiEvent], None] | None = None, on_approval: Callable[[ApprovalRequest], None] | None = None,
                 on_requeue: Callable[[str], None] | None = None, on_done: Callable[[dict[str, Any]], None] | None = None,
                 panel: CodePanel | None = None, label: str = "", autosave: bool = True,
                 approval_timeout: float = APPROVAL_TIMEOUT_S) -> None:
        self.engine = engine
        self.registry, self.bus = registry, bus
        self.on_event, self.on_approval, self.on_requeue, self.on_done = on_event, on_approval, on_requeue, on_done
        self.panel = panel or CodePanel()
        self.label = label
        self.autosave = autosave
        self.approval_timeout = approval_timeout
        self.events: deque[UiEvent] = deque(maxlen=500)
        self.activity: Activity | None = None
        self.result: dict[str, Any] | None = None
        self.requeued: list[str] = []
        self.thread: threading.Thread | None = None
        self.declined_plan = False
        self._lock = threading.RLock()
        self._pending: ApprovalRequest | None = None
        self._answered = threading.Event()
        self._seq = 0
        self._stopping = False
        self._plan_confirm = threading.Event()
        self._plan_ok = False
        self._unsub: Callable[[], None] | None = None
        self._hook: Callable[[], None] | None = None
        self._attach()

    # ── wiring ──────────────────────────────────────────────────────────
    def _attach(self) -> None:
        eng = self.engine
        self._unsub = eng.events.subscribe(self._on_record)
        if self.autosave:
            self._hook = self._autosave
            eng.add_step_hook(self._hook)
        eng.approve_fn = self._approve          # the engine's own policy still decides whether to ask at all
        eng._explicit_approve = True            # nobody on stdin: the bridge is the one who answers

    def detach(self) -> None:
        if self._unsub:
            self._unsub()
            self._unsub = None
        if self._hook is not None:
            with contextlib.suppress(Exception):
                self.engine.remove_step_hook(self._hook)
            self._hook = None

    def _autosave(self) -> None:
        from agent.codeagent import session

        if getattr(self.engine, "session_id", ""):
            session.save_session(self.engine)

    def _on_record(self, rec: dict[str, Any]) -> None:
        ev = from_record(rec, source="code")
        self.events.append(ev)
        self.panel.feed(ev)
        kind = rec.get("kind")
        if self.registry and self.activity:
            if kind == "now":
                self.registry.update(self.activity.id, now=str(rec.get("text", "")))
            elif kind == "pause":
                self.registry.update(self.activity.id, state="paused" if rec.get("paused") else "running")
        if self.bus is not None:
            self.bus.emit(ev)
        if self.on_event is not None:
            with contextlib.suppress(Exception):
                self.on_event(ev)

    def _emit(self, kind: str, **data: Any) -> None:
        ev = UiEvent(kind, data, source="code")
        self.events.append(ev)
        if self.bus is not None:
            self.bus.emit(ev)
        if self.on_event is not None:
            with contextlib.suppress(Exception):
                self.on_event(ev)

    # ── state / controls ────────────────────────────────────────────────
    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        from agent.codeagent import livestate

        snap = livestate.snapshot(self.engine)
        if self._pending is not None:
            snap["state"], snap["waiting"] = "waiting", "approval"
        self.panel.update(snap)
        return snap

    def say(self, text: str) -> None:
        """Steer the running turn: the engine adds the text before its next model call."""
        self.engine.queue_user_message(text)

    def compact(self) -> None:
        self.engine.request_compact()

    def pause(self) -> None:
        self.engine.request_pause()
        if self.registry and self.activity:
            self.registry.update(self.activity.id, state="paused")

    def resume(self) -> None:
        self.engine.request_resume()
        if self.registry and self.activity:
            self.registry.update(self.activity.id, state="running")

    def stop(self) -> None:
        """End at the next step boundary (also wakes a paused run and denies an open approval). Work so far is kept."""
        self._stopping = True
        self.engine.request_stop()
        self._plan_ok = False
        self._plan_confirm.set()
        self._answered.set()

    # ── approvals ───────────────────────────────────────────────────────
    def pending_approval(self) -> ApprovalRequest | None:
        return self._pending

    def _approve(self, description: str, reason: str) -> bool:
        with self._lock:
            self._seq += 1
            req = ApprovalRequest(self._seq, description, reason, dangerous="dangerous" in (reason or "").lower())
            self._pending = req
            self._answered.clear()
        self._emit("awaiting_approval", id=req.id, description=description, reason=reason, dangerous=req.dangerous)
        if self.registry and self.activity:
            self.registry.update(self.activity.id, state="waiting")
        with contextlib.suppress(Exception):
            if self.on_approval is not None:
                self.on_approval(req)
        try:
            self._answered.wait(timeout=self.approval_timeout)
            ok = bool(req.allowed) and not self._stopping
        finally:
            with self._lock:
                self._pending = None
            if self.registry and self.activity and not self._stopping:
                self.registry.update(self.activity.id, state="running")
        if req.allowed is None:
            req.allowed = False
        return ok

    def answer_approval(self, allow: bool, typed: str = "", approval_id: int | None = None) -> bool:
        """Decide the open request. DANGEROUS needs typed == 'yes' (anything else denies). Returns True when a request was answered."""
        with self._lock:
            req = self._pending
            if req is None or (approval_id is not None and approval_id != req.id):
                return False
            ok = bool(allow) and (not req.dangerous or typed.strip().lower() == "yes")
            req.allowed = ok
        self._answered.set()
        return True

    # ── plan-first ──────────────────────────────────────────────────────
    def confirm_plan(self, proceed: bool) -> None:
        self._plan_ok = bool(proceed)
        self._plan_confirm.set()

    def _plan_first(self, task: str, cfg: dict[str, Any], confirm: Callable[[Any], bool] | None) -> bool:
        from agent.codeagent import planview
        from agent.codeagent import stages as stages_mod

        eng = self.engine
        plan = stages_mod.split(task, llm=eng.llm, cfg=cfg, model=eng.model, cache=eng.split_cache, root=eng.root)
        view = planview.from_parts(plan, eng.plan.plan, len(eng.dead_ends.items), cfg)
        plan_dict = planview.to_dict(view)
        self.panel.update({**self.snapshot(), "plan": plan_dict})
        self._emit("plan", plan=plan_dict, awaiting_confirm=True)
        if confirm is not None:
            proceed = bool(confirm(self.panel.plan()))
        else:
            self._plan_confirm.clear()
            self._plan_confirm.wait(timeout=self.approval_timeout)
            proceed = self._plan_ok and not self._stopping
        if not proceed:
            self.declined_plan = True
            return False
        eng.stage_plan = plan
        eng.ctx.user_requests.append(task)
        return True

    # ── running ─────────────────────────────────────────────────────────
    def start(self, goal: str, *, plan_first: bool = False, confirm: Callable[[Any], bool] | None = None, staged: bool = True,
              autopilot: bool = False, mode: str | None = None) -> threading.Thread:
        """Run `goal` (or a follow-up prompt) in a worker thread; returns it. Only one run at a time. `confirm(plan_renderable)->bool` is
        called on the worker thread for plan_first (default: wait for confirm_plan())."""
        if self.busy:
            raise RuntimeError("a code run is already in progress")
        self._stopping = False
        self.result = None
        self.declined_plan = False
        self.panel.result = None
        self.activity = None
        if self.registry is not None:
            self.activity = self.registry.register(
                "code", self.label or (goal.strip().splitlines() or ["code run"])[0][:60], now="starting",
                stop=self.stop, pause=self.pause, resume=self.resume, detail={"root": str(getattr(self.engine, "root", ""))})
        self.thread = threading.Thread(target=self._main, args=(goal, plan_first, confirm, staged, autopilot, mode), daemon=True, name="code-runner")
        self.thread.start()
        return self.thread

    def join(self, timeout: float | None = None) -> bool:
        if self.thread is not None:
            self.thread.join(timeout)
        return not self.busy

    def _execute(self, task: str, staged: bool, autopilot: bool, mode: str | None) -> dict[str, Any]:
        from agent.codeagent import session as S

        eng = self.engine
        if autopilot:
            code = S.run_autopilot(eng, task, mode=mode)
            return {"text": None if code else "All stages done.", "stopped": "" if code == 0 else "partial" if code == 3 else "blocked" if code == 2 else
                    "interrupted" if code == 130 else "error", "verified": None, "not_verified": [], "files": [], "exit_code": code}
        if staged:
            report = S.run_task_staged(eng, task)
            if report is not None:
                ok = report.verdict == "done"
                return {"text": f"Staged run: {report.verdict}.", "stopped": "" if ok else "partial", "verified": None, "not_verified": [],
                        "files": [], "exit_code": 0 if ok else 1}
        res = eng.turn(task)
        return {"text": res.text, "stopped": res.stopped, "verified": res.verified, "not_verified": list(res.not_verified),
                "files": list(res.files), "reason": res.reason, "steps": res.steps,
                "exit_code": 0 if res.stopped == "" and res.text is not None else 1}

    def _main(self, goal: str, plan_first: bool, confirm: Callable[[Any], bool] | None, staged: bool, autopilot: bool, mode: str | None) -> None:
        state = "done"
        result: dict[str, Any] = {}
        try:
            if plan_first and not self._plan_first(goal, self.engine.config, confirm):
                state = "stopped"
                result = {"text": None, "stopped": "declined" if not self._stopping else "interrupted", "verified": None, "not_verified": [],
                          "files": [], "exit_code": 1}
            else:
                result = self._execute(goal, staged or plan_first, autopilot, mode)
                if result.get("stopped") == "interrupted":
                    state = "stopped"
                elif result.get("exit_code"):
                    state = "failed" if result.get("stopped") == "error" else "done"
        except KeyboardInterrupt:
            state, result = "stopped", {"text": None, "stopped": "interrupted", "verified": None, "not_verified": [], "files": [], "exit_code": 130}
        except Exception as exc:  # noqa: BLE001 - a bug in the run ends this prompt, not the UI
            state = "failed"
            result = {"text": None, "stopped": "error", "reason": f"{type(exc).__name__}: {exc}", "verified": None, "not_verified": [],
                      "files": [], "exit_code": 1}
        finally:
            with contextlib.suppress(Exception):
                if self.autosave:
                    self._autosave()
            self._finish(result, state)

    def _finish(self, result: dict[str, Any], state: str) -> None:
        # steer messages the engine never drained (the run ended before the next step boundary) become the next prompt
        left: list[str] = []
        with contextlib.suppress(Exception):
            left = self.engine.live.drain_messages()
        if left:
            text = "\n".join(left)
            self.requeued.append(text)
            result["requeued"] = text
            self._emit("steer_requeued", text=text, count=len(left))
        with contextlib.suppress(Exception):
            result["files_detail"] = self.snapshot().get("files", [])
        self.result = result
        self.panel.finish(result)
        if self.registry and self.activity:
            self.registry.finish(self.activity.id, state)
        self._emit("done", **{k: v for k, v in result.items() if k != "files_detail"})
        if left and self.on_requeue is not None:
            with contextlib.suppress(Exception):
                self.on_requeue("\n".join(left))
        if self.on_done is not None:
            with contextlib.suppress(Exception):
                self.on_done(result)

    # ── convenience for the integration ─────────────────────────────────
    @staticmethod
    def open_engine(config: dict[str, Any], root: Any, console: Any, *, test: str | None = None) -> tuple[Any, Any, Any]:
        """Build a real engine (session.open_engine, non-interactive so nothing reads stdin) and return (engine, llm, store).
        Needs a model server; not exercised by the unit tests. The caller owns acquire_lock / foreground.session / free_gpu and closes `store`."""
        from pathlib import Path

        from agent.codeagent import session as S

        return S.open_engine(console, config, Path(root), False, test)
