"""TurnWorker: runs one turn function at a time in a background thread.

    turn_fn(prompt, emit, stop, approve)

  emit(event)   UiEvent or chat-core dict ({"type": ...}); published on the EventBus
  stop          threading.Event; the turn must return soon after it is set
  approve       the worker's WorkerHandle: call it with an approval request to block for a decision;
                `approve.wait_if_paused()` is the cooperative pause point (returns False once stopped)

States: idle | running | paused | waiting_approval. Finished turns feed the PromptQueue: the next queued
item starts automatically unless the queue is paused. An interrupted (stopped) or failed turn pauses the queue
and says so (visible notice event) instead of silently marching on."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.ui.activity import ActivityRegistry
from agent.ui.events import EventBus, UiEvent
from agent.ui.queue import PromptQueue, QueueItem
from agent.ui.render import to_event

IDLE, RUNNING, PAUSED, WAITING = "idle", "running", "paused", "waiting_approval"
BUSY_STATES = frozenset({RUNNING, PAUSED, WAITING})

APPROVE_KEYS = {"y": "approve", "yes": "approve", "n": "deny", "no": "deny", "a": "always", "always": "always"}


@dataclass
class ApprovalRequest:
    tool: str = "action"
    command: str = ""
    risk: str = "CAUTION"
    reason: str = ""
    diff: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    id: int = 0

    @classmethod
    def coerce(cls, req: Any) -> "ApprovalRequest":
        if isinstance(req, ApprovalRequest):
            return req
        if isinstance(req, dict):
            d = dict(req)
        elif hasattr(req, "arguments") or hasattr(req, "description"):    # chat_adapter.ApprovalRequest and friends
            d = {k: getattr(req, k) for k in ("name", "risk", "description", "arguments", "diff", "reason") if hasattr(req, k)}
            if getattr(req, "from_web", False):
                d["reason"] = (str(d.get("reason") or "") + " The text of this call was copied from a web page.").strip()
        else:
            d = {"command": str(req)}
        risk = d.get("risk_level") or d.get("risk") or "CAUTION"
        return cls(tool=str(d.get("tool_name") or d.get("tool") or d.get("name") or "action"),
                   command=str(d.get("command") or d.get("description") or ""), risk=str(getattr(risk, "name", risk)).upper(),
                   reason=str(d.get("reason") or ""), diff=str(d.get("diff") or ""),
                   arguments=dict(d.get("arguments") or {}), id=int(d.get("approval_id") or d.get("id") or 0))

    def as_event_data(self) -> dict[str, Any]:
        return {"approval_id": self.id, "tool_name": self.tool, "command": self.command, "risk_level": self.risk,
                "reason": self.reason, "diff": self.diff, "arguments": self.arguments}


@dataclass
class ApprovalDecision:
    approve: bool = False
    always: bool = False
    reason: str = ""

    def __bool__(self) -> bool:
        return self.approve


def _plugin_event(event: str, **data: Any) -> None:
    """Tell plugin hooks (agent/plugin_api.py) about a turn boundary. Never raises."""
    try:
        from agent import plugin_api

        plugin_api.emit(event, **data)
    except Exception:  # noqa: BLE001
        pass


class WorkerHandle:
    """What the turn function receives as `approve`: callable (approval request -> decision) plus pause helpers."""

    def __init__(self, worker: "TurnWorker", stop: threading.Event) -> None:
        self._w, self.stop = worker, stop

    def __call__(self, req: Any, timeout: float | None = None) -> ApprovalDecision:
        return self._w.request_approval(req, timeout)

    @property
    def paused(self) -> bool:
        return self._w.state == PAUSED

    def as_chat_approve(self) -> Callable[[Any], Any]:
        """approve_fn for chat_adapter.run_chat_turn: returns the exact string "yes" (what DANGEROUS requires)
        when allowed, False otherwise."""
        def fn(req: Any) -> Any:
            return "yes" if self(req) else False
        return fn

    def wait_if_paused(self, poll: float = 0.05) -> bool:
        return self._w.checkpoint(poll)


class TurnWorker:
    def __init__(self, turn_fn: Callable[..., Any], *, bus: EventBus | None = None, queue: PromptQueue | None = None,
                 activities: ActivityRegistry | None = None, dispatch: Callable[[QueueItem], Any] | None = None,
                 approval_timeout: float = 600.0, label: str = "chat", on_turn_start: Callable[[str], Any] | None = None,
                 on_turn_end: Callable[[str], Any] | None = None) -> None:
        self.turn_fn = turn_fn
        self.bus = bus or EventBus()
        self.queue = queue if queue is not None else PromptQueue()
        self.activities = activities
        self.dispatch = dispatch              # runs a queued "command" item; prompts start turns by themselves
        self.approval_timeout = approval_timeout
        self.label = label
        self.on_turn_start, self.on_turn_end = on_turn_start, on_turn_end
        self._lock = threading.RLock()
        self._state = IDLE
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._approval: ApprovalRequest | None = None
        self._approval_evt = threading.Event()
        self._decision = ApprovalDecision()
        self._turn = 0
        self._prompt = ""
        self._started = 0.0
        self._act_id: str | None = None
        self._stats: dict[str, Any] = {}
        self.last_outcome = ""              # done | interrupted | error
        self._idle = threading.Event()
        self._idle.set()

    # ---- state
    @property
    def state(self) -> str:
        return self._state

    @property
    def busy(self) -> bool:
        return self._state in BUSY_STATES

    @property
    def pending_approval(self) -> ApprovalRequest | None:
        return self._approval

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    # ---- emit
    def emit(self, item: Any) -> None:
        ev = to_event(item)
        if ev.turn is None:
            ev.turn = self._turn
        self._observe(ev)
        self.bus.emit(ev)

    def notice(self, text: str, source: str = "ui") -> None:
        self.bus.emit(UiEvent("log", {"text": text, "notice": True}, source=source, turn=self._turn))

    def _observe(self, ev: UiEvent) -> None:
        d, s = ev.data, self._stats
        k = ev.kind
        if k == "context":
            s["ctx_used"], s["ctx_limit"] = int(d.get("used") or 0), int(d.get("limit") or 0)
            if d.get("tokens_per_s"):
                s["tok_s"] = float(d["tokens_per_s"])
        elif k == "token":
            s.setdefault("first_out", time.time())
            s["streamed"] = s.get("streamed", 0) + max(1, len(str(d.get("text", ""))) // 4)
            s["now"] = "writing"
        elif k == "thinking":
            s.setdefault("first_out", time.time())
            if d.get("tokens"):
                s["think_tokens"] = int(d["tokens"])
            s["now"] = "thinking"
        elif k == "status":
            if str(d.get("label") or "") == "waiting for model":
                s["wait_since"] = float(d.get("since") or time.time())
                s["prompt_tokens"] = int(d.get("prompt_tokens") or 0)
                s.pop("first_out", None)
            elif d.get("label") == "thinking":
                s.setdefault("first_out", time.time())
                if d.get("tokens"):
                    s["think_tokens"] = int(d["tokens"])
            else:
                s.pop("wait_since", None)
            s["now"] = " ".join(x for x in (str(d.get("label") or d.get("state") or ""), str(d.get("detail") or "")) if x)
        elif k == "now":
            s["now"] = str(d.get("text", ""))
        elif k in ("tool_call", "tool_start"):
            s["tools"] = s.get("tools", 0) + 1
            s["now"] = _now_for_tool(str(d.get("name", "")), d.get("arguments"))
        elif k in ("tool_result", "tool"):
            s["now"] = "waiting for the model"
        elif k in ("stage_start",):
            s["now"] = "stage " + str(d.get("name") or d.get("stage") or "")
        if self._act_id and self.activities and "now" in s:
            self.activities.update(self._act_id, now=s["now"])

    # ---- start / run
    def submit(self, prompt: str) -> bool:
        """Start a turn if idle. Returns False when busy (the caller queues instead)."""
        with self._lock:
            if self._state != IDLE:
                return False
            self._state = RUNNING
            self._idle.clear()
            self._turn += 1
            self._stop = threading.Event()
            self._resume.set()
            self._prompt = prompt
            self._started = time.time()
            self._stats = {"now": "starting"}
            self._approval = None
            stop = self._stop
            if self.activities is not None:
                a = self.activities.register("chat", prompt[:60] or self.label, stop=self.stop, pause=self.pause,
                                             resume=self.resume, now="starting")
                self._act_id = a.id
            t = threading.Thread(target=self._run, args=(prompt, stop), name=f"turn-{self._turn}", daemon=True)
            self._thread = t
        if self.on_turn_start:
            try:
                self.on_turn_start(prompt)
            except Exception:  # noqa: BLE001
                pass
        _plugin_event("turn_start", prompt=prompt)
        t.start()
        return True

    def _run(self, prompt: str, stop: threading.Event) -> None:
        outcome = "done"
        handle = WorkerHandle(self, stop)
        try:
            self.turn_fn(prompt, self.emit, stop, handle)
        except BaseException as exc:  # noqa: BLE001 - a crashing turn must never take the UI down
            if stop.is_set() or type(exc).__name__ == "GenerationAborted":
                outcome = "interrupted"
            else:
                outcome = "error"
                self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        else:
            if stop.is_set():
                outcome = "interrupted"
        self._finish(outcome)

    def _finish(self, outcome: str) -> None:
        with self._lock:
            self.last_outcome = outcome
            act, self._act_id = self._act_id, None
            self._approval = None
            self._approval_evt.set()
            self._state = IDLE
            self._resume.set()
        if act and self.activities:
            self.activities.finish(act, {"done": "done", "interrupted": "stopped", "error": "failed"}[outcome])
        if self.on_turn_end:
            try:
                self.on_turn_end(outcome)
            except Exception:  # noqa: BLE001
                pass
        _plugin_event("turn_end", outcome=outcome)
        if outcome != "done":
            n = len(self.queue)
            self.queue.pause()
            why = "interrupted" if outcome == "interrupted" else "failed"
            self.notice(f"turn {why}" + (f" - queue paused ({n} waiting), /resume to continue or /queue clear" if n else ""))
        self._idle.set()
        if outcome == "done":
            self.advance()

    def advance(self) -> None:
        """Run queued items until one starts a turn (or the queue is empty/paused). Called after each turn."""
        while True:
            with self._lock:
                if self._state != IDLE:
                    return
            item = self.queue.pop_next()
            if item is None:
                return
            if item.kind == "command":
                if self.dispatch is not None:
                    try:
                        self.dispatch(item)
                    except Exception as exc:  # noqa: BLE001
                        self.notice(f"queued command failed: {exc}")
                continue
            if self.submit(item.text):
                return

    # ---- control
    def stop(self) -> bool:
        with self._lock:
            if self._state == IDLE:
                return False
            self._stop.set()
            self._resume.set()
            self._approval_evt.set()
        self.notice("stopping...")
        return True

    def pause(self) -> bool:
        with self._lock:
            if self._state != RUNNING:
                return False
            self._state = PAUSED
            self._resume.clear()
        if self._act_id and self.activities:
            self.activities.update(self._act_id, state="paused")
        self.notice("paused - /resume to continue")
        return True

    def resume(self) -> bool:
        with self._lock:
            if self._state != PAUSED:
                return False
            self._state = RUNNING
            self._resume.set()
        if self._act_id and self.activities:
            self.activities.update(self._act_id, state="running")
        self.notice("resumed")
        return True

    def checkpoint(self, poll: float = 0.05) -> bool:
        """Cooperative pause point for the turn function. Blocks while paused; False once stop is set."""
        while not self._resume.wait(poll):
            if self._stop.is_set():
                return False
        return not self._stop.is_set()

    # ---- approvals
    def request_approval(self, req: Any, timeout: float | None = None) -> ApprovalDecision:
        r = ApprovalRequest.coerce(req)
        with self._lock:
            if self._stop.is_set():
                return ApprovalDecision(False, reason="stopped")
            prev = self._state
            self._approval, self._decision = r, ApprovalDecision(False, reason="no answer")
            self._approval_evt.clear()
            self._state = WAITING
        if self._act_id and self.activities:
            self.activities.update(self._act_id, state="waiting", now=f"waiting for approval: {r.tool}")
        self._stats["now"] = f"waiting for approval: {r.tool}"
        self.bus.emit(UiEvent("awaiting_approval", r.as_event_data(), turn=self._turn))
        limit = self.approval_timeout if timeout is None else timeout
        deadline = time.monotonic() + limit
        answered = False
        while time.monotonic() < deadline:
            if self._approval_evt.wait(0.05):
                answered = True
                break
        with self._lock:
            decision = self._decision if answered and not self._stop.is_set() else ApprovalDecision(
                False, reason="stopped" if self._stop.is_set() else "timed out")
            self._approval = None
            if self._state == WAITING:
                self._state = PAUSED if not self._resume.is_set() else RUNNING
        if self._act_id and self.activities:
            self.activities.update(self._act_id, state="running" if self._state == RUNNING else "paused")
        if not answered and not self._stop.is_set():
            self.notice(f"approval for {r.tool} timed out - denied")
        return decision

    def answer_approval(self, decision: ApprovalDecision | str | bool) -> bool:
        """Answer the pending request. Strings: y/yes, n/no, a/always. For DANGEROUS requests only the
        full word `yes` approves and `always` is refused (a typo or a stray `y` must never run it)."""
        with self._lock:
            req = self._approval
            if req is None:
                return False
            if isinstance(decision, str):
                word = decision.strip().lower()
                kind = APPROVE_KEYS.get(word)
                if kind is None:
                    return False
                if req.risk == "DANGEROUS" and word != "yes" and kind != "deny":
                    return False
                decision = ApprovalDecision(kind in ("approve", "always"), kind == "always")
            elif isinstance(decision, bool):
                decision = ApprovalDecision(decision)
            self._decision = decision
            self._approval_evt.set()
        return True

    # ---- footer
    def snapshot(self) -> dict[str, Any]:
        s = dict(self._stats)
        started = self._started
        busy = self.busy
        now_t = time.time()
        waiting = 0.0                    # seconds the model has had the prompt without producing anything
        if busy and s.get("wait_since") and "first_out" not in s:
            waiting = max(0.0, now_t - s["wait_since"])
        tok_s = s.get("tok_s")
        out_tokens = s.get("streamed", 0) + s.get("think_tokens", 0)
        if busy and not tok_s and s.get("first_out") and out_tokens >= 3 and now_t - s["first_out"] >= 2.0:
            tok_s = out_tokens / (now_t - s["first_out"])
        return {
            "state": self._state, "busy": busy, "turn": self._turn, "prompt": self._prompt if busy else "",
            "elapsed": (time.time() - started) if busy and started else 0.0, "now": s.get("now", "") if busy else "",
            "tokens": s.get("streamed", 0), "think_tokens": s.get("think_tokens", 0), "tools": s.get("tools", 0),
            "tok_s": tok_s, "waiting_s": waiting, "prompt_tokens": s.get("prompt_tokens", 0), "ctx_used": s.get("ctx_used"), "ctx_limit": s.get("ctx_limit"),
            "queue": len(self.queue), "queue_paused": self.queue.paused,
            "approval": self._approval.tool if self._approval else None, "last_outcome": self.last_outcome,
        }


def _now_for_tool(name: str, args: Any) -> str:
    arg = ""
    if isinstance(args, dict):
        for k in ("path", "file", "command", "query", "url", "pattern"):
            if args.get(k):
                arg = str(args[k])
                break
    verb = {"read_file": "reading", "read": "reading", "list_dir": "listing", "grep": "searching", "search": "searching",
            "web_search": "searching the web", "run_command": "running", "shell": "running", "edit_file": "editing",
            "write_file": "writing", "edit": "editing"}.get(name, name or "working")
    return f"{verb} {arg}".strip()
