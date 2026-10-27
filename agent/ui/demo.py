"""`nightage --demo`: the whole UI driven by a scripted fake agent (no Ollama, no model).

FakeAgent is a turn_fn: it emits a realistic event stream (streamed text, thinking, tool calls with risk, an
approval request, an edit with a diff, a plan with stages, context updates, sources) and honours stop, pause
and approval exactly like a real engine would. `delay` scales every pause (0 = instant, used by tests)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

def _nap(seconds: float) -> None:
    threading.Event().wait(seconds)     # not time.sleep: test suites patch that globally


DIFF = """--- a/rps/game.py
+++ b/rps/game.py
@@ -1,6 +1,12 @@
 import random

-def play():
-    print("rock paper scissors")
+MOVES = ("rock", "paper", "scissors")
+
+
+def winner(a: str, b: str) -> str:
+    if a == b:
+        return "tie"
+    return a if (MOVES.index(a) - MOVES.index(b)) % 3 == 1 else b
+
+
+def play() -> str:
+    return winner(random.choice(MOVES), random.choice(MOVES))
"""

ANSWER = (
    "I looked at the project and made the change.\n\n"
    "## What I did\n\n"
    "- read `rps/game.py` and found the old stub\n"
    "- added a `winner()` helper with the classic rules\n"
    "- ran the tests: all green\n\n"
    "```python\ndef winner(a, b):\n    ...\n```\n\n"
    "Tell me if you want a best-of-three mode next."
)

SHORT = "Hello! This is the demo agent. Ask me to *build something* to see tools, approvals and diffs."


class FakeAgent:
    def __init__(self, delay: float = 1.0, approve_commands: bool = True) -> None:
        self.delay = delay
        self.approve_commands = approve_commands
        self.calls: list[str] = []

    # ---- helpers
    def _sleep(self, seconds: float, stop: threading.Event, approve: Any) -> bool:
        """Sleep in small steps; False when the turn must end (stopped)."""
        end = time.monotonic() + seconds * self.delay
        waiter = getattr(approve, "wait_if_paused", None)
        while True:
            if stop.is_set():
                return False
            if waiter is not None and not waiter():
                return False
            left = end - time.monotonic()
            if left <= 0:
                return True
            _nap(min(0.03, left))

    def _stream(self, text: str, emit: Callable, stop: threading.Event, approve: Any, per_word: float = 0.03) -> bool:
        for word in text.split(" "):
            if not self._sleep(per_word, stop, approve):
                return False
            emit({"type": "token", "text": word + " "})
        return True

    # ---- the turn
    def __call__(self, prompt: str, emit: Callable, stop: threading.Event, approve: Any) -> None:
        self.calls.append(prompt)
        S = lambda s: self._sleep(s, stop, approve)  # noqa: E731
        emit({"type": "status", "state": "thinking", "label": "waiting for the model", "detail": ""})
        if not S(0.4):
            return
        if len(prompt.split()) <= 2 and not prompt.startswith("!"):
            emit({"type": "context", "used": 1200, "limit": 32768, "percent": 4, "tokens_per_s": 41.0, "compacted": False})
            if self._stream(SHORT, emit, stop, approve):
                emit({"type": "done", "reply": SHORT})
            return

        emit({"type": "thinking", "tokens": 120, "text": "The user wants a change in the project. I should read the file first, "})
        if not S(0.3):
            return
        emit({"type": "thinking", "tokens": 410, "text": "then edit it and run the tests before answering."})
        if not S(0.2):
            return
        emit({"type": "plan", "items": [
            {"id": 1, "text": "read the current implementation", "status": "open"},
            {"id": 2, "text": "implement winner()", "status": "open"},
            {"id": 3, "text": "tests pass", "status": "open"}], "confirmed": 0, "total": 3})
        emit({"type": "stage_start", "name": "inspect"})

        emit({"type": "tool_call", "id": "t1", "name": "read_file", "arguments": {"path": "rps/game.py"}, "risk": "SAFE"})
        if not S(0.5):
            return
        emit({"type": "tool_result", "id": "t1", "name": "read_file", "ok": True,
              "result": "import random\n\ndef play():\n    print(\"rock paper scissors\")\n"})
        emit({"type": "context", "used": 5200, "limit": 32768, "percent": 16, "tokens_per_s": 38.5, "compacted": False})

        emit({"type": "tool_call", "id": "t2", "name": "grep", "arguments": {"pattern": "def play", "path": "."}, "risk": "SAFE"})
        if not S(0.4):
            return
        emit({"type": "tool_result", "id": "t2", "name": "grep", "ok": True, "result": "rps/game.py:3:def play():"})
        emit({"type": "stage_done", "name": "inspect"})

        emit({"type": "sources", "query": "rock paper scissors winner modulo", "items": [
            {"n": 1, "domain": "en.wikipedia.org", "trust": {"level": "high", "tier": "T1"}, "url": "https://en.wikipedia.org/wiki/Rock_paper_scissors"},
            {"n": 2, "domain": "stackoverflow.com", "trust": {"level": "medium", "tier": "T3"}, "flags": ["tertiary"], "url": "https://stackoverflow.com/q/1"},
        ]})

        emit({"type": "stage_start", "name": "implement"})
        emit({"type": "edit", "path": "rps/game.py", "added": 11, "removed": 2, "created": False, "diff": DIFF})
        emit({"type": "tool_result", "name": "edit_file", "ok": True, "result": "edited rps/game.py"})
        if not S(0.4):
            return

        cmd = "pytest -q -x"
        emit({"type": "tool_call", "id": "t3", "name": "run_command", "arguments": {"command": cmd}, "risk": "CAUTION"})
        decision = approve({"tool_name": "run_command", "command": cmd, "risk_level": "CAUTION",
                            "reason": "runs a program in your project folder"}) if self.approve_commands else True
        if not decision:
            emit({"type": "tool_result", "id": "t3", "name": "run_command", "ok": False, "result": "denied by user"})
            if self._stream("Understood, I did not run the tests.", emit, stop, approve):
                emit({"type": "done", "reply": "Understood, I did not run the tests."})
            return
        emit({"type": "status", "state": "coding", "label": "running", "detail": cmd})
        if not S(0.8):
            return
        emit({"type": "tool_result", "id": "t3", "name": "run_command", "ok": True, "result": "....                       [100%]\n4 passed in 0.31s"})
        emit({"type": "verify", "state": "passed", "command": cmd, "attempt": 1, "max_attempts": 2})
        emit({"type": "stage_done", "name": "implement"})
        emit({"type": "plan", "items": [
            {"id": 1, "text": "read the current implementation", "status": "confirmed"},
            {"id": 2, "text": "implement winner()", "status": "confirmed"},
            {"id": 3, "text": "tests pass", "status": "confirmed", "evidence": "pytest: 4 passed"}], "confirmed": 3, "total": 3})
        emit({"type": "changes", "files": [{"path": "rps/game.py", "added": 11, "removed": 2, "new": False}], "added": 11, "removed": 2,
              "undo_available": True})
        emit({"type": "context", "used": 9800, "limit": 32768, "percent": 30, "tokens_per_s": 36.1, "compacted": False})
        if self._stream(ANSWER, emit, stop, approve, per_word=0.02):
            emit({"type": "verdict", "label": "likely", "summary": "2 sources, newest 2024", "warnings": []})
            emit({"type": "done", "reply": ANSWER})


def demo_footer() -> dict[str, Any]:
    return {"model": "demo-model:9b", "backend": "ollama", "gpu": "single", "access": "ask", "ctx_limit": 32768, "ctx_used": 0}


def run_demo(delay: float = 1.0, plain: bool | None = None, **app_kwargs: Any) -> int:

    from agent.ui.app import NightageApp

    agent = FakeAgent(delay)

    def context() -> str:
        return ("system prompt     1.1k tokens   4%\nfacts             0.3k tokens   1%\nrecalls (3)       0.8k tokens   2%\n"
                "project notes     0.2k tokens   1%\nconversation      7.4k tokens  23%\n(demo data)")

    class _Control:
        def request_compact(self) -> None: print("[demo] compact requested")
        def queue_user_message(self, text: str) -> None: print(f"[demo] message delivered: {text}")

    app = NightageApp(agent, plain=plain, footer_provider=demo_footer, context_provider=context, control=_Control(),
                      status_provider=lambda: {"mode": "demo (scripted, no model)"}, **app_kwargs)
    return app.run()
