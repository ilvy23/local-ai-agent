"""Real behaviour for the command REGISTRY of the chat UI.

`ChatController(state)` owns the chat-side logic and exposes what `NightageApp` wants:

    ctl = ChatController(state)
    app = NightageApp(ctl.turn_fn, handlers=ctl.handlers(), control=ctl.control, ...)
    ctl.attach(app)

* `turn_fn(prompt, emit, stop, approve)`   one chat turn through `agent.ui.chat_adapter.run_chat_turn`, plus the
  pre-turn "work in this repo" access question of the classic loop. Prompts that are really internal commands
  that must not overlap a turn (/clear, /new, /verify) arrive here too, through `run_exclusive`.
* `handlers()`  {command name: fn(args, app)}. Every handler returns text / a renderable / None and never reads
  stdin: the dock owns input. They reuse the classic implementations (agent.tui, agent.repoaccess, ...).
* `control`     request_compact -> ChatState.request_compact (the other step controls are code-agent only).

Safety rules are untouched: approvals travel through the dock (worker `approve`), DANGEROUS needs the typed word
`yes`, "always" is treated as a single yes (a folder is only remembered via /medium-<dir>, /auto-<dir>,
/greenlist-<dir>)."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.text import Text

from agent import repoaccess
from agent.ui.chat_adapter import ChatState, run_chat_turn

CODE_ONLY = "/{name} controls the code agent - it has no effect in a plain chat (use `code` for coding tasks)"
LEVEL_KINDS = (repoaccess.AUTO, repoaccess.MEDIUM, repoaccess.ASK)


class ChatControl:
    """`control` object of the app. Only /compact exists in a chat; the others are code-agent controls."""

    def __init__(self, state: ChatState) -> None:
        self.state = state

    def request_compact(self) -> None:
        self.state.request_compact()


class ChatController:
    def __init__(self, state: ChatState, *, quiet_console: Console | None = None) -> None:
        self.state = state
        self.app: Any = None
        self.control = ChatControl(state)
        self.quiet = quiet_console or Console(file=io.StringIO(), force_terminal=False, width=100)
        self._internal: dict[str, Callable[[Any], None]] = {}
        self._resumed: set[str] = set()
        self._lock = threading.Lock()
        self.on_session_change: Callable[[int], None] | None = None   # chat_app hook (e.g. refresh a title)

    # ------------------------------------------------------------------ wiring
    def attach(self, app: Any) -> None:
        self.app = app

    def handlers(self) -> dict[str, Callable[[str, Any], Any]]:
        h: dict[str, Callable[[str, Any], Any]] = {
            "model": self.cmd_model, "clear": self.cmd_clear, "new": self.cmd_new, "undo": self.cmd_undo,
            "plan": self.cmd_plan, "project": self.cmd_project, "notes": self.cmd_notes, "verify": self.cmd_verify,
            "access": self.cmd_access, "web": self.cmd_web, "compact": self.cmd_compact,
        }
        for name in ("steps", "mode", "say"):
            h[name] = lambda args, app, _n=name: self.cmd_code_only(_n)
        for name in ("auto", "medium", "ask", "greenlist", "ungreenlist"):
            h[name] = lambda args, app, _n=name: self.cmd_access_level(_n, args)
        return h

    # ------------------------------------------------------------------ turn function
    def turn_fn(self, prompt: str, emit: Callable[[Any], None], stop: threading.Event, approve: Any) -> None:
        with self._lock:
            job = self._internal.pop(prompt, None)
            resumed = prompt in self._resumed
            self._resumed.discard(prompt)
        if job is not None:
            job(stop)
            return
        if not resumed and self._hold_for_access_level(prompt):
            return
        approve_fn = approve.as_chat_approve() if hasattr(approve, "as_chat_approve") else approve
        # thinking text is captured so /thinking show works; it lives in memory only and is never stored
        run_chat_turn(self.state, prompt, emit, stop, approve_fn, suppress_print=True, capture_thinking=True)

    def _hold_for_access_level(self, prompt: str) -> bool:
        """Classic loop: 'work in this repo: <path>' on a folder with no grant asks for a level first."""
        s = self.state
        if s.pending_task:
            s.pending_task = None                       # asked something else: default level
        work_dir = repoaccess.parse_work_in(prompt)
        if not work_dir:
            return False
        folder, note = repoaccess.resolve_folder(work_dir)
        if folder is not None and s.active_project is not None:
            s.active_project.set_from(folder)
        if folder is not None and repoaccess.mode_for(folder) == repoaccess.ASK:
            self._say(repoaccess.question(folder, note))
            s.pending_task = prompt
            return True
        if folder is not None:
            self._say(f"{folder}: {repoaccess.mode_for(folder)} access already granted", dim=True)
        return False

    def _say(self, text: str, dim: bool = False) -> None:
        r = getattr(self.app, "renderer", None)
        if r is None:
            return
        if dim:
            r.info(text)
        else:
            r.print(Text(text))

    def run_exclusive(self, text: str, job: Callable[[threading.Event], None]) -> str | None:
        """Run `job` as a pseudo-turn (so it never overlaps a chat turn). Returns a message when it had to wait."""
        with self._lock:
            self._internal[text] = job
        if self.app is None or not self.app.start_turn(text):
            with self._lock:
                self._internal.pop(text, None)
            self.app and self.app._enqueue(text, kind="command", note="runs after the current turn")
            return None
        return None

    # ------------------------------------------------------------------ /model
    def installed_models(self) -> list[str]:
        try:
            return list(self.state.client.list_models())
        except Exception:  # noqa: BLE001
            return []

    def cmd_model(self, args: str, app: Any) -> str:
        s = self.state
        models = self.installed_models()
        if not models:
            return "Couldn't reach Ollama to list installed models."
        choice = args.strip()
        current = s.config["models"]["chat"]
        if not choice:
            lines = ["Installed models:"]
            lines += [f"  {i:2}. {m}{'  (current)' if m == current else ''}" for i, m in enumerate(models, 1)]
            lines.append("Switch with /model <number> or /model <name>.")
            return "\n".join(lines)
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected = models[int(choice) - 1]
        elif choice in models:
            selected = choice
        else:
            return f"{choice!r} is not one of the installed models. Staying on {current}. (/model lists them)"
        out: list[str] = []
        from agent.settings import set_value

        try:
            set_value("chat_model", selected)
        except Exception as exc:  # noqa: BLE001
            out.append(f"Switched for now, but couldn't save it as the default: {exc}")
        s.config["models"]["chat"] = selected
        s.model = selected
        out.append(f"Switched to {selected}. It is used from the next turn on, including in new chats.")
        buf = io.StringIO()
        from agent import tui

        tui._warn_if_model_cannot_use_tools(Console(file=buf, force_terminal=False, width=100), s.client, selected)
        if buf.getvalue().strip():
            out.append(buf.getvalue().strip())
        return "\n".join(out)

    # ------------------------------------------------------------------ /clear /new
    def _fresh(self, label: str) -> None:
        from agent import tui

        s = self.state
        r = getattr(self.app, "renderer", None)
        if r is not None:
            r.info("wrapping up - taking notes before I go...")
        tui.start_fresh_session(s, s.tool_exchanges)
        s.pending_task = None
        if self.on_session_change:
            try:
                self.on_session_change(s.session_id)
            except Exception:  # noqa: BLE001
                pass
        self._say(f"{label} (session {s.session_id}) - nothing earlier carries over.", dim=True)

    def cmd_clear(self, args: str, app: Any) -> None:
        self.run_exclusive("/clear", lambda stop: self._fresh("Started a fresh conversation"))

    def cmd_new(self, args: str, app: Any) -> None:
        self.run_exclusive("/new", lambda stop: self._fresh("Started a new chat"))

    # ------------------------------------------------------------------ /undo /plan /project /notes
    def cmd_undo(self, args: str, app: Any) -> str:
        from agent import tui

        lines = tui._UNDO.describe() if args.strip().lower() == "list" else tui._UNDO.undo()
        return "\n".join(lines) if lines else "nothing to undo"

    def cmd_plan(self, args: str, app: Any) -> str:
        s = self.state
        if args.strip().lower() == "clear":
            s.plan_holder.plan = None
            return "plan  cleared"
        if s.plan_holder.plan is None:
            return "plan  none for this task (one is made for big build tasks)"
        return "\n".join(s.plan_holder.plan.lines())

    def cmd_project(self, args: str, app: Any) -> str:
        s = self.state
        ap, pm = s.active_project, s.project_memory
        if ap is None or not ap.key:
            return "project  none active yet (edit a file in a repo or say 'work in this repo: <path>')"
        if args.strip().lower() == "clear":
            return f"project  forgot {pm.clear(ap.key)} note(s)"
        return "\n".join([f"project  {ap.key}", *(f"  - {note}" for _nid, note in pm.notes(ap.key))])

    def cmd_notes(self, args: str, app: Any) -> str:
        s = self.state
        if args.strip().lower() == "clear":
            s.store.clear_working_notes(s.session_id)
            return "Notes cleared."
        notes = s.store.get_working_notes(s.session_id)
        return f"Notes this session:\n{notes}" if notes else "No notes taken this session."

    # ------------------------------------------------------------------ /verify
    def cmd_verify(self, args: str, app: Any) -> str | None:
        s = self.state
        search_tool = s.registry.get("web_search")
        if not s.last_reply or search_tool is None:
            return "verify  nothing to check yet (ask something first)"
        self.run_exclusive("/verify", lambda stop: self._verify(search_tool, stop))
        return None

    def _verify(self, search_tool: Any, stop: threading.Event) -> None:
        from agent.research import evidence as turn_evidence
        from agent.research import verify as verifymod

        def run_search(query: str) -> tuple[list[dict], dict]:
            search_tool.handler(query=query, num_results=4)
            found = turn_evidence.last_search()
            return found["items"], found["verdict"]

        self._say("verifying the last answer against the web...", dim=True)
        try:
            checked = verifymod.verify(self.state.last_reply or "", run_search)
        except Exception as exc:  # noqa: BLE001 - a failed check must not end the chat
            self._say(f"verify failed: {exc}")
            return
        self._say(verifymod.summarize(checked))

    # ------------------------------------------------------------------ access
    def cmd_access(self, args: str, app: Any) -> str:
        handled = repoaccess.handle_command("/access")
        return handled[1] if handled else "no access information"

    def cmd_access_level(self, name: str, args: str) -> str | None:
        """/auto-<dir>, /medium-<dir>, /ask-<dir>, /greenlist-<dir>, /ungreenlist-<dir> (same code as the classic loop)."""
        s = self.state
        text = f"/{name}-{args}" if args.strip() else f"/{name}"
        handled = repoaccess.handle_command(text)
        if handled is None:
            return f"unknown access command: {text}"
        kind, message = handled
        m = repoaccess.COMMAND_RE.match(text.strip())
        if m and m.group(2) and kind in (repoaccess.AUTO, repoaccess.MEDIUM, "greenlist") and s.active_project is not None:
            folder, _ = repoaccess.resolve_folder(m.group(2))
            if folder is not None:
                s.active_project.set_from(folder)
        self._say(message)
        if s.pending_task and kind in LEVEL_KINDS:
            task, s.pending_task = s.pending_task, None          # level chosen: carry on with the task
            self._resume_task(task)
        return None

    def _resume_task(self, task: str) -> None:
        if self.app is None:
            return
        with self._lock:
            self._resumed.add(task)
        if not self.app.start_turn(task):
            self.app._enqueue(task)

    # ------------------------------------------------------------------ misc
    def cmd_web(self, args: str, app: Any) -> str | None:
        q = args.strip()
        if not q:
            return "Add something to search for: /web <question>"
        if not app.start_turn(f"/web {q}"):
            app._enqueue(f"/web {q}")
        return None

    def cmd_compact(self, args: str, app: Any) -> str:
        self.state.request_compact()
        if app.worker.busy:
            return "compacting at the next tool step"
        return ("marked: the next turn compacts its tool results at its first step. Between turns the history is rebuilt "
                "from the database and older messages are summarised automatically (see /context).")

    def cmd_code_only(self, name: str) -> str:
        return CODE_ONLY.format(name=name)
