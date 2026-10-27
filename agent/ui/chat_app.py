"""The real chat on the Nightage terminal UI.

    run_chat_ui(config, session_id=None, resume=False, plain=None, demo=False, input=None, output=None) -> int

Same start-up and shut-down semantics as the classic `agent.tui.run_repl` (they share `tui.setup_chat`,
`tui.start_fresh_session`, `tui._distill_on_exit`, `tui._unload_models_on_exit`):

  start    foreground-session marker (background worker yields), store, client, plugins/tools, new session or
           resume (compact preview of the last messages), persona name, tools-capable-model warning
  run      a `NightageApp` whose turn function is `chat_adapter.run_chat_turn` - messages are persisted by that
           function exactly as the classic loop does (user / tool / assistant rows, title, embeddings), so a session
           can be continued in either UI
  exit     notes distilled from the session, models unloaded (also on Ctrl-C / crash)

Tests and embedders can inject `client`, `store` and `console` (then nothing is opened or unloaded by this module
except what it created itself)."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from agent.ui import facts as F
from agent.ui.chat_adapter import ChatState
from agent.ui.context_view import inspector
from agent.ui.handlers import ChatController

FOOTER_TTL = 1.5     # seconds: footer facts touch sysfs / the greenlist file, do not recompute them on every redraw


def open_store(config: dict[str, Any]) -> Any:
    from agent.config import PROJECT_ROOT
    from agent.memory.store import Store

    db_path = Path(config["data"]["db_path"])
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Store(db_path)


def resolve_session(store: Any, session_id: int | None, resume: bool) -> tuple[int | None, str]:
    """(session id to resume or None for a new one, error text). Same rules as `agent resume`."""
    if session_id is None and not resume:
        return None, ""
    if session_id is None:
        last = store.get_last_session_id()
        return (last, "") if last is not None else (None, "There are no sessions yet. Start one with: nightage chat")
    existing = {row["id"] for row in store.list_sessions(limit=10_000)}
    if session_id not in existing:
        return None, f"No session with id {session_id}. Try: nightage sessions"
    return session_id, ""


def history_preview(messages: list[dict[str, Any]], width: int, count: int | None = None) -> list[str]:
    """Compact lines for the last messages of a resumed session (one line each, role labelled)."""
    from agent import tui

    n = count or tui.RESUME_PREVIEW_MESSAGES
    lines = []
    for m in messages[-n:]:
        if m["role"] not in ("user", "assistant"):
            continue
        label = "you" if m["role"] == "user" else "agent"
        text = " ".join(str(m["content"]).split())
        room = max(10, width - len(label) - 4)
        lines.append(f"{label}: {text if len(text) <= room else text[: room - 3] + '...'}")
    return lines


class _FooterCache:
    def __init__(self, state: ChatState, config: dict[str, Any], folder: str | None) -> None:
        self.state, self.config, self.folder = state, config, folder
        self._at, self._val = 0.0, {}

    def __call__(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._at > FOOTER_TTL or not self._val:
            ap = getattr(self.state.active_project, "key", None)
            self._val = F.footer_facts(self.config, self.state.model, folder=self.folder or ap or os.getcwd())
            self._at = now
        return dict(self._val)


def build_chat_app(state: ChatState, config: dict[str, Any], *, plain: bool | None = None, input: Any = None,  # noqa: A002
                   output: Any = None, console: Console | None = None, folder: str | None = None, **app_kwargs: Any) -> tuple[Any, ChatController]:
    """The wired `NightageApp` + its controller (no I/O is started). Used by `run_chat_ui` and by tests."""
    from agent.ui.app import NightageApp

    ctl = ChatController(state)
    footer = _FooterCache(state, config, folder)

    def status() -> dict[str, str]:
        rows = {"session": str(state.session_id), "persona": state.persona_name}
        ap = getattr(state.active_project, "key", None)
        if ap:
            rows["project"] = str(ap)
        plan = getattr(state.plan_holder, "plan", None)
        if plan is not None:
            rows["plan"] = f"{sum(1 for i in plan.items if i.done)}/{len(plan.items)} checks"
        if state.pending_task:
            rows["waiting"] = "task waits for an access level (/auto-<dir>, /medium-<dir>, /ask-<dir>)"
        return rows

    app = NightageApp(ctl.turn_fn, console=console, plain=plain, handlers=ctl.handlers(), control=ctl.control,
                      footer_provider=footer, status_provider=status, config=config, input=input, output=output, **app_kwargs)
    ctl.attach(app)
    app.context_provider = inspector(state, plain=app.plain, width=lambda: app.console.width)
    return app, ctl


def run_chat_ui(config: dict[str, Any], *, session_id: int | None = None, resume: bool = False, plain: bool | None = None,
                demo: bool = False, input: Any = None, output: Any = None,  # noqa: A002
                client: Any = None, store: Any = None, console: Console | None = None,
                open_store_fn: Any = None, **app_kwargs: Any) -> int:
    """Run the interactive chat on the new UI. Returns a process exit code (0 ok, 1 when `resume` finds nothing)."""
    if demo:
        from agent.ui.demo import run_demo

        return run_demo(plain=plain, input=input, output=output)

    from agent import foreground, tui
    from agent.llm import OllamaClient

    own_client, own_store = client is None, store is None
    client = client if client is not None else OllamaClient()
    store = store if store is not None else (open_store_fn or open_store)(config)
    err = Console(stderr=True)
    exit_code = 0
    ctx = foreground.session(config) if own_store else _Null()
    with ctx:
        try:
            sid, problem = resolve_session(store, session_id, resume)
            if problem:
                err.print(f"[red]{problem}[/red]")
                return 1
            quiet = Console(file=io.StringIO(), force_terminal=False, width=100)
            state = tui.setup_chat(client, config, quiet, store, sid)
            app, ctl = build_chat_app(state, config, plain=plain, input=input, output=output, console=console, **app_kwargs)
            _print_start(app, state, quiet, sid, client)
            try:
                exit_code = app.run()
            finally:
                _shutdown(app, state, client, config, own_client)
        finally:
            if own_store:
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass
    return exit_code


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


def _print_start(app: Any, state: ChatState, quiet: Console, resumed_id: int | None, client: Any) -> None:
    from agent import tui

    r = app.renderer
    captured = quiet.file.getvalue().strip() if isinstance(quiet.file, io.StringIO) else ""
    for ln in captured.splitlines():
        r.info(ln)
    if resumed_id is not None:
        history = state.store.get_messages(state.session_id)
        lines = history_preview(history, app.console.width)
        if lines:
            r.info("--- resuming previous conversation ---")
            for ln in lines:
                r.info(ln)
            r.info("--- end of history ---")
    r.print(Text(f"{state.persona_name} is ready. Type /help for commands."))
    buf = io.StringIO()
    tui._warn_if_model_cannot_use_tools(Console(file=buf, force_terminal=False, width=100), client, state.model)
    if buf.getvalue().strip():
        r.notice(" ".join(buf.getvalue().split()))


def _shutdown(app: Any, state: ChatState, client: Any, config: dict[str, Any], unload: bool) -> None:
    """Notes from the session, then free the GPU - in this order, and the unload even if the notes fail."""
    from agent import tui

    try:
        tui._distill_on_exit(state.store, state.vectors, client, config, state.session_id, app.console, state.tool_exchanges)
    except Exception:  # noqa: BLE001
        pass
    if unload:
        tui._unload_models_on_exit(app.console, client, config)
