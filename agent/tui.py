"""Terminal chat REPL: Rich console, streamed replies, /quit and /help."""

from __future__ import annotations

from agent import brand

from collections.abc import Callable, Iterator
from typing import Any

import contextlib
import inspect
import io
import json
import logging
import os
import random
import re
import threading
import time
from datetime import date
from pathlib import Path

import httpx
from agent.llm import EmbeddingUnavailable
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text

from agent import autonomy, changes, projectmem, repoaccess, toolargs, undo, verify
from agent.research import evidence as turn_evidence
from agent.research import taint
from agent import plan as planmod
from agent.memory.distill import JUNK_FACT_RE, distill_session
from agent.memory.recall import build_context
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex
from agent.safety import RiskLevel, block_reason, classify_command
from agent.tools.files import safe_text
from agent.tools.registry import Tool, ToolRegistry, default_registry, parse_tool_call_fallback
from agent.chat import gates
from agent.ui.chat_adapter import ApprovalRequest, TOOL_RESULT_CAP, TurnControl, resolve_approval

from agent.chat.helpers import (  # noqa: F401 - moved out of this module; re-exported so `agent.tui.<name>` keeps working
    CHAT_NUM_CTX,
    DENIAL_FRAMING,
    DENIED_RESULT,
    REMEMBER_MAX_CHARS,
    RESULT_TRUNCATE_LEN,
    _DEFAULT_TOOL_FRAMING,
    _TOOL_RESULT_FRAMING,
    _WEB_INJECT,
    _classify_call,
    _extract_tool_call,
    _format_tool_call,
    _frame_tool_result,
    _make_remember_tool,
    _make_save_skill_tool,
    _make_take_note_tool,
    _mood,
    _normalize_arguments,
    _num_ctx,
    _one_line,
    _react_to_stuck,
    _static_risk_level,
    _strip_thinking_artifacts,
)

logger = logging.getLogger(__name__)

TITLE_MAX_LEN = 60
RESUME_PREVIEW_MESSAGES = 6


_THINK_DOTS = (".", "..", "...")


def _ui_settings() -> dict[str, Any]:
    """Status-line label and live-reply settings: env overrides config.yaml (`ui:`)."""
    ui: dict[str, Any] = {"wait_label": "thinking"}
    try:
        from agent.config import load_config

        ui.update(load_config().get("ui") or {})
    except Exception:  # noqa: BLE001 - cosmetic settings must never break the chat
        logger.debug("could not read ui settings", exc_info=True)
    label = os.environ.get("AGENT_WAIT_LABEL", "").strip()
    if label:
        ui["wait_label"] = label
    return ui


class _ThinkingSpinner:
    """The status line while a call is generating and nothing can be shown yet.

    Shows the waiting label with growing dots and the seconds waited, e.g. "waiting...  12s". Most noticeable on "hybrid reasoning" models (Qwen3
    and similar), which can spend a long time in a hidden reasoning pass before any real
    text starts - without this, that pass just looks like a hang. A background thread
    redraws it a few times a second. `stop()` removes it early (when the live reply starts);
    the label is `ui.wait_label` in config.yaml (or AGENT_WAIT_LABEL).
    """

    def __init__(
        self, console: Console, interval: float = 0.08, label: str | None = None, coding: bool = False,
        suffix: str = "", style: str = "dim",
    ) -> None:
        from agent.liveview import ThinkingText

        ui = _ui_settings()
        self._fixed = label is not None
        if coding:
            self._text = ThinkingText("coding", style="green")      # the agent is busy writing code
        else:
            self._text = ThinkingText(label or str(ui.get("wait_label") or "thinking"), style=style, suffix=suffix)
        from rich.spinner import SPINNERS

        SPINNERS.setdefault("plain", {"interval": 100, "frames": [""]})     # words and dots only, no glyph
        self._status = console.status(self._text.render(), spinner="plain")
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._running = False

    def set_coding(self) -> None:
        """Switch to the green "coding" label mid-wait (a no-op for a fixed label like "compacting")."""
        if self._fixed:
            return
        from agent.liveview import ThinkingText

        self._text = ThinkingText("coding", style="green")

    def _animate(self) -> None:
        while not self._stop.wait(self._interval):
            self._status.update(self._text.render())

    def __enter__(self) -> "_ThinkingSpinner":
        self._status.__enter__()
        self._running = True
        self._thread.start()
        return self

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop.set()
        self._thread.join(timeout=1)
        self._status.__exit__(None, None, None)

    def __exit__(self, *exc: Any) -> None:
        if self._running:
            self._running = False
            self._stop.set()
            self._thread.join(timeout=1)
            self._status.__exit__(*exc)


HELP_TEXT = (
    "Commands:\n"
    "  /help        Show this help text\n"
    "  /model       Switch to a different AI model for this chat\n"
    "  /clear       Start a fresh conversation without leaving the app\n"
    "  /notes       Show what it's noted down for this conversation so far\n"
    "  /notes clear Erase this conversation's notes\n"
    "  /auto-<dir>  Full access in a folder this session (edit + run commands, no asking)\n"
    "  /medium-<dir> Edit freely in a folder, still ask before running commands\n"
    "  /greenlist-<dir> Full access in a folder forever (/greenlist lists, /ungreenlist-<dir> removes)\n"
    "  /undo        Take back the files the last prompt wrote (/undo list shows the history)\n"
    "  /plan        Show the acceptance list of the current task (/plan clear drops it)\n"
    "  /project     Show what is remembered about the active project (/project clear forgets it)\n"
    "  /verify      Check the key claims of the last answer against fresh, independent searches\n"
    "  /quit        Exit the chat\n"
    "\n"
    "End any message with /web (or /search) to force a live web search for it,\n"
    "e.g. 'latest news on the mars mission /web'. The agent can also search on\n"
    "its own when a question needs current info.\n"
    "\n"
    "You can also just ask things like \"what did Mike say yesterday\" or\n"
    "\"what's up with my chat with Sara\" - it can look through your WhatsApp\n"
    "messages directly. And if it's learned something useful before, it'll\n"
    "remember it without you having to repeat yourself."
)

# Rotated at random each turn for the brief "looking things up" pause before
# a reply - purely cosmetic, the lookup itself is near-instant.
_MEMORY_STATUS_LINES = (
    "checking what I remember...",
    "digging through memory...",
    "seeing if I know this already...",
    "looking back through what we've talked about...",
)

# Trailing tokens that force a live web search for the message.
_WEB_TRIGGERS = {"/web", "/search", "/s", "/w", "/net", "/online"}

# Being told you're wrong is the strongest evidence your memory is unreliable —
# and it's exactly when a small model apologises and invents something worse
# instead of checking. The system prompt alone doesn't beat that reflex, so
# detect the correction and instruct it, for this turn only, to go and look.
_DISAGREE_CUE = re.compile(
    r"\b(no|nope|nah|not|isn'?t|aren'?t|wasn'?t|weren'?t|wrong|incorrect|false|"
    r"mistake|mistaken|untrue|actually|bullshit|nonsense)\b",
    re.IGNORECASE,
)

_CORRECTION_CHECK = (
    "Did the user just say the assistant's previous answer was factually wrong "
    "or inaccurate?\n"
    "Answer with ONLY 'yes' or 'no'.\n"
    "'yes' = they are disputing or correcting a fact the assistant stated.\n"
    "'no' = anything else: a new question, a follow-up, thanks, or correcting "
    "their own wording."
)

_CORRECTION_NUDGE = (
    "The user is telling you your previous answer was wrong. Your memory of this "
    "is demonstrably unreliable, so do NOT answer from memory and do NOT simply "
    "agree. Use web_search now to check the facts, then reply based on what you "
    "find."
)


def _looks_like_disagreement(text: str) -> bool:
    """Cheap gate so the classifier below only runs on plausible corrections."""
    return bool(_DISAGREE_CUE.search(text))


def _is_factual_correction(
    client: Any, config: dict[str, Any], assistant_text: str, user_text: str
) -> bool:
    """Is the user disputing a fact we just stated? Never raises.

    Both sides are needed: "no, those aren't in there" says nothing on its own —
    without the answer it refers to, the classifier can only shrug and say no.
    """
    pair = f"Assistant said: {assistant_text[:1500]}\n\nUser replied: {user_text}"
    try:
        reply = "".join(
            client.chat(
                messages=[
                    {"role": "system", "content": _CORRECTION_CHECK},
                    {"role": "user", "content": pair},
                ],
                model=config["models"]["background"],
                stream=False,
            )
        ).strip().lower()
    except Exception:  # noqa: BLE001 - if unsure, leave the turn alone
        return False
    return reply.startswith("yes")


def _split_web_trigger(text: str) -> tuple[str, bool]:
    """If the message ends with a /web-style token, strip it and flag a search."""
    parts = text.rsplit(None, 1)
    if len(parts) == 2 and parts[1].lower() in _WEB_TRIGGERS:
        return parts[0].strip(), True
    return text, False


_QUERY_REWRITE_SYSTEM = (
    "You turn a casual question into a short, effective web search query. "
    "Reply with ONLY the search query text - no quotes, no explanation, "
    "no punctuation at the end."
)


def _rewrite_search_query(raw_message: str, client: Any, config: dict[str, Any]) -> str:
    """Turn a raw /web message into a cleaner search-engine query.

    Only used for the FORCED /web path - the autonomous tool-call path
    already has the model compose its own query as part of its own
    reasoning (already "produced" text, not raw conversational phrasing),
    so a second rewrite pass there would just add latency with no clear
    gain and risks losing thread context this isolated call doesn't see.

    Uses the background model (cheap/fast), never blocks the search on
    failure - falls back to the raw message unchanged.
    """
    try:
        rewritten = "".join(
            client.chat(
                messages=[
                    {"role": "system", "content": _QUERY_REWRITE_SYSTEM},
                    {"role": "user", "content": raw_message},
                ],
                model=config["models"]["background"],
                stream=False,
            )
        ).strip()
    except Exception:  # noqa: BLE001 - never block a search on the rewrite failing
        return raw_message
    return rewritten or raw_message

CONNECTION_HINT = (
    "[red]Could not reach Ollama.[/red] Is it running? Try: [bold]ollama serve[/bold]"
)

MAX_ITERATIONS_HINT = (
    "[red]Stopped[/red]: step limit reached before the task was done. Say \"continue\" to go on "
    "(the work so far is kept), or raise tools.max_iterations."
)


def _warn_if_model_cannot_use_tools(console: Console, client: Any, model: str) -> None:
    """Say so loudly if the chat model can't call tools.

    Ollama silently ignores `tools=` for models whose template has no slot for
    them, so the model never sees the tools and cheerfully invents results —
    a made-up directory listing looks exactly like a real one. Better a warning
    at startup than a confident lie later.
    """
    try:
        supported = client.supports_tools(model)
    except Exception:  # noqa: BLE001 - a warning must never break the chat
        return
    if supported is False:
        console.print(
            f"[yellow]Heads up:[/yellow] [bold]{model}[/bold] doesn't support tool "
            "calling, so it can't read files, run commands, or search the web — "
            "and it may [bold]make up[/bold] results instead of saying so.\n"
            f"[dim]Use a tools-capable model:  {brand.CMD} settings set chat_model qwen2.5:7b[/dim]"
        )


def _handle_model_switch(client: Any, config: dict[str, Any], console: Console, current: str) -> str:
    """The /model command: list installed models, switch to the chosen one
    immediately for the rest of THIS session, and save it as the default for
    future ones. Returns the model name to keep using (unchanged if the
    switch didn't happen).

    Mutating config["models"]["chat"] in place is what makes this take
    effect on the very next turn with no restart needed: _run_agent_turn()
    re-reads that value fresh from this same config dict (passed by
    reference) on every call, rather than from a captured local variable.
    """
    try:
        models = client.list_models()
    except Exception:  # noqa: BLE001 - never crash the chat over a listing failure
        models = []
    if not models:
        console.print("[yellow]Couldn't reach Ollama to list installed models.[/yellow]")
        return current

    console.print("[bold]Installed models:[/bold]")
    for i, m in enumerate(models, 1):
        marker = " [dim](current)[/dim]" if m == current else ""
        console.print(f"  {i:2}. {m}{marker}")

    try:
        choice = console.input("Pick a number, or type a model name: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return current
    if not choice:
        return current

    if choice.isdigit() and 1 <= int(choice) <= len(models):
        selected = models[int(choice) - 1]
    elif choice in models:
        selected = choice
    else:
        console.print("[yellow]That's not one of the models I found. Staying on the current one.[/yellow]")
        return current

    from agent.settings import set_value

    try:
        set_value("chat_model", selected)
    except ValueError as exc:
        console.print(f"[yellow]Switched for now, but couldn't save it as the default: {exc}[/yellow]")

    config["models"]["chat"] = selected
    console.print(f"[green]Switched to {selected}.[/green] I'll use it from now on, including in new chats.")
    _warn_if_model_cannot_use_tools(console, client, selected)
    return selected


def _unload_models_on_exit(console: Console, client: Any, config: dict[str, Any]) -> None:
    """Free the GPU when leaving the chat, if ollama.unload_on_exit is set.

    Runs from a `finally`, so it happens on /quit, Ctrl-C, and a crash alike —
    leaving a 5GB model pinned because of an exception would defeat the point.
    """
    if not config.get("ollama", {}).get("unload_on_exit", True):
        return
    try:
        freed = client.unload_all()
    except Exception:  # noqa: BLE001 - cleanup must never raise on the way out
        return
    if freed:
        console.print(f"[dim]freed: {', '.join(freed)}[/dim]")


def _http_error_message(exc: httpx.HTTPError) -> str:
    """Render an httpx error for the console: connection issues get the
    friendly "is it running?" hint, but a status error (Ollama reachable but
    rejecting the request) is surfaced with its actual status and message
    instead, so it isn't misreported as "could not reach Ollama".
    """
    from rich.markup import escape

    from agent import friendly

    line, hint = friendly.explain(exc)
    return f"[red]{escape(line)}[/red] [dim]next: {escape(hint)}[/dim]"


class ChatSession:
    """Holds in-memory message history for one chat run and talks to a client."""

    def __init__(self, client: Any, model: str, system_prompt: str) -> None:
        self.client = client
        self.model = model
        self.history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def send(self, user_input: str, on_token: Callable[[str], None] | None = None) -> str:
        """Append user_input, call the client, append the assistant reply, return it."""
        self.history.append({"role": "user", "content": user_input})

        tokens: list[str] = []
        for token in self.client.chat(
            messages=self.history, model=self.model, stream=True, num_ctx=_num_ctx({}, self.model)
        ):
            tokens.append(token)
            if on_token:
                on_token(token)

        reply = "".join(tokens)
        self.history.append({"role": "assistant", "content": reply})
        return reply


def print_resume_preview(console: Console, messages: list[dict[str, str]]) -> None:
    """Print the last few exchanges so the user sees where they left off."""
    preview = messages[-RESUME_PREVIEW_MESSAGES:]
    console.print("[dim]--- resuming previous conversation ---[/dim]")
    for message in preview:
        label = "you" if message["role"] == "user" else "agent"
        console.print(f"[dim]{label}: {message['content']}[/dim]")
    console.print("[dim]--- end of history ---[/dim]")


def _embed_message(
    vectors: VectorIndex | None,
    client: Any,
    config: dict[str, Any],
    message_id: int,
    text: str,
) -> None:
    """Embed a persisted message into the vector index. Never raises.

    Retries: the chat model is resident while this runs, so on a smaller GPU the
    embedding model may not fit and Ollama answers 500. Without a retry the
    message is stored but never indexed, and semantic recall can't see it again.
    """
    from agent.memory.distill import embed_with_retry

    if vectors is None or not vectors.available:
        return
    embedding = embed_with_retry(client, text, config["models"]["embed"])
    if embedding is None:
        return  # repair_unembedded() picks it up later rather than losing it
    try:
        vectors.add("message", message_id, text, embedding)
    except Exception as exc:  # noqa: BLE001 - embedding must never break a turn
        logger.warning("Failed to index message %s: %s", message_id, exc)


_CAUTION_AFFIRM = {"y", "yes"}


_CODING_ASK_RE = re.compile(
    r"\b(code|coding|script|function|refactor|implement|programm?|class|module|package|bug|fix|test|tests|pytest|"
    r"game|app|website|api|cli|repo|schreib\w*|erstell\w*|baue?|bau|entwickl\w*|datei|skript|spiel|erweiter\w*)\b"
    r"|\.(py|js|ts|rs|go|c|cpp|java|sh|html|css|json|md|toml|nix)\b",
    re.IGNORECASE,
)
_THINKING_CODE_CUES = (
    "write_file", "replace_in_file", "def ", "class ", "import ", "function", "```", "implement",
    "refactor", "pytest", "test_", ".py", "the file", "create the", "write the", "let me write",
    "i'll write", "i'll create", "now write", "datei",
)


def _looks_like_coding(text: str) -> bool:
    return bool(text and _CODING_ASK_RE.search(text))


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


class _ThinkingCue:
    """Watches the streamed reasoning for signs that the model is about to write code and calls
    `on_code` once. Tool-call arguments themselves are not streamed, so this is the earliest signal."""

    def __init__(self, on_code: Callable[[], None]) -> None:
        self._on_code, self._fired, self._tail = on_code, False, ""

    def feed(self, piece: str) -> None:
        if self._fired:
            return
        self._tail = (self._tail + piece.lower())[-200:]
        if any(cue in self._tail for cue in _THINKING_CODE_CUES):
            self._fired = True
            self._on_code()


@contextlib.contextmanager
def _edit_status(console: Console, shown_path: str, is_new: bool) -> Iterator[None]:
    """Green `editing. .. ... → file` while a file is written. The write itself is instant, so the
    line is held for a moment on a real terminal - long enough to read, short enough to not matter."""
    if not console.is_terminal:
        yield
        return
    verb = "creating" if is_new else "editing"
    started = time.monotonic()
    with _ThinkingSpinner(console, label=verb, suffix=shown_path, style="green"):
        yield
        remaining = 0.7 - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


def _describe_call(console: Console, name: str, arguments: dict[str, Any]) -> str:
    """One-line, markup-ready description of a tool call for the approval prompt.

    File edits show only the file and the line delta (`edit rps/game.py +3 -1`), never the code;
    everything else is the command / call, collapsed to one line and cut to the terminal width."""
    path = str(arguments.get("path", "")) if name in changes.TRACKED_TOOLS else ""
    if path:
        shown = escape(changes.display_path(path))
        if name == "write_file":
            before = changes.read_text(path)
            added, removed = changes.line_stats(before, str(arguments.get("content", "")))
            verb = "create" if before is None else "overwrite"
        else:
            added = len(str(arguments.get("replace", "")).splitlines())
            removed = len(str(arguments.get("search", "")).splitlines())
            verb = "edit"
        return f"[bold]{verb}[/bold] {shown} [green]+{added}[/green] [red]-{removed}[/red]"
    command = str(arguments.get("command", ""))
    text = command if command else _format_tool_call(name, arguments)
    return escape(_one_line(text, max(console.width - 24, 20)))


def _quiet_console(console: Console) -> Console:
    """A console that prints nothing (the turn's `suppress_print` mode: events carry everything)."""
    return Console(file=io.StringIO(), width=getattr(console, "width", 100), quiet=True, force_terminal=False)


class _NullSpinner:
    """Stands in for _ThinkingSpinner when nothing may be drawn."""

    def __enter__(self) -> "_NullSpinner":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def stop(self) -> None:
        return None

    def set_coding(self) -> None:
        return None


def _spinner(ctl: "TurnControl | None", console: Console, **kwargs: Any) -> Any:
    if ctl is not None and ctl.suppress_print:
        return _NullSpinner()
    return _ThinkingSpinner(console, **kwargs)


_TOOL_STATUS = {
    "web_search": ("searching", "query"), "web_fetch": ("fetching", "url"), "read_file": ("reading", "path"),
    "list_dir": ("listing", "path"), "search_files": ("searching files", "pattern"), "find_files": ("finding files", "pattern"),
    "run_command": ("running command", "command"), "write_file": ("writing", "path"),
    "replace_in_file": ("editing", "path"), "deep_research": ("researching", "question"),
}


def _tool_status(ctl: "TurnControl | None", name: str, arguments: dict[str, Any]) -> None:
    if ctl is None or ctl.emit is None:
        return
    label, key = _TOOL_STATUS.get(name, (f"running {name}", ""))
    detail = _one_line(str(arguments.get(key, "")), 160) if key else ""
    ctl.event("status", state="tool", label=label, detail=detail, tool=name, since=time.time())


def _emit_search_evidence(ctl: "TurnControl | None") -> None:
    """After a web search: `sources` and `verdict` events from the turn's evidence record (once per search)."""
    if ctl is None or ctl.emit is None:
        return
    from agent.chat.executor import emit_search_evidence

    seen = {"sig": ctl.sources_sig}
    emit_search_evidence(ctl.event, seen)
    ctl.sources_sig = seen["sig"]


def _unified_diff(path: str, before: str | None, after: str, limit: int = 2000) -> tuple[str, bool]:
    from agent.chat.executor import unified_diff

    return unified_diff(path, before, after, limit)


def _ask_approval(
    console: Console, ctl: "TurnControl | None", level: RiskLevel, description: str, name: str,
    arguments: dict[str, Any], from_web: bool, plain: str, diff: str,
) -> bool:
    """Approval through `ctl.approve_fn` when given, else the console prompt. With `suppress_print` and no
    `approve_fn` there is no way to ask, so the answer is no (never a stdin read behind a hidden console)."""
    if ctl is not None:
        ctl.event("awaiting_approval", name=name, risk=level.name, description=plain, arguments=arguments,
                  from_web=from_web, diff=diff, needs_typed_yes=level is RiskLevel.DANGEROUS)
        if ctl.approve_fn is not None:
            request = ApprovalRequest(name, level, level.name, plain, dict(arguments), from_web, diff,
                                      level is RiskLevel.DANGEROUS)
            try:
                return resolve_approval(level, ctl.approve_fn(request))
            except Exception:  # noqa: BLE001 - a broken approver means no
                logger.warning("approve_fn failed", exc_info=True)
                return False
        if ctl.suppress_print:
            return False
    return _approve(console, level, description)


def _approve(console: Console, level: RiskLevel, description: str) -> bool:
    """Ask on a single line. `description` is markup (see `_describe_call`). Returns True to run.

    CAUTION: y / yes (any case), anything else is a no. DANGEROUS still needs the exact,
    case-sensitive word `yes` - a deliberate extra step for truly high-stakes actions like
    sending a real message. SAFE never gets here."""
    if level is RiskLevel.CAUTION:
        answer = console.input(f"[yellow]Run this?[/yellow] {description} [yellow]\\[y/n][/yellow] ").strip().lower()
        return answer in _CAUTION_AFFIRM
    answer = console.input(f"[red]DANGEROUS[/red] {description} [red]type yes:[/red] ").strip()
    return answer == "yes"


# Tools that only look: they run without asking (risk "safe") and print a dim line saying
# what they read. Everything that writes, sends or runs a command goes through the approval
# gate below (CAUTION: y/yes, DANGEROUS: the exact word "yes", BLOCKED: never).
READ_ONLY_TOOLS = frozenset({"read_file", "list_dir", "search_files", "find_files", "disk_usage", "list_drives"})


def _execute_tool(
    registry: ToolRegistry,
    store: Store,
    console: Console,
    config: dict[str, Any],
    name: str,
    arguments: dict[str, Any],
    ctl: "TurnControl | None" = None,
) -> str:
    """Gate, run, and audit a tool call.

    Every tool call is risk-classified and gated before execution: SAFE runs
    silently, CAUTION/DANGEROUS require approval, BLOCKED never runs. Every
    decision (run, deny, block) writes an audit row. Handler errors are caught
    and fed back to the model.

    With a `ctl` (see agent.ui.chat_adapter) the call is also reported as events: tool_call (full
    arguments + risk), awaiting_approval, status, tool_result (full result, seconds, ok) and, for file
    edits, edit (diff). Without one nothing changes.
    """
    started = time.monotonic()

    def finish(result: str, ok: bool, level: RiskLevel | None = None) -> str:
        if ctl is not None:
            shown = result if len(result) <= TOOL_RESULT_CAP else result[:TOOL_RESULT_CAP]
            ctl.event("tool_result", name=name, result=shown, truncated=len(result) > TOOL_RESULT_CAP,
                      chars=len(result), seconds=round(time.monotonic() - started, 3), ok=ok,
                      risk=level.name if level is not None else None)
        return result

    tool = registry.get(name)
    if tool is None:
        result = f"Error: no such tool '{name}'."
        if ctl is not None:
            ctl.event("tool_call", name=name, summary=name, arguments=arguments, risk=None)
        store.add_audit_log(
            kind="tool",
            detail=json.dumps({"name": name, "arguments": arguments}),
            approved=0,
            result=result,
        )
        return finish(result, False)

    arguments, bad_arguments = toolargs.validate(tool, arguments)
    if bad_arguments:
        if ctl is not None:
            ctl.event("tool_call", name=name, summary=_one_line(_format_tool_call(name, arguments), 160),
                      arguments=arguments, risk=None)
        return finish(bad_arguments, False)               # nothing ran and nothing was asked: the model just retries

    is_shell = tool.risk == "dynamic"
    audit_kind = "shell" if is_shell else "tool"
    command = str(arguments.get("command", ""))
    detail = command if is_shell else json.dumps({"name": name, "arguments": arguments})

    level = _classify_call(tool, arguments, config)
    if ctl is not None:
        ctl.event("tool_call", name=name, summary=_one_line(_format_tool_call(name, arguments), 160),
                  arguments=arguments, risk=level.name)

    # BLOCKED: never runs.
    if level is RiskLevel.BLOCKED:
        reason = block_reason(command, config) if is_shell else (
            "sending WhatsApp messages is switched off" if name.startswith("wa_send") else "tool declared blocked")
        result = f"BLOCKED by safety policy: {reason}"
        console.print(f"[red]{result}[/red]")
        store.add_audit_log(kind=audit_kind, detail=detail, approved=0, result=result)
        return finish(result, False, level)

    # Approval gate for CAUTION / DANGEROUS.
    if level is not RiskLevel.SAFE:
        description = _describe_call(console, name, arguments)
        access = gates.decide_access(name, arguments, level)
        from_web = access.from_web
        if from_web:
            description += " [yellow]- text taken from a web page[/yellow]"
        if access.auto_allowed:
            pass                                  # folder access granted: no question (never for text a web page supplied)
        else:
            if ctl is None:
                allowed = _approve(console, level, description)
            else:
                from agent.chat.turn import plain_description

                path = str(arguments.get("path", "")) if name in changes.TRACKED_TOOLS else ""
                diff = ""
                if path and name == "write_file":
                    diff = _unified_diff(changes.display_path(path), changes.read_text(path),
                                         str(arguments.get("content", "")))[0]
                try:
                    plain = plain_description(name, arguments) + (" - text taken from a web page" if from_web else "")
                except Exception:  # noqa: BLE001
                    plain = _one_line(_format_tool_call(name, arguments), 200)
                allowed = _ask_approval(console, ctl, level, description, name, arguments, from_web, plain, diff)
            if not allowed:
                console.print("[dim](denied)[/dim]")
                store.add_audit_log(
                    kind=audit_kind, detail=detail, approved=0, result=DENIED_RESULT
                )
                return finish(DENIED_RESULT, False, level)
            started = time.monotonic()                # the wait for the answer is not the tool's runtime

    # Reads never ask - but they are shown, so you always see what the model looked at.
    if level is RiskLevel.SAFE and name in READ_ONLY_TOOLS:
        console.print(f"[dim]read › {_format_tool_call(name, arguments)}[/dim]")

    _tool_status(ctl, name, arguments)

    # Approved (or SAFE): run it.
    tracker = _tracker()
    tracked = tracker.is_tracked(name, arguments)
    if tracked:
        tracker.checkpoint_before(str(arguments["path"]))
    before = changes.read_text(arguments["path"]) if tracked else None
    errored = False
    try:
        if tracked:
            with _edit_status(console, changes.display_path(str(arguments["path"])), before is None):
                result = tool.handler(**arguments)
        else:
            result = tool.handler(**arguments)
    except Exception as exc:  # noqa: BLE001 - handler errors go back to the model
        result = f"Error running tool '{name}': {exc}"
        errored = True

    # Backstop: a result that can't be encoded as UTF-8 kills the whole session
    # when it's sent to Ollama as JSON. Filenames are bytes and a single legacy
    # one (a music folder named `Alizée` in latin-1) was enough. The tools clean
    # their own output, but nothing reaching the model should be able to do this.
    result = safe_text(result)
    if tracked:
        after = changes.read_text(arguments["path"])
        tracker.record(str(arguments["path"]), before, after)
        if ctl is not None and after is not None and before != after:
            shown_path = changes.display_path(str(arguments["path"]))
            added, removed = changes.line_stats(before, after)
            diff, cut = _unified_diff(shown_path, before, after)
            ctl.event("edit", path=shown_path, abs_path=str(Path(str(arguments["path"])).expanduser()),
                      added=added, removed=removed, created=before is None, diff=diff, diff_truncated=cut)

    if is_shell and level is RiskLevel.SAFE:
        console.print(f"[dim][ran: {command}][/dim]")

    store.add_audit_log(
        kind=audit_kind, detail=detail, approved=1, result=result[:RESULT_TRUNCATE_LEN]
    )
    registry.fire_on_call(name, arguments, result)
    out = finish(result, not errored and not result.startswith(("Error", "BLOCKED")), level)
    _emit_search_evidence(ctl)
    return out


def _context_bar(used: int, num_ctx: int) -> str:
    """A short 'how full is the context window' bar, in the same spirit as
    Claude Code's own status line - printed after a reply so context
    pressure is visible before Ollama silently starts dropping older
    messages to make room, instead of that only showing up as the agent
    mysteriously forgetting something from earlier in the chat."""
    if not used or num_ctx <= 0:
        return ""
    pct = min(used / num_ctx, 1.0)
    width = 12
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[dim]context: {bar} {pct * 100:.0f}% ({used:,}/{num_ctx:,} tokens)[/dim]"


def _print_final_reply(
    console: Console, persona_name: str, content: str, used_tokens: int = 0, num_ctx: int = 0
) -> None:
    """The one-shot (no tool involved) print path: nothing was streamed, so
    there's nothing to duplicate - print the prefix and the Markdown-rendered
    reply."""
    console.print(f"[bold cyan]{persona_name}>[/bold cyan] ", end="")
    if content:
        console.print(Markdown(content))
    else:
        console.print()
    bar = _context_bar(used_tokens, num_ctx)
    if bar:
        console.print(bar)


def _live_enabled(console: Console, client: Any, model: str) -> bool:
    """Write the reply live (streamed) instead of waiting for all of it? Only on a real
    terminal, only if the client can stream tool-aware calls, never for DeepSeek R1 (its
    <think> block is part of the content) and not if `ui.live` is off."""
    from agent.llm import is_deepseek_r1

    return bool(
        _ui_settings().get("live", True)
        and console.is_terminal
        and hasattr(client, "chat_with_tools_stream")
        and not is_deepseek_r1(model)
    )


def _stream_turn(
    client: Any, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]],
    num_ctx: int, live: Any, on_thinking: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """One streamed model call; text goes to `live` as it arrives. If the stream itself fails
    before anything was shown, fall back to the plain buffered call once."""
    extra: dict[str, Any] = {}
    if on_thinking and "on_thinking" in inspect.signature(client.chat_with_tools_stream).parameters:
        extra["on_thinking"] = on_thinking
    from agent.llm import GenerationAborted

    try:
        return client.chat_with_tools_stream(
            messages=messages, model=model, tools=tools, num_ctx=num_ctx, on_content=live.on_content, **extra
        )
    except GenerationAborted as stop:                 # a client that lets the abort through: keep what arrived
        return {"role": "assistant", "content": getattr(live, "text", ""), "_aborted": str(stop) or "stopped"}
    except (httpx.TransportError, ValueError):
        if live.started:
            raise
        return client.chat_with_tools(messages=messages, model=model, tools=tools, num_ctx=num_ctx)


class _ReplySink:
    """Fans the streamed reply out to the live terminal view (if any) and to `token` events (if any), and
    raises GenerationAborted as soon as `stop` is set. Used only when a TurnControl is active."""

    def __init__(self, ctl: TurnControl, live: Any) -> None:
        from agent.chat.turn import _TokenGate

        self._ctl, self._live = ctl, live
        self.text = ""
        self._gate = _TokenGate(lambda ev: ctl.event("token", text=ev["text"]))

    @property
    def started(self) -> bool:
        return (self._live.started if self._live is not None else False) or self._gate.state == "live"

    @property
    def tool_like(self) -> bool:
        return self._gate.state == "tool"

    @property
    def streamed(self) -> bool:
        return self._gate.state == "live"

    def on_content(self, piece: str) -> None:
        if self._ctl.stopped():
            from agent.llm import GenerationAborted

            raise GenerationAborted("stopped")
        self.text += piece
        self._gate.feed(piece)
        if self._live is not None:
            self._live.on_content(piece)

    def abandon(self) -> None:
        if self._live is not None:
            self._live.abandon()


def _prompt_estimate(messages: list[dict[str, Any]], tools: Any = None) -> int:
    from agent.chat.turn import prompt_estimate

    return prompt_estimate(messages, tools)


def _thinking_feed(ctl: TurnControl, cue: "_ThinkingCue") -> Callable[[str], None]:
    """on_thinking callback: stop check, the code cue, a status pulse (token estimate) and - only when
    `capture_thinking` is on - the reasoning text as `thinking` events. Never stored anywhere."""
    state = {"chars": 0, "last": 0.0}

    def feed(piece: str) -> None:
        if ctl.stopped():
            from agent.llm import GenerationAborted

            raise GenerationAborted("stopped")
        cue.feed(piece)
        state["chars"] += len(piece)
        if ctl.capture_thinking:
            ctl.event("thinking", text=piece)
        now = time.monotonic()
        if now - state["last"] >= 1.5:
            state["last"] = now
            ctl.event("status", state="thinking", label="thinking", detail="", tokens=state["chars"] // 4)

    return feed


def _context_payload(message: dict[str, Any], num_ctx: int, compacted: bool = False) -> dict[str, Any]:
    from agent.chat.turn import _context_event

    data = _context_event(message, num_ctx, compacted)
    data.pop("type", None)
    return data


def _run_agent_turn_core(
    client: Any,
    config: dict[str, Any],
    console: Console,
    store: Store,
    session_id: int,
    registry: ToolRegistry,
    messages: list[dict[str, Any]],
    persona_name: str,
    goal_state: dict[str, Any] | None = None,
    plan: Any = None,
    *,
    ctl: "TurnControl | None" = None,
    **control: Any,
) -> tuple[str | None, set[str]]:
    """Run the tool-calling loop for one user turn.

    Optional observation/control (agent.ui.chat_adapter.TurnControl; either pass `ctl` or its fields as
    keywords: emit, stop, approve_fn, suppress_print, capture_thinking, compact): events go to `emit`,
    `stop` ends the turn cleanly (ctl.interrupted/ctl.partial are set, the partial reply is returned),
    `approve_fn` answers approvals, `suppress_print` skips every direct print (events only). With none of
    them the output is exactly the classic terminal output.

    Persists every tool-call and tool-result message as it goes so resume
    replays correctly, and persists (but does not embed) them here; the
    final assistant text reply is persisted by the caller like before.
    Returns (final text reply or None if the iteration cap was hit, the
    names of every tool actually called this turn - used by the caller to
    decide whether a learned-fact check is worth running, and to skip that
    check for a turn that was only `remember_this`, which already saved
    exactly what was asked - re-extracting it would just duplicate it).

    Printing happens IN HERE, not in the caller: the right moment to show
    the "Agent>" prefix is right before the model actually starts speaking
    (either a direct answer, or the post-tool wrap-up) - not before tool
    execution, which has its own separate narration (e.g. web_search's
    "searching..." lines).
    """
    if ctl is None and control:
        ctl = TurnControl.build(**control)
    if ctl is not None and ctl.suppress_print:
        console = _quiet_console(console)
    model = config["models"]["chat"]
    tools_schema = registry.to_ollama_schema()
    max_iterations = config["tools"]["max_iterations"]
    num_ctx = _num_ctx(config, model)
    tools_called: set[str] = set()

    from agent.llm import is_deepseek_r1

    live_ok = _live_enabled(console, client, model)
    # events want the reply as it arrives, even where the classic path would not stream (no terminal)
    stream_ok = live_ok or bool(ctl is not None and (ctl.emit is not None or ctl.stop is not None)
                                and hasattr(client, "chat_with_tools_stream") and not is_deepseek_r1(model))

    def interrupted(partial: str = "") -> tuple[str | None, set[str]]:
        ctl.interrupted, ctl.partial = True, partial        # type: ignore[union-attr]
        console.print("[dim](interrupted)[/dim]")
        return (partial or None), tools_called

    watcher = (
        autonomy.StuckWatcher(max_lookups=int((config.get("autonomy") or {}).get("max_lookups_per_turn", 3)))
        if autonomy.enabled(config) and registry.get("web_search") else None
    )
    gate = gates.make_verify_gate(config, registry)
    goal = gates.GoalReminder(goal_state)
    steps = gates.StepBudget(max_iterations, lambda done: _ask_more_steps(console, done))
    iteration = 0
    while True:
        if ctl is not None and ctl.stopped():
            return interrupted()
        if not steps.may_continue(iteration):
            break
        iteration += 1
        live = None
        sink = None
        coding = bool(tools_called & {"write_file", "replace_in_file"}) or _looks_like_coding(_last_user_text(messages))
        if ctl is not None:
            ctl.event("status", state="thinking", label="waiting for model", detail=model, since=time.time(),
                      step=iteration, prompt_tokens=_prompt_estimate(messages, tools_schema))
        with _spinner(ctl, console, coding=coding) as spinner:
            if stream_ok:
                from agent.liveview import LiveReply

                if live_ok:
                    live = LiveReply(console, persona_name, spinner.stop)
                # The reasoning streams before a tool call does, so cues in it (a file name, "write_file",
                # code words) switch the status to coding as soon as the model is clearly about to write.
                cue = _ThinkingCue(spinner.set_coding)
                if ctl is not None:
                    sink = _ReplySink(ctl, live)
                    message = _stream_turn(client, messages, model, tools_schema, num_ctx, sink, _thinking_feed(ctl, cue))
                else:
                    message = _stream_turn(client, messages, model, tools_schema, num_ctx, live, cue.feed)
            else:
                message = client.chat_with_tools(
                    messages=messages, model=model, tools=tools_schema, num_ctx=num_ctx
                )
        if ctl is not None:
            ctl.event("context", **_context_payload(message, num_ctx))
            if message.get("_aborted") or (ctl.stopped() and sink is not None and not sink.streamed
                                           and _extract_tool_call(message) is not None):
                if live is not None:
                    live.finish("")
                partial = "" if sink is None or sink.tool_like else _strip_thinking_artifacts(
                    str(message.get("content") or (sink.text if sink else "")))
                return interrupted(partial)
        call = _extract_tool_call(message)

        if call is None:
            content = _strip_thinking_artifacts(message.get("content", ""))
            if not content.strip() and tools_called:
                # A local model can occasionally return truly empty content
                # with no tool_calls either - a stochastic blip, not a real
                # "nothing left to do" signal, especially with a task
                # clearly still mid-flight (at least one tool already ran
                # this turn). Accepting that as the final answer used to
                # end the turn completely silently. Nothing was appended to
                # `messages` from this blank attempt, so retrying with the
                # exact same context is a clean, free re-roll, bounded by
                # the same max_iterations cap as everything else here.
                if live is not None:
                    live.abandon()
                continue
            # Reuse the content THIS SAME tool-aware call already produced,
            # rather than firing a second, independent completion with no
            # tools schema attached purely for a nicer streamed-typing
            # effect (the previous approach, kept for a while for both
            # branches here). That second call was a fresh, unconstrained
            # generation, not a re-read of the same answer - so it could
            # drift into fabricating its OWN tool-call-shaped text (complete
            # with invented parameter names, since no schema was
            # constraining it), which then either ran with garbage
            # arguments or got printed to the terminal as raw JSON before
            # anyone could check it. `content` here already passed through
            # _extract_tool_call's own parse_tool_call_fallback check above
            # and definitely is NOT a tool call.
            if gate is not None and content.strip() and gate.due():
                retry = _run_verification(gate, registry, store, console, config, messages, content, ctl)
                if retry is not None:
                    if live is not None:
                        live.abandon()
                    messages = retry
                    continue
            decision = gates.PlanGate.check(plan, content)
            if decision is not None:
                console.print(decision.line)
                if ctl is not None:
                    ctl.event("plan", **decision.event)
                if decision.action == "nudge":
                    if live is not None:
                        live.abandon()
                    messages = decision.retry_messages(messages, content)
                    continue
            used = message.get("_prompt_eval_count", 0) + message.get("_eval_count", 0)
            if ctl is not None and content and (sink is None or not sink.streamed):
                ctl.event("token", text=content)            # nothing was streamed: the whole reply at once
            if live is not None and live.finish(_context_bar(used, num_ctx)):
                return content, tools_called      # already written live, nothing to print again
            if ctl is None or not ctl.suppress_print:
                _print_final_reply(console, persona_name, content, used, num_ctx)
            return content, tools_called

        if live is not None:
            live.abandon()                        # text came first, a tool call follows
        assistant_call_msg = {
            "role": "assistant",
            # Deliberate lossy-but-safe persistence: Store.get_messages only
            # returns role/content, so the tool call is flattened to a JSON
            # string rather than kept as structured tool_calls metadata.
            "content": json.dumps({"tool": call["name"], "arguments": call["arguments"]}),
        }
        if ctl is not None and ctl.stopped():
            return interrupted()                  # a tool call was announced but never ran
        result = _execute_tool(
            registry, store, console, config, call["name"], call["arguments"], ctl
        )
        if result == DENIED_RESULT:
            # Without this, the model treats "User denied execution." as just
            # another tool result and confidently tells the user the action
            # succeeded anyway - a real observed failure mode. Spell out that
            # nothing happened so the wrap-up reply is honest about it.
            tool_content = DENIAL_FRAMING
        else:
            tool_content = _frame_tool_result(call["name"], result)
        tool_msg = {"role": "tool", "content": tool_content}

        store.add_message(session_id, "assistant", assistant_call_msg["content"])
        store.add_message(session_id, "tool", result)

        messages = [*messages, assistant_call_msg, tool_msg]
        tools_called.add(call["name"])
        gates.note_clean_edit(gate, call, result)
        if watcher is not None:
            messages = _react_to_stuck(
                watcher, registry, console, messages, call, result, denied=result == DENIED_RESULT, ctl=ctl
            )
        messages = _maybe_compact(client, config, console, messages, message, num_ctx, ctl)

        # Goal restatement: every N tool calls, inject a reminder of the
        # original task so the model doesn't drift on long agentic runs.
        reminder = goal.tick()
        if reminder is not None:
            console.print(f"[dim]⋯ {reminder.log_text}[/dim]")
            if ctl is not None:
                ctl.event("log", text=reminder.log_text)
            messages = [*messages, reminder.message]

    console.print()
    console.print(MAX_ITERATIONS_HINT)
    if ctl is not None:
        ctl.event("error", message=Text.from_markup(MAX_ITERATIONS_HINT).plain, code="max_steps")
    return None, tools_called


_CHANGES = gates.CHANGES              # the shared per-prompt change log, undo stack and last-edited list live in gates
_UNDO = gates.UNDO
_LAST_EDITED = gates.LAST_EDITED      # files edited by the most recent prompt (to find the active project)
_PROJECT_HOOK: Any = None             # set by run_repl: called (check command, edited file) when a check passed


_CONTROL_KEYS = ("emit", "stop", "approve_fn", "suppress_print", "capture_thinking", "compact")


def _tracker() -> "gates.ChangeTracker":
    """The change tracker over this module's singletons (looked up now, so a test that swaps `_UNDO` is honoured)."""
    return gates.ChangeTracker(_CHANGES, _UNDO, _LAST_EDITED)


def _run_agent_turn(*args: Any, **kwargs: Any) -> tuple[str | None, set[str]]:
    """One prompt's worth of model/tool turns, then a summary of the files it created or edited.

    Optional keywords (see _run_agent_turn_core): `ctl=TurnControl(...)` or its fields `emit`, `stop`,
    `approve_fn`, `suppress_print`, `capture_thinking`, `compact`. With a control the turn also emits the
    closing `changes` and `done` events (and `error` when it raises)."""
    console = kwargs.get("console") or args[2]
    ctl = kwargs.pop("ctl", None)
    options = {k: kwargs.pop(k) for k in _CONTROL_KEYS if k in kwargs}
    if ctl is None and options:
        ctl = TurnControl.build(**options)
    if ctl is not None:
        kwargs["ctl"] = ctl
    tracker = _tracker()
    tracker.begin()
    gates.PlanGate.reset(kwargs.get("plan"))
    reply: str | None = None
    tools: set[str] = set()
    failed = False
    try:
        reply, tools = _run_agent_turn_core(*args, **kwargs)
        return reply, tools
    except BaseException as exc:
        failed = True
        if ctl is not None:
            from agent import friendly

            ctl.event("error", **friendly.error_fields(exc))     # never the raw httpx text (it carries a developer-docs URL)
        raise
    finally:
        tracker.finish()
        if ctl is not None:
            change_event = tracker.change_event()
            if change_event is not None:
                ctl.event("changes", **change_event)
            ctl.event("done", reply=(ctl.partial if ctl.interrupted else reply) or "", interrupted=ctl.interrupted,
                      tools=sorted(tools), failed=failed)
        if ctl is not None and ctl.suppress_print:
            tracker.clear()
        else:
            tracker.print_summary(console)


def _run_verification(
    gate: "verify.VerifyGate", registry: ToolRegistry, store: Store, console: Console, config: dict[str, Any],
    messages: list[dict[str, Any]], answer: str, ctl: "TurnControl | None" = None,
) -> list[dict[str, Any]] | None:
    """Terminal front end of gates.run_verification: the check runs through `_execute_tool` (risk gate included)."""
    return gates.run_verification(
        gate, lambda command: _execute_tool(registry, store, console, config, "run_command", {"command": command}, ctl),
        messages, answer, say=console.print, event=ctl.event if ctl is not None else gates._no_event,
        width=console.width, on_pass=lambda command, edited: _PROJECT_HOOK(command, edited) if _PROJECT_HOOK else None,
    )


def _maybe_compact(
    client: Any, config: dict[str, Any], console: Console, messages: list[dict[str, Any]],
    last_reply: dict[str, Any], num_ctx: int, ctl: "TurnControl | None" = None,
) -> list[dict[str, Any]]:
    """Fold the middle of a long tool loop into a work log once the window is ~70% full, so the
    run never reaches the point where the server silently drops the oldest tokens.

    With a `ctl` whose compact flag was requested (the /compact command) one mask + compact pass is
    forced here, at this tool-step boundary, whatever the fill level. The work is done by the shared
    implementation in agent.chat.compaction (also used by the web chat and the new terminal chat)."""
    from agent.chat.compaction import compact_context

    force = bool(ctl is not None and ctl.compact is not None and ctl.compact.take())
    return compact_context(
        client, config, messages, last_reply, num_ctx, event=ctl.event if ctl is not None else gates._no_event,
        say=console.print, force=force, spinner=lambda: _spinner(ctl, console, label="compacting"),
    )


def _insert_before_last_user(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Put a system block right before the user's newest message (so compaction keeps it in the head)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return [*messages[:i], {"role": "system", "content": text}, *messages[i:]]
    return [*messages, {"role": "system", "content": text}]


def _ask_more_steps(console: Console, done: int) -> bool:
    """The step budget is used up: on a real terminal let the user extend it instead of
    throwing the unfinished work away."""
    if not console.is_terminal or os.environ.get("AGENT_NO_CONTINUE_PROMPT"):
        return False
    try:
        answer = console.input(f"[yellow]{done} Schritte gemacht, Aufgabe noch offen.[/yellow] Weiter? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ("y", "yes", "j", "ja")


def setup_chat(
    client: Any, config: dict[str, Any], console: Console, store: Store, session_id: int | None = None,
    vectors: VectorIndex | None = None, tool_registry: ToolRegistry | None = None,
) -> "ChatState":
    """Everything `run_repl` does before its first prompt: web/memory/plugin/note/plan/project tools, the session
    (created, or `session_id` resumed), the project and trajectory hooks. Shared with the terminal UI
    (agent.ui.chat_app) so both start a chat identically. `console` is only used to bind tools' narration."""
    from agent.ui.chat_adapter import ChatState

    registry = tool_registry if tool_registry is not None else default_registry(config)
    tool_registry = registry
    if vectors is None:
        vectors = VectorIndex(store)

    # Live web search, bound to the console so it narrates each site as it visits.
    from agent.tools.web import make_web_tools

    web_memory = None
    try:
        from agent.research.memory import WebMemory

        web_memory = WebMemory(
            store.conn, vectors, embed=lambda texts: client.embed(texts, model=config["models"]["embed"])
        )
    except Exception:  # noqa: BLE001 - research still works without its cache
        logger.warning("web research cache unavailable", exc_info=True)
    for web_tool in make_web_tools(console, web_memory):
        tool_registry.register(web_tool)

    tool_registry.register(_make_remember_tool(store, vectors, client, config, console))
    tool_registry.register(_make_save_skill_tool(store, console))

    from agent.tools.present import make_present_tool

    from agent.tools.calc import CALC_TOOL

    tool_registry.register(CALC_TOOL)
    tool_registry.register(make_present_tool(console, allow_html=bool((config.get("ui") or {}).get("allow_html_blocks"))))

    # Optional extensions from ~/.config/agent/plugins (see agent/plugins.py).
    from agent.plugins import PluginContext, load_plugins

    loaded = load_plugins(
        PluginContext(
            registry=tool_registry, store=store, vectors=vectors,
            llm=client, config=config, console=console,
        )
    )
    if loaded:
        console.print(f"[dim]plugins: {', '.join(loaded)}[/dim]")

    resuming = session_id is not None
    title_pending = not resuming
    if session_id is None:
        session_id = store.create_session()

    tool_registry.register(_make_take_note_tool(store, session_id, console))
    from agent.tools.deep_research import make_deep_research_tool

    tool_registry.register(make_deep_research_tool(client, config, console, store, session_id))
    plan_holder = planmod.Holder()
    tool_registry.register(planmod.make_tool(plan_holder, console))
    project_memory = projectmem.ProjectMemory(store.conn)
    active_project = projectmem.ActiveProject()
    tool_registry.register(projectmem.make_tool(project_memory, active_project, console))

    state = ChatState(
        client=client, config=config, store=store, registry=tool_registry, vectors=vectors, console=console,
        session_id=session_id, model=config["models"]["chat"], persona_name=config["persona"]["name"],
        plan_holder=plan_holder, active_project=active_project, project_memory=project_memory, title_pending=title_pending,
    )
    state.reset_goal()

    def _remember_check(command: str, edited_path: str) -> None:
        key = projectmem.project_key(edited_path)
        if key:
            project_memory.add(key, f"Check that passes: {command.split(' && ', 1)[-1]}", kind="verify")

    global _PROJECT_HOOK
    _PROJECT_HOOK = _remember_check

    # Trajectory hook: log every tool call to ~/.companion/trajectory.jsonl.
    # Uses a closure so the hook always captures the current session_id even
    # after /clear rebinds it.
    def _trajectory_hook(tool_name: str, arguments: dict[str, Any], result: str) -> None:
        from agent.redact import redact
        from agent.trajectory import log_action
        log_action(
            action="tool_call",
            tool=tool_name,
            input_summary=redact(json.dumps(arguments, ensure_ascii=False)[:400]),
            output_summary=redact(result[:400]),
            session_id=state.session_id,
        )

    tool_registry.set_on_call(_trajectory_hook)
    state.session_id = session_id
    return state



def start_memory_worker(state: "ChatState", config: dict[str, Any]) -> Any:
    """Start the background fact saver for this chat (see agent.memory.queue): it first catches up on whatever an
    earlier chat left in the queue, then takes each turn's messages while the model is idle. Returns it (also kept
    on `state.distiller`), or None when the store has no file to open a second connection on."""
    from agent.memory.queue import BackgroundDistiller

    db_path = getattr(state.store, "db_path", None)
    if not db_path:
        return None
    try:
        state.distiller = BackgroundDistiller(config, state.client, db_path, session_id=state.session_id).start()
    except Exception as exc:  # noqa: BLE001 - remembering must never stop a chat from starting
        logger.warning("Could not start the memory worker: %s", exc)
        state.distiller = None
    return state.distiller


def start_fresh_session(state: "ChatState", tool_exchanges: list[tuple]) -> int:
    """/clear: queue the finished session's notes (the memory worker takes them in the background), open a new
    session and rebind the session-bound tools. Resets the state's per-session fields (title, last reply, goal,
    plan) and returns the new session id."""
    from agent.memory.queue import enqueue_exchanges

    try:
        enqueue_exchanges(state.store, state.session_id, tool_exchanges)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not queue the session's notes: %s", exc)
    state.session_id = state.store.create_session()
    if state.distiller is not None:
        state.distiller.session_id = state.session_id
        state.distiller.notify()
    state.registry.register(_make_take_note_tool(state.store, state.session_id, state.console))
    from agent.tools.deep_research import make_deep_research_tool

    state.registry.register(make_deep_research_tool(state.client, state.config, state.console, state.store, state.session_id))
    state.title_pending, state.last_reply = True, None
    state.tool_exchanges = []
    state.reset_goal()
    if state.plan_holder is not None:
        state.plan_holder.plan = None
    return state.session_id


def run_repl(
    client: Any,
    config: dict[str, Any],
    console: Console | None = None,
    store: Store | None = None,
    session_id: int | None = None,
    vectors: VectorIndex | None = None,
    tool_registry: ToolRegistry | None = None,
) -> None:
    """Run the interactive chat loop until the user quits or input ends.

    If `store` is provided, a session is created (or resumed via
    `session_id`), every message is persisted and embedded, each turn's
    context is assembled from persona + facts + semantic recall, and durable
    facts are distilled from the transcript when the session ends.
    """
    from agent import friendly

    friendly.quiet_logging()
    console = console or Console()
    model = config["models"]["chat"]
    persona_name = config["persona"]["name"]
    system_prompt = config["persona"]["style"]
    tool_registry = tool_registry if tool_registry is not None else default_registry(config)

    # Without a store there is no memory; fall back to the simple in-memory loop.
    if store is None:
        _run_repl_stateless(client, config, console, model, persona_name, system_prompt)
        return

    resuming = session_id is not None
    rt = setup_chat(client, config, console, store, session_id, vectors, tool_registry)
    vectors, session_id = rt.vectors, rt.session_id
    title_pending = rt.title_pending
    plan_holder, project_memory, active_project = rt.plan_holder, rt.project_memory, rt.active_project
    if resuming:
        history = store.get_messages(session_id)
        if history:
            print_resume_preview(console, history)

    console.print(f"[bold cyan]{persona_name}[/bold cyan] is ready. Type /help for commands.")
    _warn_if_model_cannot_use_tools(console, client, model)
    memory_worker = start_memory_worker(rt, config)

    pending_task: str | None = None  # a "work in this repo: ..." message waiting for an access level
    last_reply: str | None = None  # what a "you were wrong" would be about
    tool_exchanges: list[tuple] = []  # checked for durable facts at session end
    goal_state: dict[str, Any] = {
        "total_calls": 0,
        "every_n": config.get("goal_restatement_every_n_actions", 10),
        "original_task": None,
    }
    while True:
        if memory_worker is not None:
            memory_worker.busy(False)          # waiting for you: the model is idle, facts get saved now
        try:
            user_input = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if memory_worker is not None:
            memory_worker.busy(True)

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/help":
            console.print(HELP_TEXT)
            continue
        if user_input == "/model":
            model = _handle_model_switch(client, config, console, model)
            continue
        if user_input == "/clear":
            session_id = start_fresh_session(rt, tool_exchanges)
            title_pending, last_reply, tool_exchanges = True, None, []
            goal_state = {
                "total_calls": 0,
                "every_n": config.get("goal_restatement_every_n_actions", 10),
                "original_task": None,
            }
            plan_holder.plan = None
            console.print("[dim]Started a fresh conversation - nothing earlier carries over.[/dim]")
            continue
        if user_input == "/notes":
            notes = store.get_working_notes(session_id)
            if not notes:
                console.print("[dim]No notes taken this session.[/dim]")
            else:
                console.print(f"[dim]Notes this session:[/dim]\n{notes}")
            continue
        if user_input == "/notes clear":
            store.clear_working_notes(session_id)
            console.print("[dim]Notes cleared.[/dim]")
            continue

        if user_input == "/verify":
            search_tool = tool_registry.get("web_search")
            if not last_reply or search_tool is None:
                console.print("[dim]verify  nothing to check yet (ask something first)[/dim]")
                continue
            from agent.research import verify as verifymod

            def _run_search(query: str) -> tuple[list[dict], dict]:
                search_tool.handler(query=query, num_results=4)
                found = turn_evidence.last_search()
                return found["items"], found["verdict"]

            try:
                with _ThinkingSpinner(console, label="verifying"):
                    checked = verifymod.verify(last_reply, _run_search)
            except Exception as exc:  # noqa: BLE001 - a failed check must not end the chat
                console.print(f"[red]verify failed:[/red] {escape(str(exc))}")
                continue
            console.print(escape(verifymod.summarize(checked)))
            continue
        if user_input in ("/undo", "/undo list"):
            for line in _tracker().undo_lines(user_input.endswith("list")):
                console.print(f"[dim]{escape(line)}[/dim]")
            continue
        if user_input in ("/plan", "/plan clear"):
            if user_input == "/plan clear":
                plan_holder.plan = None
                console.print("[dim]plan  cleared[/dim]")
            elif plan_holder.plan is None:
                console.print("[dim]plan  none for this task (one is made for big build tasks)[/dim]")
            else:
                for line in plan_holder.plan.lines():
                    console.print(f"[dim]{escape(line)}[/dim]")
            continue
        if user_input in ("/project", "/project clear"):
            if not active_project.key:
                console.print("[dim]project  none active yet (edit a file in a repo or say 'work in this repo: <path>')[/dim]")
            elif user_input == "/project clear":
                console.print(f"[dim]project  forgot {project_memory.clear(active_project.key)} note(s)[/dim]")
            else:
                console.print(f"[dim]project  {escape(active_project.key)}[/dim]")
                for _nid, note in project_memory.notes(active_project.key):
                    console.print(f"[dim]  - {escape(note)}[/dim]")
            continue

        resumed = False
        handled = repoaccess.handle_command(user_input)
        if handled is not None:
            kind, message = handled
            _m = repoaccess.COMMAND_RE.match(user_input.strip())
            if _m and _m.group(2) and kind in (repoaccess.AUTO, repoaccess.MEDIUM, "greenlist"):
                _folder, _ = repoaccess.resolve_folder(_m.group(2))
                if _folder is not None:
                    active_project.set_from(_folder)
            console.print(f"[green]{escape(message)}[/green]" if kind != "access" else escape(message))
            if pending_task and kind in (repoaccess.AUTO, repoaccess.MEDIUM, repoaccess.ASK):
                user_input, pending_task = pending_task, None      # level chosen: carry on with the task
                resumed = True
            else:
                continue
        elif pending_task:
            pending_task = None                                     # asked something else: default level
        work_dir = None if resumed else repoaccess.parse_work_in(user_input)
        if work_dir and pending_task is None:
            folder, note = repoaccess.resolve_folder(work_dir)
            if folder is not None:
                active_project.set_from(folder)
            if folder is not None and repoaccess.mode_for(folder) == repoaccess.ASK:
                console.print(escape(repoaccess.question(folder, note)))
                pending_task = user_input
                continue
            if folder is not None:
                console.print(f"[dim]{folder}: {repoaccess.mode_for(folder)} access already granted[/dim]")

        user_input, web_forced = _split_web_trigger(user_input)
        if web_forced and not user_input:
            console.print("[yellow]Add something to search for before /web.[/yellow]")
            continue

        # Track the very first user message as the session goal so the
        # restatement reminder can reference it on long agentic runs.
        if goal_state["original_task"] is None:
            goal_state["original_task"] = user_input

        taint.clear()                         # what the web said earlier is not what this turn read
        turn_evidence.clear()
        tool_names = [t.name for t in tool_registry.list()]
        with console.status(f"[dim]{random.choice(_MEMORY_STATUS_LINES)}[/dim]"):
            messages = build_context(
                store, vectors, client, config, session_id, user_input, tool_names=tool_names
            )
        user_msg_id = store.add_message(session_id, "user", user_input)

        if gates.plan_creation_enabled(user_input, config):
            with _ThinkingSpinner(console, label="planning"):
                new_plan = planmod.generate(client, config, user_input)
            if new_plan is not None:
                plan_holder.plan = new_plan
                console.print(f"[dim]plan  {len(new_plan.items)} checks to confirm (/plan shows them)[/dim]")
        extra_context = []
        if plan_holder.plan is not None:
            extra_context.append(plan_holder.plan.system_message())
        if active_project.key:
            note_block = project_memory.system_message(active_project.key)
            if note_block:
                extra_context.append(note_block)
        for block in extra_context:
            messages = _insert_before_last_user(messages, block)

        web_was_forced = web_forced
        # Explicit /web: run the search now (deterministic — doesn't rely on the
        # model choosing to call the tool) and feed the results into this turn.
        if web_forced:
            web = tool_registry.get("web_search")
            search_query = _rewrite_search_query(user_input, client, config)
            if search_query != user_input:
                console.print(f"[dim]{_mood('searching')}searching for: {search_query}[/dim]")
            results = web.handler(query=search_query) if web else "web search unavailable"
            results = results[:20000]  # keep in sync with agent/tools/web.py's
            # _PAGE_CHARS * 5 - this forced path doesn't pass num_results, so
            # it always reads web_search's default of 5 pages regardless of
            # the higher _HARD_MAX_READ ceiling an autonomous tool call can opt into
            # Persist the sources like any other tool result. Without this they
            # only existed for one turn: the model answered correctly from the
            # web, then on the next question had nothing but its own (wrong)
            # training memory and confidently made something up.
            store.add_message(
                session_id, "tool", f"Web search results for {search_query!r}:\n{results}"
            )
            messages = [*messages, {"role": "system", "content": _WEB_INJECT + results}]
        elif (
            last_reply
            and tool_registry.get("web_search")
            and _looks_like_disagreement(user_input)
            and _is_factual_correction(client, config, last_reply, user_input)
        ):
            # The cheap cue keeps the classifier off the ~90% of turns that
            # couldn't be a correction, so a normal message costs nothing extra.
            messages = [*messages, {"role": "system", "content": _CORRECTION_NUDGE}]

        try:
            reply, tools_used = _run_agent_turn(
                client, config, console, store, session_id, tool_registry, messages,
                persona_name, goal_state=goal_state, plan=plan_holder,
            )
            if _LAST_EDITED:
                active_project.set_from(_LAST_EDITED[0])
        except httpx.HTTPError as exc:
            console.print()
            console.print(_http_error_message(exc))
            continue

        store.touch_session(session_id)
        if title_pending:
            store.set_session_title(session_id, user_input[:TITLE_MAX_LEN])
            title_pending = False
        _embed_message(vectors, client, config, user_msg_id, user_input)

        if reply is None:
            # Iteration cap was hit; _run_agent_turn already told the user.
            continue

        reply = _strip_thinking_artifacts(reply)
        last_reply = reply
        assistant_msg_id = store.add_message(session_id, "assistant", reply)
        _embed_message(vectors, client, config, assistant_msg_id, reply)

        # remember_this already saved exactly what was asked, in full - if
        # it was the ONLY tool this turn, re-extracting a fact from the same
        # exchange at session end would just duplicate it. A turn that also
        # used another tool alongside it is still worth checking.
        worth_checking = tools_used - {"remember_this"}
        if worth_checking or web_was_forced:
            # Checked for a savable fact at session end, not now - a
            # background-model call mid-conversation is a pause the user
            # feels on every tool-using turn, for a save that isn't urgent.
            tool_exchanges.append((user_input, reply, turn_evidence.take()))

    _distill_on_exit(store, vectors, client, config, session_id, console, tool_exchanges, distiller=memory_worker)


_FACT_EXTRACT_SYSTEM = (
    "Given this exchange, is there one specific, durable fact worth "
    "remembering for later - like a product spec, a date, a name, or a "
    "number? Only say yes if it's the kind of thing that would still be "
    "useful weeks from now, and it's not about the user personally or "
    "about their computer, files, or commands. Say NONE if the search "
    "found nothing, if the answer is just 'I don't know' or a failed "
    "lookup, or if the only notable thing is which tool or app was used "
    "to find it - that's not a fact worth keeping. If sources are given, the "
    "fact must be stated by the sources themselves, not only by the answer: "
    "if the answer says something the sources do not, reply NONE. If yes, reply with "
    "ONLY that one fact as a single complete sentence that names its own "
    "subject and stands on its own with no other context - someone "
    "reading it alone, weeks later, with no memory of this conversation, "
    "must be able to tell exactly what it's about. Never reply with just "
    "a bare number, date, or name on its own (e.g. '1889' or '300 ohms') "
    "- always spell out what it refers to (e.g. 'The Eiffel Tower was "
    "completed in 1889.'). Never mention the search, the tool, or how it "
    "was found. If no, reply with exactly: NONE"
)

# URLs are fine in a fact the user states about themselves, but a learned fact should stand on its own text
_URL_RE = re.compile(r"https?://", re.IGNORECASE)


def _save_learned_facts(
    client: Any,
    config: dict[str, Any],
    store: Store,
    vectors: VectorIndex,
    tool_exchanges: list[tuple],
    strict: bool = False,
) -> None:
    """Check each tool-using exchange from the finished session for one durable fact worth remembering, and save any
    found the way `agent memory add` does (store.add_fact() then vectors.add("fact", ...)).

    An exchange is `(question, reply)` or `(question, reply, evidence)`. When web research is attached (see
    agent.research.evidence) the fact is only saved if the evidence was verified or likely - a single, weak or
    conflicting source never becomes memory - and the extractor is shown the source passages, so it can only keep what
    a source actually says (it used to see just the model's own reply and could store a hallucination as a fact). The
    saved sentence carries its origin: `(web, 2026-10-18, docs.python.org)`.

    This runs once at session end, alongside `distill_session`: a background-model call mid-chat is a pause the user
    feels on every such turn, for a save that isn't urgent. One bad exchange is logged and skipped. Recall needs no
    change: `_semantic_hits()` searches all memory_items, so a saved fact is findable the next time."""
    from agent.research.evidence import SAVEABLE

    for exchange in tool_exchanges:
        user_input, reply = exchange[0], exchange[1]
        evidence = exchange[2] if len(exchange) > 2 else None
        try:
            if evidence is not None and evidence.get("label") not in SAVEABLE:
                continue
            transcript = f"Question: {user_input}\n\nAnswer: {reply[:1500]}"
            if evidence:
                transcript += "\n\nSources:\n" + "\n".join(
                    f"- {s['domain']} ({s.get('date') or 'undated'}): {s['passage'][:600]}" for s in evidence["sources"])
            extracted = "".join(
                client.chat(
                    messages=[
                        {"role": "system", "content": _FACT_EXTRACT_SYSTEM},
                        {"role": "user", "content": transcript},
                    ],
                    model=config["models"]["background"],
                    stream=False,
                )
            ).strip()

            if not extracted or extracted.upper() == "NONE" or JUNK_FACT_RE.search(extracted) or _URL_RE.search(extracted):
                continue

            if evidence:
                origin = ", ".join(dict.fromkeys(s["domain"] for s in evidence["sources"][:3]))
                extracted += f" (web, {date.today().isoformat()}, {origin})"

            if any(f["content"] == extracted for f in store.get_active_facts()):
                continue                 # a retried exchange (crash, or queue replay) never saves the same fact twice

            # "incidental": this came from a tool call, not from the user asking to remember something or talking
            # about themselves - only worth surfacing when it's actually relevant, not every turn.
            fact_id = store.add_fact(extracted, source_session_id=None, kind="incidental",
                                     confidence="verified" if not evidence or evidence["label"] == "verified" else "assumed")
            if vectors.available:
                embedding = client.embed([extracted], model=config["models"]["embed"])[0]
                vectors.add("fact", fact_id, extracted, embedding)
        except EmbeddingUnavailable:
            pass                 # the fact is stored; it is just not searchable by meaning right now
        except Exception as exc:  # noqa: BLE001 - one bad exchange shouldn't sink the rest
            if strict:
                raise                # the durable queue keeps the exchange and retries it later
            logger.warning("Learned-fact save failed: %s", exc)


def _distill_on_exit(
    store: Store,
    vectors: VectorIndex,
    client: Any,
    config: dict[str, Any],
    session_id: int,
    console: Console,
    tool_exchanges: list[tuple],
    distiller: Any = None,
) -> None:
    """Wrap up a finished chat WITHOUT waiting for a model. Facts were already saved while chatting; what is left
    (messages and tool exchanges the memory worker has not reached) stays in the durable queue and is finished at
    the next start or by `nightage memory process`. This only stops the worker (a step already running gets
    `memory.exit_wait_seconds` at most), queues the tool exchanges and prints one dim line. See agent.memory.queue."""
    from agent.memory.queue import finish

    finish(distiller, store, session_id, tool_exchanges, config, console)


def _run_repl_stateless(
    client: Any,
    config: dict[str, Any],
    console: Console,
    model: str,
    persona_name: str,
    system_prompt: str,
) -> None:
    """Chat loop with no persistence or memory (used when no store is given)."""
    session = ChatSession(client=client, model=model, system_prompt=system_prompt)
    console.print(f"[bold cyan]{persona_name}[/bold cyan] is ready. Type /help for commands.")
    _warn_if_model_cannot_use_tools(console, client, model)

    while True:
        try:
            user_input = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/help":
            console.print(HELP_TEXT)
            continue

        console.print(f"[bold cyan]{persona_name}>[/bold cyan] ", end="")
        try:
            session.send(user_input, on_token=lambda tok: console.print(tok, end=""))
            console.print()
        except httpx.HTTPError as exc:
            console.print()
            console.print(_http_error_message(exc))
            session.history.pop()  # drop the unanswered user turn
            continue
