"""Terminal chat REPL: Rich console, streamed replies, /quit and /help."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import json
import logging
import re

import httpx
from rich.console import Console

from agent.memory.distill import distill_session
from agent.memory.recall import build_context
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex
from agent.safety import RiskLevel, block_reason, classify_command
from agent.tools.files import safe_text
from agent.tools.registry import Tool, ToolRegistry, default_registry, parse_tool_call_fallback

logger = logging.getLogger(__name__)

TITLE_MAX_LEN = 60
RESULT_TRUNCATE_LEN = 2000
RESUME_PREVIEW_MESSAGES = 6

# Context window for chat. The recall context budget is ~24k chars (~6k tokens)
# plus persona, facts, and tool schemas, which overflows Ollama's ~4k default
# and gets silently truncated. 8192 fits the whole prompt.
CHAT_NUM_CTX = 8192

HELP_TEXT = (
    "Commands:\n"
    "  /help  Show this help text\n"
    "  /quit  Exit the chat\n"
    "\n"
    "End any message with /web (or /search) to force a live web search for it,\n"
    "e.g. 'latest news on the mars mission /web'. The agent can also search on\n"
    "its own when a question needs current info."
)

# Trailing tokens that force a live web search for the message.
_WEB_TRIGGERS = {"/web", "/search", "/s", "/w", "/net", "/online"}

_WEB_INJECT = (
    "The user asked for a web-researched answer. Below are LIVE search results "
    "(titles, URLs, and page text) fetched just now. Write a detailed, "
    "well-structured answer grounded in them, and cite the source URLs you used. "
    "If the results don't cover it, say so.\n\n"
)


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

CONNECTION_HINT = (
    "[red]Could not reach Ollama.[/red] Is it running? Try: [bold]ollama serve[/bold]"
)

MAX_ITERATIONS_HINT = (
    "[red]Couldn't finish[/red]: too many tool calls in a row. Try rephrasing."
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
            "[dim]Use a tools-capable model:  agent settings set chat_model qwen2.5:7b[/dim]"
        )


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
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        return f"[red]Ollama error {response.status_code}:[/red] {detail}"
    return CONNECTION_HINT


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
            messages=self.history, model=self.model, stream=True, num_ctx=CHAT_NUM_CTX
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


def _extract_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    """Return {"name": ..., "arguments": ...} from a chat_with_tools() reply,
    using Ollama's structured tool_calls if present, else the JSON fallback
    parser on the message content. Returns None if neither yields a call.
    """
    tool_calls = message.get("tool_calls")
    if tool_calls:
        function = tool_calls[0].get("function", {})
        return {"name": function.get("name"), "arguments": function.get("arguments", {})}

    content = message.get("content", "")
    fallback = parse_tool_call_fallback(content) if content else None
    if fallback:
        return {"name": fallback["tool"], "arguments": fallback["arguments"]}
    return None


DENIED_RESULT = "User denied execution."


def _static_risk_level(risk: str) -> RiskLevel:
    """Map a static Tool.risk string to a RiskLevel for the gate.

    Unknown/unspecified risk values escalate to CAUTION (default-deny): a tool
    that forgot to declare its risk still gets a confirmation prompt rather than
    silently auto-running.
    """
    mapping = {
        "safe": RiskLevel.SAFE,
        "caution": RiskLevel.CAUTION,
        "dangerous": RiskLevel.DANGEROUS,
        "blocked": RiskLevel.BLOCKED,
    }
    return mapping.get(risk, RiskLevel.CAUTION)


def _classify_call(
    tool: Tool, arguments: dict[str, Any], config: dict[str, Any]
) -> RiskLevel:
    """Determine the risk tier for a specific tool call.

    Dynamic-risk tools (the shell tool) are classified per-command; every other
    tool uses its declared static risk.
    """
    if tool.risk == "dynamic":
        command = str(arguments.get("command", ""))
        return classify_command(command, config)
    return _static_risk_level(tool.risk)


def _approve(
    console: Console, level: RiskLevel, description: str
) -> bool:
    """Prompt the user for approval based on the risk tier. Returns True to run.

    SAFE never reaches here (auto-runs). CAUTION is a y/N (default No).
    DANGEROUS requires the user to type the literal word `yes`.
    """
    if level is RiskLevel.CAUTION:
        console.print(
            f"[yellow]Run this?[/yellow]\n  [bold]{description}[/bold]"
        )
        answer = console.input("[yellow]Run this? [y/N] [/yellow]").strip().lower()
        return answer == "y"

    # DANGEROUS
    console.print(
        f"[red]DANGEROUS command — type 'yes' to run:[/red]\n  [bold]{description}[/bold]"
    )
    answer = console.input("[red]Type 'yes' to confirm: [/red]").strip()
    return answer == "yes"


def _execute_tool(
    registry: ToolRegistry,
    store: Store,
    console: Console,
    config: dict[str, Any],
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Gate, run, and audit a tool call.

    Every tool call is risk-classified and gated before execution: SAFE runs
    silently, CAUTION/DANGEROUS require approval, BLOCKED never runs. Every
    decision (run, deny, block) writes an audit row. Handler errors are caught
    and fed back to the model.
    """
    tool = registry.get(name)
    if tool is None:
        result = f"Error: no such tool '{name}'."
        store.add_audit_log(
            kind="tool",
            detail=json.dumps({"name": name, "arguments": arguments}),
            approved=0,
            result=result,
        )
        return result

    is_shell = tool.risk == "dynamic"
    audit_kind = "shell" if is_shell else "tool"
    command = str(arguments.get("command", ""))
    detail = command if is_shell else json.dumps({"name": name, "arguments": arguments})

    level = _classify_call(tool, arguments, config)

    # BLOCKED: never runs.
    if level is RiskLevel.BLOCKED:
        reason = block_reason(command, config) if is_shell else "tool declared blocked"
        result = f"BLOCKED by safety policy: {reason}"
        console.print(f"[red]{result}[/red]")
        store.add_audit_log(kind=audit_kind, detail=detail, approved=0, result=result)
        return result

    # Approval gate for CAUTION / DANGEROUS.
    if level is not RiskLevel.SAFE:
        description = command if is_shell else name
        if not _approve(console, level, description):
            console.print("[dim](denied)[/dim]")
            store.add_audit_log(
                kind=audit_kind, detail=detail, approved=0, result=DENIED_RESULT
            )
            return DENIED_RESULT

    # Approved (or SAFE): run it.
    try:
        result = tool.handler(**arguments)
    except Exception as exc:  # noqa: BLE001 - handler errors go back to the model
        result = f"Error running tool '{name}': {exc}"

    # Backstop: a result that can't be encoded as UTF-8 kills the whole session
    # when it's sent to Ollama as JSON. Filenames are bytes and a single legacy
    # one (a music folder named `Alizée` in latin-1) was enough. The tools clean
    # their own output, but nothing reaching the model should be able to do this.
    result = safe_text(result)

    if is_shell and level is RiskLevel.SAFE:
        console.print(f"[dim][ran: {command}][/dim]")

    store.add_audit_log(
        kind=audit_kind, detail=detail, approved=1, result=result[:RESULT_TRUNCATE_LEN]
    )
    return result


def _run_agent_turn(
    client: Any,
    config: dict[str, Any],
    console: Console,
    store: Store,
    session_id: int,
    registry: ToolRegistry,
    messages: list[dict[str, Any]],
) -> str | None:
    """Run the tool-calling loop for one user turn.

    Persists every tool-call and tool-result message as it goes so resume
    replays correctly, and persists (but does not embed) them here; the
    final assistant text reply is persisted by the caller like before.
    Returns the final text reply, or None if the iteration cap was hit.
    """
    model = config["models"]["chat"]
    tools_schema = registry.to_ollama_schema()
    max_iterations = config["tools"]["max_iterations"]

    for _ in range(max_iterations):
        message = client.chat_with_tools(
            messages=messages, model=model, tools=tools_schema, num_ctx=CHAT_NUM_CTX
        )
        call = _extract_tool_call(message)

        if call is None:
            return message.get("content", "")

        assistant_call_msg = {
            "role": "assistant",
            # Deliberate lossy-but-safe persistence: Store.get_messages only
            # returns role/content, so the tool call is flattened to a JSON
            # string rather than kept as structured tool_calls metadata.
            "content": json.dumps({"tool": call["name"], "arguments": call["arguments"]}),
        }
        result = _execute_tool(
            registry, store, console, config, call["name"], call["arguments"]
        )
        tool_msg = {"role": "tool", "content": result}

        store.add_message(session_id, "assistant", assistant_call_msg["content"])
        store.add_message(session_id, "tool", result)

        messages = [*messages, assistant_call_msg, tool_msg]

    console.print()
    console.print(MAX_ITERATIONS_HINT)
    return None


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
    console = console or Console()
    model = config["models"]["chat"]
    persona_name = config["persona"]["name"]
    system_prompt = config["persona"]["style"]
    registry_provided = tool_registry is not None
    tool_registry = tool_registry if registry_provided else default_registry(config)

    # Without a store there is no memory; fall back to the simple in-memory loop.
    if store is None:
        _run_repl_stateless(client, config, console, model, persona_name, system_prompt)
        return

    if vectors is None:
        vectors = VectorIndex(store)

    # Live web search, bound to the console so it narrates each site as it visits.
    from agent.tools.web import make_web_tools

    for web_tool in make_web_tools(console):
        tool_registry.register(web_tool)

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

    if resuming:
        history = store.get_messages(session_id)
        if history:
            print_resume_preview(console, history)

    console.print(f"[bold cyan]{persona_name}[/bold cyan] is ready. Type /help for commands.")
    _warn_if_model_cannot_use_tools(console, client, model)

    last_reply: str | None = None  # what a "you were wrong" would be about
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

        user_input, web_forced = _split_web_trigger(user_input)
        if web_forced and not user_input:
            console.print("[yellow]Add something to search for before /web.[/yellow]")
            continue

        tool_names = [t.name for t in tool_registry.list()]
        messages = build_context(
            store, vectors, client, config, session_id, user_input, tool_names=tool_names
        )
        user_msg_id = store.add_message(session_id, "user", user_input)

        # Explicit /web: run the search now (deterministic — doesn't rely on the
        # model choosing to call the tool) and feed the results into this turn.
        if web_forced:
            web = tool_registry.get("web_search")
            results = web.handler(query=user_input) if web else "web search unavailable"
            results = results[:6000]
            # Persist the sources like any other tool result. Without this they
            # only existed for one turn: the model answered correctly from the
            # web, then on the next question had nothing but its own (wrong)
            # training memory and confidently made something up.
            store.add_message(
                session_id, "tool", f"Web search results for {user_input!r}:\n{results}"
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

        console.print(f"[bold cyan]{persona_name}>[/bold cyan] ", end="")
        try:
            reply = _run_agent_turn(
                client, config, console, store, session_id, tool_registry, messages
            )
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

        console.print(reply)
        last_reply = reply
        assistant_msg_id = store.add_message(session_id, "assistant", reply)
        _embed_message(vectors, client, config, assistant_msg_id, reply)

    _distill_on_exit(store, vectors, client, config, session_id, console)


def _distill_on_exit(
    store: Store,
    vectors: VectorIndex,
    client: Any,
    config: dict[str, Any],
    session_id: int,
    console: Console,
) -> None:
    """Extract facts from the finished session, showing a small spinner."""
    try:
        with console.status("[dim]remembering...[/dim]"):
            distill_session(store, vectors, client, config, session_id)
    except Exception as exc:  # noqa: BLE001 - remembering must never crash on exit
        logger.warning("Fact distillation failed: %s", exc)


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
