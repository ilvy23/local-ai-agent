"""Structured compaction for long, autonomous runs.

A local model has a fixed context window (65k tokens here). Without help, a long task either
hits the limit - and the server silently drops the oldest tokens, including the instructions -
or the agent keeps re-reading its own history. This module folds the *middle* of a running tool
loop into one structured work log (goal / done / state / decisions / open / pitfalls) once the
window is mostly full, and keeps the original task and the most recent steps verbatim.

The summary is written by the background model (`models.background`, the same 27B as the chat,
without thinking) and is deliberately concrete: exact file paths, names, commands and error
messages survive, prose does not.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.70        # compact when this fraction of the window is used
KEEP_LAST_MESSAGES = 6          # most recent loop messages stay verbatim
MIN_MIDDLE_MESSAGES = 4         # not worth a model call for less
PER_MESSAGE_CHARS = 2400        # head+tail of each old message fed to the summarizer
WORKLOG_MARKER = "WORK LOG"

WORKLOG_SYSTEM = (
    "You maintain the work log of an autonomous coding/ops agent whose conversation is being "
    "compacted. Rewrite the log so the agent can continue the task from it alone. Output ONLY "
    "these sections, as terse bullet lists, no prose:\n"
    "DONE: what has been completed - exact file paths created/changed (one line each, with what "
    "it contains), commands run and their outcome.\n"
    "STATE: current facts that matter - test results, versions, environment quirks, what is "
    "installed, what currently passes or fails.\n"
    "DECISIONS: choices made and why (design, naming, formats), so they are not re-litigated.\n"
    "OPEN: every requirement of the ORIGINAL TASK (given to you; it stays verbatim in the "
    "conversation) that the DONE list does not prove, plus follow-ups, in order, the very next step "
    "first. A requirement is only satisfied if DONE shows the file written / the command succeeded; "
    "when in doubt it stays OPEN - claiming unfinished work is done is the worst error here.\n"
    "PITFALLS: errors hit and their fixes or workarounds; things NOT to try again.\n"
    "Keep exact identifiers, paths, commands, error text and numbers. Drop pleasantries, "
    "repeated reads of the same file, and raw file contents (a file is recoverable by reading "
    "it - just say what it is). If an earlier log is given, merge it: keep its still-true "
    "items, update changed ones, never lose open items."
)


CHAT_SUMMARY_SYSTEM = (
    "You keep the running summary of an ongoing CONVERSATION between a user and their assistant, so that older messages "
    "can be dropped without losing the thread. Rewrite the summary from the earlier summary and the new messages. Output "
    "ONLY these sections as short bullets, in the language the user writes in:\n"
    "TOPICS: what was discussed, in order.\n"
    "ESTABLISHED: facts and answers settled so far; for anything from the web keep the source name and its date, and say "
    "when sources disagreed or the evidence was weak. Exact names, numbers, versions, commands.\n"
    "USER: what the user prefers or decided (language, tone, choices), and things they corrected.\n"
    "OPEN: questions or requests of the user not yet fully answered, the next one first.\n"
    "Drop small talk and anything recoverable by asking again. Never invent: if unsure, leave it out. Keep it under 250 words."
)


MASK_DEFAULT = 0.50             # stage 1: trim old tool output at this fill level (cheap, no model call)
MASK_KEEP_LAST = 8              # the most recent messages are never masked
MASK_MIN_CHARS = 600            # shorter results are left alone
MASK_MIN_SAVING = 3000          # not worth invalidating the server's prompt cache for less (chars)
TOOL_LOG_MARKER = "TOOL LOG (exact, recorded by the harness):"
TOOL_LOG_MAX_LINES = 60
_EXIT = re.compile(r"exit code (-?\d+)")


def mask_threshold() -> float:
    raw = os.environ.get("AGENT_MASK_AT", "").strip()
    try:
        value = float(raw) if raw else MASK_DEFAULT
    except ValueError:
        value = MASK_DEFAULT
    return value if 0.1 <= value < threshold() else MASK_DEFAULT


def should_mask(used_tokens: int, num_ctx: int | None) -> bool:
    return bool(num_ctx and used_tokens > 0 and used_tokens >= mask_threshold() * int(num_ctx))


def _first_line(text: str, limit: int = 160) -> str:
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def mask_old_observations(
    messages: list[dict[str, Any]], keep_last: int = MASK_KEEP_LAST
) -> tuple[list[dict[str, Any]], int]:
    """Replace the bulky output of OLD tool calls by a one-line stub. Returns (messages, chars saved).

    Tool results (file contents, command output, search pages) and the code inside old write_file calls
    dominate the window but are recoverable (the file is on disk, the command can be re-run), so the stub
    keeps what it was and how big, and the model's own reasoning and calls stay untouched. This is the
    cheap first stage before the model-written work log; it cannot invent anything."""
    cut = max(len(messages) - keep_last, 0)
    out: list[dict[str, Any]] = []
    saved = 0
    for i, m in enumerate(messages):
        content = str(m.get("content", ""))
        if i < cut and len(content) >= MASK_MIN_CHARS:
            if m.get("role") == "tool":
                stub = (f"[old tool output omitted to save context: {_first_line(content)} "
                        f"({len(content)} chars) - run the tool again if you need it]")
                out.append({**m, "content": stub})
                saved += len(content) - len(stub)
                continue
            if m.get("role") == "assistant":
                try:
                    data = json.loads(content)
                except ValueError:
                    data = None
                args = data.get("arguments") if isinstance(data, dict) else None
                if isinstance(args, dict) and isinstance(args.get("content"), str) and len(args["content"]) >= MASK_MIN_CHARS:
                    slim = {**data, "arguments": {**args, "content": f"[{len(args['content'])} chars - now on disk]"}}
                    new = json.dumps(slim)
                    out.append({**m, "content": new})
                    saved += len(content) - len(new)
                    continue
        out.append(m)
    return out, saved


def tool_log(chunk: list[dict[str, Any]]) -> list[str]:
    """Exact facts from the tool calls in `chunk`: files written/edited, commands run and their exit code."""
    lines: list[str] = []
    pending: tuple[str, str] | None = None
    for m in chunk:
        role, content = m.get("role"), str(m.get("content", ""))
        if role == "assistant":
            try:
                data = json.loads(content)
            except ValueError:
                continue
            if not isinstance(data, dict) or "tool" not in data:
                continue
            name, args = str(data["tool"]), data.get("arguments") or {}
            if name in ("write_file", "replace_in_file") and args.get("path"):
                lines.append(f"- {'wrote' if name == 'write_file' else 'edited'} {args['path']}")
            elif name == "run_command":
                pending = ("run", str(args.get("command", ""))[:140])
            elif name in ("web_search", "web_fetch"):
                lines.append(f"- {name}: {str(args.get('query') or args.get('url') or '')[:100]}")
        elif role == "tool" and pending:
            code = _EXIT.search(content[:400])
            lines.append(f"- ran `{pending[1]}` -> exit {code.group(1) if code else '?'}")
            pending = None
    return lines


def split_tool_log(log_text: str) -> tuple[str, list[str]]:
    """(text without the tool-log section, the section's lines) - so a re-compaction can extend it."""
    if TOOL_LOG_MARKER not in log_text:
        return log_text, []
    body, _, tail = log_text.partition(TOOL_LOG_MARKER)
    return body.rstrip(), [ln for ln in tail.splitlines() if ln.startswith("- ")]


def threshold() -> float:
    raw = os.environ.get("AGENT_COMPACT_AT", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_THRESHOLD
    except ValueError:
        value = DEFAULT_THRESHOLD
    return value if 0.2 <= value <= 0.95 else DEFAULT_THRESHOLD


def should_compact(used_tokens: int, num_ctx: int | None) -> bool:
    """True once the last model call showed the window is mostly full."""
    if not num_ctx or used_tokens <= 0:
        return False
    return used_tokens >= threshold() * int(num_ctx)


def _clip(text: str, limit: int = PER_MESSAGE_CHARS) -> str:
    """Head and tail of a long text: the start says what it is, the end says how it ended."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[… {len(text) - limit} chars omitted …]\n{text[-half:]}"


def render(messages: list[dict[str, Any]], per_message_chars: int = PER_MESSAGE_CHARS) -> str:
    return "\n\n".join(f"[{m.get('role', '?')}] {_clip(m.get('content', ''), per_message_chars)}" for m in messages)


def _is_worklog(message: dict[str, Any]) -> bool:
    return message.get("role") == "system" and str(message.get("content", "")).startswith(WORKLOG_MARKER)


def summarize(
    llm: Any, config: dict[str, Any], chunk: list[dict[str, Any]], existing: str | None = None,
    task: str | None = None, system: str | None = None,
) -> str | None:
    """Fold `chunk` (and an earlier log) into a fresh work log, or with `system=CHAT_SUMMARY_SYSTEM` a conversation
    summary. None if the model call fails."""
    text = render(chunk)
    if existing:
        text = f"EARLIER LOG:\n{existing}\n\nNEW ACTIVITY TO FOLD IN:\n{text}"
    if task:
        text = f"ORIGINAL TASK:\n{_clip(task, 4000)}\n\n{text}"
    try:
        out = "".join(
            llm.chat(
                messages=[{"role": "system", "content": system or WORKLOG_SYSTEM}, {"role": "user", "content": text}],
                model=config["models"]["background"],
                stream=False,
                background=True,
            )
        ).strip()
    except Exception as exc:  # noqa: BLE001 - compaction must never break a turn
        logger.warning("Work-log compaction failed: %s", exc)
        return None
    return out or None


def compact_turn_messages(
    llm: Any, config: dict[str, Any], messages: list[dict[str, Any]], keep_last: int = KEEP_LAST_MESSAGES,
) -> list[dict[str, Any]] | None:
    """Replace the middle of a running tool loop by one work-log message.

    Kept verbatim: everything up to and including the last user message (system prompt, history,
    the task) and the final `keep_last` messages, cut so the tail starts at an assistant message
    (a tool result never loses its call). Returns the new message list, or None if there is
    nothing worth compacting or the summary failed (the caller then carries on unchanged)."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    head_end = last_user + 1
    cut = max(len(messages) - keep_last, head_end)
    while cut < len(messages) and messages[cut].get("role") != "assistant":
        cut += 1                                   # the tail must start at a call, not a result
    middle = messages[head_end:cut]
    if len([m for m in middle if not _is_worklog(m)]) < MIN_MIDDLE_MESSAGES:
        return None
    existing = next((str(m["content"]).split("\n", 1)[-1] for m in middle if _is_worklog(m)), None)
    old_lines: list[str] = []
    if existing:
        existing, old_lines = split_tool_log(existing)
    task = str(messages[last_user].get("content", "")) if last_user >= 0 else None
    fresh = [m for m in middle if not _is_worklog(m)]
    log = summarize(llm, config, fresh, existing, task)
    if not log:
        return None
    lines = (old_lines + tool_log(fresh))[-TOOL_LOG_MAX_LINES:]
    if lines:
        log = f"{log}\n\n{TOOL_LOG_MARKER}\n" + "\n".join(lines)
    note = {
        "role": "system",
        "content": f"{WORKLOG_MARKER} (auto-compacted from {len(middle)} earlier steps; trust this, "
                   f"re-read files if you need exact contents):\n{log}",
    }
    return [*messages[:head_end], note, *messages[cut:]]
