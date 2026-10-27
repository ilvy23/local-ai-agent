"""Context compaction for the tool loop: ONE implementation for the classic terminal loop (agent.tui), the new terminal
chat and the web chat (agent.chat.turn).

Two stages, as before: mask old tool output at ~50 % of the window (no model call), then fold the middle of the run
into a model-written work log at ~70 %. `force=True` (the /compact command) runs one pass whatever the fill level.
Nothing here prints or reads stdin: `event(kind, **data)` reports, `say(markup)` is the optional console line.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Callable

from agent.memory import worklog

Event = Callable[..., None]


def _no_event(kind: str, /, **data: Any) -> None:
    return None


def _context_payload(message: dict[str, Any], num_ctx: int, compacted: bool = False) -> dict[str, Any]:
    from agent.chat.turn import _context_event

    data = _context_event(message, num_ctx, compacted)
    data.pop("type", None)
    return data


def compact_context(
    client: Any, config: dict[str, Any], messages: list[dict[str, Any]], last_reply: dict[str, Any], num_ctx: int, *,
    event: Event = _no_event, say: Callable[[str], None] | None = None, force: bool = False,
    spinner: Callable[[], Any] | None = None,
) -> list[dict[str, Any]]:
    """Return `messages`, trimmed and/or compacted when the window is filling up (or `force`)."""
    say = say or (lambda _line: None)
    used = int(last_reply.get("_prompt_eval_count", 0) or 0) + int(last_reply.get("_eval_count", 0) or 0)
    changed = False
    if force or worklog.should_mask(used, num_ctx):
        # Stage 1 (no model call): old tool output and old file contents become one-line stubs.
        masked, saved = worklog.mask_old_observations(messages)
        if saved >= (1 if force else worklog.MASK_MIN_SAVING):
            messages = masked
            used = max(used - saved // 4, 0)
            changed = True
            say(f"[dim]context trimmed  -{saved // 4:,} tokens of old tool output[/dim]".replace(",", "."))
            event("log", text=f"context trimmed: -{saved // 4:,} tokens of old tool output")
    if not force and not worklog.should_compact(used, num_ctx):
        if changed:
            event("context", **_context_payload({**last_reply, "_prompt_eval_count": used, "_eval_count": 0}, num_ctx, True))
        return messages
    before_tokens = used
    event("status", state="compacting", label="compacting", detail="", since=time.time())
    with (spinner() if spinner is not None else contextlib.nullcontext()):
        compacted = worklog.compact_turn_messages(client, config, messages)
    if compacted is None:
        if force:
            event("log", text="compact: nothing worth compacting yet")
        if changed:
            event("context", **_context_payload({**last_reply, "_prompt_eval_count": used, "_eval_count": 0}, num_ctx, True))
        return messages
    after_tokens = sum(len(str(m.get("content", ""))) for m in compacted) // 4
    say(f"[dim]✓ context compacted  {before_tokens:,} → ~{after_tokens:,} tokens[/dim]".replace(",", "."))
    event("context", **_context_payload({"_prompt_eval_count": after_tokens, "_eval_count": 0}, num_ctx, True),
          before=before_tokens)
    return compacted
