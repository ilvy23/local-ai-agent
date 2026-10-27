"""The core send-message endpoint: builds context, runs the agent turn,
streams SSE events. Mirrors the sequence agent/tui.py's run_repl does per
turn (build_context -> run turn -> persist -> embed), adapted to push
events instead of printing them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agent.memory import background
from agent.memory.recall import build_context
from agent.research import evidence as turn_evidence
from agent.memory.vectors import VectorIndex
from agent.tui import _embed_message
from agent.web.agent_runtime import begin_web_turn, create_plan, run_agent_turn_web
from agent.web.deps import build_registry, get_client, get_config, get_store
from agent.web.schemas import SendMessageIn

router = APIRouter(prefix="/api/sessions", tags=["chat"])

_TERMINAL_EVENTS = {"done", "error", "awaiting_approval"}


def _format_sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _add_mood(event: dict[str, Any], config: dict[str, Any] | None) -> None:
    """Additive kaomoji side-channel (agent/mood.py) on an existing status-ish
    event - sets event["mood"] when there's a match, touches nothing a
    client already reads otherwise. `config` missing/odd just means no
    kaomoji this time, same as if the setting were off."""
    if config is None:
        return
    etype = event.get("type")
    key = {
        "status": {"searching": "searching", "verifying": "waiting", "thinking": "waiting",
                    "compacting": "waiting", "planning": "waiting"}.get(event.get("state"), event.get("state")),
        "stuck": "stuck",
        "error": "error",
        "done": "done",
        "verify": {"passed": "test_passed", "failed": "error"}.get(event.get("state")),
    }.get(etype)
    if not key:
        return
    try:
        from agent import mood
        kaomoji = mood.mood_for(key, config)
    except Exception:  # noqa: BLE001 - cosmetic only, never worth breaking the stream
        kaomoji = ""
    if kaomoji:
        event["mood"] = kaomoji


def _persisting_emit(
    store: Any, session_id: int, emit: Callable[[dict[str, Any]], Awaitable[None]],
    config: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Wraps `emit` so a `file_ready` event (the player/download card a tool
    like send_file pushes directly via console.emit_event, bypassing the
    model's own reply) also gets a durable row in `messages` - otherwise
    the card only ever existed on the live SSE stream, and reopening the
    session (a page reload, or iOS Safari reclaiming a backgrounded tab -
    both routine on a phone) replayed the text reply claiming "it's
    playing" with no player anywhere to back it up. Stored under a role
    other than "assistant"/"user"/"tool" so build_context's history walk
    (agent/memory/recall.py) never feeds this synthetic event to the model -
    it isn't something the model said or a tool result it needs to see
    again, only something the frontend needs to redraw the card."""

    async def wrapped(event: dict[str, Any]) -> None:
        _add_mood(event, config)
        if event.get("type") == "file_ready":
            store.add_message(session_id, "file_card", json.dumps(event))
        elif event.get("type") == "block":
            store.add_message(session_id, "block", json.dumps(event))
        elif event.get("type") in ("sources", "verdict", "verify_result"):
            store.add_message(session_id, "evidence", json.dumps(event))
        await emit(event)

    return wrapped


async def _sse_stream(
    run: Callable[[Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[None]],
) -> AsyncIterator[str]:
    """Bridge a "push events into emit()" coroutine into a "pull one SSE
    chunk at a time" async generator, which is what StreamingResponse
    needs. `run` does the actual work and is expected to call the `emit`
    it's given; this just drains whatever it emits, in order, as it runs.
    """
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def emit(event: dict[str, Any]) -> None:
        await queue.put(event)

    task = asyncio.create_task(run(emit))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {get_task, task}, return_when=asyncio.FIRST_COMPLETED
            )
            if get_task in done:
                yield _format_sse(get_task.result())
            else:
                get_task.cancel()
            if task in done:
                while not queue.empty():
                    yield _format_sse(queue.get_nowait())
                if task.exception():
                    raise task.exception()
                break
    finally:
        if not task.done():
            task.cancel()


@router.post("/{session_id}/messages")
async def send_message(
    session_id: int,
    body: SendMessageIn,
    store=Depends(get_store),
    client=Depends(get_client),
    config=Depends(get_config),
) -> StreamingResponse:
    vectors = VectorIndex(store)
    user_input = body.text

    async def run(emit: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        loop = asyncio.get_running_loop()
        # what the web said in an earlier turn is not what this turn read (clears taint + evidence); also the shared
        # gates' per-turn state (change log, verify, goal, plan holder - the plan tool in the registry shares it)
        runtime = begin_web_turn(config, session_id, user_input)
        # One wrapped emit for both the registry's tool events and the turn's own status/done
        # events, so the additive mood side-channel (agent/mood.py) reaches whichever one a
        # given event actually comes from, not just tool-triggered ones.
        pemit = _persisting_emit(store, session_id, emit, config)
        registry = build_registry(
            store, vectors, client, config, session_id,
            emit=pemit,
        )
        tool_names = [t.name for t in registry.list()]
        messages = await asyncio.to_thread(
            build_context, store, vectors, client, config, session_id, user_input, tool_names
        )
        messages = await asyncio.to_thread(create_plan, runtime, client, config, user_input, messages,
                                           lambda ev: asyncio.run_coroutine_threadsafe(pemit(ev), loop).result(timeout=30))
        user_msg_id = store.add_message(session_id, "user", user_input)
        row = store.conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is not None and not row[0]:
            store.set_session_title(session_id, " ".join(user_input.split("\n", 1)[0].split())[:60])       # like the terminal chat: the first message names the chat
        persona_name = config["persona"]["name"]
        reply, _tools_used = await run_agent_turn_web(
            client, config, store, session_id, registry, messages, persona_name, pemit
        )
        research = turn_evidence.take()
        store.touch_session(session_id)
        await asyncio.to_thread(_embed_message, vectors, client, config, user_msg_id, user_input)
        if reply is not None:
            assistant_msg_id = store.add_message(session_id, "assistant", reply)
            await asyncio.to_thread(
                _embed_message, vectors, client, config, assistant_msg_id, reply
            )
            background.learn_after_turn(client, config, user_input, reply, research)

    return StreamingResponse(_sse_stream(run), media_type="text/event-stream")


@router.post("/{session_id}/verify")
async def verify_last_answer(
    session_id: int,
    store=Depends(get_store),
    client=Depends(get_client),
    config=Depends(get_config),
) -> StreamingResponse:
    """Deeper check of the last answer (the GUI's "Verify" button): fresh counter-searches for its key claims, then a
    `verify_result` event. Every search also emits its own `sources` / `verdict` events, so the UI can show the evidence."""
    from agent.research import verify as verifymod

    vectors = VectorIndex(store)

    async def run(emit: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        pemit = _persisting_emit(store, session_id, emit, config)
        last = ""
        for m in reversed(store.get_messages(session_id)):
            if m["role"] == "assistant" and not m["content"].lstrip().startswith("{"):
                last = m["content"]
                break
        if not last.strip():
            await pemit({"type": "error", "message": "There is no answer to verify yet."})
            return
        registry = build_registry(store, vectors, client, config, session_id, emit=pemit)
        search = registry.get("web_search")
        if search is None:
            await pemit({"type": "error", "message": "Web search is not available."})
            return

        def run_search(query: str) -> tuple[list[dict], dict]:
            search.handler(query=query, num_results=4)
            found = turn_evidence.last_search()
            return found["items"], found["verdict"]

        await pemit({"type": "status", "state": "verifying", "label": "verifying", "detail": "", "since": ""})
        result = await asyncio.to_thread(verifymod.verify, last, run_search)
        await pemit({"type": "verify_result", **result})
        await pemit({"type": "done", "reply": verifymod.summarize(result)})

    return StreamingResponse(_sse_stream(run), media_type="text/event-stream")
