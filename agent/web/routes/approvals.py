"""Pending tool-approval endpoints: list what's waiting, decide one and
resume its agent turn.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent.memory.vectors import VectorIndex
from agent.web.agent_runtime import resolve_pending_approval, run_agent_turn_web
from agent.web.deps import build_registry, get_client, get_config, get_store
from agent.web.routes.chat import _persisting_emit, _sse_stream
from agent.web.schemas import ApprovalDecisionIn, PendingApprovalOut

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


@router.get("/pending", response_model=list[PendingApprovalOut])
def list_pending(session_id: int | None = None, store=Depends(get_store)) -> list[PendingApprovalOut]:
    rows = store.list_pending_approvals(session_id=session_id, status="pending")
    return [
        PendingApprovalOut(
            id=r["id"], session_id=r["session_id"], tool_name=r["tool_name"],
            arguments=r["arguments"], risk_level=r["risk_level"],
            description=r["description"], status=r["status"], created_at=r["created_at"],
        )
        for r in rows
    ]


@router.post("/{approval_id}/decide")
async def decide(
    approval_id: int,
    body: ApprovalDecisionIn,
    store=Depends(get_store),
    client=Depends(get_client),
    config=Depends(get_config),
) -> StreamingResponse:
    row = store.get_pending_approval(approval_id)
    if row is None:
        raise HTTPException(404, "No such pending approval.")
    if row["status"] != "pending":
        raise HTTPException(409, f"Already {row['status']}.")

    session_id = row["session_id"]
    vectors = VectorIndex(store)

    async def run(emit: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        resolved = resolve_pending_approval(
            build_registry(
                store, vectors, client, config, session_id,
                emit=_persisting_emit(store, session_id, emit),
            ),
            store, config, approval_id, body.approve,
        )
        if resolved is None:
            await emit({"type": "error", "message": "Approval was already decided elsewhere."})
            return
        result, turn_state = resolved

        from agent.web.agent_runtime import _tool_message_pair

        assistant_call_msg, tool_msg = _tool_message_pair(
            row["tool_name"], row["arguments"], result
        )
        store.add_message(session_id, "assistant", assistant_call_msg["content"])
        store.add_message(session_id, "tool", result)
        messages = [*turn_state["messages"], assistant_call_msg, tool_msg]
        tools_called = set(turn_state["tools_called"]) | {row["tool_name"]}

        registry = build_registry(
            store, vectors, client, config, session_id,
            emit=_persisting_emit(store, session_id, emit),
        )
        persona_name = config["persona"]["name"]
        reply, _tools_used = await run_agent_turn_web(
            client, config, store, session_id, registry, messages, persona_name, emit,
            tools_called=tools_called, start_iteration=turn_state["iteration"] + 1,
        )
        store.touch_session(session_id)
        if reply is not None:
            from agent.tui import _embed_message

            assistant_msg_id = store.add_message(session_id, "assistant", reply)
            _embed_message(vectors, client, config, assistant_msg_id, reply)

    return StreamingResponse(_sse_stream(run), media_type="text/event-stream")
