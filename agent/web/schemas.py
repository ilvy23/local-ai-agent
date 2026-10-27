"""Pydantic request/response models for the web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SessionOut(BaseModel):
    id: int
    title: str | None = None
    started_at: str | None = None
    message_count: int = 0
    pinned: bool = False


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str | None = None


class SendMessageIn(BaseModel):
    text: str
    web_forced: bool = False


class ApprovalDecisionIn(BaseModel):
    approve: bool


class PendingApprovalOut(BaseModel):
    id: int
    session_id: int
    tool_name: str
    arguments: dict[str, Any]
    risk_level: str
    description: str
    status: str
    created_at: str


class NotesOut(BaseModel):
    notes: str | None = None


class ModelsOut(BaseModel):
    current: str
    available: list[str]
    loaded: list[str]


class SelectModelIn(BaseModel):
    model: str
