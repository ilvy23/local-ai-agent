"""Live 'what the worker is doing right now' marker.

The watcher writes a one-line activity (e.g. "summarizing a chat") to a small
JSON file in the data dir as it moves item to item; `agent worker status`
reads it back so you can see the current task, not just the last pass. Best-
effort — a failed write never disrupts the actual work.

Also exposes ``AgentStatus`` / ``StatusRegistry`` for the live status dashboard
(``agent/panel.py``).  The registry is an in-process singleton; it is only
meaningful for agents running in the same Python process.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar


def _activity_path(config: dict) -> Path:
    from agent.config import PROJECT_ROOT

    db = Path(config["data"]["db_path"])
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    return db.parent / "worker_activity.json"


def set_activity(config: dict, text: str) -> None:
    """Record what the worker is doing right now (best-effort)."""
    try:
        _activity_path(config).write_text(json.dumps({"text": text, "at": int(time.time())}))
    except OSError:
        pass


def read_activity(config: dict) -> tuple[str, int] | None:
    """(text, seconds_ago) of the current activity, or None if unknown."""
    try:
        d = json.loads(_activity_path(config).read_text())
        return d.get("text", ""), max(0, int(time.time()) - int(d.get("at", 0)))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# In-process agent status registry (used by StatusPanel in panel.py)
# ---------------------------------------------------------------------------

_MISSING = object()  # sentinel for optional update() kwargs


@dataclass
class AgentStatus:
    """Snapshot of a single running agent session."""

    agent_id: str          # e.g. "Agt-1"
    model: str             # e.g. "qwen3.5:9b"
    milestone: str = ""    # e.g. "M2 Coding"
    last_action: str = ""  # e.g. "write auth.py"
    status: str = "idle"   # "running" | "idle" | "waiting_review" | "done"
    review_pending: bool = False


class StatusRegistry:
    """In-process singleton tracking active agent sessions.

    Thread-safe: all mutations hold ``_lock``.
    """

    _instance: ClassVar[StatusRegistry | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._agents: dict[str, AgentStatus] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> StatusRegistry:
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def register(self, agent_id: str, model: str) -> None:
        """Add a new agent entry (replaces any prior entry with the same id)."""
        with self._lock:
            self._agents[agent_id] = AgentStatus(agent_id=agent_id, model=model)

    def update(
        self,
        agent_id: str,
        *,
        milestone: object = _MISSING,
        last_action: object = _MISSING,
        status: object = _MISSING,
        review_pending: object = _MISSING,
    ) -> None:
        """Partially update a registered agent's fields.

        Only the kwargs that differ from ``_MISSING`` are applied, so callers
        can update a single field without clobbering the rest.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return
            if milestone is not _MISSING:
                agent.milestone = str(milestone)
            if last_action is not _MISSING:
                agent.last_action = str(last_action)
            if status is not _MISSING:
                agent.status = str(status)
            if review_pending is not _MISSING:
                agent.review_pending = bool(review_pending)

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the registry (no-op if not present)."""
        with self._lock:
            self._agents.pop(agent_id, None)

    def all_agents(self) -> list[AgentStatus]:
        """Return a snapshot list of all currently registered agents.

        The AgentStatus objects are *copies*. Handing out the live ones only
        copied the list: the dashboard thread then read fields off objects that
        worker threads were still mutating, so one rendered row could mix an
        old milestone with a new status. Copying under the lock makes each
        returned row a consistent point-in-time view.
        """
        with self._lock:
            return [replace(agent) for agent in self._agents.values()]

    def mark_review_pending(self, agent_id: str) -> None:
        """Flag the agent as needing human review and flip status to 'waiting_review'."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is not None:
                agent.review_pending = True
                agent.status = "waiting_review"

    def clear_review(self, agent_id: str) -> None:
        """Clear the review-pending flag; restores status to 'running' if it was 'waiting_review'."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is not None:
                agent.review_pending = False
                if agent.status == "waiting_review":
                    agent.status = "running"
