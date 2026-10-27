"""Memory work that must not make a reply wait.

Saving a learned fact costs a model call on the same single slot the chat uses, so it runs on one background worker,
after the reply is out and in order. Failures are logged, never raised.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-bg")


def submit(fn: Callable[..., Any], *args: Any) -> Future:
    def guarded() -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - background memory work is a bonus
            logger.warning("background memory task failed", exc_info=True)

    return _pool.submit(guarded)


def learn_after_turn(client: Any, config: dict[str, Any], user_input: str, reply: str, evidence: dict | None) -> Future | None:
    """Queue the learned-fact check for one finished turn - only when the web evidence was strong enough to save.

    The worker opens its own database connection (a Store belongs to the request that made it)."""
    from agent.research.evidence import SAVEABLE

    if not evidence or evidence.get("label") not in SAVEABLE or not reply.strip():
        return None

    def work() -> None:
        from agent.memory.store import Store
        from agent.memory.vectors import VectorIndex
        from agent.tui import _save_learned_facts

        with Store(config["data"]["db_path"]) as store:
            _save_learned_facts(client, config, store, VectorIndex(store), [(user_input, reply, evidence)])

    return submit(work)
