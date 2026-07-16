"""Assemble the message list sent to the chat model for one turn.

Layers the system prompt (persona + distilled facts + semantically recalled
past memories) on top of the current session's history and the new user
message, trimming oldest session messages to fit a char budget. Facts and
persona are never dropped, and any embedding failure degrades to "no recall"
rather than breaking the turn.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

logger = logging.getLogger(__name__)


def _tool_instructions(tool_names: list[str]) -> str:
    names = ", ".join(tool_names)
    parts = [
        "You have access to these tools: "
        + names
        + ". If you need one, reply with ONLY a JSON object of the form "
        '{"tool": "<name>", "arguments": {...}} and nothing else. Otherwise, '
        "answer normally."
    ]
    if "web_search" in tool_names:
        # Without this the model searches for "what songs are on X" but not for
        # "do you know anything about X" — and, worst of all, not when the user
        # says it got something wrong. It cannot feel the difference between
        # recalling a detail and inventing one, so name the situations for it.
        parts.append(
            "When to use web_search: your knowledge is out of date and has gaps, "
            "and inventing a detail feels exactly like remembering one. Look it "
            "up instead of answering from memory when:\n"
            "- the user corrects you or says you got something wrong. That is the "
            "clearest evidence your memory is unreliable here, so check it rather "
            "than guessing again or simply agreeing.\n"
            "- the answer turns on specifics: names, dates, versions, prices, "
            "track listings, who did what, or anything current.\n"
            "- you are about to state something you are not certain is right.\n"
            "Just answer, without searching, for conversation, opinions, "
            "arithmetic, writing, and general concepts you reliably know.\n"
            "Never present a guess as fact. If you did not look it up and you are "
            "unsure, say so."
        )
    return "\n\n".join(parts)


def _semantic_hits(
    vectors: VectorIndex, llm: Any, config: dict[str, Any], user_input: str
) -> list[str]:
    """Return recalled memory texts for user_input, or [] on any failure."""
    if not vectors.available:
        return []
    k = config["memory"]["recall_k"]
    try:
        embedding = llm.embed([user_input], model=config["models"]["embed"])[0]
    except Exception as exc:  # noqa: BLE001 - recall must never break a turn
        logger.warning("Recall embedding failed, skipping semantic memory: %s", exc)
        return []
    return [text for text, _kind, _ref, _dist in vectors.search(embedding, k=k)]


def _build_system_prompt(
    persona: str, facts: list[str], memories: list[str], tool_names: list[str] | None,
    environment: list[str] | None = None,
) -> str:
    parts = [persona]
    if tool_names:
        parts.append(_tool_instructions(tool_names))
    if environment:
        parts.append(
            "System environment (always current — use these real paths):\n"
            + "\n".join(f"- {e}" for e in environment)
        )
    if facts:
        parts.append("Known facts about the user:\n" + "\n".join(f"- {f}" for f in facts))
    if memories:
        parts.append(
            "Possibly relevant past memories:\n" + "\n".join(f"- {m}" for m in memories)
        )
    return "\n\n".join(parts)


def build_context(
    store: Store,
    vectors: VectorIndex,
    llm: Any,
    config: dict[str, Any],
    session_id: int,
    user_input: str,
    tool_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build the full message list for this turn (system + history + user).

    `tool_names`, if given, adds a brief instruction on how to call a tool —
    mainly useful for the JSON fallback path when the model doesn't use
    Ollama's structured tool_calls.
    """
    from agent.memory.environment import system_facts

    persona = config["persona"]["style"]
    facts = [f["content"] for f in store.get_active_facts()]
    memories = _semantic_hits(vectors, llm, config, user_input)
    environment = system_facts(config)

    system = {
        "role": "system",
        "content": _build_system_prompt(persona, facts, memories, tool_names, environment),
    }
    history = store.get_messages(session_id)
    new_user = {"role": "user", "content": user_input}

    budget = config["memory"]["context_char_budget"]
    fixed_len = len(system["content"]) + len(user_input)
    # Keep newest history messages until adding an older one would exceed budget.
    kept: list[dict[str, str]] = []
    used = fixed_len
    for message in reversed(history):
        cost = len(message["content"])
        if used + cost > budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()

    # A "tool" message is always preceded by the assistant tool-call message
    # that produced it. If the trim above kept the tool message but dropped
    # its preceding assistant call, the replayed context would start with an
    # orphaned tool result with no call to explain it. Drop any leading tool
    # messages left dangling like this.
    while kept and kept[0]["role"] == "tool":
        kept.pop(0)

    return [system, *kept, new_user]
