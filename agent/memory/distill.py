"""Extract durable user facts from a finished session transcript.

Sends the transcript to the background model asking for a JSON array of short
facts, parses it robustly (junk around the array is tolerated; parse failures
are logged and skipped), then stores each new fact — deduped against existing
active facts by exact text or high embedding similarity — and embeds it into
the vector index.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.jsonx import extract_json_value
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

logger = logging.getLogger(__name__)

MIN_USER_MESSAGES = 2
DEDUPE_SIMILARITY = 0.9

# A "fact" that looks like a file path, tool echo, dir listing, or timestamp is
# noise the model scraped from a shell/web tool result — never a personal fact.
# Second line of defence behind only feeding user messages to the distiller.
JUNK_FACT_RE = re.compile(
    r"(^\s*[./~])|(/home/)|(\b[A-Za-z]:\\)|"
    r"(list_dir|image deleted|\[image\]|\.git\b|\.py\b|README|pyproject|"
    r"\.lock|\.gitignore|pytest|\bdir\b|http[s]?://)|"
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2})",
    re.IGNORECASE,
)

_PROMPT = (
    "Extract durable PERSONAL facts the user explicitly stated about THEMSELVES "
    "in this conversation: their identity, preferences, relationships, work, or "
    "ongoing personal projects.\n"
    "STRICT RULES — follow exactly:\n"
    "- Only include a fact the user DIRECTLY stated about their own life.\n"
    "- Do NOT include anything about the computer, operating system, files, "
    "directories, network, hardware, or the output of any command or tool.\n"
    "- Do NOT include anything about the assistant, this app, or the session itself.\n"
    "- Do NOT guess, infer, or speculate. Never use words like 'presumably', "
    "'possibly', 'probably', or 'implied'. If you are not certain the user "
    "stated it about themselves, leave it out.\n"
    "Reply with ONLY a JSON array of short factual strings. If there are no such "
    "facts, reply with []. (Format only — do NOT reuse this content: "
    '["prefers tea over coffee", "has a sister named Lena"].)'
)


def _extract_json_array(text: str) -> list[str] | None:
    """Return the first JSON array of strings in `text`, or None if none parses."""
    parsed = extract_json_value(text, "[", "]")
    if not isinstance(parsed, list):
        return None
    return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]


def distill_session(
    store: Store,
    vectors: VectorIndex,
    llm: Any,
    config: dict[str, Any],
    session_id: int,
) -> list[str]:
    """Extract, dedupe, store, and embed new facts. Returns the facts added."""
    history = store.get_messages(session_id)
    user_messages = [m for m in history if m["role"] == "user"]
    if len(user_messages) < MIN_USER_MESSAGES:
        return []

    # ONLY the user's own words. Feeding assistant replies and tool results (file
    # listings, web page text, command output) is what filled memory with file
    # paths and book plots — the model "extracted" them as facts.
    transcript = "\n".join(m["content"] for m in user_messages)
    reply = "".join(
        llm.chat(
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": transcript},
            ],
            model=config["models"]["background"],
            stream=False,
        )
    )

    candidates = _extract_json_array(reply)
    if candidates is None:
        logger.warning("Fact extraction returned no parseable JSON array; skipping.")
        return []

    existing = {f["content"] for f in store.get_active_facts()}
    added: list[str] = []
    for fact in candidates:
        if fact in existing or JUNK_FACT_RE.search(fact):
            continue
        embedding: list[float] | None = None
        try:
            embedding = llm.embed([fact], model=config["models"]["embed"])[0]
        except Exception as exc:  # noqa: BLE001 - never let distill break the app
            logger.warning("Could not embed candidate fact %r: %s", fact, exc)

        if embedding is not None:
            if vectors.max_cosine_similarity(embedding, "fact") > DEDUPE_SIMILARITY:
                continue

        fact_id = store.add_fact(fact, source_session_id=session_id)
        if embedding is not None:
            vectors.add("fact", fact_id, fact, embedding)
        existing.add(fact)
        added.append(fact)

    return added
