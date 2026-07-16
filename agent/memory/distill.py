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
import time
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
    "You pull out durable facts about the USER from what they said, so an "
    "assistant can remember them in future conversations.\n"
    "\n"
    "Include anything lasting they revealed about themselves:\n"
    "- who they are: name, age, where they live, job, what they study\n"
    "- what they like or dislike: favourites, preferences, hobbies, taste\n"
    "- the people and pets in their life\n"
    "- what they are working on, learning, or planning\n"
    "\n"
    "How they phrase it does not matter. All of these are facts about the user:\n"
    '  "X is my favourite band"      -> the user\'s favourite band is X\n'
    '  "my dog is called Rex"        -> the user has a dog called Rex\n'
    '  "I work nights"               -> the user works nights\n'
    '  "been playing piano for years" -> the user plays piano\n'
    "A sentence whose grammatical subject is a band, a pet or a hobby is still "
    "a fact about the user if they said it about their own life.\n"
    "\n"
    "Leave out:\n"
    "- their computer: files, paths, directories, commands, or any tool output\n"
    "- the assistant, this app, or this conversation\n"
    "- one-off requests and questions ('list my music', 'what's the weather')\n"
    "- anything you would have to guess at — only what they actually said\n"
    "\n"
    "Reply with ONLY a JSON array of short strings, each written in the third "
    "person about the user, e.g. the form 'likes …' / 'has a …' / 'works as …'. "
    "Reply [] if they revealed nothing lasting about themselves."
)


def embed_with_retry(llm: Any, text: str, model: str, attempts: int = 3) -> list[float] | None:
    """Embed `text`, retrying transient failures. None if it never succeeds.

    Embedding runs right after a chat turn, while the chat model is still
    resident — on a smaller GPU there may not be room for the embedding model
    too, and Ollama answers 500. That's transient: a moment later it fits.
    """
    for attempt in range(attempts):
        try:
            return llm.embed([text], model=model)[0]
        except Exception as exc:  # noqa: BLE001 - never let embedding break a turn
            if attempt == attempts - 1:
                logger.warning("Could not embed %r after %d tries: %s", text[:60], attempts, exc)
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


_UNEMBEDDED = {
    # kind -> (sql selecting id + text for rows with no vector)
    "fact": (
        "SELECT id, content AS text FROM facts WHERE active = 1 AND id NOT IN "
        "(SELECT ref_id FROM memory_items WHERE kind = 'fact' AND ref_id IS NOT NULL)"
    ),
    "message": (
        "SELECT id, content AS text FROM messages WHERE role IN ('user','assistant') "
        "AND content != '' AND id NOT IN "
        "(SELECT ref_id FROM memory_items WHERE kind = 'message' AND ref_id IS NOT NULL)"
    ),
}


def repair_unembedded(
    store: Store, vectors: VectorIndex, llm: Any, config: dict[str, Any]
) -> dict[str, int]:
    """Embed facts and messages that have no vector. Returns counts per kind.

    Anything whose embedding failed is stored but unsearchable: it never reached
    `memory_items`, so semantic recall can't see it and even `reembed` (which
    reads from `memory_items`) can't rescue it. Without this, one blip means the
    agent keeps something it can never remember.
    """
    fixed = {kind: 0 for kind in _UNEMBEDDED}
    if not vectors.available:
        return fixed
    for kind, sql in _UNEMBEDDED.items():
        for row in store.conn.execute(sql).fetchall():
            embedding = embed_with_retry(llm, row["text"], config["models"]["embed"])
            if embedding is None:
                continue  # still can't; try again next time rather than losing it
            try:
                vectors.add(kind, row["id"], row["text"], embedding)
                fixed[kind] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not index %s %s: %s", kind, row["id"], exc)
    if any(fixed.values()):
        logger.info("Repaired unembedded memory: %s", fixed)
    return fixed


def repair_unembedded_facts(
    store: Store, vectors: VectorIndex, llm: Any, config: dict[str, Any]
) -> int:
    """Facts-only repair. Returns how many were fixed."""
    return repair_unembedded(store, vectors, llm, config)["fact"]


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
        embedding = embed_with_retry(llm, fact, config["models"]["embed"])

        if embedding is not None:
            if vectors.max_cosine_similarity(embedding, "fact") > DEDUPE_SIMILARITY:
                continue

        fact_id = store.add_fact(fact, source_session_id=session_id)
        if embedding is not None:
            vectors.add("fact", fact_id, fact, embedding)
        existing.add(fact)
        added.append(fact)

    # Catch up anything a previous run stored but couldn't embed — otherwise the
    # agent keeps facts it can never recall.
    repair_unembedded(store, vectors, llm, config)
    return added
