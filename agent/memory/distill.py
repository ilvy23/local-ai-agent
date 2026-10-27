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
from dataclasses import dataclass, field
from typing import Any

from agent.jsonx import extract_json_value
from agent.llm import EmbeddingUnavailable, strip_thinking
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex

logger = logging.getLogger(__name__)

# One message is enough. People routinely say the whole thing in a single line
# ("type o negative is my favourite band") and leave — requiring two threw that
# away before the distiller ever saw it. A throwaway session costs one cheap
# call that correctly returns [].
MIN_USER_MESSAGES = 1
DEDUPE_SIMILARITY = 0.9

# A "fact" that looks like a file path, tool echo, dir listing, or timestamp is noise the model scraped from a
# shell/web tool result - never a personal fact. Second line of defence behind only feeding user messages to the
# distiller. File names, tool names and URLs the USER mentions about their own setup ("uses pytest", "my site is
# https://...") are legitimate facts and no longer rejected; real paths and tool echoes still are.
JUNK_FACT_RE = re.compile(
    r"(^\s*[./~])|(/home/)|(\b[A-Za-z]:\\)|"
    r"(list_dir|image deleted|\[image\])|"
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
    "- their own setup: operating system, hardware, software and games they use or play\n"
    "\n"
    "How they phrase it does not matter. All of these are facts about the user:\n"
    '  "X is my favourite band"       -> {"text": "favourite band is X", "confidence": "verified"}\n'
    '  "my dog is called Rex"         -> {"text": "has a dog called Rex", "confidence": "verified"}\n'
    '  "I work nights"                -> {"text": "works nights", "confidence": "verified"}\n'
    '  "been playing piano for years" -> {"text": "plays piano", "confidence": "verified"}\n'
    '  "I run Debian and play chess" -> two facts: "uses Debian", "plays chess" (verified)\n'
    '  "I moved to Munich last month" -> {"text": "lives in Munich", "confidence": "verified"} (a stated change is still stated)\n'
    '  "I\'ve been doing this a while" -> {"text": "has experience with this area", "confidence": "assumed"}\n'
    "A sentence whose grammatical subject is a band, a pet or a hobby is still "
    "a fact about the user if they said it about their own life.\n"
    "\n"
    "Use confidence='verified' when they stated it directly. "
    "Use confidence='assumed' only for reasonable inferences (they implied it but didn't say it outright).\n"
    "\n"
    "Leave out:\n"
    "- the contents of their computer: files, paths, directories, commands, or any tool output "
    "(what they SAY about their own setup, hardware, software or games is a fact - keep it)\n"
    "- the assistant, this app, or this conversation\n"
    "- one-off requests and questions ('list my music', 'what's the weather')\n"
    "- anything you would have to invent — only what they actually said or clearly implied\n"
    "\n"
    "Reply with ONLY a JSON array of objects with 'text' and 'confidence' fields, "
    "each fact written in the third person about the user. "
    "Reply [] if they revealed nothing lasting about themselves."
)


# Facts that are close but not near-identical (cosine between these) get a second look:
# is the new one a paraphrase (skip it), an update that replaces the old one ("lives in
# Hamburg" -> "lives in Munich"), or simply a different fact about the same topic?
OVERLAP_MIN_SIMILARITY = 0.6

_JUDGE_PROMPT = (
    "You keep a list of facts about one person. A NEW fact may overlap with an OLD one.\n"
    "Answer with exactly one word:\n"
    "SAME - the new fact says nothing the old one does not already say (a paraphrase, or less detail)\n"
    "UPDATE - the new fact contradicts or replaces the old one (something changed, or the old one was wrong)\n"
    "DIFFERENT - both can be true at the same time; the new fact adds something\n"
    "\n"
    "OLD: lives in Hamburg / NEW: lives in Munich -> UPDATE\n"
    "OLD: likes Black Metal and Gothic Rock / NEW: favourite music is Black Metal -> SAME\n"
    "OLD: has a cat called Luna / NEW: has a dog called Rex -> DIFFERENT\n"
    "OLD: works as a developer / NEW: works as a software developer in Berlin -> UPDATE\n"
)
_VERDICT_RE = re.compile(r"\b(SAME|UPDATE|DIFFERENT)\b", re.IGNORECASE)


def judge_overlap(llm: Any, config: dict[str, Any], old: str, new: str) -> str:
    """"same" | "update" | "different". Any failure or unclear answer is "different": keep
    both, which is what happened before this check existed - never lose a fact on a guess."""
    try:
        reply = "".join(
            llm.chat(
                messages=[
                    {"role": "system", "content": _JUDGE_PROMPT},
                    {"role": "user", "content": f"OLD: {old}\nNEW: {new}"},
                ],
                model=config["models"]["background"],
                stream=False,
                background=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a failed judgement must not break distillation
        logger.warning("Fact overlap check failed: %s", exc)
        return "different"
    match = _VERDICT_RE.search(strip_thinking(reply))
    return match.group(1).lower() if match else "different"


def embed_with_retry(llm: Any, text: str, model: str, attempts: int = 3) -> list[float] | None:
    """Embed `text`, retrying transient failures. None if it never succeeds.

    Embedding runs right after a chat turn, while the chat model is still
    resident — on a smaller GPU there may not be room for the embedding model
    too, and Ollama answers 500. That's transient: a moment later it fits.
    """
    for attempt in range(attempts):
        try:
            return llm.embed([text], model=model)[0]
        except EmbeddingUnavailable:
            return None          # known missing/unsupported: no retries (the client already said so once)
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
    """Return the first JSON array of strings in `text`, or None if none parses.

    Kept for backward compatibility and direct testing. Production code uses
    _extract_facts_with_confidence, which handles the newer object format too.
    """
    parsed = extract_json_value(text, "[", "]")
    if not isinstance(parsed, list):
        return None
    return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]


def _extract_facts_with_confidence(
    text: str,
) -> list[dict[str, str]] | None:
    """Parse the distiller's reply into a list of {text, confidence} dicts.

    Accepts two formats from the model:
    - New: [{"text": "...", "confidence": "verified|assumed"}, ...]
    - Legacy fallback: ["...", ...] — all treated as "verified"

    Returns None if no parseable JSON array is found at all.
    """
    parsed = extract_json_value(text, "[", "]")
    if not isinstance(parsed, list) or not parsed:
        return None if not isinstance(parsed, list) else []

    result: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict):
            fact_text = str(item.get("text", "")).strip()
            confidence = str(item.get("confidence", "verified")).strip()
            if confidence not in ("verified", "assumed"):
                confidence = "verified"
        elif isinstance(item, str):
            fact_text = item.strip()
            confidence = "verified"
        else:
            continue
        if fact_text:
            result.append({"text": fact_text, "confidence": confidence})
    return result if result else []


@dataclass
class PlannedFact:
    """A candidate that passed every check and only waits to be written."""

    text: str
    confidence: str
    embedding: list[float] | None
    supersedes: list[int] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def extract_candidates(llm: Any, config: dict[str, Any], texts: list[str]) -> list[dict[str, str]] | None:
    """One background-model call over the user's own words. None = the reply held no parseable JSON array."""
    reply = "".join(
        llm.chat(
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": "\n".join(texts)},
            ],
            model=config["models"]["background"],
            stream=False,
            background=True,
        )
    )
    return _extract_facts_with_confidence(reply)


def plan_facts(
    store: Store, vectors: VectorIndex, llm: Any, config: dict[str, Any], candidates: list[dict[str, str]]
) -> list[PlannedFact]:
    """Apply the quality gates (junk filter, exact / near-identical duplicates, overlap judge) WITHOUT writing
    anything, so the slow part (embedding, judge calls) never holds a database write lock."""
    existing = {f["content"] for f in store.get_active_facts()}
    planned: list[PlannedFact] = []
    for item in candidates:
        fact = item["text"]
        if fact in existing or JUNK_FACT_RE.search(fact):
            continue
        embedding = embed_with_retry(llm, fact, config["models"]["embed"])

        supersedes: list[int] = []
        if embedding is not None:
            if any(p.embedding is not None and len(p.embedding) == len(embedding)
                   and _cosine(embedding, p.embedding) > DEDUPE_SIMILARITY for p in planned):
                continue                      # a paraphrase of a fact planned a moment ago in this same batch
            neighbours = vectors.similar(embedding, "fact", k=3, min_similarity=OVERLAP_MIN_SIMILARITY)
            if neighbours and neighbours[0][2] > DEDUPE_SIMILARITY:
                continue
            skip = False
            for old_id, old_text, _sim in neighbours:
                verdict = judge_overlap(llm, config, old_text, fact)
                if verdict == "same":
                    skip = True
                    break
                if verdict == "update" and old_id is not None:
                    supersedes.append(old_id)
            if skip:
                continue
        planned.append(PlannedFact(fact, item["confidence"], embedding, supersedes))
        existing.add(fact)
    return planned


def index_fact(vectors: VectorIndex, fact_id: int, p: PlannedFact) -> None:
    """Add the vector of a stored fact. A dimension mismatch (embedder changed without `reembed`) leaves the fact
    stored but unsearchable by meaning; it never raises."""
    if p.embedding is None:
        return
    try:
        vectors.add("fact", fact_id, p.text, p.embedding)
    except Exception as exc:  # noqa: BLE001 - the fact is saved; indexing can be repaired later
        logger.warning("Fact %s saved but not indexed: %s", fact_id, exc)


def distill_session(
    store: Store,
    vectors: VectorIndex,
    llm: Any,
    config: dict[str, Any],
    session_id: int,
) -> list[str]:
    """Extract, dedupe, store, and embed new facts. Returns the fact texts added.

    The whole-session, synchronous path (CLI tools, tests). Chats no longer call it at exit: facts are saved while
    chatting by agent.memory.queue, which uses the same extraction and gates."""
    history = store.get_messages(session_id)
    user_messages = [m for m in history if m["role"] == "user"]
    if len(user_messages) < MIN_USER_MESSAGES:
        return []

    # ONLY the user's own words. Feeding assistant replies and tool results (file
    # listings, web page text, command output) is what filled memory with file
    # paths and book plots — the model "extracted" them as facts.
    candidates = extract_candidates(llm, config, [m["content"] for m in user_messages])
    if candidates is None:
        logger.warning("Fact extraction returned no parseable JSON array; skipping.")
        return []

    added: list[str] = []
    for p in plan_facts(store, vectors, llm, config, candidates):
        for old_id in p.supersedes:
            store.deactivate_fact(old_id)       # the new fact replaces the one it contradicts
        fact_id = store.add_fact(p.text, source_session_id=session_id, confidence=p.confidence)
        index_fact(vectors, fact_id, p)
        added.append(p.text)

    # Catch up anything a previous run stored but couldn't embed — otherwise the
    # agent keeps facts it can never recall.
    repair_unembedded(store, vectors, llm, config)
    return added
