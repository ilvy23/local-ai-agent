"""Does remembering actually work here? A throwaway round trip through the real memory code.

Writes one fact into a temporary database, then asks a question about it the way a chat does (`recall._semantic_hits`):
by meaning with the configured embedding model when it is reachable, by keyword when it is not. Nothing of the user's
own data is read or written. Used by `nightage doctor` (check `memory.roundtrip`) and `nightage memory check`."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.llm import EmbeddingUnavailable

FACT = "has a cat called Pixel"
DISTRACTOR = "lives in Lisbon"
QUESTION = "What is my cat called?"


@dataclass
class Roundtrip:
    status: str          # ok | warn | fail
    detail: str          # one sentence with the exact reason
    path: str            # meaning | keyword | none


class _Embedder:
    """The shape `recall._semantic_hits` expects (`.embed(texts, model=...)`), over any callable."""

    def __init__(self, fn: Callable[[list[str]], list[list[float]]] | None, why: str = "") -> None:
        self.fn, self.why = fn, why

    def embed(self, texts: list[str], model: str = "", **_: Any) -> list[list[float]]:
        if self.fn is None:
            raise EmbeddingUnavailable(self.why or model)
        return self.fn(texts)


def roundtrip(config: dict[str, Any], embed: Callable[[list[str]], list[list[float]]] | None = None, *,
              unavailable: str = "") -> Roundtrip:
    """`embed(texts) -> vectors` is the configured embedder, or None when it is not reachable (`unavailable` says why)."""
    from agent.memory.recall import _semantic_hits
    from agent.memory.store import Store
    from agent.memory.vectors import VectorIndex

    model = (config.get("models") or {}).get("embed", "?")
    cfg = {"models": {"embed": model}, "memory": {**(config.get("memory") or {}), "recall_k": 4}}
    with tempfile.TemporaryDirectory(prefix="nightage-memcheck-") as tmp:
        with Store(Path(tmp) / "check.db") as store:
            sid = store.create_session()
            for text in (FACT, DISTRACTOR):
                store.add_fact(text, sid)
            vectors = VectorIndex(store)
            by_keyword = FACT in _semantic_hits(_NoVectors(vectors), _Embedder(None), cfg, QUESTION, store)
            if not by_keyword:
                return Roundtrip("fail", "a fact was written but the keyword search could not find it again", "none")
            if not vectors.available:
                return Roundtrip("warn", "the vector extension (sqlite-vec) is not available here: facts are saved and found "
                                 "by keyword only", "keyword")
            if embed is None:
                why = unavailable or "not reachable"
                return Roundtrip("warn", f"embedding model {model} is {why}: facts are saved and found by keyword, not by meaning", "keyword")
            try:
                for text in (FACT, DISTRACTOR):
                    vectors.add("fact", store.conn.execute("SELECT id FROM facts WHERE content = ?", (text,)).fetchone()[0],
                                text, embed([text])[0])
                strict = {**cfg, "memory": {**cfg["memory"], "recall_min_similarity": 0.0}}
                hits = _semantic_hits(vectors, _Embedder(embed), strict, QUESTION, store)
                kept = _semantic_hits(vectors, _Embedder(embed), cfg, QUESTION, store)
            except Exception as exc:  # noqa: BLE001 - the reason is the point
                return Roundtrip("warn", f"embedding model {model} failed ({type(exc).__name__}: {exc}): facts are found by keyword only", "keyword")
            if not hits or hits[0] != FACT:
                return Roundtrip("warn", f"embedding model {model} did not rank the saved fact first for a question about it "
                                 f"(got {hits[:2]!r}): recall by meaning is unreliable with it", "keyword")
            if FACT not in kept:
                return Roundtrip("warn", f"the saved fact is found by meaning with {model}, but its similarity is below "
                                 f"memory.recall_min_similarity ({cfg['memory'].get('recall_min_similarity')}): lower that value", "meaning")
            return Roundtrip("ok", f"a saved fact was recalled by meaning with {model} ({len(embed([FACT])[0])}-dim)", "meaning")


class _NoVectors:
    """A VectorIndex stand-in that reports 'unavailable', which sends `_semantic_hits` down its keyword path."""

    available = False

    def __init__(self, real: Any) -> None:
        self.DIM = getattr(real, "DIM", None)
