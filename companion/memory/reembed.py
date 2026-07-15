"""Re-embed all stored memory with a new embedding model.

Swapping the embedding model (e.g. nomic-embed-text -> a multilingual model like
bge-m3) changes the vector size, so the whole vec0 index has to be rebuilt and
every item re-embedded. The source text lives in `memory_items`, so this reads
from there — no need to re-scan anything else.

Governor-aware (yields while gaming) and resumable-ish: it rebuilds the vector
table from scratch each run, committing per batch. Safe to re-run.
"""

from __future__ import annotations

from typing import Any, Callable

from companion.memory.vectors import _load_extension, _pack


def _probe_dim(llm: Any, model: str) -> int:
    vec = llm.embed(["dimension probe"], model)[0]
    return len(vec)


def reembed_all(
    store,
    llm: Any,
    config: dict[str, Any],
    model: str,
    batch: int = 64,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, int]:
    """Rebuild the vector index with `model`. Returns counts.

    Steps: probe the model's vector size, drop+recreate `memory_vectors` at that
    size, persist the size (so VectorIndex agrees), then re-embed every
    `memory_items.text` in batches. Runs at full speed (not governor-paced): it's
    an explicit, user-launched job whose own embedding load would otherwise make
    the GPU-busy check pause it against itself. Ctrl-C to stop; it's resumable."""
    conn = store.conn
    try:
        _load_extension(conn)
    except Exception:  # noqa: BLE001 - surfaced clearly to the caller
        raise RuntimeError("sqlite-vec unavailable; cannot rebuild the vector index") from None

    dim = _probe_dim(llm, model)
    log(f"{model} → {dim}-dim; rebuilding index")

    conn.execute("DROP TABLE IF EXISTS memory_vectors")
    conn.execute(f"CREATE VIRTUAL TABLE memory_vectors USING vec0(embedding float[{dim}])")
    conn.execute(
        "INSERT INTO app_state (k, v) VALUES ('embed_dim', ?) "
        "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (str(dim),),
    )
    conn.commit()

    rows = conn.execute(
        "SELECT id, text FROM memory_items WHERE text IS NOT NULL AND text != '' ORDER BY id"
    ).fetchall()
    total = len(rows)
    done = failed = 0
    for start in range(0, total, batch):
        chunk = rows[start : start + batch]
        try:
            embs = llm.embed([r["text"] for r in chunk], model)
        except Exception:  # noqa: BLE001 - one bad batch shouldn't abort the whole run
            failed += len(chunk)
            log(f"batch at {start} failed; continuing")
            continue
        for r, e in zip(chunk, embs, strict=False):
            conn.execute(
                "INSERT INTO memory_vectors (rowid, embedding) VALUES (?, ?)",
                (r["id"], _pack(e)),
            )
        conn.commit()
        done += len(chunk)
        if start % (batch * 20) == 0 or start + batch >= total:
            log(f"re-embedded {done}/{total}")

    return {"reembedded": done, "failed": failed, "dim": dim, "total": total}
