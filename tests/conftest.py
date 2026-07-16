from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_embedding_backoff(monkeypatch):
    """Don't make the suite wait out the embedding retry's backoff.

    `embed_with_retry` sleeps between attempts so a busy GPU has a moment to free
    up — real and wanted in production, pure dead time in tests (it took the suite
    from 2s to 96s). The retry logic itself still runs; only the waiting is gone.
    """
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _seconds: None)
