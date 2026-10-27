"""A missing / non-embedding model: one calm notice, no retries, keyword recall instead."""
import logging

import httpx
import pytest

from agent import llm
from agent.llm import EmbeddingUnavailable, OllamaClient
from agent.memory.distill import embed_with_retry


@pytest.fixture(autouse=True)
def _fresh():
    llm.reset_embedding_state()
    yield
    llm.reset_embedding_state()


def _client(status, calls):
    def fake(request):
        calls.append(request.url.path)
        if status == 200:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        return httpx.Response(status, json={"error": "model not found"})
    return OllamaClient(base_url="http://x", transport=httpx.MockTransport(fake))


@pytest.mark.parametrize("status", [404, 501])
def test_missing_model_is_one_request_one_notice(status, caplog):
    calls: list[str] = []
    c = _client(status, calls)
    with caplog.at_level(logging.WARNING, logger="agent.llm"):
        for _ in range(4):
            with pytest.raises(EmbeddingUnavailable):
                c.embed(["a"], model="bge-m3")
    assert len(calls) == 1                                   # cached after the first failure
    notes = [r.getMessage() for r in caplog.records]
    assert notes == ["semantic memory is off: embedding model 'bge-m3' is not available - run: ollama pull bge-m3"]


def test_reprobe_after_the_recheck_window(monkeypatch):
    calls: list[str] = []
    c = _client(404, calls)
    with pytest.raises(EmbeddingUnavailable):
        c.embed(["a"], model="m")
    llm._embed_down["m"] -= llm.EMBED_RECHECK_S + 1
    with pytest.raises(EmbeddingUnavailable):
        c.embed(["a"], model="m")
    assert len(calls) == 2


def test_server_error_500_is_not_cached():
    calls: list[str] = []
    c = _client(500, calls)
    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            c.embed(["a"], model="m")
    assert not llm.embedding_unavailable("m")


def test_embed_with_retry_does_not_retry_when_unavailable(monkeypatch):
    slept = []
    monkeypatch.setattr("agent.memory.distill.time.sleep", slept.append)

    class Fake:
        n = 0
        def embed(self, texts, model):
            Fake.n += 1
            raise EmbeddingUnavailable(model)
    assert embed_with_retry(Fake(), "t", "m") is None
    assert Fake.n == 1 and slept == []


def test_recall_falls_back_to_keywords(tmp_path, caplog):
    from agent.memory import recall
    class V:
        available = True
    class Boom:
        def embed(self, *a, **k):
            raise EmbeddingUnavailable("m")
    seen = {}
    orig = recall._keyword_hits
    recall._keyword_hits = lambda s, c, u: seen.setdefault("hit", ["kw"])
    try:
        with caplog.at_level(logging.WARNING):
            out = recall._semantic_hits(V(), Boom(), {"memory": {"recall_k": 3}, "models": {"embed": "m"}}, "hello", None)
    finally:
        recall._keyword_hits = orig
    assert out == ["kw"] and not caplog.records


def test_stop_event_ends_a_stream_that_is_still_reading_the_prompt():
    """Nothing arrives for a callback to raise in: the cancel event itself must end the call (real socket, local server)."""
    import http.server
    import threading
    import time as _t

    class Silent(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.end_headers()
            self.wfile.flush()
            threading.Event().wait(8)                     # a prompt evaluation on a CPU: headers, then silence

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Silent)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cancel = threading.Event()
    try:
        c = OllamaClient(base_url=f"http://127.0.0.1:{server.server_port}")
        threading.Timer(0.5, cancel.set).start()
        t0 = _t.monotonic()
        out = c.chat_with_tools_stream([{"role": "user", "content": "hi"}], "m", [], cancel=cancel)
        assert _t.monotonic() - t0 < 4 and out.get("_aborted")
    finally:
        server.shutdown()
