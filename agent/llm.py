"""Thin client around the local Ollama HTTP API.

Kept deliberately small so the chat loop (tui.py) and later persistence /
memory tasks can depend on a stable interface without knowing about HTTP,
streaming, or NDJSON parsing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx


def _default_base_url() -> str:
    """Ollama host, honoring the standard OLLAMA_HOST env var.

    On an offload node (e.g. the laptop) set OLLAMA_HOST to the main PC's
    address and every client here talks to that server with no code change —
    the same variable Ollama's own CLI uses. Accepts `host`, `host:port`, or a
    full `http(s)://…` URL.
    """
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    if not raw:
        return "http://localhost:11434"
    if "://" in raw:
        return raw.rstrip("/")
    if ":" not in raw:
        raw += ":11434"
    return f"http://{raw}"


DEFAULT_BASE_URL = _default_base_url()

# Embedding models have a fixed context window; inputs longer than this are
# truncated before embedding so a single oversized text can't 400 the whole
# batch. ~4000 chars stays comfortably under nomic-embed-text's ~2048 tokens.
EMBED_MAX_CHARS = 4000

# Keep the model resident in VRAM this long after each call so back-to-back CLI
# commands (chat, then a summary, then a profile) don't each pay dolphin's cold
# reload — the single biggest speedup on repeated use. It frees after 30 min
# idle, so it won't hold VRAM while you game.
KEEP_ALIVE = "30m"


def _is_tools_unsupported_error(response: httpx.Response) -> bool:
    """True if `response` is Ollama's 400 for a model that rejects `tools=`."""
    if response.status_code != 400:
        return False
    try:
        error = response.json().get("error", "")
    except ValueError:
        return False
    return "does not support tools" in error


class OllamaClient:
    """Talks to a local Ollama server. No other network access is used."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._base_url = base_url
        # Long read timeout (a 27B generation is slow) but a short *connect*
        # timeout, so a mistyped OLLAMA_HOST fails in seconds, not 10 minutes.
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        self._tools_unsupported: set[str] = set()
        self._tool_support: dict[str, bool | None] = {}

    def supports_tools(self, model: str) -> bool | None:
        """Does `model` advertise tool support? None if it can't be determined.

        This matters more than it looks: Ollama only rejects `tools=` outright
        for *some* models. For others (dolphin3, gemma2) it accepts the request
        and silently drops the tools, because the model's chat template has no
        slot for them. The model never learns the tools exist, so when you ask
        it to list a directory it invents a plausible listing instead of calling
        anything. Asking up front is the only way to catch that.
        """
        if model in self._tool_support:
            return self._tool_support[model]
        try:
            response = self._client.post("/api/show", json={"model": model})
            response.raise_for_status()
            caps = response.json().get("capabilities")
            result = ("tools" in caps) if isinstance(caps, list) else None
        except (httpx.HTTPError, ValueError):
            result = None  # unknown — never cry wolf on a transport hiccup
        self._tool_support[model] = result
        return result

    def loaded_models(self) -> list[str]:
        """Models currently resident in memory, per Ollama's /api/ps."""
        try:
            response = self._client.get("/api/ps")
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", []) if m.get("name")]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    def unload(self, model: str) -> bool:
        """Evict `model` from memory now. True if Ollama accepted it.

        `keep_alive: 0` with no prompt is Ollama's documented way to unload —
        it's what `ollama stop` does.
        """
        try:
            response = self._client.post(
                "/api/generate", json={"model": model, "keep_alive": 0}
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def unload_all(self) -> list[str]:
        """Free every resident model. Returns the ones actually unloaded."""
        return [m for m in self.loaded_models() if self.unload(m)]

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        stream: bool = True,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Send a chat request and yield assistant content tokens as they arrive.

        `num_ctx` sets the context window Ollama allocates for this call.
        Ollama's default is small (~4k), which silently truncates long inputs.
        `num_gpu` caps how many layers go on the GPU — needed for a model too
        big for VRAM (e.g. Gemma-27B on 8GB), where Ollama's auto-split
        over-commits the GPU and crashes with cudaMalloc OOM. `options` passes
        any additional Ollama sampling options straight through (repeat_penalty,
        top_p, top_k) — coding mode needs these and they merge under the explicit
        arguments above. Raises httpx exceptions if Ollama is unreachable;
        callers handle those.
        """
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": stream, "keep_alive": KEEP_ALIVE,
        }
        merged: dict[str, Any] = dict(options) if options else {}
        if num_ctx:
            merged["num_ctx"] = num_ctx
        if num_gpu is not None:
            merged["num_gpu"] = num_gpu
        if temperature is not None:
            merged["temperature"] = temperature
        if merged:
            payload["options"] = merged

        with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        """Send a non-streaming chat request with a `tools` schema and return
        the raw `message` dict (may contain `content` and/or `tool_calls`).

        Ollama does not reliably stream tool calls, so this always sends
        stream=False; plain conversational turns should keep using `chat()`.

        Some models (e.g. mixtral:8x7b) reject `tools=` outright with a 400
        "does not support tools" response. When that happens, this retries
        the same request without `tools` so the plain-text reply can still
        flow through the JSON fallback tool-call parser, and remembers the
        model for the rest of the process so later calls skip the doomed
        tools= attempt entirely.
        """
        if model in self._tools_unsupported:
            return self._post_chat(messages, model, tools=None, num_ctx=num_ctx)

        try:
            return self._post_chat(messages, model, tools=tools, num_ctx=num_ctx)
        except httpx.HTTPStatusError as exc:
            if not _is_tools_unsupported_error(exc.response):
                raise
            self._tools_unsupported.add(model)
            return self._post_chat(messages, model, tools=None, num_ctx=num_ctx)

    def _post_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False, "keep_alive": KEEP_ALIVE}
        if tools is not None:
            payload["tools"] = tools
        if num_ctx:
            payload["options"] = {"num_ctx": num_ctx}
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed a batch of texts via the local embedding model.

        Returns one vector per input text (Ollama `/api/embed`). Raises httpx
        exceptions if Ollama is unreachable; callers embedding memory are
        expected to catch those so a failed embed never breaks chat.

        Each input is truncated to ``EMBED_MAX_CHARS`` first: embedding models
        have a fixed context window (nomic-embed-text ~2048 tokens) and Ollama
        rejects the *entire* batch with a 400 if any single input exceeds it,
        so one pasted article/code dump would otherwise sink 49 innocent
        messages. The leading slice is more than enough signal for semantic
        search.
        """
        capped = [t[:EMBED_MAX_CHARS] for t in texts]
        response = self._client.post("/api/embed", json={"model": model, "input": capped})
        response.raise_for_status()
        return response.json()["embeddings"]
