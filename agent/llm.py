"""Thin client around the local Ollama HTTP API.

Kept deliberately small so the chat loop (tui.py) and later persistence /
memory tasks can depend on a stable interface without knowing about HTTP,
streaming, or NDJSON parsing.

ENGINE SEAM - the interface the coding agent (codeagent/engine, stages, supervisor, session, reviewer, context) depends on.
Anything else on OllamaClient (option shaping, /api/ps, keep_alive, Vulkan routing, speculative decoding) is private to this
module; a backend that offers only the REQUIRED methods must work.

  REQUIRED
    chat_with_tools(messages, model, tools, num_ctx=None, think=None, extra_options=None) -> message dict
        ({"role", "content", "tool_calls"?, "thinking"?, "_prompt_eval_count"?, "_eval_count"?}; extra_options carries
        temperature / top_p / top_k / repeat_penalty / num_predict - a backend maps or ignores what it does not know)
    chat(messages=[...], model=..., stream=False, background=True) -> iterable of text pieces
  OPTIONAL (the module-level helpers below return a neutral default when a method is missing or raises)
    chat_with_tools_stream(..., on_content=, on_thinking=)   streaming variant; without it the engine uses chat_with_tools
    server_url(model) -> str                                  health-probe target; without it a failed call is just resent
    revive(model) -> bool                                     restart a dead secondary server
    loaded_models() / unload(model) / unload_all()            free VRAM before and after a session
    set_read_timeout(seconds)                                 coding.read_timeout_s

Engine code must call the optional ones through server_url_of / revive_server / loaded_models_of / unload_model /
unload_all_models / set_read_timeout / streaming_chat (bottom of this module) and must not name Ollama endpoints or settings.
"""

from __future__ import annotations

import json
import logging
import os
import re
import contextlib
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)


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

# --- Remote gateway (phone UI + model on the home PC) ------------------------------------------------------------------
# `ollama.host` in config.yaml (or OLLAMA_HOST in the environment, which wins) points every client at another machine,
# e.g. a PC elsewhere on your network or VPN: `host: 192.0.2.10:11434`. A remote host is used for ALL models: the local dual-GPU
# Vulkan server (127.0.0.1:11436) is skipped, because it would be this machine's, not the gateway's.
_CONFIG_HOST = ""


def normalize_host(raw: str) -> str:
    """`host`, `host:port` or a full URL -> `http(s)://host:port` ('' for empty input)."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        return raw.rstrip("/")
    if raw.count(":") == 0 or raw.endswith("]"):
        raw += ":11434"
    return f"http://{raw}"


def current_base_url() -> str:
    """The default Ollama server now: OLLAMA_HOST env > `ollama.host` (see `configure`) > localhost:11434."""
    if os.environ.get("OLLAMA_HOST", "").strip():
        return _default_base_url()
    return normalize_host(_CONFIG_HOST) or DEFAULT_BASE_URL


def is_remote(url: str | None = None) -> bool:
    """True when the Ollama server is not on this machine (loopback / localhost / unix hostnames count as local)."""
    from urllib.parse import urlparse

    host = (urlparse(url or current_base_url()).hostname or "").lower()
    return host not in ("", "localhost", "127.0.0.1", "::1", "0.0.0.0") and not host.endswith(".localhost")


def configure(config: dict | None) -> None:
    """Apply the `ollama:` block of a loaded config: `host` (remote gateway) and `keep_alive`. Cheap; called by load_config."""
    global _CONFIG_HOST, KEEP_ALIVE
    block = (config or {}).get("ollama") if isinstance(config, dict) else None
    block = block if isinstance(block, dict) else {}
    _CONFIG_HOST = str(block.get("host") or "").strip()
    ka = str(block.get("keep_alive") or "").strip()
    if ka:
        KEEP_ALIVE = ka

# --- Big models on the Vulkan Ollama (spans BOTH GPUs) ----------------------
# A 27B model does not fit one card. A second Ollama server built on the Vulkan
# backend can split layers across an AMD and an NVIDIA GPU at once, which the
# ROCm/CUDA servers cannot. Any model whose name matches AGENT_VULKAN_MODELS is
# sent to that server (AGENT_VULKAN_OLLAMA_URL); everything else keeps using
# OLLAMA_HOST. Set AGENT_VULKAN_MODELS="" to switch the routing off.
#
#   AGENT_VULKAN_MODELS      regex on the model name   (default: every generating model, no embedders)
#   AGENT_VULKAN_OLLAMA_URL  that server's URL         (default: 127.0.0.1:11436)
#   AGENT_VULKAN_START_CMD   command that starts it     (default: `ollama-vk list`,
#                            found on PATH or in ~/.local/bin; the server is
#                            started on demand and left running)
_VULKAN_DEFAULT_MODELS = r"^(?!.*(?:embed|bge-|nomic|minilm)).+"      # every generating model; embedding models never
_VULKAN_DEFAULT_URL = "http://127.0.0.1:11436"
_VULKAN_START_TIMEOUT_S = 90.0


def vulkan_base_url() -> str:
    raw = os.environ.get("AGENT_VULKAN_OLLAMA_URL", "").strip() or _VULKAN_DEFAULT_URL
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


_VULKAN_ENV = ("AGENT_VULKAN_MODELS", "AGENT_VULKAN_OLLAMA_URL", "AGENT_VULKAN_START_CMD")
_VULKAN_PROBE_TTL_S = 60.0
_vulkan_probe: tuple[float, bool] | None = None   # (checked at, result)


def _vulkan_configured() -> bool:
    return any(os.environ.get(v, "").strip() for v in _VULKAN_ENV)


def vulkan_available() -> bool:
    """Is there a Vulkan Ollama to route to at all? Routing is opt-in by capability:
    configured through AGENT_VULKAN_* (the owner said so), or discoverable (the ollama-vk unit /
    command exists, or the URL answers within a moment) AND a Vulkan-capable GPU is present.
    A machine without either simply uses the default server, silently. Cached for a minute."""
    global _vulkan_probe
    if _vulkan_configured():
        return True
    now = time.monotonic()
    if _vulkan_probe is not None and now - _vulkan_probe[0] < _VULKAN_PROBE_TTL_S:
        return _vulkan_probe[1]
    found = False
    try:
        from agent import gpumode

        if gpumode.gpu_mode_available():
            found = _vulkan_start_command() is not None
            if not found:
                try:
                    found = httpx.get(vulkan_base_url() + "/api/version", timeout=0.5).status_code == 200
                except httpx.HTTPError:
                    found = False
    except Exception:  # noqa: BLE001 - detection must never break a chat
        found = False
    _vulkan_probe = (now, found)
    log.debug("Vulkan Ollama routing %s", "on" if found else "off (no GPU / no server found)")
    return found


def uses_vulkan(model: str) -> bool:
    """True if `model` should run on the dual-GPU Vulkan Ollama server."""
    pattern = os.environ.get("AGENT_VULKAN_MODELS")
    if pattern is None:
        if not vulkan_available():
            return False
        pattern = _VULKAN_DEFAULT_MODELS
    pattern = pattern.strip()
    if not pattern or not model:
        return False
    try:
        return re.search(pattern, model, re.IGNORECASE) is not None
    except re.error:
        log.warning("AGENT_VULKAN_MODELS is not a valid regex: %r", pattern)
        return False


def wants_vulkan(model: str) -> bool:
    """Should `model` run on the dual-GPU Vulkan server?

    The session's GPU choice (agent.gpumode: asked once at the start) wins: "dual"
    sends every generating model there, "single" keeps everything on the default
    server. With no choice made (non-interactive runs) the name pattern decides."""
    from agent import gpumode

    if is_remote():                      # a remote gateway serves everything; the local Vulkan server is not part of it
        return False
    mode = gpumode.current_mode()
    if mode == gpumode.DUAL:
        return bool(model)
    if mode == gpumode.SINGLE:
        return False
    return uses_vulkan(model)


def _vulkan_unit_exists() -> bool:
    """True when a persistent user unit ollama-vk.service is installed (then `systemctl --user start` is the way to bring it up)."""
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    try:
        return subprocess.run([systemctl, "--user", "cat", "ollama-vk.service"], capture_output=True, timeout=5, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _vulkan_start_command() -> list[str] | None:
    """The command that brings the Vulkan server up, or None if none is known."""
    custom = os.environ.get("AGENT_VULKAN_START_CMD", "").strip()
    if custom:
        import shlex

        return shlex.split(custom)
    if _vulkan_unit_exists():            # the persistent user unit wins: the wrapper's `systemd-run --unit=ollama-vk` cannot start a stopped service
        return ["systemctl", "--user", "start", "ollama-vk.service"]
    found = shutil.which("ollama-vk")
    if not found:
        local = Path.home() / ".local" / "bin" / "ollama-vk"
        found = str(local) if local.is_file() and os.access(local, os.X_OK) else None
    return [found, "list"] if found else None

# Embedding models have a fixed context window; inputs longer than this are
# truncated before embedding so a single oversized text can't 400 the whole
# batch. ~4000 chars stays comfortably under nomic-embed-text's ~2048 tokens.
EMBED_MAX_CHARS = 4000

# Keep the model resident in VRAM this long after each call so back-to-back CLI
# commands (chat, then a summary, then a profile) don't each pay a cold
# reload — the single biggest speedup on repeated use. It frees after 30 min
# idle, so it won't hold VRAM while you game.
KEEP_ALIVE = os.environ.get("AGENT_KEEP_ALIVE", "").strip() or "30m"   # `ollama.keep_alive` (config) rebinds it, see configure()

# Bounds a single tool-loop turn's worst-case generation length. Ollama's own
# default (num_predict: -1) is unbounded - fine for a streamed reply the user
# can watch and interrupt, but chat_with_tools's request is non-streaming, so
# an unbounded reasoning pass just blocks in silence for however long the
# model rambles, up to the full context window. 2048 tokens is generous for
# a tool-call decision or a normal reply and was never approached by a
# well-behaved turn during testing.
DEFAULT_NUM_PREDICT = 2048
BACKGROUND_NUM_PREDICT = 3072  # utility calls: JSON facts / a verdict / a summary, plus hidden thinking
_MAX_AUTO_NUM_PREDICT = 32768


# Ollama 0.34's MTP self-speculative decoding (the qwen3.5/3.8 family ships its own
# draft head) is lossless - the main model verifies every drafted token - but the
# Modelfile default of 4 drafted tokens was too greedy on the dual-GPU setup it was
# measured on (a 27B model: 4 -> 19 tok/s, 3 -> 24, 2 -> 36, 1 -> 32). `AGENT_DRAFT_TOKENS=<n>` overrides, `-1` leaves the model's default.
DEFAULT_DRAFT_TOKENS = 4


def speculative_options(model: str) -> dict[str, Any]:
    """`draft_num_predict` for models served by the Vulkan (0.34+, MTP-capable) server."""
    raw = os.environ.get("AGENT_DRAFT_TOKENS", "").strip()
    value = int(raw) if raw.lstrip("-").isdigit() else DEFAULT_DRAFT_TOKENS
    if value < 0 or not wants_vulkan(model):
        return {}
    return {"draft_num_predict": value}


# Back-compat: this used to be a qwen-only ceiling; it's now the default for every model's
# context budget (agent.ctxbudget), overridable via AGENT_CTX_CEILING or `models.max_ctx_ceiling`.
# AGENT_MAX_QWEN_CTX still works as an alias so existing setups don't silently change behaviour.
# Keep OLLAMA_CONTEXT_LENGTH of the server at the same value: a different num_ctx between the
# preload and the first chat makes Ollama reload the model.
MAX_QWEN_CTX = int(os.environ.get("AGENT_CTX_CEILING", "").strip()
                    or os.environ.get("AGENT_MAX_QWEN_CTX", "").strip() or 95000)


def cap_ctx(model: str | None, num_ctx: int | None, config: dict | None = None,
            hw_summary: Any | None = None) -> int | None:
    """`num_ctx` clamped to this model's computed context budget (agent.ctxbudget.budget_for) -
    free VRAM/RAM, a KV-cache heuristic, and the ceiling above, for EVERY model (no more
    name-based qwen special case). An explicit `models.overrides.<model>.num_ctx` always wins.
    `None`/falsy `num_ctx` is passed through unchanged, same as before. `hw_summary` lets a
    caller (chiefly tests) supply fake hardware instead of real detection."""
    if not num_ctx:
        return num_ctx
    from . import ctxbudget

    return ctxbudget.budget_for(model, None, hw_summary=hw_summary, requested=int(num_ctx), config=config)


# Context size last used per model. A differing `num_ctx` makes Ollama reload the model (8-10 s
# for the 27B), so utility calls on the chat model reuse the chat's window instead of the default.
_last_num_ctx: dict[str, int] = {}


def _remember_ctx(model: str, num_ctx: int | None) -> None:
    if num_ctx:
        _last_num_ctx[model] = int(num_ctx)


_CLEAN_THINK_OFF = re.compile(r"qwen3\.[5-9]", re.IGNORECASE)


def supports_think_off(model: str) -> bool:
    """True if `think: false` returns clean content for `model` on Ollama 0.34.

    Checked live: qwen3.5+ (renderer/parser pair) does; the older qwen3:4b leaks its raw
    chain-of-thought into `content` and burns the whole token budget instead."""
    return bool(_CLEAN_THINK_OFF.search(model or ""))


def output_token_cap(num_ctx: int | None) -> int | None:
    """Most tokens one tool-loop turn may generate (None = unlimited).

    A fixed 2048 is far too little for work like "merge these two long files into one
    document": the written document is itself the model's output (a write_file argument),
    and for reasoning models the hidden thinking counts against the same budget - hitting
    the cap leaves a cut-off tool call or an empty reply. So the cap grows with the context
    window: a quarter of it, between 2048 and 32768 tokens. `AGENT_MAX_OUTPUT=<n>` sets it
    explicitly; `0` (or a negative number) removes the cap entirely."""
    raw = os.environ.get("AGENT_MAX_OUTPUT", "").strip()
    if raw.lstrip("-").isdigit():
        value = int(raw)
        return value if value > 0 else None
    if not num_ctx:
        return DEFAULT_NUM_PREDICT
    return max(DEFAULT_NUM_PREDICT, min(int(num_ctx) // 4, _MAX_AUTO_NUM_PREDICT))


class EmbeddingUnavailable(RuntimeError):
    """The embedding model is missing or cannot embed on this server (HTTP 404 / 501).

    Remembered per model for a few minutes, so a missing model costs one request, not three per message.
    Memory callers treat it as "no semantic memory right now" and fall back to keyword recall."""


EMBED_RECHECK_S = 300.0
_embed_down: dict[str, float] = {}     # model -> when it was found unavailable
_embed_notice_shown = False


def embedding_unavailable(model: str) -> bool:
    """True while `model` is known not to embed (until the periodic re-probe)."""
    since = _embed_down.get(model)
    return since is not None and time.monotonic() - since < EMBED_RECHECK_S


def reset_embedding_state() -> None:
    global _embed_notice_shown
    _embed_down.clear()
    _embed_notice_shown = False


def _note_embedding_down(model: str) -> None:
    """Remember the failure and say it ONCE per session, calmly."""
    global _embed_notice_shown
    _embed_down[model] = time.monotonic()
    if not _embed_notice_shown:
        _embed_notice_shown = True
        log.warning("semantic memory is off: embedding model '%s' is not available - run: ollama pull %s", model, model)


class GenerationAborted(Exception):
    """Raised by a streaming callback to stop the generation (the message is the reason)."""


def _is_tools_unsupported_error(response: httpx.Response) -> bool:
    """True if `response` is Ollama's 400 for a model that rejects `tools=`."""
    if response.status_code != 400:
        return False
    try:
        error = response.json().get("error", "")
    except ValueError:
        return False
    return "does not support tools" in error


_DEFAULT_ROUTING_CONFIG: dict = {
    "models": {"main": "qwen3.5:9b", "orchestrator": "qwen3.5:9b", "fallback": "qwen3.5:9b"}
}

# Whole words only: plain substring matching routed "encoding", "decoding" and
# "barcoding" to the big model because they all contain "coding".
_CODING_PLANNING_RE = re.compile(r"\b(?:coding|planning)\b")


# ---------------------------------------------------------------------------
# DeepSeek R1 helpers
# ---------------------------------------------------------------------------

def is_deepseek_r1(model: str) -> bool:
    """True if `model` is a DeepSeek R1 variant (any parameter size)."""
    return "deepseek-r1" in model.lower()


def strip_thinking(text: str) -> str:
    """Remove DeepSeek R1 <think>...</think> reasoning blocks from a response.

    DeepSeek R1 prefixes its answer with a chain-of-thought block wrapped in
    <think>…</think> tags.  End-users should see only the final answer; this
    function strips those blocks so callers need not care which model produced
    the text.
    """
    # Remove complete <think>…</think> blocks (there may be several).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove a dangling, unclosed <think> (e.g. output truncated mid-thought):
    # strip from the stray opening tag to the end, or the raw reasoning leaks
    # through to the user because the paired pattern above never matched.
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL)
    return text.strip()


def _log_think_block(model: str, text: str) -> None:
    """Log the <think> block from a DeepSeek R1 response before it is stripped.

    Never raises — a broken trajectory write must not interrupt a chat turn.
    """
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return
    think_content = match.group(1)
    try:
        from agent.trajectory import log_action  # noqa: PLC0415
        log_action("reasoning", None, model, think_content[:500])
    except Exception:  # noqa: BLE001
        pass


def _adapt_system_prompt(model: str, system_prompt: str) -> str:
    """Shorten system prompts for models that perform better with terse context.

    DeepSeek R1 is instruction-tuned on short, direct prompts; long role
    descriptions with elaborate framing tend to hurt rather than help.  We
    keep the first meaningful sentence (up to 500 chars, split at the last
    period) so the model still knows its role without drowning in preamble.
    """
    if is_deepseek_r1(model) and len(system_prompt) > 500:
        return system_prompt[:500].rsplit(".", 1)[0] + "."
    return system_prompt


HARNESS_PREFIX = "[harness note] "


def fold_system_messages(messages: list[dict]) -> list[dict]:
    """Make system messages survive the chat template of the Qwen3.5+ models.

    Measured on qwen3.8:27b via Ollama (a codeword that appears only in the message): a `system` message that comes
    directly AFTER an assistant text turn is not seen - the model answers with a single end-of-turn token, deterministically
    (it did so 291 times in a row in a real run). Every nudge an agent appends after the model's own text (plan reminders,
    test failures, syntax errors) has exactly that shape. System messages elsewhere are seen, user messages everywhere.
    So the leading system messages are merged into ONE system message and every later one is sent as a user message
    starting with `[harness note]`. Deterministic per message, so the prompt prefix (and the server's cache) stays stable."""
    out: list[dict] = []
    head: list[str] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        head.append(str(messages[i].get("content", "")))
        i += 1
    if head:
        out.append({"role": "system", "content": "\n\n".join(h for h in head if h)})
    for m in messages[i:]:
        if m.get("role") == "system":
            out.append({"role": "user", "content": HARNESS_PREFIX + str(m.get("content", ""))})
        else:
            out.append(m)
    return out


def _adapt_messages(model: str, messages: list[dict]) -> list[dict]:
    """Apply model-specific message pre-processing before sending to Ollama.

    System messages are folded (see `fold_system_messages`) and DeepSeek R1 additionally gets its system prompt
    truncated by `_adapt_system_prompt`. Returns new lists; the caller's `messages` is never mutated."""
    messages = fold_system_messages(messages)
    if not is_deepseek_r1(model):
        return messages
    adapted = []
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            adapted.append({**msg, "content": _adapt_system_prompt(model, msg["content"])})
        else:
            adapted.append(msg)
    return adapted


# A cold 78k-token prompt takes 120-150 s before the first byte (a slower GPU more): the read timeout must outlast the prefill. The
# streaming call is covered too - httpx applies it per chunk, and the first chunk only comes after the prefill. Connecting stays short.
READ_TIMEOUT_S = 1800.0
CONNECT_TIMEOUT_S = 10.0


class OllamaClient:
    """Talks to a local Ollama server. No other network access is used."""

    def __init__(
        self,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = READ_TIMEOUT_S,
    ) -> None:
        base_url = base_url or current_base_url()
        self._base_url = base_url
        # Long read timeout (a 27B generation is slow) but a short *connect*
        # timeout, so a mistyped OLLAMA_HOST fails in seconds, not 10 minutes.
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_S),
        )
        self._tools_unsupported: set[str] = set()
        self._tool_support: dict[str, bool | None] = {}
        self._transport = transport
        self._timeout = timeout
        self._vk_client: httpx.Client | None = None
        self._vulkan_ok: bool | None = None  # None = not tried yet
        self._vulkan_used = False  # this process has put a model on the Vulkan server

    def set_read_timeout(self, seconds: float) -> None:
        """Change the read timeout of every chat client (coding.read_timeout_s); the connect timeout stays short."""
        self._timeout = float(seconds)
        timeout = httpx.Timeout(self._timeout, connect=CONNECT_TIMEOUT_S)
        self._client.timeout = timeout
        if self._vk_client is not None:
            self._vk_client.timeout = timeout

    def server_url(self, model: str) -> str:
        """Base URL of the server that serves `model` (what a health probe has to ask)."""
        return vulkan_base_url() if wants_vulkan(model) and self._vulkan_ok is not False else self._base_url

    def revive(self, model: str) -> bool:
        """After a crash: let the Vulkan server be checked (and started, if there is a start command) again. True if it answers."""
        if not wants_vulkan(model):
            return False
        self._vulkan_ok = None
        return self._ensure_vulkan_server()

    # -- routing between the default server and the dual-GPU Vulkan server ----

    def _vulkan(self) -> httpx.Client:
        if self._vk_client is None:
            self._vk_client = httpx.Client(
                base_url=vulkan_base_url(),
                transport=self._transport,
                timeout=httpx.Timeout(self._timeout, connect=CONNECT_TIMEOUT_S),
            )
        return self._vk_client

    @staticmethod
    def _reachable(client: httpx.Client, wait_s: float = 0.0) -> bool:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                if client.get("/api/version", timeout=2.0).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)

    def _ensure_vulkan_server(self) -> bool:
        """Make sure the Vulkan server answers, starting it on demand (once).

        Returns False - and the caller falls back to the default server - if it
        can't be reached and there is no way to start it."""
        if self._vulkan_ok:
            return True
        vk = self._vulkan()
        if self._reachable(vk):
            self._vulkan_ok = True
            return True
        cmd = _vulkan_start_command()
        if cmd is None:
            log.warning("Vulkan Ollama not reachable at %s and no start command found "
                        "(set AGENT_VULKAN_START_CMD); using the default server", vulkan_base_url())
            return False  # not cached: a later call may succeed once it is started
        log.info("starting the Vulkan Ollama (both GPUs): %s", " ".join(cmd))
        try:
            subprocess.run(cmd, capture_output=True, timeout=_VULKAN_START_TIMEOUT_S, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("could not start the Vulkan Ollama: %s", exc)
            return False
        if self._reachable(vk, wait_s=_VULKAN_START_TIMEOUT_S):
            self._vulkan_ok = True
            return True
        log.warning("Vulkan Ollama did not come up at %s; using the default server", vulkan_base_url())
        return False

    @staticmethod
    def _resident_in_vram(client: httpx.Client) -> list[str]:
        try:
            response = client.get("/api/ps", timeout=3.0)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])
                    if m.get("name") and m.get("size_vram", 0) > 0]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    @staticmethod
    def _unload_on(client: httpx.Client, model: str) -> bool:
        try:
            client.post("/api/generate", json={"model": model, "keep_alive": 0}).raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def _client_for(self, model: str) -> httpx.Client:
        """The HTTP client that should serve `model`.

        A 27B model on the Vulkan server takes the VRAM of BOTH cards, so before
        it runs, anything the default (single-GPU) server still holds in VRAM is
        unloaded - and the other way round when a normal model is used next.
        The embedding model runs with num_gpu 0 (no VRAM) and never goes through
        here, so it is never evicted."""
        if wants_vulkan(model) and self._ensure_vulkan_server():
            for name in self._resident_in_vram(self._client):
                self._unload_on(self._client, name)
            self._vulkan_used = True
            return self._vulkan()
        if self._vulkan_used:
            vk = self._vulkan()
            for name in self._resident_in_vram(vk):
                self._unload_on(vk, name)
        return self._client

    def _servers(self) -> list[httpx.Client]:
        """Every server worth asking about resident models: the default one, plus
        the Vulkan one if it is already running (never started just to ask)."""
        servers = [self._client]
        if self._vulkan_used or (os.environ.get("AGENT_VULKAN_MODELS") or "").strip() or (
                "AGENT_VULKAN_MODELS" not in os.environ and vulkan_available()):
            vk = self._vulkan()
            try:
                if vk.get("/api/version", timeout=1.0).status_code == 200:
                    servers.append(vk)
            except httpx.HTTPError:
                pass
        return servers

    def supports_tools(self, model: str) -> bool | None:
        """Does `model` advertise tool support? None if it can't be determined.

        This matters more than it looks: Ollama only rejects `tools=` outright
        for *some* models. For others (gemma2, for example) it accepts the request
        and silently drops the tools, because the model's chat template has no
        slot for them. The model never learns the tools exist, so when you ask
        it to list a directory it invents a plausible listing instead of calling
        anything. Asking up front is the only way to catch that.
        """
        if model in self._tool_support:
            return self._tool_support[model]
        try:
            response = self._client_for(model).post("/api/show", json={"model": model})
            response.raise_for_status()
            caps = response.json().get("capabilities")
            result = ("tools" in caps) if isinstance(caps, list) else None
        except (httpx.HTTPError, ValueError):
            result = None  # unknown — never cry wolf on a transport hiccup
        self._tool_support[model] = result
        return result

    def loaded_models(self, vram_only: bool = False) -> list[str]:
        """Models currently resident in memory, per Ollama's /api/ps.

        `vram_only=True` excludes entries with `size_vram == 0` - the
        embedding model runs with `num_gpu: 0` specifically so it never
        touches VRAM (see embed()'s docstring), but it still shows up in
        plain /api/ps. A caller that's actually asking "is the GPU free"
        (e.g. image generation waiting for room to fit) needs this, or it
        ends up waiting on - and repeatedly trying to unload - a model
        that was never occupying the VRAM it cares about."""
        names: list[str] = []
        for server in self._servers():
            try:
                response = server.get("/api/ps")
                response.raise_for_status()
                models = response.json().get("models", [])
                if vram_only:
                    models = [m for m in models if m.get("size_vram", 0) > 0]
                names.extend(m["name"] for m in models if m.get("name"))
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue
        return list(dict.fromkeys(names))

    def vram_in_use_mb(self) -> int:
        """VRAM currently held by resident models on all reachable servers (MiB)."""
        total = 0
        for server in self._servers():
            try:
                response = server.get("/api/ps")
                response.raise_for_status()
                total += sum(int(m.get("size_vram", 0)) for m in response.json().get("models", []))
            except (httpx.HTTPError, ValueError, TypeError):
                continue
        return total // (1024 * 1024)

    def model_facts(self, model: str) -> dict[str, Any] | None:
        """Size on disk and the numbers needed to estimate the KV cache of `model`.

        Returns None if Ollama doesn't know the model. Keys: size_mb, layers,
        kv_elems_per_token (sum over layers of kv_heads * (key_len + value_len)),
        ctx_max. Reads /api/show and /api/tags, which work without loading anything."""
        try:
            show = self._client.post("/api/show", json={"model": model})
            show.raise_for_status()
            info = show.json().get("model_info") or {}
            tags = self._client.get("/api/tags")
            tags.raise_for_status()
        except (httpx.HTTPError, ValueError):
            return None
        size = 0
        for m in tags.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size = int(m.get("size", 0))
        arch = info.get("general.architecture", "")

        def num(key: str, default: float = 0) -> float:
            value = info.get(f"{arch}.{key}", default)
            return value if isinstance(value, (int, float)) else default

        layers = int(num("block_count"))
        heads = info.get(f"{arch}.attention.head_count", 0)
        kv_heads = info.get(f"{arch}.attention.head_count_kv", heads)
        emb = num("embedding_length")
        head_dim_default = (emb / heads) if isinstance(heads, (int, float)) and heads else 0
        key_len = num("attention.key_length", head_dim_default)
        val_len = num("attention.value_length", key_len)
        approx = False
        if isinstance(kv_heads, list):  # per-layer kv heads (sliding-window / hybrid models)
            per_layer = [float(x) for x in kv_heads]
        else:
            if not isinstance(kv_heads, (int, float)) or not kv_heads:
                # /api/show hides big arrays (null): assume full multi-head attention, a
                # conservative upper bound for the KV cache (real GQA models need less)
                kv_heads = heads if isinstance(heads, (int, float)) else 0
                approx = True
            per_layer = [float(kv_heads)] * layers
        # hybrid models (attention every Nth layer, the rest state-space) cache KV only there
        interval = int(num("full_attention_interval", 0))
        if interval > 1 and layers:
            per_layer = per_layer[: max(layers // interval, 1)]
        elems = sum(h * (key_len + val_len) for h in per_layer)
        if not size or not layers or not elems:
            return None
        return {
            "size_mb": size // (1024 * 1024), "layers": layers, "approx": approx,
            "kv_elems_per_token": elems, "ctx_max": int(num("context_length", 0)) or 131072,
        }

    def installed_sizes(self) -> dict[str, int]:
        """{model name: size on disk in MiB} for every pulled model."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            return {m["name"]: int(m.get("size", 0)) // (1024 * 1024)
                    for m in response.json().get("models", []) if m.get("name")}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {}

    def list_models(self) -> list[str]:
        """Every model pulled locally (Ollama /api/tags) - not just what's
        currently loaded. Used to offer a picker instead of making the user
        type a model name from memory."""
        names: list[str] = []
        for server in self._servers():
            try:
                response = server.get("/api/tags")
                response.raise_for_status()
                names.extend(m["name"] for m in response.json().get("models", []) if m.get("name"))
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue
        return list(dict.fromkeys(names))

    def unload(self, model: str) -> bool:
        """Evict `model` from memory now. True if Ollama accepted it.

        `keep_alive: 0` with no prompt is Ollama's documented way to unload —
        it's what `ollama stop` does.
        """
        # Route by name, but never start the Vulkan server just to unload.
        if wants_vulkan(model) and (self._vulkan_ok or self._reachable(self._vulkan())):
            return self._unload_on(self._vulkan(), model)
        return self._unload_on(self._client, model)

    def unload_all(self) -> list[str]:
        """Free every resident model. Returns the ones actually unloaded."""
        return [m for m in self.loaded_models() if self.unload(m)]

    def preload(self, model: str) -> bool:
        """Force `model` into memory right now, without waiting for the
        first real chat message to trigger Ollama's own lazy loading. An
        empty-prompt /api/generate call is Ollama's documented way to do
        this - it loads the model and returns immediately without running
        any actual inference. True if Ollama accepted it."""
        try:
            response = self._client_for(model).post(
                "/api/generate", json={"model": model, "prompt": "", "keep_alive": KEEP_ALIVE,
                                      **({"options": {"num_ctx": _last_num_ctx[model]}} if model in _last_num_ctx else {})}
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        stream: bool = True,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
        temperature: float | None = None,
        extra_options: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
        background: bool = False,
        json_mode: bool = False,
    ) -> Iterator[str]:
        """Send a chat request and yield assistant content tokens as they arrive.

        `background=True` marks quick utility calls (distill, overlap judge, compaction):
        they get a hard `num_predict` cap (a runaway "thinking" pass once blocked the
        single slot for minutes). Models with clean `think: false` support (qwen3.5+,
        e.g. the 27B itself - no second model in VRAM, no reload) skip thinking; any
        other model on the dual-GPU server runs CPU-only (`num_gpu: 0`) so it never takes
        VRAM from the big model (`AGENT_BG_CPU=0` keeps it on the GPU).

        `num_ctx` sets the context window Ollama allocates for this call.
        Ollama's default is small (~4k), which silently truncates long inputs.
        `num_gpu` caps how many layers go on the GPU — needed for a model too
        big for VRAM (e.g. Gemma-27B on 8GB), where Ollama's auto-split
        over-commits the GPU and crashes with cudaMalloc OOM. Raises httpx
        exceptions if Ollama is unreachable; callers handle those.

        If `usage` is passed, it's filled in place with `prompt_eval_count`
        and `eval_count` from the final chunk once the response finishes -
        Ollama's own token counts for that call, used to show how full the
        context window actually is. Left untouched (not even added as empty
        keys) if the final chunk never carries them.

        `think` stays unset except for `background=True` on models where `think: false` is
        known to work (see `supports_think_off`): on qwen3:4b it backfired - the raw
        chain-of-thought leaked straight into `content` and it was slower too.
        """
        send_messages = _adapt_messages(model, messages)
        payload: dict[str, Any] = {
            "model": model, "messages": send_messages, "stream": stream, "keep_alive": KEEP_ALIVE,
        }
        if json_mode:                                  # grammar-constrained JSON: a utility call that must return a JSON object cannot narrate instead
            payload["format"] = "json"
        options = dict(speculative_options(model))
        if background:
            num_ctx = num_ctx or _last_num_ctx.get(model)
            options["num_predict"] = BACKGROUND_NUM_PREDICT
            if supports_think_off(model):
                payload["think"] = False
            elif num_gpu is None and wants_vulkan(model) and os.environ.get("AGENT_BG_CPU", "1") != "0":
                num_gpu = 0
        num_ctx = cap_ctx(model, num_ctx)
        _remember_ctx(model, num_ctx)
        if num_ctx:
            options["num_ctx"] = num_ctx
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        if temperature is not None:
            options["temperature"] = temperature
        if extra_options:  # code sampling: repeat_penalty, top_p, top_k, …
            options.update(extra_options)
        if options:
            payload["options"] = options

        with self._client_for(model).stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            if is_deepseek_r1(model):
                # Buffer the full response so the complete <think>…</think> block
                # can be logged and stripped before any content reaches the caller.
                # True per-token streaming would require a state machine and still
                # couldn't log the full think block; the trade-off is acceptable
                # because R1's reasoning pass is internal, not UI output.
                full_content: list[str] = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        full_content.append(content)
                    if chunk.get("done"):
                        if usage is not None:
                            usage["prompt_eval_count"] = chunk.get("prompt_eval_count", 0)
                            usage["eval_count"] = chunk.get("eval_count", 0)
                        break
                assembled = "".join(full_content)
                _log_think_block(model, assembled)
                yield strip_thinking(assembled)
            else:
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        if usage is not None:
                            usage["prompt_eval_count"] = chunk.get("prompt_eval_count", 0)
                            usage["eval_count"] = chunk.get("eval_count", 0)
                        break

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
        think: bool | None = None,
        extra_options: dict[str, Any] | None = None,
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
        extra = {"think": think, "extra_options": extra_options}
        if model in self._tools_unsupported:
            return self._post_chat(messages, model, tools=None, num_ctx=num_ctx, **extra)

        try:
            return self._post_chat(messages, model, tools=tools, num_ctx=num_ctx, **extra)
        except httpx.HTTPStatusError as exc:
            if not _is_tools_unsupported_error(exc.response):
                raise
            self._tools_unsupported.add(model)
            return self._post_chat(messages, model, tools=None, num_ctx=num_ctx, **extra)

    def chat_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
        on_content: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        think: bool | None = None,
        extra_options: dict[str, Any] | None = None,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Like chat_with_tools(), but streamed: `on_content(piece)` is called with each piece
        of the reply text the moment it arrives, so the caller can show it live. `on_thinking`
        gets the hidden reasoning pieces the same way (the caller only peeks at them to know
        what kind of work is coming; they are never shown).

        Returns the same shape as chat_with_tools() (`content`, optional `tool_calls`, plus
        `_prompt_eval_count` / `_eval_count`). Hidden reasoning (`thinking`) is collected but
        never passed to `on_content`. DeepSeek R1 keeps the buffered path: its <think> block
        sits inside the content and has to be stripped as a whole.

        A callback may raise `GenerationAborted` to stop the generation: the stream is closed
        (checked live on Ollama 0.34: the server stops generating at once) and the partial reply
        comes back with `_aborted` set to the reason. `cancel` (an Event) does the same from outside, also
        while the model is still reading the prompt and nothing has arrived for a callback to raise in."""
        if is_deepseek_r1(model):
            return self.chat_with_tools(messages, model, tools, num_ctx=num_ctx, think=think, extra_options=extra_options)
        use_tools = None if model in self._tools_unsupported else tools
        extra = {"think": think, "extra_options": extra_options, "cancel": cancel}
        try:
            return self._stream_chat(messages, model, use_tools, num_ctx, on_content, on_thinking, **extra)
        except httpx.HTTPStatusError as exc:
            if use_tools is None or not _is_tools_unsupported_error(exc.response):
                raise
            self._tools_unsupported.add(model)
            return self._stream_chat(messages, model, None, num_ctx, on_content, on_thinking, **extra)

    def _stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        num_ctx: int | None,
        on_content: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None = None,
        think: bool | None = None,
        extra_options: dict[str, Any] | None = None,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model, "messages": _adapt_messages(model, messages), "stream": True,
            "keep_alive": KEEP_ALIVE,
        }
        if tools is not None:
            payload["tools"] = tools
        if think is not None and (think or supports_think_off(model)):
            payload["think"] = think
        options: dict[str, Any] = dict(speculative_options(model))
        num_ctx = cap_ctx(model, num_ctx)
        cap = output_token_cap(num_ctx)
        if cap:
            options["num_predict"] = cap
        _remember_ctx(model, num_ctx)
        if num_ctx:
            options["num_ctx"] = num_ctx
        if extra_options:
            options.update(extra_options)
        payload["options"] = options
        content: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        aborted = ""
        with self._client_for(model).stream("POST", "/api/chat", json=payload) as response:
            if response.status_code >= 400:
                response.read()
            response.raise_for_status()
            watcher_done = threading.Event()
            if cancel is not None:
                def _watch() -> None:        # closing the connection also ends a long prompt evaluation on the server
                    while not watcher_done.wait(0.2):
                        if cancel.is_set():
                            with contextlib.suppress(Exception):      # shutdown wakes the reader blocked in recv and tells the server
                                sock = response.extensions["network_stream"].get_extra_info("socket")
                                sock.shutdown(socket.SHUT_RDWR)
                            with contextlib.suppress(Exception):
                                response.close()
                            return
                threading.Thread(target=_watch, name="stream-cancel", daemon=True).start()
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):                # a failure after the 200 header arrives as an error line
                        raise httpx.HTTPStatusError(f"model error: {chunk['error']}", request=response.request, response=response)
                    message = chunk.get("message") or {}
                    if on_thinking and message.get("thinking"):
                        on_thinking(message["thinking"])
                    piece = message.get("content") or ""
                    if piece:
                        content.append(piece)
                        if on_content:
                            on_content(piece)
                    tool_calls.extend(message.get("tool_calls") or [])
                    if chunk.get("done"):
                        final = chunk
                        break
            except GenerationAborted as stop:
                aborted = str(stop) or "aborted"      # leaving the `with` closes the connection: the server stops generating
            except (httpx.StreamClosed, httpx.ReadError, httpx.RemoteProtocolError, httpx.CloseError):
                if cancel is None or not cancel.is_set():
                    raise
                aborted = "stopped"
            finally:
                watcher_done.set()
            if cancel is not None and cancel.is_set() and not final and not aborted:
                aborted = "stopped"
        result: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if tool_calls:
            result["tool_calls"] = tool_calls
        result["_prompt_eval_count"] = final.get("prompt_eval_count", 0)
        result["_eval_count"] = final.get("eval_count", 0)
        result["_prompt_eval_ns"] = final.get("prompt_eval_duration", 0)
        result["_eval_ns"] = final.get("eval_duration", 0)
        result["_load_ns"] = final.get("load_duration")             # None when the server sends none (timing.py)
        result["_done_reason"] = final.get("done_reason", "")
        if aborted:
            result["_aborted"] = aborted
        return result

    def _post_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        num_ctx: int | None = None,
        think: bool | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        send_messages = _adapt_messages(model, messages)
        payload: dict[str, Any] = {"model": model, "messages": send_messages, "stream": False, "keep_alive": KEEP_ALIVE}
        if tools is not None:
            payload["tools"] = tools
        if think is not None and (think or supports_think_off(model)):
            payload["think"] = think                  # tool steps run with think=False: no hidden reasoning pass
        # Caught live: a reasoning model (Qwen3) can fall into an unbounded
        # hidden "thinking" pass with no natural stop, and since this call
        # is non-streaming, the request just blocks in total silence until
        # it finishes - reproduced with a real request that hit 500+
        # generated tokens at ~7 tok/s and was still climbing over a
        # minute in, on a machine already known for exactly this symptom
        # ("thinking for 5 minutes on a simple request"). num_predict
        # bounds the worst case to something a tool-call decision or a
        # normal reply should never need this much of anyway.
        options: dict[str, Any] = dict(speculative_options(model))
        num_ctx = cap_ctx(model, num_ctx)
        cap = output_token_cap(num_ctx)
        if cap:
            options["num_predict"] = cap
        _remember_ctx(model, num_ctx)
        if num_ctx:
            options["num_ctx"] = num_ctx
        if extra_options:
            options.update(extra_options)
        payload["options"] = options
        response = self._client_for(model).post("/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
        message = body["message"]
        # Strip DeepSeek R1 <think>…</think> blocks from the content field so
        # tool-call routing and plain-text reply parsing never see raw reasoning.
        if is_deepseek_r1(model) and isinstance(message.get("content"), str) and message["content"]:
            _log_think_block(model, message["content"])
            message["content"] = strip_thinking(message["content"])
        # Non-standard keys tacked onto Ollama's own message dict, not part of
        # its shape - existing callers reading content/tool_calls are
        # unaffected. Lets a caller show context-window usage without this
        # method's return type changing for the two other call sites.
        message["_prompt_eval_count"] = body.get("prompt_eval_count", 0)
        message["_eval_count"] = body.get("eval_count", 0)
        message["_prompt_eval_ns"] = body.get("prompt_eval_duration", 0)
        message["_eval_ns"] = body.get("eval_duration", 0)
        message["_load_ns"] = body.get("load_duration")
        message["_done_reason"] = body.get("done_reason", "")
        return message

    def route(self, task: str, context: str = "", config: dict | None = None) -> str:
        """Return the model name best suited for `task`.

        If the task mentions coding or planning it warrants the main (larger)
        model; everything else goes to the cheaper orchestrator. Both default to
        the same model so the caller always gets *something* sensible even when
        the config doesn't distinguish them yet.
        """
        models = (config or {}).get("models") or {}
        defaults = _DEFAULT_ROUTING_CONFIG["models"]
        key = "main" if _CODING_PLANNING_RE.search(task.lower()) else "orchestrator"
        return models.get(key) or defaults[key]

    def check_model_loaded(self, model_name: str) -> bool:
        """True if `model_name` is currently resident in Ollama memory.

        Matches on the full name or the base name (no tag) so
        ``qwen3.5:9b`` matches a running ``qwen3.5:9b`` and vice-versa.
        Returns False on any connection error so callers can treat a
        missing Ollama server as "not loaded" without a crash.
        """
        base = model_name.split(":")[0]
        for server in self._servers():
            try:
                response = server.get("/api/ps")
                response.raise_for_status()
                for m in response.json().get("models", []):
                    name = m.get("name", "")
                    if name == model_name or name.split(":")[0] == base:
                        return True
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                continue
        return False

    def with_fallback(self, preferred: str, config: dict) -> str:
        """Return `preferred` if it is loaded, else ``config["models"]["fallback"]``.

        Never raises — a failed check is treated as "not loaded", and a config
        with no (or a blank) `models.fallback` falls back to `preferred` rather
        than raising KeyError or handing the caller an empty model name.
        """
        try:
            if self.check_model_loaded(preferred):
                return preferred
        except Exception:  # noqa: BLE001 - a failed check means "not loaded"
            pass
        return ((config or {}).get("models") or {}).get("fallback") or preferred

    def embed(
        self, texts: list[str], model: str, num_gpu: int | None = 0
    ) -> list[list[float]]:
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

        `num_gpu` defaults to 0: embedding runs on the CPU by default so it
        never competes with the main chat model for VRAM, mirroring the
        same option `chat()` already uses (see its docstring) but applied
        as this method's default rather than something every caller has to
        remember to pass. Pass `None` to let Ollama place it automatically
        instead. Confirmed via `/api/ps` (not just assumed): passing
        `num_gpu: 0` drops bge-m3's `size_vram` to 0, vs. the full model
        size when the option is omitted - `/api/embed` does honor it. The
        try/fallback below is kept anyway as a safety net in case a future
        Ollama version or a different embedding model rejects the option.
        """
        if embedding_unavailable(model):
            raise EmbeddingUnavailable(model)
        capped = [t[:EMBED_MAX_CHARS] for t in texts]
        payload: dict[str, Any] = {"model": model, "input": capped}
        if num_gpu is not None:
            payload["options"] = {"num_gpu": num_gpu}
        try:
            try:
                response = self._client.post("/api/embed", json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (404, 501):
                    raise
                if "options" not in payload:
                    raise
                response = self._client.post("/api/embed", json={"model": model, "input": capped})
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (404, 501):
                _embed_down.pop(model, None)
                _note_embedding_down(model)
                raise EmbeddingUnavailable(model) from exc
            raise
        _embed_down.pop(model, None)
        return response.json()["embeddings"]


# ── engine seam helpers: optional client methods with neutral defaults ─────────────────────────────────

def streaming_chat(client: Any) -> Callable[..., Any] | None:
    """The streaming variant of chat_with_tools, or None when the backend has none."""
    fn = getattr(client, "chat_with_tools_stream", None)
    return fn if callable(fn) else None


def server_url_of(client: Any, model: str) -> str | None:
    """Health-probe URL of the server that serves `model`; None = unknown (no health endpoint, or it failed)."""
    fn = getattr(client, "server_url", None)
    if not callable(fn):
        return None
    try:
        return fn(model)
    except Exception:  # noqa: BLE001 - a probe target is a convenience
        return None


def revive_server(client: Any, model: str) -> bool:
    """Try to bring a dead server back (only some backends can); False when unsupported or it did not work."""
    fn = getattr(client, "revive", None)
    if not callable(fn):
        return False
    try:
        return bool(fn(model))
    except Exception:  # noqa: BLE001 - only an attempt
        return False


def loaded_models_of(client: Any) -> list[str]:
    fn = getattr(client, "loaded_models", None)
    if not callable(fn):
        return []
    try:
        return list(fn())
    except Exception:  # noqa: BLE001
        return []


def unload_model(client: Any, model: str) -> bool:
    fn = getattr(client, "unload", None)
    if not callable(fn):
        return False
    try:
        return bool(fn(model))
    except Exception:  # noqa: BLE001
        return False


def unload_all_models(client: Any) -> list[str]:
    fn = getattr(client, "unload_all", None)
    if not callable(fn):
        return []
    try:
        return list(fn())
    except Exception:  # noqa: BLE001
        return []


def set_read_timeout(client: Any, seconds: Any) -> bool:
    """True when the client accepted a read timeout."""
    fn = getattr(client, "set_read_timeout", None)
    if not callable(fn):
        return False
    try:
        fn(seconds)
        return True
    except Exception:  # noqa: BLE001
        return False
