"""Three-layer task routing: NATIVE (no LLM), SMALL (orchestrator model), LARGE (main model).

Routing is purely rule-based — no network calls, no model loading. It
answers "which layer should handle this?" so the caller can then wire up
the right model or skip the LLM entirely for trivial status checks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum


# Keywords whose *sole* presence in a task means no LLM is required.
_STATUS_KEYWORDS: frozenset[str] = frozenset(
    ["status", "alive", "exists", "exit code", "ping"]
)

#: Phrases that flag uncertainty in a context string → escalate to the LARGE layer.
UNCERTAINTY_MARKERS: frozenset[str] = frozenset(
    [
        "i'm not sure",
        "unclear",
        "need clarification",
        "i don't know",
        "not certain",
        "unsure",
    ]
)

_DEFAULT_CONFIG: dict = {
    "models": {
        "main": "qwen3.5:9b",
        "orchestrator": "qwen3.5:9b",
        "fallback": "qwen3.5:9b",
        "reasoning": "deepseek-r1:7b",
    }
}


class Layer(str, Enum):
    NATIVE    = "native"     # pure Python, no LLM
    SMALL     = "small"      # orchestrator model
    LARGE     = "large"      # main model
    REASONING = "reasoning"  # DeepSeek R1 — analysis, strategy, complex problem-solving


#: Keywords that suggest analytical / reasoning work best served by DeepSeek R1.
#: Checked AFTER coding/planning (those stay on LARGE); only tasks that contain
#: one of these words *without* being a coding/planning task reach REASONING.
REASONING_KEYWORDS: frozenset[str] = frozenset([
    "analyze", "analyse", "strategy", "explain", "reason",
    "why", "how does", "compare", "evaluate", "audit",
    "security", "vulnerability", "exploit",
])


@dataclass
class RoutingDecision:
    layer: Layer
    model: str | None  # None for NATIVE
    reason: str


def has_uncertainty(text: str) -> bool:
    """True if `text` contains any of the uncertainty marker phrases."""
    lower = text.lower()
    return any(m in lower for m in UNCERTAINTY_MARKERS)


#: "coding"/"planning" as whole words. Substring matching sent "encoding",
#: "decoding" and "scaffolding" straight to the LARGE layer.
_CODING_PLANNING_RE = re.compile(r"\b(?:coding|planning)\b")


def _is_status_only(task: str) -> bool:
    """True if `task` contains only status-check keywords and no other content.

    At least one keyword must actually be present: without that check a task
    with no alphanumerics at all ("", "   ", "???") stripped down to nothing
    and was declared a "pure status-check", routing to NATIVE with model=None
    — i.e. an empty prompt silently got no model at all.

    Each keyword is then stripped out (longest first to handle multi-word ones
    like "exit code"), and the remainder is checked for any alphanumeric char.
    """
    remaining = task.lower()
    if not any(kw in remaining for kw in _STATUS_KEYWORDS):
        return False
    for kw in sorted(_STATUS_KEYWORDS, key=len, reverse=True):
        remaining = remaining.replace(kw, " ")
    return not re.search(r"[a-z0-9]", remaining)


def decide(
    task: str,
    context: str = "",
    failure_count: int = 0,
    config: dict | None = None,
) -> RoutingDecision:
    """Return a :class:`RoutingDecision` for the given task.

    Routing order (first match wins):

    1. **NATIVE** — task is a pure status check (only status-check keywords,
       nothing else).
    2. **LARGE** — task is coding or planning, ``failure_count >= 2``, or the
       context carries an uncertainty marker.
    3. **REASONING** — task contains analysis/strategy/security keywords (but
       is not a coding/planning task — those stay on LARGE).  Routes to the
       DeepSeek R1 model which excels at step-by-step reasoning.
    4. **SMALL** — everything else.
    """
    # A partial config (no "models" block, or a key missing/blank) must still
    # yield a usable model name rather than a KeyError — routing is supposed to
    # always return *something* sensible.
    models = (config or {}).get("models") or {}
    defaults = _DEFAULT_CONFIG["models"]

    def _model(key: str) -> str:
        return models.get(key) or defaults[key]

    task_lower = task.lower()

    # Layer 1 — no LLM needed
    if _is_status_only(task):
        return RoutingDecision(
            layer=Layer.NATIVE,
            model=None,
            reason="pure status-check task",
        )

    # Layer 3 — escalate to main model
    is_coding_or_planning = bool(_CODING_PLANNING_RE.search(task_lower))
    if is_coding_or_planning:
        reason = "coding/planning task"
    elif failure_count >= 2:
        reason = "failure_count >= 2"
    elif has_uncertainty(context):
        reason = "uncertainty in context"
    else:
        reason = ""

    if reason:
        return RoutingDecision(
            layer=Layer.LARGE,
            model=_model("main"),
            reason=reason,
        )

    # Layer 3.5 — analytical / reasoning tasks → DeepSeek R1
    # Only reached when the task is NOT coding/planning (checked above).
    if any(kw in task_lower for kw in REASONING_KEYWORDS):
        return RoutingDecision(
            layer=Layer.REASONING,
            model=_model("reasoning"),
            reason="reasoning/analysis keyword",
        )

    # Layer 2 — orchestrator handles it
    return RoutingDecision(
        layer=Layer.SMALL,
        model=_model("orchestrator"),
        reason="generic task",
    )


def ensure_models_available(config: dict | None = None) -> dict:
    """Check whether main, orchestrator and reasoning models are available in Ollama.

    Uses OllamaClient.check_model_loaded() to query Ollama's /api/ps endpoint.
    Falls back to False on any connection error so this never raises.

    Returns a dict with keys:
      - "main": bool — whether the main model is loaded
      - "orchestrator": bool — whether the orchestrator model is loaded
      - "reasoning": bool — whether the reasoning (DeepSeek R1) model is loaded
      - "warnings": list[str] — human-readable warnings for missing models

    Logs an error when the orchestrator model is missing (it is required for
    most tasks), and a warning when the main or reasoning model is missing.

    Deliberately offers no auto-pull: side effects in the startup path are
    hard to test and can stall the CLI waiting for a large download.
    """
    # Deferred import avoids a circular import at module load time because
    # llm.py also contains routing helpers that depend on this module's
    # constants.
    from agent.llm import OllamaClient  # noqa: PLC0415

    _DEFAULT_MAIN = "qwen3.8:27b"
    _DEFAULT_ORCH = "qwen3.5:9b"
    _DEFAULT_REASONING = "deepseek-r1:7b"

    models_cfg = (config or {}).get("models") or {}
    main_model = models_cfg.get("main") or _DEFAULT_MAIN
    orch_model = models_cfg.get("orchestrator") or _DEFAULT_ORCH
    reasoning_model = models_cfg.get("reasoning") or _DEFAULT_REASONING

    main_ok = False
    orch_ok = False
    reasoning_ok = False

    try:
        client = OllamaClient()
        try:
            main_ok = bool(client.check_model_loaded(main_model))
        except Exception:  # noqa: BLE001
            main_ok = False
        try:
            orch_ok = bool(client.check_model_loaded(orch_model))
        except Exception:  # noqa: BLE001
            orch_ok = False
        try:
            reasoning_ok = bool(client.check_model_loaded(reasoning_model))
        except Exception:  # noqa: BLE001
            reasoning_ok = False
    except Exception:  # noqa: BLE001
        pass

    warnings: list[str] = []
    if not orch_ok:
        logging.error(
            "Orchestrator model %r is not available in Ollama — "
            "run: ollama pull %s",
            orch_model,
            orch_model,
        )
    if not main_ok:
        logging.warning(
            "Main model %r is not available in Ollama — "
            "large tasks may degrade",
            main_model,
        )
        warnings.append(
            f"main model {main_model!r} not loaded — large tasks may fail"
        )
    if not reasoning_ok:
        logging.warning(
            "Reasoning model %r is not available in Ollama — "
            "analysis/security tasks will degrade (run: ollama pull %s)",
            reasoning_model,
            reasoning_model,
        )
        warnings.append(
            f"reasoning model {reasoning_model!r} not loaded — analysis tasks may fail"
        )

    return {"main": main_ok, "orchestrator": orch_ok, "reasoning": reasoning_ok, "warnings": warnings}
