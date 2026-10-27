"""`write_prose`: a second, small text model writes the graphic or mature passages of a creative project.

The coding model (stock Qwen) stays the agent and places the text into the files; it softens graphic violence when it writes it itself.
Off by default: the tool exists only when `coding.prose_model` is set.
The model runs through `llm.chat(background=True, ...)`, CPU-only by default (`coding.prose_gpu` false -> `num_gpu 0`), so it never
evicts the main model from VRAM. Nothing here raises: every failure is a short string for the agent and an event."""

from __future__ import annotations

from agent import brand

import re
import threading
from pathlib import Path
from typing import Any

from agent.codeagent.context import MARKERS

STYLE_FILES = ("STYLE.md", ".agent/style.md", "docs/STYLE.md")      # first existing one wins
STYLE_CAP = 3000
DEFAULT_WORDS, MAX_NUM_PREDICT, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT_S, DEFAULT_MAX_CALLS = 120, 600, 0.8, 300.0, 12
START, END = "----- PASSAGE START -----", "----- PASSAGE END -----"

SYSTEM = ("You are a fiction writer for a video game. Write exactly what is requested in the genre's conventions, concrete and "
          "unflinching, no disclaimers, no moralizing, no meta commentary, no refusals; output ONLY the passage text.")
PROMPT_SENTENCE = ("For graphic or mature narrative passages (deaths, violence), call write_prose instead of writing them yourself, "
                   "then insert the returned text verbatim.")
TOOL_DESCRIPTION = ("Have a separate fiction model write ONE graphic or mature narrative passage (a death, violence, an ending text). "
                    "Returns the passage between PASSAGE START / PASSAGE END lines: copy exactly what is between them into the file with your "
                    "edit tools, without the delimiter lines. It does not write any file itself.")
REFUSED = "The prose model refused or failed; try rephrasing the request or write a milder version yourself."
LIMIT = "write_prose limit reached for this prompt ({n} calls); write the remaining passages yourself."

_REFUSAL = re.compile(r"(?i)\b(i can(?:'|’)?t|i cannot|i can not|i(?:'|’)m sorry|i am sorry|i(?:'|’)m unable|i am unable|i won(?:'|’)t|"
                      r"as an ai|as a language model|i(?:'|’)m not able|i am not able|cannot (?:assist|help|fulfil|fulfill|comply))\b")
_BOILER = re.compile(r"(?i)^\s*(?:(?:here(?:'|’)s|here is|here are|below is)\b.*|(?:sure|certainly|of course|okay)\b[^\n]{0,120}(?:\bhere\b|:\s*)$)")
_THINK = re.compile(r"<think>.*?</think>|<think>.*\Z", re.S | re.I)
_FENCE = re.compile(r"^```[\w-]*\n?|\n?```\s*$")
MIN_WORDS = 5


def enabled(config: dict[str, Any] | None) -> bool:
    return bool(str(((config or {}).get("coding") or {}).get("prose_model") or "").strip())


def _num(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float((config.get("coding") or {}).get(key, default))
    except (TypeError, ValueError):
        return default


def style_samples(root: Path | str | None) -> str:
    """Text of the first existing style file in the repo root (STYLE.md, .agent/style.md, docs/STYLE.md), capped; '' when none."""
    if root is None:
        return ""
    for rel in STYLE_FILES:
        p = Path(root) / rel
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace").strip()[:STYLE_CAP]
        except OSError:
            continue
    return ""


def build_messages(request: str, style_hint: str, max_words: int, style: str) -> list[dict[str, str]]:
    parts = []
    if style:
        parts.append("Style samples of this game (match their voice, tense and level of detail):\n" + style)
    parts.append("Request: " + request.strip())
    if style_hint.strip():
        parts.append("Style hint: " + style_hint.strip())
    parts.append(f"Length: about {max_words} words, at most {max_words + max_words // 4}. Output only the passage.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def strip_markers(text: str) -> str:
    """The passage must not carry a harness marker (context.MARKERS): the model would write it into a game file."""
    for _name, rx in MARKERS:
        text = rx.sub("", text)
    return text


def clean(text: str) -> str:
    """Drop <think> blocks, 'Here is ...' lines, code fences and surrounding quotes, and any harness marker."""
    t = _THINK.sub("", text or "").strip()
    t = _FENCE.sub("", t).strip()
    lines = t.splitlines()
    while lines and (not lines[0].strip() or _BOILER.match(lines[0])):
        lines.pop(0)
    t = "\n".join(lines).strip()
    pairs = {'"': '"', "'": "'", "“": "”", "”": "”"}
    if len(t) >= 2 and pairs.get(t[0]) == t[-1] and t.count(t[0]) + (t.count(t[-1]) if t[-1] != t[0] else 0) <= 2:
        t = t[1:-1].strip()
    return strip_markers(t).replace(START, "").replace(END, "").strip()


def refused(passage: str) -> bool:
    """A refusal sounds like one in the first lines, or the answer is too short to be a passage."""
    if len(passage.split()) < MIN_WORDS:
        return True
    return bool(_REFUSAL.search(" ".join(passage.split()[:40])))


def _call(llm: Any, messages: list[dict[str, str]], model: str, kw: dict[str, Any], timeout: float) -> str:
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["text"] = "".join(str(p) for p in llm.chat(messages, model=model, **kw))
        except BaseException as exc:  # noqa: BLE001 - reported to the agent, never raised
            box["err"] = exc

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"no answer within {timeout:.0f}s")
    if "err" in box:
        raise box["err"]
    return box.get("text", "")


def write_prose(llm: Any, config: dict[str, Any], root: Path | str | None, request: str = "", style_hint: str = "",
                max_words: int = DEFAULT_WORDS, *, model: str | None = None, events: Any = None, gpu: bool | None = None) -> str:
    """One passage from the prose model, delimited, with a footer line. Never raises."""
    coding = config.get("coding") or {}
    model = model or str(coding.get("prose_model") or "").strip()

    def emit(kind: str, **data: Any) -> None:
        if events is not None:
            events.emit(kind, **data)

    if not model:
        return "error: no prose model configured (coding.prose_model)."
    if not str(request or "").strip():
        return "error: write_prose needs a request."
    try:
        words = max(10, min(int(max_words), 400))
    except (TypeError, ValueError):
        words = DEFAULT_WORDS
    use_gpu = bool(coding.get("prose_gpu", False)) if gpu is None else gpu
    kw: dict[str, Any] = {"stream": False, "background": True, "temperature": _num(config, "prose_temperature", DEFAULT_TEMPERATURE),
                          "extra_options": {"num_predict": min(MAX_NUM_PREDICT, words * 3)}}
    kw["num_gpu"] = -1 if use_gpu else 0                                 # -1 = the server decides; 0 = CPU only, VRAM stays with the main model
    messages = build_messages(str(request), str(style_hint or ""), words, style_samples(root))
    try:
        raw = _call(llm, messages, model, kw, _num(config, "prose_timeout_s", DEFAULT_TIMEOUT_S))
    except Exception as exc:  # noqa: BLE001
        why = "timed out (" + str(exc) + ")" if isinstance(exc, TimeoutError) else f"{type(exc).__name__}: {str(exc)[:120]}"
        emit("prose_error", model=model, error=why)
        return f"error: prose model {model} failed: {why}"
    passage = clean(raw)
    if refused(passage):
        emit("prose_refused", model=model, words=len(passage.split()))
        return REFUSED
    n = len(passage.split())
    emit("prose", model=model, words=n)
    return f"{START}\n{passage}\n{END}\nprose from {model}, {n} words - place it in the file yourself with edit/replace tools"


def prose_cli(request: str, model: str | None = None, gpu: bool = False, folder: str | None = None) -> int:
    """`agent code prose "request..." [--model M] [--gpu]`: the same function the agent calls; prints the passage. No agent, no coding model."""
    from rich.console import Console

    from agent.config import load_config
    from agent.main import _open_llm

    console = Console()
    config = load_config()
    if not (model or enabled(config)):
        console.print("[yellow]no prose model: pass --model M or set coding.prose_model[/yellow]")
        return 1
    if not request.strip():
        console.print(f"[yellow]usage: {brand.CMD} code prose \"request...\" [--model M] [--gpu][/yellow]")
        return 1
    root = Path(folder).expanduser() if folder else Path.cwd()
    out = write_prose(_open_llm(), config, root, request, model=model, gpu=gpu or None)
    console.print(out, markup=False, highlight=False, soft_wrap=True)
    return 0 if out.startswith(START) else 1
