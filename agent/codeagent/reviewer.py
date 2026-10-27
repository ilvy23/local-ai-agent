"""The reviewer: a fresh-context model call that judges ONE criterion against the program's ACTUAL output.

Principle (reviewer principle): the author must not be the judge. The same model with the same context called a character rewrite
"mostly DONE" although it missed the brief. The reviewer gets no history - only the brief card, one criterion and the
output the HARNESS produced by running scenario commands - and must answer strict JSON:
    {"pass": true|false, "evidence": ["exact quote from the output", ...], "problems": ["...", ...]}
Deterministic guards around the model:
  * the input is bounded (brief <= 3.5k chars, output head+tail <= 11k chars: ~5k tokens), output <= 300 tokens,
    no thinking, low temperature, the conversation's own num_ctx (another value would reload the model),
  * every evidence quote is checked against the output; a "pass" without one real quote is not a pass ("unreviewed"),
  * invalid JSON: one retry with a stricter instruction, then "unreviewed" - never a guess.
Cost note: the server has one slot, so a review evicts the conversation's prompt cache - that is why it runs at the
final gate and at stage boundaries (where the next stage starts a fresh context anyway), not after every step."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

MAX_BRIEF, MAX_CRITERION, OUT_HEAD, OUT_TAIL = 3500, 600, 4000, 7000
REVIEW_PREDICT = 300
REVIEW_OPTIONS = {"temperature": 0.1, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.0, "num_predict": REVIEW_PREDICT}

SYSTEM = (
    "You are a strict, independent reviewer. You did not write this program and you owe its author nothing. You get the user's "
    "brief, ONE criterion and the program's real output (produced by the harness, not by the author). Decide whether the OUTPUT "
    "shows that the criterion is met. Judge only what is in the output; if it does not clearly show the criterion, pass is false. "
    "Answer with JSON only, no other text:\n"
    '{"pass": true or false, "evidence": ["short exact quotes copied from the output"], "problems": ["what is wrong or missing"]}'
)
RETRY = "Your answer was not valid JSON. Reply with exactly one JSON object with the keys pass, evidence, problems - nothing else."


def _clip(text: str, head: int, tail: int) -> str:
    text = str(text or "")
    if len(text) <= head + tail + 40:
        return text
    # harness marker, same shape as tools.cap_output (context.MARKERS "chars_omitted")
    return f"{text[:head]}\n[… {len(text) - head - tail} chars omitted …]" + (f"\n{text[-tail:]}" if tail else "")


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(s).lower()).split())


def parse(reply: str) -> dict[str, Any] | None:
    """The JSON object of a reply (code fences and chatter around it tolerated), or None."""
    from agent.jsonx import extract_json_value

    text = re.sub(r"<think>.*?</think>", "", str(reply or ""), flags=re.S)
    data = extract_json_value(text, "{", "}")
    if not isinstance(data, dict) or not isinstance(data.get("pass"), bool):
        return None
    ev = data.get("evidence") or []
    pr = data.get("problems") or []
    return {"pass": data["pass"], "evidence": [str(x) for x in (ev if isinstance(ev, list) else [ev])][:8],
            "problems": [str(x) for x in (pr if isinstance(pr, list) else [pr])][:8]}


def judge(verdict: dict[str, Any] | None, output: str) -> dict[str, Any]:
    """Apply the deterministic rules to a parsed verdict: quotes must be real, a pass needs one."""
    if verdict is None:
        return {"status": "unreviewed", "pass": None, "evidence": [], "problems": ["the reviewer gave no valid JSON"]}
    hay = _norm(output)
    real = [q for q in verdict["evidence"] if _norm(q) and _norm(q) in hay]
    fake = [q for q in verdict["evidence"] if q not in real]
    problems = list(verdict["problems"])
    if fake:
        problems.append(f"{len(fake)} evidence quote(s) are not in the output")
    if verdict["pass"] and not real:
        return {"status": "unreviewed", "pass": None, "evidence": [], "problems": problems + ["a pass without a real quote from the output"]}
    return {"status": "pass" if verdict["pass"] else "fail", "pass": verdict["pass"], "evidence": real, "problems": problems}


def review(llm: Any, model: str, brief: str, criterion: str, output: str, *, num_ctx: int | None = None,
           events: Any = None) -> dict[str, Any]:
    """{status: pass|fail|unreviewed, pass, evidence, problems, seconds, prompt}. Never raises."""
    user = (f"USER'S BRIEF:\n{_clip(brief, MAX_BRIEF, 0) or '(none)'}\n\nCRITERION:\n{_clip(criterion, MAX_CRITERION, 0)}\n\n"
            f"PROGRAM OUTPUT (produced by the harness):\n{_clip(output, OUT_HEAD, OUT_TAIL) or '(no output)'}")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    t0 = time.monotonic()
    verdict, prompt = None, 0
    for attempt in range(2):
        try:
            msg = llm.chat_with_tools(messages, model, [], num_ctx=num_ctx, think=False, extra_options=dict(REVIEW_OPTIONS))
        except Exception as exc:  # noqa: BLE001 - an unreachable reviewer leaves the criterion unreviewed, it does not end the run
            logger.warning("review failed: %s", exc)
            break
        prompt += int(msg.get("_prompt_eval_count", 0) or 0)
        reply = str(msg.get("content", ""))
        verdict = parse(reply)
        if verdict is not None:
            break
        if attempt == 0:
            messages = [*messages, {"role": "assistant", "content": reply[:800]}, {"role": "user", "content": RETRY}]
    out = judge(verdict, output)
    out.update(seconds=round(time.monotonic() - t0, 1), prompt=prompt)
    if events is not None:
        events.emit("review", status=out["status"], criterion=criterion[:100], seconds=out["seconds"], prompt=prompt)
    return out


def summary(result: dict[str, Any]) -> str:
    """One short block for the model / the report."""
    lines = [f"reviewer: {result.get('status', 'unreviewed').upper()}"]
    lines += [f'  evidence: "{q[:140]}"' for q in result.get("evidence", [])[:3]]
    lines += [f"  problem: {p[:200]}" for p in result.get("problems", [])[:4]]
    return "\n".join(lines)


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
