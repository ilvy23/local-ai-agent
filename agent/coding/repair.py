"""Repair loop: retry, detect stalls by AST, escalate. The orchestrator decides.

Each attempt is stateless — it always sees the *current* file state plus the
latest error, never a growing transcript. That sidesteps context poisoning by
construction (no accumulated flailing to clear), so the plan's tier-1/tier-2
split collapses into "try again with fresh eyes." What remains meaningful:

  * stall detection — identical AST or identical error twice means the 7B is in
    a rut; stop wasting attempts.
  * tier-3 escalation — hand a distilled trail to a bigger model. Gated behind
    config (the 30B is unpulled and unproven >5 tok/s), but the hook is here.
"""

from __future__ import annotations

import ast
import hashlib

from agent.coding import config as coding_config
from agent.coding.session import Outcome, attempt
from agent.coding.verify.checks import run_ladder
from agent.coding.workspace import Workspace
from agent.events import NULL, Events


def _signature(ws, changed: list[str]) -> str:
    """Normalised-AST fingerprint of the changed files after an attempt."""
    parts: list[str] = []
    for rel in sorted(set(changed)):
        try:
            src = ws.read(rel)
        except OSError:
            continue
        try:
            parts.append(ast.dump(ast.parse(src)))
        except SyntaxError:
            parts.append("SYNTAX:" + src)  # broken code is always "different"
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()


def _distilled(trail: list[str]) -> str:
    return "Previous attempts (all failed):\n" + "\n".join(f"- {t}" for t in trail[-6:])


def run(repo_root, task: str, files: list[str], config: dict | None = None,
        test_target: str | None = None, llm=None, events: Events = NULL,
        seed: int | None = None) -> Outcome:
    cfg = coding_config.load(config)
    if llm is None:
        from agent.llm import OllamaClient
        llm = OllamaClient()

    max_attempts = cfg["executor"]["max_repair_attempts"]
    esc = cfg.get("escalation", {})

    with Workspace(repo_root) as ws:
        baseline = run_ladder(ws.path, [ws.path / f for f in files], test_target, events)
        if baseline.ok:
            return Outcome(ok=True, stage="already-passing")

        prior_error = baseline.error
        trail: list[str] = []
        last_sig: str | None = None
        last_err: str | None = None
        malformed_total = 0
        real = 0
        gen = 0                        # varies the seed so retries differ
        model_over: str | None = None  # tier-3 escalation swaps this in

        while real < max_attempts:
            events.emit("attempt", n=real + 1, of=max_attempts, model=model_over or cfg["executor"]["model"])
            attempt_seed = None if seed is None else seed * 100 + gen
            gen += 1
            att = attempt(ws, llm, _with_model(cfg, model_over), task, files,
                          test_target, prior_error, events, seed=attempt_seed)

            if att.ok:
                diff = ws.diff()
                if cfg.get("auto_commit", True):
                    ws.commit(f"agent code: {task[:60]}")
                    ws.keep = True
                trail.append(f"attempt {real + 1}: fixed")
                return Outcome(ok=True, stage="fixed", diff=diff,
                               branch=ws.branch if ws.keep else "", attempts=real + 1,
                               malformed_count=malformed_total, trail=trail)

            note = att.retry_note
            if att.result is None and note:  # malformed / test-edit reject → re-ask, don't count
                malformed_total += 1
                prior_error = note
                trail.append(f"re-ask: {note.splitlines()[0][:70]}")
                if malformed_total > max_attempts + 2:
                    break
                continue

            real += 1
            stage = att.result.stage if att.result else "malformed"
            err = att.result.error if att.result else (note or "unknown failure")
            first = err.splitlines()[0][:80] if err else ""
            trail.append(f"attempt {real}: {stage} — {first}")
            events.emit("attempt_failed", n=real, stage=stage, error=first)

            sig = _signature(ws, att.changed)
            if last_sig is not None and (sig == last_sig or err == last_err):
                events.emit("stall", n=real)
                trail.append("stalled: identical AST/error twice")
                if esc.get("tier3_enabled") and model_over != esc.get("tier3_model"):
                    model_over = esc.get("tier3_model")
                    prior_error = _distilled(trail)  # hand the big model a clean summary
                    last_sig = last_err = None
                    events.emit("escalate", to=model_over)
                    continue
                break

            last_sig, last_err = sig, err
            prior_error = err

        diff = ws.diff()
        return Outcome(ok=False, stage="stuck", error=prior_error, diff=diff,
                       attempts=real, malformed_count=malformed_total, trail=trail)


def _with_model(cfg: dict, model_over: str | None) -> dict:
    if not model_over:
        return cfg
    out = {**cfg, "executor": {**cfg["executor"], "model": model_over}}
    return out
