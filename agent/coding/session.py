"""One coding attempt: prompt → generate → parse → apply → verify → report.

This is a single rung. repair.py wraps it with retries, AST-stall detection and
escalation. Keeping them separate keeps each testable — this module works with a
fake LLM in tests and the real coder-7B in anger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.coding import config as coding_config
from agent.coding.edit.apply import apply_edits
from agent.coding.edit.format import parse_edits
from agent.coding.verify import guards
from agent.coding.verify.checks import CheckResult, run_ladder
from agent.coding.workspace import Workspace
from agent.events import NULL, Events

_SYSTEM = """You are a precise coding assistant working inside a Python repository.
Fix the task by editing files. Output ONLY file edits — no explanation.

For a SMALL file, output the complete new file:

### FILE: <path>
```python
<the COMPLETE new contents of the file>
```

For a LARGE file, output ONLY the changed regions as search/replace blocks:

### FILE: <path>
```
<<<<<<< SEARCH
<exact existing lines to find>
=======
<the replacement lines>
>>>>>>> REPLACE
```

Rules:
- Whole-file: output the ENTIRE file, never a fragment.
- Search/replace: the SEARCH text must match the current file EXACTLY, byte for byte.
- Never edit a test file (name starts with `test_` or ends with `_test.py`).
- Make the smallest change that makes the checks pass."""


_DIAGNOSE = """A check is failing. Do NOT write code yet — first find the ROOT CAUSE by
SIMULATING the failing test by hand.

Task: {task}

Failing check output:
```
{error}
```

Current code:
{code}

Do this concretely — do not hand-wave:
1. Start from the failing assertion. Execute the relevant functions BY HAND, step by
   step, and WRITE OUT the actual value of the key variables/data structures after
   EACH operation. e.g. "total = 0  → after add(5): total = 5  → after add(3): total = 8".
2. Compare the value you computed to what the assertion expects. The exact line where
   they diverge IS the bug — name that line and say WHY it produces the wrong value.
{test_clause}4. State the ONE root cause and the concrete fix (which line to change and to what).

Answer in 5-8 short lines. Show the traced values. No code blocks."""

_TEST_CLAUSE = (
    "3. Decide where the bug really is: the IMPLEMENTATION, or the TEST's own setup /\n"
    "   expectation. A test CAN be wrong — e.g. it sets up a state that can't produce\n"
    "   the expected result (wrong assumptions about the order of operations, the\n"
    "   state, or the expected values). Say which.\n"
)


def _diagnose(llm, cfg: dict, task: str, files: list[str], ws: Workspace,
              error: str, events: Events, seed: int | None = None,
              tests_editable: bool = False) -> str:
    """One focused reasoning pass that yields a plain-English root cause. A small
    model diagnoses far better when it (a) does this BEFORE writing a fix and
    (b) is forced to trace concrete values rather than hand-wave. `seed` varies
    the hypothesis across repair attempts so a wrong first guess isn't repeated."""
    code = "\n".join(f"### {r}\n```python\n{ws.read(r)}\n```" for r in files)
    # Only let the diagnosis blame the test when the caller can actually edit it
    # (build mode). In fix mode tests are read-only, so blaming the test is a dead end.
    prompt = _DIAGNOSE.format(task=task, error=error[:1500], code=code[:6000],
                              test_clause=_TEST_CLAUSE if tests_editable else "")
    ex = cfg["executor"]
    events.emit("diagnose_start")
    # Slightly higher temperature than the fix pass: we want varied hypotheses
    # when the first diagnosis was wrong, not the same deterministic mistake.
    opts = _code_options(ex, seed)
    try:
        out = "".join(llm.chat([{"role": "user", "content": prompt}], model=ex["model"],
                               num_ctx=ex["num_ctx"], num_gpu=ex.get("num_gpu"),
                               temperature=max(ex["temperature"], 0.4),
                               extra_options=opts))
    except Exception as exc:  # noqa: BLE001 - diagnosis is optional; never crash the fix
        events.emit("diagnose_end", text=f"(skipped: {type(exc).__name__})")
        return ""
    events.emit("diagnose_end", text=out.strip()[:400])
    return out.strip()


@dataclass
class Attempt:
    ok: bool
    result: CheckResult | None = None
    changed: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # test-file edits, etc.
    raw: str = ""

    @property
    def retry_note(self) -> str | None:
        """A message to feed back that shouldn't count as a real attempt."""
        if self.rejected or self.malformed:
            return "\n".join(self.malformed)
        return None


@dataclass
class Outcome:
    ok: bool
    stage: str = ""          # already-passing | fixed | stuck | syntax | lint | tests | judge
    error: str = ""
    diff: str = ""
    branch: str = ""
    attempts: int = 0
    malformed_count: int = 0
    trail: list[str] = field(default_factory=list)


def _code_options(ex: dict, seed: int | None) -> dict:
    opts = {"repeat_penalty": ex["repeat_penalty"], "top_p": ex["top_p"], "top_k": ex["top_k"]}
    if seed is not None:
        opts["seed"] = seed
    return opts


def _build_user(task: str, files: list[str], ws: Workspace, error: str | None,
                threshold: int, diagnosis: str = "") -> str:
    parts = [f"Task: {task}\n"]
    if error:
        parts.append("The checks currently fail:\n```\n" + error[:2000] + "\n```\n")
    if diagnosis:
        parts.append("Root-cause diagnosis (implement this fix):\n" + diagnosis + "\n")
    parts.append("Current file(s):\n")
    for rel in files:
        content = ws.read(rel)
        n = content.count("\n") + 1
        mode = "search/replace (large file)" if n > threshold else "whole-file"
        parts.append(f"### FILE: {rel}  [{n} lines — use {mode}]\n```python\n{content}\n```\n")
    parts.append("\nReturn the corrected file(s) in the FILE format above.")
    return "\n".join(parts)


def _generate(llm, cfg: dict, messages: list[dict], events: Events, seed: int | None) -> str:
    ex = cfg["executor"]
    events.emit("generate_start", model=ex["model"])
    out: list[str] = []
    for tok in llm.chat(messages, model=ex["model"], num_ctx=ex["num_ctx"],
                        num_gpu=ex.get("num_gpu"),
                        temperature=ex["temperature"], extra_options=_code_options(ex, seed)):
        out.append(tok)
        events.emit("token", text=tok)
    events.emit("generate_end", tokens=len("".join(out)))
    return "".join(out)


def attempt(ws: Workspace, llm, cfg: dict, task: str, files: list[str],
            test_target: str | None, prior_error: str | None, events: Events = NULL,
            seed: int | None = None) -> Attempt:
    """A single generate→apply→verify cycle inside an existing workspace."""
    threshold = cfg["executor"].get("whole_file_threshold", 200)
    # Diagnose first (only when debugging a real failure and enabled): a separate
    # reasoning pass so the model finds the root cause before it has to also write
    # the fix. This is what lets the 7B self-solve subtle bugs instead of stalling.
    diagnosis = ""
    if prior_error and cfg.get("diagnose_first", True):
        diagnosis = _diagnose(llm, cfg, task, files, ws, prior_error, events, seed=seed)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _build_user(task, files, ws, prior_error, threshold, diagnosis)},
    ]
    raw = _generate(llm, cfg, messages, events, seed)
    parsed = parse_edits(raw, default_path=files[0] if len(files) == 1 else None)
    if not parsed.edits:
        return Attempt(ok=False, malformed=parsed.malformed or ["no edits parsed"], raw=raw)

    g = cfg.get("guards", {})
    # Guard 1: test files are read-only.
    if g.get("test_files_readonly", True):
        violations = guards.test_edit_violations([e.path for e in parsed.edits])
        if violations:
            events.emit("guard_reject", name="test-edit", paths=violations)
            return Attempt(ok=False, rejected=violations, raw=raw, malformed=[
                f"Rejected: you edited test file(s) {violations}. Tests are read-only — "
                "fix the implementation so the existing tests pass."])

    changed, errors = apply_edits(ws, parsed.edits)
    if errors:
        return Attempt(ok=False, malformed=errors, changed=changed, raw=raw)

    result = run_ladder(ws.path, [ws.path / c for c in changed], test_target, events)
    if not result.ok:
        return Attempt(ok=False, result=result, changed=changed, malformed=parsed.malformed, raw=raw)

    # Post-pass guards: only reached when the ladder is green.
    if g.get("coverage_gate", True):
        cov = guards.coverage_gate(ws, changed, test_target)
        events.emit("guard", name="coverage", ok=cov.ok)
        if not cov.ok:
            return Attempt(ok=False, result=cov, changed=changed, raw=raw)
    if g.get("llm_judge", True):
        verdict = guards.judge_diff(llm, cfg, task, ws.diff())
        events.emit("guard", name="judge", ok=verdict.ok)
        if not verdict.ok:
            return Attempt(ok=False, changed=changed, raw=raw,
                           result=CheckResult(False, "judge",
                                              "Reviewer flagged the diff as gaming the test: "
                                              + verdict.reason))

    return Attempt(ok=True, result=result, changed=changed, malformed=parsed.malformed, raw=raw)


def run_once(repo_root, task: str, files: list[str], config: dict | None = None,
             test_target: str | None = None, llm=None, events: Events = NULL) -> Outcome:
    """Single-shot fix in an isolated worktree. On success, auto-commit + keep the branch."""
    cfg = coding_config.load(config)
    if llm is None:
        from agent.llm import OllamaClient
        llm = OllamaClient()

    with Workspace(repo_root) as ws:
        baseline = run_ladder(ws.path, [ws.path / f for f in files], test_target, events)
        if baseline.ok:
            return Outcome(ok=True, stage="already-passing")

        att = attempt(ws, llm, cfg, task, files, test_target, baseline.error, events)
        diff = ws.diff()
        if att.ok:
            if cfg.get("auto_commit", True):
                ws.commit(f"agent code: {task[:60]}")
                ws.keep = True
            return Outcome(ok=True, stage="fixed", diff=diff,
                           branch=ws.branch if ws.keep else "", attempts=1,
                           malformed_count=len(att.malformed))
        stage = att.result.stage if att.result else "malformed"
        error = att.result.error if att.result else "; ".join(att.malformed)
        return Outcome(ok=False, stage=stage, error=error, diff=diff, attempts=1,
                       malformed_count=len(att.malformed))
