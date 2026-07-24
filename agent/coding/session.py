from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.coding.edit.apply import apply_edits
from agent.coding.edit.format import parse_edits
from agent.coding.repair import (
    Attempt,
    Decision,
    RepairState,
    Tier,
    decide_next,
)
from agent.coding.verify.checks import ladder_passed, run_ladder
from agent.coding.verify.guards import (
    build_judge_prompt,
    coverage_gate,
    parse_judge_verdict,
    reject_test_edits,
)
from agent.coding.verify.report import enrich_traceback, summarize_failure
from agent.coding.workspace import Workspace
from agent.events import EventBus

MAX_MALFORMED_REASKS = 3

_EDIT_INSTRUCTIONS = (
    "You are a coding executor. You edit files to satisfy a task and make the "
    "tests pass. You must NOT edit any test file.\n\n"
    "Reply with edit blocks and nothing else. Two forms:\n\n"
    "Whole-file rewrite (use for files under ~200 lines):\n"
    "<<<< FILE: path/to/file.py\n"
    "<the complete new contents of the file>\n"
    ">>>>\n\n"
    "Search/replace (use for large files):\n"
    "<<<< FILE: path/to/file.py\n"
    "------ SEARCH\n"
    "<exact existing text>\n"
    "====== REPLACE\n"
    "<new text>\n"
    ">>>>\n\n"
    "Emit one block per file you change. Do not explain."
)


@dataclass(frozen=True)
class SessionResult:
    success: bool
    tier: Tier
    reason: str
    diff: str = ""
    changed_files: tuple[str, ...] = ()
    attempts: int = 0
    malformed: int = 0


@dataclass
class CodingSession:
    client: Any
    config: dict[str, Any]
    workspace: Workspace
    bus: EventBus = field(default_factory=EventBus)

    def run(self, task: str, files: dict[str, str] | None = None) -> SessionResult:
        cfg = self.config["coding"]
        executor = cfg["executor"]
        guards = cfg["guards"]
        escalation = cfg["escalation"]
        sandbox = cfg["sandbox"]

        model = executor["model"]
        max_attempts = executor["max_repair_attempts"]
        state = RepairState(max_attempts=max_attempts)

        conversation = self._initial_conversation(task, files)
        error = ""
        malformed = 0
        tier = Tier.ACCUMULATING

        while True:
            self.bus.emit_kind("attempt", f"tier {int(tier)}", tier=int(tier), model=model)
            response = self._complete(conversation, model, executor)
            conversation = [*conversation, {"role": "assistant", "content": response}]

            parsed = parse_edits(response)
            if not parsed.ok:
                malformed += 1
                self.bus.emit_kind("malformed", "; ".join(parsed.malformed), count=malformed)
                if malformed > MAX_MALFORMED_REASKS:
                    return SessionResult(
                        False, tier, "too many malformed responses",
                        attempts=state.count, malformed=malformed,
                    )
                conversation = [
                    *conversation,
                    {"role": "user", "content": _reask(parsed.malformed)},
                ]
                continue

            outcome = self._attempt(parsed.edits, task, guards, sandbox)
            if outcome.success:
                self.bus.emit_kind("success", f"tier {int(tier)}", tier=int(tier))
                return SessionResult(
                    True, tier, "tests passed",
                    diff=outcome.diff, changed_files=outcome.changed,
                    attempts=state.count + 1, malformed=malformed,
                )

            error = outcome.error
            self.bus.emit_kind("check", error, ok=False)

            attempt = Attempt.make(outcome.source, error=error)
            decision = decide_next(state, attempt)

            if decision.action in (Decision.RETRY, Decision.SATISFY_BLOCKER):
                conversation = [*conversation, {"role": "user", "content": _fix_prompt(error)}]
                continue

            if decision.action is Decision.HANDBACK or decision.tier is Tier.HANDBACK:
                return self._handback(tier, state, malformed)

            tier = decision.tier
            state.tier = tier
            self.bus.emit_kind("escalate", decision.reason, tier=int(tier))

            if tier is Tier.FRESH_CONTEXT:
                conversation = self._fresh_conversation(task, files, error)
            elif tier is Tier.ESCALATED:
                if not escalation["tier3_enabled"]:
                    return self._handback(tier, state, malformed)
                model = escalation["tier3_model"]
                conversation = self._distilled_conversation(task, files, state)

    def _attempt(self, edits, task, guards, sandbox) -> _Outcome:
        self.workspace.reset()

        result = apply_edits(edits, self.workspace.path)
        if not result.ok:
            return _Outcome(error="; ".join(result.errors), source=_source_of(edits))

        if guards["test_files_readonly"]:
            guard = reject_test_edits(result.changed)
            if not guard.ok:
                return _Outcome(error=guard.reason, source=_source_of(edits))

        changed_abs = [self.workspace.path / p for p in result.changed]
        source = self._read_changed(result.changed)

        ladder = run_ladder(
            changed_abs, self.workspace.path,
            timeout_seconds=sandbox["timeout_seconds"],
        )
        for check in ladder:
            self.bus.emit_kind("check", check.name, name=check.name, status=check.status.value)
        if not ladder_passed(ladder):
            return _Outcome(error=self._ladder_error(ladder), source=source)

        if guards["coverage_gate"]:
            cov = coverage_gate(changed_abs, self.workspace.path)
            if not cov.ok:
                return _Outcome(error=cov.reason, source=source)

        diff = self.workspace.diff()
        if guards["llm_judge"]:
            verdict = self._judge(task, diff)
            if not verdict.ok:
                return _Outcome(error=verdict.reason, source=source)

        return _Outcome(success=True, diff=diff, changed=result.changed, source=source)

    def _ladder_error(self, ladder) -> str:
        failed = next((c for c in ladder if c.blocking), None)
        if failed is None:
            return "check failed"
        detail = enrich_traceback(failed.detail, self.workspace.path)
        return summarize_failure(failed.name, detail)

    def _judge(self, task: str, diff: str):
        model = self.config["coding"]["executor"]["model"]
        prompt = build_judge_prompt(task, diff)
        response = self._complete([{"role": "user", "content": prompt}], model, temperature=0.0)
        return parse_judge_verdict(response)

    def _read_changed(self, changed: tuple[str, ...]) -> str:
        parts = []
        for rel in changed:
            try:
                parts.append(self.workspace.read(rel))
            except OSError:
                continue
        return "\n".join(parts)

    def _initial_conversation(self, task: str, files: dict[str, str] | None) -> list[dict]:
        return [
            {"role": "system", "content": _EDIT_INSTRUCTIONS},
            {"role": "user", "content": _task_prompt(task, files)},
        ]

    def _fresh_conversation(self, task, files, error) -> list[dict]:
        body = (
            f"{_task_prompt(task, files)}\n\n"
            f"A previous attempt failed with this error. Start over and fix it:\n{error}"
        )
        return [
            {"role": "system", "content": _EDIT_INSTRUCTIONS},
            {"role": "user", "content": body},
        ]

    def _distilled_conversation(self, task, files, state: RepairState) -> list[dict]:
        history = "\n".join(
            f"- attempted a fix, failed: {_first_line(a.error)}" for a in state.attempts
        )
        body = (
            f"{_task_prompt(task, files)}\n\n"
            f"Smaller models already tried and failed:\n{history}\n\n"
            f"Produce a correct fix."
        )
        return [
            {"role": "system", "content": _EDIT_INSTRUCTIONS},
            {"role": "user", "content": body},
        ]

    def _handback(self, tier, state: RepairState, malformed: int) -> SessionResult:
        self.bus.emit_kind("handback", "handing back to user", attempts=state.count)
        return SessionResult(
            False, tier, "could not reach a passing state",
            attempts=state.count, malformed=malformed,
        )

    def _complete(self, messages, model, executor=None, temperature=None) -> str:
        options = None
        num_ctx = None
        if executor is not None:
            options = {
                "repeat_penalty": executor["repeat_penalty"],
                "top_p": executor["top_p"],
                "top_k": executor["top_k"],
            }
            num_ctx = executor["num_ctx"]
            temperature = executor["temperature"]
        return "".join(
            self.client.chat(
                messages=messages,
                model=model,
                stream=False,
                num_ctx=num_ctx,
                temperature=temperature,
                options=options,
            )
        )


@dataclass
class _Outcome:
    success: bool = False
    error: str = ""
    diff: str = ""
    changed: tuple[str, ...] = ()
    source: str = ""


def _source_of(edits) -> str:
    return "\n".join(e.replace for e in edits)


def _task_prompt(task: str, files: dict[str, str] | None) -> str:
    parts = [f"## Task\n{task.strip()}"]
    if files:
        for path, content in files.items():
            parts.append(f"## File: {path}\n```\n{content}\n```")
    return "\n\n".join(parts)


def _reask(reasons: tuple[str, ...]) -> str:
    joined = "; ".join(reasons)
    return (
        f"Your response could not be parsed as edits ({joined}). "
        f"Re-send using the exact edit-block format. Do not explain."
    )


def _fix_prompt(error: str) -> str:
    return f"That did not work. Fix it. Error:\n{error}"


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""
