from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from agent.coding.verify.runner import run_python

_TEST_NAME_RE = re.compile(r"(^test_.*\.py$)|(.*_test\.py$)|(^conftest\.py$)")


def is_test_file(rel_path: str) -> bool:
    p = Path(rel_path)
    if _TEST_NAME_RE.match(p.name):
        return True
    return any(part in {"tests", "test"} for part in p.parts[:-1])


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""


def reject_test_edits(changed_paths: list[str] | tuple[str, ...]) -> GuardResult:
    offenders = [p for p in changed_paths if is_test_file(p)]
    if not offenders:
        return GuardResult(ok=True)
    listed = ", ".join(offenders)
    return GuardResult(
        ok=False,
        reason=(
            f"rejected: this task must not modify test files, but the edit "
            f"changed {listed}. Fix the implementation so the existing tests "
            f"pass unchanged; do not edit the tests."
        ),
    )


def coverage_gate(
    changed_files: list[Path],
    cwd: Path,
    *,
    timeout_seconds: float = 120.0,
) -> GuardResult:
    py_files = [f for f in changed_files if f.suffix == ".py"]
    if not py_files:
        return GuardResult(ok=True)

    rel = [str(f.relative_to(cwd)) if f.is_absolute() else str(f) for f in py_files]
    report_path = cwd / ".coverage_gate.json"

    run_result = run_python(
        ["coverage", "run", "--source", ",".join(rel), "-m", "pytest",
         "-q", "-p", "no:cacheprovider"],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    if run_result.returncode == 127 or "No module named coverage" in (run_result.stderr or ""):
        return GuardResult(ok=True, reason="coverage.py not installed — gate skipped")

    export = run_python(
        ["coverage", "json", "-o", str(report_path), "--quiet"],
        cwd=cwd,
        timeout_seconds=30.0,
    )
    if not report_path.exists():
        return GuardResult(ok=True, reason=f"coverage report unavailable — gate skipped ({_short(export)})")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return GuardResult(ok=True, reason=f"coverage report unreadable — gate skipped ({exc})")
    finally:
        report_path.unlink(missing_ok=True)

    untouched = _files_with_no_coverage(report, rel)
    if untouched:
        listed = ", ".join(untouched)
        return GuardResult(
            ok=False,
            reason=(
                f"tests pass but do not execute your changes in {listed}. "
                f"The passing tests never run the changed code, so 'passed' "
                f"proves nothing. Add or adjust the implementation so the tests "
                f"actually exercise it."
            ),
        )
    return GuardResult(ok=True)


def _files_with_no_coverage(report: dict, rel_paths: list[str]) -> list[str]:
    files = report.get("files", {})
    by_base: dict[str, dict] = {}
    for key, data in files.items():
        by_base.setdefault(Path(key).name, data)

    untouched: list[str] = []
    for rel in rel_paths:
        data = by_base.get(Path(rel).name)
        if data is None:
            continue
        summary = data.get("summary", {})
        num_statements = summary.get("num_statements", 0)
        covered = summary.get("covered_lines", 0)
        if num_statements > 0 and covered == 0:
            untouched.append(rel)
    return untouched


_JUDGE_INSTRUCTIONS = (
    "You are a strict code reviewer checking for test-gaming, not style.\n"
    "Given a task and a diff, decide whether the diff genuinely solves the "
    "stated task, or whether it games the tests (hardcodes an expected value, "
    "special-cases the test input, weakens an assertion, or is unrelated).\n"
    'Reply with ONLY a JSON object: {"verdict": "solves" | "games", '
    '"reason": "<one sentence>"}.'
)


def build_judge_prompt(task: str, diff: str) -> str:
    return (
        f"{_JUDGE_INSTRUCTIONS}\n\n"
        f"## Task\n{task.strip()}\n\n"
        f"## Diff\n{diff.strip()}\n"
    )


def parse_judge_verdict(response: str) -> GuardResult:
    verdict, reason = _extract_verdict(response)
    if verdict is None:
        return GuardResult(ok=True, reason="judge verdict unparseable — allowed without judgment")
    if verdict == "games":
        return GuardResult(ok=False, reason=f"LLM judge: diff appears to game the test — {reason}")
    return GuardResult(ok=True, reason=reason)


def _extract_verdict(response: str) -> tuple[str | None, str]:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict in {"solves", "games"}:
                return verdict, str(obj.get("reason", "")).strip()
        except ValueError:
            pass
    low = response.lower()
    if "games" in low and "solves" not in low:
        return "games", "keyword match"
    if "solves" in low and "games" not in low:
        return "solves", "keyword match"
    return None, ""


def _short(result) -> str:
    return (result.stderr or result.stdout or "").strip()[:200]
