from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from agent.coding.verify.runner import RunResult, run, run_python


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASSED

    @property
    def blocking(self) -> bool:
        return self.status is CheckStatus.FAILED


def check_syntax(paths: list[Path]) -> CheckResult:
    for path in paths:
        if path.suffix != ".py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return CheckResult("ast", CheckStatus.FAILED, f"cannot read {path.name}: {exc}")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            where = f"{Path(exc.filename).name}:{exc.lineno}:{exc.offset}"
            return CheckResult("ast", CheckStatus.FAILED, f"{where}: {exc.msg}")
    return CheckResult("ast", CheckStatus.PASSED)


def check_ruff(cwd: Path, timeout_seconds: float = 30.0) -> CheckResult:
    result = run(["ruff", "check", "."], cwd=cwd, timeout_seconds=timeout_seconds)
    if _tool_missing(result):
        return CheckResult("ruff", CheckStatus.SKIPPED, "ruff not installed")
    if result.ok:
        return CheckResult("ruff", CheckStatus.PASSED)
    return CheckResult("ruff", CheckStatus.FAILED, _trim(result.stdout or result.stderr))


def autofix_ruff(cwd: Path, timeout_seconds: float = 30.0) -> bool:
    result = run(["ruff", "check", "--fix", "."], cwd=cwd, timeout_seconds=timeout_seconds)
    return not _tool_missing(result)


def check_pytest(cwd: Path, timeout_seconds: float = 120.0) -> CheckResult:
    result = run_python(
        ["pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    if result.timed_out:
        return CheckResult("pytest", CheckStatus.FAILED, _trim(result.stdout + result.stderr))
    if result.returncode == 5:
        return CheckResult("pytest", CheckStatus.SKIPPED, "no tests collected")
    if _tool_missing(result):
        return CheckResult("pytest", CheckStatus.SKIPPED, "pytest not installed")
    if result.ok:
        return CheckResult("pytest", CheckStatus.PASSED, _trim(result.stdout))
    return CheckResult("pytest", CheckStatus.FAILED, _trim(result.stdout + result.stderr))


def run_ladder(
    changed_files: list[Path],
    cwd: Path,
    *,
    timeout_seconds: float = 120.0,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    syntax = check_syntax(changed_files)
    results.append(syntax)
    if syntax.blocking:
        return results

    autofix_ruff(cwd, timeout_seconds=min(timeout_seconds, 30.0))
    ruff = check_ruff(cwd, timeout_seconds=min(timeout_seconds, 30.0))
    results.append(ruff)
    if ruff.blocking:
        return results

    results.append(check_pytest(cwd, timeout_seconds=timeout_seconds))
    return results


def ladder_passed(results: list[CheckResult]) -> bool:
    if any(r.blocking for r in results):
        return False
    return any(r.passed for r in results)


def _tool_missing(result: RunResult) -> bool:
    if result.returncode == 127:
        return True
    text = (result.stderr or "") + (result.stdout or "")
    return "No module named" in text or "command not found" in text


def _trim(text: str, max_chars: int = 4000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n[... trimmed ...]\n{tail}"
