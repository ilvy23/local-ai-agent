"""Guards on what an attempt did to the project, checked from git (against the last green checkpoint), not from what the model says.

The classic overnight failure is a model that gets the tests green by weakening them. `check()` finds, in this order of seriousness:
  protected   a file the prompt must never touch changed (CI, lockfiles, .git*, the charter's own files)            fatal -> reverted
  deleted     a test file was deleted                                                                              fatal -> reverted
  weakened    a test file has fewer `assert`s than before, or got a skip / xfail marker                           fatal -> reverted
  syntax      a changed .py file does not compile                                                                 fatal (the agent must fix it)
  deleted_src a source file was deleted, or one of 40+ lines shrank by more than 60%, and the prompt did not ask for removing / refactoring  fatal -> reverted
  bloat       a NEW file over 5 MB (a venv, weights, a log dump)                                                                                  fatal -> removed
  vacuous     new tests that contain no assertion at all                                                                                          warning
  lint        real errors ruff finds in the changed Python files (undefined names ...), if ruff is installed                warning
  stubs       added lines that are placeholders (raise NotImplementedError, a bare pass, TODO/FIXME)               warning
  scope       more files / more added lines than the budget                                                       warning
`revert()` restores the fatal paths from the checkpoint so the next attempt starts from honest ground."""
from __future__ import annotations

import fnmatch
import ast
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agent.night import gitbox

TEST_PATH = re.compile(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_[^/]*\.py$|_test\.(py|go|js|ts)$|\.(test|spec)\.(js|ts|jsx|tsx)$")
ASSERT = re.compile(r"\bassert\b|\bexpect\(|\bassert[A-Z]\w*\(|\bassert_\w+\(|\bt\.(?:Error|Fatal)f?\(")
SKIP = re.compile(r"@pytest\.mark\.(?:skip|skipif|xfail)|\bpytest\.(?:skip|xfail)\(|@unittest\.(?:skip|skipIf|skipUnless)|\bt\.Skip\(|\b(?:it|test|describe)\.skip\(|\bxit\(|\bxdescribe\(")
STUB = re.compile(r"raise\s+NotImplementedError|^\+\s*pass\s*(#.*)?$|\bTODO\b|\bFIXME\b|^\+\s*\.\.\.\s*$|unimplemented!\(\)|panic\(\"not implemented\"")
PROTECTED = (".github/*", ".gitlab-ci.yml", ".gitignore", ".git/*", "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
             "Cargo.lock", "go.sum", ".night/*", "Jenkinsfile", ".circleci/*")


REMOVAL_WORDS = re.compile(r"\b(remove|delete|drop|rename|move|refactor|clean ?up|migrate|replace|split|extract|merge|consolidate|simplif\w*|rewrite)\b", re.I)
CODE = re.compile(r"\.(py|js|ts|tsx|jsx|go|rs|java|c|h|cpp|rb|php|cs|kt|swift)$")
BLOAT_BYTES = 5 * 1024 * 1024
SHRINK_MIN_LINES, SHRINK_RATIO = 40, 0.4


@dataclass
class Budget:
    max_files: int = 60
    max_added: int = 4000
    protected: tuple[str, ...] = PROTECTED
    allow_new_tests: bool = True


@dataclass
class Finding:
    kind: str
    path: str
    detail: str
    fatal: bool = False


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def fatal(self) -> list[Finding]:
        return [f for f in self.findings if f.fatal]

    def text(self) -> str:
        return "; ".join(f"{f.kind} {f.path}: {f.detail}" for f in self.findings)


def is_test(path: str) -> bool:
    return bool(TEST_PATH.search(path))


def _show(root: Path, tag: str, path: str) -> str | None:
    p = gitbox.git(root, "show", f"{tag}:{path}")
    return p.stdout if p.returncode == 0 else None


def _changed(root: Path, tag: str) -> list[tuple[str, str]]:
    """[(status, path)] of the working tree against `tag`: A added, M modified, D deleted (untracked files count as added)."""
    gitbox.git(root, "add", "-N", ".")
    out = gitbox.git(root, "diff", "--name-status", tag).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0][0], parts[-1]))
    return rows


def _added_lines(root: Path, tag: str) -> list[str]:
    return [ln for ln in gitbox.git(root, "diff", "-U0", tag).stdout.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def vacuous_tests(path: Path) -> int:
    """The number of test functions in a Python file that contain no assertion, no `pytest.raises` and no `self.assert*` call."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (SyntaxError, OSError, ValueError):
        return 0
    bad = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            checks = (isinstance(n, ast.Assert) or (isinstance(n, ast.Call) and (getattr(n.func, "attr", "").startswith(("assert", "raises", "fail")) or getattr(n.func, "id", "") == "fail"))
                      for n in ast.walk(node))
            bad += 0 if any(checks) else 1
    return bad


def check(root: Path, since_tag: str, budget: Budget | None = None, allow_removal: bool = False) -> Report:
    b = budget or Budget()
    rep = Report()
    changed = _changed(root, since_tag)
    for status, path in changed:
        if any(fnmatch.fnmatch(path, pat) for pat in b.protected):
            rep.findings.append(Finding("protected", path, "a protected file was changed", True))
            continue
        if not is_test(path):
            continue
        if status == "D":
            rep.findings.append(Finding("deleted", path, "a test file was deleted", True))
        elif status == "M":
            old, new = _show(root, since_tag, path) or "", (root / path).read_text(errors="replace") if (root / path).exists() else ""
            if len(ASSERT.findall(new)) < len(ASSERT.findall(old)):
                rep.findings.append(Finding("weakened", path, f"{len(ASSERT.findall(old))} -> {len(ASSERT.findall(new))} assertions", True))
            elif len(SKIP.findall(new)) > len(SKIP.findall(old)):
                rep.findings.append(Finding("weakened", path, "a skip / xfail marker was added", True))
        elif status == "A" and (root / path).exists() and SKIP.search((root / path).read_text(errors="replace")):
            rep.findings.append(Finding("weakened", path, "a new test file contains skip / xfail markers", True))
    for status, path in changed:
        if is_test(path) or any(fnmatch.fnmatch(path, pat) for pat in b.protected) or allow_removal or not CODE.search(path):
            continue
        if status == "D":
            rep.findings.append(Finding("deleted_src", path, "a source file was deleted", True))
        elif status == "M" and (root / path).is_file():
            old = _show(root, since_tag, path) or ""
            n_old, n_new = old.count("\n"), (root / path).read_text(errors="replace").count("\n")
            if n_old >= SHRINK_MIN_LINES and n_new < n_old * SHRINK_RATIO:
                rep.findings.append(Finding("deleted_src", path, f"shrank from {n_old} to {n_new} lines", True))
    for status, path in changed:
        f = root / path
        if status == "A" and f.is_file() and f.stat().st_size > BLOAT_BYTES:
            rep.findings.append(Finding("bloat", path, f"a new file of {f.stat().st_size // 2**20} MB", True))
        if status == "A" and is_test(path) and path.endswith(".py") and f.is_file():
            n = vacuous_tests(f)
            if n:
                rep.findings.append(Finding("vacuous", path, f"{n} new test(s) without any assertion"))
    for status, path in changed:
        if status != "D" and path.endswith(".py") and (root / path).is_file():
            problem = syntax_problem(root / path)
            if problem:
                rep.findings.append(Finding("syntax", path, problem, True))
    lint = lint_problems(root, [p for st, p in changed if st != "D"])
    if lint:
        rep.findings.append(Finding("lint", "", f"{len(lint)} real error(s), e.g. {lint[0][:120]}"))
    added = _added_lines(root, since_tag)
    stubs = [ln for ln in added if STUB.search(ln)]
    if len(stubs) >= 3:
        rep.findings.append(Finding("stubs", "", f"{len(stubs)} placeholder lines were added (e.g. {stubs[0][1:60].strip()!r})"))
    if len(changed) > b.max_files or len(added) > b.max_added:
        rep.findings.append(Finding("scope", "", f"{len(changed)} files, {len(added)} added lines (budget {b.max_files} / {b.max_added})"))
    return rep


def lint_problems(root: Path, paths: list[str], run=subprocess.run, limit: int = 5) -> list[str]:
    """Real errors only (syntax, undefined names, bad f-strings: ruff's E9, F63, F7, F82) in the changed Python files, when ruff is installed. [] otherwise."""
    py = [p for p in paths if p.endswith(".py") and (root / p).is_file()]
    exe = shutil.which("ruff")
    if not py or not exe:
        return []
    try:
        out = run([exe, "check", "--select", "E9,F63,F7,F82", "--output-format", "concise", "--no-cache", *py], cwd=root, capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.strip() for ln in out.splitlines() if re.match(r"^[\w./-]+:\d+:\d+:", ln.strip())][:limit]


def revert(root: Path, since_tag: str, report: Report) -> list[str]:
    """Restore the fatal test/protected paths from the checkpoint (syntax errors are left: the model has to fix its own code). Returns the paths."""
    paths = [f.path for f in report.fatal if f.kind in ("protected", "deleted", "weakened", "deleted_src", "bloat")]
    done = []
    for p in paths:
        if _show(root, since_tag, p) is not None:
            if gitbox.git(root, "checkout", since_tag, "--", p).returncode == 0:
                done.append(p)
        elif (root / p).exists():                                   # a NEW protected / weakened file: it did not exist before
            (root / p).unlink()
            done.append(p)
    return done


def syntax_problem(path: Path) -> str | None:
    """None when the Python file compiles, else 'message (line N)'. Compiles in memory: no .pyc is written anywhere."""
    try:
        compile(path.read_bytes(), str(path), "exec")
    except SyntaxError as e:
        return f"{e.msg} (line {e.lineno})"
    except (OSError, ValueError) as e:
        return None if isinstance(e, OSError) else str(e)[:100]
    return None


def python() -> str:
    return sys.executable
