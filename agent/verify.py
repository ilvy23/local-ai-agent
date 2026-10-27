"""Verification gate: before the agent may call code work "done", run the project's own check.

A 27B model happily says "all green" without having run anything. After a turn in which files were
written or edited, the final answer is held back until a check ran:

  * the project's test suite when one is detectable (pytest / npm test / cargo test / go test), else
  * a syntax check of the edited Python files.

The check goes through the normal tool gate (`run_command`, so the user is asked, or `/auto-<dir>` lets it
run). A failure goes back to the model as "you are not done, fix this" - at most `retries` times, then
the answer is shown with an honest note. `autonomy.verify: false` or `AGENT_VERIFY=0` switches it off.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CODE_EXTENSIONS = frozenset({".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".cc", ".cpp", ".h",
                             ".java", ".kt", ".rb", ".php", ".sh", ".lua", ".nix"})
_ROOT_MARKERS = ("pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "package.json", "Cargo.toml", "go.mod",
                 "tests", "test", ".git")
_EXIT = re.compile(r"exit code (-?\d+)")
OUTPUT_TAIL = 1800


def enabled(config: dict[str, Any]) -> bool:
    if os.environ.get("AGENT_VERIFY", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool((config.get("autonomy") or {}).get("verify", True))


def is_code(path: str) -> bool:
    return Path(path).suffix.lower() in CODE_EXTENSIONS


def project_root(path: str) -> Path | None:
    """Nearest folder above `path` that looks like a project root (never home or the filesystem root)."""
    start = Path(path).expanduser()
    for folder in [start.parent, *start.parent.parents][:7]:
        if folder == Path.home() or len(folder.parts) < 3:
            return None
        if any((folder / m).exists() for m in _ROOT_MARKERS):
            return folder
    return None


def _has_pytest_signs(root: Path) -> bool:
    if (root / "pytest.ini").exists() or (root / "tox.ini").exists() or (root / "tests").is_dir():
        return True
    try:
        if "pytest" in (root / "pyproject.toml").read_text(errors="replace"):
            return True
    except OSError:
        pass
    return any(root.glob("test_*.py")) or any(root.glob("*_test.py"))


def check_command(root: Path, edited: list[str]) -> str | None:
    """The shell command that checks `root`, or None when nothing sensible exists."""
    cd = f"cd {shlex.quote(str(root))} && "
    if (root / "Cargo.toml").exists():
        return cd + "cargo test -q"
    if (root / "go.mod").exists():
        return cd + "go test ./..."
    pkg = root / "package.json"
    if pkg.exists():
        try:
            script = (json.loads(pkg.read_text()).get("scripts") or {}).get("test", "")
        except (OSError, ValueError):
            script = ""
        if script and "no test specified" not in script:
            return cd + "npm test --silent"
    py_files = [f for f in edited if f.endswith(".py")]
    if py_files and _has_pytest_signs(root):
        runner = "uv run --no-project --with pytest python -m pytest" if shutil.which("uv") else "python3 -m pytest"
        return cd + runner + " -q -x"
    if py_files:
        rel = []
        for f in py_files:
            try:
                rel.append(shlex.quote(str(Path(f).expanduser().resolve().relative_to(root))))
            except ValueError:
                rel.append(shlex.quote(f))
        return cd + "python3 -m py_compile " + " ".join(rel)
    return None


def exit_code(result: str) -> int | None:
    m = _EXIT.search((result or "")[:500])
    return int(m.group(1)) if m else None


def output_tail(result: str) -> str:
    return result[-OUTPUT_TAIL:] if len(result) > OUTPUT_TAIL else result


@dataclass
class VerifyGate:
    retries: int = 2
    tries: int = 0
    dirty: list[str] = field(default_factory=list)      # code files edited since the last passing check

    def note_edit(self, path: str) -> None:
        if is_code(path) and path not in self.dirty:
            self.dirty.append(path)

    def due(self) -> bool:
        return bool(self.dirty) and self.tries <= self.retries

    def command(self) -> str | None:
        root = project_root(self.dirty[0]) if self.dirty else None
        return check_command(root, self.dirty) if root else None

    def passed(self) -> None:
        self.dirty.clear()
        self.tries = 0

    def failed(self) -> bool:
        """Record a failed check. True if the model gets another try."""
        self.tries += 1
        return self.tries <= self.retries

    def give_up(self) -> None:
        self.dirty.clear()


def failure_message(command: str, result: str) -> str:
    return (
        "VERIFICATION FAILED. You said the work was done, but the project's own check does not pass. "
        f"Command: {command}\nResult (end of output):\n{output_tail(result)}\n\n"
        "You are NOT done. Fix the cause, re-run the check yourself, and only then report completion. "
        "If you cannot fix it, say exactly what fails instead of claiming success."
    )
