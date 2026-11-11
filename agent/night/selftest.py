"""`nightage night selftest`: the whole night on a toy project, in a temp folder, in minutes - to be run after any change and before trusting a long night with a new model
or a new setting. It builds a tiny Python project with a real test suite and two prompts (the second depends on the first), runs the real `night run` chain on it
(canary included) and then CHECKS the result by measurement: the new files exist, the project's tests pass, there are git checkpoints and a report, and no prompt was left open."""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT = {
    "calc.py": '"""A tiny calculator."""\n\n\ndef add(a, b):\n    return a + b\n',
    "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
}
PROMPTS = {
    "01-subtract.md": ("---\nname: subtract\n---\nAdd a function `sub(a, b)` to `calc.py` that returns a minus b, and a test in `tests/test_calc.py` named test_sub that checks "
                       "sub(5, 3) == 2. The whole test suite must pass."),
    "02-divide.md": ("---\nname: divide\ndepends: subtract\n---\nAdd a function `div(a, b)` to `calc.py` that returns a divided by b and raises ValueError('division by zero') when b is 0, "
                     "and tests for both cases in `tests/test_calc.py`. The whole test suite must pass."),
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def build(root: Path) -> Path:
    """Write the toy project and its prompt folder; returns the prompt folder."""
    for rel, text in PROJECT.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    prompts = root.parent / (root.name + "-prompts")
    prompts.mkdir(exist_ok=True)
    for name, text in PROMPTS.items():
        (prompts / name).write_text(text + "\n")
    return prompts


def command(root: Path, prompts: Path, hours: float) -> list[str]:
    return [sys.executable, "-c", "from agent.main import app; app()", "night", "run", str(prompts), "--folder", str(root), "--hours", str(hours), "--yes", "--no-model", "--ignore-games"]


def test_command() -> list[str]:
    """pytest is not part of the nightage tool environment: borrow one (uv) when there is uv, else the interpreter that runs this."""
    if shutil.which("uv"):
        return ["uv", "run", "--no-project", "--with", "pytest", "python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]


def verify(root: Path) -> list[Check]:
    """The measured outcome of a selftest night."""
    out = []
    calc = (root / "calc.py").read_text() if (root / "calc.py").exists() else ""
    out.append(Check("sub exists", "def sub" in calc))
    out.append(Check("div exists", "def div" in calc))
    t = subprocess.run(test_command(), cwd=root, capture_output=True, text=True, timeout=300)
    out.append(Check("tests pass", t.returncode == 0, t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-120:]))
    tags = subprocess.run(["git", "tag", "-l", "night/green-*"], cwd=root, capture_output=True, text=True).stdout.split()
    out.append(Check("git checkpoints", len(tags) >= 1, f"{len(tags)} green checkpoint(s)"))
    out.append(Check("report written", (root / ".night" / "NIGHT_REPORT.md").exists()))
    report = (root / ".night" / "NIGHT_REPORT.md").read_text() if (root / ".night" / "NIGHT_REPORT.md").exists() else ""
    out.append(Check("both prompts done", "| subtract | done |" in report and "| divide | done |" in report))
    return out


def text(checks: list[Check]) -> str:
    rows = [f"{'PASS' if c.ok else 'FAIL'}  {c.name}" + (f": {c.detail}" if c.detail else "") for c in checks]
    return "\n".join(rows) + ("\n\nselftest PASSED" if all(c.ok for c in checks) else "\n\nselftest FAILED")
