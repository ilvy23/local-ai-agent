"""The 'tests run the standard way' check: the harness runs `python -m pytest` (the cwd is on sys.path), a user types plain `pytest`.

Measured live: a model-built 3-module CLI had tests/test_*.py doing `from models import Book`. They passed with `python -m pytest`, but plain
`pytest` died with 3 collection errors (ModuleNotFoundError: No module named 'models') and nobody noticed. No model call: one deterministic,
cheap run (collection only) of the bare form, and only when no pytest configuration already puts the project root on sys.path.

    standard_command(test_cmd)           the bare-pytest command (collection only) or None when pytest cannot be started
    applies(root, test_cmd)              is the check meaningful here (python -m pytest + test files below the root + no import-path config)
    check(root, test_cmd, runner)        Finding or None. `runner(cmd, timeout) -> (exit code | None, output, seconds)` is the engine's `_exec`
    stamp(root)                          content stamp of the python sources, test files and pytest config: the cache key

The engine owns the nudge (cap 1, then NOT VERIFIED), the cache and the policy (engine._std_gate); verified.build shows one TESTS line."""

from __future__ import annotations

import hashlib
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.codeagent import edits, profile

Runner = Callable[[str, int], "tuple[int | None, str, float]"]

TIMEOUT = 60
CONFIG_FILES = ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "conftest.py", "tests/conftest.py", "tests/__init__.py")
_NO_MODULE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
LINE = "plain pytest: FAILS (import path)"


@dataclass(frozen=True)
class Finding:
    module: str                                  # a top-level module / package of the repo that plain pytest cannot import

    @property
    def not_verified(self) -> str:
        return f"tests only pass with python -m pytest: plain pytest fails to import {self.module}"

    @property
    def nudge(self) -> str:
        return (f"Plain `pytest` fails with ModuleNotFoundError: {self.module} because the project root is not on sys.path. Fix it for real users: "
                "add a conftest.py in the project root containing the single comment line `# makes pytest put the project root on sys.path` (pytest then inserts the root) or a pytest.ini with `pythonpath = .`; "
                "then both `pytest` and `python -m pytest` must pass.")


def standard_command(test_cmd: str) -> str | None:
    """Plain `pytest` (never the cwd on sys.path), collection only: that is where the import errors show and it costs seconds, not a suite run."""
    if "-m pytest" not in test_cmd:
        return None
    if test_cmd.lstrip().startswith("uv run"):
        return "uv run --no-project --with pytest pytest -q --collect-only -p no:cacheprovider"
    if shutil.which("pytest"):
        return "pytest -q --collect-only -p no:cacheprovider"
    if profile.probe_python()["pytest"]:              # what the console script does: no '' / cwd entry in sys.path
        return ("python3 -c \"import sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')]; from pytest import console_main; "
                "sys.exit(console_main())\" -q --collect-only -p no:cacheprovider")
    if profile.probe_python()["uv"]:
        return "uv run --no-project --with pytest pytest -q --collect-only -p no:cacheprovider"
    return None


def module_command(test_cmd: str) -> str:
    """The harness' own form, collection only (every failure listed: no -x)."""
    cmd = re.sub(r"(?<=\s)-x(?=\s|$)", "", test_cmd)
    return re.sub(r"\s+", " ", cmd).strip() + " --collect-only"


def _test_files(root: Path) -> list[Path]:
    out = []
    for p in edits.iter_files(root, 400):
        if p.suffix == ".py" and (p.name.startswith("test_") or p.name.endswith("_test.py")):
            out.append(p)
    return out


def _fixes_path(root: Path) -> bool:
    """A pytest configuration that already fixes the import path (or makes the tests a package): the check is skipped."""
    if (root / "conftest.py").exists() or (root / "tests" / "__init__.py").exists():
        return True                                     # rootdir conftest: pytest inserts the root; package tests: the top package's parent
    try:
        if "sys.path" in (root / "tests" / "conftest.py").read_text(errors="replace"):
            return True
    except OSError:
        pass
    for name in ("pytest.ini", "tox.ini", "setup.cfg"):
        try:
            text = (root / name).read_text(errors="replace")
        except OSError:
            continue
        if re.search(r"(?im)^\s*pythonpath\s*[=:]", text):
            return True
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text()) if (root / "pyproject.toml").exists() else {}
    except (OSError, tomllib.TOMLDecodeError):
        data = {}
    opts = ((data.get("tool") or {}).get("pytest") or {}).get("ini_options") or {}
    return bool(opts.get("pythonpath") or opts.get("rootdir"))


def applies(root: Path | str, test_cmd: str | None) -> bool:
    root = Path(root)
    if not test_cmd or "-m pytest" not in test_cmd:
        return False
    if not any(p.parent != root for p in _test_files(root)):           # flat test files: pytest puts their folder (= the root) on sys.path
        return False
    return not _fixes_path(root)


def _repo_module(root: Path, output: str) -> str | None:
    """The first top-level name in a `No module named` line that exists as a .py file / package in the repo."""
    for m in _NO_MODULE.finditer(output):
        top = m.group(1).split(".")[0]
        if (root / f"{top}.py").is_file() or (root / top / "__init__.py").is_file() or (root / top).is_dir() and any((root / top).glob("*.py")):
            return top
    return None


def check(root: Path | str, test_cmd: str | None, runner: Runner, timeout: int = TIMEOUT) -> Finding | None:
    root = Path(root)
    if not applies(root, test_cmd):
        return None
    bare = standard_command(str(test_cmd))
    if bare is None:
        return None
    code, out, _secs = runner(bare, timeout)
    if code is None or code == 0:
        return None                                      # stopped / fine
    module = _repo_module(root, out)
    if module is None:
        return None
    code, out, _secs = runner(module_command(str(test_cmd)), timeout)
    if code is None or (code != 0 and _repo_module(root, out) == module):
        return None                                      # both fail the same way: another problem (or the module really is missing)
    return Finding(module)


def stamp(root: Path | str) -> str:
    """Content of the python sources, the test files and the pytest config: the same code is judged once."""
    root = Path(root)
    h = hashlib.sha1()
    paths: list[Path] = [p for p in edits.iter_files(root, 400) if p.suffix == ".py"]
    paths += [root / n for n in CONFIG_FILES]
    for p in sorted(set(paths)):
        try:
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
        except OSError:
            h.update(str(p.relative_to(root)).encode() + b"\0gone\0")
    return h.hexdigest()


def touched(changed: Any) -> bool:
    """Did this prompt / stage edit a python file (a test or a source)?"""
    return any(str(p).endswith(".py") for p in changed)
