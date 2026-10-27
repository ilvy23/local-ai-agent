"""What kind of project is this, and how is it tested? Detected from files, never guessed by the model.

The profile feeds three things: the system prompt (a few lines the model should not have to rediscover every run), the
verification gate (which command proves the work), and the lint step. Everything is derived from marker files, so it is
cheap, deterministic and the same across runs - which also keeps the start of the prompt byte-identical (cache friendly).
"""

from __future__ import annotations

import functools
import json
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from agent.codeagent import edits


@dataclass
class Profile:
    root: Path
    kinds: list[str] = field(default_factory=list)          # python, node, rust, go, make, nix, html ...
    test_cmd: str | None = None
    lint_cmd: str | None = None
    build_cmd: str | None = None
    run_hint: str | None = None
    tools: list[str] = field(default_factory=list)          # available helpers: ruff, pytest, mypy, node ...
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"PROJECT {self.root.name}  ({', '.join(self.kinds) or 'no code yet'})",
                 f"root: {self.root}  (every command already runs here: use relative paths, never `cd` somewhere else, never guess another path)"]
        if self.test_cmd:
            lines.append(f"tests: {self.test_cmd}")
        if self.lint_cmd:
            lines.append(f"lint: {self.lint_cmd}")
        if self.build_cmd:
            lines.append(f"build: {self.build_cmd}")
        if self.run_hint:
            lines.append(f"run: {self.run_hint}")
        if self.tools:
            lines.append("available: " + ", ".join(self.tools))
        lines += [f"- {n}" for n in self.notes[:6]]
        return "\n".join(lines)


NIXOS_MARKER = Path("/etc/NIXOS")
OS_RELEASE = Path("/etc/os-release")


def is_nixos() -> bool:
    """NixOS detected from /etc/NIXOS or ID=nixos in /etc/os-release."""
    try:
        if NIXOS_MARKER.exists():
            return True
        return bool(re.search(r"(?im)^ID=\"?nixos\"?\s*$", OS_RELEASE.read_text(errors="replace")))
    except OSError:
        return False


_PROBE = ("import sys, os, sysconfig\nprint(sys.version.split()[0])\n"
          "try:\n import pytest\n print('pytest')\nexcept Exception:\n print('no')\n"
          "print('ro' if not os.access(sysconfig.get_paths()['purelib'], os.W_OK) else 'rw')")


@functools.lru_cache(maxsize=1)
def probe_python() -> dict:
    """What the system python can do: version, is pytest importable, is uv there, is its site-packages read-only. (Cached for the process.)"""
    info = {"version": "", "pytest": False, "uv": bool(shutil.which("uv")), "readonly": False}
    try:
        r = subprocess.run(["python3", "-c", _PROBE], capture_output=True, text=True, timeout=8)
        out = r.stdout.split()
        if out:
            info["version"] = out[0]
            info["pytest"] = len(out) > 1 and out[1] == "pytest"
            info["readonly"] = len(out) > 2 and out[2] == "ro"
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def pytest_command() -> str:
    """How to run pytest HERE: directly when importable, else through uv (installing into the system python is not an option)."""
    info = probe_python()
    if info["pytest"] or not info["uv"]:
        return "python3 -m pytest -q -x"
    return "uv run --no-project --with pytest python3 -m pytest -q -x"


def _has(root: Path, *names: str) -> bool:
    return any((root / n).exists() for n in names)


def _make_targets(root: Path) -> set[str]:
    f = root / "Makefile"
    try:
        return set(re.findall(r"^([A-Za-z0-9_.-]+):", f.read_text(errors="replace"), re.M)) if f.exists() else set()
    except OSError:
        return set()


def _py_files(root: Path) -> list[Path]:
    return [p for p in edits.iter_files(root, 400) if p.suffix == ".py"]


def detect(root: Path | str, override_test: str | None = None) -> Profile:
    root = Path(root).resolve()
    prof = Profile(root)
    py = _py_files(root)
    if py or _has(root, "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"):
        prof.kinds.append("python")
        has_tests = any(p.name.startswith("test_") or p.name.endswith("_test.py") or "tests" in p.parts for p in py)
        pytest_cfg = _has(root, "pytest.ini", "tox.ini", "conftest.py") or (root / "tests").is_dir()
        try:
            data = tomllib.loads((root / "pyproject.toml").read_text()) if (root / "pyproject.toml").exists() else {}
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        if "pytest" in json.dumps(data.get("tool", {})) or pytest_cfg:
            prof.test_cmd = pytest_command()
        elif has_tests:
            prof.test_cmd = "python3 -m unittest discover -q" if not probe_python()["pytest"] and not probe_python()["uv"] else pytest_command()
        mains = [p.name for p in py if p.name in ("main.py", "app.py", "__main__.py", "game.py", "cli.py")]
        if mains:
            prof.run_hint = f"python3 {mains[0]}"
        if "ruff" in json.dumps(data.get("tool", {})) or shutil.which("ruff"):
            prof.lint_cmd = "ruff check ."
        elif shutil.which("pyflakes"):
            prof.lint_cmd = "python -m pyflakes ."
        for t in ("ruff", "mypy", "pyright", "pytest", "black"):
            if shutil.which(t):
                prof.tools.append(t)
    pkg = root / "package.json"
    if pkg.exists():
        prof.kinds.append("node")
        try:
            scripts = (json.loads(pkg.read_text()).get("scripts") or {})
        except (OSError, ValueError):
            scripts = {}
        runner = "pnpm" if _has(root, "pnpm-lock.yaml") else "yarn" if _has(root, "yarn.lock") else "npm"
        if "test" in scripts:
            prof.test_cmd = prof.test_cmd or f"{runner} test --silent"
        if "lint" in scripts:
            prof.lint_cmd = prof.lint_cmd or f"{runner} run lint"
        if "build" in scripts:
            prof.build_cmd = f"{runner} run build"
        if "dev" in scripts or "start" in scripts:
            prof.run_hint = prof.run_hint or f"{runner} run {'dev' if 'dev' in scripts else 'start'}"
        if shutil.which("node"):
            prof.tools.append("node")
    if (root / "Cargo.toml").exists():
        prof.kinds.append("rust")
        prof.test_cmd = prof.test_cmd or "cargo test -q"
        prof.build_cmd = prof.build_cmd or "cargo build"
        prof.lint_cmd = prof.lint_cmd or "cargo clippy -q"
    if (root / "go.mod").exists():
        prof.kinds.append("go")
        prof.test_cmd = prof.test_cmd or "go test ./..."
        prof.build_cmd = prof.build_cmd or "go build ./..."
    targets = _make_targets(root)
    if targets:
        prof.kinds.append("make")
        if "test" in targets and not prof.test_cmd:
            prof.test_cmd = "make test"
        if "build" in targets and not prof.build_cmd:
            prof.build_cmd = "make build"
        if "lint" in targets and not prof.lint_cmd:
            prof.lint_cmd = "make lint"
    if _has(root, "flake.nix", "default.nix", "shell.nix"):
        prof.kinds.append("nix")
    if any(p.suffix == ".html" for p in edits.iter_files(root, 200)):
        prof.kinds.append("web")
    if (root / ".git").exists():
        prof.notes.append("git repo: use `git diff` to review, commit only when asked")
    else:
        prof.notes.append("not a git repo: changes are tracked by the harness (/undo restores them)")
    info = probe_python()
    if info["version"]:
        ro = " (NixOS: the system python is read-only)" if info.get("readonly") and is_nixos() else ""
        env = f"python {info['version']}; " + ("pytest is installed" if info["pytest"] else
              f"pytest is NOT installed - never pip install into the system python{ro}. " +
              ("Run tests with: " + pytest_command() if info["uv"] else "write tests with unittest"))
        prof.notes.insert(0, env)
    if override_test:
        prof.test_cmd = override_test
    if "python" in prof.kinds and not prof.test_cmd:
        prof.notes.append("no tests yet: write test_*.py with real assertions for new logic, or run the program with sample input")
    return prof


def _syntax_commands(edited: list[str], check_cmd: str | None) -> list[str]:
    """Cheap syntax commands for the non-Python files among `edited`; a tool that is not installed adds nothing."""
    def quoted(suffixes: tuple[str, ...]) -> list[str]:
        return [shlex.quote(e) for e in edited if e.lower().endswith(suffixes)][:40]

    out: list[str] = []
    if check_cmd:
        custom = edits.custom_suffixes()
        for e in [e for e in edited if Path(e).suffix.lower() in custom][:10]:
            cmd = edits.custom_cmd(check_cmd, e)
            if cmd:
                out.append(cmd)
        return out
    js = quoted((".js", ".mjs", ".cjs"))
    if js and shutil.which("node"):
        out.append(" && ".join(f"node --check {f}" for f in js[:10]))
    sh = quoted((".sh", ".bash"))
    if sh and shutil.which("bash"):
        out.append(" && ".join(f"bash -n {f}" for f in sh[:10]))
    go = quoted((".go",))
    if go and shutil.which("gofmt"):
        out.append("gofmt -e " + " ".join(go) + " >/dev/null")
    return out


def check_commands(prof: Profile, edited: list[str], check_cmd: str | None = None) -> list[str]:
    """Cheap-to-expensive commands that prove the edited files are sane. Python always gets py_compile (so a bare
    folder of scripts is still checked); node / bash / gofmt syntax checks follow for their files when installed (or the
    project's `coding.check_cmd`); the project's own test/lint command comes last."""
    cmds: list[str] = []
    py = [e for e in edited if e.endswith(".py")]
    if py:
        quoted = " ".join(shlex.quote(p) for p in py[:40])           # any file name, quotes and unicode included
        cmds.append(f"python3 -m py_compile {quoted}")
    if prof.lint_cmd and py and "ruff" in prof.lint_cmd:
        cmds.append(f"ruff check --select E9,F63,F7,F82,F821,F822,F823 {quoted}")
    cmds += _syntax_commands(edited, check_cmd)
    if prof.test_cmd:
        cmds.append(prof.test_cmd)
    return cmds
