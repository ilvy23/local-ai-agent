"""The only things `nightage doctor --fix` can ever run.

Rules (enforced here, not by the individual checks):
  * user-level only: a small allow-list of programs (`ollama pull`, `chmod`, `mkdir -p`); never sudo/su/doas, never a shell;
  * every fix is a closure that knows its exact command line; `describe()` shows it BEFORE anything runs;
  * `chmod` only touches the owner's own files with owner-only modes, `mkdir` only creates folders under the home folder or
    the project data folder, `ollama pull` only takes a plain model name;
  * a profile key is written with one line edit of config.yaml (a `.bak` copy is kept).
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

FORBIDDEN = frozenset({"sudo", "su", "doas", "pkexec", "sh", "bash", "zsh", "dash", "env", "xargs", "eval", "systemctl", "rm"})
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,100}$")
_MODES = frozenset({"600", "700", "u+rw", "u+rwx", "u+w", "u+r", "u+x", "0600", "0700"})


class FixRefused(Exception):
    """The command is not one doctor is allowed to run."""


def validate(argv: list[str], *, home: Path | None = None, extra_roots: tuple[Path, ...] = ()) -> None:
    if not argv or not all(isinstance(a, str) for a in argv):
        raise FixRefused("empty command")
    prog = os.path.basename(argv[0])
    if prog in FORBIDDEN or any(os.path.basename(a) in FORBIDDEN for a in argv[:1]):
        raise FixRefused(f"{prog} is never run by doctor")
    if prog == "ollama":
        if len(argv) != 3 or argv[1] != "pull" or not _MODEL.match(argv[2]) or argv[2].startswith("-"):
            raise FixRefused("only `ollama pull <model>` is allowed")
        return
    roots = (home or Path.home(), *extra_roots)
    if prog == "chmod":
        if len(argv) != 3 or argv[1] not in _MODES:
            raise FixRefused("chmod is only allowed with an owner-only mode (600, 700, u+rw, u+w, ...)")
        _under(Path(argv[2]), roots)
        return
    if prog == "mkdir":
        if len(argv) != 3 or argv[1] != "-p":
            raise FixRefused("only `mkdir -p <folder>` is allowed")
        _under(Path(argv[2]), roots)
        return
    raise FixRefused(f"{prog} is not on the allow-list (ollama pull, chmod, mkdir -p)")


def _under(path: Path, roots: tuple[Path, ...]) -> None:
    try:
        resolved = path.expanduser().resolve()
    except OSError as exc:
        raise FixRefused(f"cannot resolve {path}") from exc
    for r in roots:
        try:
            resolved.relative_to(r.expanduser().resolve())
            return
        except ValueError:
            continue
    raise FixRefused(f"{path} is outside your home folder and the project data folder")


def command_line(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def user_command(argv: list[str], *, timeout: float = 600.0, home: Path | None = None,
                 extra_roots: tuple[Path, ...] = ()) -> tuple[str, Callable[[], str]]:
    """(display string, closure). Raises FixRefused at construction so a bad fix never even appears in the list."""
    argv = list(argv)
    validate(argv, home=home, extra_roots=extra_roots)

    def run() -> str:
        validate(argv, home=home, extra_roots=extra_roots)             # again at run time: nothing changed meanwhile
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError:
            return f"{argv[0]} is not installed"
        except subprocess.TimeoutExpired:
            return f"timed out after {timeout:g} s"
        except OSError as exc:
            return f"could not run: {type(exc).__name__}"
        tail = ((r.stderr or r.stdout or "").strip().splitlines() or [""])[-1][:160]
        return "done" if r.returncode == 0 else f"failed (exit {r.returncode}): {tail}"

    return command_line(argv), run


def pull_model(name: str) -> tuple[str, Callable[[], str]]:
    return user_command(["ollama", "pull", name], timeout=3600.0)


def chmod_path(mode: str, path: Path, *, home: Path | None = None, extra_roots: tuple[Path, ...] = ()) -> tuple[str, Callable[[], str]]:
    return user_command(["chmod", mode, str(path)], timeout=10.0, home=home, extra_roots=extra_roots)


def make_dir(path: Path, *, home: Path | None = None, extra_roots: tuple[Path, ...] = ()) -> tuple[str, Callable[[], str]]:
    return user_command(["mkdir", "-p", str(path)], timeout=10.0, home=home, extra_roots=extra_roots)


def set_config_key(config_path: Path, key: str, value: str) -> tuple[str, Callable[[], str]]:
    """Write a top-level scalar key (e.g. `profile: phone`) into config.yaml by editing one line. Keeps config.yaml.bak."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", key) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise FixRefused("only simple top-level keys and values can be written")

    def run() -> str:
        try:
            text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
            if config_path.exists():
                shutil.copy2(config_path, config_path.with_name(config_path.name + ".bak"))
            line = f"{key}: {value}"
            pat = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
            new = pat.sub(line, text, count=1) if pat.search(text) else line + "\n" + text
            config_path.write_text(new, encoding="utf-8")
            return f"wrote `{line}` (backup: {config_path.name}.bak)"
        except OSError as exc:
            return f"could not write: {type(exc).__name__}"

    return f"set `{key}: {value}` in {config_path}", run
