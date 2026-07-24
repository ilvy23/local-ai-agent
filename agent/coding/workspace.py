from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self


class WorkspaceError(RuntimeError):
    pass


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def is_git_repo(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


@dataclass
class Workspace:
    root: Path
    branch: str
    _base: Path

    @property
    def path(self) -> Path:
        return self.root

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def diff(self) -> str:
        _git(["add", "-A"], cwd=self.root)
        return _git(["diff", "--cached"], cwd=self.root).stdout

    def changed_files(self) -> list[str]:
        _git(["add", "-A"], cwd=self.root)
        out = _git(["diff", "--cached", "--name-only"], cwd=self.root).stdout
        return [line for line in out.splitlines() if line]

    def reset(self) -> None:
        _git(["reset", "--hard", "HEAD"], cwd=self.root)
        _git(["clean", "-fd"], cwd=self.root)

    def commit(self, message: str) -> str:
        _git(["add", "-A"], cwd=self.root)
        _git(["commit", "-m", message, "--no-verify"], cwd=self.root)
        return _git(["rev-parse", "HEAD"], cwd=self.root).stdout.strip()

    def close(self, *, keep: bool = False) -> None:
        if keep:
            return
        try:
            _git(["worktree", "remove", "--force", str(self.root)], cwd=self._base)
        except WorkspaceError:
            shutil.rmtree(self.root, ignore_errors=True)
        try:
            _git(["branch", "-D", self.branch], cwd=self._base)
        except WorkspaceError:
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def create_workspace(base: Path, prefix: str = "agent-code") -> Workspace:
    base = Path(base).resolve()
    if not is_git_repo(base):
        raise WorkspaceError(f"{base} is not a git repository")

    token = uuid.uuid4().hex[:8]
    branch = f"{prefix}/{token}"
    root = Path(tempfile.mkdtemp(prefix=f"{prefix}-{token}-"))
    root.rmdir()

    _git(["worktree", "add", "-b", branch, str(root), "HEAD"], cwd=base)
    return Workspace(root=root, branch=branch, _base=base)
