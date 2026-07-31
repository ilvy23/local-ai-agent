"""Isolated git-worktree sandbox for a coding session.

A worktree gives us a free snapshot of the repo on a scratch branch: the model
edits and verifies there, never touching your working tree. On success the branch
is kept (auto-committed, ready to merge or cherry-pick); on abandon it's removed.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from agent.coding.verify import runner


class WorkspaceError(RuntimeError):
    pass


def _git(args: list[str], cwd: str | Path) -> runner.RunResult:
    return runner.run(["git", *args], cwd=cwd, timeout=60)


class Workspace:
    """Context manager: `with Workspace(repo) as ws: ... edit ws.path ...`."""

    def __init__(self, repo_root: str | Path, branch: str | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.branch = branch or f"agent-code/{int(time.time())}"
        self.keep = False  # session sets True to preserve the branch on success
        self.path: Path | None = None
        self._tmp: Path | None = None

    def __enter__(self) -> "Workspace":
        if not _git(["rev-parse", "--git-dir"], self.repo_root).ok:
            raise WorkspaceError(f"{self.repo_root} is not a git repository")
        if not _git(["rev-parse", "HEAD"], self.repo_root).ok:
            raise WorkspaceError("repository has no commits yet — commit once first")
        self._tmp = Path(tempfile.mkdtemp(prefix="agent-code-"))
        self.path = self._tmp / "wt"
        r = _git(["worktree", "add", "-b", self.branch, str(self.path), "HEAD"], self.repo_root)
        if not r.ok:
            shutil.rmtree(self._tmp, ignore_errors=True)
            raise WorkspaceError(f"git worktree add failed: {r.output}")
        return self

    def read(self, rel: str) -> str:
        return (self.path / rel).read_text(encoding="utf-8")

    def write(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def diff(self) -> str:
        _git(["add", "-A"], self.path)  # stage so new files show in the diff too
        return _git(["diff", "--cached"], self.path).stdout

    def commit(self, message: str) -> None:
        _git(["add", "-A"], self.path)
        _git(["commit", "-m", message, "--no-verify"], self.path)

    def __exit__(self, *exc_info: object) -> None:
        if self.path is not None:
            _git(["worktree", "remove", "--force", str(self.path)], self.repo_root)
            if not self.keep:
                _git(["branch", "-D", self.branch], self.repo_root)
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
