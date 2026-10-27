"""`/undo`: take back what the agent wrote in its last prompts.

Two layers:
  * a stack of per-prompt file snapshots (the content of every edited file before the prompt and after
    it), so `/undo` restores exactly those files - and moves files the agent CREATED to a trash folder
    rather than deleting them. A file changed by someone else since the agent's edit is left alone.
  * inside a git repository a dated checkpoint ref `refs/agent/checkpoints/<time>` of the whole working
    tree is made before the first edit of a prompt (a private commit; your branch, index and working tree
    are not touched), so even a crash or restart leaves a way back: `git checkout <ref> -- .`
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from agent import osport
from typing import Any

MAX_TURNS = 10
TRASH_DIR = osport.data_home() / "agent" / "undo-trash"


def _git(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15,
                              env={**os.environ, **(env or {})}, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_checkpoint(path: str) -> str | None:
    """Private checkpoint commit of the working tree of the repo containing `path`; its ref name or None."""
    folder = Path(path).expanduser().parent
    if not folder.is_dir():
        for parent in folder.parents:
            if parent.is_dir():
                folder = parent
                break
    top = _git(["rev-parse", "--show-toplevel"], folder)
    head = _git(["rev-parse", "--verify", "-q", "HEAD"], folder) if top else None
    if not top or not head:
        return None
    root = Path(top)
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index"), "GIT_AUTHOR_NAME": "agent", "GIT_AUTHOR_EMAIL": "agent@localhost",
               "GIT_COMMITTER_NAME": "agent", "GIT_COMMITTER_EMAIL": "agent@localhost"}
        if _git(["read-tree", "HEAD"], root, env) is None or _git(["add", "-A"], root, env) is None:
            return None
        tree = _git(["write-tree"], root, env)
        commit = _git(["commit-tree", tree, "-p", head, "-m", "agent checkpoint before edits"], root, env) if tree else None
        if not commit:
            return None
        ref = f"refs/agent/checkpoints/{time.strftime('%Y%m%d-%H%M%S')}"
        return ref if _git(["update-ref", ref, commit], root) is not None else None


def trash(path: Path) -> Path:
    """Move `path` out of the way (never delete). Tries the desktop trash first."""
    for cmd in (["gio", "trash", str(path)], ["trash-put", str(path)]):
        if shutil.which(cmd[0]):
            try:
                if subprocess.run(cmd, capture_output=True, timeout=15, check=False).returncode == 0:
                    return path
            except (OSError, subprocess.SubprocessError):
                pass
    dest = TRASH_DIR / time.strftime("%Y%m%d-%H%M%S") / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return dest


class UndoStack:
    def __init__(self, limit: int = MAX_TURNS) -> None:
        self._turns: list[dict[str, Any]] = []
        self._limit = limit

    def push(self, files: dict[str, tuple[str | None, str]], checkpoint: str | None = None) -> None:
        if files:
            self._turns.append({"files": dict(files), "checkpoint": checkpoint, "time": time.strftime("%H:%M")})
            del self._turns[:-self._limit]

    def __len__(self) -> int:
        return len(self._turns)

    def describe(self) -> list[str]:
        if not self._turns:
            return ["nothing to undo"]
        lines = []
        for i, turn in enumerate(reversed(self._turns), 1):
            names = ", ".join(Path(p).name for p in list(turn["files"])[:4]) + (" ..." if len(turn["files"]) > 4 else "")
            ref = f"  git: {turn['checkpoint']}" if turn["checkpoint"] else ""
            lines.append(f"{i}. {turn['time']}  {len(turn['files'])} file(s): {names}{ref}")
        return lines

    def undo(self) -> list[str]:
        """Undo the most recent prompt's edits. Returns lines describing what happened."""
        if not self._turns:
            return ["nothing to undo"]
        turn = self._turns.pop()
        out = []
        for path, (before, after) in turn["files"].items():
            p = Path(path).expanduser()
            current = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
            if current != after:
                out.append(f"skipped {p.name}: it was changed after the agent's edit")
                continue
            try:
                if before is None:
                    moved = trash(p)
                    out.append(f"removed {p.name} (new file, moved to trash)")
                    del moved
                else:
                    p.write_text(before, encoding="utf-8")
                    out.append(f"restored {p.name}")
            except OSError as exc:
                out.append(f"could not undo {p.name}: {exc}")
        return out or ["nothing changed"]
