"""Checkpoints in git. The agent's own commits are off during a night run; these are the only ones: a state whose tests passed."""
from __future__ import annotations

import subprocess
from pathlib import Path

_ID = ["-c", "user.name=nightage", "-c", "user.email=night@localhost", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]


GIT_TIMEOUT_S = 180


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """git with a time limit: a hung git (a stale lock, a dead network mount) must cost one failed command, never the night."""
    try:
        return subprocess.run(["git", *_ID, *args], cwd=root, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(["git", *args], 124, "", f"git {' '.join(args[:2])} timed out after {GIT_TIMEOUT_S} s")


def ensure_repo(root: Path) -> None:
    """A repo with .night/ excluded and a baseline commit tagged night/base (so a roll back always has somewhere to go)."""
    if not (root / ".git").exists():
        git(root, "init", "-q")
    exclude = root / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    text = exclude.read_text() if exclude.exists() else ""
    missing = [e for e in (".night/", "__pycache__/", ".pytest_cache/", ".scratch/", "*.pyc") if e not in text.split()]
    if missing:
        exclude.write_text(text + ("\n" if text and not text.endswith("\n") else "") + "\n".join(missing) + "\n")
    if git(root, "tag", "-l", "night/base").stdout.strip():
        return
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", "night: baseline before the run")
    git(root, "tag", "night/base")


def _tags(root: Path, kind: str) -> list[str]:
    out = git(root, "tag", "-l", f"night/{kind}-*").stdout.split()
    return sorted(out, key=lambda t: int(t.rsplit("-", 1)[1]))


def checkpoint(root: Path, label: str) -> str:
    """Commit the working tree (if it changed) and tag it night/green-N. Returns the tag."""
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", f"night: {label}")
    green = _tags(root, "green")
    tag = f"night/green-{(int(green[-1].rsplit('-', 1)[1]) + 1) if green else 1}"
    git(root, "tag", tag)
    return tag


def last_green(root: Path) -> str:
    green = _tags(root, "green")
    return green[-1] if green else "night/base"


def restore(root: Path, tag: str) -> str:
    """Roll the tree back to `tag`, keeping the work it throws away: committed under night/red-N first (nothing is lost)."""
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", "night: red state kept before a roll back")
    red = _tags(root, "red")
    keep = f"night/red-{(int(red[-1].rsplit('-', 1)[1]) + 1) if red else 1}"
    git(root, "tag", keep)
    git(root, "reset", "-q", "--hard", tag)
    return keep


def head_summary(root: Path, n: int = 40) -> str:
    return git(root, "log", "--oneline", f"-{n}", "--decorate=short").stdout.strip()


def snapshot(root: Path, label: str) -> str:
    """Commit the working tree as it is (no tag) and return the commit id: a candidate of a tournament, or work to look at later."""
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", f"night: {label}")
    return git(root, "rev-parse", "HEAD").stdout.strip()


def reset_to(root: Path, ref: str) -> None:
    git(root, "reset", "-q", "--hard", ref)


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-").lower() or "prompt"


def mark_done(root: Path, name: str) -> str:
    """Tag the current state as the end of prompt `name` (night/done-<name>): the report's per-prompt diff stats and the morning review use these."""
    tag = f"night/done-{slug(name)}"
    git(root, "tag", "-f", tag)
    return tag


def numstat(root: Path, a: str, b: str = "HEAD") -> tuple[int, int, int]:
    """(files, added, removed) between two refs."""
    out = git(root, "diff", "--numstat", a, b).stdout.splitlines()
    files = added = removed = 0
    for ln in out:
        x, y, *_ = ln.split("\t") + ["", ""]
        files += 1
        added += int(x) if x.isdigit() else 0
        removed += int(y) if y.isdigit() else 0
    return files, added, removed


def tags(root: Path, prefix: str) -> list[str]:
    return git(root, "tag", "-l", f"{prefix}*").stdout.split()


def undo(root: Path) -> str:
    """Put the project back to exactly where it was before the night. Nothing is lost: the whole night stays under night/undone-N (a tag)."""
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", "night: state kept before an undo")
    n = len(tags(root, "night/undone-")) + 1
    keep = f"night/undone-{n}"
    git(root, "tag", keep)
    git(root, "reset", "-q", "--hard", "night/base")
    return keep


def squash(root: Path, message: str) -> str | None:
    """All of the night's commits become ONE commit on top of the baseline (the old tip stays under night/before-squash-N). None when there is nothing to squash."""
    if git(root, "rev-list", "--count", "night/base..HEAD").stdout.strip() in ("", "0"):
        return None
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", "night: final state before squash")
    n = len(tags(root, "night/before-squash-")) + 1
    git(root, "tag", f"night/before-squash-{n}")
    git(root, "reset", "-q", "--soft", "night/base")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "--short", "HEAD").stdout.strip()


def review_text(root: Path) -> str:
    stat = git(root, "diff", "--stat", "night/base", "HEAD").stdout.strip().splitlines()
    done = [t for t in tags(root, "night/done-")]
    red = tags(root, "night/red-")
    lines = ["What the night changed (against the baseline night/base):", *(stat[-25:] or ["  nothing"]), ""]
    if done:
        lines += ["Finished prompts: " + ", ".join(t.removeprefix("night/done-") for t in done)]
    if red:
        lines += [f"Work that was rolled back is kept under {len(red)} tag(s): " + ", ".join(red[-5:]) + "  (look at one with: git diff night/base <tag>)"]
    lines += ["", "Look at it:      git log --oneline night/base..HEAD     git diff night/base",
              "Keep it as one:  nightage night squash        (one commit on top of the baseline)",
              "Throw it away:   nightage night undo          (back to the baseline; the night stays under a tag)"]
    return "\n".join(lines)


def work_token(root: Path) -> str:
    """A fingerprint of everything the work tree changed since HEAD (diff + untracked files with their sizes): it only moves when the work moves."""
    import hashlib

    h = hashlib.sha1()
    git(root, "add", "-N", ".")
    h.update(git(root, "diff", "HEAD").stdout.encode("utf-8", "replace"))
    h.update(git(root, "ls-files", "--others", "--exclude-standard").stdout.encode("utf-8", "replace"))
    return h.hexdigest()
