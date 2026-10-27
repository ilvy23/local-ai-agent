"""Spill instead of truncate: a big tool result is kept WHOLE in `.scratch/spill/<tool>-<n>.txt`; the model gets a head+tail preview and one
precise line saying where the rest is and which range to read next.

Seen live: a command re-run with another `sed -n` range to see the middle of its output (22 times); a cut-off result is lost for good, so the
model repeats the call that produced it. With the whole result on disk the middle costs one `read_file` of a few hundred lines, not a re-run.

  * `Spill.clip(tool, text)`: the generic path (engine.execute): text up to SPILL_MIN chars comes back unchanged, bigger text becomes
    head lines + a gap line + tail lines + the locator line. When the file cannot be written the ORIGINAL text is returned (nothing is lost).
  * `Spill.note(tool, text, start)`: only the locator line, for callers that built their own preview (`run`, `search`).
  * Files: at most KEEP, newest kept (pruned on every save and at every stage start); the same text is stored once (a repeated identical
    command does not create a new file); long lines are wrapped at WRAP so line ranges always make sense; `.scratch/` is kept out of git.
  * Both harness lines are in context.MARKERS ("spill_gap", "spill_locator"): a write that contains one is refused, like every other note
    the model could copy into a file.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from agent.codeagent import edits

SPILL_MIN = 6000          # chars: bigger results are spilled
HEAD, TAIL = 1500, 3000   # chars of the preview (whole lines): the start, and the end where failures and summaries live
KEEP = 30                 # files kept in the spill folder
WRAP = 400                # a longer line is broken here in the stored file (a one-line 50 KB blob stays readable by line range)
WINDOW = 9000             # chars of the range the locator line suggests (a read_file range stays well below tools.READ_CAP)
DIR = ".scratch/spill"
LINE_PREFIX = "full output: " + DIR + "/"
_NAME = re.compile(r"^([\w.-]+?)-(\d+)\.txt$")


def _wrap(text: str) -> list[str]:
    lines: list[str] = []
    for ln in text.splitlines():
        if len(ln) <= WRAP:
            lines.append(ln)
        else:
            lines.extend(ln[i:i + WRAP] for i in range(0, len(ln), WRAP))
    return lines


def _slug(tool: str) -> str:
    return re.sub(r"[^\w.]+", "_", tool).strip("_") or "tool"


class Spill:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.dir = self.root / DIR
        self.last: dict[str, str] = {}                      # tool -> repo-relative path of its newest spill file
        self._stored: dict[tuple[str, str], str] = {}       # (tool, text hash) -> path: the same text is stored once
        self._counter: dict[str, int] = {}

    # ── files ───────────────────────────────────────────────────────────
    def _number(self, slug: str) -> int:
        if slug not in self._counter:
            top = 0
            try:
                for f in self.dir.iterdir():
                    m = _NAME.match(f.name)
                    if m and m.group(1) == slug:
                        top = max(top, int(m.group(2)))
            except OSError:
                pass
            self._counter[slug] = top
        self._counter[slug] += 1
        return self._counter[slug]

    def save(self, tool: str, text: str) -> tuple[str, list[str]] | None:
        """(repo-relative path, stored lines) of the whole text, or None when it could not be written."""
        lines = _wrap(text)
        body = "\n".join(lines) + "\n"
        key = (_slug(tool), hashlib.sha1(body.encode("utf-8", "replace")).hexdigest())
        try:
            known = self._stored.get(key)
            if known and (self.root / known).is_file():
                os.utime(self.root / known)                 # still the newest: pruning keeps it
                self.last[tool] = known
                return known, lines
            edits.ensure_scratch_excluded(self.root)
            self.dir.mkdir(parents=True, exist_ok=True)
            rel = f"{DIR}/{key[0]}-{self._number(key[0])}.txt"
            (self.root / rel).write_text(body, encoding="utf-8", errors="replace")
        except OSError:
            return None
        self._stored[key] = rel
        self.last[tool] = rel
        self.prune()
        return rel, lines

    def prune(self, keep: int = KEEP) -> int:
        """Delete all but the newest `keep` files of the spill folder. Returns how many were removed."""
        try:
            files = [f for f in self.dir.iterdir() if _NAME.match(f.name)]
        except OSError:
            return 0

        def age(f: Path) -> tuple[int, int]:
            try:
                return f.stat().st_mtime_ns, int(_NAME.match(f.name).group(2))     # type: ignore[union-attr]
            except OSError:
                return 0, 0

        files.sort(key=age)
        gone = 0
        for f in files[:max(0, len(files) - keep)]:
            try:
                f.unlink()
                gone += 1
            except OSError:
                continue
        return gone

    # ── what the model reads ────────────────────────────────────────────
    def locator(self, rel: str, lines: list[str], start: int = 1, what: str = "") -> str:
        """The one line: file, size, and the exact range to read next (a window of about WINDOW chars from `start`)."""
        n = len(lines)
        a = max(1, min(start, max(n, 1)))
        end, used = a, 0
        while end <= n and (end == a or used + len(lines[end - 1]) + 1 <= WINDOW):
            used += len(lines[end - 1]) + 1
            end += 1
        b = max(a, end - 1)
        return f"full output: {rel} ({n} lines{what}) - read it with read_file path={rel} start={a} end={b}; do not re-run it to see more"

    def note(self, tool: str, text: str, start: int = 1, what: str = "") -> str:
        """The locator line alone (with a leading newline), '' when the text could not be saved."""
        saved = self.save(tool, text)
        return "" if saved is None else "\n" + self.locator(saved[0], saved[1], start, what)

    def clip(self, tool: str, text: str, head: int = HEAD, tail: int = TAIL, minimum: int = SPILL_MIN) -> str:
        if len(text) <= minimum:
            return text
        saved = self.save(tool, text)
        if saved is None:
            return text                                     # no place to put it: keep the original inline
        rel, lines = saved
        n = len(lines)
        h = used = 0
        while h < n and (h == 0 or used + len(lines[h]) + 1 <= head):
            used += len(lines[h]) + 1
            h += 1
        t = used = 0
        while t < n - h and (t == 0 or used + len(lines[n - 1 - t]) + 1 <= tail):
            used += len(lines[n - 1 - t]) + 1
            t += 1
        if h + t >= n:                                      # nothing left out (cannot happen above SPILL_MIN with these sizes)
            return text
        gap = f"({n - h - t} of {n} lines not shown: lines {h + 1}-{n - t})"      # harness marker (context.MARKERS "spill_gap")
        shown = lines[:h] + [gap] + lines[n - t:]
        return "\n".join(shown) + "\n" + self.locator(rel, lines, h + 1)      # harness marker (context.MARKERS "spill_locator")
