"""Look vs. act: the one place that decides whether a step only LOOKED (read, search, a read-only shell command) or ACTED
(a write, a real command), and what follows from long stretches of either.

Seen live, each a real incident:
  * after a resume the 27B re-read the whole code base in 100-line pieces - 89 read_file calls, 13 ignored nudges, no edit
    in 9 minutes. After LOOK_BLOCK_AT look-steps in a row a look call is simply not run (zero tokens, deterministic);
    any write or run starts reading again. The streak is saved with the session (a restart is not a fresh allowance).
  * with reading paused the model switched to `run sed -n ...` (D2): a read-only shell command is a look, not an act.
  * with reading paused and edit_file needing the verbatim old text, the model could do neither for 15 minutes (D1):
    write tools are never paused, and replace_lines / insert_after answer a stale or unseen line number with the
    current lines - so a paused model always has a way to get the lines it needs by acting.
  * 66 and 97 steps of scripted test runs without one edit to the project: an edit-gap note (never a block).
These counters used to be scattered over the engine; they live here now, with no knowledge of the conversation:
`blocked()` before a call runs, `observe()` after it, which returns the reactions (event + message) for the engine."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

LOOK_TOOLS = frozenset({"read_file", "read_files", "search", "outline", "list_dir", "repo_map", "show_symbol", "where_used"})
LOOK_BLOCK_AT = 16                 # look-steps in a row after which reading is paused until the model changes something (was 25: streaks of 54 were measured)
FREE_STAGE_STEPS = 6               # the first steps of a stage whose reads do not count toward that streak (orientation)
READ_VOLUME_NUDGE = 700            # lines read in a row without writing or running anything
TEST_STREAK_NUDGE = 15             # steps in a row without editing a real project file (running and reading only)
WRITE_TOOLS = frozenset({"write_file", "append_file", "edit_file", "apply_patch", "delete_file", "move_file", "replace_lines", "insert_after"})
SHELL_TOOLS = frozenset({"run", "run_tests", "lint", "git_commit", "tui_smoke"})
NEUTRAL_ACTS = frozenset({"mark_done", "project_note", "add_check", "ask_user", "write_prose"})
_WROTE = ("wrote ", "edited ", "appended ", "patched ", "deleted ", "moved ")
SCRATCH_DIRS = frozenset({".scratch", "scratch", "tmp", ".tmp", "__pycache__", ".pytest_cache"})


def is_scratch_path(target: str) -> bool:
    """True when ANY path segment is a scratch folder (`tmp/x.py`, `src/.scratch/t.py`); `tmpfile.py`, `tmpl/x.py` and
    `scratchpad_notes.py` are real files. An absolute path is judged relative to the working directory (a repo that itself lives
    under /tmp or ~/scratch is not all scratch); outside it, it is never scratch."""
    raw = str(target or "").replace("\\", "/")
    if raw.startswith("/"):
        cwd = os.getcwd().replace("\\", "/").rstrip("/") + "/"
        if not raw.startswith(cwd):
            return False
        raw = raw[len(cwd):]
    return any(p in SCRATCH_DIRS for p in raw.split("/"))


# The ONE list of read-only shell commands. FILTER_CMDS are the ones that filter a pipe (tools.filter_split imports them);
# READONLY_CMDS adds the commands that only look at files. Some are read-only only without certain flags: see _segment_readonly.
FILTER_CMDS = frozenset({"sed", "head", "tail", "grep", "egrep", "fgrep", "rg", "awk", "cut", "sort", "uniq", "wc", "nl", "tr", "column", "fold", "cat", "less", "more"})
READONLY_CMDS = FILTER_CMDS | frozenset({"ls", "find", "tree", "file", "stat", "diff", "git", "jq", "xxd", "od", "strings", "base64", "fd", "realpath",
                                         "dirname", "basename", "echo", "printf", "python", "python3"})
_GIT_READ = frozenset({"show", "log", "diff", "status", "blame", "ls-files", "grep", "rev-parse", "branch"})
_FIND_ACTS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"})
_PY_BAD = re.compile(
    r"\.write|write_text|write_bytes|os\.(?:remove|unlink|rename|replace|system|popen|rmdir|exec\w*|spawn\w*|kill)|shutil|subprocess|rmtree|makedirs|"
    r"touch|mkdir|chmod|exec\s*\(|eval\s*\(|__import__|importlib|getattr\s*\(|compile\s*\(|ctypes|\.(?:unlink|rename|rmdir|symlink_to|hardlink_to)\s*\(|"
    r"(?<![\w.])(?:remove|unlink|rename|system|popen)\s*\(")


def _words(cmd: str) -> list[list[str]] | None:
    """Split a command into segments (at | || && ; & newline) of words, quotes respected and removed. None when it holds something that can
    write or run other code: a redirect (except `2>&1`, `>/dev/null`), a heredoc, `$(...)`, backticks, `<(...)`, an open quote."""
    segs: list[list[str]] = []
    words: list[str] = []
    cur, started, quote, i, n = "", False, "", 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if quote == "'":
            if ch == "'":
                quote = ""
            else:
                cur += ch
        elif quote == '"':
            if ch == '"':
                quote = ""
            elif ch == "\\" and i + 1 < n:
                i += 1
                cur += cmd[i]
            elif ch == "`" or cmd.startswith("$(", i):
                return None
            else:
                cur += ch
        elif ch == "\\" and i + 1 < n:
            i += 1
            cur += cmd[i]
            started = True
        elif ch in "'\"":
            quote, started = ch, True
        elif ch == "`" or cmd.startswith(("$(", "<<", "<(", ">("), i):
            return None
        elif ch == ">" or (ch == "&" and cmd.startswith("&>", i)):
            m = re.match(r"&?>>?\s*(?:&\d+|/dev/null(?![\w/.]))", cmd[i:])
            if not m or cur not in ("", "1", "2"):
                return None
            cur, started, i = "", False, i + m.end() - 1
        elif ch in "|;&\n" or ch.isspace():
            if started:
                words.append(cur)
            cur, started = "", False
            if ch in "|;&\n" and words:
                segs.append(words)
                words = []
        else:
            cur += ch
            started = True
        i += 1
    if quote:
        return None
    if started:
        words.append(cur)
    if words:
        segs.append(words)
    return segs


def _sed_readonly(args: list[str]) -> bool:
    scripts: list[str] = []
    rest = list(args)
    plain: list[str] = []
    while rest:
        a = rest.pop(0)
        if a in ("--in-place", "--file") or a.startswith(("--in-place=", "--file=")) or re.fullmatch(r"-[A-Za-z]*[if][A-Za-z.]*", a):
            return False                      # -i / -ni / -i.bak / -f script.sed (the script is a file we cannot see)
        if a == "--expression" or a == "-e" or re.fullmatch(r"-[A-Za-z]*e", a):
            if rest:
                scripts.append(rest.pop(0))
        elif a.startswith("--expression="):
            scripts.append(a.split("=", 1)[1])
        elif not a.startswith("-") or a == "-":
            plain.append(a)
    if not scripts and plain:
        scripts.append(plain[0])
    for sc in scripts:
        for m in re.finditer(r"s(.)((?:\\.|(?!\1).)*)\1((?:\\.|(?!\1).)*)\1([a-zA-Z0-9]*)", sc):
            if re.search(r"[weW]", m.group(4)):
                return False                  # s///w file, s///e
        sc = re.sub(r"[sy](.)(?:\\.|(?!\1).)*\1(?:\\.|(?!\1).)*\1[a-zA-Z0-9]*", "s", sc)
        sc = re.sub(r"/(?:[^/\\]|\\.)*/", "/", sc)
        if re.search(r"(?:^|[;{}\s!,/\d$])[wWe]", sc):
            return False                      # w / W (write a file), e (run a command)
    return True


def _awk_readonly(args: list[str]) -> bool:
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-f", "--file") or a.startswith("--file="):
            return False
        if a in ("-F", "-v"):
            i += 2
            continue
        if a.startswith("-") and a != "-":
            i += 1
            continue
        return not re.search(r"system\s*\(|>|\|", a)
    return True


def _python_readonly(args: list[str]) -> bool:
    """`python -c CODE` whose code cannot write, delete, run commands or evaluate strings; everything else with python acts."""
    if "-c" not in args or args.index("-c") + 1 >= len(args):
        return False
    code = args[args.index("-c") + 1]
    if _PY_BAD.search(code):
        return False
    for m in re.finditer(r"\bopen\s*\(([^)]*)\)", code):
        parts = [x.strip() for x in m.group(1).split(",")]
        mode_kw = re.search(r"\bmode\s*=\s*(['\"])([^'\"]*)\1", m.group(1))
        method = m.start() > 0 and code[m.start() - 1] == "."          # Path.open(mode): the mode is the FIRST argument
        pos = parts[0] if method else parts[1] if len(parts) > 1 else ""
        if mode_kw:
            mode = mode_kw.group(2)
        elif pos and not re.match(r"^\w+\s*=", pos):
            lit = re.fullmatch(r"(['\"])([^'\"]*)\1", pos)
            if not lit:
                return False                  # a mode we cannot read
            mode = lit.group(2)
        else:
            continue
        if re.search(r"[wax+]", mode):
            return False
    return True


def _git_readonly(args: list[str]) -> bool:
    i, sub = 0, None
    while i < len(args):
        if args[i] in ("-C", "-c"):
            i += 2
            continue
        if args[i].startswith("-"):
            i += 1
            continue
        sub = args[i]
        break
    if sub not in _GIT_READ:
        return False
    rest = args[i + 1:]
    if any(a.startswith("--output") for a in rest):
        return False
    if sub == "branch":
        listing = any(a in ("-l", "--list") for a in rest)
        for a in rest:
            if a.startswith("--"):
                if a in ("--delete", "--move", "--copy", "--force", "--track", "--no-track", "--edit-description", "--create-reflog") or a.startswith(("--set-upstream", "--unset-upstream")):
                    return False
            elif a.startswith("-"):
                if re.search(r"[dDmMcCfutT]", a[1:]):
                    return False
            elif not listing:
                return False                  # `git branch name` creates a branch
    return True


def _segment_readonly(words: list[str]) -> bool:
    cmd, args = words[0], words[1:]
    if cmd not in READONLY_CMDS:
        return False
    if cmd == "find":
        return not any(a in _FIND_ACTS for a in args)
    if cmd == "sed":
        return _sed_readonly(args)
    if cmd == "awk":
        return _awk_readonly(args)
    if cmd == "sort":
        return not any(re.fullmatch(r"-[A-Za-z]*o.*", a) and not a.startswith("--") or a.startswith(("--output", "--compress-program")) for a in args)
    if cmd == "uniq":
        return len([a for a in args if not a.startswith("-")]) <= 1       # `uniq in out` writes out
    if cmd == "git":
        return _git_readonly(args)
    if cmd in ("python", "python3"):
        return _python_readonly(args)
    if cmd == "base64":
        return not any(a.startswith(("-o", "--output")) for a in args)
    if cmd == "tree":
        return not any(a == "-o" for a in args)
    if cmd == "xxd":
        return "-r" not in args
    if cmd == "rg":
        return not any(a.startswith("--pre") for a in args)
    if cmd == "fd":
        return not any(a in ("-x", "-X") or a.startswith(("--exec")) for a in args)
    return True


def readonly_shell(command: Any) -> bool:
    """True for a shell command that only LOOKS at files (`sed -n '60,252p' f.py`, `cat f | head`, `git log`, `python3 -c "print(open(f).read())"`).
    Seen live: when read_file was paused after 25 read steps in a row, the model switched to `run sed -n ...`, and a run counted as an
    action, so it could read 25 more times. A pipeline is read-only only when EVERY segment is; anything that can write, delete or run
    other code (redirects, tee, xargs, `$(...)`, `find -exec`, `sed -i`, `git commit` ...) is an act."""
    cmd = str(command or "").strip()
    segs = _words(cmd) if cmd else None
    return bool(segs) and all(_segment_readonly(w) for w in segs)


def classify(name: str, args: dict[str, Any] | None, result: str | None = None) -> str:
    """'look', 'act' or 'neutral'. A write tool that changed nothing and answered with the current lines (replace_lines'
    NOT CHANGED, NOT APPLIED, ALREADY APPLIED) looked; a read-only shell command looked."""
    args = args or {}
    if name in LOOK_TOOLS or (name == "run" and readonly_shell(args.get("command"))):
        return "look"
    if name in WRITE_TOOLS:
        if "is protected (the player's own data)" in str(result or "")[:200]:
            return "neutral"                   # refused: coding.protect (the player's save file) - nothing was changed, nothing looked at
        return "look" if result is not None and str(result).startswith(("NOT CHANGED", "NOT APPLIED", "ALREADY APPLIED")) else "act"
    if name in SHELL_TOOLS:
        return "act"
    if name in NEUTRAL_ACTS:
        return "neutral"                   # bookkeeping: neither a change nor a look (a rejected mark_done is the same)
    return "look" if name not in ("ask_user", "write_prose", "web_search", "web_fetch") else "neutral"


_RUN_NOTE = re.compile(r"^NOTE: this command changed (.+?)(?: and \d+ more)? \(recorded\)\.$", re.M)


def run_changed(result: str) -> list[str]:
    """The (listed) files a `run` result says the command changed (tools.run appends that NOTE line), else []."""
    m = _RUN_NOTE.search(str(result))
    return [x.strip() for x in m.group(1).split(", ")] if m else []


def real_edit(name: str, args: dict[str, Any] | None, result: str) -> bool:
    """A successful write to a project file (not a scratch copy), also one a shell command made."""
    if name == "run":
        return any(not is_scratch_path(f) for f in run_changed(result))
    if name not in WRITE_TOOLS or not str(result).startswith(_WROTE):
        return False
    target = str((args or {}).get("path") or (args or {}).get("dst") or (args or {}).get("dest") or "")
    return not is_scratch_path(target)


@dataclass
class Reaction:
    event: str
    data: dict[str, Any]
    text: str
    think: bool = False


@dataclass
class LookAct:
    look_streak: int = 0
    wander_nudges: int = 0
    edit_gap: int = 0
    edit_nudges: int = 0
    free_steps: int = 0                    # steps of the current stage whose reads are still free (begin_stage)
    probing: bool = False                  # a command ran since the last real edit (reads count toward the edit gap from then on)
    read_lines: Any = field(default=None, repr=False)     # () -> lines read since the last act (tools.read_lines)
    reset_lines: Any = field(default=None, repr=False)
    limit_fn: Any = field(default=None, repr=False)       # () -> the pause threshold (config coding.look_block_at); None = LOOK_BLOCK_AT
    chunk_fn: Any = field(default=None, repr=False)       # () -> the lower threshold of an editing-by-passes stage (coding.chunk_look_block) or None
    hint_fn: Any = field(default=None, repr=False)        # () -> 'Rewrite lines 201-280 of story.py now ...' (the next chunk) or ''

    @property
    def block_at(self) -> int:
        try:
            base = int(self.limit_fn()) if self.limit_fn else LOOK_BLOCK_AT
        except Exception:  # noqa: BLE001 - a broken config value must not break the guard
            base = LOOK_BLOCK_AT
        try:
            chunk = self.chunk_fn() if self.chunk_fn else None
            return min(base, int(chunk)) if chunk is not None else base
        except Exception:  # noqa: BLE001
            return base

    def begin_stage(self) -> None:
        """A fresh stage context: its first FREE_STAGE_STEPS steps may read without counting toward the pause."""
        self.free_steps = FREE_STAGE_STEPS

    def state(self) -> dict[str, Any]:
        """The counters as plain JSON (saved with the session)."""
        return {"look_streak": self.look_streak, "wander_nudges": self.wander_nudges, "edit_gap": self.edit_gap,
                "edit_nudges": self.edit_nudges, "probing": self.probing}

    def load(self, d: Any) -> None:
        d = d if isinstance(d, dict) else {}
        for key in ("look_streak", "wander_nudges", "edit_gap", "edit_nudges"):
            try:
                setattr(self, key, max(int(d.get(key) or 0), 0))
            except (TypeError, ValueError):
                setattr(self, key, 0)
        self.probing = d.get("probing") is True

    def _lines(self) -> int:
        try:
            return int(self.read_lines()) if self.read_lines else 0
        except Exception:  # noqa: BLE001
            return 0

    def pause(self) -> None:
        """Start with reading paused (the supervisor's second strategy after a re-read loop)."""
        self.look_streak = max(self.look_streak, self.block_at)

    def blocked(self, name: str, args: dict[str, Any] | None) -> str | None:
        """The NOT RUN answer for a look call while reading is paused, else None. Write tools are never paused."""
        if self.look_streak < self.block_at or not (name in LOOK_TOOLS or (name == "run" and readonly_shell((args or {}).get("command")))):
            return None
        try:
            hint = (self.hint_fn() if self.hint_fn else "") or ""
        except Exception:  # noqa: BLE001
            hint = ""
        return (f"NOT RUN: {self.look_streak} read/search steps in a row without a change - reading is paused. What you read is in your context "
                "and the outlines above. " + (hint + " " if hint else "") + "Make the next change now (replace_lines / insert_after need no old text and show you the current "
                "lines if your numbers are off; or edit_file / write_file) or run a command, then you can read again.")

    def observe(self, name: str, args: dict[str, Any] | None, result: str) -> list[Reaction]:
        out: list[Reaction] = []
        kind = classify(name, args, result)
        if self.free_steps > 0:
            self.free_steps -= 1
            if kind == "look":
                return out                                # an orientation read of a new stage: not counted, no note
        w = self._wander(kind)
        if w:
            out.append(w)
        g = self._edit_gap(name, args, result, kind)
        if g:
            out.append(g)
        return out

    def _wander(self, kind: str) -> Reaction | None:
        """Only looking for 10 steps or ~700 lines without writing or running anything: say so, every 10 steps (after a resume or a
        compaction the model re-reads the whole code base - seen live, 60 read_file calls in a row, two nudges ignored); one merged
        message (the step and the line nudge used to fire back to back). No forced reasoning step: measured live, it took 4.7 min."""
        if kind == "act":
            self.look_streak = 0
            self.wander_nudges = 0
            if self.reset_lines:
                self.reset_lines()
            return None
        if kind == "neutral":
            return None
        self.look_streak += 1
        n = self.wander_nudges + 1
        lines = self._lines()
        if self.look_streak < 10 * n and lines < READ_VOLUME_NUDGE * n:
            return None
        self.wander_nudges = n
        if n <= 2:
            text = (f"You have spent {self.look_streak} steps reading (about {lines} lines) without writing or "
                    "running anything. That is enough to start: write the first piece now (a new module or a first edit) or run something, "
                    "and read more only for the exact lines you need next.")
        else:
            text = (f"{self.look_streak} read/search steps in a row (about {lines} lines), nothing written. Stop re-reading: what you read "
                    "earlier is in the work log and the outlines above, and every extra read makes each following step slower. Pick the ONE next "
                    "change the task needs, use `show_symbol` for the exact lines, and make the edit in your next call. Write big new "
                    "content in parts of at most ~250 lines (write_file, then append_file).")
        return Reaction("wander", {"steps": self.look_streak, "lines": lines}, text)

    def _edit_gap(self, name: str, args: dict[str, Any] | None, result: str, kind: str) -> Reaction | None:
        """After 15 steps without a successful edit of a real file (scratch folders do not count) once a command ran: one note and
        one reasoning step; once more at 30. Never blocks. Reading before the first run is exploring - the wander note's business."""
        if real_edit(name, args, result):
            self.edit_gap = 0
            self.edit_nudges = 0
            self.probing = False
            return None
        if name in SHELL_TOOLS and kind == "act":
            self.probing = True
        elif not self.probing:
            return None
        self.edit_gap += 1
        n = self.edit_nudges + 1
        if n > 2 or self.edit_gap < TEST_STREAK_NUDGE * n:
            return None
        self.edit_nudges = n
        return Reaction("edit_gap", {"steps": self.edit_gap},
                        f"{self.edit_gap} steps in a row without editing any project file - you are testing and probing, not changing "
                        "anything. Decide now: what is the smallest edit that moves the task forward? Make it, then test once. If a test run "
                        "shows something you do not understand, read the code that produces it instead of re-running with other inputs.",
                        think=True)
