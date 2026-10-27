"""The coding agent's tools, bound to one repo folder.

Design rules, each from a measured failure of the earlier agent:
  * Edits are small and exact: `edit_file` (tolerant search/replace, several edits atomically) and `apply_patch`.
    A whole-file `write_file` over an existing large file is refused, and one write is capped, because a 23k-character
    payload takes ~3 minutes to generate and a single wrong quote throws all of it away.
  * Every write runs a syntax gate and answers with the exact error line (or an outline of what was written), so the
    model never has to guess whether its change worked.
  * Reads are cheap: line ranges, outlines, a repo map, and a read cache ("unchanged since step N") instead of
    re-sending the same 20k characters.
  * `run` can feed stdin and has a timeout, so playtesting a text game needs no heredoc tricks. The command runs in its
    own process group with stdin closed (or fed), output goes to a temp file (capped), and on timeout, Ctrl-C or exit the
    whole group is killed - a child can never outlive its command or hang the harness on an open pipe.
  * Writes are atomic (temp file + rename), keep the file's line endings (CRLF stays CRLF) and its mode, and a
    read-only file is never written - its mode is the user's protection (a game's save file, for example).
All paths resolve inside the repo; the engine checks the policy (ask / medium / auto) before a handler runs.
"""

from __future__ import annotations

import contextlib
import difflib
import fnmatch
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from agent import changes, osport
from agent.codeagent import edits, outputs, profile as profile_mod, spill as spill_mod, static, symbols, tuihint, tuismoke, xref
from agent.codeagent.context import MARKERS, ContextManager, FileEntry, new_markers
from agent.codeagent.lookact import FILTER_CMDS, is_scratch_path, readonly_shell
from agent.tools.registry import Tool, ToolRegistry

MAX_WRITE_CHARS = 14000          # advice threshold only
HARD_WRITE_CAP = 120000
TINY_APPEND_CHARS = 300
OVERWRITE_GUARD_CHARS = 3000
READ_CAP = 14000                 # characters of an explicit line range (start/end) in one answer
READ_MAX_CHARS = 9000            # default of coding.read_max_chars: a read WITHOUT a range (~250 lines); measured 229 reads vs 32 edits of whole files
READ_MAX_LINES = 250
REMOVED_NAMES_CHECKED = 5        # removed-symbol note: at most this many removed names are looked up per write
OUT_HEAD, OUT_TAIL = 900, 3200
SEARCH_MAX = 40
MAX_OUTPUT_BYTES = 8_000_000     # a command printing more than this is stopped (the model only ever sees head and tail)
KEEP_OUTPUT = (100_000, 300_000)  # bytes of a long output kept for summarizing: head, tail
SLOW_RUN_S = 4.0                 # a run this slow that used under SLEEPY_CPU of its time on the CPU is mostly sleeping
SLEEPY_CPU = 0.3
NO_SLEEP_DIR = Path(__file__).with_name("nosleep")      # holds a sitecustomize.py that turns time.sleep into a no-op
_FILTERS = FILTER_CMDS            # one list of read-only commands: lookact.py


_WRITES_RX = re.compile(r"<<|>|\btee\b|open\(|write_text|write_bytes|sed\s+-\w*i|\bcp\b|\bmv\b")     # a command that can write a file


def filter_split(command: str) -> int | None:
    """Index of the `|` that starts the trailing chain of pure output filters (`prog | sed -n '1,45p'`, `prog 2>&1 | grep x | head`),
    or None. Only the last pipeline of the command counts; quotes are respected; anything with substitutions or heredocs is
    left alone (None). Used to keep the unfiltered output of a slow program so it need not be re-run to see another part."""
    if any(tok in command for tok in ("$(", "`", "<<", "<(", ">(")):
        return None
    pipes: list[int] = []
    quote, i, n = "", 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"':
                i += 1
        elif ch == "\\":
            i += 1
        elif ch in "'\"":
            quote = ch
        elif ch == "|" and command[i + 1:i + 2] == "|" or ch == "&" and command[i + 1:i + 2] == "&":
            pipes, i = [], i + 2                          # `||` / `&&`: a new pipeline starts
            continue
        elif ch == "&" and (command[i - 1:i] in (">", "<") or command[i + 1:i + 2] == ">"):
            pass                                          # `2>&1`, `&>file`: a redirect, not a background `&`
        elif ch in ";\n&":
            pipes = []
        elif ch == "|":
            pipes.append(i)
            if command[i + 1:i + 2] == "&":               # `|&`
                i += 1
        i += 1
    if quote or not pipes:
        return None
    start = None
    for k in range(len(pipes) - 1, -1, -1):              # walk back over the trailing stages while they are filters
        stage = command[pipes[k] + 1:pipes[k + 1] if k + 1 < len(pipes) else n].lstrip("&").split()
        if not stage or stage[0] not in _FILTERS:
            break
        start = pipes[k]
    return start

READ, WRITE, SHELL, NET, OTHER = "read", "write", "shell", "net", "other"
KINDS = {
    "read_file": READ, "read_files": READ, "list_dir": READ, "search": READ, "repo_map": READ, "outline": READ,
    "git_status": READ, "git_diff": READ, "show_symbol": READ, "where_used": READ,
    "write_file": WRITE, "append_file": WRITE, "edit_file": WRITE, "apply_patch": WRITE, "delete_file": WRITE,
    "move_file": WRITE, "replace_lines": WRITE, "insert_after": WRITE,
    "run": SHELL, "run_tests": SHELL, "lint": SHELL, "git_commit": SHELL, "tui_smoke": SHELL,
    "web_search": NET, "web_fetch": NET,
}


SNAPSHOT_CAP = 20000          # more files than this: no snapshot around a `run` (the tree is too big to watch cheaply)
_SYMBOL_SUFFIXES = edits.CODE_SUFFIXES - {".md", ".json", ".toml", ".yaml", ".yml", ".html", ".css", ".sql"}
SNAPSHOT_SKIP = edits.SKIP_DIRS | {".scratch"}
RO_MAX_FILES, RO_MAX_BYTES = 50, 1_000_000       # read-only files whose bytes are kept around a `run`
# coding.protect: the player's own data (a game's save file). Independent of file modes: the human must keep write access.
DEFAULT_PROTECT = ("save.json", "*.sav", "savegame*", "*.save")
PROTECT_MAX_BYTES = 5_000_000                    # total size of the protected files whose bytes are kept around a `run`
PROTECT_MARK = "is protected (the player's own data)"
PROTECT_COPY = "Run the game or script on a copy in .scratch/ instead (copy the folder, run there)."


def protect_patterns(extra: Any = None) -> tuple[str, ...]:
    """The built-in patterns plus the user's `coding.protect` (extend-only: a built-in cannot be removed, garbage is ignored)."""
    more = [x.strip() for x in extra if isinstance(x, str) and x.strip()] if isinstance(extra, (list, tuple)) else []
    return tuple(dict.fromkeys([*DEFAULT_PROTECT, *more]))


def is_protected_rel(rel: str, patterns: tuple[str, ...]) -> bool:
    """Does the repo-relative path match a protect pattern (the whole relative path or the basename)? Anything below a
    `.scratch/` folder never does: scratch copies must stay writable, that is where the model experiments."""
    rel = str(rel).replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    parts = rel.split("/")
    if rel.startswith("/") or ".." in parts or ".scratch" in parts:
        return False
    return any(fnmatch.fnmatchcase(rel, pat) or fnmatch.fnmatchcase(parts[-1], pat) for pat in patterns)
SHIFT_WINDOW = 80                                # replace_lines: a stale stamp is looked for this many lines up and down (symbols.shifted_starts)
EDIT_APPLIED_MIN_CHARS = 30                      # edit_file: a replacement shorter than this is never taken for 'already applied'
_REJECT_SUFFIXES = {".py", ".pyi", ".json", ".toml", ".yaml", ".yml"}   # coding.syntax_reject: the types the stdlib parses (edits.check_syntax)
MAX_OWNED = 500                                  # files remembered as written / created by the agent (what its own `rm` may delete)
MAX_RUN_TEXT, MAX_RUN_RECORDS = 2_000_000, 200   # text files a command changed: the biggest one recorded, the most files recorded


def snapshot_tree(root: Path, keep_ro: bool = False, protect: tuple[str, ...] | None = None) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[bytes, int, int]]] | None:
    """({path: (mtime_ns, size)} of the regular files below root without build / cache / scratch folders, {path: (bytes, mode,
    mtime_ns)} of the read-only ones when keep_ro and of those matching the `protect` patterns, whatever their mode, up to
    PROTECT_MAX_BYTES together), or None above SNAPSHOT_CAP files."""
    files: dict[str, tuple[int, int]] = {}
    kept = 0
    ro: dict[str, tuple[bytes, int, int]] = {}
    stack = [str(root)]
    while stack:
        try:
            it = os.scandir(stack.pop())
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name not in SNAPSHOT_SKIP:
                            stack.append(e.path)
                        continue
                    if not e.is_file(follow_symlinks=False):
                        continue
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                files[e.path] = (st.st_mtime_ns, st.st_size)
                if len(files) > SNAPSHOT_CAP:
                    return None
                if keep_ro and not st.st_mode & stat.S_IWUSR and len(ro) < RO_MAX_FILES and st.st_size <= RO_MAX_BYTES:
                    try:
                        ro[e.path] = (Path(e.path).read_bytes(), stat.S_IMODE(st.st_mode), st.st_mtime_ns)
                    except OSError:
                        pass
                if protect and e.path not in ro and st.st_size + kept <= PROTECT_MAX_BYTES and is_protected_rel(os.path.relpath(e.path, root), protect):
                    try:
                        ro[e.path] = (Path(e.path).read_bytes(), stat.S_IMODE(st.st_mode), st.st_mtime_ns)
                        kept += st.st_size
                    except OSError:
                        pass
    return files, ro


def _uses_crlf(text: str) -> bool:
    return "\r\n" in text and text.count("\r\n") * 2 >= text.count("\n")


def atomic_write(p: Path, text: str) -> None:
    """Write via a temp file in the same folder + rename: a Ctrl-C, a full disk or a crash leaves the old file or the new
    one, never half of it. Keeps the old file's mode."""
    mode = stat.S_IMODE(p.stat().st_mode) if p.exists() else None
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _kill_group(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        osport.kill_group(proc.pid, getattr(signal, "SIGKILL", None))
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def _child_cpu() -> float:
    """CPU seconds (user + system) of all finished child processes so far."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime
    except (ImportError, OSError):                  # no resource module (Windows): never call a run sleepy
        return float("inf")


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _feed(pipe: Any, data: bytes) -> None:
    with contextlib.suppress(OSError):              # the program may exit (or close stdin) before reading it all
        pipe.write(data)
    with contextlib.suppress(OSError):
        pipe.close()


def _read_capped(f: Any) -> str:
    size = f.seek(0, os.SEEK_END)
    f.seek(0)
    head, tail = KEEP_OUTPUT
    if size <= head + tail:
        return f.read().decode("utf-8", errors="replace")
    first = f.read(head)
    f.seek(size - tail)
    # harness marker (context.MARKERS "bytes_omitted")
    return (first.decode("utf-8", errors="replace") + f"\n[… {size - head - tail} bytes of output omitted …]\n"
            + f.read().decode("utf-8", errors="replace"))


PLACEHOLDER_ERROR = ("error: that content is the placeholder the history shows in place of text you already wrote ('[N lines, M chars - now on disk]'), "
                     "not real file text. What you wrote earlier IS in the file - call read_file to see it, and send only the real new lines to add.")


def has_placeholder(content: Any) -> bool:
    """Seen live: the model copied the history marker as the content of append_file 94 times; with 'always accept' it ended up in the file.
    Never stripped (text around it would be lost silently): content with the placeholder ANYWHERE is refused."""
    return isinstance(content, str) and dict(MARKERS)["on_disk"].search(content) is not None


def cap_output(text: str, head: int = OUT_HEAD, tail: int = OUT_TAIL) -> str:
    text = text or ""
    if len(text) <= head + tail + 200:
        return text
    # harness marker (context.MARKERS "chars_omitted")
    return f"{text[:head]}\n[… {len(text) - head - tail} chars omitted …]\n{text[-tail:]}"


class CodeTools:
    def __init__(self, root: Path, profile: profile_mod.Profile, ctx: ContextManager, console: Any | None = None,
                 web_tools: list[Tool] | None = None, plan_tool: Tool | None = None, note_tool: Tool | None = None,
                 ask: Callable[[str], str] | None = None, run_timeout: int = 60,
                 check_cmd: str | None = None, read_max_chars: int | None = None, protect: Any = None, syntax_reject: bool = True,
                 smoke_timeout: int = tuismoke.DEFAULT_TIMEOUT, smoke_max_chars: int = tuismoke.MAX_CHARS) -> None:
        self.root = Path(root).resolve()
        self.protect = protect_patterns(protect)         # coding.protect (+ the built-in patterns): the player's data, see _protected / run
        self.read_max_chars = max(500, int(read_max_chars)) if read_max_chars else READ_MAX_CHARS
        self.test_override = profile.test_cmd if getattr(profile, "_override", False) else None
        self.profile = profile
        self.ctx = ctx
        self.console = console
        self.web_tools = web_tools or []
        self.plan_tool, self.note_tool = plan_tool, note_tool
        self.ask = ask
        self.run_timeout = run_timeout
        self.smoke_timeout = smoke_timeout                # coding.tui_smoke_timeout: seconds of one tui_smoke call when it names none
        self.smoke_max_chars = smoke_max_chars            # coding.tui_smoke_max_chars: size of its answer
        self.smoke_allow: Callable[[str], bool] | None = None   # the engine sets it: would the repo policy run this command without asking?
        self._tui_told: set[str] = set()                  # programs (and "fake") that already got the long tui_smoke hint in this session (tuihint.py)
        self.check_cmd = check_cmd or None             # coding.check_cmd: project-specific syntax check, {file} placeholder
        self.syntax_reject = bool(syntax_reject)       # coding.syntax_reject: replace_lines / edit_file / insert_after never turn a valid file into a broken one
        self.next_hint: Callable[[str], int | None] | None = None   # the engine sets it: next unchanged line of the stage window of a file (-1 = window done, None = unknown)
        self.spill = spill_mod.Spill(self.root)           # big results kept whole in .scratch/spill/ (spill.py)
        self.changes = changes.FileChangeLog()
        self.dirty: set[str] = set()                 # edited since the last passing check
        self.syntax_bad: dict[str, str] = {}         # path -> latest syntax error
        self.owned: dict[str, str] = {}              # rel path -> "write" (the model wrote it) / "run" (a command created it): what `rm` may delete
        self.static_bad: dict[str, list[str]] = {}   # path -> undefined names / duplicate defs / bad project imports (static.py)
        self._static_sig: dict[str, set[tuple[str, str]]] = {}   # path -> signatures (static.signatures) of its problems when last checked: the baseline of the next note
        self._pre_edit: dict[str, str | None] = {}   # path -> text before the write in progress (set by _bookkeep, used by _static_note once)
        # (path, start, end) of a request -> (mtime_ns, size, step, inode) of a COMPLETE answer that is still in the context; cleared with it
        self.read_cache: dict[tuple[str, int, int], tuple[int, int, int, int]] = {}
        self.ctx.on_clear_cache = self.read_cache.clear
        self.evidence: list[tuple[int, str, int | None]] = []     # (step, command, exit) since the last edit
        self.last_edit_step = 0
        self.read_lines = 0                             # lines returned by reads since the last write/run (volume nudge)
        self.read_counts: dict[str, int] = {}           # read_file calls per relative path, whole run (stages.py: which files a stage keeps re-reading)
        self.last_cpu = 0.0                             # CPU seconds of the last command (sleep detection)
        self._tiny_appends: dict[str, int] = {}         # path -> consecutive tiny appends
        self._write_log: list[tuple[int, str]] = []     # (step, path) of recent whole-file writes
        self._removed_note = ""                         # set by _bookkeep for a write that dropped a still-used symbol; _xref_note appends it once
        self.on_write: Any = None                       # (rel path, old text or None, new text) after every file-tool write (the stage cursor)
        self.on_read: Any = None                        # (rel path, first line, last line) of what a read really showed; (rel, None, None) for a placeholder (fileseen.py)
        self.line_views: dict[str, str] = {}            # path -> file hash when the model last saw its line numbers (replace_lines checks it)

    # ── paths ───────────────────────────────────────────────────────────
    def resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(str(path)))
        return (p if p.is_absolute() else self.root / p).resolve()

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def _read_text(self, p: Path) -> tuple[str | None, str]:
        if not p.exists():
            return None, f"error: {self.rel(p)} does not exist"
        if p.is_dir():
            return None, f"error: {self.rel(p)} is a directory (use list_dir)"
        raw = p.read_bytes()
        if b"\0" in raw[:4096]:
            return None, f"{self.rel(p)} is a binary file ({len(raw)} bytes)"
        return raw.decode("utf-8", errors="replace"), ""

    # ── reads ───────────────────────────────────────────────────────────
    def read_file(self, path: str, start: Any = None, end: Any = None, outline: bool = False, **_: Any) -> str:
        p = self.resolve(path)
        text, err = self._read_text(p)
        if text is None:
            return err
        self.read_counts[self.rel(p)] = self.read_counts.get(self.rel(p), 0) + 1
        crlf = _uses_crlf(text)
        if crlf:
            text = text.replace("\r\n", "\n")              # shown with plain newlines; edits keep the file's CRLF
        lines = text.splitlines()
        total = len(lines)
        try:
            s = int(start) if start not in (None, "") else None
            e = int(end) if end not in (None, "") else None
        except (TypeError, ValueError):
            s = e = None
        st = p.stat()
        key = (str(p), s or 0, e or 0)
        stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        hit = self.read_cache.get(key)
        if hit and (hit[0], hit[1], hit[3]) == stamp and not outline:
            self._saw(p, None, None)
            return (f"{self.rel(p)} is unchanged since step {hit[2]} ({total} lines) - its content is already in your context above. "
                    "Read a different line range, or search for what you need.")
        self.read_cache.pop(key, None)
        header = f"{self.rel(p)}  ({total} lines, {len(text)} chars{', CRLF line endings' if crlf else ''})"
        self.line_views[self.rel(p)] = symbols.file_hash(text)      # every answer below shows line numbers (range, outline) or the whole text
        if outline:
            body = "\n".join(edits.outline(str(p), text)) or "(no top-level definitions found)"
            return f"{header}\nOUTLINE (line: definition)\n{body}"
        # only a COMPLETE answer arms the placeholder: after a cut one the model never saw the rest, and "already in your context" would be false
        if s is None and e is None:
            if len(text) <= self.read_max_chars and total <= READ_MAX_LINES:
                self.read_cache[key] = (*stamp[:2], self.ctx.step, st.st_ino)
                self.read_lines += total
                self._saw(p, 1, total)
                return f"{header}\n{text}"
            last, used = 0, 0                                # a long file: the first lines up to the cap, then ONE line saying where to go on
            while last < total and last < READ_MAX_LINES and (last == 0 or used + len(lines[last]) + 1 <= self.read_max_chars):
                used += len(lines[last]) + 1
                last += 1
            chunk = "\n".join(lines[:last])[:self.read_max_chars]
            self.read_lines += last
            self._saw(p, 1, last)
            return f"{header}\n--- lines 1-{last} ---\n{chunk}\nlines 1-{last} of {total}; continue with start={last + 1}"
        s = max(1, s or 1)
        e = min(total, e or min(total, s + 199))
        chunk = "\n".join(lines[s - 1:e])
        shown = e
        if len(chunk) > READ_CAP:
            shown = s - 1 + chunk[:READ_CAP].count("\n")      # complete lines that fit
            chunk = chunk[:READ_CAP] + "\n[truncated - use a smaller range]"       # harness marker (context.MARKERS "truncated")
        else:
            self.read_cache[key] = (*stamp[:2], self.ctx.step, st.st_ino)
        self.read_lines += e - s + 1
        self._saw(p, s, shown)
        return f"{header}\n--- lines {s}-{e} ---\n{chunk}"

    def _forget_reads(self, p: Path) -> None:
        """Drop the cached answers of one file (a write: the text the model saw is not the text on disk any more)."""
        for key in [k for k in self.read_cache if k[0] == str(p)]:
            del self.read_cache[key]

    def _saw(self, p: Path, a: int | None, b: int | None) -> None:
        """Tell the stage record (fileseen.py) what a read showed; (None, None) = a placeholder answer, a read without new lines."""
        if self.on_read is not None:
            try:
                self.on_read(self.rel(p), a, b if b is not None and a is not None and b >= a else None)
            except Exception:  # noqa: BLE001 - the record is a bonus, never a failed read
                pass

    def read_files(self, paths: Any = None, **_: Any) -> str:
        if isinstance(paths, str):
            paths = [x for x in re.split(r"[,\n]", paths) if x.strip()]
        out = []
        for p in (paths or [])[:6]:
            out.append(self.read_file(str(p).strip()))
        return "\n\n=====\n\n".join(out) or "error: give paths=[...]"

    def outline(self, path: str, **_: Any) -> str:
        return self.read_file(path, outline=True)

    def list_dir(self, path: str = ".", depth: Any = 1, **_: Any) -> str:
        p = self.resolve(path)
        if not p.is_dir():
            return f"error: {self.rel(p)} is not a directory"
        try:
            depth = max(1, min(int(depth), 3))
        except (TypeError, ValueError):
            depth = 1
        out: list[str] = []

        def walk(d: Path, level: int):
            try:
                entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except OSError:
                return
            for x in entries:
                if x.name in edits.SKIP_DIRS or (x.name.startswith(".") and x.name not in (".github", ".gitignore", ".env.example")):
                    continue
                pad = "  " * (level - 1)
                if x.is_dir():
                    out.append(f"{pad}{x.name}/")
                    if level < depth:
                        walk(x, level + 1)
                else:
                    try:
                        size = x.stat().st_size
                    except OSError:
                        size = 0
                    out.append(f"{pad}{x.name}  ({size}b)" if size < 1000 else f"{pad}{x.name}  ({size // 1000 + 1}k)")
                if len(out) >= 300:
                    return
        walk(p, 1)
        return "\n".join(out) or "(empty)"

    def search(self, pattern: str, path: str = ".", glob: str | None = None, context: Any = 0, ignore_case: bool = False,
               **_: Any) -> str:
        base = self.resolve(path)
        ctxn = 0
        try:
            ctxn = max(0, min(int(context), 3))
        except (TypeError, ValueError):
            pass
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "-n", "--no-heading", "--color", "never", "-m", "8", "--max-columns", "200"]
            if ignore_case:
                cmd.append("-i")
            if glob:
                cmd += ["-g", glob]
            if ctxn:
                cmd += ["-C", str(ctxn)]
            cmd += ["-e", pattern, str(base)]
        else:
            cmd = ["grep", "-rnIE", "-m", "8", "--exclude-dir=.git", "--exclude-dir=node_modules", "--exclude-dir=__pycache__"]
            if ignore_case:
                cmd.append("-i")
            if glob:
                cmd.append(f"--include={glob}")
            if ctxn:
                cmd += ["-C", str(ctxn)]
            cmd += ["-e", pattern, str(base)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return "search timed out - narrow the path or pattern"
        lines = [ln.replace(str(self.root) + "/", "") for ln in r.stdout.splitlines()]
        if not lines:
            return f"no matches for {pattern!r}" + (" (bad regex?)" if r.returncode == 2 else "")
        self._saw_numbers(lines)
        shown, used = [], 0
        for ln in lines[:SEARCH_MAX]:                      # the first matches, within the spill threshold
            if shown and used + min(len(ln), 220) + 1 > spill_mod.SPILL_MIN:
                break
            shown.append(ln[:220])
            used += len(shown[-1]) + 1
        if len(shown) == len(lines):
            return "\n".join(shown)
        more = f"\n[… {len(lines) - len(shown)} more matches - narrow the search]"          # harness marker (MARKERS "more_matches")
        full = "\n".join(lines)
        return "\n".join(shown) + more + self.spill.note("search", full, start=len(shown) + 1)

    def repo_map(self, **_: Any) -> str:
        return edits.repo_map(self.root, 4500, focus=sorted(self.dirty))

    # ── symbols and line ranges (edits that need no verbatim old text) ─────
    def _saw_numbers(self, rows: list[str]) -> None:
        """`file:line: text` rows were shown: the model now knows line numbers of those files (replace_lines checks this)."""
        for rel in list(dict.fromkeys(r.split(":", 1)[0] for r in rows if ":" in r))[:10]:
            p = self.resolve(rel)
            if p.is_file() and self.root in p.parents:
                text, _e = self._read_text(p)
                if text is not None:
                    self.line_views[self.rel(p)] = symbols.file_hash(text.replace("\r\n", "\n"))

    def _plain(self, path: str) -> tuple[Path, str | None, str, bool, str]:
        """(path, raw text or None, text with plain newlines, crlf, error)."""
        p = self.resolve(path)
        raw, err = self._read_text(p)
        if raw is None:
            return p, None, "", False, err
        crlf = _uses_crlf(raw)
        return p, raw, raw.replace("\r\n", "\n") if crlf else raw, crlf, ""

    def show_symbol(self, name: str = "", path: str = "", **_: Any) -> str:
        if not str(name).strip():
            return "error: show_symbol needs name (a function, class, method as Class.method, or a constant/dict) and path."
        if not str(path).strip():
            hits = []
            for f in edits.iter_files(self.root, 600):
                if f.suffix.lower() in _SYMBOL_SUFFIXES:           # Python by AST; other languages by the definition-line regex
                    try:
                        found, _c = symbols.find_symbol(str(f), f.read_text(encoding="utf-8", errors="replace"), name)
                    except OSError:
                        continue
                    if found:
                        hits.append(self.rel(f))
            if len(hits) != 1:
                return (f"'{name}' is defined in {len(hits)} files ({', '.join(hits[:8])}) - give path=..." if hits
                        else f"no definition of '{name}' found in the repo's code files - check the name with search or where_used")
            path = hits[0]
        p, raw, text, _crlf, err = self._plain(path)
        if raw is None:
            return err
        span, cands = symbols.find_symbol(str(p), text, name)
        if span is None:
            if cands:
                return (f"'{name}' is ambiguous or not exact in {self.rel(p)}; candidates: "
                        + ", ".join(f"{c.name} (lines {c.start}-{c.end})" for c in cands[:12]) + ". Ask again with one of these names.")
            items = edits.outline(str(p), text, 40)
            return f"no symbol '{name}' in {self.rel(p)}. Its outline:\n" + ("\n".join(items) or "(none)")
        self.line_views[self.rel(p)] = symbols.file_hash(text)
        self.read_lines += span.end - span.start + 1
        return (symbols.show(self.rel(p), text, span) + "\n[change it with replace_lines(path, start, end, new_text, expect=\"<the #hash>\") - "
                "no need to quote the old text; the numbers are not part of the code]")

    def where_used(self, name: str = "", **_: Any) -> str:
        out = symbols.where_used(self.root, str(name))
        self._saw_numbers([ln for ln in out.splitlines()[1:] if re.match(r"[^:\s]+:\d+:", ln)])
        return out

    def _stale(self, rel: str, text: str, lines: list[str], s: int, e: int, expect: Any) -> str:
        """'' when the model's line numbers are current, else why not. With `expect` (the #hash it was shown) the range
        itself is compared; without, the file must be unchanged since it last saw line numbers of it."""
        if expect not in (None, ""):
            want = str(expect).strip().lstrip("#").lower()
            got = symbols.range_hash(lines[s - 1:e])
            return "" if want == got else f"lines {s}-{e} are not the lines you saw (#{want}; now #{got})"
        seen = self.line_views.get(rel)
        if seen is None:
            return "you have not seen line numbers of this file yet"
        return "" if seen == symbols.file_hash(text) else "the file changed since you last saw its line numbers (an edit moves the lines after it)"

    def _current_lines(self, rel: str, text: str, lines: list[str], s: int, e: int, why: str, tool: str) -> str:
        self.line_views[rel] = symbols.file_hash(text)               # it sees the current numbers now
        a, b = max(1, s - 3), min(len(lines), max(e, s) + 3)
        h = symbols.range_hash(lines[s - 1:e]) if 1 <= s <= e <= len(lines) else ""
        return (f"NOT CHANGED: {why}. {rel} lines {a}-{b} now ({len(lines)} lines in the file):\n{symbols.numbered(lines[a - 1:b], a)}\n"
                + (f"Lines {s}-{e} are #{h}. " if h else "") + f"Send {tool} again with the right numbers" + (f" and expect=\"{h}\"." if h else "."))

    @staticmethod
    def _new_lines(new_text: Any) -> list[str]:
        text = str(new_text or "")
        rows = text.split("\n")
        if rows and rows[-1] == "":
            rows = rows[:-1]
        body = [r for r in rows if r.strip()]
        if body and all(re.match(r"^\s*\d+\| ", r) for r in body):           # a numbered view copied as code: drop the numbers
            rows = [re.sub(r"^\s*\d+\| ?", "", r, count=1) for r in rows]
        return rows

    def _dup_note(self, before: str | None, after: str) -> str:
        """ONE factual line when the lines this edit wrote also stand verbatim elsewhere in the file (edits.duplicate_block; no model call).
        Seen live: a replace_lines with a too small end line left the old 141-line block behind the new one. Advice only, never a gate."""
        try:
            hit = edits.duplicate_block(before.replace("\r\n", "\n") if before else before, after.replace("\r\n", "\n"))
        except Exception:  # noqa: BLE001 - advice only; never break a write
            return ""
        return "\n" + edits.duplicate_text(*hit) if hit else ""

    def _applied(self, rel: str, text: str, lines: list[str], x: int, y: int) -> str:
        """The ONE answer to an edit whose new text already stands in the file (x-y). Nothing in it varies between repeats, so the loop guard
        counts a re-sent edit as a repeat; it is no error and it changes nothing."""
        self.line_views[rel] = symbols.file_hash(text)
        try:
            nxt = self.next_hint(rel) if self.next_hint is not None else None
        except Exception:  # noqa: BLE001 - the cursor is a hint, never a failed call
            nxt = None
        if nxt == -1:
            tail = "Your window of this file has no unchanged lines left."
        else:
            nxt = nxt if isinstance(nxt, int) and nxt > y else y + 1
            tail = (f"Next line of your window that is still unchanged: {nxt}." if nxt <= len(lines)
                    else f"The file ends at line {len(lines)}; there is nothing after it.")
        return f"ALREADY APPLIED: these lines already contain your new text (lines {x}-{y}). Nothing to do here. {tail}"

    def _keep_proposal(self, p: Path, content: str) -> str:
        """The rejected text goes to .scratch/unsaved/<file>.<n>.txt (never overwritten) so the work of generating it is not lost."""
        try:
            edits.ensure_scratch_excluded(self.root)
            d = self.root / ".scratch" / "unsaved"
            d.mkdir(parents=True, exist_ok=True)
            dest = next(d / f"{p.name}.{n}.txt" for n in range(1, 10_000) if not (d / f"{p.name}.{n}.txt").exists())
            atomic_write(dest, content)
            return f" Your proposal was kept in {self.rel(dest)}."
        except (OSError, StopIteration):
            return ""

    def _syntax_break(self, p: Path, rel: str, before: str, after: str, proposal: str) -> str:
        """'' unless this edit would turn a file that parses (py, json, toml, yaml) into one that does not: then the NOT APPLIED answer.
        A file that was already broken is edited as before (the write answers with SYNTAX GATE FAILED)."""
        if not self.syntax_reject or p.suffix.lower() not in _REJECT_SUFFIXES:
            return ""
        self._refuse_markers(before, after)                      # a pasted history marker is named as such, not as a syntax problem
        if edits.check_syntax(str(p), before) is not None:
            return ""
        err = edits.check_syntax(str(p), after)
        if err is None:
            return ""
        head = err.splitlines()[0].replace(str(p), rel)
        m = re.search(r"line (\d+)", head)
        rows = after.split("\n")
        around = ""
        if m and rows:
            at = min(max(int(m.group(1)), 1), len(rows))
            a, b = max(1, at - 1), min(len(rows), at + 1)
            around = " would leave these lines around the error:\n" + "\n".join(f"{'>>' if n == at else '  '} {n:4} | {rows[n - 1][:200]}" for n in range(a, b + 1)) + "\n"
        return (f"NOT APPLIED: this edit would break the syntax of {rel} ({head}). Your replacement text{around or ' leaves the file unparsable.'} "
                "The file is unchanged. Re-send the edit with the correct range or complete block (a multi-step refactor that needs a temporarily "
                "broken state: use apply_patch, which keeps writing, or set coding.syntax_reject to false)." + self._keep_proposal(p, proposal))

    def _commit_lines(self, p: Path, raw: str, lines: list[str], crlf: bool, trailing_nl: bool, what: str, first: int, count: int, note: str = "",
                      proposal: str | None = None) -> str:
        after = "\n".join(lines) + ("\n" if trailing_nl and lines else "")
        if proposal is not None:
            refused = self._syntax_break(p, self.rel(p), raw.replace("\r\n", "\n") if crlf else raw, after, proposal)
            if refused:
                return refused
        gate = self._commit_write(p, raw, after.replace("\n", "\r\n") if crlf else after, "edited")
        rel = self.rel(p)
        self.line_views[rel] = symbols.file_hash(after)
        a, b = max(1, first - 2), min(len(lines), first + max(count, 1) + 1)
        region = symbols.numbered(lines[a - 1:b], a)
        tag = f"new lines {first}-{first + count - 1} are #{symbols.range_hash(lines[first - 1:first + count - 1])}" if count else "lines removed"
        head = f"edited {rel}: {what} (file now {len(lines)} lines; {tag})."
        warn = self._dup_note(raw, after) + note
        if gate:
            return f"{head}\nSYNTAX GATE FAILED:\n{gate}\nFix it with another replace_lines on the lines shown.{warn}\n{region}"
        return f"{head} Syntax OK.{self._xref_note(p, after) if p.suffix == '.py' else ''}{warn}\n{region}"

    def replace_lines(self, path: str = "", start: Any = None, end: Any = None, new_text: Any = None, expect: Any = None, **kw: Any) -> str:
        if new_text is None:
            new_text = next((kw[k] for k in ("text", "content", "replace", "new") if k in kw), None)
        if new_text is None:
            return "error: replace_lines needs path, start, end and new_text (the new lines; \"\" deletes the range)."
        if has_placeholder(new_text):
            return PLACEHOLDER_ERROR
        p, raw, text, crlf, err = self._plain(path)
        if raw is None:
            return err
        refused = self._protected(p)
        if refused:
            return refused
        lines = text.split("\n")
        trailing = text.endswith("\n")
        if trailing:
            lines = lines[:-1]
        try:
            s = int(start)
            e = int(end) if end not in (None, "") else s
        except (TypeError, ValueError):
            return "error: start and end must be line numbers (integers)."
        rel = self.rel(p)
        new = self._new_lines(new_text)
        if not (1 <= s <= e <= len(lines)):
            hit = edits.find_applied(lines, new, min(max(s, 1), max(len(lines), 1)))
            if hit:
                return self._applied(rel, text, lines, *hit)
            return self._current_lines(rel, text, lines, min(max(s, 1), max(len(lines), 1)), min(max(e, 1), max(len(lines), 1)),
                                       f"lines {s}-{e} are outside the file (1-{len(lines)})", "replace_lines")
        shifted = 0
        why = self._stale(rel, text, lines, s, e, expect)
        if why:
            want = str(expect or "").strip().lstrip("#").lower()
            at = symbols.shifted_starts(lines, want, e - s + 1, s, SHIFT_WINDOW) if want else []
            if len(at) == 1:                                           # the stamped block stands at exactly one other place: only the numbers moved
                shifted = at[0] - s
                s, e = at[0], at[0] + (e - s)
            else:                                                      # none, or two (never guess): the new text may be there already, else reject
                hit = edits.find_applied(lines, new, s)
                if hit:
                    return self._applied(rel, text, lines, *hit)
                return self._current_lines(rel, text, lines, s, e, why, "replace_lines")
        out = lines[:s - 1] + new + lines[e:]
        if out == lines:
            return self._applied(rel, text, lines, s, s + len(new) - 1)
        delta = len(new) - (e - s + 1)
        what = (f"applied at lines {s}-{e} (the file had shifted by {shifted:+d} since you read it), replaced by {len(new)} line(s)" if shifted
                else f"lines {s}-{e} replaced by {len(new)} line(s)")
        return self._commit_lines(p, raw, out, crlf, trailing or not text, what + (f", the lines after moved by {delta:+d}" if delta else ""),
                                  s, len(new), self._retyped_note(lines, new, s, e), proposal=str(new_text or ""))

    @staticmethod
    def _retyped_note(lines: list[str], new: list[str], s: int, e: int) -> str:
        """A short range replaced by far more text that contains the line right after the range: the model re-typed text that follows,
        so the old copy will be there twice (a hint to the end line; the result already shows the numbers)."""
        if e >= len(lines) or len(new) <= 3 * (e - s + 1) or e - s + 1 >= 10:
            return ""
        nxt = edits._significant(lines[e])
        if nxt and any(r.strip() == nxt for r in new):
            return f"\nNOTE: your replacement re-includes the line that follows the range ({nxt[:80]}) - check the end line."
        return ""

    def insert_after(self, path: str = "", new_text: Any = None, line: Any = None, symbol: str = "", expect: Any = None, **kw: Any) -> str:
        if new_text is None:
            new_text = next((kw[k] for k in ("text", "content", "code") if k in kw), None)
        if has_placeholder(new_text):
            return PLACEHOLDER_ERROR
        if not str(new_text or "").strip():
            return "error: insert_after needs path, new_text and either line (0 = top of the file) or symbol (insert after that definition)."
        p, raw, text, crlf, err = self._plain(path)
        if raw is None:
            return err
        refused = self._protected(p)
        if refused:
            return refused
        lines = text.split("\n")
        trailing = text.endswith("\n")
        if trailing:
            lines = lines[:-1]
        rel = self.rel(p)
        if str(symbol or "").strip():
            span, cands = symbols.find_symbol(str(p), text, str(symbol))
            if span is None:
                return (f"error: no single definition '{symbol}' in {rel}" + (": candidates " + ", ".join(f"{c.name} (line {c.start})" for c in cands[:10]) if cands else "")
                        + ". Use show_symbol or outline to find it, or give line=N.")
            at = span.end
        else:
            try:
                at = int(line)
            except (TypeError, ValueError):
                return "error: give line (a number; 0 inserts at the top) or symbol (a definition name)."
            if not 0 <= at <= len(lines):
                return self._current_lines(rel, text, lines, max(1, min(at, len(lines))), max(1, min(at, len(lines))),
                                           f"line {at} is outside the file (0-{len(lines)})", "insert_after")
            if at > 0:
                why = self._stale(rel, text, lines, at, at, expect)
                if why:
                    return self._current_lines(rel, text, lines, at, at, why, "insert_after")
        new = self._new_lines(new_text)
        out = lines[:at] + new + lines[at:]
        return self._commit_lines(p, raw, out, crlf, trailing or not text, f"inserted {len(new)} line(s) after line {at}", at + 1, len(new),
                                  proposal=str(new_text or ""))

    # ── writes ──────────────────────────────────────────────────────────
    @staticmethod
    def _refuse_markers(before: str | None, after: str) -> None:
        added = new_markers(before, after)     # seen live: the model copied `...[cut]` into a replace_lines call and broke a file
        if added:
            raise ValueError(f"the new text contains the harness marker '{added[0]}' (a note the history shows, not real text). "
                             "Nothing was written. Fetch the real lines (show_symbol / read_file) and send the complete real text.")

    def _commit_write(self, p: Path, before: str | None, after: str, action: str) -> str:
        """Write, run the syntax gate, update bookkeeping; the one place every write goes through."""
        self._refuse_markers(before, after)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.ctx.step and not self.changes.checkpoint_tried:
            from agent import undo

            self.changes.checkpoint_tried = True
            self.changes.checkpoint = undo.git_checkpoint(str(p))
        atomic_write(p, after)
        self._forget_reads(p)
        if self.on_write is not None:
            try:
                self.on_write(self.rel(p), before, after)
            except Exception:  # noqa: BLE001 - the cursor is a bonus, never a failed write
                pass
        return self._bookkeep(p, before, after, action)

    def _bookkeep(self, p: Path, before: str | None, after: str, action: str) -> str:
        """Everything that follows a write (also one a shell command made, with before=None): change log, dirty set, ledger,
        syntax gate. Returns the syntax error text or ''."""
        self.changes.record(str(p), before, after)
        rel = self.rel(p)
        if action in ("run", "appended"):
            self._pre_edit.pop(rel, None)        # a chunked file is compared with its last CHECKED state, not with the unfinished chunk before
        else:
            self._pre_edit[rel] = before
        self.dirty.add(rel)
        if self.profile.test_cmd is None and rel.endswith(".py"):
            self.profile = profile_mod.detect(self.root, self.test_override)       # a first test file changes how we verify
        self.evidence.clear()
        self.last_edit_step = self.ctx.step
        self.ctx.ledger.note_file(rel, after, self.ctx.step, action)
        if action != "run":
            self._own(rel, "write")
        err = edits.check_syntax(str(p), after, self.check_cmd)
        self._removed_note = ""
        if err:
            self.syntax_bad[rel] = err
            return err
        self.syntax_bad.pop(rel, None)
        if before is not None and rel.endswith(".py"):
            self._removed_note = self._removed_symbols(rel, before, after)
        return ""

    def _removed_symbols(self, rel: str, before: str, after: str) -> str:
        """ONE factual line when this write removed a def / class / assignment name that other .py files still mention (where_used, no model).
        Seen live: State.mood_change was deleted while dialogue.py still called it, 33 tests were red mid-run."""
        try:
            old = {sp.name for sp in symbols.python_spans(before)}
            new = {sp.name for sp in symbols.python_spans(after)}
            kept = {n.rsplit(".", 1)[-1] for n in new}
            gone = list(dict.fromkeys(n.rsplit(".", 1)[-1] for n in sorted(old - new) if n.rsplit(".", 1)[-1] not in kept))
            gone = [n for n in gone if len(n) >= 3 and not (n.startswith("__") and n.endswith("__"))][:REMOVED_NAMES_CHECKED]
            parts = []
            for name in gone:
                hits = []
                for ln in symbols.where_used(self.root, name, 200).splitlines()[1:]:
                    m = re.match(r"([^:\s]+\.py):(\d+):", ln)
                    if m and m.group(1) != rel and "[definition]" not in ln and m.group(1) not in self._defining(name, m.group(1)):
                        hits.append(f"{m.group(1)}:{m.group(2)}")
                if hits:
                    parts.append(f"Removed {name} but it is still used in: " + ", ".join(hits[:4]) + (f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""))
            return "; ".join(parts)
        except Exception:  # noqa: BLE001 - advice only; never break a write
            return ""

    def _defining(self, name: str, rel: str) -> list[str]:
        """[rel] when that file defines `name` itself (then its uses are its own, not the removed symbol's)."""
        try:
            text = (self.root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [rel] if any(sp.name.rsplit(".", 1)[-1] == name for sp in symbols.python_spans(text)) else []

    def _protected(self, p: Path) -> str:
        """'' when `p` may be written, else the refusal. A read-only file is the user's own protection (a game's save file):
        a rename-based write would silently replace it, so it is checked here - and the model is told not to retry.
        A path matching `coding.protect` (a save file the human plays with, writable for him) is refused the same way."""
        if self._is_protected(p):
            return (f"error: {self.rel(p)} {PROTECT_MARK}, so it was NOT changed. Do not retry this; the player plays with it. "
                    "If you need a changed version or want to test, work on a copy in .scratch/.")
        if p.exists() and not os.access(p, os.W_OK):
            return (f"error: {self.rel(p)} is read-only (its file mode protects it), so it was NOT changed. Do not retry this; "
                    "if you need a changed version, work on a copy in .scratch/.")
        if p.is_dir():
            return f"error: {self.rel(p)} is a directory"
        return ""

    def _is_protected(self, p: Path) -> bool:
        try:
            rel = p.resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return False
        return is_protected_rel(rel, self.protect)

    def _protected_inside(self, d: Path) -> str:
        """The refusal when a folder that is about to be moved contains a protected file ('' otherwise)."""
        if not d.is_dir():
            return ""
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if x not in SNAPSHOT_SKIP]
            for fn in fns:
                if self._is_protected(Path(dp) / fn):
                    return self._protected(Path(dp) / fn)
        return ""

    def _keep_rejected(self, p: Path, content: str) -> str:
        """A big generated file that could not go where it was meant to is kept in .scratch/ rather than thrown away."""
        if len(content) < OVERWRITE_GUARD_CHARS:
            return ""
        try:
            dest = self.root / ".scratch" / "unsaved" / p.name
            edits.ensure_scratch_excluded(self.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(dest, content)
            return f" Your {len(content)}-char content was kept in {self.rel(dest)} so it is not lost."
        except OSError:
            return ""

    def write_file(self, path: str, content: str = "", overwrite: bool = False, **_: Any) -> str:
        """Always accepts what the model generated: refusing a file AFTER it took minutes to write throws the work away.
        Advice (prefer edit_file, split big files, stop rewriting) comes as a note on the result, never as a refusal.
        The only refusals: a read-only target (the user protected it; the content is kept in .scratch/) and absurd sizes."""
        if not isinstance(content, str) or not content:
            return "error: write_file needs path and content (the complete file text)."
        if has_placeholder(content):
            return PLACEHOLDER_ERROR
        if len(content) > HARD_WRITE_CAP:
            return f"error: {len(content)} chars is beyond the {HARD_WRITE_CAP} char cap of one call - split it into modules."
        p = self.resolve(path)
        refused = self._protected(p)
        if refused:
            return refused + self._keep_rejected(p, content)
        notes = []
        existing = None
        if p.exists():
            existing, _err = self._read_text(p)
            if existing is None:                          # a binary file: move it to the trash first so /undo-style recovery stays possible
                from agent import undo

                undo.trash(p)
                notes.append(f"the old binary {self.rel(p)} was moved to the trash")
        text = content if content.endswith("\n") else content + "\n"
        if existing is not None and _uses_crlf(existing) and "\r\n" not in text:
            text = text.replace("\n", "\r\n")                 # keep the file's line endings
        rel = self.rel(p)
        recent = [pth for st, pth in self._write_log if self.ctx.step - st <= 8 and pth == rel]
        self._write_log.append((self.ctx.step, rel))
        del self._write_log[:-12]
        err = self._commit_write(p, existing, text, "wrote")
        if existing is not None and len(existing) > OVERWRITE_GUARD_CHARS:
            notes.append(f"replaced an existing {len(existing)}-char file; for small changes edit_file is cheaper next time")
        if len(content) > MAX_WRITE_CHARS:
            notes.append("large single write accepted; splitting big modules into several files makes later edits cheaper")
        if existing is not None and len(recent) >= 2:
            notes.append(f"{rel} was rewritten {len(recent) + 1} times in a few steps - fix problems with edit_file instead of rewriting")
        tail = ("\nNOTE: " + "; ".join(notes)) if notes else ""
        tail += self._dup_note(existing, text)
        head = f"wrote {rel} ({text.count(chr(10))} lines)"
        if err:
            return f"{head}\nSYNTAX GATE FAILED:\n{err}\nFix it with edit_file (a small search/replace) - do not rewrite the whole file.{tail}"
        out = edits.outline(str(p), text, 40)
        warn = self._xref_note(p, text)
        return f"{head}. Syntax OK." + (("\nOUTLINE\n" + "\n".join(out)) if out else "") + warn + tail

    def _xref_note(self, p: Path, text: str) -> str:
        """Cross-module warnings (xref.py) and the static checks (static.py) for a Python write that passed the syntax gate."""
        if p.suffix != ".py" or self.root not in p.parents:
            return ""
        removed, self._removed_note = (f"\n{self._removed_note}" if self._removed_note else ""), ""
        return xref.note(self.root, p, text) + self._static_note(p, text) + removed

    def _static_note(self, p: Path, text: str) -> str:
        """Static problems of a Python write. static_bad keeps ALL of them (the final gate needs the whole list); the note shows only
        the NEW ones - not in the file as it was before this edit (compared as message + content of the line, so a shifted line is not
        new) - and sums up the rest in one line. Without a known earlier state everything counts as new."""
        try:
            probs = static.check(self.root, p, text)
        except Exception:  # noqa: BLE001 - advice only; never break a write
            probs = []
        rel = self.rel(p)
        sigs = set(static.signatures(probs, text))
        if rel in self._pre_edit:
            earlier = self._pre_edit.pop(rel)
            try:
                old = set() if earlier is None else set(static.signatures(static.check(self.root, p, earlier), earlier))
            except Exception:  # noqa: BLE001
                old = set()
        else:
            old = self._static_sig.get(rel, set())
        self._static_sig[rel] = sigs
        if probs:
            self.static_bad[rel] = probs
        else:
            self.static_bad.pop(rel, None)
        fresh = static.new_problems(probs, text, old)
        return static.note(fresh, unchanged=len(probs) - len(fresh))

    def append_file(self, path: str, content: str = "", final: bool = False, **_: Any) -> str:
        if not content:
            return "error: append_file needs path and content."
        if has_placeholder(content):
            return PLACEHOLDER_ERROR
        p = self.resolve(path)
        rel0 = self.rel(p)
        if self._is_protected(p):
            return self._protected(p) + self._keep_rejected(p, content)
        if len(content) < TINY_APPEND_CHARS and not final:
            n = self._tiny_appends.get(rel0, 0) + 1
            self._tiny_appends[rel0] = n
            if n >= 3:
                return (f"refused: that is the {n}th tiny append ({len(content)} chars) to {rel0} in a row. Appending line by line wastes steps and "
                        "never converges. Write the rest of the file as ONE bigger chunk (up to ~8000 chars), or if the file is complete call "
                        "append_file with final=true / move on and run it.")
        else:
            self._tiny_appends[rel0] = 0
        refused = self._protected(p)
        if refused:
            return refused + self._keep_rejected(p, content)
        before = ""
        if p.exists():
            before, err = self._read_text(p)
            if before is None:
                return f"error: {err} - nothing appended"
        chunk = content if content.endswith("\n") else content + "\n"
        nl = "\r\n" if before and _uses_crlf(before) else "\n"
        if nl == "\r\n" and "\r\n" not in chunk:
            chunk = chunk.replace("\n", "\r\n")
        after = before + ("" if not before or before.endswith("\n") else nl) + chunk
        err = self._commit_write(p, before or None, after, "appended")
        rel = self.rel(p)
        warn = self._dup_note(before, after)
        if err and final:
            return f"appended to {rel} ({after.count(chr(10))} lines total)\nSYNTAX GATE FAILED:\n{err}{warn}"
        if err:
            self.syntax_bad.pop(rel, None)            # may just be incomplete: only the last chunk (final=true) is gated
            return f"appended to {rel} ({after.count(chr(10))} lines total). Not checked yet (more chunks expected); pass final=true on the last chunk."
        return f"appended to {rel} ({after.count(chr(10))} lines total). Syntax OK." + (self._static_note(p, after) if p.suffix == ".py" else "") + warn

    def edit_file(self, path: str, edits_: Any = None, replace_all: bool = False, **kw: Any) -> str:
        items = edits_ if edits_ is not None else kw.get("edits")
        if items is None and "search" in kw:                     # a single edit passed flat
            items = [{"search": kw.get("search"), "replace": kw.get("replace", "")}]
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list) or not items or not all(isinstance(e, dict) for e in items):
            return "error: edit_file needs path plus search and replace (exact old text -> new text), or edits=[{\"search\": \"...\", \"replace\": \"...\"}, ...]"
        p = self.resolve(path)
        raw, err = self._read_text(p)
        if raw is None:
            return err
        refused = self._protected(p)
        if refused:
            return refused
        crlf = _uses_crlf(raw)
        before = raw.replace("\r\n", "\n") if crlf else raw          # edit in plain newlines, write back in the file's own
        if crlf:
            items = [{**e, "search": str(e.get("search", "")).replace("\r\n", "\n"), "replace": str(e.get("replace", "")).replace("\r\n", "\n")} for e in items]
        res = edits.apply_edits(before, items, replace_all)
        rel = self.rel(p)
        if res.text is None or res.text == before:
            hit = self._edits_applied(before, items)
            if hit:
                return self._applied(rel, before, before.split("\n")[:-1] if before.endswith("\n") else before.split("\n"), *hit)
        if res.text is None:
            return (f"edit_file failed, nothing changed: {res.error}\n(No need to quote old text: show_symbol(name, path) gives the exact lines, "
                    "then replace_lines(path, start, end, new_text, expect=#hash) changes them.)")
        if res.text == before:
            return "no change: the replacement is identical to the existing text."
        refused = self._syntax_break(p, rel, before, res.text, "\n--- next edit ---\n".join(str(e.get("replace", "")) for e in items))
        if refused:
            return refused
        gate = self._commit_write(p, raw, res.text.replace("\n", "\r\n") if crlf else res.text, "edited")
        diff = "\n".join(list(difflib.unified_diff(before.splitlines(), res.text.splitlines(), lineterm="", n=2))[2:40])
        head = f"edited {rel}: {len(items)} edit(s) applied ({res.how})."
        warn = self._dup_note(before, res.text)
        if gate:
            return f"{head}\nSYNTAX GATE FAILED:\n{gate}\nFix it now with another small edit_file.{warn}\n{diff}"
        return f"{head} Syntax OK.{self._xref_note(p, res.text)}{warn}\n{diff}"

    @staticmethod
    def _edits_applied(before: str, items: list[dict]) -> tuple[int, int] | None:
        """(first, last) line when EVERY edit's replacement already stands once in the file (the search text is gone because the edit was
        applied before); None when an edit deletes text, a replacement is short or stands more than once (nothing is guessed)."""
        src = before.split("\n")
        spans = []
        for e in items:
            rows = str(e.get("replace", "")).split("\n")
            while rows and not rows[0].strip():
                rows.pop(0)
            while rows and not rows[-1].strip():
                rows.pop()
            if not rows or sum(len(edits._norm(r)) for r in rows) < EDIT_APPLIED_MIN_CHARS:
                return None
            hits = edits._find_lines(src, rows)
            if len(hits) != 1:
                return None
            spans.append((hits[0] + 1, hits[0] + len(rows)))
        return (min(a for a, _ in spans), max(b for _, b in spans)) if spans else None

    def apply_patch(self, patch: str = "", path: str | None = None, **_: Any) -> str:
        files = edits.parse_patch(patch or "", path)
        if not files:
            return ("error: no files found in the patch. Send a unified diff: '--- a/path', '+++ b/path', then '@@' hunks whose lines start with ' ', '-' or '+' "
                    "(or pass path=... when the diff has no file header). For a small change edit_file is simpler.")
        planned: list[tuple[Path, str | None, str]] = []
        for path, hunks in files:
            p = self.resolve(path)
            refused = self._protected(p)
            if refused:
                return refused.replace("error:", "apply_patch failed:", 1)
            before = None
            if p.exists():
                before, err = self._read_text(p)
                if before is None:
                    return f"apply_patch failed on {path}, nothing changed: {err}"
            crlf = bool(before) and _uses_crlf(before)
            res = edits.apply_hunks((before or "").replace("\r\n", "\n") if crlf else before or "", hunks)
            if res.text is None:
                return f"apply_patch failed on {path}, nothing changed: {res.error}"
            planned.append((p, before, res.text.replace("\n", "\r\n") if crlf else res.text))
        out = []
        for p, before, after in planned:
            gate = self._commit_write(p, before, after, "patched" if before is not None else "wrote")
            out.append(f"{self.rel(p)}: {'SYNTAX GATE FAILED' if gate else 'ok'}" + (f"\n{gate}" if gate else self._xref_note(p, after)))
        return "patched " + str(len(planned)) + " file(s)\n" + "\n".join(out)

    def _own(self, rel: str, origin: str) -> None:
        self.owned.pop(rel, None)
        self.owned[rel] = origin
        while len(self.owned) > MAX_OWNED:
            del self.owned[next(iter(self.owned))]

    def _forget(self, rel: str) -> None:
        """A file is gone: no ledger entry (it would render as MISSING), no pending problems."""
        self.owned.pop(rel, None)
        self.dirty.discard(rel)
        self.syntax_bad.pop(rel, None)
        self.static_bad.pop(rel, None)
        self._static_sig.pop(rel, None)
        self._pre_edit.pop(rel, None)
        self.ctx.ledger.files.pop(rel, None)

    def _moved(self, old: str, new: str) -> None:
        """A file or folder moved: every ledger entry, dirty mark and pending problem below `old` follows it to `new`."""
        led = self.ctx.ledger.files
        for key in [k for k in list(led) + list(self.dirty) + list(self.syntax_bad) + list(self.static_bad) if k == old or k.startswith(old + "/")]:
            dst = new + key[len(old):]
            if key in led:
                e = led.pop(key)
                led[dst] = FileEntry("moved", self.ctx.step, e.lines, e.size, e.names)
            if key in self.dirty:
                self.dirty.discard(key)
                self.dirty.add(dst)
            if key in self._static_sig:
                self._static_sig[dst] = self._static_sig.pop(key)
            for bad in (self.syntax_bad, self.static_bad):
                if key in bad:
                    bad[dst] = bad.pop(key)
        if not (self.root / new).is_dir():
            self.dirty.add(new)

    def delete_file(self, path: str, **_: Any) -> str:
        p = self.resolve(path)
        if not p.is_file():
            return f"error: {self.rel(p)} is not a file"
        refused = self._protected(p)
        if refused:
            return refused
        before, _e = self._read_text(p)
        from agent import undo

        undo.trash(p)
        self.changes.record(str(p), before, None)
        self._forget(self.rel(p))
        return f"deleted {self.rel(p)} (kept in the undo trash)"

    def move_file(self, src: str, dst: str, **_: Any) -> str:
        a, b = self.resolve(src), self.resolve(dst)
        if not a.exists():
            return f"error: {self.rel(a)} does not exist"
        if b.exists():
            return f"error: {self.rel(b)} already exists"
        refused = self._protected(a) if a.is_file() else self._protected_inside(a)
        if not refused and self._is_protected(b):
            refused = self._protected(b)
        if refused:
            return refused
        b.parent.mkdir(parents=True, exist_ok=True)
        before, _e = self._read_text(a) if a.is_file() else (None, "")
        a.rename(b)
        if before is not None:
            self.changes.record(str(a), before, None)
            self.changes.record(str(b), None, before)
        self._moved(self.rel(a), self.rel(b))
        return f"moved {self.rel(a)} -> {self.rel(b)}"

    # ── running things ──────────────────────────────────────────────────
    def _exec(self, command: str, timeout: int, stdin: str | None = None, no_sleep: bool = False,
              watch: Path | None = None) -> tuple[int | None, str, float]:
        """(exit code or None when it was stopped, output, seconds). Own process group, stdin closed unless fed, stdout+stderr
        interleaved into a temp file; on timeout, too much output, Ctrl-C or normal exit the whole group is killed (a
        `server &` or a grandchild holding the pipe used to keep subprocess.run waiting long after the timeout).
        `no_sleep`: Python's time.sleep is a no-op in the command's interpreters. `watch`: a side file (the tee of a filtered
        pipeline) that counts against the output cap too. CPU seconds used by the command land in self.last_cpu."""
        env = tuismoke.headless_env({**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1", "CI": "1",
                                     "GIT_TERMINAL_PROMPT": "0"})          # SDL's dummy drivers unless the user set others (games run headless)
        if no_sleep:
            env["PYTHONPATH"] = os.pathsep.join(x for x in (str(NO_SLEEP_DIR), env.get("PYTHONPATH", "")) if x)
        cpu0 = _child_cpu()
        t0 = time.monotonic()
        note = ""
        argv = osport.shell_argv(command)
        if argv is None:
            return 127, "could not start the command: " + osport.NO_SHELL_MESSAGE, 0.0
        with tempfile.TemporaryFile() as out:
            try:
                proc = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                        stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            except OSError as exc:
                return 127, f"could not start the command: {exc}", 0.0
            if stdin:
                threading.Thread(target=_feed, args=(proc.stdin, stdin.encode("utf-8", errors="replace")), daemon=True).start()
            try:
                while True:
                    try:
                        proc.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if time.monotonic() - t0 > timeout:
                        note = f"\n[timed out after {timeout}s and was stopped. Programs that wait for input need stdin=... or a --demo flag.]"
                        break
                    if os.fstat(out.fileno()).st_size > MAX_OUTPUT_BYTES or (watch is not None and _size(watch) > MAX_OUTPUT_BYTES):
                        note = f"\n[stopped: it printed more than {MAX_OUTPUT_BYTES // 1_000_000} MB of output]"
                        break
            finally:
                _kill_group(proc)                       # leftovers too: nothing a command started outlives it
            text = _read_capped(out)
        self.last_cpu = _child_cpu() - cpu0
        return (None if note else proc.returncode), text + note, time.monotonic() - t0

    def run(self, command: str = "", timeout: Any = None, stdin: str | None = None, no_sleep: Any = False, **_: Any) -> str:
        if not command.strip():
            return "error: run needs command"
        hit = new_markers(None, command) if _WRITES_RX.search(command) else []
        if hit:
            return f"error: the command contains the harness marker '{hit[0]}' (a note the history shows, not real text) and can write. Nothing was run."
        try:
            t = max(1, min(int(timeout or self.run_timeout), 600))
        except (TypeError, ValueError):
            t = self.run_timeout
        no_sleep = no_sleep is True or str(no_sleep).strip().lower() in ("true", "1", "yes")
        # `prog | sed -n '1,45p'`: the program's own output is teed to a temp file (outside the repo, so a command that
        # empties .scratch cannot break it) and kept, so another part of it never needs another run of the program.
        cut, tee, actual = filter_split(command), None, command
        if cut is not None:
            fd, name = tempfile.mkstemp(prefix="agent-run-", suffix=".txt")
            os.close(fd)
            tee = Path(name)
            op = "|&" if command[cut + 1:cut + 2] == "&" else "|"
            actual = f"{command[:cut]}{op} tee {shlex_quote(name)} {command[cut:]}"
        snap = None if readonly_shell(command) else snapshot_tree(self.root, keep_ro=True, protect=self.protect)    # a look cannot change files: no snapshot
        try:
            code, out, secs = self._exec(actual, t, stdin, no_sleep=no_sleep, watch=tee)
            unfiltered = tee.read_bytes()[:2_000_000].decode("utf-8", errors="replace") if tee else None
        finally:
            if tee:
                tee.unlink(missing_ok=True)
        changed = self._after_run(snap)
        if snap is None and not readonly_shell(command):
            self.read_cache.clear()                                 # no measurement (too many files): nobody knows what the command changed
        if (self.root / ".scratch").is_dir():                       # the model made it itself (the prompt tells it to): keep it out of `git add -A`
            edits.ensure_scratch_excluded(self.root)
        self.ctx.ledger.note_run(self.ctx.step, command, code, out.strip().splitlines()[-1] if out.strip() else "")
        self.evidence.append((self.ctx.step, command, code))
        raw = out.strip()
        out = outputs.summarize_pytest(raw)
        extra = outputs.traceback_snippets(self.root, out) if (code not in (0, None) or "Traceback" in out) else ""
        body = cap_output(out.strip(), OUT_HEAD, OUT_TAIL) + extra
        cut_line = raw[:OUT_HEAD].count("\n") + 1 if out == raw and len(raw) > OUT_HEAD + OUT_TAIL + 200 else 1       # first line the head does not show completely
        body += self._spill(raw, secs, unfiltered, command[cut:].strip() if cut is not None else "", cut_line)
        body += self._sleepy(command, code, secs, no_sleep)
        label = code if code is not None else "stopped" if "[stopped: it printed" in out else "timeout"
        head = f"exit code {label}  ({secs:.1f}s)\n{body}" if body else f"exit code {label}  ({secs:.1f}s)  (no output)"
        return head + ("\n" + changed if changed else "") + self._tui_redirect(command, stdin, raw)

    def _tui_redirect(self, command: str, stdin: str | None, output: str) -> str:
        """Seen live: the prompt sentence about tui_smoke was in context and the model never called it - it faked the screen and proved a symptom
        patch. So `run` points at the tool when the command is such a case (tuihint.py): one line, the long form once per program, and for a
        terminal program that failed for lack of a terminal the program is also run ONCE through tui_smoke (no keys, 3 s) and its screen appended."""
        found = tuihint.scan(command, stdin, output, self.root)
        if not found:
            return ""
        key, kind = found
        first = key not in self._tui_told
        self._tui_told.add(key)
        text = "\n" + (tuihint.LONG_HINT if first else tuihint.SHORT_HINT)
        if kind == "tui" and first and tuihint.auto_run_ok(command) and (self.smoke_allow is None or self.smoke_allow(command)):
            try:
                res, changed = self._smoke(command, None, tuismoke.DEFAULT_SIZE, None, tuihint.AUTO_TIMEOUT, None)
                text += (f"\ntui_smoke (automatic, no keys, {tuihint.AUTO_TIMEOUT} s timeout):\n" + tuismoke.format_result(res, min(self.smoke_max_chars, 1800))
                         + ("\n" + changed if changed else ""))
            except Exception:  # noqa: BLE001 - the hint is a convenience: the run result stands without it
                pass
        return text

    def _after_run(self, before: tuple[dict, dict] | None) -> str:
        """Account for what a shell command did to the tree (before = snapshot_tree taken just before it): read-only files it
        touched are restored; created / modified text files get the bookkeeping of a tool write (before unknown), deleted ones
        leave the ledger. Returns the note lines for the result ('' when nothing changed or no snapshot could be taken)."""
        if before is None:
            return ""
        after = snapshot_tree(self.root)
        if after is None:
            return ""
        old, now = before[0], after[0]
        lines = self._restore_read_only(before[1])
        for path in sorted(p for p in now if p not in old and is_protected_rel(os.path.relpath(p, self.root), self.protect)):
            try:
                Path(path).unlink()                         # a protected file the command created: gone again
            except OSError:
                continue
            rel = self.rel(Path(path))
            lines.append((f"NOTE: {rel} {PROTECT_MARK.split(' (')[0]}; this command created it and it was deleted. {PROTECT_COPY}", rel))
            now.pop(path, None)
        restored = {self.root / x[1] for x in lines}
        lines = [x[0] for x in lines]
        created = sorted(p for p in now if p not in old)
        modified = sorted(p for p in now if p in old and now[p] != old[p])
        deleted = sorted(p for p in old if p not in now)
        for path in (*created, *modified, *deleted):               # also the ones too big / too many to be recorded below
            self._forget_reads(Path(path))
        done: list[str] = []
        errors: list[str] = []
        for path in created:
            self._own(self.rel(Path(path)), "run")
        for path in [*created, *modified][:MAX_RUN_RECORDS]:
            p = Path(path)
            if p in restored or now[path][1] > MAX_RUN_TEXT:
                continue
            try:
                text, _e = self._read_text(p)
            except OSError:
                continue
            if text is None:
                continue
            err = self._bookkeep(p, None, text, "run")
            done.append(self.rel(p))
            if err:
                errors.append(err)
            elif p.suffix == ".py" and len(done) <= 20:
                self._static_note(p, text)
        for path in deleted:
            rel = self.rel(Path(path))
            self.owned.pop(rel, None)
            if Path(path) not in restored and rel in self.ctx.ledger.files:
                self._forget(rel)
                done.append(rel)
        if restored:
            self.read_cache.clear()
        if done:
            self.read_cache.clear()
            done.sort(key=lambda r: (is_scratch_path(r), r))
            shown = ", ".join(done[:8]) + (f" and {len(done) - 8} more" if len(done) > 8 else "")
            lines.insert(0, f"NOTE: this command changed {shown} (recorded).")
        return "\n".join([*lines, *(f"SYNTAX GATE FAILED:\n{e}" for e in errors[:3])])

    def _restore_read_only(self, saved: dict) -> list[tuple[str, str]]:
        """Put back every file that was read-only (or protected, coding.protect) before the command and is now missing, different
        or has another mode. [(note, rel)]"""
        out: list[tuple[str, str]] = []
        for path, (data, mode, mtime) in saved.items():
            p = Path(path)
            try:
                st = p.stat()
                if stat.S_IMODE(st.st_mode) == mode and st.st_size == len(data) and (st.st_mtime_ns == mtime or p.read_bytes() == data):
                    continue                               # untouched (or only touch-ed)
            except OSError:
                pass                                       # gone
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.exists():
                    p.chmod(0o600)
                p.write_bytes(data)
                p.chmod(mode)
                os.utime(p, ns=(mtime, mtime))
            except OSError:
                continue
            rel = self.rel(p)
            self.dirty.discard(rel)
            if self._is_protected(p):
                out.append((f"NOTE: {rel} is protected; this command changed it and the change was undone. {PROTECT_COPY}", rel))
            else:
                out.append((f"NOTE: this command tried to change the read-only file {rel}; it was restored.", rel))
        return out

    def _spill(self, out: str, secs: float, unfiltered: str | None = None, filt: str = "", start: int = 1) -> str:
        """Seen live: a 17 s scripted game run piped through `sed -n '1,45p'`, then '1,50p', '1,55p' ... 22 times - the model
        re-ran the program to page through its output. Long or slow output is kept whole in .scratch/spill/run-<n>.txt and the
        result ends with one line saying where (spill.py); for a filtered pipeline it is the program's output BEFORE the filter (the
        part the model cannot see). `start`: the first line of the output that the shown text does not contain."""
        if unfiltered is not None:
            text = unfiltered.strip()
            if not text or text == out:
                return ""                                  # nothing the filter hid: no file, nothing to point at
            return self.spill.note("run", text, what=f", the program's output before `{filt[:60]}`")
        if len(out) > OUT_HEAD + OUT_TAIL + 200 or (secs >= 5 and len(out) > 600):
            return self.spill.note("run", out, start=start)
        return ""

    def _sleepy(self, command: str, code: int | None, secs: float, no_sleep: bool) -> str:
        """Seen live: ~50 scripted runs of a text game at 7-26 s each - it prints with a typing effect (time.sleep per line), so
        most of the session's wall clock was the game sleeping. A slow Python run that hardly used the CPU says so, once per run,
        and names the switch that removes the waiting."""
        if no_sleep or code is None or secs < SLOW_RUN_S or self.last_cpu > SLEEPY_CPU * secs:
            return ""
        if not re.search(r"\bpython[\d.]*\b|\.py\b|\bpytest\b", command) or re.search(r"\b(pip|install|curl|wget|requests)\b", command):
            return ""
        return (f"\n[it took {secs:.0f}s but used only {self.last_cpu:.1f}s of CPU: the program mostly waits in time.sleep (pacing, a typing "
                "effect). For test runs pass no_sleep=true to `run` - Python's time.sleep then returns at once and the run no longer waits]")

    def _smoke(self, command: str, keys: Any, size: Any, expect: Any, timeout: Any, settle_ms: Any) -> tuple[tuismoke.SmokeResult, str]:
        """One tui_smoke run with the same guards as `run`: protected files (coding.protect, read-only files) are snapshotted before and put back
        after, a protected file the program created is deleted again. Returns (result, the NOTE lines of what the program changed)."""
        snap = snapshot_tree(self.root, keep_ro=True, protect=self.protect)
        try:
            res = tuismoke.run_smoke(command, keys, size, expect, timeout or self.smoke_timeout, settle_ms or tuismoke.DEFAULT_SETTLE_MS, cwd=self.root)
        finally:
            changed = self._after_run(snap)
        if snap is None:
            self.read_cache.clear()                                  # no measurement (too many files): nobody knows what the program changed
        if (self.root / ".scratch").is_dir():
            edits.ensure_scratch_excluded(self.root)
        return res, changed

    def tui_smoke(self, command: str = "", keys: Any = None, size: Any = tuismoke.DEFAULT_SIZE, expect: Any = None, timeout: Any = None,
                  settle_ms: Any = None, **_: Any) -> str:
        """Run an interactive terminal program on a real pty, send keys, return the screen text and PASS/FAIL per expected text (tuismoke.py).
        The command goes through the same policy as `run` (engine.decide); here: the marker guard, the protect snapshot, the ledger."""
        if not str(command).strip():
            return "error: tui_smoke needs command"
        if _WRITES_RX.search(command) and new_markers(None, command):
            return "error: the command contains a harness marker (a note the history shows, not real text) and can write. Nothing was run."
        res, changed = self._smoke(command, keys, size, expect, timeout, settle_ms)
        full = tuismoke.format_result(res, 10 ** 9)
        text = tuismoke.format_result(res, self.smoke_max_chars)
        if len(full) > len(text) + 40:
            text += self.spill.note("tui_smoke", full, what=", the whole screen and transcript")
        if res.error:
            return text
        code = res.exit_code if res.status == "exited" else None
        self.ctx.ledger.note_run(self.ctx.step, command, code, text.splitlines()[0])
        self.evidence.append((self.ctx.step, command, code))
        return text + ("\n" + changed if changed else "")

    def tui_check(self, spec: dict[str, Any]) -> tuismoke.SmokeResult:
        """The `tui` check kind (checks.py): the same run, from a spec {command, keys, expect, size, timeout, settle_ms}."""
        return self._smoke(str(spec.get("command") or ""), spec.get("keys"), spec.get("size") or tuismoke.DEFAULT_SIZE, spec.get("expect"),
                           spec.get("timeout"), spec.get("settle_ms"))[0]

    def check_command_list(self) -> list[str]:
        return profile_mod.check_commands(self.profile, sorted(self.dirty), self.check_cmd)

    def run_tests(self, filter: str | None = None, **_: Any) -> str:
        cmd = self.profile.test_cmd
        if not cmd:
            return "no test command is known for this project. Write tests (test_*.py) or tell the user how to run it; `run` can execute any command."
        if filter:
            cmd = f"{cmd} -k {shlex_quote(filter)}" if "pytest" in cmd else cmd
        return self.run(cmd, timeout=300)

    def lint(self, **_: Any) -> str:
        cmds = self.check_command_list() or ([self.profile.lint_cmd] if self.profile.lint_cmd else [])
        if not cmds:
            return "nothing to lint: no files edited yet"
        out = []
        for c in cmds[:3]:
            out.append(f"$ {c}\n{self.run(c, timeout=120)}")
        return "\n".join(out)

    def git_status(self, **_: Any) -> str:
        code, out, _s = self._exec("git status --short --branch 2>&1 | head -60", 20)
        return out.strip() or "(clean)"

    def git_diff(self, path: str | None = None, staged: bool = False, **_: Any) -> str:
        cmd = "git diff --stat && git diff" + (" --cached" if staged else "") + (f" -- {shlex_quote(path)}" if path else "")
        code, out, _s = self._exec(cmd + " 2>&1 | head -400", 30)
        return cap_output(out.strip() or "(no changes)", 1500, 5000)

    def git_commit(self, message: str = "", **_: Any) -> str:
        if not message.strip():
            return "error: git_commit needs message"
        code, out, _s = self._exec(f"git add -A && git commit -q -m {shlex_quote(message)} 2>&1 | tail -5", 60)
        return f"exit code {code}\n{out.strip()}"

    # ── registry ────────────────────────────────────────────────────────
    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        S = {"type": "string"}

        def add(name: str, desc: str, props: dict[str, Any], required: list[str], handler: Callable[..., str]) -> None:
            risk = {READ: "safe", WRITE: "caution", SHELL: "dynamic", NET: "safe", OTHER: "safe"}[KINDS.get(name, OTHER)]
            reg.register(Tool(name=name, description=desc, parameters={"type": "object", "properties": props, "required": required},
                              handler=handler, risk=risk))

        add("read_file", "Read a file (exact text, no line numbers). Without start/end you get at most the first ~250 lines (~9000 chars) and a last line "
            "saying where to continue; use start/end for the part you need, or outline=true for just the structure. Repeated reads of an unchanged "
            "file are answered cheaply.",
            {"path": S, "start": {"type": "integer"}, "end": {"type": "integer"}, "outline": {"type": "boolean"}}, ["path"], self.read_file)
        add("read_files", "Read up to 6 small files in one step.", {"paths": {"type": "array", "items": S}}, ["paths"], self.read_files)
        add("outline", "Structure of a file: definitions with line numbers (cheap way to learn an API).", {"path": S}, ["path"], self.outline)
        add("list_dir", "List a folder (depth 1-3).", {"path": S, "depth": {"type": "integer"}}, [], self.list_dir)
        add("search", "Regex search across the repo (ripgrep). Returns file:line: text.",
            {"pattern": S, "path": S, "glob": S, "context": {"type": "integer"}, "ignore_case": {"type": "boolean"}}, ["pattern"], self.search)
        add("repo_map", "Map of all files with their top-level definitions.", {}, [], self.repo_map)
        add("write_file", "Create a NEW file with its complete content. Existing large files must be changed with edit_file. "
            "A syntax gate runs and reports the exact error line.",
            {"path": S, "content": S, "overwrite": {"type": "boolean"}}, ["path", "content"], self.write_file)
        add("append_file", "Append a chunk to a file (for building a big file in parts). Pass final=true on the last chunk to run the syntax gate.",
            {"path": S, "content": S, "final": {"type": "boolean"}}, ["path", "content"], self.append_file)
        add("edit_file", "Change a file by replacing exact text: `search` (text that is in the file now, unique; small whitespace slips are tolerated) "
            "becomes `replace`. For several changes call it again, or pass edits=[{search, replace}, ...] (applied atomically). Prefer this over rewriting a file.",
            {"path": S, "search": S, "replace": S, "edits": {"type": "array", "items": {"type": "object", "properties": {"search": S, "replace": S}}},
             "replace_all": {"type": "boolean"}}, ["path"],
            lambda **kw: self.edit_file(kw.pop("path", ""), kw.pop("edits", None), kw.pop("replace_all", False), **kw))
        add("apply_patch", "Apply a unified diff ('--- a/x', '+++ b/x', '@@' hunks) to one or more files; matched by content, not line numbers. "
            "If the diff has no file header give path=...",
            {"patch": S, "path": S}, ["patch"], self.apply_patch)
        add("show_symbol", "Show exactly one definition (function, class, Class.method, or a constant/dict) with line numbers and its #hash. "
            "Cheaper than reading a range; path may be omitted when the name is unique in the repo.",
            {"name": S, "path": S}, ["name"], self.show_symbol)
        add("where_used", "Every line in the repo that uses a name (whole word), definitions first.", {"name": S}, ["name"], self.where_used)
        add("replace_lines", "Replace lines start..end (as numbered by show_symbol / read_file / outline / search) with new_text - no need to quote "
            "the old text. Pass expect=<the #hash shown for that range>; if the lines changed meanwhile nothing is changed and you get the "
            "current lines. new_text \"\" deletes them.",
            {"path": S, "start": {"type": "integer"}, "end": {"type": "integer"}, "new_text": S, "expect": S},
            ["path", "start", "end", "new_text"], self.replace_lines)
        add("insert_after", "Insert new_text after a line number (0 = top of the file) or after a whole definition (symbol=name, e.g. a new "
            "function after an existing one). No old text needed.",
            {"path": S, "new_text": S, "line": {"type": "integer"}, "symbol": S, "expect": S}, ["path", "new_text"], self.insert_after)
        add("delete_file", "Delete a file (recoverable with /undo).", {"path": S}, ["path"], self.delete_file)
        add("move_file", "Move or rename a file.", {"src": S, "dst": S}, ["src", "dst"], self.move_file)
        add("run", "Run a shell command in the repo (tests, scripts, installs). `stdin` is fed to the program - use it to play a "
            "interactive program or CLI with scripted input (without it, stdin is empty). no_sleep=true makes Python's time.sleep return at once "
            "(for programs that pause or type text slowly). Long output is trimmed to head+tail and kept whole in .scratch/spill/ (the last line names the file and the range to read_file). "
            "Default timeout 60 s, max 600; background processes are stopped when the command ends.",
            {"command": S, "timeout": {"type": "integer"}, "stdin": S, "no_sleep": {"type": "boolean"}}, ["command"], self.run)
        add("run_tests", "Run the project's test command (optional pytest -k filter).", {"filter": S}, [], self.run_tests)
        add("tui_smoke", "Run an interactive terminal program (curses, TUI, text game) in a real terminal, send keystrokes, get the final screen text and "
            "PASS/FAIL per expect. Use this instead of faking a screen. keys: text, or Enter Esc Tab Space Up Down Left Right PageUp Ctrl-C F1, "
            "wait:<ms>, text:<literal>. expect: what the player must SEE (e.g. game over banner): substrings or re:regex.",
            {"command": S, "keys": {"type": "array", "items": S}, "size": S, "expect": {"type": "array", "items": S},
             "timeout": {"type": "integer"}, "settle_ms": {"type": "integer"}}, ["command"], self.tui_smoke)
        add("lint", "Syntax/lint checks on the files you changed.", {}, [], self.lint)
        add("git_status", "git status (short).", {}, [], self.git_status)
        add("git_diff", "git diff (optionally one path, or staged).", {"path": S, "staged": {"type": "boolean"}}, [], self.git_diff)
        add("git_commit", "Commit all changes with a message (only when the user asked or the repo policy allows).",
            {"message": S}, ["message"], self.git_commit)
        for t in self.web_tools:
            reg.register(t)
        for t in (self.plan_tool, self.note_tool):
            if t is not None:
                reg.register(t)
        return reg


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(str(s))
