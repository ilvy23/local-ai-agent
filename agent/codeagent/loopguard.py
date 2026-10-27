"""Helpers of the engine's loop guard: near-duplicate calls, the outcome of a result, the one-line hints, and the DEAD ENDS ledger.

Seen in real runs (55 `loop` events and 5 aborts in ~3300 steps, nearly all on `run`): the model re-runs slightly edited
`python3 -c "..."` / `printf ... | python3 main.py` variants, or the same failing pytest, and gets the same outcome every time. So:
  * `near_calls`: two `run` / `search` calls are "the same attempt" when the token sets of their texts (digits and quotes ignored)
    overlap by >= NEAR_JACCARD. The engine counts them only when the OUTCOME is also equal (see `outcome_line`),
  * the hints appended to a tool result (a repeat result, a traceback or failing test to open, a piped interactive program),
  * `DeadEnds`: what was tried with the same result, kept in the engine (and the saved session) and shown to a fresh context
    (verified.py, engine.begin_stage, session resume), so a restart after a loop abort does not walk the same dead end again.
  * `FamilyGuard`: counts calls of the same KIND (not the same outcome) and read-only run spam, because a run whose output changes
    a little every time (other regex matches) never counts as a near-duplicate (measured: ~30 `python3 -c` reads of one test file in a row).
  * `PatternGuard`: the same tool with changing arguments, alternating calls, re-reads in a window; `ChantDetector`: repetition inside one
    streamed reply or thought (both after qwen-code's loop detection, see docs/CODE_ENGINE.md 'Ported ideas').
No model call anywhere. Config (coding.*): loop_note (2), loop_repeat (3), loop_abort (4), family_note (5), family_abort (8),
run_spam_note (10), run_spam_abort (14), same_tool_note (8), same_tool_abort (12), alt_note (3), alt_abort (4), read_repeat_note (4),
read_repeat_abort (8), chant_detect, chant_hits (14), chant_think_hits (10); see docs/CODE_ENGINE.md."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

NEAR_JACCARD = 0.8
NEAR_WINDOW = 10                 # a token-set match only counts against the last 10 calls
SAME_RESULT_NOTE = "Same result as your last call; it will not change."
PHASE_NOTE = ("Your files are written and nothing is failing. Stop experimenting: give your final answer now, and say plainly what you could "
              "not verify.")                # the verification phase's note (phase.py), in place of every 'try a different approach'
DEAD_ENDS_HEADER = "DEAD ENDS (already tried, same result - do not repeat):"
DEAD_ENDS_CAP, DEAD_ENDS_CHARS, HEAD_CHARS = 12, 600, 120
INTERACTIVE_HINT = ("Drive the program from a script (import the module and call the functions) instead of piping stdin; "
                    "a piped run only shows the first screens.")
CONTENT_TOOLS = ("read_file", "read_files", "show_symbol")       # their result is file text that the model copies from: no note appended
_FIXED_NOTES = re.compile(rf"\n?(?:{re.escape(SAME_RESULT_NOTE)}|Traceback ends at \S+ - open it with read_file/show_symbol instead of re-running\."
                          r"|Failing test: \S+ - open it with read_file/show_symbol instead of re-running\."
                          rf"|{re.escape(INTERACTIVE_HINT)}|{re.escape(PHASE_NOTE)})")
_TIMING = re.compile(r"\(\d+(?:\.\d+)?s\)")
_TOKEN = re.compile(r"[a-z_][a-z_0-9]*")
_FILE = re.compile(r"[\w./-]+\.(?:py|js|ts|json|txt|md|sh|toml|cfg|ini|yaml|yml|c|cc|cpp|h|rs|go|java|html|css|csv)\b")
_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
_FAILED = re.compile(r"(?m)^FAILED (\S+)")
_NO_DOT = re.compile(r"^(?:\./)+")
_PIPED = re.compile(r"\|\s*(?:python3?|node|ruby|perl)\b")
_FOREIGN = ("site-packages", "dist-packages", "node_modules", "/.venv/", "/venv/", "/usr/lib/", "/lib/python")


def strip_notes(text: str) -> str:
    """A tool result without the one-line hints the guard appended (so earlier copies still compare equal)."""
    return _FIXED_NOTES.sub("", str(text))


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def clip(text: Any, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


# ── near-duplicate calls ────────────────────────────────────────────────────

def call_text(name: str, args: dict[str, Any] | None) -> str:
    """The part of a call that says what it does: the command of `run`, the pattern of `search`, else its arguments."""
    args = args or {}
    for key in ("command", "pattern", "query"):
        if isinstance(args.get(key), str) and args[key].strip():
            return args[key]
    return json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)


def text_of_key(ckey: str | None) -> str | None:
    """call_text from the engine's stored call key (the JSON of the arguments); None for a call restored from a saved session."""
    if ckey is None:
        return None
    try:
        args = json.loads(ckey)
    except ValueError:
        return ckey
    return call_text("", args) if isinstance(args, dict) else ckey


def tokens(text: str) -> frozenset[str]:
    """Lower-case words of a command; quotes, punctuation and bare numbers are dropped (identifiers like `p1` or `test_2` stay): an
    edited `python3 -c "..."` body keeps most of its set."""
    return frozenset(_TOKEN.findall(str(text).lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def near_calls(name: str, a: str | None, b: str | None) -> bool:
    """Only `run` and `search` texts: token-set Jaccard >= NEAR_JACCARD. The caller also requires an equal outcome."""
    return name in ("run", "search") and a is not None and b is not None and jaccard(tokens(a), tokens(b)) >= NEAR_JACCARD


# ── the outcome of a result ─────────────────────────────────────────────────

def outcome_line(result: Any) -> str:
    """The line that says what happened: a pytest `FAILED id`, the final line of a Python traceback, else the first line that is not
    just the exit code (timings removed)."""
    raw = [ln.rstrip() for ln in _TIMING.sub("", str(result)).splitlines() if ln.strip()]
    lines = [ln.strip() for ln in raw]
    for ln in lines:
        if ln.startswith(("FAILED ", "ERROR ")) and "::" in ln:
            return clip(ln, HEAD_CHARS)
    at = max((i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last)")), default=-1)
    if at >= 0:
        tail = [ln for ln in raw[at + 1:] if not ln[0].isspace() and not ln.startswith(("File ", "During handling", "The above exception"))]
        if tail:
            return clip(tail[-1], HEAD_CHARS)
    for ln in lines:
        if not re.match(r"exit code \d+\b", ln):
            return clip(ln, HEAD_CHARS)
    return clip(lines[0], HEAD_CHARS) if lines else ""


def _inside(root: Path, raw: str) -> str | None:
    if any(m in raw for m in _FOREIGN):
        return None
    p = Path(raw)
    try:
        full = p.resolve() if p.is_absolute() else (root / p).resolve()
        rel = full.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return rel.as_posix() if full.is_file() else None


def traceback_end(result: Any, root: Path) -> str | None:
    """`file:line` of the last frame of a Python traceback in `result` when that file is inside the repo."""
    text = str(result)
    if "Traceback (most recent call last)" not in text:
        return None
    frames = _FRAME.findall(text)
    if not frames:
        return None
    raw, line = frames[-1]
    rel = _inside(Path(root), raw)
    return f"{rel}:{line}" if rel else None


def failed_test(result: Any) -> str | None:
    m = _FAILED.search(str(result))
    return m.group(1) if m else None


def error_hint(result: Any, root: Path) -> str | None:
    """The one line for the SECOND identical error (the engine never calls this for the first)."""
    where = traceback_end(result, root)
    if where:
        return f"Traceback ends at {where} - open it with read_file/show_symbol instead of re-running."
    test = failed_test(result)
    if test:
        return f"Failing test: {test} - open it with read_file/show_symbol instead of re-running."
    return None


def piped_prompt_run(args: dict[str, Any] | None, result: Any) -> bool:
    """A program driven by piped stdin whose output is only prompts (no traceback, last line is a prompt)."""
    args = args or {}
    if not (args.get("stdin") or _PIPED.search(str(args.get("command", "")))):
        return False
    text = str(result)
    if "Traceback" in text:
        return False
    lines = [ln for ln in _TIMING.sub("", text).splitlines() if ln.strip() and not re.match(r"exit code \d+", ln.strip())]
    return bool(lines) and bool(re.search(r"[>:?]\s*$", lines[-1]))


def files_of(name: str, args: dict[str, Any] | None, result: Any, root: Path) -> list[str]:
    """Files an attempt concerned (command words that look like files, the failing frame, the failing test's file, the path argument)."""
    args = args or {}
    found = [str(args[k]) for k in ("path", "dst", "dest") if args.get(k)]
    if name in ("run", "tui_smoke"):
        found += _FILE.findall(str(args.get("command", "")))
    where = traceback_end(result, root)
    if where:
        found.append(where.rsplit(":", 1)[0])
    test = failed_test(result)
    if test:
        found.append(test.split("::")[0])
    out: list[str] = []
    for f in found:
        f = _NO_DOT.sub("", f) or f
        if f not in out:
            out.append(f)
    return out[:6]


def attempts_text(rows: list[tuple[str, str]]) -> str:
    """'1. `cmd` -> outcome' lines of the last attempts."""
    return "\n".join(f"{i}. `{clip(cmd, 90)}` -> {clip(out, 90) or '(no output)'}" for i, (cmd, out) in enumerate(rows, 1))


# ── DEAD ENDS ───────────────────────────────────────────────────────────────

def signature(name: str, text: str) -> str:
    """Stable id of an attempt: run / search -> the sorted token set (so an edited variant is the same dead end), else a hash."""
    if name in ("run", "search"):
        return f"{name}:" + " ".join(sorted(tokens(text))[:80])
    return f"{name}:{sha(text)[:12]}"


def _sig_tokens(sig: str) -> frozenset[str]:
    return frozenset(sig.split(":", 1)[1].split()) if sig.startswith(("run:", "search:")) else frozenset()


def _base(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


class DeadEnds:
    """Attempts that were repeated with the same result, newest last. Entry: tool, cmd (<= 120 chars), out (first outcome line,
    <= 120), times, sig, error (the repeated result was an error), files, rsha (hash of the result, for 'same' entries)."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def __bool__(self) -> bool:
        return bool(self.items)

    def clear(self) -> None:
        self.items = []

    def record(self, tool: str, text: str, out: str, times: int, *, error: bool, files: list[str], rsha: str) -> None:
        sig = signature(tool, text)
        old = next((e for e in self.items if e["sig"] == sig), None)
        if old is not None:
            self.items.remove(old)
            times = max(times, int(old.get("times", 0)))
        self.items.append({"tool": tool, "cmd": clip(text, HEAD_CHARS), "out": clip(out, HEAD_CHARS), "times": int(times), "sig": sig,
                           "error": bool(error), "files": list(files)[:6], "rsha": rsha})
        del self.items[:-DEAD_ENDS_CAP]

    def _matches(self, e: dict[str, Any], tool: str, text: str) -> bool:
        if e["tool"] != tool:
            return False
        sig = signature(tool, text)
        return sig == e["sig"] or (tool in ("run", "search") and jaccard(tokens(text), _sig_tokens(e["sig"])) >= NEAR_JACCARD)

    def attempt(self, tool: str, text: str, errorish: bool, rsha: str) -> int:
        """A call was made: a related dead end ends when this attempt no longer fails the same way (an error entry: the call
        succeeded; a 'same' entry: the result changed). Returns how many entries were dropped."""
        keep = [e for e in self.items
                if not (self._matches(e, tool, text) and ((e["error"] and not errorish) or (not e["error"] and e["rsha"] != rsha)))]
        dropped = len(self.items) - len(keep)
        self.items = keep
        return dropped

    def edited(self, path: str) -> int:
        """A file was edited: dead ends that concerned it are stale."""
        if not path:
            return 0
        want = _NO_DOT.sub("", path.replace("\\", "/")) or path
        keep = [e for e in self.items if not any(_base(f) == _base(want) for f in e.get("files") or [])]
        dropped = len(self.items) - len(keep)
        self.items = keep
        return dropped

    def lines(self) -> list[str]:
        """The section, newest first, within DEAD_ENDS_CHARS ([] when empty)."""
        if not self.items:
            return []
        out, size = [DEAD_ENDS_HEADER], len(DEAD_ENDS_HEADER)
        for e in reversed(self.items):
            line = f"  - {e['tool']}: {clip(e['cmd'], 100)} -> {clip(e['out'], 80) or '(no output)'} ({e['times']}x)"
            if size + len(line) + 1 > DEAD_ENDS_CHARS:
                break
            out.append(line)
            size += len(line) + 1
        return out if len(out) > 1 else []

    def text(self) -> str:
        return "\n".join(self.lines())

    def to_json(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.items]

    def load(self, raw: Any) -> None:
        """Safe for an old session (no key) and for damaged values."""
        self.items = []
        for e in raw if isinstance(raw, list) else []:
            if not isinstance(e, dict) or not e.get("tool") or not e.get("sig"):
                continue
            try:
                self.items.append({"tool": str(e["tool"]), "cmd": clip(e.get("cmd", ""), HEAD_CHARS), "out": clip(e.get("out", ""), HEAD_CHARS),
                                   "times": int(e.get("times") or 1), "sig": str(e["sig"]), "error": bool(e.get("error")),
                                   "files": [str(f) for f in (e.get("files") or [])][:6], "rsha": str(e.get("rsha") or "")})
            except (TypeError, ValueError):
                continue
        del self.items[:-DEAD_ENDS_CAP]


# ── command families and read-only run spam ─────────────────────────────────

FAMILY_NOTE, FAMILY_ABORT, FAMILY_WINDOW = 5, 8, 12      # calls of one family among the last 12 tool calls: a note at 5, the segment ends at 8
RUN_SPAM_NOTE, RUN_SPAM_ABORT, RUN_SPAM_WINDOW = 10, 14, 20   # unchanged `run` calls among the last 20 calls without an edit
GUARD_FREE_CALLS = 6             # the first calls of a stage (orientation) never fire either guard
EDIT_GAP_FREE = 6                # fewer than this many calls since the last edit: an edit/test cycle, never counted
_PY = re.compile(r"python[\d.]*$")
_IMPORT = re.compile(r"(?:^|[;\n\s\"'(])import\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)")
_FROM = re.compile(r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\b")
_TEST_CMD = re.compile(r"\b(?:pytest|py\.test|unittest|nosetests|tox)\b|\b(?:npm|yarn|pnpm|cargo|go|make)\s+test\b")
_TEST_SUMMARY = re.compile(r"\d+\s+(?:passed|failed|errors?|skipped|deselected)|^OK\b|^Ran \d+ tests?|^FAILED \(")


def _paths_in_text(text: str) -> list[str]:
    return sorted({_NO_DOT.sub("", f) or f for f in _FILE.findall(text)})


def _smoke_family(command: str) -> str | None:
    """`tui_smoke` family: the program's head word plus the files it names (`tui python3 snake.py`), else its first three words. Twelve smoke
    tests of one program with other keys each time are one family: a program that is RUN is a probe, and the guard counts the probes."""
    command = command.strip()
    if not command:
        return None
    try:
        words = shlex.split(command.splitlines()[0], posix=True)
    except ValueError:
        words = command.splitlines()[0].split()
    words = [w for w in words if not re.match(r"[A-Za-z_]\w*=", w)]
    if not words:
        return None
    paths = _paths_in_text(command)
    return " ".join(["tui", words[0].rsplit("/", 1)[-1], *(paths or words[1:3])])


def family_key(name: str, args: dict[str, Any] | None) -> str | None:
    """The kind of a `run` / `search` call: python -c / heredoc scripts -> interpreter + file names in the text + imported modules; other shell
    -> first word + file/path operands (a script file run as `python3 main.py` has no family: it executes a program, not a look) (numeric ranges, line numbers and quoted patterns ignored); search -> its file/path scope only
    (a whole-repo search has no scope, so no family). None when nothing identifies the target."""
    args = args or {}
    if name == "search":
        scope = [str(args[k]).strip() for k in ("path", "glob") if args.get(k) and str(args[k]).strip() not in ("", ".", "./")]
        return "search " + " ".join(_NO_DOT.sub("", s) or s for s in scope) if scope else None
    if name == "tui_smoke":
        return _smoke_family(str(args.get("command") or ""))
    if name != "run":
        return None
    cmd = str(args.get("command") or "").strip()
    if not cmd:
        return None
    try:
        words = shlex.split(cmd.splitlines()[0], posix=True)
    except ValueError:
        words = cmd.splitlines()[0].split()
    words = [w for w in words if not re.match(r"[A-Za-z_]\w*=", w)]
    if not words:
        return None
    first = words[0].rsplit("/", 1)[-1]
    if _PY.match(first):
        if not (re.search(r"(?:^|\s)-c\b", cmd.splitlines()[0]) or "<<" in cmd or re.search(r"(?:^|\s)-(?:\s|$)", cmd.splitlines()[0])):
            return None                     # `python3 main.py` runs a program (often with new input each time): only run_spam watches that
        mods = sorted({m.split(".")[0] for g in _IMPORT.findall(cmd) for m in re.split(r"\s*,\s*", g)} | {m.split(".")[0] for m in _FROM.findall(cmd)})
        paths = _paths_in_text(cmd)
        if not paths and not mods:
            return None
        return " ".join([first, *paths] + ([f"[import {','.join(mods)}]"] if mods else []))
    ops = []
    for w in words[1:]:
        if w.startswith("-") or w in ("|", "||", "&&", ";", ">", ">>"):
            continue
        if _FILE.fullmatch(w) or (("/" in w) and re.fullmatch(r"[\w./~-]+", w)):
            ops.append(_NO_DOT.sub("", w) or w)
    if not ops:
        return None
    return " ".join([first, *sorted(set(ops))])


def is_test_command(command: Any) -> bool:
    return bool(_TEST_CMD.search(str(command or "")))


def test_summary(result: Any) -> str:
    """The summary line of a test run (`3 failed, 10 passed in 1.2s`, timings removed), else its last line."""
    lines = [_TIMING.sub("", ln).strip() for ln in str(result).splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _TEST_SUMMARY.search(ln):
            return re.sub(r"\s+in\s+[\d.]+s?\b.*$", "", ln)
    return lines[-1] if lines else ""


class FamilyGuard:
    """Two deterministic counters over the last tool calls, both reset by a successful edit (the only real progress):
      * family: >= FAMILY_NOTE calls of one `family_key` among the last FAMILY_WINDOW -> a note; >= FAMILY_ABORT -> abort,
      * run_spam: >= RUN_SPAM_NOTE `run` calls that changed nothing among the last RUN_SPAM_WINDOW -> a note; >= RUN_SPAM_ABORT -> abort.
    A test run whose summary differs from the previous test run's is real information and counts for neither. Nothing fires in the
    first GUARD_FREE_CALLS calls of a stage or within EDIT_GAP_FREE calls after an edit. State is not saved (a resumed stage starts clean)."""

    def __init__(self) -> None:
        self.reset_stage()

    def reset_stage(self) -> None:
        self.calls: list[tuple[str | None, bool]] = []
        self.stage_calls = 0
        self.since_edit: int | None = None
        self.noted: set[str] = set()
        self.last_summary: str | None = None

    def edited(self) -> None:
        self.calls = []
        self.noted = set()
        self.since_edit = 0

    def observe(self, name: str, args: dict[str, Any] | None, result: Any, *, changed: bool,
                note_at: int = FAMILY_NOTE, abort_at: int = FAMILY_ABORT,
                spam_note: int = RUN_SPAM_NOTE, spam_abort: int = RUN_SPAM_ABORT) -> list[tuple[str, str, str, int, str]]:
        """One finished non-edit call. Returns events (action 'note' | 'abort', shape 'family' | 'run_spam', family, count, text)."""
        self.stage_calls += 1
        if self.since_edit is not None:
            self.since_edit += 1
        fam: str | None = family_key(name, args)
        spam = name == "run" and not changed
        if name == "run" and is_test_command((args or {}).get("command")):
            summary = test_summary(result)
            if self.last_summary is not None and summary != self.last_summary:
                fam, spam = None, False                  # the verdict moved (green -> red ...): information, not circling
            self.last_summary = summary
        self.calls.append((fam, spam))
        del self.calls[:-RUN_SPAM_WINDOW]
        if self.stage_calls <= GUARD_FREE_CALLS or (self.since_edit is not None and self.since_edit < EDIT_GAP_FREE):
            return []
        out: list[tuple[str, str, str, int, str]] = []
        if fam is not None:
            n = sum(1 for f, _ in self.calls[-FAMILY_WINDOW:] if f == fam)
            abort_at = max(abort_at, note_at)
            if n >= abort_at:
                out.append(("abort", "family", fam, n, ""))
            elif n >= note_at and ("family:" + fam) not in self.noted:
                self.noted.add("family:" + fam)
                out.append(("note", "family", fam, n, f"Same kind of command as your last {note_at - 1} ({fam}); the output keeps changing but you are "
                                                       "not acting on it. Make an edit now, or try a different approach."))
        if spam:
            n = sum(1 for _, sp in self.calls if sp)
            spam_abort = max(spam_abort, spam_note)
            if n >= spam_abort:
                out.append(("abort", "run_spam", "read-only runs", n, ""))
            elif n >= spam_note and "run_spam" not in self.noted:
                self.noted.add("run_spam")
                out.append(("note", "run_spam", "read-only runs", n, f"{n} of your last {len(self.calls)} calls were commands that changed nothing and "
                                                                     "you made no edit; stop investigating. Make an edit now, or try a different approach."))
        return out


# ── patterns over the last calls: same tool, alternation, re-reads ──────────
# Ideas taken from qwen-code's loopDetectionService (Apache-2.0), reimplemented here; see docs/CODE_ENGINE.md, 'Ported ideas'.

SAME_TOOL_NOTE, SAME_TOOL_ABORT = 8, 12          # the same tool name in a row (changing arguments, no edit): a note at 8, the segment ends at 12
ALT_NOTE, ALT_ABORT = 3, 4                       # rounds of an A-B-A-B / A-B-C-A-B-C pattern (identical calls, any outcome)
READ_NOTE, READ_ABORT, READ_WINDOW = 4, 8, 15    # re-reads of text already read among the last 15 calls, no edit in between
PATTERN_WINDOW = 16
_WHOLE = 10 ** 9
_READ_SPAN = 240                                 # a read_file without end shows about this many lines
_WROTE_OK = ("wrote ", "edited ", "appended ", "patched ", "deleted ", "moved ")
_ACTS_NEUTRAL = frozenset({"mark_done", "project_note", "add_check", "ask_user", "write_prose"})
_WRITES = frozenset({"write_file", "append_file", "edit_file", "apply_patch", "delete_file", "move_file", "replace_lines", "insert_after"})


def read_spans(name: str, args: dict[str, Any] | None) -> list[tuple[str, int, int]] | None:
    """What a read call shows, as (target, first line, last line); None for every other tool."""
    args = args or {}
    path = str(args.get("path") or "")
    if name == "read_file" and path:
        try:
            lo = max(1, int(args.get("start") or 1))
            hi = int(args["end"]) if args.get("end") else lo + _READ_SPAN - 1
        except (TypeError, ValueError):
            lo, hi = 1, _WHOLE
        return [(path, lo, max(lo, hi))]
    if name == "read_files" and isinstance(args.get("paths"), list) and args["paths"]:
        return [(str(p), 1, _WHOLE) for p in args["paths"]]
    if name == "outline" and path:
        return [(f"outline {path}", 1, _WHOLE)]
    if name == "show_symbol" and args.get("name"):
        return [(f"{path}::{args['name']}", 1, _WHOLE)]
    return None


def _overlaps(span: tuple[str, int, int], seen: list[tuple[str, int, int]]) -> bool:
    return any(t == span[0] and lo <= span[2] and span[1] <= hi for t, lo, hi in seen)


class PatternGuard:
    """Three deterministic counters that need no equal outcome, all reset by a successful edit (the only real progress):
      * same_tool: the same tool name SAME_TOOL_NOTE times in a row with changing arguments -> a note; SAME_TOOL_ABORT -> abort. For the
        read tools the streak is keyed on (tool, path) and only counts a call whose lines overlap text it already read in the streak
        (twelve different files, or one big file read in distinct ranges, is exploration). A successful write, a command that changed
        files and a test run with a new summary end the streak,
      * alternating: the last 2*ALT_NOTE (A-B) or 3*ALT_NOTE (A-B-C) calls repeat one pattern of identical calls -> a note; ALT_ABORT rounds -> abort,
      * read_window: READ_NOTE re-reads of already-read text among the last READ_WINDOW calls -> a note; READ_ABORT -> abort.
    Nothing fires in the first GUARD_FREE_CALLS calls of a stage or within EDIT_GAP_FREE calls after an edit; state is not saved.
    `look_only` marks a read-only shell command: it does not count toward same_tool (run_spam owns it). `quiet=True` (another guard already spoke on this call) drops the notes; the aborts always stand. Never forces thinking."""

    def __init__(self) -> None:
        self.reset_stage()

    def reset_stage(self) -> None:
        self.stage_calls = 0
        self.since_edit: int | None = None
        self.last_summary: str | None = None
        self.edited()
        self.since_edit = None

    def edited(self) -> None:
        self.since_edit = 0
        self.noted: set[str] = set()
        self.last_note = -99
        self.keys: list[str] = []                       # identity of the last calls, for alternation
        self.labels: list[str] = []                     # what they were, for the note
        self.recent: list[list[tuple[str, int, int]] | None] = []   # read spans of the last READ_WINDOW calls
        self.reread: list[bool] = []
        self.streak_key: str | None = None
        self.streak_n = 0
        self.streak_seen: list[tuple[str, int, int]] = []

    def _end_streak(self) -> None:
        self.streak_key, self.streak_n, self.streak_seen = None, 0, []

    def observe(self, name: str, args: dict[str, Any] | None, result: Any, *, changed: bool, quiet: bool = False, look_only: bool = False,
                same_note: int = SAME_TOOL_NOTE, same_abort: int = SAME_TOOL_ABORT, alt_note: int = ALT_NOTE, alt_abort: int = ALT_ABORT,
                read_note: int = READ_NOTE, read_abort: int = READ_ABORT) -> list[tuple[str, str, str, int, str]]:
        """One finished call that was not a real edit. Returns events (action 'note' | 'abort', shape 'same_tool' | 'alternating' |
        'read_window', key text, count, note text)."""
        self.stage_calls += 1
        if self.since_edit is not None:
            self.since_edit += 1
        args = args or {}
        resets = changed or name in _ACTS_NEUTRAL
        if name in _WRITES and str(result).startswith(_WROTE_OK) and "SYNTAX GATE FAILED" not in str(result):
            resets = True                                # a scratch write: no progress, but no streak either
        if name == "run" and is_test_command(args.get("command")):
            summary = test_summary(result)
            if self.last_summary is not None and summary != self.last_summary:
                resets = True                            # the verdict moved: information, not circling
            self.last_summary = summary
        spans = read_spans(name, args)
        window = self.recent[-(READ_WINDOW - 1):]
        again = spans is not None and all(_overlaps(s, [x for r in window if r for x in r]) for s in spans)
        self.recent = [*window, spans]
        self.reread = [*self.reread[-(READ_WINDOW - 1):], again]
        if resets:
            self._end_streak()
            self.keys, self.labels = [], []
        else:
            self.keys.append(f"{name}:{signature(name, call_text(name, args))}")
            self.labels.append(f"{name} {clip(call_text(name, args), 40)}")
            del self.keys[:-PATTERN_WINDOW], self.labels[:-PATTERN_WINDOW]
            # same tool: content tools are keyed by (tool, path); the others by name
            key = f"{name} {spans[0][0]}" if spans else name
            if not look_only:                              # a read-only shell command is run_spam's business (its own note and stop): neutral here
                if key != self.streak_key:
                    self._end_streak()
                    self.streak_key = key
                if spans is None or _overlaps(spans[0], self.streak_seen):
                    self.streak_n += 1
            self.streak_seen += spans or []
        if self.stage_calls <= GUARD_FREE_CALLS or (self.since_edit is not None and self.since_edit < EDIT_GAP_FREE):
            return []
        found: list[tuple[str, str, str, int, str]] = []
        if not resets and not look_only and self.streak_n >= 1:
            n = self.streak_n
            label = self.streak_key or name
            same_abort = max(same_abort, same_note)
            if n >= same_abort:
                found.append(("abort", "same_tool", label, n, ""))
            elif n >= same_note:
                found.append(("note", "same_tool", label, n, f"`{name}` {n} times in a row"
                              + (f" on {spans[0][0]}" if spans else " with changing arguments")
                              + " and no edit in between; more of the same will not answer it. Make an edit now, or try a different approach."))
        alt = self._alternation(alt_note, max(alt_abort, alt_note)) if not resets else None
        if alt is not None:
            action, rounds, texts = alt
            found.append((action, "alternating", " <-> ".join(texts), rounds,
                          f"You are alternating between {len(texts)} calls ({'; '.join(texts)}), {rounds} rounds without an edit; "
                          "the answer will not change. Make an edit now, or try a different approach."))
        if again and not resets:
            n = sum(self.reread)
            read_abort = max(read_abort, read_note)
            if n >= read_abort:
                found.append(("abort", "read_window", "re-reads", n, ""))
            elif n >= read_note:
                found.append(("note", "read_window", "re-reads", n, f"{n} of your last {len(self.reread)} calls re-read text you already have and "
                              "you made no edit; use what is in your context and make an edit now."))
        out: list[tuple[str, str, str, int, str]] = []
        for act, shape, label, n, text in found:
            if act == "note":
                token = f"{shape}:{label}"
                if quiet or token in self.noted or self.stage_calls - self.last_note < 4:
                    continue
                self.noted.add(token)
                self.last_note = self.stage_calls
            out.append((act, shape, label, n, text))
        return out

    def _alternation(self, note: int, abort: int) -> tuple[str, int, list[str]] | None:
        """('note' | 'abort', rounds, the distinct calls) when the tail of the call history is a pattern of 2 or 3 identical calls repeated."""
        best: tuple[str, int, list[str]] | None = None
        for p in (2, 3):
            keys = self.keys
            run = p                                       # length of the periodic suffix
            while run < len(keys) and keys[-1 - run] == keys[-1 - run + p]:
                run += 1
            tail = keys[-run:] if run <= len(keys) else keys
            rounds = run // p
            if run >= p * note and len(set(keys[-p:])) == p and len(set(tail)) == p:
                action = "abort" if rounds >= abort else "note"
                if best is None or rounds > best[1]:
                    best = (action, rounds, self.labels[-p:])
        return best


# ── repetition inside one streamed generation ───────────────────────────────

CHANT_CHUNK, CHANT_WINDOW, CHANT_GAP = 50, 4000, 75      # a 50-char chunk, looked up in the last 4000 chars; units up to ~75 chars repeat 'clustered'
CHANT_HITS = {"content": 14, "thinking": 10}             # occurrences of the chunk at <= CHANT_GAP spacing; text in a tool call gets more room than thinking
CHANT_DISTINCT = {"content": 8, "thinking": 5}           # a chunk of fewer distinct characters (dividers, zero rows, blank lines) is never a chant
CHANT_REGION = {"content": 2000, "thinking": 1000}       # a longer unit must be exactly periodic over at least this many chars ...
CHANT_PERIODS = {"content": (5, 4), "thinking": (5, 3)}  # ... after this many occurrences (units shorter than CHANT_LONG chars, then longer ones)
CHANT_LONG, CHANT_EVERY = 800, 100


class ChantDetector:
    """Early stop of a generation that chants: fed every streamed piece of ONE kind ('content' or 'thinking'), it looks every CHANT_EVERY chars
    at the last CHANT_CHUNK chars and where else they occur in the last CHANT_WINDOW chars:
      * clustered: >= hits occurrences in a row, each within CHANT_GAP of the previous one (a short unit said over and over),
      * periodic: occurrences at one equal stride (a long unit repeated) AND the whole stretch is exactly periodic.
    The older `engine.repeating` (one block over 8000 chars) stays as the backstop for units longer than the window. feed() returns the
    reason ('clustered' | 'periodic') or None. No model call."""

    def __init__(self, kind: str = "content", hits: int | None = None) -> None:
        self.kind = kind if kind in CHANT_HITS else "content"
        self.hits = max(2, int(hits)) if hits else CHANT_HITS[self.kind]
        self.text = ""
        self.fresh = 0

    def feed(self, piece: str) -> str | None:
        self.text += piece
        self.fresh += len(piece)
        if len(self.text) > CHANT_WINDOW + 1000:
            self.text = self.text[-CHANT_WINDOW:]
        if self.fresh < CHANT_EVERY:
            return None
        self.fresh = 0
        return self._look()

    def _look(self) -> str | None:
        win = self.text[-CHANT_WINDOW:]
        tail = win[-CHANT_CHUNK:]
        if len(win) < 2 * CHANT_CHUNK or len(set(tail)) < CHANT_DISTINCT[self.kind]:
            return None
        pos, i = [], win.find(tail)
        while i >= 0:
            pos.append(i)
            i = win.find(tail, i + 1)
        if len(pos) < 3:
            return None
        run = 1
        for a, b in zip(reversed(pos[:-1]), reversed(pos[1:])):
            if b - a > CHANT_GAP:
                break
            run += 1
        if run >= self.hits:
            return "clustered"
        stride = pos[-1] - pos[-2]
        if stride <= CHANT_GAP:
            return None
        run, j = 2, len(pos) - 2
        while j > 0 and pos[j] - pos[j - 1] == stride:
            run += 1
            j -= 1
        short_need, long_need = CHANT_PERIODS[self.kind]
        region = (run - 1) * stride + CHANT_CHUNK
        if run >= (long_need if stride >= CHANT_LONG else short_need) and region >= CHANT_REGION[self.kind] and region <= len(win):
            seg = win[-region:]
            if seg[stride:] == seg[:-stride]:
                return "periodic"
        return None
