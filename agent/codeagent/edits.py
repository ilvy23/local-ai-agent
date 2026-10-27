"""Pure helpers for editing code safely: syntax gates, tolerant search/replace, unified-diff apply, outlines.

Nothing here touches the model or the policy layer - these are the building blocks the code tools use so a small
local model gets (a) edits that survive whitespace slips, (b) an immediate, precise syntax error instead of a later
mystery, and (c) cheap structural views of a file instead of re-reading all of it.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── duplicate blocks ────────────────────────────────────────────────────────

DUP_MIN_FILE = 40          # shorter files are not checked
DUP_RUN = 8                # lines in a row that must repeat verbatim
DUP_SHORT = 12             # a line shorter than this (stripped) is "short" ...
DUP_SHORT_SHARE = 0.6      # ... and a run of which at least this share is short is ignored (`"",` / `}` / `),` boilerplate)
_TRIVIAL = re.compile(r"^[\s{}\[\]()<>,;:'\"`*#/\\-]*$")          # only brackets, quotes, comment markers: carries no content


def _significant(line: str) -> str:
    """The normalised line (stripped), or '' for a blank / trivial one."""
    s = line.strip()
    return "" if _TRIVIAL.match(s) else s


def duplicate_block(before: str | None, after: str) -> tuple[int, int, int, int, int] | None:
    """(n, a, b, c, d): of the lines an edit introduced (`after` minus the unchanged head and tail it shares with `before`), a run of at
    least DUP_RUN non-blank, non-trivial lines at a-b (n lines) also occurs verbatim at c-d outside that region. None when there is none, the
    file is short, or nothing was added (a delete). Language neutral, O(n): the lines are compared as DUP_RUN-grams in a dict, so a moved block
    (its old place is gone) and a replaced block (the old text is part of the region) are never a duplicate of themselves."""
    new = after.split("\n")
    if len(new) < DUP_MIN_FILE:
        return None
    old = before.split("\n") if before else []
    head = 0
    limit = min(len(old), len(new))
    while head < limit and old[head] == new[head]:
        head += 1
    tail = 0
    while tail < limit - head and old[len(old) - 1 - tail] == new[len(new) - 1 - tail]:
        tail += 1
    lo, hi = head, len(new) - tail                      # new[lo:hi] is what the edit wrote
    if hi - lo < DUP_RUN:
        return None
    outside: dict[tuple[str, ...], tuple[int, int]] = {}
    for seg_lo, seg_hi in ((0, lo), (hi, len(new))):
        rows = [(i, _significant(new[i])) for i in range(seg_lo, seg_hi)]
        rows = [(i, t) for i, t in rows if t]
        for k in range(len(rows) - DUP_RUN + 1):
            gram = tuple(t for _i, t in rows[k:k + DUP_RUN])
            outside.setdefault(gram, (rows[k][0], rows[k + DUP_RUN - 1][0]))
    if not outside:
        return None
    rows = [(i, _significant(new[i])) for i in range(lo, hi)]
    rows = [(i, t) for i, t in rows if t]
    best: tuple[int, int, int, int, int] | None = None
    run: list[tuple[int, int, int, int]] = []           # consecutive matching grams: (new start, new end, old start, old end)
    for k in range(len(rows) - DUP_RUN + 1):
        gram = tuple(t for _i, t in rows[k:k + DUP_RUN])
        hit = outside.get(gram)
        if hit is not None and sum(len(t) < DUP_SHORT for t in gram) < DUP_SHORT_SHARE * DUP_RUN:
            run.append((rows[k][0], rows[k + DUP_RUN - 1][0], hit[0], hit[1]))
        else:
            run = []
            continue
        a, b, c, d = run[0][0], run[-1][1], run[0][2], run[-1][3]
        if best is None or b - a + 1 > best[0]:
            best = (b - a + 1, a + 1, b + 1, c + 1, d + 1)
    return best


def duplicate_text(n: int, a: int, b: int, c: int, d: int) -> str:
    return (f"WARNING: the {n} lines you just wrote at {a}-{b} duplicate lines {c}-{d} that are still in the file. "
            "If you meant to replace that block, your end line was too small; fix it now.")


# ── syntax gates ────────────────────────────────────────────────────────────

SYNTAX_CONTEXT = 2


def _excerpt(lines: list[str], lineno: int, col: int | None = None) -> str:
    lo, hi = max(1, lineno - SYNTAX_CONTEXT), min(len(lines), lineno + SYNTAX_CONTEXT)
    out = []
    for n in range(lo, hi + 1):
        mark = ">>" if n == lineno else "  "
        out.append(f"{mark} {n:4} | {lines[n - 1][:200]}")
        if n == lineno and col:
            out.append("        | " + " " * max(col - 1, 0) + "^")
    return "\n".join(out)


def check_syntax(path: str, text: str, check_cmd: str | None = None) -> str | None:
    """None when the text is fine (or the type is not checked), else a precise error with surrounding lines.

    Python, JSON and TOML (and YAML when PyYAML is importable) use the standard library; shell, JS, TS, Rust, Go, C and C++
    use the compiler/parser if it is installed. A missing or hanging tool is NEVER an error: the file is simply not checked.
    `check_cmd` (config coding.check_cmd, a command with {file}) replaces the built-in external check of a project."""
    suffix = Path(path).suffix.lower()
    lines = text.splitlines() or [""]
    if check_cmd and suffix in custom_suffixes():
        return _custom_syntax(path, text, suffix, check_cmd)
    try:
        if suffix in (".py", ".pyi"):
            ast.parse(text, filename=path)
        elif suffix == ".json":
            json.loads(text)
        elif suffix == ".toml":
            tomllib.loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:                     # no PyYAML here: not checked, not an error
                return None
            yaml.safe_load(text)
        elif suffix in _EXTERNAL:
            return _EXTERNAL[suffix](path, text, suffix)
        else:
            return None
    except SyntaxError as exc:
        lineno = exc.lineno or 1
        hint = ""
        if "unterminated string" in str(exc) or "was never closed" in str(exc) or "invalid character" in str(exc):
            hint = ("\nHint: this usually means a quote or bracket inside a string was not escaped. In a JSON tool "
                    "argument a Python backslash must be doubled (\\\\n for a newline inside a string literal); "
                    "prefer edit_file for small changes and put long text in triple-quoted strings.")
        return f"SyntaxError in {path}: {exc.msg} (line {lineno})\n{_excerpt(lines, lineno, exc.offset)}{hint}"
    except json.JSONDecodeError as exc:
        return f"JSON error in {path}: {exc.msg} (line {exc.lineno}, col {exc.colno})\n{_excerpt(lines, exc.lineno, exc.colno)}"
    except tomllib.TOMLDecodeError as exc:
        return f"TOML error in {path}: {exc}"
    except Exception as exc:  # noqa: BLE001 - yaml.YAMLError and friends
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        return f"Syntax error in {path}{where}: {str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__}"
    return None


def _run_tool(argv: list[str], *, text: str | None = None, suffix: str = "", timeout: int = 15, cwd: str | None = None):
    """Run a checker on `text` (stdin when `suffix` is empty, else a temp file whose name replaces the {} item of argv).
    Returns the CompletedProcess, or None when the tool cannot run (missing, hangs, crashes) - callers treat that as 'fine'."""
    name = None
    try:
        if suffix:
            with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8", errors="replace") as f:
                f.write(text or "")
                name = f.name
            argv = [name if a == "{}" else a for a in argv]
            return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return subprocess.run(argv, input=text, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    finally:
        if name:
            try:
                os.unlink(name)
            except OSError:
                pass


def _short(r: Any, limit: int = 400) -> str:
    return ((r.stderr or "").strip() + "\n" + (r.stdout or "").strip()).strip()[:limit]


def _shell_syntax(path: str, text: str, suffix: str = "") -> str | None:
    bash = shutil.which("bash")
    if not bash:
        return None
    r = _run_tool([bash, "-n"], text=text, timeout=10)
    return None if r is None or r.returncode == 0 else f"Shell syntax error: {r.stderr.strip()[:300]}"


def _node_syntax(path: str, text: str, suffix: str) -> str | None:
    node = shutil.which("node")
    if not node:
        return None
    r = _run_tool([node, "--check", "{}"], text=text, suffix=suffix)
    return None if r is None or r.returncode == 0 else f"JavaScript syntax error: {r.stderr.strip()[:400]}"


def _local_bin(path: str, name: str) -> str | None:
    """`name` from PATH, else from the closest node_modules/.bin above `path`."""
    found = shutil.which(name)
    if found:
        return found
    try:
        for parent in list(Path(path).resolve().parents)[:8]:
            cand = parent / "node_modules" / ".bin" / name
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    except OSError:
        pass
    return None


_TS_SYNTAX = re.compile(r"error TS1[01]\d\d:")           # TS1000-1199 are syntax errors; the rest (missing modules, types) is noise here


def _ts_syntax(path: str, text: str, suffix: str) -> str | None:
    tsc = _local_bin(path, "tsc")
    if not tsc:
        return None
    r = _run_tool([tsc, "--noEmit", "--skipLibCheck", "--jsx", "preserve", "--target", "esnext", "{}"], text=text, suffix=suffix, timeout=30)
    if r is None:
        return None
    hits = [ln for ln in (r.stdout + "\n" + r.stderr).splitlines() if _TS_SYNTAX.search(ln)]
    return (f"TypeScript syntax error in {path}: " + "\n".join(h[:200] for h in hits[:4])) if hits else None


def _rust_syntax(path: str, text: str, suffix: str) -> str | None:
    fmt = shutil.which("rustfmt")
    if not fmt:
        return None
    r = _run_tool([fmt, "--check", "--edition", "2021"], text=text)
    if r is None or r.returncode == 0:
        return None
    out = _short(r, 600)
    if "Diff in" in out or "error" not in out.lower():           # exit 1 for plain formatting differences: not a syntax error
        return None
    return f"Rust syntax error in {path}: {out[:400]}"


def _go_syntax(path: str, text: str, suffix: str) -> str | None:
    gofmt = shutil.which("gofmt")
    if not gofmt:
        return None
    r = _run_tool([gofmt, "-e"], text=text)
    return None if r is None or r.returncode == 0 else f"Go syntax error in {path}: {(r.stderr or r.stdout).strip()[:400]}"


def _c_syntax(path: str, text: str, suffix: str) -> str | None:
    cpp = suffix in (".cc", ".cpp", ".cxx")
    cc = next((shutil.which(n) for n in (("c++", "g++", "clang++") if cpp else ("cc", "gcc", "clang")) if shutil.which(n)), None)
    if not cc:
        return None
    here = str(Path(path).parent) if Path(path).is_absolute() else "."
    r = _run_tool([cc, "-fsyntax-only", "-x", "c++" if cpp else "c", f"-I{here}", "-"], text=text, timeout=20)
    if r is None or r.returncode == 0:
        return None
    out = _short(r, 1500)
    if "No such file or directory" in out:                        # a header we cannot see: the rest of the errors prove nothing
        return None
    errs = [ln for ln in out.splitlines() if "error" in ln]
    return (f"C/C++ syntax error in {path}: " + "\n".join(e[:200] for e in errs[:4])) if errs else None


_EXTERNAL = {".sh": _shell_syntax, ".bash": _shell_syntax, ".js": _node_syntax, ".mjs": _node_syntax, ".cjs": _node_syntax,
             ".ts": _ts_syntax, ".tsx": _ts_syntax, ".mts": _ts_syntax, ".cts": _ts_syntax, ".rs": _rust_syntax, ".go": _go_syntax,
             ".c": _c_syntax, ".cc": _c_syntax, ".cpp": _c_syntax, ".cxx": _c_syntax}
# Java is deliberately not checked by default: javac cannot judge a single file (classpath, public class = file name); set coding.check_cmd.


def custom_suffixes() -> set[str]:
    """File types coding.check_cmd applies to: every code file except Python (own gate), data and markup."""
    return CODE_SUFFIXES - {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".html", ".css", ".sql", ".nix"}


def custom_cmd(check_cmd: str, file: str) -> str | None:
    """The configured command for one file, or None when its first word is not an installed tool."""
    try:
        words = shlex.split(check_cmd)
    except ValueError:
        return None
    if not words or not shutil.which(words[0]):
        return None
    return check_cmd.replace("{file}", shlex.quote(file))


def _custom_syntax(path: str, text: str, suffix: str, check_cmd: str) -> str | None:
    """Run the project's own check on a temp copy of the new text; a missing tool or a timeout means 'not checked'."""
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8", errors="replace") as f:
            f.write(text)
            name = f.name
        cmd = custom_cmd(check_cmd, name)
        if cmd is None:
            return None
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    finally:
        if name:
            try:
                os.unlink(name)
            except OSError:
                pass
    if r.returncode == 0 or r.returncode in (126, 127):
        return None
    return f"Syntax error in {path}: {_short(r, 500).replace(name, path)}"


# ── tolerant search/replace ─────────────────────────────────────────────────

@dataclass
class EditResult:
    text: str | None
    how: str = ""
    error: str = ""


def _norm(line: str) -> str:
    return " ".join(line.split())


def _find_lines(src_lines: list[str], search_lines: list[str], norm=_norm) -> list[int]:
    """Start indexes where search_lines match src_lines ignoring indentation/whitespace differences (and whatever else `norm` folds)."""
    want = [norm(x) for x in search_lines]
    while want and not want[0]:
        want.pop(0)
        search_lines = search_lines[1:]
    while want and not want[-1]:
        want.pop()
    if not want:
        return []
    have = [norm(x) for x in src_lines]
    hits = []
    for i in range(len(have) - len(want) + 1):
        if have[i:i + len(want)] == want:
            hits.append(i)
    return hits


# Typographic variants a model emits for the characters of a text (clean-room version of the "ignore typographic quote / dash / space
# variants" level of opencode's edit tool): curly quotes, dashes, the ellipsis character and the space variants.
_TYPO = {**{c: "'" for c in "\u2018\u2019\u201a\u201b\u2032"}, **{c: '"' for c in "\u201c\u201d\u201e\u201f\u2033"},
         **{c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}, "\u2026": "...",
         **{c: " " for c in "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"}}
_TYPO_RX = re.compile("[" + "".join(_TYPO) + "]")
HOW_TYPO = "matched ignoring typographic quotes/dashes"
HOW_ANCHOR = "matched by its first and last line, the middle lines differ"
ANCHOR_MIN_LINES, ANCHOR_MIN_RATIO, ANCHOR_MIN_CHARS = 3, 0.7, 8


APPLIED_WINDOW, APPLIED_RATIO, APPLIED_MIN_CHARS = 60, 0.98, 12


def find_applied(lines: list[str], new: list[str], near: int, window: int = APPLIED_WINDOW) -> tuple[int, int] | None:
    """(first, last) 1-based line of the block in `lines` that already holds the text `new` (first and last line equal after whitespace
    folding, the whole block >= 0.98 similar), within `window` lines of `near` - the one closest to `near`; None when there is none.
    A replacement with under 12 characters of content never counts (a blank line or a brace would match anywhere)."""
    want = [_norm(x) for x in new]
    while want and not want[0]:
        want.pop(0)
    while want and not want[-1]:
        want.pop()
    if not want or sum(len(x) for x in want) < APPLIED_MIN_CHARS:
        return None
    have = [_norm(x) for x in lines]
    n, joined = len(want), "\n".join(want)
    best: tuple[int, int] | None = None
    for i in range(max(0, near - 1 - window), min(len(have) - n, near - 1 + window) + 1):
        if have[i] != want[0] or have[i + n - 1] != want[-1]:
            continue
        block = have[i:i + n]
        if block != want and difflib.SequenceMatcher(None, "\n".join(block), joined).ratio() < APPLIED_RATIO:
            continue
        if best is None or abs(i + 1 - near) < abs(best[0] - near):
            best = (i + 1, i + n)
    return best


def _typo(text: str) -> str:
    return _TYPO_RX.sub(lambda m: _TYPO[m.group()], text)


def _typo_norm(line: str) -> str:
    return _norm(_typo(line))


def _typo_text(text: str) -> tuple[str, list[int]]:
    """The text with the variants folded, and for every folded character the index of the source character it came from."""
    out, idx = [], []
    for i, ch in enumerate(text):
        rep = _TYPO.get(ch, ch)
        out.append(rep)
        idx += [i] * len(rep)
    return "".join(out), idx


def _typo_spans(src: str, search: str) -> list[tuple[int, int]]:
    """Source spans (start, end) whose folded text equals the folded `search` (overlaps counted, so a repeated pattern is ambiguous)."""
    folded, idx = _typo_text(src)
    want = _typo(search)
    spans, k = [], folded.find(want)
    while k != -1:
        e = k + len(want)
        whole = (k == 0 or idx[k] != idx[k - 1]) and (e >= len(folded) or idx[e] != idx[e - 1])     # never cut a character that was widened
        if whole:
            spans.append((idx[k], idx[e - 1] + 1))
        k = folded.find(want, k + 1)
    return spans


def _anchor_blocks(src_lines: list[str], search_lines: list[str]) -> list[tuple[int, int, float]]:
    """(first line, end line exclusive, similarity of the middle) of every block that starts and ends with the search's first and
    last line (folded) and whose middle is similar enough. The block may be a little longer or shorter than the search."""
    want = [_typo_norm(x) for x in search_lines]
    while want and not want[0]:
        want.pop(0)
    while want and not want[-1]:
        want.pop()
    if len(want) < ANCHOR_MIN_LINES or len(want[0]) + len(want[-1]) < ANCHOR_MIN_CHARS:
        return []
    have = [_typo_norm(x) for x in src_lines]
    mid_want = "\n".join(want[1:-1])
    found = []
    for i, line in enumerate(have):
        if line != want[0]:
            continue
        for j in range(i + 2, min(len(have), i + 2 * len(want))):
            if have[j] == want[-1]:
                ratio = difflib.SequenceMatcher(None, "\n".join(have[i + 1:j]), mid_want, autojunk=False).ratio()
                if ratio >= ANCHOR_MIN_RATIO:
                    found.append((i, j + 1, ratio))
                break                                                   # the nearest closing line only
    return found


def _reindent(replace: str, search: str, src_block: str) -> str:
    """Shift `replace` so its indentation matches the file when the model's search block was indented differently."""
    def lead(s: str) -> int:
        for line in s.splitlines():
            if line.strip():
                return len(line) - len(line.lstrip())
        return 0
    delta = lead(src_block) - lead(search)
    if delta == 0:
        return replace
    out = []
    for line in replace.splitlines(keepends=True):
        if not line.strip():
            out.append(line)
        elif delta > 0:
            out.append(" " * delta + line)
        else:
            cut = min(-delta, len(line) - len(line.lstrip()))
            out.append(line[cut:])
    return "".join(out)


def _lines_of(src: str, starts: list[int]) -> str:
    return ", ".join(str(src.count("\n", 0, k) + 1) for k in starts[:6])


def _splice_lines(src_lines: list[str], search: str, search_lines: list[str], replace: str, hits: list[int]) -> str:
    """Replace the line blocks that start at `hits` (indexes into src_lines) by `replace`, re-indented to the file."""
    new = src_lines
    for start in reversed(hits):
        count = len([x for x in search_lines if x.strip()])
        # span of the matched block in the source (skipping the blank edge lines the search may carry)
        end = start
        seen = 0
        while end < len(new) and seen < count:
            if new[end].strip():
                seen += 1
            end += 1
        block = "".join(new[start:end])
        rep = _reindent(replace, search, block)
        if block.endswith("\n") and rep and not rep.endswith("\n"):
            rep += "\n"
        new = new[:start] + [rep] + new[end:]
    return "".join(new)


def _ambiguous(how: str, src: str, starts: list[int]) -> EditResult:
    return EditResult(None, error=f"search text matches {len(starts)} places ({how}; lines {_lines_of(src, starts)}) - add context lines")


def replace_once(src: str, search: str, replace: str, replace_all: bool = False) -> EditResult:
    """Replace `search` by `replace`. Levels, each tried only when the one before found nothing:
    exact; whitespace / indentation-tolerant line match; the same with typographic variants folded (curly quotes, dashes, ellipsis,
    no-break space - also inside a line); a block whose first and last line match and whose middle is similar. The three tolerant
    levels need exactly ONE candidate (the whitespace level also accepts several with replace_all). A search that matches several
    places is refused (unless replace_all at the exact / whitespace level) so the wrong one is never edited; `how` names the level.
    The replacement is always written as the model gave it."""
    if not search:
        return EditResult(None, error="empty search text")
    n = src.count(search)
    if n == 1 or (n > 1 and replace_all):
        return EditResult(src.replace(search, replace) if replace_all else src.replace(search, replace, 1), "exact")
    if n > 1:
        starts, k = [], src.find(search)
        while k != -1 and len(starts) < 6:
            starts.append(k)
            k = src.find(search, k + 1)
        return EditResult(None, error=f"search text matches {n} places (lines {_lines_of(src, starts)}) - add a few surrounding lines "
                                      "to make it unique (or pass replace_all=true)")
    src_lines = src.splitlines(keepends=True)
    plain = [ln.rstrip("\n") for ln in src_lines]
    search_lines = search.splitlines()
    hits = _find_lines(plain, search_lines)
    if len(hits) > 1 and not replace_all:
        return _ambiguous("ignoring whitespace", src, [sum(len(x) for x in src_lines[:h]) for h in hits])
    if hits:
        return EditResult(_splice_lines(src_lines, search, search_lines, replace, hits if replace_all else hits[:1]), "whitespace-tolerant")
    # level: typographic variants (quotes, dashes, ellipsis, special spaces) folded on BOTH sides
    spans = _typo_spans(src, search)
    if len(spans) == 1:
        a, b = spans[0]
        return EditResult(src[:a] + replace + src[b:], HOW_TYPO)
    if len(spans) > 1:
        return _ambiguous("ignoring typographic quotes/dashes", src, [a for a, _ in spans])
    hits = _find_lines(plain, search_lines, _typo_norm)
    if len(hits) == 1:
        return EditResult(_splice_lines(src_lines, search, search_lines, replace, hits), HOW_TYPO)
    if len(hits) > 1:
        return _ambiguous("ignoring typographic quotes/dashes and whitespace", src, [sum(len(x) for x in src_lines[:h]) for h in hits])
    # level: block anchor
    blocks = _anchor_blocks(plain, search_lines)
    if len(blocks) == 1:
        i, j, ratio = blocks[0]
        block = "".join(src_lines[i:j])
        rep = _reindent(replace, search, block)
        if block.endswith("\n") and rep and not rep.endswith("\n"):
            rep += "\n"
        return EditResult("".join(src_lines[:i]) + rep + "".join(src_lines[j:]), f"{HOW_ANCHOR} (similarity {ratio:.2f}, lines {i + 1}-{j})")
    if len(blocks) > 1:
        return _ambiguous("same first and last line", src, [sum(len(x) for x in src_lines[:i]) for i, _j, _r in blocks])
    return EditResult(None, error=_not_found_hint(src, search))


HINT_BLOCK_LINES, HINT_BLOCK_CHARS = 30, 2400


def _not_found_hint(src: str, search: str) -> str:
    """Point at the closest real line, name the first line of the search that differs from the file, and show the file's real
    lines there - seen live: "closest line is 29" sent the model to read_file and only then to a second edit; with the real
    text in the error it can copy it and fix the edit in the next step."""
    want_lines = [ln for ln in search.splitlines() if ln.strip()]
    if want_lines:
        want = _norm(want_lines[0])
        src_lines = src.splitlines()
        best, best_score = None, 0.0
        for i, line in enumerate(src_lines):
            h = _norm(line)
            if not h:
                continue
            common = len(set(want.split()) & set(h.split()))
            score = common / max(len(want.split()), 1)
            if score > best_score:
                best, best_score = i, score
        if best is not None and best_score >= 0.5:
            have = [(n, ln) for n, ln in enumerate(src_lines[best:], best + 1) if ln.strip()]
            differs = ""
            for k, w in enumerate(want_lines):
                if k >= len(have):
                    differs = f" The file ends before your search line {k + 1}."
                    break
                if _norm(w) != _norm(have[k][1]):
                    differs = f" Your line {_one(w)} differs from the file's line {have[k][0]}: {_one(have[k][1])}."
                    break
            span = min(len(search.splitlines()) + 3, HINT_BLOCK_LINES)
            block = "\n".join(src_lines[best:best + span])[:HINT_BLOCK_CHARS]
            return (f"search text not found. The closest match starts at line {best + 1}.{differs} The file's real lines "
                    f"{best + 1}-{min(best + span, len(src_lines))} (copy from these exactly):\n{block}")
    return "search text not found - read the file (read_file with a line range) and copy the lines exactly."


def _one(line: str) -> str:
    return repr(line.strip()[:120])


def apply_edits(src: str, edits: list[dict], replace_all: bool = False) -> EditResult:
    """Apply several edits in order; all or nothing."""
    text, hows = src, []
    for i, e in enumerate(edits, 1):
        res = replace_once(text, str(e.get("search", "")), str(e.get("replace", "")), bool(e.get("replace_all", replace_all)))
        if res.text is None:
            return EditResult(None, error=f"edit {i}/{len(edits)} failed: {res.error}")
        text = res.text
        hows.append(res.how)
    return EditResult(text, ",".join(sorted(set(hows))))


# ── unified diff ────────────────────────────────────────────────────────────

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch: str, default_path: str | None = None) -> list[tuple[str, list[list[str]]]]:
    """[(path, hunks)] where a hunk is a list of diff lines (' ', '-', '+' prefixed). Accepts what local models actually send:
    a proper unified diff (`--- a/x` / `+++ b/x`), `diff --git` headers, the `*** Update File: x` block style, and hunks with
    no file header at all (then `default_path` names the file)."""
    files: list[tuple[str, list[list[str]]]] = []
    cur: list[list[str]] | None = None
    for raw in (patch or "").splitlines():
        stripped = raw.strip()
        if raw.startswith("+++ "):
            path = raw[4:].strip().split("\t")[0]
            path = path[2:] if path.startswith(("b/", "a/")) else path
            cur = []
            files.append((path, cur))
        elif stripped.startswith(("*** Update File:", "*** Add File:", "*** Modify File:")):
            cur = []
            files.append((stripped.split(":", 1)[1].strip(), cur))
        elif raw.startswith("diff --git "):
            parts = raw.split()
            if len(parts) >= 4 and cur is None:
                cur = []
                files.append((parts[3][2:] if parts[3].startswith("b/") else parts[3], cur))
        elif raw.startswith(("--- ", "index ", "new file", "deleted file", "*** Begin Patch", "*** End Patch", "*** End of File")) or stripped == "\\ No newline at end of file":
            continue
        elif raw.startswith("@@"):
            if cur is None and default_path:
                cur = []
                files.append((default_path, cur))
            if cur is not None:
                cur.append([])
        elif cur is not None and cur and raw[:1] in (" ", "-", "+"):
            cur[-1].append(raw)
        elif cur is not None and cur and raw == "":
            cur[-1].append(" ")
    return [(p_, h) for p_, h in files if h]


def apply_hunks(src: str, hunks: list[list[str]]) -> EditResult:
    """Apply hunks by content (line numbers in model-written diffs are unreliable)."""
    lines = src.splitlines()
    ends_nl = src.endswith("\n") or not src
    for n, hunk in enumerate(hunks, 1):
        old = [ln[1:] for ln in hunk if ln[:1] in (" ", "-")]
        new = [ln[1:] for ln in hunk if ln[:1] in (" ", "+")]
        if not old:
            lines.extend(new)                      # pure addition with no context: append
            continue
        hits = _find_lines(lines, old)
        if not hits:
            return EditResult(None, error=f"hunk {n}: the context/removed lines were not found in the file")
        if len(hits) > 1:
            return EditResult(None, error=f"hunk {n}: matches {len(hits)} places - include more context lines")
        i = hits[0]
        # keep the file's own indentation for lines that only differ in whitespace
        lines[i:i + len([x for x in old])] = new
    out = "\n".join(lines)
    return EditResult(out + ("\n" if ends_nl and out else ""), "patch")


# ── outlines and repo map ───────────────────────────────────────────────────

def _sig(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}" + (f"({bases})" if bases else "")
    args = ast.unparse(node.args) if hasattr(node, "args") else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
    return f"{prefix} {node.name}({args}){ret}"


def outline_python(text: str, max_items: int = 120) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"(syntax error: {exc.msg} at line {exc.lineno})"]
    out: list[str] = []

    def visit(body, depth):
        for node in body:
            if len(out) >= max_items:
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                tail = f"  # {doc[0][:80]}" if doc else ""
                out.append(f"{'  ' * depth}{node.lineno:>5}: {_sig(node)[:160]}{tail}")
                if isinstance(node, ast.ClassDef):
                    visit(node.body, depth + 1)
            elif depth == 0 and isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if names and all(n.isupper() or n.startswith("_") for n in names):
                    out.append(f"{node.lineno:>5}: {', '.join(names)} = ...")
    visit(tree.body, 0)
    return out


_GENERIC = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|def\s+\w+|fn\s+\w+|func\s+\w+|"
    r"(?:pub\s+)?(?:struct|enum|trait|impl)\s+\w+|interface\s+\w+|type\s+\w+\s*=|const\s+\w+\s*=\s*(?:async\s*)?\()")


def outline(path: str, text: str, max_items: int = 120) -> list[str]:
    if Path(path).suffix.lower() in (".py", ".pyi"):
        return outline_python(text, max_items)
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if _GENERIC.match(line):
            out.append(f"{n:>5}: {line.strip()[:150]}")
            if len(out) >= max_items:
                break
    return out


def _git_exclude_file(root: Path) -> Path | None:
    """The `info/exclude` of the repo `root` belongs to (a .git folder here or above, or a worktree's .git file), else None."""
    for d in [root, *list(root.parents)[:10]]:
        g = d / ".git"
        if g.is_dir():
            return g / "info" / "exclude"
        if g.is_file():                               # worktree / submodule: let git say where its info folder is
            try:
                r = subprocess.run(["git", "rev-parse", "--git-path", "info/exclude"], cwd=root, capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                return None
            out = r.stdout.strip()
            return (root / out) if r.returncode == 0 and out else None
    return None


def ensure_scratch_excluded(root: Path | str) -> bool:
    """Put `.scratch/` into .git/info/exclude (local, untracked - no .gitignore change) so `git add -A` can never commit scratch
    files. True when the line was added now; False when it was there already, there is no git repo, or the file is not writable."""
    ex = _git_exclude_file(Path(root).resolve())
    if ex is None:
        return False
    try:
        old = ex.read_text(encoding="utf-8", errors="replace") if ex.exists() else ""
        if any(ln.strip() in (".scratch", ".scratch/", "/.scratch", "/.scratch/") for ln in old.splitlines()):
            return False
        ex.parent.mkdir(parents=True, exist_ok=True)
        ex.write_text(old + ("" if not old or old.endswith("\n") else "\n") + ".scratch/\n", encoding="utf-8")
        return True
    except OSError:
        return False


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             "dist", "build", "target", ".tox", ".idea", ".vscode", ".next", ".cache", "site-packages"}
CODE_SUFFIXES = {".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".h", ".cc", ".cpp", ".hpp", ".java", ".kt",
                 ".rb", ".php", ".cs", ".swift", ".sh", ".lua", ".nix", ".sql", ".html", ".css", ".md", ".toml", ".yaml", ".yml", ".json"}


def iter_files(root: Path, limit: int = 4000):
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") or d in (".github",))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix.lower() in CODE_SUFFIXES or name in ("Makefile", "Dockerfile", "justfile"):
                yield p
                n += 1
                if n >= limit:
                    return


def repo_map(root: Path, budget_chars: int = 5000, focus: list[str] | None = None) -> str:
    """Compact map: every file with its size, and for code files the top-level names. Fits `budget_chars`;
    files in `focus` (recently touched) get their outline first."""
    root = Path(root)
    files = list(iter_files(root))
    if not files:
        return "(no code files yet)"
    focus_set = {str(Path(f)) for f in (focus or [])}
    lines: list[str] = []
    used = 0

    def add(text: str) -> bool:
        nonlocal used
        if used + len(text) + 1 > budget_chars:
            return False
        lines.append(text)
        used += len(text) + 1
        return True

    for p in files:
        rel = p.relative_to(root)
        try:
            size = p.stat().st_size
        except OSError:
            continue
        head = f"{rel}  ({size // 1000 + 1}k)" if size >= 1000 else f"{rel}  ({size}b)"
        if not add(head):
            lines.append("… (map truncated - use search or list_dir)")
            break
        if (p.suffix in (".py", ".js", ".ts", ".rs", ".go") and size < 200_000) or str(p) in focus_set:
            try:
                body = outline(str(p), p.read_text(encoding="utf-8", errors="replace"), 14)
            except OSError:
                continue
            for item in body:
                if not add("    " + item.strip()):
                    break
    return "\n".join(lines)


def numbered(text: str, start: int = 1, end: int | None = None) -> str:
    lines = text.splitlines()
    end = len(lines) if end is None else min(end, len(lines))
    start = max(1, start)
    return "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, end + 1))
