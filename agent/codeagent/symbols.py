"""Symbols and line ranges: edit without quoting the old text.

Seen live (D1): the model would not edit without the exact old text (edit_file needs a verbatim `search`), and once the
harness paused reading it could do neither - 15 minutes of deadlock. These helpers let it point at code instead of
quoting it:
  * find_symbol / show_symbol: exactly one function, class or assignment (dict, list ...) with line numbers,
  * range_hash: a short content hash of a line range, shown with every numbered view; `replace_lines` and `insert_after`
    compare it (or the whole-file hash of the last numbered view) before they change anything, so a stale line number
    never edits the wrong lines - on a mismatch the tool answers with the current lines instead,
  * where_used: every place a name occurs, definitions marked.
Pure functions; the tools in tools.py add the bookkeeping."""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_SHOW_LINES = 300               # a symbol longer than this is shown head + tail
HASH_LEN = 6


def range_hash(lines: list[str]) -> str:
    """Short hash of a block of lines (whitespace at line ends ignored, so an editor's trailing space does not matter)."""
    text = "\n".join(ln.rstrip() for ln in lines)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:HASH_LEN]


def file_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def shifted_starts(lines: list[str], want: str, count: int, near: int, window: int = 80) -> list[int]:
    """1-based start lines within `window` of `near` where `count` lines hash to `want` (a range stamp): where a block the model read
    now stands after earlier edits moved the lines. `near` itself is included, so a caller sees an exact hit as well."""
    if count < 1 or not want:
        return []
    lo, hi = max(1, near - window), min(len(lines) - count + 1, near + window)
    return [i for i in range(lo, hi + 1) if range_hash(lines[i - 1:i - 1 + count]) == want]


def numbered(lines: list[str], first: int) -> str:
    """`  120| code` - the line number is NOT part of the code (never copy it into new_text)."""
    return "\n".join(f"{n:>5}| {ln}" for n, ln in enumerate(lines, first))


@dataclass
class Span:
    name: str                      # qualified: "Cls.method"
    kind: str                      # def, class, assign
    start: int                     # 1-based, decorators included
    end: int                       # inclusive


def python_spans(text: str) -> list[Span]:
    """Every def / class (any depth) and every module- or class-level assignment, with qualified names."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[Span] = []

    def visit(body: list[ast.stmt], prefix: str, in_func: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                q = prefix + node.name
                out.append(Span(q, "class" if isinstance(node, ast.ClassDef) else "def", start, node.end_lineno or node.lineno))
                visit(node.body, q + ".", not isinstance(node, ast.ClassDef))
            elif not in_func and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        out.append(Span(prefix + t.id, "assign", node.lineno, node.end_lineno or node.lineno))
            elif not in_func and isinstance(node, (ast.If, ast.Try, ast.With)):
                for part in ("body", "orelse", "finalbody"):            # `if TYPE_CHECKING:` / try-import blocks
                    visit(getattr(node, part, []) or [], prefix, in_func)
                for h in getattr(node, "handlers", []) or []:
                    visit(h.body, prefix, in_func)
    visit(tree.body, "", False)
    return out


_MODS = r"(?:(?:export|default|pub(?:\([^)]*\))?|public|private|protected|internal|static|final|abstract|async|unsafe|extern|override|open|data|sealed)\s+)*"
_DEF_LINE = (r"^(\s*)" + _MODS + r"(?:def|function\*?|class|fn|func(?:\s*\([^)]*\))?|struct|enum|trait|interface|type|const|let|var|val|impl|object|module|namespace)"
             r"\s+{name}\b")
# C / C++ / Java style: `int add(int a, int b)` or `static void foo(...)` at the start of a line (a call, a `return`, a prototype ending in `;` is not a definition)
_C_DEF = (r"^()(?!\s)(?!return\b|else\b|if\b|while\b|for\b|switch\b|case\b|typedef\b|#)[A-Za-z_][\w\s\*:<>,&~]*?[\s\*&:]{name}\s*\([^;]*$")


def generic_span(text: str, name: str) -> Span | None:
    """Non-Python files: the definition line, then up to the matching closing brace, or (no braces) the indented block."""
    lines = text.splitlines()
    pats = [re.compile(_DEF_LINE.format(name=re.escape(name))), re.compile(_C_DEF.format(name=re.escape(name)))]
    for i, ln in enumerate(lines):
        m = next((m for m in (p.match(ln) for p in pats) if m), None)
        if not m:
            continue
        depth, seen = 0, False
        for j in range(i, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            seen = seen or "{" in lines[j]
            if seen and depth <= 0:
                return Span(name, "def", i + 1, j + 1)
        if seen:
            return Span(name, "def", i + 1, len(lines))
        indent = len(m.group(1))
        end = i
        for j in range(i + 1, len(lines)):
            if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= indent:
                break
            end = j
        return Span(name, "def", i + 1, end + 1)
    return None


def find_symbol(path: str, text: str, name: str) -> tuple[Span | None, list[Span]]:
    """(the one span, []) or (None, candidates) when the name is ambiguous or unknown (candidates = near names)."""
    name = name.strip().strip("`").removesuffix("()")
    if Path(path).suffix.lower() not in (".py", ".pyi"):
        sp = generic_span(text, name.split(".")[-1])
        return (sp, []) if sp else (None, [])
    spans = python_spans(text)
    exact = [s for s in spans if s.name == name]
    if len(exact) == 1:
        return exact[0], []
    tail = [s for s in spans if s.name.split(".")[-1] == name.split(".")[-1]] if not exact else exact
    if len(tail) == 1:
        return tail[0], []
    if tail:
        top = [s for s in tail if "." not in s.name]
        if len(top) == 1 and len(tail) > 1 and "." not in name:
            return top[0], []
        return None, tail
    low = name.lower().split(".")[-1]
    near = [s for s in spans if low in s.name.lower()][:12]
    return None, near


def show(path: str, text: str, span: Span) -> str:
    """The numbered block with its range hash (what replace_lines(expect=...) compares)."""
    lines = text.splitlines()
    block = lines[span.start - 1:span.end]
    h = range_hash(block)
    head = f"{path}: {span.kind} {span.name}  lines {span.start}-{span.end}  #{h}  (file {len(lines)} lines)"
    if len(block) > MAX_SHOW_LINES:
        keep = MAX_SHOW_LINES // 2
        body = (numbered(block[:keep], span.start) + f"\n  [… {len(block) - 2 * keep} lines …]\n"
                + numbered(block[-keep:], span.end - keep + 1))
    else:
        body = numbered(block, span.start)
    return f"{head}\n{body}"


def where_used(root: Path, name: str, limit: int = 40) -> str:
    """Every line that mentions `name` as a whole word, definitions first and marked."""
    name = name.strip().strip("`").removesuffix("()").split(".")[-1]
    if not re.fullmatch(r"[A-Za-z_]\w*", name or ""):
        return "error: where_used needs a plain identifier (a function, class or variable name)."
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "-n", "--no-heading", "--color", "never", "-w", "-F", "--max-columns", "200", "-e", name, str(root)]
    else:
        cmd = ["grep", "-rnIwF", "--exclude-dir=.git", "--exclude-dir=__pycache__", "--exclude-dir=node_modules", "-e", name, str(root)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "where_used timed out"
    rows = [ln.replace(str(root) + "/", "", 1) for ln in r.stdout.splitlines()]
    rows = [ln for ln in rows if not ln.startswith((".scratch/", ".git/"))]
    if not rows:
        return f"'{name}' is not used anywhere in the repo."
    defn = re.compile(rf"^[^:]+:\d+:\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b|^[^:]+:\d+:{re.escape(name)}\s*(?::[^=]*)?=")
    first = [f"{ln[:220]}   [definition]" for ln in rows if defn.search(ln)]
    rest = [ln[:220] for ln in rows if not defn.search(ln)]
    files = len({ln.split(":", 1)[0] for ln in rows})
    shown = (first + rest)[:limit]
    more = f"\n[… {len(rows) - limit} more - narrow with search(path=...)]" if len(rows) > limit else ""
    return f"'{name}': {len(rows)} line(s) in {files} file(s)\n" + "\n".join(shown) + more
