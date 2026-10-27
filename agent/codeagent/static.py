"""Static checks on every Python write and at the final gate: errors the tests may never execute.

Seen live (B4): `_advance_chapter` was called but never defined - a NameError on every `examine` - and two functions
were defined twice (the second silently replaced the first); the tests passed because no test went down that path.
pyflakes is not installed on this system (and the system python is read-only), so this is a small, conservative
re-implementation of the three checks that matter, with the standard library only:
  * undefined names (pyflakes F821): `symtable` knows every scope's bindings; a name that is read (outside an
    annotation) and bound nowhere - not locally, not in an enclosing function, not at module level, not a builtin -
    is reported with its first line. Files with `import *`, a module `__getattr__` or `globals()` writes are skipped
    (their names cannot be known statically), so a report is never a false alarm from those.
  * duplicate definitions: the same def / class name twice in one module or class body (property setters and
    @overload excluded) - the later one wins and the earlier one is dead code,
  * duplicate bindings: a module-level name (def, class or plain assignment, also a data dict) bound twice at the top level without a
    read in between - the first is dead (seen live: a 141-line `ELLA_OPEN = {...}` block twice after a replace_lines with a too small end line),
  * imports inside the project: `from pkg.mod import name` where pkg/mod.py does not define `name`, and project
    modules that do not exist. Flat sibling modules (`from engine import x` next to engine.py) are xref.py's job.
Every result is a short list of `line N: ...` strings; nothing here raises."""

from __future__ import annotations

import ast
import re
import builtins
import symtable
from pathlib import Path

MAX_PROBLEMS = 8
_EXTRA_BUILTINS = {"__file__", "__name__", "__doc__", "__spec__", "__loader__", "__package__", "__builtins__", "__path__",
                   "__annotations__", "__dict__", "__module__", "__qualname__", "__class__", "__debug__", "__cached__",
                   "WindowsError", "reveal_type", "__conditional_annotations__"}
BUILTINS = set(dir(builtins)) | _EXTRA_BUILTINS
_OVERLOAD = {"overload", "typing.overload", "typing_extensions.overload", "singledispatchmethod", "register"}


def _walk_tables(t: symtable.SymbolTable):
    for child in t.get_children():
        yield child
        yield from _walk_tables(child)


def _annotation_nodes(tree: ast.AST) -> set[int]:
    """ids of every node inside an annotation (with postponed annotations an unknown name there is no NameError)."""
    out: set[int] = set()

    def mark(node: ast.AST | None) -> None:
        if node is not None:
            out.update(id(n) for n in ast.walk(node))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mark(node.returns)
            for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg]:
                if a is not None:
                    mark(a.annotation)
        elif isinstance(node, ast.AnnAssign):
            mark(node.annotation)
    return out


def _dynamic(tree: ast.Module) -> bool:
    """Names may come from somewhere symtable cannot see: `from x import *`, a module __getattr__, globals() / exec."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("globals", "exec", "vars"):
            return True
    return any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body)


def undefined_names(text: str, filename: str = "<file>") -> list[tuple[str, int]]:
    """(name, first line) of names that are read but bound nowhere. [] on a syntax error or a dynamic module."""
    try:
        tree = ast.parse(text)
        top = symtable.symtable(text, filename, "exec")
    except (SyntaxError, ValueError, RecursionError):
        return []
    if _dynamic(tree):
        return []
    tables = [top, *_walk_tables(top)]
    bound_at_module = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    for t in tables:                                         # `global x; x = 1` inside a function binds x at module level
        for s in t.get_symbols():
            if s.is_declared_global() and (s.is_assigned() or s.is_imported() or s.is_namespace()):
                bound_at_module.add(s.get_name())
    suspects: set[str] = set()
    for t in tables:
        for s in t.get_symbols():
            if not s.is_referenced():
                continue
            name = s.get_name()
            if name in BUILTINS or name in bound_at_module:
                continue
            if t is top or s.is_global():
                suspects.add(name)
    if not suspects:
        return []
    skip = _annotation_nodes(tree)
    first: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in suspects and id(node) not in skip:
            first[node.id] = min(first.get(node.id, node.lineno), node.lineno)
    return sorted(first.items(), key=lambda kv: kv[1])


def _decorated_away(node: ast.AST) -> bool:
    for d in getattr(node, "decorator_list", []) or []:
        name = ast.unparse(d)
        if name.split("(")[0] in _OVERLOAD or name.endswith((".setter", ".getter", ".deleter", ".register")) or ".register(" in name:
            return True
    return False


def duplicate_defs(text: str) -> list[tuple[str, int, int]]:
    """(qualified name, first line, second line) for a def/class defined twice in the same body."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    out: list[tuple[str, int, int]] = []

    def scan(body: list[ast.stmt], prefix: str) -> None:
        seen: dict[str, int] = {}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not _decorated_away(node):
                    if node.name in seen:
                        out.append((prefix + node.name, seen[node.name], node.lineno))
                    seen[node.name] = node.lineno
                if isinstance(node, ast.ClassDef):
                    scan(node.body, prefix + node.name + ".")
    scan(tree.body, "")
    return out


def _loaded(node: ast.AST) -> set[str]:
    """Every name read anywhere inside `node` (function bodies included: a read there may happen at any time) plus names a `global`
    statement lets a function rebind and `x += ...` targets - all mean "the earlier binding may matter"."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Load, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.Global):
            out.update(n.names)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def _stored(node: ast.AST) -> set[str]:
    out = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out


def duplicate_bindings(text: str) -> list[tuple[str, int, int]]:
    """(name, first line, second line) for a module-level name that is bound twice at the top level with no read of it in between:
    pyflakes F811 for def / class, and also plain data (`ELLA_OPEN = {...}` twice - seen live after a replace_lines whose end line was too
    small). Two defs of one name stay with duplicate_defs. Not reported: bindings inside if / try / for / with, `x = x + [...]`, `x += ...`,
    `x.update(...)` or any other read in between (a function that reads it counts), names a function rebinds with `global`, decorated
    defs (overload / setter / register), a constant first value (`X = None` placeholder), `_` and dunder names."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    if _dynamic(tree):
        return []
    live: dict[str, tuple[int, bool]] = {}              # name -> (line, is_def) of a binding nobody has read yet
    out: list[tuple[str, int, int]] = []
    for node in tree.body:
        for name in _loaded(node) & live.keys():
            del live[name]
        names: list[tuple[str, bool]] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.decorator_list:
                names = [(node.name, True)]
        elif isinstance(node, ast.Assign) and not isinstance(node.value, ast.Constant):
            names = [(t.id, False) for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name) and not isinstance(node.value, ast.Constant):
            names = [(node.target.id, False)]
        for name in _stored(node) - {n for n, _d in names}:       # a loop / with / tuple / import / nested / decorated / constant binding: not tracked
            live.pop(name, None)
        for name, is_def in names:
            if name == "_" or (name.startswith("__") and name.endswith("__")):
                continue
            if name in live and not (is_def and live[name][1]):
                out.append((name, live[name][0], node.lineno))
            live[name] = (node.lineno, is_def)
    return out


# ── imports inside the project ──────────────────────────────────────────────

def _module_file(root: Path, dotted: str) -> Path | None:
    parts = dotted.split(".")
    for base in (root, root / "src"):
        f = base.joinpath(*parts)
        if f.with_suffix(".py").is_file():
            return f.with_suffix(".py")
        if (f / "__init__.py").is_file():
            return f / "__init__.py"
        if f.is_dir() and any(f.glob("*.py")):                       # namespace package
            return f
    return None


def _project_top(root: Path, top: str) -> bool:
    return any((base / f"{top}.py").is_file() or (base / top).is_dir() and any((base / top).glob("*.py")) for base in (root, root / "src"))


def _defined_names(f: Path) -> tuple[set[str], bool]:
    """Top-level names of a module (and submodules of a package); True when it cannot be known (star import, __getattr__)."""
    if f.is_dir():
        return {p.stem for p in f.glob("*.py")} | {p.name for p in f.iterdir() if p.is_dir()}, False
    try:
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set(), True
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node in tree.body:
            names.add(node.name)
    top = list(tree.body)
    i = 0
    while i < len(top):                                             # module level incl. if/try blocks
        node = top[i]
        i += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(a.name == "*" for a in node.names):
                return names, True
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            top.extend(getattr(node, "body", []) + getattr(node, "orelse", []) + getattr(node, "finalbody", []))
            for h in getattr(node, "handlers", []) or []:
                top.extend(h.body)
    if "__getattr__" in names:
        return names, True
    if f.name == "__init__.py":
        names |= {p.stem for p in f.parent.glob("*.py")} | {p.name for p in f.parent.iterdir() if p.is_dir()}
    return names, False


def import_problems(root: Path, path: Path, text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    root = Path(root)
    try:
        here = Path(path).resolve().parent.relative_to(root.resolve())
    except ValueError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = here.parts[: len(here.parts) - (node.level - 1)] if node.level - 1 <= len(here.parts) else None
                if pkg is None:
                    continue
                dotted = ".".join([*pkg, *(node.module.split(".") if node.module else [])])
                shown = "." * node.level + (node.module or "")
            else:
                if not node.module:
                    continue
                dotted, shown = node.module, node.module
                top = dotted.split(".")[0]
                if not _project_top(root, top) or ("." not in dotted and (root / f"{dotted}.py").is_file()):
                    continue                                        # not ours, or a flat sibling module (xref.py checks those)
            if not dotted:
                continue
            f = _module_file(root, dotted)
            if f is None:
                out.append(f"line {node.lineno}: `from {shown} import ...`: there is no module {dotted.replace('.', '/')}.py in the project")
                continue
            names, unknown = _defined_names(f)
            if unknown:
                continue
            for a in node.names:
                if a.name != "*" and a.name not in names:
                    out.append(f"line {node.lineno}: `from {shown} import {a.name}`: {f.relative_to(root) if f.is_relative_to(root) else f} does not define '{a.name}'")
        elif isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if "." in a.name and _project_top(root, top) and _module_file(root, a.name) is None:
                    out.append(f"line {node.lineno}: `import {a.name}`: there is no module {a.name.replace('.', '/')}.py in the project")
    return out


# ── the checks together ─────────────────────────────────────────────────────

def check(root: Path | str, path: Path | str, text: str) -> list[str]:
    """All problems of one Python file, most important first, each `line N: ...`."""
    path = Path(path)
    if path.suffix != ".py":
        return []
    out = [f"line {ln}: undefined name '{name}' (NameError when this line runs)" for name, ln in undefined_names(text, str(path))]
    out += [f"line {b}: '{name}' is defined twice (lines {a} and {b}) - the second replaces the first" for name, a, b in duplicate_defs(text)]
    out += [f"line {b}: {name} is defined twice at module level (line {a} and line {b}) - the first definition is dead. If you rewrote a block "
            "with replace_lines, your end line was too small: the old text is still there. Remove the stale copy." for name, a, b in duplicate_bindings(text)]
    try:
        out += import_problems(Path(root), path, text)
    except (OSError, ValueError, RecursionError):
        pass
    return out[:MAX_PROBLEMS]


_PROBLEM_LINE = re.compile(r"^line (\d+): (.*)$", re.S)
_LINE_REFS = re.compile(r"\blines? \d+(?: and \d+)?")


def signatures(problems: list[str], text: str) -> list[tuple[str, str]]:
    """One (message without line numbers, whitespace-normalised content of the line it points at) per problem. Two problems of two
    states of a file are "the same" when these match, so an edit that only shifts lines does not make an old problem look new."""
    rows = text.splitlines()
    out = []
    for prob in problems:
        m = _PROBLEM_LINE.match(prob)
        if not m:
            out.append((_LINE_REFS.sub("line", prob), ""))
            continue
        n = int(m.group(1))
        out.append((_LINE_REFS.sub("line", m.group(2)), " ".join(rows[n - 1].split()) if 0 < n <= len(rows) else ""))
    return out


def new_problems(problems: list[str], text: str, before: set[tuple[str, str]]) -> list[str]:
    """The problems of `text` that its earlier state (the signatures `before`) did not have."""
    return [prob for prob, sig in zip(problems, signatures(problems, text)) if sig not in before]


def note(problems: list[str], unchanged: int = 0) -> str:
    """The write-result note: the NEW problems in full, the ones the file already had before this edit as one short line."""
    tail = f"\n{unchanged} earlier issue{'s' if unchanged != 1 else ''} in this file unchanged" if unchanged else ""
    if not problems:
        return tail
    return "\nSTATIC CHECK (these fail at run time even if no test reaches them - fix them now):\n" + "\n".join(f"- {p}" for p in problems) + tail


def check_files(root: Path | str, files: list[str]) -> dict[str, list[str]]:
    """path -> problems, for the existing .py files among `files` (relative to root)."""
    root = Path(root)
    out: dict[str, list[str]] = {}
    for rel in files:
        p = root / rel if not Path(rel).is_absolute() else Path(rel)
        if p.suffix != ".py" or not p.is_file():
            continue
        try:
            probs = check(root, p, p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if probs:
            out[str(p.relative_to(root)) if p.is_relative_to(root) else str(p)] = probs
    return out
