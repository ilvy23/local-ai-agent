"""Cross-module consistency check for Python projects: does the code call things that really exist?

A local model writing several modules in a row invents API names (`State.gain_item` when the class defines `take`),
imports names a sibling never defined, or calls a function with the wrong arguments - and only finds out when it runs.
This check reads the sibling modules with `ast` and reports exactly those mismatches right after a file is written.

It is conservative on purpose: a class is only checked when everything about it is known (all bases are local, no
`__getattr__`, no unknown decorators), because a false alarm costs the model a wasted step. Warnings are advice, never a
blocker.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

MAX_WARNINGS = 6
_KNOWN_DECORATORS = {"dataclass", "dataclasses.dataclass", "staticmethod", "classmethod", "property", "total_ordering", "unique"}
_OPEN_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "NamedTuple", "TypedDict", "Exception", "ValueError", "RuntimeError", "object"}


@dataclass
class Sig:
    params: list[str]
    required: int
    vararg: bool
    kwarg: bool
    kwonly: list[str] = field(default_factory=list)

    def accepts(self, n_pos: int, keywords: list[str]) -> str | None:
        if not self.vararg and n_pos > len(self.params):
            return f"takes at most {len(self.params)} positional argument(s) but {n_pos} were given"
        if not self.kwarg:
            bad = [k for k in keywords if k not in self.params and k not in self.kwonly]
            if bad:
                return f"has no parameter named {', '.join(repr(b) for b in bad)} (parameters: {', '.join(self.params + self.kwonly) or 'none'})"
        given = min(n_pos, len(self.params)) + sum(1 for k in keywords if k in self.params)
        if given < self.required and not self.vararg:
            missing = [p for p in self.params[n_pos:self.required] if p not in keywords]
            if missing:
                return f"is missing required argument(s): {', '.join(missing)}"
        return None


def _sig(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_self: bool) -> Sig:
    a = fn.args
    params = [x.arg for x in [*a.posonlyargs, *a.args]]
    if drop_self and params:
        params = params[1:]
    defaults = len(a.defaults)
    required = max(len(params) - defaults, 0)
    return Sig(params, required, a.vararg is not None, a.kwarg is not None, [x.arg for x in a.kwonlyargs])


@dataclass
class ClassInfo:
    name: str
    members: set[str] = field(default_factory=set)
    init: Sig | None = None
    bases: list[str] = field(default_factory=list)
    closed: bool = True                        # everything about it is known


@dataclass
class ModuleInfo:
    name: str
    names: set[str] = field(default_factory=set)
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    funcs: dict[str, Sig] = field(default_factory=dict)
    star_import: bool = False


def _dec_name(d: ast.expr) -> str:
    node = d.func if isinstance(d, ast.Call) else d
    return ast.unparse(node)


def _class_info(node: ast.ClassDef, known: dict[str, ClassInfo]) -> ClassInfo:
    info = ClassInfo(node.name, bases=[ast.unparse(b) for b in node.bases])
    is_dc = any(_dec_name(d).split(".")[-1] == "dataclass" for d in node.decorator_list)
    if any(_dec_name(d) not in _KNOWN_DECORATORS for d in node.decorator_list):
        info.closed = False
    for b in info.bases:
        if b in known:
            info.members |= known[b].members
            info.closed = info.closed and known[b].closed
        elif b not in _OPEN_BASES:
            info.closed = False
    if any(b in _OPEN_BASES - {"object"} for b in info.bases):
        info.closed = False if any(b in ("Enum", "IntEnum", "StrEnum", "Flag", "NamedTuple", "TypedDict") for b in info.bases) else info.closed
    dc_params: list[str] = []
    dc_required = 0
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.members.add(item.name)
            if item.name == "__init__":
                info.init = _sig(item, True)
            if item.name in ("__getattr__", "__getattribute__", "__slots__"):
                info.closed = False
            for sub in ast.walk(item):                       # self.x = ... anywhere in a method
                if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    for t in targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                            info.members.add(t.attr)
                        elif isinstance(t, ast.Tuple):
                            for e in t.elts:
                                if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self":
                                    info.members.add(e.attr)
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    info.members.add(t.id)
                    if t.id == "__slots__":
                        info.closed = False
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            info.members.add(item.target.id)
            if is_dc:
                dc_params.append(item.target.id)
                if item.value is None:
                    dc_required += 1
    if is_dc and info.init is None:
        info.init = Sig(dc_params, dc_required, False, False)
    if info.init is None and not info.bases:
        info.init = Sig([], 0, False, False)
    if info.init is None:
        for b in info.bases:
            if b in known and known[b].init is not None:
                info.init = known[b].init
    info.members |= {"__class__", "__dict__", "__doc__", "__init__", "__repr__", "__eq__", "__hash__", "__str__"}
    return info


def read_module(path: Path) -> ModuleInfo | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None
    mod = ModuleInfo(path.stem)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            mod.names.add(node.name)
            mod.classes[node.name] = _class_info(node, {k: v for k, v in mod.classes.items()})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mod.names.add(node.name)
            mod.funcs[node.name] = _sig(node, False)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        mod.names.add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            mod.names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name == "*":
                    mod.star_import = True
                mod.names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.If, ast.Try, ast.With)):           # names defined conditionally
            for n in ast.walk(node):
                if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    mod.names.add(n.name)
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    mod.names.add(n.id)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        mod.names.add((a.asname or a.name).split(".")[0])
    return mod


class _Checker(ast.NodeVisitor):
    def __init__(self, root: Path, rel: str, mods: dict[str, ModuleInfo]) -> None:
        self.root, self.rel, self.mods = root, rel, mods
        self.out: list[str] = []
        self.alias_mod: dict[str, str] = {}            # local name -> module name
        self.alias_cls: dict[str, tuple[str, str]] = {}  # local name -> (module, class)
        self.alias_fn: dict[str, tuple[str, str]] = {}
        self.var_types: list[dict[str, tuple[str, str]]] = [{}]
        self.self_types: dict[str, tuple[str, str]] = {}
        self.cur_class_info: ClassInfo | None = None

    def warn(self, node: ast.AST, msg: str) -> None:
        if len(self.out) < MAX_WARNINGS + 4:
            self.out.append(f"{self.rel}:{getattr(node, 'lineno', '?')}: {msg}")

    # imports
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            top = a.name.split(".")[0]
            if top in self.mods and "." not in a.name:
                self.alias_mod[a.asname or a.name] = top

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module or node.module not in self.mods:
            return
        mod = self.mods[node.module]
        for a in node.names:
            if a.name == "*":
                continue
            local = a.asname or a.name
            if a.name in mod.classes:
                self.alias_cls[local] = (mod.name, a.name)
            elif a.name in mod.funcs:
                self.alias_fn[local] = (mod.name, a.name)
            elif a.name not in mod.names and not mod.star_import:
                near = sorted(mod.names)[:12]
                self.warn(node, f"`from {mod.name} import {a.name}`: {mod.name}.py does not define '{a.name}' (it defines: {', '.join(near)})")

    # type tracking (very small on purpose)
    def _ctor_type(self, value: ast.expr | None) -> tuple[str, str] | None:
        if isinstance(value, ast.Call):
            f = value.func
            if isinstance(f, ast.Name) and f.id in self.alias_cls:
                return self.alias_cls[f.id]
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in self.alias_mod:
                mod = self.mods[self.alias_mod[f.value.id]]
                if f.attr in mod.classes:
                    return (mod.name, f.attr)
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        t = self._ctor_type(node.value)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                if t:
                    self.var_types[-1][tgt.id] = t
                else:
                    self.var_types[-1].pop(tgt.id, None)
            elif isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self" and t:
                self.self_types[tgt.attr] = t
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.var_types.append(dict(self.var_types[-1]))
        self.generic_visit(node)
        self.var_types.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for sub in ast.walk(node):                       # pre-scan self.x = Cls() so later methods can use it
            if isinstance(sub, ast.Assign):
                t = self._ctor_type(sub.value)
                for tgt in sub.targets:
                    if t and isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                        self.self_types[tgt.attr] = t
        self.generic_visit(node)

    # checks
    def _class(self, ref: tuple[str, str]) -> ClassInfo | None:
        return self.mods[ref[0]].classes.get(ref[1])

    def _resolve(self, node: ast.expr) -> tuple[str, tuple] | None:
        if isinstance(node, ast.Name):
            if node.id in self.alias_mod:
                return "module", (self.alias_mod[node.id],)
            if node.id in self.alias_cls:
                return "class", self.alias_cls[node.id]
            if node.id in self.var_types[-1]:
                return "instance", self.var_types[-1][node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in self.self_types:
            return "instance", self.self_types[node.attr]
        if isinstance(node, ast.Call):
            t = self._ctor_type(node)
            if t:
                return "instance", t
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            r = self._resolve(node.value)
            if r:
                kind, ref = r
                if kind == "module":
                    mod = self.mods[ref[0]]
                    if node.attr not in mod.names and not mod.star_import and not node.attr.startswith("__"):
                        self.warn(node, f"`{ref[0]}.{node.attr}`: {ref[0]}.py defines no '{node.attr}' (it defines: {', '.join(sorted(mod.names)[:12])})")
                else:
                    cls = self._class(ref)
                    if cls and cls.closed and node.attr not in cls.members and not node.attr.startswith("__"):
                        self.warn(node, f"`{ref[1]}` (from {ref[0]}.py) has no attribute '{node.attr}' "
                                        f"(it has: {', '.join(sorted(m for m in cls.members if not m.startswith('__'))[:14])})")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        sig, label = None, ""
        f = node.func
        if isinstance(f, ast.Name) and f.id in self.alias_cls:
            cls = self._class(self.alias_cls[f.id])
            sig, label = (cls.init if cls and cls.closed else None), f"{f.id}(...)"
        elif isinstance(f, ast.Name) and f.id in self.alias_fn:
            sig, label = self.mods[self.alias_fn[f.id][0]].funcs.get(self.alias_fn[f.id][1]), f"{f.id}(...)"
        elif isinstance(f, ast.Attribute):
            r = self._resolve(f.value)
            if r and r[0] == "module":
                mod = self.mods[r[1][0]]
                if f.attr in mod.funcs:
                    sig, label = mod.funcs[f.attr], f"{r[1][0]}.{f.attr}(...)"
                elif f.attr in mod.classes and mod.classes[f.attr].closed:
                    sig, label = mod.classes[f.attr].init, f"{r[1][0]}.{f.attr}(...)"
        if sig and not any(isinstance(a, ast.Starred) for a in node.args) and not any(k.arg is None for k in node.keywords):
            why = sig.accepts(len(node.args), [k.arg for k in node.keywords if k.arg])
            if why:
                self.warn(node, f"call `{label}` {why}")
        self.generic_visit(node)


def check_file(root: Path, path: Path, text: str) -> list[str]:
    """Warnings for one Python file against its sibling modules in the same project folder."""
    root = Path(root)
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    mods: dict[str, ModuleInfo] = {}
    for f in sorted(root.glob("*.py")):
        if f.stem != path.stem and not f.stem.startswith("test_"):
            info = read_module(f)
            if info:
                mods[f.stem] = info
    if not mods:
        return []
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    checker = _Checker(root, rel, mods)
    checker.visit(tree)
    seen, out = set(), []
    for w in checker.out:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:MAX_WARNINGS]


def note(root: Path, path: Path, text: str) -> str:
    """The block appended to a write result ('' when everything checks out)."""
    try:
        warnings = check_file(root, path, text)
    except Exception:  # noqa: BLE001 - advice only; never break a write
        return ""
    if not warnings:
        return ""
    return "\nXREF (these calls do not match the sibling modules - fix them or check with `outline`):\n" + "\n".join(f"- {w}" for w in warnings)
