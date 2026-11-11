"""A persistent map of a project that keeps itself true, for EVERY coding run (not only nights).

The model used to spend the first stage of every run re-reading the code base to learn what is where. This map is written by the harness from the code itself (ast for
Python, a few regexes for other languages; no model call): per file its size, its top-level classes and functions with the first docstring line, and who imports it. It is
stored between runs (in <project>/.night/ when that folder exists, else in the user's state folder keyed by the project path, so no repo gets a new folder) and put into the system
prompt of every run; `refresh()` runs at the start and at the end of every run:
  * files are only re-read when their size or mtime changed (a stat per file otherwise), so a big repo costs nothing;
  * a renamed function / class / constant (found by its unchanged signature, or the only one removed and the only one added in a file) and a renamed or moved file (same content,
    new path) are updated in the model-written `## Notes` section below the map; a note that mentions something that is gone gets `[stale: ...]` instead of silently lying.
The factual part above `## Notes` is never written by a model."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent import osport

CODE_EXT = {".py": "py", ".js": "js", ".ts": "js", ".tsx": "js", ".jsx": "js", ".go": "go", ".rs": "rs", ".java": "java", ".c": "c", ".h": "c", ".cpp": "c"}
SKIP_DIRS = {".git", ".night", ".scratch", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build", "target", ".mypy_cache", ".ruff_cache"}
NOTES_MARK = "## Notes"
MAX_FILE_BYTES = 400_000
_JS_DEF = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()", re.M)
_GO_DEF = re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)|^type\s+(\w+)\s+(?:struct|interface)", re.M)
_RS_DEF = re.compile(r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait)\s+(\w+)", re.M)
_C_DEF = re.compile(r"^[\w\*\s]+\s+\**(\w+)\s*\([^;{]*\)\s*\{", re.M)
_NAME = re.compile(r"^(?:class\s+)?([A-Za-z_]\w*)")


@dataclass
class FileInfo:
    path: str
    lines: int
    defs: list[str] = field(default_factory=list)          # "class Box [open] - A box.", "run(a, b) - Run the thing.", "DEFAULT"
    imports: list[str] = field(default_factory=list)
    sha: str = ""
    mtime_ns: int = 0
    size: int = 0


@dataclass
class Changes:
    changed: list[str] = field(default_factory=list)              # added, modified or removed paths
    renamed_files: dict[str, str] = field(default_factory=dict)
    renamed_names: dict[str, str] = field(default_factory=dict)
    stale: list[str] = field(default_factory=list)                # names / paths that notes mention and that are gone

    def __bool__(self) -> bool:
        return bool(self.changed)


# ---------------------------------------------------------------------------------------------------------------- where the map lives
def store_dir(root: Path) -> Path:
    root = Path(root).resolve()
    if (root / ".night").is_dir():
        return root / ".night"
    return osport.state_home() / "agent" / "maps" / hashlib.sha1(str(root).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------------- scanning
def _py(text: str) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [], []
    defs, imports = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = (ast.get_docstring(node) or "").strip().splitlines()
            tail = f" - {doc[0][:60]}" if doc else ""
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")][:6]
                defs.append(f"class {node.name}" + (f" [{', '.join(methods)}]" if methods else "") + tail)
            else:
                args = ", ".join(a.arg for a in node.args.args[:4])
                defs.append(f"{node.name}({args})" + tail)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper():
            defs.append(node.targets[0].id)
        elif isinstance(node, ast.Import):
            imports += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imports.append(node.module.split(".")[0])
    return defs, imports


def _other(kind: str, text: str) -> list[str]:
    rx = {"js": _JS_DEF, "go": _GO_DEF, "rs": _RS_DEF, "c": _C_DEF, "java": _JS_DEF}.get(kind)
    if rx is None:
        return []
    return [next(g for g in m.groups() if g) for m in rx.finditer(text) if any(m.groups())][:30]


def scan(root: Path, cache: dict[str, dict] | None = None) -> list[FileInfo]:
    """Every source file under `root`. A file whose size and mtime equal its entry in `cache` is not read again."""
    root = Path(root)
    cache = cache or {}
    out: list[FileInfo] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix not in CODE_EXT:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > MAX_FILE_BYTES or not p.is_file():
                continue
            rel = str(p.relative_to(root))
            c = cache.get(rel)
            if c and c.get("mtime_ns") == st.st_mtime_ns and c.get("size") == st.st_size:
                out.append(FileInfo(rel, c["lines"], c["defs"], c["imports"], c["sha"], st.st_mtime_ns, st.st_size))
                continue
            data = p.read_bytes()
            text = data.decode("utf-8", errors="replace")
            kind = CODE_EXT[p.suffix]
            defs, imports = _py(text) if kind == "py" else (_other(kind, text), [])
            out.append(FileInfo(rel, text.count("\n") + 1, defs, imports, hashlib.sha1(data).hexdigest()[:12], st.st_mtime_ns, st.st_size))
    return sorted(out, key=lambda f: f.path)


# ---------------------------------------------------------------------------------------------------------------- rendering
def _importance(f: FileInfo, importers: dict[str, int]) -> float:
    return importers.get(Path(f.path).stem, 0) * 3 + len(f.defs) + f.lines / 200 - (2 if "test" in f.path.lower() else 0)


def render(files: list[FileInfo], budget: int = 8000, title: str = "Project map") -> str:
    local = {Path(f.path).stem for f in files}
    importers: dict[str, int] = {}
    imported_by: dict[str, list[str]] = {}
    for f in files:
        for m in set(f.imports) & local:
            importers[m] = importers.get(m, 0) + 1
            imported_by.setdefault(m, []).append(Path(f.path).stem)
    ranked = sorted(files, key=lambda f: -_importance(f, importers))
    head = f"# {title}\n(written by the harness from the code itself; {len(files)} source files; re-checked at the start and the end of every run)\n\n"
    detail = {f.path: min(len(f.defs), 12) for f in ranked}

    def block(f: FileInfo) -> str:
        stem = Path(f.path).stem
        used = f"  <- used by {', '.join(sorted(imported_by.get(stem, []))[:6])}" if imported_by.get(stem) else ""
        return "\n".join([f"- {f.path} ({f.lines} lines){used}"] + [f"    {d}" for d in f.defs[:detail[f.path]]])

    text = head + "\n".join(block(f) for f in files)
    for f in reversed(ranked):                                   # the least important files lose their details first
        if len(text) <= budget:
            break
        detail[f.path] = 0
        text = head + "\n".join(block(g) for g in files)
    kept = list(files)
    while len(text) > budget and len(kept) > 1:                  # still too long with bare lines: whole files go, least important first
        worst = [g for g in ranked if g in kept][-1]
        kept = [f for f in kept if f is not worst]
        text = head + "\n".join(block(g) for g in kept) + f"\n... ({len(files) - len(kept)} more files not shown)"
    if len(text) > budget:
        text = text[:budget - 20].rsplit("\n", 1)[0] + "\n... (map cut)"
    return text + "\n"


# ---------------------------------------------------------------------------------------------------------------- renames and stale notes
def _name_of(defstr: str) -> str:
    m = _NAME.match(defstr)
    return m.group(1) if m else ""


def _def_table(defs: list[str]) -> dict[str, str]:
    return {_name_of(d): d[len(_name_of(d)) + (6 if d.startswith("class ") else 0):] for d in defs if _name_of(d)}


def renamed_names(old: list[str], new: list[str]) -> dict[str, str]:
    """{old name: new name} for one file: a removed name whose signature (args + doc) reappears under a new name, or the only removed and the only added name."""
    a, b = _def_table(old), _def_table(new)
    gone, fresh = [n for n in a if n not in b], [n for n in b if n not in a]
    out: dict[str, str] = {}
    for g in list(gone):
        twin = [f for f in fresh if b[f] == a[g] and (a[g] or len(gone) == 1)]
        if len(twin) == 1:
            out[g] = twin[0]
            gone.remove(g)
            fresh.remove(twin[0])
    if len(gone) == 1 and len(fresh) == 1:
        out[gone[0]] = fresh[0]
    return out


def update_notes(notes: str, name_map: dict[str, str], file_map: dict[str, str], gone_names: list[str], gone_files: list[str]) -> str:
    """The model-written notes after a rename pass: new names/paths in place of the old ones; lines about something that no longer exists are marked, once."""
    for old, new in sorted(file_map.items(), key=lambda kv: -len(kv[0])):
        notes = notes.replace(old, new)
    for old, new in name_map.items():
        if len(old) >= 3:
            notes = re.sub(rf"(?<![\w.]){re.escape(old)}(?!\w)", new, notes)
    lines = notes.split("\n")
    for what, pat in [(n, rf"(?<![\w.]){re.escape(n)}(?!\w)") for n in gone_names if len(n) >= 4] + [(f, re.escape(f)) for f in gone_files]:
        mark = f"[stale: `{what}` is gone]"
        lines = [ln + " " + mark if re.search(pat, ln) and "[stale:" not in ln else ln for ln in lines]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------- the stored map
def _load(store: Path) -> dict[str, dict]:
    try:
        data = json.loads((store / "map.json").read_text())
    except (OSError, ValueError):
        return {}
    files = data.get("files") if isinstance(data, dict) else None
    return files if isinstance(files, dict) and all(isinstance(v, dict) for v in files.values()) else {}


def refresh(root: Path, store: Path | None = None, budget: int = 8000) -> Changes:
    """Bring the stored map up to date with the code: only changed files are re-read, renames are carried into the notes. Writes MAP.md and map.json when anything changed."""
    root = Path(root).resolve()
    store = store or store_dir(root)
    store.mkdir(parents=True, exist_ok=True)
    old = _load(store)
    files = scan(root, old)
    now = {f.path: f for f in files}
    mapfile = store / "MAP.md"
    ch = Changes(changed=sorted([p for p, f in now.items() if old.get(p, {}).get("sha") != f.sha] + [p for p in old if p not in now]))
    if not ch.changed and mapfile.exists():
        return ch
    removed, added = [p for p in old if p not in now], [p for p in now if p not in old]
    for r in removed:
        twin = [a for a in added if now[a].sha == old[r].get("sha")]
        if len(twin) == 1:
            ch.renamed_files[r] = twin[0]
    gone_files = [r for r in removed if r not in ch.renamed_files]
    gone_names: list[str] = []
    for path in set(now) & set(old):
        if now[path].sha != old[path].get("sha"):
            ch.renamed_names.update(renamed_names(old[path].get("defs", []), now[path].defs))
            gone_names += [n for n in _def_table(old[path].get("defs", [])) if n not in _def_table(now[path].defs) and n not in ch.renamed_names]
    notes = ""
    if mapfile.exists():
        existing = mapfile.read_text()
        if NOTES_MARK in existing:
            notes = existing[existing.index(NOTES_MARK):]
    if notes and (ch.renamed_files or ch.renamed_names or gone_names or gone_files):
        notes = update_notes(notes, ch.renamed_names, ch.renamed_files, gone_names, gone_files)
        ch.stale = [w for w in gone_names + gone_files if f"[stale: `{w}`" in notes]
    mapfile.write_text(render(files, budget) + ("\n" + notes if notes else ""))
    (store / "map.json").write_text(json.dumps({"version": 2, "files": {f.path: f.__dict__ for f in files}}))
    return ch


def refresh_safe(root: Path) -> Changes | None:
    """For the hooks at the start and end of a run: a problem with the map must never stop a run."""
    try:
        return refresh(root)
    except Exception:  # noqa: BLE001
        return None


def text(root: Path, budget: int = 10000, store: Path | None = None, notes_budget: int = 3000) -> str:
    """The stored map for the system prompt: the factual part, then as much of the model's notes as fits. '' when there is none (no source files)."""
    f = (store or store_dir(root)) / "MAP.md"
    try:
        body = f.read_text()
    except OSError:
        return ""
    head, _, notes = body.partition(NOTES_MARK)
    if "source files; " in head and head.count("\n- ") == 0:
        return ""                                              # a project without source files: the plain repo map is better
    room = max(budget - len(head), 0)
    notes = (NOTES_MARK + notes)[:min(notes_budget, room)] if notes and room > 200 else ""
    out = head[:budget] + (("\n" + notes) if notes else "")
    return out.rstrip() + "\n"
