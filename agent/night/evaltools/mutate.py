"""Injected-bug benchmark: python -m agent.night.evaltools.mutate {inject,score,restore}."""
import argparse
import ast
import json
import os
import random
import re
import sys

from . import runner as R

TEST_TIMEOUT = 300
MAX_TRIES = 40
SKIP_DIRS = {"tests", "test", "__pycache__", "node_modules", "venv", "site-packages"}
CMP = {ast.Eq: (r"==", "!="), ast.NotEq: (r"!=", "=="), ast.Lt: (r"<(?!=)", ">="),
       ast.GtE: (r">=", "<"), ast.Gt: (r">(?!=)", "<="), ast.LtE: (r"<=", ">"),
       ast.In: (r"\bin\b", "not in"), ast.NotIn: (r"\bnot\s+in\b", "in")}


def _b(line, c0, c1):
    return line.encode()[c0:c1].decode()


class Finder(ast.NodeVisitor):
    def __init__(self, lines):
        self.lines, self.stack, self.out, self.scopes = lines, [], [], {}

    def add(self, ln, c0, c1, new, kind):
        self.out.append({"lineno": ln, "c0": c0, "c1": c1, "new": new, "kind": kind,
                         "function": ".".join(self.stack) or "<module>"})

    def between(self, a, b, op_re, new, kind):
        """Operator text sitting between nodes a and b on one line (only parens/space around it)."""
        if a.end_lineno != b.lineno:
            return
        seg = _b(self.lines[a.end_lineno - 1], a.end_col_offset, b.col_offset)
        m = re.search(op_re, seg)
        if m and re.sub(r"[()\s]", "", seg) == re.sub(r"[()\s]", "", m.group()):
            self.add(a.end_lineno, a.end_col_offset + len(seg[:m.start()].encode()),
                     a.end_col_offset + len(seg[:m.end()].encode()), new, kind)

    def scope(self, n):
        self.stack.append(n.name)
        self.scopes.setdefault(".".join(self.stack), n)
        self.generic_visit(n)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = scope

    def visit_Compare(self, n):
        left = n.left
        for op, c in zip(n.ops, n.comparators):
            if type(op) in CMP:
                self.between(left, c, CMP[type(op)][0], CMP[type(op)][1], "cmp_flip")
            if isinstance(c, ast.Constant) and type(c.value) is int and c.lineno == c.end_lineno:
                self.add(c.lineno, c.col_offset, c.end_col_offset, str(c.value + 1), "off_by_one")
            left = c
        self.generic_visit(n)

    def visit_BoolOp(self, n):
        a, b = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
        for x, y in zip(n.values, n.values[1:]):
            self.between(x, y, rf"\b{a}\b", b, "bool_swap")
        self.generic_visit(n)

    def visit_BinOp(self, n):
        txt = lambda x: isinstance(x, ast.JoinedStr) or (isinstance(x, ast.Constant) and isinstance(x.value, (str, bytes)))
        if isinstance(n.op, (ast.Add, ast.Sub)) and not (txt(n.left) or txt(n.right)):
            self.between(n.left, n.right, r"\+" if isinstance(n.op, ast.Add) else "-",
                         "-" if isinstance(n.op, ast.Add) else "+", "plus_minus")
        self.generic_visit(n)

    def _cond(self, n):
        t = n.test
        for c in ast.walk(t):
            if isinstance(c, ast.Constant) and isinstance(c.value, bool) and c.lineno == c.end_lineno:
                self.add(c.lineno, c.col_offset, c.end_col_offset, str(not c.value), "bool_flip")
        if isinstance(n, ast.If) and t.lineno == t.end_lineno:
            self.add(t.lineno, t.col_offset, t.end_col_offset,
                     f"not ({_b(self.lines[t.lineno - 1], t.col_offset, t.end_col_offset)})", "negate_if")
        self.generic_visit(n)

    visit_If = visit_While = _cond

    def visit_Return(self, n):
        v = n.value
        if v is not None and not (isinstance(v, ast.Constant) and v.value is None) and v.lineno == v.end_lineno:
            self.add(v.lineno, v.col_offset, v.end_col_offset, "None", "return_none")
        self.generic_visit(n)


def parse(text):
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def source_files(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in SKIP_DIRS)
        for f in sorted(files):
            p = os.path.join(d, f)
            if (f.endswith(".py") and not f.startswith("test_") and not f.endswith("_test.py")
                    and f != "conftest.py" and os.path.getsize(p) <= 400 * 1024):
                yield os.path.relpath(p, root).replace(os.sep, "/")


def tests_hashes(root):
    out = {}
    for d, dirs, files in os.walk(os.path.join(root, "tests")):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = R.sha1(open(p, "rb").read())
    return out


def run_tests(root, cmd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    rc, out, to = R.run(cmd, str(root), "", TEST_TIMEOUT, env, shell=True)
    R.log(out[-400:])
    return rc, to


def load(store):
    f = store / "mutations.json"
    if not f.is_file():
        print("no mutations.json in store; run inject first", file=sys.stderr)
        sys.exit(2)
    return json.loads(f.read_text())


def restore(root, store):
    st = load(store)
    n = 0
    for rel, sha in st["originals"].items():
        p = (root / rel).resolve()
        if root not in p.parents:
            continue
        p.write_bytes((store / "originals" / sha).read_bytes())
        n += 1
    return n


def candidates(root):
    out = []
    for rel in source_files(root):
        try:
            text = (root / rel).read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        tree = parse(text)
        if tree is None:
            continue
        f = Finder(text.split("\n"))
        f.visit(tree)
        out += [dict(c, file=rel) for c in f.out]
    out.sort(key=lambda c: (c["file"], c["lineno"], c["c0"], c["kind"]))
    return out


def inject(root, store, a):
    if (store / "mutations.json").is_file():
        restore(root, store)  # never stack on an earlier injection
    rc, _ = run_tests(root, a.test)
    if rc != 0:
        R.emit({"injected": 0, "tries": 0, "tests_red": True, "baseline_green": False, "ids": []})
        return
    cands = candidates(root)
    random.Random(a.seed).shuffle(cands)
    (store / "originals").mkdir(exist_ok=True)
    state = {"mutations": [], "tests_sha1": tests_hashes(root), "originals": {}}
    used, tries, kept = set(), 0, []

    def save():
        (store / "mutations.json").write_text(json.dumps(state, indent=1))

    # Each candidate is tested ALONE on the pristine tree; kept ones are applied together at the end.
    for c in cands:
        if tries >= MAX_TRIES or len(kept) >= a.n:
            break
        if (c["file"], c["function"]) in used:
            continue
        path = root / c["file"]
        cur = path.read_bytes()
        lines = cur.decode("utf-8").split("\n")
        old = lines[c["lineno"] - 1]
        raw = old.encode()
        new = (raw[:c["c0"]] + c["new"].encode() + raw[c["c1"]:]).decode()
        lines[c["lineno"] - 1] = new
        text = "\n".join(lines)
        if new == old or parse(text) is None:
            continue
        if c["file"] not in state["originals"]:
            sha = R.sha1(cur)
            (store / "originals" / sha).write_bytes(cur)
            state["originals"][c["file"]] = sha
            save()
        try:
            path.write_bytes(text.encode("utf-8"))
            tries += 1
            rc, to = run_tests(root, a.test)
        finally:
            path.write_bytes(cur)
        if rc != 0 and not to:
            used.add((c["file"], c["function"]))
            kept.append((c, old, new))
    for c, old, new in kept:
        path = root / c["file"]
        lines = path.read_bytes().decode("utf-8").split("\n")
        lines[c["lineno"] - 1] = new
        path.write_bytes("\n".join(lines).encode("utf-8"))
        state["mutations"].append({
            "id": f"m{len(state['mutations']) + 1}", "file": c["file"], "lineno": c["lineno"],
            "kind": c["kind"], "original_line": old.rstrip("\r"), "mutated_line": new.rstrip("\r"),
            "function": c["function"]})
    save()
    k = len(state["mutations"])
    red = run_tests(root, a.test)[0] != 0 if k else False
    R.emit({"injected": k, "tries": tries, "tests_red": red, "ids": [m["id"] for m in state["mutations"]]})


def function_lines(path, qual):
    """Stripped source lines of the function/class `qual` (whole file if module level or not found)."""
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.split("\n")
    tree = parse(text)
    if tree is not None and qual != "<module>":
        f = Finder(lines)
        f.visit(tree)
        n = f.scopes.get(qual)
        if n is not None:
            lines = lines[n.lineno - 1:n.end_lineno]
    return [l.strip() for l in lines]


def score(root, store, a):
    st = load(store)
    details, unfixed = [], []
    for m in st["mutations"]:
        ls = function_lines(root / m["file"], m["function"])
        # fixed = the mutated line is gone from the function (restored original or removed)
        fixed = ls is not None and m["mutated_line"].strip() not in ls
        details.append({"id": m["id"], "file": m["file"], "function": m["function"], "fixed": fixed})
        if not fixed:
            unfixed.append(m["id"])
    rc, to = run_tests(root, a.test)
    R.emit({"injected": len(details), "fixed": len(details) - len(unfixed),
            "tests_green": rc == 0 and not to, "tests_changed": tests_hashes(root) != st["tests_sha1"],
            "unfixed": unfixed, "details": details})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mutate")
    sp = ap.add_subparsers(dest="sub", required=True)
    i = R.common_args(sp.add_parser("inject"))
    i.add_argument("--n", type=int, default=3)
    i.add_argument("--seed", type=int, default=7)
    i.add_argument("--test", required=True)
    s = R.common_args(sp.add_parser("score"))
    s.add_argument("--test", required=True)
    R.common_args(sp.add_parser("restore"))
    a = ap.parse_args(argv)
    root, store = R.setup(ap, a)
    if a.sub == "inject":
        if a.n < 1:
            ap.error("--n must be >= 1")
        inject(root, store, a)
    elif a.sub == "score":
        score(root, store, a)
    else:
        R.emit({"restored": restore(root, store)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
