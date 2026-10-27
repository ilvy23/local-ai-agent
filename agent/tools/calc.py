"""`calc`: exact arithmetic and statistics, so numbers are computed instead of guessed.

A language model is unreliable at arithmetic and happily invents a plausible total. This tool evaluates a small
Python-like expression language - numbers, + - * / // % **, comparisons, lists, comprehensions over range(), and a fixed
set of functions (sum, mean, median, stdev, round, sqrt, log, percent, pct_change, cagr, ...). It is a whitelist
interpreter written over the `ast` module: no imports, no attribute access, no names except the ones below, no file or
network access, bounded work. Results are recorded as "computed" so a chart or table built from them is marked as such
(see tools/present.py provenance).

Examples: `a = [499, 459, 429]; pct_change(a[0], a[-1])`   `sum(x * 1.19 for x in prices)`   `cagr(100, 180, 5)`
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import Any

from agent.research import taint
from agent.tools.registry import Tool

MAX_NODES = 2_000
MAX_STEPS = 200_000          # loop iterations in comprehensions, across the whole call
MAX_RANGE = 1_000_000
MAX_POW_BITS = 200_000


class CalcError(ValueError):
    """The expression cannot be evaluated; the message says why in plain words."""


def _percent(part: float, whole: float) -> float:
    return 100.0 * part / whole


def _pct_change(old: float, new: float) -> float:
    return 100.0 * (new - old) / old


def _cagr(start: float, end: float, years: float) -> float:
    return 100.0 * ((end / start) ** (1.0 / years) - 1.0)


_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len, "sorted": sorted, "range": range,
    "int": int, "float": float, "list": list, "reversed": lambda x: list(reversed(x)), "all": all, "any": any,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp, "floor": math.floor,
    "ceil": math.ceil, "sin": math.sin, "cos": math.cos, "tan": math.tan, "hypot": math.hypot, "gcd": math.gcd,
    "factorial": lambda n: math.factorial(n) if 0 <= n <= 5000 else (_ for _ in ()).throw(CalcError("factorial is limited to 0..5000")),
    "mean": statistics.fmean, "median": statistics.median, "stdev": statistics.stdev, "pstdev": statistics.pstdev,
    "variance": statistics.variance, "percent": _percent, "pct_change": _pct_change, "cagr": _cagr,
}
_CONSTS = {"pi": math.pi, "e": math.e, "True": True, "False": False, "None": None}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


class _Eval:
    def __init__(self) -> None:
        self.names: dict[str, Any] = dict(_CONSTS)
        self.steps = 0

    def tick(self, n: int = 1) -> None:
        self.steps += n
        if self.steps > MAX_STEPS:
            raise CalcError("this takes too many steps; use a smaller range or fewer items")

    def ev(self, node: ast.AST, local: dict[str, Any] | None = None) -> Any:
        self.tick()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str, bool)) or node.value is None:
                return node.value
            raise CalcError("only numbers, text and true/false are allowed as values")
        if isinstance(node, ast.Name):
            if local and node.id in local:
                return local[node.id]
            if node.id in self.names:
                return self.names[node.id]
            if node.id in _FUNCS:
                return _FUNCS[node.id]
            raise CalcError(f"unknown name '{node.id}' (functions: {', '.join(sorted(_FUNCS))})")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            a, b = self.ev(node.left, local), self.ev(node.right, local)
            if isinstance(node.op, ast.Pow):
                self._check_pow(a, b)
            if isinstance(node.op, ast.Mult) and isinstance(a, (str, list)) and isinstance(b, int) and len(a) * max(b, 0) > 100_000:
                raise CalcError("result would be too large")
            return _BIN[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self.ev(node.operand, local))
        if isinstance(node, ast.BoolOp):
            vals = (self.ev(v, local) for v in node.values)
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            left = self.ev(node.left, local)
            for op, right_node in zip(node.ops, node.comparators):
                if type(op) not in _CMP:
                    raise CalcError("this comparison is not supported")
                right = self.ev(right_node, local)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.ev(node.body, local) if self.ev(node.test, local) else self.ev(node.orelse, local)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise CalcError("only the listed functions can be called")
            if node.keywords:
                raise CalcError("keyword arguments are not supported")
            args = [self.ev(a, local) for a in node.args]
            if node.func.id == "range":
                if any(not isinstance(a, int) for a in args) or (args and abs(args[-1] if len(args) == 1 else args[1] - args[0]) > MAX_RANGE):
                    raise CalcError(f"range is limited to {MAX_RANGE:,} items")
            return _FUNCS[node.func.id](*args)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [self.ev(e, local) for e in node.elts]
            return items if isinstance(node, ast.List) else tuple(items) if isinstance(node, ast.Tuple) else set(items)
        if isinstance(node, ast.Dict):
            return {self.ev(k, local): self.ev(v, local) for k, v in zip(node.keys, node.values) if k is not None}
        if isinstance(node, ast.Subscript):
            seq, idx = self.ev(node.value, local), self.ev(node.slice, local)
            return seq[idx]
        if isinstance(node, ast.Slice):
            return slice(*(self.ev(p, local) if p is not None else None for p in (node.lower, node.upper, node.step)))
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            out = list(self._comprehend(node.generators, 0, dict(local or {}), node.elt))
            return set(out) if isinstance(node, ast.SetComp) else out
        raise CalcError(f"{type(node).__name__} is not supported")

    def _check_pow(self, base: Any, exp: Any) -> None:
        if isinstance(base, int) and isinstance(exp, int) and exp > 0 and abs(base) > 1 and exp * base.bit_length() > MAX_POW_BITS:
            raise CalcError("that power is too large")
        if isinstance(exp, (int, float)) and abs(exp) > 100_000:
            raise CalcError("that exponent is too large")

    def _comprehend(self, gens: list[ast.comprehension], i: int, local: dict[str, Any], elt: ast.AST):
        if i == len(gens):
            yield self.ev(elt, local)
            return
        g = gens[i]
        if not isinstance(g.target, ast.Name):
            raise CalcError("only simple loop variables are supported")
        for item in self.ev(g.iter, local):
            self.tick()
            inner = {**local, g.target.id: item}
            if all(self.ev(cond, inner) for cond in g.ifs):
                yield from self._comprehend(gens, i + 1, inner, elt)


def evaluate(source: str) -> list[tuple[str, Any]]:
    """Run `source` (statements separated by `;` or newlines). Returns [(label, value)] for every statement."""
    source = (source or "").strip()
    if not source:
        raise CalcError("give an expression to calculate")
    if len(source) > 4_000:
        raise CalcError("the expression is too long (max 4000 characters)")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise CalcError(f"syntax error: {exc.msg} (column {exc.offset})") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalcError("the expression is too complicated")
    ev, results = _Eval(), []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name in _FUNCS or name in _CONSTS:
                raise CalcError(f"'{name}' is a built-in name; pick another variable name")
            value = ev.ev(stmt.value)
            ev.names[name] = value
            results.append((name, value))
        elif isinstance(stmt, ast.Expr):
            results.append(("", ev.ev(stmt.value)))
        else:
            raise CalcError("only assignments (x = ...) and expressions are supported")
    return results


def _show(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_show(v) for v in value[:50]) + (", ..." if len(value) > 50 else "")
        return f"[{inner}]" if isinstance(value, list) else f"({inner})"
    return str(value)


def run(code: str = "", **_: Any) -> str:
    try:
        results = evaluate(code)
    except CalcError as exc:
        return f"Error: {exc}"
    except (ZeroDivisionError, OverflowError, ValueError, TypeError, IndexError, KeyError, statistics.StatisticsError) as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    lines = [f"{label} = {_show(v)}" if label else _show(v) for label, v in results]
    taint.add_computed("\n".join(lines))
    return "\n".join(lines)


CALC_TOOL = Tool(
    name="calc",
    description=(
        "Calculate exactly instead of estimating: arithmetic, percentages, totals, averages, growth rates. Use it for ANY "
        "calculation whose result you will state or chart. Python-like syntax; statements separated by ; or newlines; the "
        "last value is shown. Functions: sum mean median stdev variance min max round abs sqrt log log10 exp floor ceil "
        "percent(part,whole) pct_change(old,new) cagr(start,end,years) range len sorted; lists and comprehensions work, "
        "e.g. `p = [499, 459, 429]; pct_change(p[0], p[-1])` or `sum(x * 1.19 for x in [10, 20])`."
    ),
    parameters={"type": "object", "properties": {"code": {"type": "string", "description": "The calculation."}}, "required": ["code"]},
    handler=run,
    risk="safe",
)
