"""The calc tool: exact results, the function set, and refusing everything that is not arithmetic."""

from __future__ import annotations

import pytest

from agent.research import taint
from agent.tools import calc
from agent.tools.calc import CalcError, evaluate, run


@pytest.fixture(autouse=True)
def _clean():
    taint.clear()
    yield
    taint.clear()


@pytest.mark.parametrize("code,expected", [
    ("1 + 2 * 3", "7"), ("(1 + 2) * 3", "9"), ("7 // 2", "3"), ("7 % 4", "3"), ("2 ** 10", "1024"), ("10 / 4", "2.5"),
    ("-3 + abs(-5)", "2"), ("round(3.14159, 2)", "3.14"), ("sqrt(16)", "4"), ("1 < 2 < 3", "True"), ("3 if 1 > 2 else 4", "4"),
    ("sum([1, 2, 3])", "6"), ("mean([2, 4, 6])", "4"), ("median([5, 1, 3])", "3"), ("max(3, 9, 4)", "9"), ("len([1, 2, 3])", "3"),
    ("percent(25, 200)", "12.5"), ("pct_change(499, 429)", "-14.02805611"), ("cagr(100, 200, 5)", "14.8698355"),
    ("sorted([3, 1, 2])", "[1, 2, 3]"), ("[x * x for x in range(4)]", "[0, 1, 4, 9]"),
    ("sum(x for x in range(10) if x % 2 == 0)", "20"), ("[a + b for a in [1, 2] for b in [10, 20]]", "[11, 21, 12, 22]"),
    ("p = [499, 459, 429]; pct_change(p[0], p[-1])", "p = [499, 459, 429]\n-14.02805611"), ("x = 5\ny = x * 2\ny + 1", "x = 5\ny = 10\n11"),
    ("pi", "3.141592654"), ("0.1 + 0.2", "0.3"), ("[1, 2, 3, 4][1:3]", "[2, 3]"), ("stdev([2, 4, 4, 4, 5, 5, 7, 9])", "2.138089935"),
    ("3 in [1, 2, 3]", "True"), ("factorial(5)", "120"),
])
def test_results(code, expected):
    assert run(code) == expected


@pytest.mark.parametrize("code,fragment", [
    ("__import__('os').system('id')", "only the listed functions"),
    ("().__class__.__bases__", "Attribute is not supported"),
    ("open('/etc/passwd')", "only the listed functions"),
    ("print(1)", "only the listed functions"),
    ("os", "unknown name 'os'"),
    ("lambda x: x", "Lambda is not supported"),
    ("[1,2,3].append(4)", "only the listed functions"),
    ("2 ** 100000000", "too large"), ("9 ** 9 ** 9", "too large"),
    ("range(10**9)", "limited"), ("[x for x in range(900000) for y in range(900000)]", "too many steps"),
    ("'a' * 10**9", "too large"),
    ("import os", "only assignments"), ("def f(): pass", "only assignments"), ("x, y = 1, 2", "only assignments"),
    ("pi = 3", "built-in name"), ("1 +", "syntax error"), ("", "give an expression"),
    ("foo(1)", "only the listed functions"), ("mean([1], key=2)", "keyword arguments"),
    ("factorial(99999)", "limited"),
])
def test_everything_that_is_not_arithmetic_is_refused(code, fragment):
    out = run(code)
    assert out.startswith("Error") and fragment in out, out


def test_runtime_errors_are_reported_not_raised():
    assert "ZeroDivisionError" in run("1 / 0")
    assert "Error" in run("[1, 2][5]") and "Error" in run("stdev([1])") and "Error" in run("sqrt(-1)")


def test_size_limits():
    with pytest.raises(CalcError, match="too long"):
        evaluate("1+" * 5000 + "1")
    with pytest.raises(CalcError, match="complicated"):
        evaluate("[" + ",".join(["-1"] * 1250) + "]")   # short text, but ~3700 syntax nodes


def test_results_are_recorded_as_computed_for_the_present_tool():
    run("p = [499, 459, 429]; pct_change(p[0], p[-1])")
    assert taint.has_computed_number(-14.02805611) is False            # rounded to 10 digits in the text, not the exact float
    assert taint.has_computed_number(499) and taint.has_computed_number(429)
    assert not taint.has_computed_number(777)
    taint.clear()
    assert not taint.has_computed_number(499)


def test_the_tool_is_registered_and_safe():
    assert calc.CALC_TOOL.name == "calc" and calc.CALC_TOOL.risk == "safe"
    assert calc.CALC_TOOL.handler(code="2+2") == "4"
