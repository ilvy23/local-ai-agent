from __future__ import annotations

import math
from dataclasses import dataclass


def pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0 or n <= 0 or k > n:
        raise ValueError("require 0 < k <= n")
    if n - c < k:
        return 1.0
    prob = 1.0
    for i in range(n - c + 1, n + 1):
        prob *= 1.0 - k / i
    return 1.0 - prob


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Interval:
    if total <= 0:
        return Interval(0.0, 0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return Interval(point=p, low=max(0.0, center - margin), high=min(1.0, center + margin))


@dataclass(frozen=True)
class McNemarResult:
    statistic: float
    p_value: float
    b: int
    c: int


def mcnemar(b: int, c: int) -> McNemarResult:
    if b + c == 0:
        return McNemarResult(statistic=0.0, p_value=1.0, b=b, c=c)
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    statistic = max(statistic, 0.0)
    p_value = math.erfc(math.sqrt(statistic / 2))
    return McNemarResult(statistic=statistic, p_value=p_value, b=b, c=c)


def intervals_overlap(a: Interval, b: Interval) -> bool:
    return not (a.high < b.low or b.high < a.low)
