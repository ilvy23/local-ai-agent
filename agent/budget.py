"""Agent self-control: action/token/time budget with hard limits.

A Budget instance tracks three independent resource axes: discrete tool
actions, cumulative tokens, and wall-clock elapsed seconds. Any single axis
crossing its ceiling raises BudgetExhausted so the caller can exit cleanly.
"""

from __future__ import annotations

import time


class BudgetExhausted(Exception):
    """Raised by Budget.tick() when any configured limit is exceeded."""


class Budget:
    def __init__(
        self,
        max_actions: int = 50,
        max_tokens: int = 50_000,
        max_seconds: float = 1800,
    ) -> None:
        self._max_actions = max_actions
        self._max_tokens = max_tokens
        self._max_seconds = max_seconds
        self._actions = 0
        self._tokens = 0
        self._start = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, tokens_used: int = 0) -> None:
        """Record one action and `tokens_used` tokens; raise if any limit exceeded."""
        self._actions += 1
        self._tokens += tokens_used

        elapsed = time.monotonic() - self._start

        if self._actions > self._max_actions:
            raise BudgetExhausted(
                f"action limit exceeded: {self._actions}/{self._max_actions}"
            )
        if self._tokens > self._max_tokens:
            raise BudgetExhausted(
                f"token limit exceeded: {self._tokens}/{self._max_tokens}"
            )
        if elapsed > self._max_seconds:
            raise BudgetExhausted(
                f"time limit exceeded: {elapsed:.1f}s/{self._max_seconds}s"
            )

    def is_exhausted(self) -> bool:
        """True if any limit has already been reached or exceeded."""
        elapsed = time.monotonic() - self._start
        return (
            self._actions >= self._max_actions
            or self._tokens >= self._max_tokens
            or elapsed >= self._max_seconds
        )

    @property
    def actions(self) -> int:
        return self._actions

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start

    def summary(self) -> str:
        elapsed = time.monotonic() - self._start
        return (
            f"{self._actions}/{self._max_actions} actions, "
            f"{self._tokens}/{self._max_tokens} tokens, "
            f"{elapsed:.0f}s/{self._max_seconds:.0f}s"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "Budget":
        """Construct from config dict, reading config['budget'] with fallback defaults.

        Tolerates the shapes real YAML produces: a `budget:` key with nothing
        under it parses as None, not {}, and `max_actions:` with no value parses
        as None too. Either would otherwise blow up — the first with an
        AttributeError right here, the second later inside tick() comparing an
        int to None, which would abort a whole coding run.
        """
        section = (config or {}).get("budget")
        if not isinstance(section, dict):
            section = {}

        def _limit(key: str, default):
            value = section.get(key, default)
            return default if value is None else value

        return cls(
            max_actions=_limit("max_actions", 50),
            max_tokens=_limit("max_tokens", 50_000),
            max_seconds=_limit("max_seconds", 1800),
        )
