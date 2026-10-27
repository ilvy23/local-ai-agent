"""Milestone mode: interactive checkpoints between coding-agent phases.

MilestoneSession (Mode A) — pauses after each major step and asks the user to
  review before continuing.  Raised KeyboardInterrupt aborts the run cleanly.

AutoSession (Mode B) — auto-approves every checkpoint.  Only available after
  the user has completed at least one Mode-A run (tracked in app_state).

Usage in coding_cli.py:
    session = MilestoneSession(console)
    session.complete_milestone("Diagnosed the issue: ...")
    if not session.await_review():
        raise KeyboardInterrupt  # abort; caught by run_code / run_goal_cli
"""

from __future__ import annotations

from rich.console import Console


class MilestoneSession:
    """Interactive review gate between coding-agent milestones (Mode A).

    Each major phase (diagnosis → generation → verification) calls
    `complete_milestone` to announce what just happened, then calls
    `await_review` to block until the user confirms (or aborts).
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._n = 0

    @property
    def milestone_count(self) -> int:
        return self._n

    def complete_milestone(self, summary: str) -> None:
        """Print a milestone header and summary."""
        self._n += 1
        self._console.print()
        self._console.print(
            f"[bold cyan]── Milestone {self._n} ──[/bold cyan]  [dim]{summary}[/dim]"
        )

    def await_review(self) -> bool:
        """Block until the user types 'ok' (continue) or 'stop' (abort).

        Returns True to continue, False to abort.  An empty Enter defaults to
        continue so a fast confirm is just pressing Enter.  EOF / Ctrl-C → abort.
        """
        self._console.print(
            "[dim]Continue?  [bold]ok[/bold] / Enter  to proceed, "
            "[bold]stop[/bold] to abort:[/dim]",
            end=" ",
        )
        try:
            response = self._console.input("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return response not in ("stop", "s", "n", "no", "abort", "quit", "q")


class AutoSession(MilestoneSession):
    """Autonomous mode (Mode B) — checkpoints pass through without blocking.

    Prints the milestone summary so the user can see progress, but never
    waits for input.  Only offered in the menu after Mode A has been used once.
    """

    def await_review(self) -> bool:
        self._console.print("[dim](auto — continuing)[/dim]")
        return True


# ── Mode-B unlock tracking via app_state ─────────────────────────────────────

_AUTO_UNLOCK_KEY = "milestone_mode_used"


def auto_mode_unlocked(store) -> bool:
    """True if the user has completed at least one Mode-A run."""
    row = store.conn.execute(
        "SELECT v FROM app_state WHERE k = ?", (_AUTO_UNLOCK_KEY,)
    ).fetchone()
    return row is not None


def mark_milestone_mode_used(store) -> None:
    """Record that Mode A has been used — unlocks Mode B."""
    store.conn.execute(
        "INSERT OR REPLACE INTO app_state (k, v) VALUES (?, '1')",
        (_AUTO_UNLOCK_KEY,),
    )
    store.conn.commit()
