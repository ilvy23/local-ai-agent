"""Integration test: Priority 4 Milestone system fires correctly end-to-end.

Validates that run_goal_cli correctly:
  - calls mode.complete_milestone after a generate_end event
  - allows AutoSession to auto-approve and complete normally (exit 0)
  - turns a False from await_review into a KeyboardInterrupt → exit 130

No live Ollama or GPU is required — build() and the model helpers are fully patched.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.coding.session import Outcome
from agent.coding_cli import run_goal_cli
from agent.events import Events
from agent.modes import AutoSession, MilestoneSession


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_build(root, goal, config=None, llm=None, events: Events = None,
                max_rounds: int = 8) -> Outcome:
    """Fake build: emits the generate-phase events then reports success."""
    events.emit("build_round", n=1, of=1, phase="generate")
    events.emit("generate_end")
    return Outcome(ok=True, stage="built", diff="", branch="test-branch", attempts=1)


def _mock_console():
    c = MagicMock()
    c.print = MagicMock()
    return c


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("agent.config.load_config", return_value={})
@patch("agent.coding.build.build", side_effect=_fake_build)
@patch("agent.coding_cli._ensure_repo", return_value=Path("/tmp/fake-repo"))
@patch("agent.coding_cli._prep_model", return_value=(MagicMock(), "fake-model:7b"))
@patch("agent.coding_cli._release_model")
def test_auto_session_fires_one_milestone(
    mock_release, mock_prep, mock_repo, mock_build, mock_config
):
    """AutoSession auto-approves the generate_end gate; milestone_count must be 1."""
    console = _mock_console()
    session = AutoSession(console)

    rc = run_goal_cli("test goal", mode=session)

    assert rc == 0
    assert session.milestone_count == 1


@patch("agent.config.load_config", return_value={})
@patch("agent.coding.build.build", side_effect=_fake_build)
@patch("agent.coding_cli._ensure_repo", return_value=Path("/tmp/fake-repo"))
@patch("agent.coding_cli._prep_model", return_value=(MagicMock(), "fake-model:7b"))
@patch("agent.coding_cli._release_model")
def test_milestone_abort_returns_130(
    mock_release, mock_prep, mock_repo, mock_build, mock_config
):
    """await_review() returning False (user typed 'stop') → run_goal_cli returns 130."""
    console = _mock_console()
    console.input = MagicMock(return_value="stop")
    session = MilestoneSession(console)

    rc = run_goal_cli("test goal", mode=session)

    assert rc == 130


# ── Mode-B unlock is actually wired to a guided run ──────────────────────────

def test_guided_run_unlocks_auto_mode(tmp_path, monkeypatch):
    """Nothing called mark_milestone_mode_used, so mode B never unlocked."""
    from agent.ui import launcher as menu_mod
    from agent.memory.store import Store
    from agent.modes import auto_mode_unlocked

    store = Store(tmp_path / "agent.db")
    assert auto_mode_unlocked(store) is False

    class _KeepOpen:
        """_open_store's result is used as a context manager; don't close ours."""

        def __enter__(self_inner):
            return store

        def __exit__(self_inner, *exc):
            return False

    import agent.main as main_mod
    monkeypatch.setattr(main_mod, "_open_store", lambda config: _KeepOpen())
    monkeypatch.setattr("agent.config.load_config", lambda: {})

    ran: list[str] = []

    def _fake_run_goal_cli(goal, mode=None):
        ran.append(goal)
        mode.complete_milestone("round 1: generated a solution")
        mode.await_review()
        return 0

    monkeypatch.setattr("agent.coding_cli.run_goal_cli", _fake_run_goal_cli)

    console = MagicMock()
    console.input = MagicMock(return_value="build a snake game")
    menu_mod.build_guided(console)

    assert ran == ["build a snake game"]
    assert auto_mode_unlocked(store) is True
    store.close()


def test_guided_run_that_is_cancelled_does_not_unlock(tmp_path, monkeypatch):
    """An empty goal (cancelled at the prompt) must not unlock auto mode."""
    from agent.ui import launcher as menu_mod
    from agent.memory.store import Store
    from agent.modes import auto_mode_unlocked

    store = Store(tmp_path / "agent.db")

    import agent.main as main_mod
    monkeypatch.setattr(main_mod, "_open_store", lambda config: store)
    monkeypatch.setattr("agent.config.load_config", lambda: {})

    console = MagicMock()
    console.input = MagicMock(return_value="   ")
    menu_mod.build_guided(console)

    assert auto_mode_unlocked(store) is False
    store.close()


def test_auto_build_resolves_the_db_path_against_the_project_root(monkeypatch, tmp_path):
    """It opened Store(cfg['data']['db_path']) directly, resolving against cwd."""
    from agent.ui import launcher as menu_mod

    seen: list[object] = []

    class _Dummy:
        def __enter__(self_inner):
            seen.append("opened")
            return MagicMock(**{"conn.execute.return_value.fetchone.return_value": None})

        def __exit__(self_inner, *exc):
            return False

    import agent.main as main_mod
    monkeypatch.setattr(main_mod, "_open_store", lambda config: _Dummy())
    monkeypatch.setattr("agent.config.load_config",
                        lambda: {"data": {"db_path": "data/agent.db"}})
    monkeypatch.chdir(tmp_path)

    console = MagicMock()
    menu_mod.build_auto(console)

    assert seen == ["opened"]          # went through _open_store, not raw Store()
    assert not (tmp_path / "data").exists()  # no stray DB created next to cwd
