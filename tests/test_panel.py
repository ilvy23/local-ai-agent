"""Tests for agent.panel.StatusPanel."""
from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from agent.panel import StatusPanel
from agent.worker_status import StatusRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the StatusRegistry singleton before and after each test."""
    StatusRegistry._instance = None
    yield
    StatusRegistry._instance = None


@pytest.fixture(autouse=True)
def isolated_trajectory(tmp_path, monkeypatch):
    """render() tails the trajectory log — keep it off the real ~/.companion one.

    Without this the panel tests render whatever happens to be in the developer's
    own log, so their output (and any assertion about it) depends on machine state.
    """
    import agent.trajectory as traj_mod
    monkeypatch.setattr(traj_mod, "_TRAJECTORY_PATH", tmp_path / "trajectory.jsonl")


def _render_to_text(renderable) -> str:
    """Render a Rich renderable to plain text for assertions."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, highlight=False)
    console.print(renderable)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_instantiate_no_args():
    """StatusPanel() can be instantiated with no arguments."""
    panel = StatusPanel()
    assert panel is not None


def test_instantiate_with_args():
    """StatusPanel accepts log_lines and refresh_rate kwargs."""
    panel = StatusPanel(log_lines=20, refresh_rate=1.0)
    assert panel._log_lines == 20
    assert panel._refresh_rate == 1.0


# ---------------------------------------------------------------------------
# render() return value
# ---------------------------------------------------------------------------


def test_render_returns_non_none():
    """render() returns a non-None renderable."""
    panel = StatusPanel()
    result = panel.render()
    assert result is not None


def test_render_produces_output():
    """render() produces non-empty text when printed."""
    panel = StatusPanel()
    output = _render_to_text(panel.render())
    assert output.strip()  # must have something


# ---------------------------------------------------------------------------
# render() content: agent data
# ---------------------------------------------------------------------------


def test_render_shows_agent_id():
    """render() includes the registered agent's ID."""
    reg = StatusRegistry.get()
    reg.register("Agt-Test", "qwen3.5:9b")
    reg.update("Agt-Test", milestone="M1 Planning", last_action="read specs", status="running")

    panel = StatusPanel()
    output = _render_to_text(panel.render())

    assert "Agt-Test" in output


def test_render_shows_agent_milestone():
    """render() shows the agent's current milestone."""
    reg = StatusRegistry.get()
    reg.register("Agt-M", "llama3")
    reg.update("Agt-M", milestone="M2 Coding")

    panel = StatusPanel()
    output = _render_to_text(panel.render())
    assert "M2 Coding" in output


def test_render_shows_agent_status():
    """render() shows the agent's status field."""
    reg = StatusRegistry.get()
    reg.register("Agt-S", "llama3")
    reg.update("Agt-S", status="running")

    panel = StatusPanel()
    output = _render_to_text(panel.render())
    assert "running" in output


def test_render_no_agents_still_renders():
    """render() works with an empty registry (no agents registered)."""
    panel = StatusPanel()
    output = _render_to_text(panel.render())
    assert output.strip()  # table header and log panel at minimum


# ---------------------------------------------------------------------------
# render() content: log panel
# ---------------------------------------------------------------------------


def test_render_includes_live_log_header():
    """render() output contains 'Live Log' panel title."""
    panel = StatusPanel()
    output = _render_to_text(panel.render())
    assert "Live Log" in output


def test_render_includes_hint_line():
    """render() output contains the keyboard-shortcut hint line."""
    panel = StatusPanel()
    output = _render_to_text(panel.render())
    assert "approve" in output
    assert "reject" in output
    assert "kill all" in output


# ---------------------------------------------------------------------------
# Events exposed on the panel
# ---------------------------------------------------------------------------


def test_review_events_are_threading_events():
    """review_approved and review_rejected are threading.Event instances."""
    import threading

    panel = StatusPanel()
    assert isinstance(panel.review_approved, threading.Event)
    assert isinstance(panel.review_rejected, threading.Event)


# ---------------------------------------------------------------------------
# render() content: log tail actually shows records
# ---------------------------------------------------------------------------


def test_render_shows_logged_tool_calls():
    """The Live Log panel tails the trajectory file."""
    from agent.trajectory import log_action

    log_action("tool_call", "list_dir", "{'path': '.'}", "README.md")
    output = _render_to_text(StatusPanel().render())
    assert "list_dir" in output


def test_render_shows_placeholder_when_log_is_empty():
    output = _render_to_text(StatusPanel().render())
    assert "no log entries yet" in output


def test_render_honours_the_log_lines_cap():
    from agent.trajectory import log_action

    for i in range(10):
        log_action("tool_call", f"tool_{i}", "in", "out")
    output = _render_to_text(StatusPanel(log_lines=3).render())
    assert "tool_9" in output
    assert "tool_5" not in output


def test_render_survives_an_unreadable_log(monkeypatch, tmp_path):
    import agent.trajectory as traj_mod

    bad = tmp_path / "broken.jsonl"
    bad.write_text("{not json at all\nalso not json\n")
    monkeypatch.setattr(traj_mod, "_TRAJECTORY_PATH", bad)
    assert _render_to_text(StatusPanel().render()).strip()


def test_render_shows_review_marker_for_a_waiting_agent():
    reg = StatusRegistry.get()
    reg.register("Agt-R", "llama3")
    reg.mark_review_pending("Agt-R")
    output = _render_to_text(StatusPanel().render())
    assert "waiting_review" in output
