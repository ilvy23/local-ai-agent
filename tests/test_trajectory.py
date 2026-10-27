"""Tests for agent/trajectory.py — append-only JSONL action log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.trajectory import log_action, read_recent


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect trajectory writes to a temp directory for every test."""
    import agent.trajectory as traj_mod
    monkeypatch.setattr(traj_mod, "_TRAJECTORY_PATH", tmp_path / "trajectory.jsonl")


# ── Basic write / read round-trip ─────────────────────────────────────────────

def test_log_action_creates_jsonl_file(tmp_path: Path) -> None:
    from agent.trajectory import _TRAJECTORY_PATH  # post-patch value
    log_action("tool_call", "read_file", "/home/user/notes.txt", "file contents", session_id=1)
    assert _TRAJECTORY_PATH.exists()


def test_log_action_writes_valid_json(tmp_path: Path) -> None:
    from agent.trajectory import _TRAJECTORY_PATH
    log_action("tool_call", "write_file", "{'path': 'out.txt'}", "ok", session_id=2)
    line = _TRAJECTORY_PATH.read_text().strip()
    record = json.loads(line)
    assert record["action"] == "tool_call"
    assert record["tool"] == "write_file"
    assert record["session_id"] == 2


def test_log_action_appends_multiple_records(tmp_path: Path) -> None:
    from agent.trajectory import _TRAJECTORY_PATH
    log_action("tool_call", "tool_a", "in_a", "out_a")
    log_action("tool_call", "tool_b", "in_b", "out_b")
    lines = [l for l in _TRAJECTORY_PATH.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["tool"] == "tool_a"
    assert json.loads(lines[1])["tool"] == "tool_b"


def test_record_has_required_fields() -> None:
    from agent.trajectory import _TRAJECTORY_PATH
    log_action("tool_call", "shell", "cmd", "output", session_id=5)
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert {"ts", "session_id", "action", "tool", "input", "output"} <= record.keys()


def test_session_id_can_be_none() -> None:
    from agent.trajectory import _TRAJECTORY_PATH
    log_action("tool_call", "tool", "in", "out", session_id=None)
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert record["session_id"] is None


# ── Redaction in the log ──────────────────────────────────────────────────────

def test_secrets_are_redacted_in_log() -> None:
    from agent.trajectory import _TRAJECTORY_PATH
    secret_input = '{"api_key": "sk-abcdefghijklmnopqrstuvwxyz123456"}'
    log_action("tool_call", "api_call", secret_input, "ok")
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in record["input"]


def test_secrets_redacted_from_output() -> None:
    from agent.trajectory import _TRAJECTORY_PATH
    secret_output = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefgh"
    log_action("tool_call", "http", "{}", secret_output)
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in record["output"]


# ── Truncation ────────────────────────────────────────────────────────────────

def test_long_input_is_truncated() -> None:
    from agent.trajectory import _TRAJECTORY_PATH, _SUMMARY_MAX_CHARS
    long_input = "x" * 2000
    log_action("tool_call", "tool", long_input, "ok")
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert len(record["input"]) <= _SUMMARY_MAX_CHARS


def test_long_output_is_truncated() -> None:
    from agent.trajectory import _TRAJECTORY_PATH, _SUMMARY_MAX_CHARS
    log_action("tool_call", "tool", "in", "y" * 2000)
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert len(record["output"]) <= _SUMMARY_MAX_CHARS


# ── read_recent ───────────────────────────────────────────────────────────────

def test_read_recent_returns_last_n() -> None:
    for i in range(10):
        log_action("tool_call", f"tool_{i}", f"in_{i}", f"out_{i}")
    records = read_recent(3)
    assert len(records) == 3
    assert records[-1]["tool"] == "tool_9"


def test_read_recent_returns_empty_if_no_file(tmp_path: Path, monkeypatch) -> None:
    import agent.trajectory as traj_mod
    monkeypatch.setattr(traj_mod, "_TRAJECTORY_PATH", tmp_path / "nonexistent.jsonl")
    assert read_recent() == []


# ── Resilience ────────────────────────────────────────────────────────────────

def test_log_action_never_raises_on_unwritable_path(monkeypatch, tmp_path: Path) -> None:
    import agent.trajectory as traj_mod
    # Point to a path that can't be created (file exists as dir).
    bad = tmp_path / "trajectory.jsonl"
    bad.mkdir()
    monkeypatch.setattr(traj_mod, "_TRAJECTORY_PATH", bad / "subfile.jsonl")
    # Should not raise.
    log_action("tool_call", "tool", "in", "out")


# ── on_call hook via registry ─────────────────────────────────────────────────

def test_registry_set_on_call_fires_after_tool_execution(tmp_path: Path) -> None:
    """End-to-end: set_on_call → log written."""
    from agent.tools.registry import Tool, ToolRegistry

    registry = ToolRegistry()
    fired: list[tuple[str, dict, str]] = []
    registry.set_on_call(lambda name, args, result: fired.append((name, args, result)))

    tool = Tool(
        name="noop",
        description="does nothing",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda **_: "done",
    )
    registry.register(tool)

    result = registry.get("noop").handler()
    registry.fire_on_call("noop", {}, result)

    assert fired == [("noop", {}, "done")]


def test_registry_fire_on_call_suppresses_callback_exceptions() -> None:
    from agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.set_on_call(lambda name, args, result: 1 / 0)  # always raises
    # Should not raise.
    registry.fire_on_call("tool", {}, "ok")


# ── Redaction runs BEFORE truncation ──────────────────────────────────────────

def test_secret_beyond_truncation_point_is_still_redacted() -> None:
    """Truncating first would cut a secret in half and log the surviving prefix."""
    from agent.trajectory import _TRAJECTORY_PATH, _SUMMARY_MAX_CHARS
    padding = "pad " * ((_SUMMARY_MAX_CHARS - 20) // 4)
    secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
    log_action("tool_call", "tool", padding + secret, "ok")
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert "sk-abcdefghijklmnopq" not in record["input"]
    assert len(record["input"]) <= _SUMMARY_MAX_CHARS


def test_private_key_block_spanning_truncation_is_redacted() -> None:
    """A key block cut before its -----END----- marker stopped matching at all."""
    from agent.trajectory import _TRAJECTORY_PATH
    block = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        + "MIIEpAIBAAKCAQEA\n" * 60
        + "-----END RSA PRIVATE KEY-----"
    )
    log_action("tool_call", "read_file", block, "ok")
    record = json.loads(_TRAJECTORY_PATH.read_text().strip())
    assert "MIIEpAIBAAKCAQEA" not in record["input"]
    assert "[REDACTED:private-key]" in record["input"]
