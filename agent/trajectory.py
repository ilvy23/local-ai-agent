"""Append-only trajectory log: every tool call the agent makes is recorded as
a JSONL line at ~/.companion/trajectory.jsonl.

Tool inputs and outputs are redacted before writing so no secrets end up on
disk even if a tool is called with credentials in its arguments.  Each line is
a self-contained JSON object — the file can be tailed, grepped, or fed to an
analysis script without parsing the whole thing.

Design constraints:
- Never raises: a broken log must not interrupt a tool-calling turn.
- Input/output summaries are capped at 500 chars before writing — enough for
  useful context, small enough to keep the file from growing unbounded.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRAJECTORY_PATH = Path.home() / ".companion" / "trajectory.jsonl"
_SUMMARY_MAX_CHARS = 500


def log_action(
    action: str,
    tool: str | None,
    input_summary: str,
    output_summary: str,
    session_id: int | None = None,
) -> None:
    """Append one action record to the trajectory log.

    `action` is a short verb like "tool_call".  `tool` is the tool name, or
    None for non-tool actions.  Input and output summaries are redacted and
    then truncated before writing.  `session_id` links the record back to a
    chat session in the main database.  Never raises.

    Order matters: redaction runs on the FULL text, before truncation. Cutting
    first would slice a secret in half — the surviving prefix no longer matches
    its pattern (worst case a PRIVATE KEY block loses its `-----END` marker, so
    the DOTALL rule stops matching entirely) and the fragment lands on disk.
    """
    from agent.redact import redact

    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "action": action,
        "tool": tool,
        "input": redact(input_summary)[:_SUMMARY_MAX_CHARS],
        "output": redact(output_summary)[:_SUMMARY_MAX_CHARS],
    }
    try:
        _TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TRAJECTORY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - trajectory must never break a turn
        logger.debug("Trajectory log write failed: %s", exc)


def read_recent(n: int = 50) -> list[dict[str, Any]]:
    """Return the last `n` records from the trajectory log, newest last.

    Returns an empty list if the file doesn't exist or can't be read.
    Used by audit/diagnostic commands — not in the hot path.
    """
    if not _TRAJECTORY_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with _TRAJECTORY_PATH.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("Trajectory log read failed: %s", exc)
    return records[-n:]
