"""The shell tool: run a command on the user's machine, behind the safety gate.

The tool itself does NOT classify or gate — that is the agent loop's job (it
calls `agent.safety.classify_command` and enforces approval before ever
invoking this handler). This module only executes an already-approved command
and formats the result for the model.

`risk="dynamic"` signals the loop to classify per-call rather than trust a
static tier.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from agent.tools.registry import Tool

DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_TIMEOUT_S = 300
OUTPUT_LIMIT = 10_000
TRUNCATE_MARKER = "\n[truncated]"


def run_command(
    command: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_timeout_s: int = DEFAULT_MAX_TIMEOUT_S,
    cwd: str | None = None,
) -> str:
    """Execute `command` via bash, returning merged stdout+stderr and exit code.

    Output is truncated to OUTPUT_LIMIT chars. `timeout_s` is capped at
    `max_timeout_s`. cwd defaults to the process working directory (the
    directory agent was launched from).
    """
    effective_timeout = min(max(1, int(timeout_s)), max_timeout_s)

    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return f"$ {command}\n(in {cwd or os.getcwd()})\nTimed out after {effective_timeout}s."

    output = (completed.stdout or "") + (completed.stderr or "")
    if len(output) > OUTPUT_LIMIT:
        output = output[:OUTPUT_LIMIT] + TRUNCATE_MARKER

    # Echo the command and the directory it ran in. Bare output is
    # unattributable — across several calls the model can't tell which result
    # belongs to which command, and the cwd (wherever the CLI was launched) is
    # otherwise invisible to it.
    head = f"$ {command}\n(in {cwd or os.getcwd()}, exit code {completed.returncode})"
    return f"{head}\n{output}".rstrip()


def make_shell_tool(config: dict[str, Any]) -> Tool:
    """Build the run_command Tool, wiring the config timeout cap into the handler."""
    max_timeout_s = int(config.get("safety", {}).get("max_timeout_s", DEFAULT_MAX_TIMEOUT_S))

    def handler(command: str, timeout_s: int = DEFAULT_TIMEOUT_S, **_kwargs: Any) -> str:
        return run_command(command, timeout_s=timeout_s, max_timeout_s=max_timeout_s)

    return Tool(
        name="run_command",
        description=(
            "Run a shell command on the user's machine and return its output "
            "and exit code. Commands are risk-classified and may require the "
            "user's approval before running."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout_s": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 60).",
                },
            },
            "required": ["command"],
        },
        handler=handler,
        risk="dynamic",
    )
