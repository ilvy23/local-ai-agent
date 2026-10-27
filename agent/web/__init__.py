"""Web frontend/backend for the `agent` chat CLI.

Exposes the same chat/tool-calling core (agent/tui.py's turn logic,
agent/memory/*, agent/tools/*) over an HTTP API instead of a terminal REPL,
so it can be reached from a browser - a phone, from anywhere, over a
private Tailscale network. See the plan this was built from for the full
design rationale (approval flow, deployment, iOS notes).

Not installed by the CLI-only home-PC setup: `uv sync --group web` on
whichever machine actually runs this.
"""
