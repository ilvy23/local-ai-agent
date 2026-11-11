"""`nightage night watch`: a live view of a running night - the queue with statuses, the clock, the last events, the model server and the GPU temperature.
`render()` is a pure function of the state file's data, so it can be tested; the CLI just redraws it every few seconds."""
from __future__ import annotations

import json
from pathlib import Path

ICON = {"done": "✓", "active": "▶", "partial": "◐", "pending": "·"}


def render(state: dict, ps_line: str = "", temps: list[float] | None = None, control: str = "", tail: int = 8) -> str:
    left = state.get("left_s", 0)
    lines = [f"night: {left // 3600} h {left % 3600 // 60:02d} min left" + (f"   [{control.upper()} requested]" if control else "")]
    items = state.get("items", [])
    done = sum(1 for i in items if i.get("status") == "done")
    lines.append(f"prompts: {done}/{len(items)} done")
    for i in items:
        kinds = i.get("kinds") or []
        lines.append(f"  {ICON.get(i.get('status'), '?')} {i.get('name', '?'):<24} attempts {i.get('attempts', 0):<3} {i.get('seconds', 0) / 60:5.0f} min  {kinds[-1] if kinds else ''}")
    if state.get("flaky"):
        lines.append(f"flaky tests: {len(state['flaky'])}")
    if ps_line:
        lines.append(f"model: {ps_line}")
    if temps:
        lines.append("gpu temp: " + ", ".join(f"{t:.0f} C" for t in temps))
    lines += ["", *state.get("events", [])[-tail:]]
    return "\n".join(lines)


def load(root: Path) -> dict:
    try:
        return json.loads((root / ".night" / "state.json").read_text())
    except (OSError, ValueError):
        return {}
