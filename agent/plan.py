"""Acceptance list: what must be true before a big task counts as finished.

For a substantial task the harness (not the model) owns a short checklist of concrete, verifiable criteria.
The model cannot edit it; it can only confirm an item with `mark_done(id, evidence)` once it has proof
(a command and its result, a file path). The list rides in the context head, so compaction never drops it,
and the final answer is held back (twice at most) while items are unconfirmed - the local model's habit of
declaring victory early is the failure this targets. `/plan` shows it, `/plan clear` drops it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MAX_ITEMS = 8
MIN_EVIDENCE_CHARS = 12
MIN_TASK_CHARS = 160

_SYSTEM = (
    "Turn the user's task into a checklist of at most 8 concrete acceptance criteria: what must be true "
    "when the work is finished. Each one short (under 100 characters) and checkable (a file exists with X, "
    "the tests pass, the command prints Y, the old data still loads). Cover EVERY explicit requirement; no "
    "setup steps, no duplicates, no vague items like 'good quality'. Output ONLY a JSON array of strings."
)
_CODE_ASK = re.compile(
    r"(?i)\b(code|script|function|refactor|implement|class|module|package|tests?|pytest|game|app|cli|repo|"
    r"programm\w*|schreib\w*|erstell\w*|baue?|entwickl\w*|datei|skript|erweiter\w*)\b|\.(py|js|ts|rs|go|sh|nix)\b"
)


def should_create(text: str) -> bool:
    """A task big enough for a checklist: long and about building something, or an explicit multi-item list."""
    text = text.strip()
    if text.startswith("/") or len(text) < 80:
        return False
    bullets = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", text))
    return bool(bullets >= 3 or (len(text) >= MIN_TASK_CHARS and _CODE_ASK.search(text)))


def parse_items(reply: str) -> list[str]:
    reply = reply or ""
    start, end = reply.find("["), reply.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(reply[start:end + 1])
            items = [str(x).strip() for x in data if str(x).strip()]
            if items:
                return items[:MAX_ITEMS]
        except ValueError:
            pass
    lines = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip() for ln in reply.splitlines()
             if re.match(r"^\s*(?:[-*]|\d+[.)])\s+\S", ln)]
    return [ln for ln in lines if ln][:MAX_ITEMS]


@dataclass
class Item:
    id: int
    text: str
    done: bool = False
    evidence: str = ""


@dataclass
class Plan:
    task: str
    items: list[Item] = field(default_factory=list)
    nudges: int = 0

    def open_items(self) -> list[Item]:
        return [i for i in self.items if not i.done]

    def confirmed(self) -> int:
        return len(self.items) - len(self.open_items())

    def system_message(self) -> str:
        lines = [f"{i.id}. [{'x' if i.done else ' '}] {i.text}" for i in self.items]
        return ("ACCEPTANCE LIST (owned by the harness - you cannot edit it). The task is finished only when every "
                "item is confirmed. Confirm an item by calling mark_done(id, evidence) once you have proof "
                "(a command you ran and its result, or the file path); do not confirm from memory.\n" + "\n".join(lines))

    def mark_done(self, item_id: Any, evidence: str) -> str:
        try:
            wanted = int(item_id)
        except (TypeError, ValueError):
            return f"Error: id must be one of {[i.id for i in self.items]}."
        item = next((i for i in self.items if i.id == wanted), None)
        if item is None:
            return f"Error: there is no item {wanted}. Items: {[i.id for i in self.items]}."
        if len((evidence or "").strip()) < MIN_EVIDENCE_CHARS:
            return "Error: give concrete evidence (the command you ran and its result, or the file path)."
        item.done, item.evidence = True, evidence.strip()
        left = self.open_items()
        return (f"Item {item.id} confirmed ({self.confirmed()} of {len(self.items)} done)."
                + (f" Still open: {', '.join(str(i.id) for i in left)}." if left else " All items are confirmed."))

    def reminder(self) -> str:
        left = self.open_items()
        return ("ACCEPTANCE LIST NOT COMPLETE. These items are not confirmed yet:\n"
                + "\n".join(f"{i.id}. {i.text}" for i in left)
                + "\nEither finish them and confirm each with mark_done(id, evidence), or tell the user plainly "
                  "which of them are NOT done and why. Do not claim the task is complete.")

    def lines(self) -> list[str]:
        return [f"  {i.id}  {'done' if i.done else 'open'}  {i.text}" for i in self.items]


def generate(llm: Any, config: dict[str, Any], task: str) -> Plan | None:
    """Ask the background model for the checklist. None on failure or if it came back empty."""
    try:
        reply = "".join(llm.chat(
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": task[:6000]}],
            model=config["models"]["background"], stream=False, background=True,
        ))
    except Exception:  # noqa: BLE001 - a missing checklist must never block the task
        return None
    items = parse_items(reply)
    return Plan(task=task, items=[Item(i, t) for i, t in enumerate(items, 1)]) if len(items) >= 2 else None


class Holder:
    """The session's current plan (None when there is none)."""

    def __init__(self) -> None:
        self.plan: Plan | None = None


def make_tool(holder: Holder, console: Any):
    from agent.tools.registry import Tool

    def mark_done(id: Any = None, evidence: str = "", ids: Any = None, **_: Any) -> str:  # noqa: A002 - the tool's argument name
        plan = holder.plan
        if plan is None:
            return "Error: there is no acceptance list for this task."
        wanted = list(ids) if isinstance(ids, (list, tuple)) else ([] if ids in (None, "") else [ids])
        if id not in (None, ""):
            wanted.insert(0, id)
        if not wanted:
            return "Error: give id (or ids=[...]) and evidence."
        results = [plan.mark_done(i, evidence) for i in wanted]
        if any(not r.startswith("Error") for r in results):
            console.print(f"[dim]plan  {plan.confirmed()} of {len(plan.items)} confirmed[/dim]")
        return results[-1] if len(results) == 1 else "\n".join(results)

    return Tool(
        name="mark_done",
        description=(
            "Confirm ONE item of the acceptance list as finished. Call it only after you have proof: run the "
            "check or read the file first. `evidence` = the command you ran and its result, or the file path "
            "and what it contains. The task is not finished until every item is confirmed."
        ),
        parameters={"type": "object", "properties": {
            "id": {"type": "integer", "description": "Number of the acceptance item."},
            "ids": {"type": "array", "items": {"type": "integer"}, "description": "Several items proven by the same evidence (one call instead of many)."},
            "evidence": {"type": "string", "description": "Concrete proof: command + result, or file path."}},
            "required": ["evidence"]},
        handler=mark_done,
        risk="safe",
    )
