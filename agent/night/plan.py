"""`nightage night plan`: everything the night WOULD do, without starting it - the order of the prompts and the minutes each would get, what the prompt check
says, the files each one must create, the charter, whether the test command works, and how fit the machine is right now."""
from __future__ import annotations

from pathlib import Path

from agent.night import deliverables, preflight
from agent.night.governor import MIN_SLICE_S, RESERVE_S
from agent.night.queue import Item


def slices(items: list[Item], hours: float) -> list[tuple[str, float]]:
    """(name, minutes): the first-round time slices, weights respected; a prompt never gets less than MIN_SLICE_S while the time lasts."""
    total_s = max(hours * 3600 - RESERVE_S, 0)
    weights = [max(i.weight, 0.1) for i in items]
    return [(i.name, max(MIN_SLICE_S, total_s * w / sum(weights)) / 60) for i, w in zip(items, weights)] if items else []


def render(items: list[Item], root: Path, hours: float, charter_text: str = "", facts_text: str = "", health: str | None = None) -> str:
    lines = [f"Night plan for {root}: {len(items)} prompts, {hours:.1f} h (the last {RESERVE_S // 60} min are reserved for the end state)", ""]
    for n, ((name, minutes), it) in enumerate(zip(slices(items, hours), items), 1):
        lines.append(f"{n}. {name}  ~{minutes:.0f} min first round" + (f"  weight {it.weight:g}" if it.weight != 1 else "") + (f"  kind {it.kind}" if it.kind else "")
                     + (f"  after {', '.join(it.depends)}" if it.depends else "") + (f"  verify `{it.verify}`" if it.verify else ""))
        want = deliverables.expected(it.text, root)
        if want:
            lines.append(f"     must create: {', '.join(want)}")
    warnings = preflight.check(items, root)
    lines += ["", "Prompt check: " + ("fine" if not warnings else "")] + [f"  - {w}" for w in warnings]
    if facts_text:
        lines += ["", facts_text]
    if charter_text:
        lines += ["", "Charter:", charter_text]
    lines += ["", "Machine: " + (health or "fit to start")]
    return "\n".join(lines)
