"""The handoff between attempts, written by the supervisor from facts only (git diff, the test result, the event log) - never from what the
model said it did. A fresh context starts with this text instead of a story, and a story can be wrong."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.night import gitbox
from agent.night.testwatch import TestState, fixed, regressions

_NUMSTAT = re.compile(r"^(\d+|-)\s+(\d+|-)\s+(.+)$", re.M)


@dataclass
class EventSummary:
    steps: int = 0
    tools: dict[str, int] = field(default_factory=dict)
    loops: int = 0
    look_blocked: int = 0
    model_errors: int = 0
    server_waits: int = 0
    edits: int = 0
    last_tools: list[str] = field(default_factory=list)
    stage_caps: int = 0
    gen_tokens: int = 0                                     # what the model generated / how long decoding and prompt reading took (from the step events)
    decode_s: float = 0.0
    prefill_s: float = 0.0
    peak_prompt: int = 0
    think_steps: int = 0                                    # forced reasoning passes (a model call without tools)
    loop_aborts: int = 0
    verify_budgets: int = 0
    masks: int = 0                                          # old tool results cut out of the prompt (each one costs a prompt re-read)
    edit_gaps: int = 0
    wanders: int = 0

    @property
    def limits_hit(self) -> int:
        """How often a guard of the coding agent interfered: blocked reads, forced thinking, aborts, stage caps, masks, nudges."""
        return self.look_blocked + self.think_steps + self.loop_aborts + self.verify_budgets + self.stage_caps + self.masks + self.edit_gaps + self.wanders

    @property
    def guard_share(self) -> float:
        """Guard interventions per model step (0 when there were no steps): how much of the run the guards, not the task, filled."""
        return self.limits_hit / self.steps if self.steps else 0.0

    @property
    def tps(self) -> float:
        """Decode speed in tokens per second over the attempt (0 when unknown)."""
        return self.gen_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def prefill_share(self) -> float:
        total = self.decode_s + self.prefill_s
        return self.prefill_s / total if total > 0 else 0.0

    @property
    def reads(self) -> int:
        return sum(n for t, n in self.tools.items() if t in {"read_file", "read_files", "search", "outline", "list_dir", "repo_map", "show_symbol", "where_used"})


_EDIT_TOOLS = frozenset({"write_file", "append_file", "edit_file", "apply_patch", "replace_lines", "insert_after", "delete_file", "move_file"})


def summarize_events(events: Path, since: float, prefix: str = "", until: float | None = None) -> EventSummary:
    """Fold the coding agent's event log (JSON lines) between `since` and `until` for the sessions whose name starts with `prefix`."""
    es = EventSummary()
    try:
        lines = events.read_text(errors="replace").splitlines()
    except OSError:
        return es
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ts = d.get("ts") or 0
        if ts < since or (until is not None and ts > until) or (prefix and not str(d.get("session", "")).startswith(prefix)):
            continue
        kind = d.get("kind")
        if kind == "step":
            es.steps += 1
            es.gen_tokens += int(d.get("gen") or 0)
            es.decode_s += float(d.get("decode_s") or 0)
            es.prefill_s += float(d.get("prefill_s") or 0)
            es.peak_prompt = max(es.peak_prompt, int(d.get("prompt") or 0))
            es.think_steps += 1 if d.get("think") else 0
        elif kind == "tool":
            name = str(d.get("name"))
            es.tools[name] = es.tools.get(name, 0) + 1
            es.last_tools = (es.last_tools + [name])[-6:]
            if name in _EDIT_TOOLS and d.get("ok") is not False:
                es.edits += 1
        elif kind in ("loop", "loop_abort"):
            es.loops += 1
            es.loop_aborts += 1 if kind == "loop_abort" else 0
        elif kind == "verify_budget":
            es.verify_budgets += 1
        elif kind in ("stage_mask", "mask", "emergency_mask"):
            es.masks += 1
        elif kind == "edit_gap":
            es.edit_gaps += 1
        elif kind == "wander":
            es.wanders += 1
        elif kind == "look_blocked":
            es.look_blocked += 1
        elif kind == "model_error":
            es.model_errors += 1
        elif kind == "server_wait":
            es.server_waits += 1
        elif kind == "stage_cap":
            es.stage_caps += 1
    return es


@dataclass
class DiffStat:
    files: list[tuple[str, int, int]] = field(default_factory=list)       # (path, added, removed)

    @property
    def added(self) -> int:
        return sum(a for _, a, _ in self.files)

    @property
    def removed(self) -> int:
        return sum(r for _, _, r in self.files)

    def empty(self) -> bool:
        return not self.files


def diff_since(root: Path, tag: str) -> DiffStat:
    """What the working tree (committed or not, untracked files included) changed relative to `tag`."""
    gitbox.git(root, "add", "-N", ".")                      # untracked files show up in numstat
    out = gitbox.git(root, "diff", "--numstat", tag).stdout
    return DiffStat([(m.group(3), 0 if m.group(1) == "-" else int(m.group(1)), 0 if m.group(2) == "-" else int(m.group(2))) for m in _NUMSTAT.finditer(out)])


@dataclass
class Handoff:
    text: str
    diff: DiffStat
    events: EventSummary


def build(root: Path, since_tag: str, before: TestState | None, after: TestState, events: EventSummary, exit_code: int, why: str = "", limit: int = 1800) -> Handoff:
    d = diff_since(root, since_tag)
    lines = [f"the last attempt ended with exit code {exit_code}" + (f" ({why})" if why else "") + f"; it ran {events.steps} model steps ({events.reads} reads, {events.edits} edits)."]
    if d.empty():
        lines.append("It changed NOTHING on disk.")
    else:
        shown = ", ".join(f"{p} (+{a}/-{r})" for p, a, r in d.files[:8]) + (f" and {len(d.files) - 8} more" if len(d.files) > 8 else "")
        lines.append(f"Since the last good state it changed: {shown}. That work is on disk - do not redo it.")
    reg, fix = regressions(before, after), fixed(before, after)
    lines.append("Tests: " + ("all green." if after.ok else ("DID NOT FINISH IN TIME (something loops or waits for input) - " if after.timed_out else "RED - ") + after.digest))
    if reg and before is not None:
        lines.append("It BROKE these tests that passed before: " + ", ".join(reg[:6]) + ".")
    if fix:
        lines.append("It FIXED: " + ", ".join(fix[:6]) + ".")
    if events.loops:
        lines.append(f"It looped {events.loops} time(s) (the same call again and again; last tools: {', '.join(events.last_tools)}). Do not repeat the same call: act or change approach.")
    elif events.reads > 6 * max(events.edits, 1) and events.reads > 20:
        lines.append(f"It mostly read ({events.reads} reads, {events.edits} edits): start writing sooner.")
    text = " ".join(lines)
    return Handoff(text if len(text) <= limit else text[:limit - 3] + "...", d, events)


def slice_events(events: Path, since: float, prefix: str, dest: Path, until: float | None = None) -> int:
    """Copy the event-log lines of one attempt (its session prefix, its time window) to `dest`, so the morning analysis has them per attempt. Returns how many lines."""
    n = 0
    try:
        lines = events.read_text(errors="replace").splitlines()
    except OSError:
        return 0
    keep = []
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ts = d.get("ts") or 0
        if ts >= since and (until is None or ts <= until) and str(d.get("session", "")).startswith(prefix):
            keep.append(line)
            n += 1
    if keep:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(keep) + "\n")
    return n
