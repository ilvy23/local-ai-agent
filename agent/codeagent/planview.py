"""Plan view: one read-only snapshot of a staged run, for humans - before it starts and while it runs.

Nothing here is new engine state. It aggregates what already exists: the stage plan (stages.py: Stage.status, the
budget fields spent_s/spent_steps/real_edits), the harness-owned acceptance checklist (plan.py: Item.done/evidence),
the loop guard's dead ends (loopguard.DeadEnds, exposed on the engine as `dead_ends`) and the event log
(eventlog.py: the latest stage_* event gives a sharper "what's happening now" than the stage statuses alone).

    build(engine) -> PlanView                 the live snapshot (used by the web run status and the CLI gate)
    from_parts(stage_plan, acc_plan, dead_ends_count, cfg, events=None) -> PlanView   the pure core, no engine needed
    table(view) -> rich renderable             `agent code --plan-first` prints this directly (never agent/panel.py)
    to_dict(view) -> dict                      JSON-safe shape for the web run status response
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from agent.codeagent import stages as stages_mod

STATUS_STYLE = {"pending": "dim", "active": "bold cyan", "done": "green", "blocked": "red"}


@dataclass
class StageView:
    id: int
    title: str
    status: str                       # pending | active | done | blocked
    attempts: int
    budget_used_s: float
    budget_limit_s: float             # 0 = no time limit configured
    budget_used_steps: int
    budget_limit_steps: int           # 0 = no step limit configured
    budget_share: float               # the larger of the time/step share used (0 when both limits are off); stages.budget_share
    files: list[str] = field(default_factory=list)
    reason: str = ""                  # why it is blocked, if it is
    summary: str = ""                 # why it is done, if it is


@dataclass
class ChecklistItem:
    id: int
    text: str
    done: bool
    evidence: str


@dataclass
class PlanView:
    stages: list[StageView] = field(default_factory=list)
    current: int = -1                 # index into `stages` of the active one; -1 = none active
    source: str = "single"            # StagePlan.source: single | headings | model | ranges
    task: str = ""
    checklist: list[ChecklistItem] = field(default_factory=list)
    checklist_confirmed: int = 0
    checklist_total: int = 0
    dead_ends: int = 0
    now: str = "no plan yet"          # one short line: what is happening right now


# ── the "what's happening now" line ──────────────────────────────────────────

def _event_now(rec: dict[str, Any]) -> str | None:
    kind = rec.get("kind")
    if kind == "stage_start":
        return f"starting stage {rec.get('id')}: {rec.get('title')}"
    if kind == "segment_start":
        return f"working on stage {rec.get('id')}: {rec.get('title')}"
    if kind == "stage_cap":
        return f"stage {rec.get('id')} hit the context cap, continuing in a fresh context"
    if kind == "stage_tests":
        return f"running tests for stage {rec.get('id')} ({rec.get('status')})"
    if kind == "stage_resplit":
        return f"splitting stage {rec.get('from')} into smaller parts"
    if kind == "stage_resplit_failed":
        return f"could not split stage {rec.get('stage')}: {rec.get('reason')}"
    if kind == "stage_done":
        return f"finished stage {rec.get('id')}: {rec.get('title')}"
    if kind == "stage_blocked":
        return f"stage {rec.get('id')} blocked: {rec.get('reason')}"
    return None


def _now_from_events(events: Any) -> str | None:
    for rec in reversed(list(getattr(events, "recent", None) or [])):
        line = _event_now(rec)
        if line:
            return line
    return None


def _now_from_stages(stages: list[StageView], current: int) -> str:
    if not stages:
        return "no stage plan yet"
    if current >= 0:
        st = stages[current]
        return f"stage {st.id}/{len(stages)}: {st.title}"
    if all(s.status == "done" for s in stages):
        return "all stages done"
    pending = [s for s in stages if s.status == "pending"]
    if pending:
        return f"waiting to start stage {pending[0].id}/{len(stages)}: {pending[0].title}"
    blocked = [s for s in stages if s.status == "blocked"]
    if blocked:
        return f"blocked: {blocked[-1].reason or 'needs a human'}"
    return "idle"


# ── building ──────────────────────────────────────────────────────────────

def _stage_view(st: Any, cfg: Any) -> StageView:
    max_s, max_steps, _min_edits = stages_mod.budget_limits(cfg, st)
    return StageView(id=st.id, title=st.title, status=st.status, attempts=st.attempts,
                      budget_used_s=round(st.spent_s, 1), budget_limit_s=round(max_s, 1),
                      budget_used_steps=int(st.spent_steps), budget_limit_steps=int(max_steps),
                      budget_share=round(stages_mod.budget_share(st, cfg), 3),
                      files=list(st.files), reason=st.reason, summary=st.summary)


def from_parts(stage_plan: Any, acc_plan: Any, dead_ends_count: int, cfg: Any, events: Any = None) -> PlanView:
    """The pure core: no engine needed, so it is the same function a unit test and `build()` both run."""
    stages = [_stage_view(st, cfg) for st in (stage_plan.stages if stage_plan is not None else [])]
    current = next((i for i, s in enumerate(stages) if s.status == "active"), -1)
    checklist = [ChecklistItem(i.id, i.text, i.done, i.evidence) for i in (acc_plan.items if acc_plan is not None else [])]
    now = (_now_from_events(events) if events is not None else None) or _now_from_stages(stages, current)
    return PlanView(stages=stages, current=current, source=(stage_plan.source if stage_plan is not None else "single"),
                     task=(stage_plan.task if stage_plan is not None else ""), checklist=checklist,
                     checklist_confirmed=(acc_plan.confirmed() if acc_plan is not None else 0),
                     checklist_total=len(checklist), dead_ends=max(0, int(dead_ends_count)), now=now)


def build(engine: Any) -> PlanView:
    """The live snapshot of a CodeEngine: its stage plan, its acceptance checklist, its dead ends, its event log. Safe on a brand
    new engine (no stage plan yet: an empty PlanView) and on one mid-run (reflects whatever has happened so far)."""
    dead = getattr(engine, "dead_ends", None)
    dead_count = len(getattr(dead, "items", None) or [])
    acc_plan = getattr(getattr(engine, "plan", None), "plan", None)
    return from_parts(getattr(engine, "stage_plan", None), acc_plan, dead_count, getattr(engine, "config", None),
                       events=getattr(engine, "events", None))


# ── rendering ─────────────────────────────────────────────────────────────

def _budget_cell(s: StageView) -> str:
    time_part = f"{s.budget_used_s:.0f}s" + (f"/{s.budget_limit_s:.0f}s" if s.budget_limit_s else "")
    step_part = f"{s.budget_used_steps}" + (f"/{s.budget_limit_steps}" if s.budget_limit_steps else "") + " steps"
    pct = f" ({s.budget_share * 100:.0f}%)" if s.budget_limit_s or s.budget_limit_steps else ""
    return f"{time_part} · {step_part}{pct}"


def stage_table(view: PlanView) -> Table:
    t = Table(title=f"Plan ({view.source})" if view.stages else "Plan", expand=False)
    t.add_column("#", justify="right")
    t.add_column("Stage")
    t.add_column("Status")
    t.add_column("Budget used/limit")
    for s in view.stages:
        style = STATUS_STYLE.get(s.status, "")
        t.add_row(str(s.id), s.title, f"[{style}]{s.status}[/{style}]" if style else s.status, _budget_cell(s))
    return t


def checklist_table(view: PlanView) -> Table | None:
    if not view.checklist:
        return None
    t = Table(title=f"Acceptance checklist ({view.checklist_confirmed}/{view.checklist_total})", expand=False)
    t.add_column("#", justify="right")
    t.add_column("")
    t.add_column("Item")
    t.add_column("Evidence")
    for it in view.checklist:
        t.add_row(str(it.id), "✓" if it.done else " ", it.text, it.evidence if it.done else "")
    return t


def table(view: PlanView) -> Group:
    """Everything `agent code --plan-first` prints: the stage table, the checklist (if any) and a one-line status panel.
    A renderable the CLI hands to its own console - never wired into agent/panel.py's status dashboard."""
    parts: list[Any] = [stage_table(view)]
    cl = checklist_table(view)
    if cl is not None:
        parts.append(cl)
    now = view.now + (f" · {view.dead_ends} dead end(s) hit" if view.dead_ends else "")
    parts.append(Panel(now, title="Right now", expand=False))
    return Group(*parts)


def lines(view: PlanView) -> list[str]:
    """A plain-text rendering (no rich markup): scripting, logging, and assertions in tests that don't want to parse a Table."""
    out = [f"PLAN ({view.source}): {len(view.stages)} stage(s)"]
    for s in view.stages:
        out.append(f"  {s.id}  {s.status:<8} {_budget_cell(s)}  {s.title}")
    if view.checklist:
        out.append(f"ACCEPTANCE: {view.checklist_confirmed}/{view.checklist_total} confirmed")
        for it in view.checklist:
            out.append(f"  {it.id}  [{'x' if it.done else ' '}]  {it.text}")
    out.append(f"NOW: {view.now}" + (f" ({view.dead_ends} dead end(s))" if view.dead_ends else ""))
    return out


def to_dict(view: PlanView) -> dict[str, Any]:
    """The JSON-safe shape the web run status response adds `plan` as."""
    return {"source": view.source, "task": view.task, "current": view.current, "now": view.now, "dead_ends": view.dead_ends,
            "stages": [asdict(s) for s in view.stages], "checklist": [asdict(i) for i in view.checklist],
            "checklist_confirmed": view.checklist_confirmed, "checklist_total": view.checklist_total}
