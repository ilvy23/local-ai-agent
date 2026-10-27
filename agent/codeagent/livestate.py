"""One dict with everything a live UI footer / panel shows about a coding engine. Reads the same engine attributes the web run view reads
(agent/web/routes/code_runs.py `_public`), never raises: a value that cannot be read is left at its default."""

from __future__ import annotations

import contextlib
import time
from typing import Any


def snapshot(engine: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "state": "idle", "now": "", "step": 0, "budget_now": 0, "step_ceiling": 0, "budget_source": "",
        "ctx_used": 0, "ctx_limit": 0, "ctx_percent": 0.0, "tps": 0.0, "model": "", "mode": "", "elapsed_s": 0.0,
        "files_changed": 0, "lines_added": 0, "lines_removed": 0, "files": [], "dead_ends": 0, "plan": None,
        "queued": 0, "paused": False, "waiting": "", "undo_available": 0,
    }
    if engine is None:
        return snap
    with contextlib.suppress(Exception):
        snap.update(step=int(engine.ctx.step), budget_now=int(engine.budget_now), step_ceiling=int(engine.step_ceiling), budget_source=str(engine.budget_source))
    with contextlib.suppress(Exception):
        used, limit = int(engine.ctx.estimate()), int(engine.num_ctx)
        snap.update(ctx_used=used, ctx_limit=limit, ctx_percent=round(100.0 * used / limit, 1) if limit else 0.0)
    with contextlib.suppress(Exception):
        snap["tps"] = round(float(engine.avg_tps()), 1)
    with contextlib.suppress(Exception):
        snap.update(model=str(engine.model), mode=str(engine.pol.mode))
    with contextlib.suppress(Exception):
        snap["now"] = str(engine.now)
        live = engine.live
        snap.update(paused=bool(live.paused), waiting=str(live.waiting), queued=int(live.queued()))
        in_turn = bool(engine.in_turn)
        snap["state"] = ("paused" if live.paused else "waiting" if live.waiting else "running") if in_turn else "idle"
        if in_turn and engine._turn_t0:
            snap["elapsed_s"] = round(time.monotonic() - engine._turn_t0, 1)
    with contextlib.suppress(Exception):
        rows = engine.tools.changes.totals()
        snap.update(files_changed=len(rows), lines_added=sum(r[1] for r in rows), lines_removed=sum(r[2] for r in rows),
                    files=[{"path": r[0], "added": r[1], "removed": r[2], "created": bool(r[3])} for r in rows][:50])
    with contextlib.suppress(Exception):
        snap["dead_ends"] = len(engine.dead_ends.items)
    with contextlib.suppress(Exception):
        snap["undo_available"] = len(engine.undo)
    with contextlib.suppress(Exception):
        if getattr(engine, "stage_plan", None) is not None or engine.plan.plan is not None:
            from agent.codeagent import planview

            view = planview.build(engine)
            if view.stages or view.checklist:
                snap["plan"] = planview.to_dict(view)
    return snap
