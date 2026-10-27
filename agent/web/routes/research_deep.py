"""Deep research as a background job for the GUI: start it, poll it, stop it.

Same registry the other long GUI jobs use (`agent.web.routes._jobs`), and the
same job function the `deep_research` chat tool starts (`agent.tools.deep_research.run_job`) - a run
started from either place shows up here. Kept in its own file (additive, no existing research route
touched) since `agent/web/routes/research.py` already covers the ordinary web_search screen (engines,
trust rules, cache, SSRF samples) and may be edited by someone else at the same time as this.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.tools.deep_research import JOB_KIND, run_job
from agent.web.console_shim import WebConsole
from agent.web.deps import get_client, get_config, get_store
from agent.web.routes import _jobs

router = APIRouter(prefix="/api/research/deep", tags=["research-deep"])


class StartIn(BaseModel):
    question: str
    max_subquestions: int = 12
    budget_chars: int = 20000
    session_id: int | None = None          # when given, the finished report is also offered as a download card


@router.post("")
def start_deep_research(body: StartIn, client=Depends(get_client), config=Depends(get_config)) -> dict[str, Any]:
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "question must not be empty.")
    if not 1 <= body.max_subquestions <= 60:
        raise HTTPException(422, "max_subquestions must be from 1 to 60.")
    if not 2000 <= body.budget_chars <= 200000:
        raise HTTPException(422, "budget_chars must be from 2000 to 200000.")
    tag = question[:200]
    existing = _jobs.running(JOB_KIND, question=tag)
    if existing is not None:
        return {"id": existing["id"], "already_running": True}
    store = get_store() if body.session_id is not None else None
    console = WebConsole(None, None)
    jid = _jobs.start(
        JOB_KIND,
        lambda job: run_job(job, question, client, config, console, store, body.session_id,
                            max_steps=body.max_subquestions, budget_chars=body.budget_chars),
        question=tag,
    )
    return {"id": jid, "already_running": False}


@router.get("/{job_id}")
def deep_research_status(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job (jobs are kept in memory only; a server restart forgets them).")
    return {k: job.get(k) for k in ("id", "status", "text", "error", "result", "elapsed_s", "log")}


@router.post("/{job_id}/stop")
def deep_research_stop(job_id: str) -> dict[str, bool]:
    if not _jobs.stop(job_id):
        raise HTTPException(404, "No such job.")
    return {"ok": True}
