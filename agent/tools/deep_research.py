"""Chat tool: deep_research.

Starts a `agent.research.deep.DeepResearch` run as a background job, the same pattern
other long GUI jobs use (`agent.web.routes._jobs`: start a function in a
thread, poll it, ask it to stop) - a multi-step research pass can take minutes and must not block the chat
turn the way `web_search` does. The job function is also reused directly by the `/api/research/deep` route
(`agent/web/routes/research_deep.py`), so a run started from either the chat tool or the web GUI shows up in
the same job table.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.memory.store import Store
from agent.research import deep
from agent.tools.registry import Tool
from agent.web.routes import _jobs

JOB_KIND = "deep_research"
DEST_DIR = "deep_research"


def make_complete(client: Any, model: str) -> Callable[[list[dict[str, str]]], str]:
    """Adapt `agent.llm.OllamaClient.chat` (a streaming generator) to DeepResearch's plain
    `complete(messages) -> str`. `background=True` caps the call like the other short utility calls
    (distill, overlap judge) this codebase already makes, rather than running it as a full chat turn."""
    def complete(messages: list[dict[str, str]]) -> str:
        return "".join(client.chat(messages, model=model, background=True))
    return complete


def run_job(
    job: dict[str, Any], question: str, client: Any, config: dict[str, Any], console: Any,
    store: Store | None = None, session_id: int | None = None, **kwargs: Any,
) -> dict[str, Any]:
    """The function `_jobs.start` runs in a thread. Writes progress onto `job` as it goes.

    `make_web_tools`/`make_create_file_tool`/`write_file` are imported here rather than at module level so
    a test can monkeypatch `agent.tools.web.make_web_tools` (etc.) right before calling this, the same way
    `agent/main.py`'s CLI command and `agent/tui.py`'s registry setup already defer those imports."""
    from agent.tools.create_file import make_create_file_tool
    from agent.tools.files import write_file
    from agent.tools.web import make_web_tools

    model = (config.get("code") or {}).get("model") or config["models"]["chat"]
    complete = make_complete(client, model)
    search_tool, fetch_tool = make_web_tools(console)
    create_file = make_create_file_tool(store, session_id, console).handler if (store is not None and session_id is not None) else None

    def progress(event: str, fields: dict[str, Any]) -> None:
        if event == "question_start":
            job["text"] = f"[{fields['step']}] researching: {fields['question']}"
            job["log"].append(job["text"])
        elif event == "question_done":
            extra = f", +{fields['new_questions']} new sub-question(s)" if fields.get("new_questions") else ""
            job["log"].append(f"  -> {fields['status']}{extra}")
        elif event == "compacted":
            job["log"].append(f"  (compacted running notes: {fields['before_chars']} -> {fields['after_chars']} chars)")
        elif event == "run_done":
            job["text"] = f"done: {fields['findings']} answered, {fields['still_open']} still open ({fields['reason']})"

    out = deep.run_deep_research(
        question, complete=complete, web_search=search_tool.handler, web_fetch=fetch_tool.handler,
        write_file=write_file, create_file=create_file, dest_dir=DEST_DIR, progress=progress,
        stop_check=lambda: bool(job.get("stop")), **kwargs,
    )
    result = out["result"]
    summary = {
        "path": out["path"], "markdown": out["markdown"],
        "questions_answered": sum(1 for f in result.findings if f.status == "answered"),
        "still_open": result.still_open, "steps": result.steps, "stop_reason": result.stop_reason,
        "compactions": len(result.compactions),
    }
    job["result"] = summary
    return summary


def make_deep_research_tool(
    client: Any, config: dict[str, Any], console: Any,
    store: Store | None = None, session_id: int | None = None,
) -> Tool:
    def handler(question: str = "", max_subquestions: int = 12, budget_chars: int = 20000, **_: Any) -> str:
        question = question.strip()
        if not question:
            return "Error: give a question to research."
        tag = question[:200]
        if _jobs.running(JOB_KIND, question=tag):
            return "A deep research job for this question is already running - ask me for its status instead of starting another."
        jid = _jobs.start(
            JOB_KIND,
            lambda job: run_job(job, question, client, config, console, store, session_id,
                                max_steps=max(1, min(int(max_subquestions), 60)),
                                budget_chars=max(2000, min(int(budget_chars), 200000))),
            question=tag,
        )
        return (f"Started deep research job {jid} for: {question!r}. This runs in the background (search, read "
                "several pages per sub-question, follow newly discovered sub-questions, write a report) and can "
                f"take several minutes - ask me for its status later, or poll /api/research/deep/{jid}. The "
                f"finished report is written to {DEST_DIR}/ and, in the web chat, also offered as a download card.")

    return Tool(
        name="deep_research",
        description=(
            "Start a deep, multi-step research run on a broad question: it keeps a growing list of "
            "sub-questions, searches and reads several pages per sub-question, follows newly discovered "
            "sub-questions, and writes a detailed Markdown report with inline source citations and trust "
            "ratings. Runs as a BACKGROUND JOB that can take several minutes - returns immediately with a "
            "job id, not the finished report. Use the ordinary web_search tool instead for a quick single "
            "lookup; call this only when the user explicitly wants a thorough, report-style investigation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The broad research question to investigate."},
                "max_subquestions": {"type": "integer",
                                     "description": "Upper bound on sub-questions explored (default 12, max 60)."},
                "budget_chars": {"type": "integer",
                                 "description": "Running-notes character budget before they get compacted (default 20000)."},
            },
            "required": ["question"],
        },
        handler=handler,
        risk="safe",
    )
