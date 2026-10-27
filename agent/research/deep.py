"""Deep research: a budget-aware loop over a growing todo list of sub-questions.

Phase 4 of the research roadmap. Where `web_search` is one search-then-answer turn, `DeepResearch` keeps
asking: it seeds a queue with the user's question, pops one sub-question at a time, searches/reads for it,
notices new sub-questions worth chasing and pushes them onto the same queue, and keeps a running size
estimate of what it has accumulated. When that estimate crosses `compact_at` (default 75%) of a configurable
character budget, it asks the model to compress everything learned so far into short notes and keeps going
with those notes standing in for the detailed history. It stops when the queue is empty or a hard step/time
budget is hit, then renders a Markdown report: one section per sub-question with inline source citations
(trust tier included), and a "Still open" section for anything left unanswered.

Model-agnostic on purpose: this module never imports `agent.llm` or talks to Ollama directly. It is built on
three plain callables the caller supplies:

  * `complete(messages) -> str`            one chat-completion call, no streaming, no model name.
  * `web_search(**kwargs) -> str`          exactly `agent.tools.web`'s `web_search` tool handler.
  * `web_fetch(**kwargs) -> str`           exactly `agent.tools.web`'s `web_fetch` tool handler.

`web_search`/`web_fetch` are the actual handlers from `agent.tools.web.make_web_tools(console)` (or any
fake with the same signature in tests) — DeepResearch never touches httpx, the HTML extractor, trust
scoring or the prompt-injection scrubber itself. Every page it reads still goes through netguard
(`agent.research.netguard`), trust tiering (`agent.research.trust`) and injection scrubbing
(`agent.research.injection`) exactly as an ordinary `web_search` call would, because it *is* one. For the
same reason `DeepResearch` does not call `agent.research.connectors.plan()/lookup()` a second time itself:
`web_search` already calls `connectors.lookup()` before it falls back to ordinary search engines (see
`agent/tools/web.py::web_search`), so doing it again here would just repeat the same network calls. What
`DeepResearch` adds on top is the structured per-source citation list, which it reads back from
`agent.research.evidence.last_search()` — the same record `web_search` leaves behind for the verify step —
rather than re-parsing `web_search`'s formatted text output.

Compaction choice (continue in-process, not a fresh sub-run): when the running estimate crosses budget,
this driver asks `complete()` for a compact summary and keeps going in the SAME process/instance, with that
summary replacing the detailed prompt-side history (`context_notes`). A "fresh sub-run" (new process/thread
seeded with just the notes) was considered and rejected: this driver is already invoked from a background
job (CLI command or a web job thread, see `agent/web/routes/research_deep.py`), so spawning a second job per
compaction would only add bookkeeping (job ids, polling, a queue that has to be handed across the boundary)
without buying anything that replacing a few instance attributes does not already buy for a single-process
tool. The queue and the per-question report sections are never touched by compaction — only the text fed
back into `complete()` shrinks. `agent/chat/turn.py`'s own context/compaction machinery is untouched; this
is an independent, smaller counter scoped to one DeepResearch run (chars, not that turn's token accounting).
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from agent.jsonx import extract_json_value
from agent.research import evidence as turn_evidence

Complete = Callable[[list[dict[str, str]]], str]
WebSearch = Callable[..., str]
WebFetch = Callable[..., str]
Progress = Callable[[str, dict[str, Any]], None]

DEFAULT_BUDGET_CHARS = 20000
DEFAULT_COMPACT_AT = 0.75
DEFAULT_MAX_STEPS = 25
DEFAULT_MAX_SECONDS = 1800.0
DEFAULT_MAX_QUESTIONS = 40
DEFAULT_RESULTS_PER_QUESTION = 4
DEFAULT_NEW_QUESTIONS_PER_STEP = 3

_SYS_PROMPT = (
    "You are a careful research assistant working through one sub-question of a larger investigation. "
    "You are given numbered source passages (each already rated for trust). Using ONLY those sources:\n"
    "1. Answer the sub-question, citing sources as [n].\n"
    "2. If the sources do not answer it, say so plainly instead of guessing.\n"
    "3. List up to 3 NEW sub-questions that would help the larger investigation and are not already "
    "answered by what you were given.\n"
    "Reply with exactly one JSON object: {\"answer\": \"...\", \"new_questions\": [\"...\"]}. "
    "new_questions may be an empty list."
)

_COMPACT_PROMPT = (
    "The notes below are getting long. Compress them into a short brief (at most a few hundred words) that "
    "keeps every concrete fact, figure, date and source domain a later step would need, dropping only "
    "redundant wording. Reply with the compacted notes as plain text, nothing else."
)


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit].rstrip("-")) or "research"


@dataclass
class TodoItem:
    question: str
    origin: str                    # "seed" | "discovered"
    source_q: str = ""             # which question raised this one, for traceability


@dataclass
class Finding:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "answered"       # "answered" | "no_evidence" | "search_failed"
    origin: str = "seed"


@dataclass
class RunResult:
    question: str
    findings: list[Finding]
    still_open: list[str]
    compactions: list[str]
    steps: int
    questions_seen: int
    stop_reason: str
    chars_used: int
    budget_chars: int
    elapsed_s: float


class DeepResearch:
    """One deep-research run. Construct it, call `run()` once, read back a `RunResult`."""

    def __init__(
        self,
        question: str,
        *,
        complete: Complete,
        web_search: WebSearch,
        web_fetch: WebFetch,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        compact_at: float = DEFAULT_COMPACT_AT,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        max_questions: int = DEFAULT_MAX_QUESTIONS,
        results_per_question: int = DEFAULT_RESULTS_PER_QUESTION,
        new_questions_per_step: int = DEFAULT_NEW_QUESTIONS_PER_STEP,
        progress: Progress | None = None,
        stop_check: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        self.question = question
        self._complete = complete
        self._web_search = web_search
        self._web_fetch = web_fetch
        self.budget_chars = max(200, int(budget_chars))
        self.compact_at = max(0.1, min(0.99, float(compact_at)))
        self.max_steps = max(1, int(max_steps))
        self.max_seconds = max(1.0, float(max_seconds))
        self.max_questions = max(1, int(max_questions))
        self.results_per_question = max(1, int(results_per_question))
        self.new_questions_per_step = max(0, int(new_questions_per_step))
        self._progress = progress
        self._stop_check = stop_check
        self._clock = clock

        self.todo: deque[TodoItem] = deque([TodoItem(question, "seed")])
        self.seen_questions: set[str] = {_norm(question)}
        self.findings: list[Finding] = []
        self.compactions: list[str] = []
        self.context_notes = ""
        self.steps = 0
        self.stop_requested = False
        self._stop_reason = ""
        self._t0 = clock()  # reset again in run(); set here too so step() alone (e.g. in tests) has a clock to measure against

    # -------------------------------------------------------------------------------------------- plumbing

    def _emit(self, event: str, **fields: Any) -> None:
        if self._progress is not None:
            try:
                self._progress(event, fields)
            except Exception:  # noqa: BLE001 - a lost progress callback must never break the run
                pass

    def _ask(self, prompt: str) -> str:
        messages = [{"role": "system", "content": _SYS_PROMPT}, {"role": "user", "content": prompt}]
        try:
            return self._complete(messages) or ""
        except Exception as exc:  # noqa: BLE001 - a dead model must not crash the whole research run
            return f'{{"answer": "(the model could not be reached: {type(exc).__name__}) ", "new_questions": []}}'

    def _parse(self, raw: str) -> tuple[str, list[str]]:
        obj = extract_json_value(raw, "{", "}")
        if isinstance(obj, dict) and "answer" in obj:
            answer = str(obj.get("answer") or "").strip()
            new_qs = [str(q).strip() for q in (obj.get("new_questions") or []) if str(q).strip()]
            return answer or raw.strip(), new_qs
        return raw.strip(), []

    # -------------------------------------------------------------------------------------------- the loop

    def _budget_fraction(self) -> float:
        return len(self.context_notes) / self.budget_chars

    def _time_left(self) -> bool:
        return (self._clock() - self._t0) < self.max_seconds

    def _enqueue(self, text: str, source_q: str) -> bool:
        text = text.strip().rstrip("?").strip()
        if not text:
            return False
        text = text + "?" if not text.endswith(("?", ".", "!")) else text
        key = _norm(text)
        if key in self.seen_questions or len(self.seen_questions) >= self.max_questions:
            return False
        self.seen_questions.add(key)
        self.todo.append(TodoItem(text, "discovered", source_q=source_q))
        return True

    def _compact(self) -> None:
        notes = self.context_notes
        summary = self._ask(f"{_COMPACT_PROMPT}\n\n{notes}")
        parsed, _ = self._parse(summary)
        compacted = parsed if parsed else summary.strip()
        compacted = compacted or notes[-2000:]
        self.compactions.append(compacted)
        self.context_notes = compacted
        self._emit("compacted", before_chars=len(notes), after_chars=len(compacted), step=self.steps)

    def step(self) -> bool:
        """Do one unit of work. Returns False when the run is over (nothing left, or a budget was hit)."""
        if self.stop_requested or (self._stop_check is not None and self._stop_check()):
            self._stop_reason = self._stop_reason or "stopped"
            return False
        if not self.todo:
            self._stop_reason = "queue exhausted"
            return False
        if self.steps >= self.max_steps:
            self._stop_reason = "step budget reached"
            return False
        if not self._time_left():
            self._stop_reason = "time budget reached"
            return False

        item = self.todo.popleft()
        self.steps += 1
        self._emit("question_start", question=item.question, step=self.steps, queue_len=len(self.todo))

        try:
            self._web_search(query=item.question, num_results=self.results_per_question, read=True)
        except Exception as exc:  # noqa: BLE001 - one dead search must not end the whole run
            self.findings.append(Finding(item.question, f"Search failed: {exc}", [], "search_failed", item.origin))
            self.context_notes += f"\nQ: {item.question}\nA: (search failed: {exc})\n"
            self._emit("question_done", question=item.question, status="search_failed")
            if self._budget_fraction() >= self.compact_at:
                self._compact()
            return True

        record = turn_evidence.last_search()
        items = [it for it in record.get("items", []) if it.get("status") in ("ok", "cached") and it.get("passage")]
        if not items:
            self.findings.append(Finding(item.question, "No sources with usable evidence were found for this question.",
                                         [], "no_evidence", item.origin))
            self.context_notes += f"\nQ: {item.question}\nA: no evidence found.\n"
            self._emit("question_done", question=item.question, status="no_evidence")
            if self._budget_fraction() >= self.compact_at:
                self._compact()
            return True

        passage_block = "\n\n".join(
            f"[{it['n']}] {it.get('title', '')} - {it.get('domain', '')}"
            f"{' - ' + it['date'] if it.get('date') else ''}"
            f"{' - ' + it['trust']['label'] if it.get('trust') and it['trust'].get('label') else ''}\n{it['passage']}"
            for it in items
        )
        # web_search's own passages are capped short (it budgets across many hits at once); read the
        # single best source for this sub-question more deeply instead of just trusting that excerpt.
        for it in items[:1]:
            try:
                deeper = self._web_fetch(url=it["url"], focus=item.question, max_chars=2500)
            except Exception:  # noqa: BLE001 - a deep-read failure just means we keep the shorter passage
                continue
            if deeper and not deeper.startswith("Could not read"):
                passage_block += f"\n\n[{it['n']}, deeper read] {deeper}"
        context_block = f"What is already known so far:\n{self.context_notes}\n\n" if self.context_notes else ""
        prompt = f"{context_block}Sub-question: {item.question}\n\nSources:\n{passage_block}"
        analysis = self._ask(prompt)
        answer, new_qs = self._parse(analysis)

        citations = [
            {"n": it["n"], "url": it["url"], "domain": it.get("domain", ""),
             "date": it.get("date") or "", "trust": (it.get("trust") or {}).get("label", "")}
            for it in items
        ]
        self.findings.append(Finding(item.question, answer, citations, "answered", item.origin))
        added = sum(1 for q in new_qs[: self.new_questions_per_step] if self._enqueue(q, item.question))
        self.context_notes += f"\nQ: {item.question}\nA: {answer}\n"
        self._emit("question_done", question=item.question, status="answered", new_questions=added)

        if self._budget_fraction() >= self.compact_at:
            self._compact()
        return True

    def stop(self) -> None:
        """Ask the run to end after its current step (checked at the top of the next `step()`)."""
        self.stop_requested = True

    def run(self) -> RunResult:
        self._t0 = self._clock()
        while self.step():
            pass
        still_open = [f.question for f in self.findings if f.status != "answered"]
        still_open += [t.question for t in self.todo]
        elapsed = self._clock() - self._t0
        self._emit("run_done", steps=self.steps, findings=len(self.findings), still_open=len(still_open),
                   reason=self._stop_reason)
        return RunResult(
            question=self.question, findings=list(self.findings), still_open=still_open,
            compactions=list(self.compactions), steps=self.steps, questions_seen=len(self.seen_questions),
            stop_reason=self._stop_reason or "queue exhausted", chars_used=len(self.context_notes),
            budget_chars=self.budget_chars, elapsed_s=elapsed,
        )


# ------------------------------------------------------------------------------------------------------ report


_STATUS_LABEL = {"answered": "answered", "no_evidence": "no evidence found", "search_failed": "search failed"}


def render_markdown(result: RunResult) -> str:
    """The detailed `.md` report: one section per sub-question, inline citations, a "Still open" section."""
    lines = [f"# Deep research: {result.question}", "",
              f"*Generated {date.today().isoformat()} - {result.steps} sub-question(s) explored, "
              f"{len(result.still_open)} still open, stopped because: {result.stop_reason}.*", ""]
    for i, f in enumerate(result.findings, 1):
        tag = "" if f.status == "answered" else f" ({_STATUS_LABEL.get(f.status, f.status)})"
        lines.append(f"## {i}. {f.question}{tag}")
        lines.append("")
        lines.append(f.answer or "(no answer)")
        if f.citations:
            lines.append("")
            lines.append("**Sources:**")
            for c in f.citations:
                tier = f" ({c['trust']})" if c.get("trust") else ""
                when = f" - {c['date']}" if c.get("date") else ""
                lines.append(f"- [{c['n']}] {c.get('domain', c['url'])}{tier}{when} - {c['url']}")
        lines.append("")

    lines.append("## Still open")
    lines.append("")
    if result.still_open:
        for q in result.still_open:
            lines.append(f"- {q}")
    else:
        lines.append("Nothing - every sub-question in the queue was answered.")
    lines.append("")

    if result.compactions:
        lines.append("## Research notes (compaction log)")
        lines.append("")
        lines.append(f"*The running notes were compacted {len(result.compactions)} time(s) to stay inside the "
                     f"{result.budget_chars}-character budget. The last compacted summary:*")
        lines.append("")
        lines.append(result.compactions[-1])
        lines.append("")

    lines.append("---")
    lines.append(f"*{result.questions_seen} question(s) seen in total - {result.chars_used} of "
                 f"{result.budget_chars} notes characters used - {result.elapsed_s:.1f}s.*")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------- orchestration


WriteFile = Callable[..., str]
CreateFile = Callable[..., str]


def run_deep_research(
    question: str,
    *,
    complete: Complete,
    web_search: WebSearch,
    web_fetch: WebFetch,
    write_file: WriteFile,
    create_file: CreateFile | None = None,
    dest_dir: str = "deep_research",
    filename: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a full `DeepResearch` pass and deliver its report.

    The report is always written to disk via `agent.tools.files.write_file` (so it lands wherever that tool
    lands - the project/session directory the caller is already working in). When `create_file` is given
    (the handler of `agent.tools.create_file`'s tool, built for one chat session), the same Markdown is also
    handed to it so the web/terminal chat shows a downloadable file card - this reuses that tool's own
    file-registration call instead of duplicating it.
    """
    dr = DeepResearch(question, complete=complete, web_search=web_search, web_fetch=web_fetch, **kwargs)
    result = dr.run()
    markdown = render_markdown(result)
    name = filename or f"{slug(question)}.md"
    path = f"{dest_dir.rstrip('/')}/{name}"
    write_note = write_file(path=path, content=markdown)
    out: dict[str, Any] = {"path": path, "write_note": write_note, "markdown": markdown, "result": result}
    if create_file is not None:
        out["create_file_note"] = create_file(filename=name, content=markdown)
    return out
