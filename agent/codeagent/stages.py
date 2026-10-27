"""Stages: a big task runs as several small tasks, each in a FRESH, small context.

Measured live: one long session grows to a 78k-token prompt, prefill then eats 35-84% of the wall time (2.5 steps/min instead of
7-9) and every resume makes the model re-read the repo. So a big task is split into STAGES. Each stage runs in a new context that
holds exactly: the system prompt, the brief card (brief.py), the STAGE GOAL (pinned, context.py), one line per earlier stage, and a
VERIFIED STATE block (verified.py) measured by the harness. State that is not context - ledger, change log, checks, look/act
counters - lives on in the engine, so nothing is lost and no model prose carries the work from one stage to the next.

    split(task, llm=None, cfg=...) -> StagePlan       structure first (headers, headings, lists), a model only if asked to, else ONE stage
    run_stages(engine, plan, save=...) -> StageReport  CodeEngine unchanged as the per-stage worker (engine.begin_stage / stage_cap)

A stage ends when its final answer is accepted AND the engine's final gate (checks, tests) is green. Red: it stays active for up to
coding.stage_attempts engine segments, then it is BLOCKED (reason recorded) and the run goes on with the next stage. A segment that
reaches the absolute prompt cap (coding.stage_prompt_cap, default 28000 tokens) is not a failure: the same stage continues in a new
fresh context; two such segments in a row that changed no file end it as blocked ("no progress").
Config (coding.*): stages 'off' (default) | 'auto' (staged when split() finds >= 2 stages) | 'on', stage_prompt_cap, stage_attempts,
stage_split_with_model (default true: a task without structure of stage_split_min_chars (160) or more is split by ONE main-model call).
A stage that ends at the cap for the coding.resplit_after_caps-th time (default 2) is RE-SPLIT once (coding.resplit 'auto' | 'off'): it is too big
for one context, so ONE main-model call (or, if its answer is unusable, the files named in the goal / most read, <= ~600 lines per part) replaces
it in the plan by 2..6 smaller sub-stages (parent / depth set, ids renumbered, sub-stages never split again) that name their files.
A stage whose goal NAMES 1-3 existing files (explicit names / paths, or 'the X module' with an X.py) is scoped to exactly those files: neither the
model split nor the deterministic fallback may use any other repo file. A named file above coding.range_split_lines (default 260) is split WITHOUT a
model call into consecutive LINE-RANGE parts of about 120-160 lines (aligned to top-level definitions / blank lines, at most 8 parts over all named
files, `Stage.lines` = (first, last)); the cursor, the scope gate and the read notes of such a part speak in that range (cursor.py), and
coding.out_of_scope_read_weight (default 2) makes a read of another file count that many steps toward the look pause.
HARD BUDGET per stage (coding.stage_max_minutes 25, stage_max_steps 180, stage_min_edits_per_100_steps 3): the stage records spent_s / spent_steps /
real_edits over all its segments (segment durations only, no idle gaps). At a segment boundary the supervisor blocks it ('budget: ...') when time
OR steps are used up and the edits do not keep pace (real_edits < ceil(steps / 100 * min_edits)); a stage that edits at that rate may run up to 3x the
budget. `budget_blocked` is the pure rule; the retry round / the harness close rule live in supervisor.py."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from rich.markup import escape

from agent.codeagent import brief as brief_mod, checks as checks_mod, cursor as cursor_mod, fileseen, verified

DEFAULT_CAP = 28000
DEFAULT_ATTEMPTS = 3
MAX_STAGES = 12
MIN_LIST_BLOCK = 1500                                  # a list/heading block smaller than this is a requirement of one job, not a stage of its own
MIN_BLOCK = 80                     # a block shorter than this is merged with its neighbour
MODEL_SPLIT_MIN_CHARS = 160                  # default of coding.stage_split_min_chars
MODEL_MAX_STAGES, MODEL_MAX_TITLE, MODEL_GOAL_RANGE, MODEL_NUM_PREDICT = 8, 80, (20, 1200), 600
SUMMARY_MAX = 600
RESPLIT_AFTER_CAPS = 2                       # default of coding.resplit_after_caps
RESPLIT_RANGE, RESPLIT_NUM_PREDICT, RESPLIT_MAX_FILES = (2, 6), 900, 12
RESPLIT_CHUNK_LINES, RESPLIT_LISTED_FILES = 600, 60
RANGE_SPLIT_LINES, RANGE_TARGET_LINES, RANGE_MAX_PARTS = 260, 140, 8    # coding.range_split_lines; a part is about this long; parts over all named files
RANGE_MIN_PART, RANGE_WINDOW, MAX_NAMED = 40, 25, 3                    # a part is never shorter; cuts move at most this far to a boundary; named-file limit
BROAD_MIN_FILES, BROAD_MIN_LINES = 3, 1200            # coding.broad_split: a broad goal is split at stage start in a repo of at least this size
TEST_IDS_SHOWN = 6                                     # failing tests named in the next segment's note after a red stage-end test run
_CREATES = re.compile(r"^\W*(?:stage \d+\W+)?\s*(?:add|implement|build|create|introduce|design|develop|write|set up|erstelle|baue|füge|implementiere|entwickle)\b", re.I | re.M)   # a goal that opens with a creation verb is a feature
TESTS_TITLE = "Run the tests and fix what YOUR rewrites broke"       # the last stage of a range plan
_TESTS_WORD = re.compile(r"\b(?:tests?|pytest|unittests?)\b", re.I)
_REMOVES = re.compile(r"\b(?:remov|delet|strip|eliminat|drop all|entfern|lösch)", re.I)
_BROAD = re.compile(r"\b(?:all|every|each|whole|entire|throughout|alle|jede|jeden|jedes|sämtliche|überall)\b", re.I)
LARGE_FILE_CHARS = 12000                     # a sub-stage file above this is read by line range only (LARGE FILES hint)
STATUSES = ("pending", "active", "done", "blocked")
STAGE_MAX_MINUTES, STAGE_MAX_STEPS, STAGE_MIN_EDITS = 25.0, 180, 3.0       # defaults of the coding.stage_* budget keys
BUDGET_CEILING, BUDGET_CLOSE_SHARE = 3, 0.6        # a healthy stage may use 3x the budget; the harness may close a satisfied stage at 60 % of it
CLOSE_NOTE = "closed by the harness: scope satisfied, no new test failures, no edits in the last 2 segments"
CAP_NOTE = "continuing: previous segment ended at the context cap; see VERIFIED STATE"

_STAGE_HEADER = re.compile(r"^[ \t]*[#>*_ \t]*(?:stage|stufe|phase|part|teil)[ \t]+\d+\b[ \t]*(?:[:.\-–—)]|$)", re.I)
_HEADING = re.compile(r"^(#{1,6})[ \t]+\S")
_ITEM = re.compile(r"^(?:\d+[.)]|[-*•])[ \t]+\S")                  # top level only: no indentation
_MODEL_SYSTEM = """Split the coding task into stages that are done one after the other, each in a fresh context.
Rules:
- ONE stage if the task is one coherent change (a bug fix, a small feature, a refactor). Most tasks are one stage.
- 2 to 6 stages ONLY for separable deliverables that can each be built and verified alone (for example: data model, game loop, menu, tests).
- Each goal is self-contained: it names the files or behaviour to build. Later stages may rely on what earlier stages built.
- The LAST stage may be "integrate and verify", only if it is really needed.
Output ONLY JSON: {"stages": [{"title": "<short>", "goal": "<what to do in this stage>"}]}"""
_RESPLIT_SYSTEM = """One stage of a coding task is too big for one context: it keeps filling the context while re-reading the same files.
Split THAT stage into 2 to 6 smaller sub-stages that are done one after the other, each in a fresh context.
Rules:
- Each sub-stage must be small enough to finish in about 20 steps and names the exact files (and sections of a big file) it covers.
- The sub-stages are DISJOINT in scope (no two sub-stages do the same work) and ordered.
- Parts that are already finished (the files listed under CHANGED SO FAR) are NOT done again: start with what is NOT yet done.
- Each goal is self-contained, 20 to 1200 characters. Only use files from the REPO FILES list or new file names that the stage needs.
- Every sub-stage must DELIVER something that changes files: code, text, or a test. Never create "verify", "audit", "review", "check" or "investigate" sub-stages (investigating is part of doing the work); a final sub-stage may run the test suite only if it also fixes what breaks.
Output ONLY JSON: {"stages": [{"title": "<short>", "goal": "<what to do>", "files": ["<path>", ...]}]}"""
_SAFE_PATH = re.compile(r"^[\w][\w./+-]*$")


def _coding(cfg: Any) -> dict[str, Any]:
    return (cfg or {}).get("coding") or {}


def _one_line(text: str, limit: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def budget_limits(cfg: Any, st: Stage | None = None) -> tuple[float, float, float]:
    """(seconds, steps, min edits per 100 steps) of one stage's budget from coding.*; 0 = that limit is off. A retry stage has half of it."""
    c, scale = _coding(cfg), (st.budget_scale if st is not None else 1.0)
    return (max(0.0, _num(c.get("stage_max_minutes", STAGE_MAX_MINUTES), STAGE_MAX_MINUTES)) * 60 * scale,
            max(0.0, _num(c.get("stage_max_steps", STAGE_MAX_STEPS), STAGE_MAX_STEPS)) * scale,
            max(0.0, _num(c.get("stage_min_edits_per_100_steps", STAGE_MIN_EDITS), STAGE_MIN_EDITS)))


def budget_numbers(st: Stage) -> str:
    return f"{st.spent_s / 60:.0f} min / {st.spent_steps} steps / {st.real_edits} edits"


def budget_share(st: Stage, cfg: Any) -> float:
    """How much of the base budget the stage has used: the larger of the time and the step share (0 when both limits are off)."""
    max_s, max_steps, _edits = budget_limits(cfg, st)
    return max(st.spent_s / max_s if max_s else 0.0, st.spent_steps / max_steps if max_steps else 0.0)


def budget_blocked(st: Stage, cfg: Any) -> str:
    """'' while the stage may go on, else the reason ('budget: <min> / <steps> / <edits>'). Pure and deterministic. The base budget is
    exceeded (time OR steps) and the stage did not make clear progress (real edits < ceil(steps / 100 * min_edits), at least 1); a stage that
    does edit at that rate may use up to BUDGET_CEILING x the budget."""
    max_s, max_steps, min_edits = budget_limits(cfg, st)
    over = (max_s and st.spent_s >= max_s) or (max_steps and st.spent_steps >= max_steps)
    ceiling = (max_s and st.spent_s >= BUDGET_CEILING * max_s) or (max_steps and st.spent_steps >= BUDGET_CEILING * max_steps)
    healthy = st.real_edits >= max(1, math.ceil(st.spent_steps / 100 * min_edits))
    return f"budget: {budget_numbers(st)}" if ceiling or (over and not healthy) else ""


@dataclass
class Stage:
    id: int
    title: str
    goal: str
    status: str = "pending"          # pending | active | done | blocked
    attempts: int = 0                # engine segments that ended for another reason than the prompt cap
    summary: str = ""                # <= 600 chars, from measurements + the last answer
    files: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    prompt_peak: int = 0
    seconds: float = 0.0             # time actually spent in the stage (started_at/finished_at also span a pause between processes)
    reason: str = ""                 # why a stage is blocked
    cap_segments: int = 0            # segments of this stage that ended at the prompt cap (the re-split trigger)
    parent: int = 0                  # id (before the re-split) of the stage this one was split from; 0 = an original stage
    depth: int = 0                   # 1 = a sub-stage: never split again
    scope: list[str] = field(default_factory=list)      # the files a sub-stage covers (planned; `files` = the ones it changed)
    waived: list[str] = field(default_factory=list)     # scope files the model said (mark_done / project_note naming them) need no change
    edited: dict[str, list[list[int]]] = field(default_factory=dict)     # per file: the line ranges written during the stage (cursor.py)
    spent_s: float = 0.0             # budget: model / tool wall seconds in this stage (segment durations; no idle gaps), over all segments and resumes
    spent_steps: int = 0             # budget: model steps (engine step counter delta per segment)
    real_edits: int = 0              # budget: successful non-scratch writes
    no_edit_segments: int = 0        # consecutive segments without a real edit (the harness close rule)
    budget_scale: float = 1.0        # 1.0 = the configured budget; 0.5 = the final retry round
    budget_round: int = 0            # 0 = first run; 1 = the one retry after a budget block
    seen: dict[str, dict[str, Any]] = field(default_factory=dict)       # per file: line ranges read, number of reads, last action (fileseen.py)
    lines: tuple[int, int] | None = None   # a ranged part: the (first, last) line of scope[0] it covers, in the file's CURRENT numbering (cursor.sync_range)
    lines_total: int = 0             # the length of that file the range was valid for (the shift of earlier parts = file length now - this)


def _lines_of(raw: Any) -> tuple[int, int] | None:
    """A saved line range: [first, last] with 1 <= first <= last, else None."""
    try:
        a, b = int(raw[0]), int(raw[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (a, b) if len(raw) == 2 and 1 <= a <= b else None


@dataclass
class StagePlan:
    stages: list[Stage] = field(default_factory=list)
    current: int = 0
    source: str = "single"           # model | headings | single
    task: str = ""
    baseline_red: list[str] | None = None   # failing test ids known BEFORE the current stage (None = not measured yet; old sessions)
    baseline_ok: bool = True                # False = the baseline run itself gave no result (timeout ...): the scope-name fallback applies

    def unfinished(self) -> bool:
        return any(s.status in ("pending", "active") for s in self.stages)

    def to_json(self) -> dict[str, Any]:
        return {"source": self.source, "current": self.current, "task": self.task, "stages": [asdict(s) for s in self.stages],
                "baseline_red": None if self.baseline_red is None else sorted(self.baseline_red), "baseline_ok": self.baseline_ok}

    @classmethod
    def from_json(cls, data: Any) -> "StagePlan | None":
        """None when `data` is absent or unusable (an old session has no `stages` key)."""
        if not isinstance(data, dict) or not isinstance(data.get("stages"), list):
            return None
        stages: list[Stage] = []
        for i, d in enumerate(data["stages"], 1):
            try:
                st = Stage(id=int(d.get("id", i)), title=str(d.get("title") or f"stage {i}"), goal=str(d["goal"]))
                st.status = str(d.get("status")) if d.get("status") in STATUSES else "pending"
                st.attempts, st.prompt_peak = int(d.get("attempts") or 0), int(d.get("prompt_peak") or 0)
                st.summary, st.reason = str(d.get("summary") or "")[:SUMMARY_MAX], str(d.get("reason") or "")
                st.files = [str(x) for x in d.get("files") or []]
                st.scope = [str(x) for x in d.get("scope") or []]
                st.waived = [str(x) for x in d.get("waived") or []]
                raw = d.get("edited")                           # absent in a session saved before the cursor existed
                st.edited = {str(k): cursor_mod.merge(v) for k, v in raw.items() if isinstance(v, list)} if isinstance(raw, dict) else {}
                st.cap_segments, st.parent, st.depth = (int(d.get(k) or 0) for k in ("cap_segments", "parent", "depth"))
                st.started_at, st.finished_at, st.seconds = (float(d.get(k) or 0) for k in ("started_at", "finished_at", "seconds"))
                st.spent_steps, st.real_edits, st.no_edit_segments, st.budget_round = (int(d.get(k) or 0) for k in
                                                                                       ("spent_steps", "real_edits", "no_edit_segments", "budget_round"))
                st.spent_s, st.budget_scale = float(d.get("spent_s") or 0), float(d.get("budget_scale") or 1.0)       # absent in older sessions
                st.seen = fileseen.load(d.get("seen"))          # absent in a session saved before the record existed
                st.lines, st.lines_total = _lines_of(d.get("lines")), max(0, int(d.get("lines_total") or 0))      # absent before ranged parts existed
            except (AttributeError, KeyError, TypeError, ValueError):
                return None
            stages.append(st)
        if not stages:
            return None
        plan = cls(stages, int(data.get("current") or 0), str(data.get("source") or "single"), str(data.get("task") or ""))
        raw_red = data.get("baseline_red")                      # absent in a session saved before the baseline existed
        plan.baseline_red = [str(x) for x in raw_red] if isinstance(raw_red, list) else None
        plan.baseline_ok = data.get("baseline_ok") is not False
        return plan

    def table(self) -> list[str]:
        return [f"  {s.id:>2}  {s.status:<8} {s.attempts}x  {s.title}" for s in self.stages]


@dataclass
class StageReport:
    verdict: str                                  # done | partial | failed
    stages: list[Stage] = field(default_factory=list)


# ── splitting ───────────────────────────────────────────────────────────────

def _title(block: str) -> str:
    first = next((ln for ln in block.splitlines() if ln.strip()), "")
    return _one_line(re.sub(r"^[#>*_\s]+|^(?:\d+[.)]|[-*•])\s+", "", first.strip()).strip("*_ "), 70) or "stage"


def _fence_free(lines: list[str]) -> list[bool]:
    """True for each line that is outside a ``` fence (a `# comment` in a code block is not a heading)."""
    out, inside = [], False
    for ln in lines:
        if ln.lstrip().startswith("```"):
            inside = not inside
            out.append(False)
        else:
            out.append(not inside)
    return out


def _cut(lines: list[str], starts: list[int]) -> list[str]:
    """The text between consecutive start lines (what comes before the first start belongs to no stage: the brief card keeps it)."""
    ends = [*starts[1:], len(lines)]
    return ["\n".join(lines[a:b]).strip() for a, b in zip(starts, ends)]


def _structure(task: str) -> tuple[list[str], bool]:
    """(blocks, explicit): explicit = the task itself says 'Stage N'; headings and bullet lists are often just the requirements of ONE job."""
    lines = task.splitlines()
    free = _fence_free(lines)
    headers = [i for i, ln in enumerate(lines) if free[i] and _STAGE_HEADER.match(ln)]
    if len(headers) >= 2:
        return _cut(lines, headers), True
    heads = [(i, len(m.group(1))) for i, ln in enumerate(lines) if free[i] and (m := _HEADING.match(ln))]
    for level in sorted({lv for _i, lv in heads}):                  # the highest level that occurs at least twice
        starts = [i for i, lv in heads if lv == level]
        if len(starts) >= 2:
            return _cut(lines, starts), False
    items = [i for i, ln in enumerate(lines) if free[i] and _ITEM.match(ln)]
    return (_cut(lines, items), False) if len(items) >= 3 else ([], False)


def _merge_tiny(blocks: list[str], minimum: int = MIN_BLOCK) -> list[str]:
    out: list[str] = []
    carry = ""
    for b in blocks:
        b = f"{carry}\n{b}".strip() if carry else b
        carry = ""
        if len(b) < minimum:
            carry = b
        else:
            out.append(b)
    if carry:
        if out:
            out[-1] += "\n" + carry
        else:
            out = [carry]
    return out


def _cap_count(blocks: list[str], limit: int = MAX_STAGES) -> list[str]:
    blocks = list(blocks)
    while len(blocks) > limit:
        i = min(range(len(blocks) - 1), key=lambda k: len(blocks[k]) + len(blocks[k + 1]))
        blocks[i:i + 2] = [blocks[i] + "\n" + blocks[i + 1]]
    return blocks


def _plan_of(task: str, pairs: list[tuple[str, str]], source: str) -> StagePlan:
    head = f"Overall task: {_one_line(task, 300)}"
    return StagePlan([Stage(n, title, f"{head}\n\n{goal}") for n, (title, goal) in enumerate(pairs, 1)], 0, source, task)


def _single(task: str) -> StagePlan:
    return StagePlan([Stage(1, _title(task), task)], 0, "single", task)       # the goal IS the task: behaviour identical to a plain turn


def split(task_text: str, *, llm: Any = None, cfg: dict[str, Any] | None = None, model: str | None = None,
          cache: dict[str, Any] | None = None, root: Any = None) -> StagePlan:
    """A task that names 1-3 files of `root`, one above coding.range_split_lines, and edits them by passes gets the deterministic RANGE PLAN
    (source 'ranges', no model call, see `range_plan`) before anything else. Else deterministic first (stage headers, markdown headings, a list of
    >= 3 top-level items), then - when coding.stage_split_with_model is on (default), the task has coding.stage_split_min_chars (default 160) or
    more and an llm is given - one non-thinking call of the MAIN model `model`, else a single stage. `cache` (the engine's, saved with the
    session) keeps the answer per task text: a repeat does not ask again."""
    task = str(task_text or "").strip()
    found, explicit = _structure(task)
    blocks = _cap_count(_merge_tiny([b for b in found if b], MIN_BLOCK if explicit else MIN_LIST_BLOCK))
    ranged = range_plan(task, root, cfg, blocks if len(blocks) >= 2 else None)
    if ranged is not None:
        return ranged
    if len(blocks) >= 2:
        return _plan_of(task, [(_title(b), b) for b in blocks], "headings")
    c = _coding(cfg)
    if llm is not None and len(task) >= _min_chars(c) and c.get("stage_split_with_model", True):
        pairs = _model_split(llm, cfg or {}, task, model, cache)
        if pairs:
            return _plan_of(task, pairs, "model")
    return _single(task)


def _min_chars(c: dict[str, Any]) -> int:
    try:
        return max(0, int(c.get("stage_split_min_chars", MODEL_SPLIT_MIN_CHARS)))
    except (TypeError, ValueError):
        return MODEL_SPLIT_MIN_CHARS


def _valid_pairs(data: Any) -> list[tuple[str, str]] | None:
    """2..8 stages, a title of at most 80 characters, a goal of 20..1200, no duplicate titles or goals; else None."""
    items = data.get("stages") if isinstance(data, dict) else None
    if not isinstance(items, list) or not 2 <= len(items) <= MODEL_MAX_STAGES:
        return None
    pairs: list[tuple[str, str]] = []
    for i in items:
        if not isinstance(i, dict) or not isinstance(i.get("title"), str) or not isinstance(i.get("goal"), str):
            return None
        title, goal = " ".join(i["title"].split()), i["goal"].strip()
        if not title or len(title) > MODEL_MAX_TITLE or not MODEL_GOAL_RANGE[0] <= len(goal) <= MODEL_GOAL_RANGE[1]:
            return None
        pairs.append((title, goal))
    if len({t.lower() for t, _g in pairs}) != len(pairs) or len({g.lower() for _t, g in pairs}) != len(pairs):
        return None
    return pairs


def _ask(llm: Any, cfg: dict[str, Any], system: str, user: str, model: str | None, num_predict: int) -> str | None:
    """ONE non-thinking background call of the MAIN model (no GPU swap); None when the call fails."""
    models = cfg.get("models") or {}
    model = model or (cfg.get("code") or {}).get("model") or models.get("chat") or models.get("background")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        try:
            return "".join(llm.chat(messages=msgs, model=model, stream=False, background=True, json_mode=True, extra_options={"num_predict": num_predict}))
        except TypeError:                              # a backend/stub without json_mode
            return "".join(llm.chat(messages=msgs, model=model, stream=False, background=True, extra_options={"num_predict": num_predict}))
    except Exception:  # noqa: BLE001 - a missing split must never block the task: it falls through to one stage (and is asked again next time)
        return None


def _model_split(llm: Any, cfg: dict[str, Any], task: str, model: str | None = None, cache: dict[str, Any] | None = None) -> list[tuple[str, str]] | None:
    key = hashlib.sha1(task.encode("utf-8")).hexdigest()
    if cache is not None and key in cache:
        return [(str(t), str(g)) for t, g in cache[key]] or None
    reply = _ask(llm, cfg, _MODEL_SYSTEM, task[:8000], model, MODEL_NUM_PREDICT)
    if reply is None:
        return None
    try:
        pairs = _valid_pairs(json.loads(reply[reply.find("{"):reply.rfind("}") + 1]))
    except Exception:  # noqa: BLE001 - unreadable JSON is the same as a one-stage answer
        pairs = None
    if cache is not None:
        cache[key] = [list(p) for p in pairs or []]
    return pairs


# ── re-splitting a stage that is too big for one context ────────────────────

_DATA_SUFFIX = (".json", ".db", ".sqlite", ".sqlite3", ".lock", ".log", ".pyc", ".flac", ".mp3", ".ogg", ".wav", ".png", ".jpg", ".jpeg", ".zip", ".bin")
_DATA_NAMES = ("save.json", "savegame", "save.dat", ".env", "config.local.yaml")


def _is_data_file(rel: str) -> bool:
    """Save games, databases, media and secrets are data, never a stage's scope (a stage must not be pushed to edit a player's save)."""
    low = rel.lower()
    base = low.rsplit("/", 1)[-1]
    return low.endswith(_DATA_SUFFIX) or base in _DATA_NAMES or base.startswith(("save.", ".env"))


def _repo_files(root: Any) -> list[tuple[str, int, int]]:
    """(relative path, lines, characters) of the code files of the repo (scratch and huge files left out), at most 400."""
    from pathlib import Path

    from agent.codeagent import edits
    from agent.codeagent.lookact import is_scratch_path

    base, out = Path(root), []
    for f in edits.iter_files(base, 400):
        try:
            rel = f.relative_to(base).as_posix()
            if is_scratch_path(rel) or f.stat().st_size > 2_000_000 or _is_data_file(rel):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        out.append((rel, text.count("\n") + 1, len(text)))
    return out


def _safe_path(name: Any) -> str | None:
    """A relative file name the model may name (an existing file or one to create), normalised; None for anything else."""
    if not isinstance(name, str):
        return None
    n = name.strip().removeprefix("./")
    if not n or len(n) > 120 or not _SAFE_PATH.match(n) or n.startswith("/") or ".." in n.split("/") or n.endswith("/") or _is_data_file(n):
        return None
    return n


def _valid_substages(data: Any) -> list[tuple[str, str, list[str]]] | None:
    """2..6 sub-stages: title <= 80, goal 20..1200, no duplicate titles / goals, `files` a list of <= 12 safe relative names; else None."""
    items = data.get("stages") if isinstance(data, dict) else None
    if not isinstance(items, list) or not RESPLIT_RANGE[0] <= len(items) <= RESPLIT_RANGE[1]:
        return None
    out: list[tuple[str, str, list[str]]] = []
    for i in items:
        if not isinstance(i, dict) or not isinstance(i.get("title"), str) or not isinstance(i.get("goal"), str):
            return None
        raw = i.get("files", [])
        if not isinstance(raw, list) or len(raw) > RESPLIT_MAX_FILES:
            return None
        files = [_safe_path(f) for f in raw]
        if None in files:
            return None
        title, goal = " ".join(i["title"].split()), i["goal"].strip()
        if not title or len(title) > MODEL_MAX_TITLE or not MODEL_GOAL_RANGE[0] <= len(goal) <= MODEL_GOAL_RANGE[1]:
            return None
        out.append((title, goal, list(dict.fromkeys(f for f in files if f))))
    if len({t.lower() for t, _g, _f in out}) != len(out) or len({g.lower() for _t, g, _f in out}) != len(out):
        return None
    return out


def _body(st: Stage) -> str:
    return st.goal.split("\n\n", 1)[1] if st.goal.startswith("Overall task:") and "\n\n" in st.goal else st.goal


_MODULE = re.compile(r"\bthe\s+([A-Za-z_]\w*)\s+module\b", re.I)


def _mention(text: str, name: str) -> int:
    """Position of the first whole-word mention of the file / path `name` in `text` (not inside a longer name or path), else -1."""
    m = re.search(r"(?<![\w./-])" + re.escape(name) + r"(?![\w])", text)
    return m.start() if m else -1


def named_files(st: Stage, repo: list[tuple[str, int, int]]) -> list[str]:
    """The existing repo files the stage's title + goal NAMES, in the order of the text: an explicit path, a bare file name (also for a file in a
    folder), or 'the X module' when X.py exists. 1..3 files, else [] (nothing named, or too many to be a scope). The 'Overall task:' head of a
    sub-stage is not part of its goal."""
    text, found = f"{st.title}\n{_body(st)}", {}
    for rel, _n, _c in repo:
        pos = [p for p in (_mention(text, rel), _mention(text, rel.rsplit("/", 1)[-1]) if "/" in rel else -1) if p >= 0]
        if pos:
            found[rel] = min(pos)
    for m in _MODULE.finditer(text):
        for rel, _n, _c in repo:
            if rel == m.group(1) + ".py" or rel.endswith("/" + m.group(1) + ".py"):
                found[rel] = min(found.get(rel, m.start()), m.start())
    ordered = sorted(found, key=lambda r: (found[r], r))
    return ordered if 1 <= len(ordered) <= MAX_NAMED else []


def _file_rows(root: Any, rel: str) -> list[str]:
    from pathlib import Path

    try:
        return (Path(root) / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _top_level_starts(rel: str, rows: list[str]) -> set[int]:
    """First line (decorators included) of every top-level def / class of a python file: the best places to cut it."""
    if not rel.endswith(".py"):
        return set()
    from agent.codeagent import symbols

    return {sp.start for sp in symbols.python_spans("\n".join(rows)) if "." not in sp.name and sp.kind in ("def", "class")}


def _cut_points(rel: str, rows: list[str], first: int, parts: int) -> list[int]:
    """The first lines of parts 2..`parts` when lines `first`..len(rows) are cut into `parts` parts of about equal length. Each cut moves (at most
    RANGE_WINDOW lines) to the nearest top-level definition, else the nearest line that follows a blank line / starts a heading, else it stays."""
    last = len(rows)
    size, defs, cuts, prev = last - first + 1, _top_level_starts(rel, rows), [], first

    def blank_before(c: int) -> bool:
        return 2 <= c <= last and rows[c - 2].strip() == "" and rows[c - 1].strip() != ""

    for k in range(1, parts):
        target = first + round(k * size / parts)
        lo, hi = max(prev + RANGE_MIN_PART, target - RANGE_WINDOW), min(last - RANGE_MIN_PART * (parts - k) + 1, target + RANGE_WINDOW)
        pool = range(lo, hi + 1)
        best = next((c for tier in (lambda c: c in defs, blank_before) for c in sorted(pool, key=lambda c: (abs(c - target), c)) if tier(c)), None)
        prev = best if best is not None else min(max(target, lo), max(lo, last - RANGE_MIN_PART + 1))
        cuts.append(prev)
    return cuts


def _range_substages(st: Stage, named: list[str], root: Any, min_lines: int) -> tuple[list[tuple[str, str, list[str]]], list[tuple[int, int, int] | None]] | None:
    """Split the named files WITHOUT a model call: every file above `min_lines` lines becomes consecutive line-range parts of about
    RANGE_TARGET_LINES (starting after the lines the stage already finished from line 1), a smaller file stays one whole-file part, at most
    RANGE_MAX_PARTS parts in all, in the order of `named`. Returns the (title, goal, files) rows and, parallel, (first, last, file length) of each ranged
    part (None for a whole-file part); None when no named file is above `min_lines` or fewer than 2 parts result."""
    if min_lines <= 0:
        return None
    plan: list[tuple[str, list[str], int, int]] = []              # (file, rows, first line, parts) of a big file; parts 0 = whole file
    for rel in named:
        rows = _file_rows(root, rel)
        first = cursor_mod.next_line((st.edited or {}).get(rel)) if (st.edited or {}).get(rel) else 1
        count = len(rows) - first + 1
        big = len(rows) > min_lines and count >= 2
        plan.append((rel, rows, first, max(2, round(count / RANGE_TARGET_LINES)) if big else 0))
    if not any(parts for _r, _w, _f, parts in plan):
        return None
    counts = [p for _r, _w, _f, p in plan]
    while sum(max(c, 1) for c in counts) > RANGE_MAX_PARTS and max(counts) > 2:
        counts[counts.index(max(counts))] -= 1
    jobs: list[tuple[str, tuple[int, int, int] | None]] = []
    for (rel, rows, first, _p), n in zip(plan, counts):
        if not n:
            jobs.append((rel, None))
            continue
        starts = [first, *_cut_points(rel, rows, first, n)]
        jobs += [(rel, (a, b, len(rows))) for a, b in zip(starts, [c - 1 for c in starts[1:]] + [len(rows)])]
    if len(jobs) < 2 or len(jobs) > RANGE_MAX_PARTS:
        return None
    instruction, total = _one_line(_body(st), 900), len(jobs)
    subs, ranges = [], []
    for i, (rel, r) in enumerate(jobs, 1):
        if r is None:
            subs.append((f"Part {i}/{total}: {rel}", f"Work ONLY on {rel} (part {i} of {total} of the stage \"{st.title}\"; the other parts cover the other files or line ranges). "
                         f"{instruction}", [rel]))
        else:
            subs.append((f"Part {i}/{total}: {rel} lines {r[0]}-{r[1]}",
                         f"Work ONLY on {rel} lines {r[0]}-{r[1]} (the rest of the file is handled by other parts). {instruction}", [rel]))
        ranges.append(r)
    return subs, ranges


def range_lines_of(cfg: Any) -> int:
    """coding.range_split_lines (default 260; 0 = never range-split)."""
    try:
        return max(0, int((_coding(cfg)).get("range_split_lines", RANGE_SPLIT_LINES)))
    except (TypeError, ValueError):
        return RANGE_SPLIT_LINES


def range_job(task: str, root: Any, cfg: Any) -> tuple[Stage, list[str], list[tuple[str, int, int]], list[str]] | None:
    """(the task as one stage, the files it names, the repo files, the named files above coding.range_split_lines) when `task` is a range job:
    1-3 named files of `root`, at least one above the threshold, editing-by-passes intent (cursor.INTENT / REMOVES) and no creation verb at its
    start. Cheap and deterministic (names + sizes only): the autopilot predicate uses it too. None otherwise."""
    limit = range_lines_of(cfg)
    if root is None or limit <= 0 or not task.strip() or _CREATES.search(task[:160]):
        return None
    if not (cursor_mod.INTENT.search(task) or _REMOVES.search(task)):
        return None
    try:
        repo = _repo_files(root)
        st = Stage(1, _title(task), task)
        named = named_files(st, repo)
        big = [rel for rel in named if len(_file_rows(root, rel)) > limit]
    except Exception:  # noqa: BLE001 - an optimisation: never break the split
        return None
    return (st, named, repo, big) if big else None


def range_plan(task: str, root: Any, cfg: Any, blocks: list[str] | None = None) -> StagePlan | None:
    """The deterministic plan of a range job (source 'ranges'): the sub-stages `_range_substages` builds (Part k/n: file lines a-b, scope = the
    file, no other repo file) plus ONE last normal stage 'Run the tests and fix what YOUR rewrites broke' when the task mentions tests or the repo
    has test files. `blocks` = the stages the task text itself is structured into (headings / stage headers): they are replaced only when each of
    them names every big file (the range plan is finer); blocks about other files keep their plan. None when `task` is no range job."""
    job = range_job(task, root, cfg)
    if job is None:
        return None
    st, named, repo, big = job
    if blocks and not all(set(big) <= set(named_files(Stage(1, _title(b), b), repo)) for b in blocks):
        return None
    ranged = _range_substages(st, named, root, range_lines_of(cfg))
    if ranged is None:
        return None
    head = f"Overall task: {_one_line(task, 300)}"
    subs, ranges = ranged
    stages = [_sub_stage(head, title, goal, files, r, 0) for (title, goal, files), r in zip(subs, ranges)]
    if _TESTS_WORD.search(task) or any(_is_test_path(rel) for rel, _n, _c in repo):
        stages.append(Stage(0, TESTS_TITLE, f"{head}\n\nRun the project's tests once and fix only what the edits of the earlier parts broke. Change nothing else."))
    for n, s2 in enumerate(stages, 1):
        s2.id = n
    return StagePlan(stages, 0, "ranges", task)


def _sub_stage(head: str, title: str, goal: str, files: list[str], r: tuple[int, int, int] | None, parent: int) -> Stage:
    """A sub-stage (id set by the caller): scope = its files, `lines` / `lines_total` when it covers a line range."""
    return Stage(0, title, f"{head}\n\n{goal}" + (f"\nFiles of this part: {', '.join(files)}" if files else ""), scope=list(files),
                 parent=parent, depth=1, lines=(r[0], r[1]) if r else None, lines_total=r[2] if r else 0)


_STOP_WORDS = frozenset("""stage stufe phase part teil remove removes removing every which their there these those would should could about after before
    where while with without from into only make makes keep keeps change changes changed also them then than that this have must messages message
    other files file code tests test including include includes""".split())


def _goal_keywords(st: Stage) -> list[str]:
    """Key nouns of the stage: words of >= 5 letters of its title (else of its goal), without filler words, most frequent first."""
    for text in (st.title, st.goal):
        words = [w for w in re.findall(r"[a-zA-Z]{5,}", text.lower()) if w not in _STOP_WORDS]
        if words:
            return [w for w, _n in sorted(((w, words.count(w)) for w in dict.fromkeys(words)), key=lambda x: -x[1])][:6]
    return []


def _is_test_path(rel: str) -> bool:
    parts = rel.lower().split("/")
    name = parts[-1]
    return any(x in ("tests", "test", "__tests__") for x in parts[:-1]) or name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".spec.js"))


def _size_groups(paths: list[str], lines: dict[str, int]) -> list[list[str]]:
    groups: list[list[str]] = []
    total = 0
    for p in paths:
        if groups and total + lines[p] <= RESPLIT_CHUNK_LINES:
            groups[-1].append(p)
            total += lines[p]
        else:
            groups.append([p])
            total = lines[p]
    return groups


def _fallback_substages(st: Stage, repo: list[tuple[str, int, int]], reads: dict[str, int], root: Any = None,
                        hot: Any = ()) -> list[tuple[str, str, list[str]]] | None:
    """No usable model answer: (1) the files the goal names, else the files this stage read most, grouped into parts of <= ~600 lines
    (files that nothing changed yet first); (2) else the files that contain a key noun of the goal (each its own part, `hot` = the ledger's
    files first); (3) else all code / text files (tests, scratch and caches left out) grouped by size, `hot` and recently changed first.
    None when that gives fewer than 2 parts."""
    lines = {rel: n for rel, n, _c in repo}
    goal = st.goal
    named = [rel for rel in lines if rel in goal or re.search(r"(?<![\w./-])" + re.escape(rel.rsplit("/", 1)[-1]) + r"(?![\w])", goal)]
    if not named:
        named = [p for p, _n in sorted(reads.items(), key=lambda kv: (-kv[1], kv[0])) if p in lines][:8]
    named.sort(key=lambda p: p in st.files)                            # stable: untouched files first
    groups = _size_groups(named, lines)
    if len(groups) < RESPLIT_RANGE[0]:
        from pathlib import Path

        hot_set = {*hot, *st.files}
        base = Path(root) if root is not None else None

        def mtime(rel: str) -> float:
            try:
                return (base / rel).stat().st_mtime if base else 0.0
            except OSError:
                return 0.0

        pool = sorted((r for r in lines if not _is_test_path(r) and not r.endswith(".json")), key=lambda r: (r not in hot_set, -mtime(r), r))
        keys, hits = _goal_keywords(st), {}
        if base is not None and keys:
            for rel in pool:
                try:
                    low = (base / rel).read_text(encoding="utf-8", errors="replace").lower()
                except OSError:
                    continue
                n = sum(low.count(k) for k in keys)
                if n:
                    hits[rel] = n
        kw = sorted(hits, key=lambda r: (r not in hot_set, -hits[r], r))
        if len(kw) >= RESPLIT_RANGE[0]:
            groups = [[p] for p in kw[:RESPLIT_RANGE[1]]]
            groups[-1] += kw[RESPLIT_RANGE[1]:] if len(kw) > RESPLIT_RANGE[1] else []
        else:
            groups = _size_groups(pool, lines)
    if len(groups) > RESPLIT_RANGE[1]:
        groups[RESPLIT_RANGE[1] - 1:] = [[p for g in groups[RESPLIT_RANGE[1] - 1:] for p in g]]
    if len(groups) < RESPLIT_RANGE[0]:
        return None
    body = goal.split("\n\n", 1)[1] if goal.startswith("Overall task:") and "\n\n" in goal else goal
    n = len(groups)
    return [(_one_line(f"Part {i}/{n}: " + ", ".join(g), MODEL_MAX_TITLE),
             f"Work ONLY on {', '.join(g)} (part {i} of {n} of the stage \"{st.title}\"; the other parts cover the other files). "
             f"Original stage goal: {_one_line(body, 600)}", g[:RESPLIT_MAX_FILES]) for i, g in enumerate(groups, 1)]


def announce(console: Any, plan: StagePlan) -> None:
    """One dim line per stage title when the plan has two or more stages (nothing for one)."""
    if len(plan.stages) < 2:
        return
    for st in plan.stages:
        console.print(f"[dim]plan: stage {st.id}/{len(plan.stages)}  {escape(st.title)}[/dim]")


# ── running ─────────────────────────────────────────────────────────────────

def _ledger_marks(engine: Any) -> dict[str, tuple]:
    return {p: (e.action, e.step, e.size) for p, e in engine.ctx.ledger.files.items()}


def _stamp(engine: Any) -> str:
    return checks_mod.content_stamp(engine.root, engine.ctx.ledger.files)


@dataclass
class Segment:
    """How one engine segment of a stage ended. kind: green (answer accepted, all green) | red (ended, not green: `why`) | cap (context cap)
    | interrupted (the user stopped it). `res` is the engine's TurnResult (res.stopped: budget / loop / error / announce ...)."""
    res: Any
    kind: str
    why: str = ""


class _Runner:
    def __init__(self, engine: Any, plan: StagePlan, save: Callable[[], None] | None) -> None:
        cfg = _coding(engine.config)
        self.engine, self.plan, self.save = engine, plan, save
        self.last_state: Any = None                           # the VerifiedState the latest segment started from
        self.cap = int(cfg.get("stage_prompt_cap", DEFAULT_CAP) or 0) or None
        try:
            self.cap = int(os.environ["AGENT_CODE_STAGE_PROMPT_CAP"]) or None       # one run's override (scripts/speed_trial.py); junk is ignored
        except (KeyError, ValueError):
            pass
        self.limit = max(1, int(cfg.get("stage_attempts", DEFAULT_ATTEMPTS) or DEFAULT_ATTEMPTS))
        self.resplit_on = str(cfg.get("resplit", "auto")).strip().lower() != "off"
        try:
            self.resplit_after = max(1, int(cfg.get("resplit_after_caps", RESPLIT_AFTER_CAPS)))
        except (TypeError, ValueError):
            self.resplit_after = RESPLIT_AFTER_CAPS
        try:
            self.range_lines = max(0, int(cfg.get("range_split_lines", RANGE_SPLIT_LINES)))
        except (TypeError, ValueError):
            self.range_lines = RANGE_SPLIT_LINES
        self.only: set[str] = set()                            # the files a re-split may use: the ones the stage's goal names (empty = the whole repo)
        self.broad_on = str(cfg.get("broad_split", True)).strip().lower() not in ("false", "off", "0", "no")
        self.tests_on = str(cfg.get("stage_tests", True)).strip().lower() not in ("false", "off", "0", "no")
        self.strict_last = str(cfg.get("stage_tests_strict_last", True)).strip().lower() not in ("false", "off", "0", "no")
        self._last_failing: list[str] | None = None            # failing ids of the latest stage-end run that produced a result
        self._preexisting: list[str] = []                      # of those: the ones that were already red before the stage
        self._broad_tried: set[int] = set()                    # id() of the stage objects whose broad split was already attempted
        self.why = ""                                          # why the last model split was unusable (for stage_resplit_failed)
        self.reads: dict[int, dict[str, int]] = {}            # per stage id: read_file calls per path while the stage ran (not saved)

    def persist(self) -> None:
        if self.save is not None:
            self.save()

    def emit(self, kind: str, st: Stage, **extra: Any) -> None:
        self.engine.events.emit(kind, id=st.id, title=st.title, seconds=round(st.seconds, 1), prompt_peak=st.prompt_peak, **extra)

    def notes_for(self, st: Stage) -> str:
        cursor_mod.sync_range(st, self.engine.root)
        lines = [f"- stage {s.id} \"{s.title}\" [{s.status}]: {s.summary or s.reason}" for s in self.plan.stages
                 if s.id != st.id and s.status in ("done", "blocked")]
        later = [f"{s.id}. {s.title}" for s in self.plan.stages if s.status in ("pending", "active") and s.id > st.id]
        if later:
            lines.append("- later stages, NOT yours now: " + "; ".join(later))
        lines += self.large_files(st)
        if st.lines and st.scope:
            f, (a, b) = st.scope[0], st.lines
            lines.append(f"- This part covers {f} lines {a}-{b} only (the file has {st.lines_total} lines; the other parts handle the rest): read {f} lines {a}-{b} only "
                         "(start/end) and do not read other files. "
                         f"Start by reading lines {a}-{b} of {f} (read_file path start={a} end={b}) and rewrite them immediately in place with replace_lines/edit_file; "
                         "do not search the rest of the repository first; tests are run once at the end.")
        if cursor_mod.is_chunk_stage(st) and not any(ln.startswith("WORKING METHOD") for ln in lines):
            lines.append(cursor_mod.METHOD)
        if cursor_mod.targets(st):
            lines.append(cursor_mod.note(cursor_mod.cursor(st, self.engine.root)))
        lines += fileseen.block(st.seen, st.edited)
        return "\n".join(lines)

    def large_files(self, st: Stage) -> list[str]:
        """The LARGE FILES hint of a sub-stage: files of its scope above 12000 characters, with their line counts, to be read by range only."""
        from pathlib import Path

        root, rows = Path(self.engine.root), []
        for rel in st.scope:
            try:
                text = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > LARGE_FILE_CHARS:
                rows.append(f"{rel} ({text.count(chr(10)) + 1} lines)")
        if not rows:
            return []
        return ["- LARGE FILES in this stage: " + ", ".join(rows) + ". Do NOT read them whole: use read_file with a line range (path start-end), "
                "show_symbol or outline, and only the part this stage covers."]

    def measure(self) -> verified.VerifiedState:
        try:
            return verified.build(self.engine)
        except Exception as exc:  # noqa: BLE001 - the stage must start even when the measurement fails
            return verified.VerifiedState(f"{verified.HEADER}\n(could not be measured: {type(exc).__name__})", {}, 0.0)

    def activate(self, st: Stage) -> bool:
        """Mark `st` as the active stage. True when it is new (its first segment makes a fresh acceptance list)."""
        fresh_plan = st.status == "pending"                   # a stage that was started before keeps its acceptance list
        st.status = "active"
        st.started_at = st.started_at or time.time()
        self.plan.current = self.plan.stages.index(st)
        self.emit("stage_start", st, attempts=st.attempts)
        self.engine.console.print(f"[bold]stage {st.id}/{len(self.plan.stages)}[/bold] {st.title}")
        return fresh_plan

    def segment(self, st: Stage, note: str = "", *, fresh_plan: bool = False) -> Segment:
        """ONE engine segment of `st`: a fresh context (VERIFIED STATE measured now), one `engine.turn`. Bookkeeping only (seconds, peak,
        files, attempts); the decision what to do next is the caller's (`run` below, or the supervisor)."""
        eng = self.engine
        cursor_mod.sync_range(st, eng.root)                   # a ranged part follows the lines earlier parts added or removed
        if self.broad_applies(st):
            self._broad_tried.add(id(st))
            if self.resplit(st, "broad"):
                return Segment(types.SimpleNamespace(stopped="", text=None), "cap", "resplit")      # no model step ran: the sub-stages start next
        self.ensure_baseline()
        self._last_failing, self._preexisting = None, []
        t0 = time.monotonic()
        eng.stage_obj, eng.stage_edited, eng.stage_seen = st, st.edited, st.seen     # the cursor (measure below shows it) and the live records of writes / reads
        eng.stage_chunk = cursor_mod.is_chunk_stage(st)
        self.last_state = state = self.measure()
        eng.begin_stage(st.goal, state.text, self.notes_for(st), reset_plan=fresh_plan)
        eng.stage_cap, eng.prompt_peak = self.cap, 0
        eng.events.emit("segment_start", id=st.id, title=st.title)
        marks = _ledger_marks(eng)
        eng.stage_scope, eng.stage_changed0, eng.stage_marks, eng.stage_waived = list(st.scope), set(st.files), marks, set(st.waived)
        reads0 = dict(getattr(getattr(eng, "tools", None), "read_counts", None) or {})
        step0, writes0 = getattr(eng.ctx, "step", 0), getattr(eng, "real_writes", 0)
        self.persist()
        try:
            res = eng.turn(f"{note}\n\n{st.goal}" if note else st.goal)
        finally:
            eng.stage_cap = None
            eng.stage_obj, eng.stage_edited, eng.stage_seen, eng.stage_chunk = None, None, None, False
            st.waived = sorted(eng.stage_waived & set(st.scope))
            eng.stage_scope = []
        took = time.monotonic() - t0
        st.seconds += took
        st.spent_s += took                                    # the budget: this segment's own duration, never the gap between processes
        st.spent_steps += max(0, getattr(eng.ctx, "step", 0) - step0)
        edits = max(0, getattr(eng, "real_writes", 0) - writes0)
        st.real_edits += edits
        st.no_edit_segments = 0 if edits else st.no_edit_segments + 1
        st.prompt_peak = max(st.prompt_peak, eng.prompt_peak)
        after = _ledger_marks(eng)
        st.files = sorted({*st.files, *(p for p, m in after.items() if marks.get(p) != m)})
        seen = self.reads.setdefault(st.id, {})
        for p, n in (getattr(getattr(eng, "tools", None), "read_counts", None) or {}).items():
            if n > reads0.get(p, 0):
                seen[p] = seen.get(p, 0) + n - reads0.get(p, 0)
        if res.stopped == "interrupted":
            self.persist()
            return Segment(res, "interrupted")
        if res.stopped == "stage_cap":
            st.cap_segments += 1
            self.emit("stage_cap", st)
            if self.should_resplit(st) and self.resplit(st):
                return Segment(res, "cap", "resplit")              # the plan changed: the caller picks the first sub-stage
            return Segment(res, "cap")
        if res.stopped == "fresh_context":                    # ended on purpose by the supervisor's measured rule: no attempt is used up
            return Segment(res, "cap", "fresh_context")
        st.attempts += 1
        why = self.not_green(res) or self.stage_tests(st)
        return Segment(res, "red" if why else "green", why)

    def run(self, st: Stage) -> str:
        """Run one stage to done / blocked. Returns its status, or 'interrupted' when the user stopped the run."""
        eng = self.engine
        fresh_plan = self.activate(st)
        idle_caps, note = 0, ""
        while True:
            stamp = _stamp(eng)
            seg = self.segment(st, note, fresh_plan=fresh_plan)
            fresh_plan = False
            if seg.kind == "interrupted":
                return "interrupted"
            if seg.why == "resplit":
                return "resplit"
            if seg.kind == "cap":
                idle_caps = idle_caps + 1 if _stamp(eng) == stamp else 0
                if idle_caps >= 2:
                    return self.block(st, "no progress: two segments in a row ended at the context cap without changing a file")
                note = CAP_NOTE
                self.persist()
                continue
            if not seg.why:
                return self.finish(st, seg.res)
            if st.attempts >= self.limit:
                return self.block(st, f"{st.attempts} attempts, last: {seg.why}")
            note = f"continuing: the previous segment ended without a green result ({seg.why}); see VERIFIED STATE"
            self.persist()

    def broad_applies(self, st: Stage) -> bool:
        """A stage whose goal says all / every / each ... (BROAD words) in a repo with >= 3 code / text files and >= 1200 lines never fits one
        context (measured: a 'rewrite all texts' stage hit the cap 8 times): it is split before its first segment, once, never a sub-stage."""
        if not (self.broad_on and self.resplit_on and st.depth == 0 and id(st) not in self._broad_tried and st in self.plan.stages):
            return False
        if st.attempts or st.cap_segments or st.files:
            return False                                       # it already ran: the cap-based re-split decides
        body = st.goal.split("\n\n", 1)[1] if st.goal.startswith("Overall task:") and "\n\n" in st.goal else st.goal
        text = f"{st.title}\n{body}"
        if not _BROAD.search(text) or not (cursor_mod.INTENT.search(text) or _REMOVES.search(text)) or _CREATES.search(body[:160]):
            return False                                       # only editing-by-passes goals; 'Add a level system ... each ending' is a feature, not a sweep
        try:
            repo = _repo_files(self.engine.root)
        except Exception:  # noqa: BLE001 - an optimisation: never break the stage
            return False
        named = named_files(st, repo)
        if named:                                              # a goal that names its files is no sweep of the repo: it is split only inside them
            return any(len(_file_rows(self.engine.root, rel)) > self.range_lines > 0 for rel in named)
        return len(repo) >= BROAD_MIN_FILES and sum(n for _p, n, _c in repo) >= BROAD_MIN_LINES

    def _can_test(self) -> bool:
        return self.tests_on and bool(getattr(getattr(self.engine, "profile", None), "test_cmd", None))

    def ensure_baseline(self) -> None:
        """Measure the failing test ids ONCE per run (at the start of the first stage, or after a resume of a plan that has none): the
        stage-end rule is 'a stage must not make things worse', so tests that were already red are not this stage's job. A run without a
        result (timeout ...) gives baseline_ok False: see `stage_tests`. coding.stage_tests false / no test command = no run at all."""
        plan = self.plan
        if plan.baseline_red is not None or not self._can_test():
            return
        try:
            _lines, data = verified.run_tests(self.engine)
        except Exception:  # noqa: BLE001 - the stage must start even when the measurement fails
            data = {}
        status = data.get("status")
        plan.baseline_ok = status in ("pass", "fail")
        plan.baseline_red = list(data.get("failing") or []) if status == "fail" else []
        self.engine.events.emit("stage_baseline", ok=plan.baseline_ok, red=len(plan.baseline_red))
        if plan.baseline_red:
            self.engine.console.print(f"[dim]{len(plan.baseline_red)} tests are already red before the first stage; they are not held against a stage[/dim]")
        elif not plan.baseline_ok:
            self.engine.console.print("[dim]test baseline not measurable (no result); only tests named after a stage file can block[/dim]")
        self.persist()

    def stage_tests(self, st: Stage) -> str:
        """'' when the stage may be accepted, else why not. Rule: a stage must not make things worse. red = failing now - plan.baseline_red
        (ids only; a known-red test that fails with another assertion is not new). Exceptions: the LAST stage of the plan is strict (every
        test must be green, nothing comes after it; coding.stage_tests_strict_last), and when the baseline run had no result only failing ids
        that contain the basename of a scope file count. No test command, coding.stage_tests false, a timeout or an unreadable run = nothing
        measured = not red. No model call."""
        eng, plan = self.engine, self.plan
        if not self._can_test():
            return ""
        try:
            lines, data = verified.run_tests(eng)
        except Exception:  # noqa: BLE001 - the stage must not fail on a broken measurement
            return ""
        self.emit("stage_tests", st, status=data.get("status"), failed=len(data.get("failing") or []))
        if data.get("status") not in ("pass", "fail"):
            return ""
        failing = list(data.get("failing") or [])
        self._last_failing = failing
        strict = self.strict_last and bool(plan.stages) and st is plan.stages[-1]
        known = set(plan.baseline_red or [])
        if strict:
            new = failing
        elif plan.baseline_ok and plan.baseline_red is not None:
            new = [t for t in failing if t not in known]
        else:                                                  # no usable baseline: only tests that name a file of this stage
            names = {os.path.basename(f) for f in st.scope}
            new = [t for t in failing if any(n in t for n in names)] if names else []
        self._preexisting = [] if strict else [t for t in failing if t not in new]
        if not new and not (strict and data.get("status") == "fail"):
            if self._preexisting:
                self.engine.console.print(f"[dim]{len(self._preexisting)} tests were already red before this stage; not held against it[/dim]")
            return ""
        shown = {}
        for ln in lines[1:]:
            t = ln.strip()
            for tid in new[:TEST_IDS_SHOWN]:
                if t.startswith(tid) or t.startswith(tid[:120]):
                    shown[tid] = t
        rows = [shown.get(t, t) for t in new[:TEST_IDS_SHOWN]]
        head = lines[0].removeprefix("TESTS: ")
        if strict:
            msg = f"tests red at the end of the last stage, all must be green ({head})"
        else:
            msg = f"{len(new)} tests are newly red ({head}); these passed before this stage"
        msg += (": " + "; ".join(rows) if rows else "")
        if self._preexisting:
            msg += f". {len(self._preexisting)} other tests are red from earlier work; they are not part of this stage"
        return _one_line(msg, 900)

    def should_resplit(self, st: Stage) -> bool:
        return self.resplit_on and st.depth == 0 and st.cap_segments >= self.resplit_after and st in self.plan.stages

    def resplit(self, st: Stage, source: str | None = None) -> bool:
        """Replace `st` in the plan by 2..6 smaller sub-stages (one main-model call, else a deterministic file grouping). True when done;
        False leaves the plan untouched. Never raises and is never silent: a decline or failure is logged (`stage_resplit_failed`) and
        printed, and the stage goes on as before."""
        try:
            repo = _repo_files(self.engine.root)
            self.why, self.only = "", set()
            named = [] if _CREATES.search(_body(st)[:160]) else named_files(st, repo)        # a feature may need new files: no scope from names
            if named:
                ranged = _range_substages(st, named, self.engine.root, self.range_lines)
                if ranged:
                    self._apply_resplit(st, ranged[0], source or "ranges", ranged[1])
                    return True
                repo, self.only = [r for r in repo if r[0] in named], set(named)
            subs, how = self._ask_substages(st, repo), "model"
            model_why = self.why
            if not subs and _CREATES.search(st.goal.split("\n\n", 1)[-1][:160]):      # grouping by FILES is wrong for a feature ('Add a level system')
                return self._decline(st, f"model: {model_why or 'no answer'}; no file grouping for a feature goal")
            if not subs:
                ledger = getattr(getattr(getattr(self.engine, "ctx", None), "ledger", None), "files", None) or {}
                subs, how = _fallback_substages(st, repo, self.reads.get(st.id, {}), self.engine.root, list(ledger)), "fallback"
            if not subs:
                return self._decline(st, f"model: {model_why or 'no answer'}; fallback: fewer than 2 file groups ({len(repo)} repo files)")
            self._apply_resplit(st, subs, source or how)
            return True
        except Exception as exc:  # noqa: BLE001 - the re-split is an optimisation: it must never end or break the run
            return self._decline(st, f"{type(exc).__name__}: {exc}")

    def _decline(self, st: Stage, reason: str) -> bool:
        try:
            self.engine.events.emit("stage_resplit_failed", stage=st.id, reason=reason[:300])
            self.engine.console.print(f"[dim]re-split of stage {st.id} not possible: {escape(reason[:200])}[/dim]")
        except Exception:  # noqa: BLE001 - reporting must not break the run either
            pass
        return False

    def _ask_substages(self, st: Stage, repo: list[tuple[str, int, int]]) -> list[tuple[str, str, list[str]]] | None:
        eng, state = self.engine, self.last_state
        sec = getattr(state, "sections", None) or {}
        task = self.plan.task or st.goal
        listed = sorted(repo, key=lambda r: (-r[1], r[0]))[:RESPLIT_LISTED_FILES]
        parts = [f"STAGE GOAL:\n{_one_line(st.goal, 1500)}", f"OVERALL TASK:\n{' '.join(task.split())[:600]}",
                 ("THE TASK NAMES THESE FILES; every sub-stage may only use them: " + ", ".join(sorted(self.only))) if self.only else "",
                 "REPO FILES (path, lines):\n" + ("\n".join(f"{p} ({n})" for p, n, _c in listed) or "(none)"),
                 "CHANGED SO FAR IN THIS STAGE:\n" + (", ".join(st.files) or "(nothing)")]
        parts = [p for p in parts if p]
        for name in ("OPEN ITEMS", "DEAD ENDS"):
            text = str((sec.get(name) or {}).get("text") or "").strip()
            if text:
                parts.append(f"{name}:\n{_one_line(text, 500)}")
        llm = getattr(eng, "llm", None)
        if llm is None:
            self.why = "no model"
            return None
        reply = _ask(llm, eng.config or {}, _RESPLIT_SYSTEM, "\n\n".join(parts), getattr(eng, "model", None), RESPLIT_NUM_PREDICT)
        if reply is None:
            self.why = "the call failed"
            return None
        try:
            out = _valid_substages(json.loads(reply[reply.find("{"):reply.rfind("}") + 1]))
        except Exception:  # noqa: BLE001 - unreadable JSON: the deterministic fallback takes over
            out = None
            self.why = "unreadable JSON: " + _one_line(reply, 80)
        if out is not None and self.only and any(f not in self.only for _t, _g, files in out for f in files):
            out, self.why = None, "answer names files outside the files the task names"
        if out is None and not self.why:
            self.why = "answer rejected (needs 2-6 stages with title, goal 20-1200 chars, files list): " + _one_line(reply, 80)
        return out

    def _apply_resplit(self, st: Stage, subs: list[tuple[str, str, list[str]]], source: str, ranges: list[tuple[int, int, int] | None] | None = None) -> None:
        plan = self.plan
        i = plan.stages.index(st)
        head = f"Overall task: {_one_line(plan.task or st.goal, 300)}"
        ranges = ranges or [None] * len(subs)
        new = [_sub_stage(head, title, goal, files, r, st.id) for (title, goal, files), r in zip(subs, ranges)]
        plan.stages[i:i + 1] = new
        for n, s2 in enumerate(plan.stages, 1):
            s2.id = n
        plan.current = i
        self.reads.clear()
        self.engine.events.emit("stage_resplit", **{"from": st.id, "to": [s2.id for s2 in new], "source": source})
        self.engine.console.print(f"[dim]re-split stage {st.id} into {len(new)} parts:[/dim]")
        for s2 in new:
            self.engine.console.print(f"[dim]  {s2.id}. {escape(s2.title)}[/dim]")
        self.persist()

    def not_green(self, res: Any) -> str:
        """'' when the engine's final gate accepted the answer with every check and test green, else why not. `verified` False alone can
        also mean "nothing was run since the last edit" (unproven, e.g. no test command): that is red only with a measured failure."""
        if res.stopped:
            return f"stopped: {res.stopped}" + (f" ({'; '.join(res.not_verified)})" if res.stopped == "gate" and res.not_verified
                                                else f" ({res.reason})" if getattr(res, "reason", "") else "")
        if res.text is None:
            return "no final answer"
        if res.not_verified:
            return "not verified: " + "; ".join(res.not_verified)
        t = self.engine.tools
        if res.verified is False:
            if t.syntax_bad or t.static_bad:
                return "syntax or static errors: " + ", ".join(sorted({*t.syntax_bad, *t.static_bad})[:4])
            failed = [c for _s, c, code in t.evidence if code not in (0, None)]
            if failed:
                return f"command failed after the last edit: {_one_line(failed[-1], 60)}"
        return ""

    def summary(self, st: Stage, text: str) -> str:
        eng = self.engine
        parts = ["changed " + ", ".join(st.files[:8]) + (f" +{len(st.files) - 8}" if len(st.files) > 8 else "") if st.files else "no file changed"]
        runs = eng.ctx.ledger.runs
        if runs:
            _step, cmd, code, _tail = runs[-1]
            parts.append(f"last run `{_one_line(cmd, 50)}` exit {code if code is not None else '?'}")
        every = eng.checks.all()
        if every:
            parts.append(f"checks {sum(c.status() == 'pass' for c in every)}/{len(every)} pass")
        if self._preexisting:
            parts.append(f"{len(self._preexisting)} tests were already red before this stage: " + ", ".join(self._preexisting[:3]))
        parts.append("said: " + _one_line(re.sub(r"^NOT VERIFIED:.*?\n\n", "", text or "", flags=re.S), 300))
        return _one_line("; ".join(parts), SUMMARY_MAX)

    def finish(self, st: Stage, res: Any) -> str:
        st.status, st.finished_at, st.reason = "done", time.time(), ""
        if self._last_failing is not None:                    # accepted: fixed tests leave the baseline, later regressions are caught
            self.plan.baseline_red, self.plan.baseline_ok = sorted(self._last_failing), True
        st.summary = self.summary(st, res.text or "")
        self.emit("stage_done", st)
        self.persist()
        return "done"

    def close(self, st: Stage, note: str) -> str:
        """The harness ends a stage the model left open (supervisor rule): done, with `note` as its summary."""
        st.status, st.finished_at, st.reason = "done", time.time(), ""
        if self._last_failing is not None:
            self.plan.baseline_red, self.plan.baseline_ok = sorted(self._last_failing), True
        st.summary = _one_line(note, SUMMARY_MAX)
        self.emit("stage_done", st, closed_by="harness")
        self.engine.console.print(f"[dim]stage {st.id} {escape(note)}[/dim]")
        self.persist()
        return "done"

    def block(self, st: Stage, reason: str) -> str:
        st.status, st.finished_at, st.reason = "blocked", time.time(), reason
        st.summary = self.summary(st, "")[:SUMMARY_MAX - 1]
        self.emit("stage_blocked", st, reason=reason)
        self.engine.console.print(f"[yellow]stage {st.id} blocked: {reason}[/yellow]")
        self.persist()
        return "blocked"


def make_runner(engine: Any, plan: StagePlan, save: Callable[[], None] | None = None) -> _Runner:
    """Bind `plan` to the engine (pin the brief card) and return the runner whose `segment` / `finish` / `block` the supervisor drives."""
    engine.stage_plan = plan
    if plan.task:
        card = brief_mod.make(plan.task)
        if card and card not in engine.briefs:
            engine._pin_brief(card)
    return _Runner(engine, plan, save)


def run_stages(engine: Any, plan: StagePlan, *, save: Callable[[], None] | None = None, max_stages: int | None = None) -> StageReport:
    """Run every pending / active stage of `plan` in order (at most `max_stages` of them in this call). `save` persists the session; it
    is called at every segment and stage boundary, so a killed run resumes at the active stage."""
    runner, ran = make_runner(engine, plan, save), 0
    try:
        while True:
            st = next((x for x in plan.stages if x.status in ("pending", "active")), None)
            if st is None or (max_stages is not None and ran >= max_stages):
                break
            ran += 1
            out = runner.run(st)
            if out == "interrupted":
                break
            if out == "resplit":
                ran -= 1                                          # the sub-stages replace the stage: it does not count as one run
    finally:
        engine.stage_cap, engine.stage_goal = None, None
        runner.persist()
    states = [s.status for s in plan.stages]
    verdict = "done" if all(s == "done" for s in states) else ("partial" if "done" in states else "failed")
    return StageReport(verdict, plan.stages)
