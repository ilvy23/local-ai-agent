"""Autopilot: `agent code --autopilot "task"` does not stop merely because the work is long.

A SUPERVISOR runs engine segments through the stage runner (stages.py) and answers every non-final stop with a GRADED recovery, until all
stages are done, or N unproductive segments in a row, or a hard budget. Everything is decided from measurements (the ledger on disk, the
checks, the tests, the clock) - never from what the model wrote.

    Supervisor(engine).run(task) -> SupervisorResult   verdict done | needs_human | partial, exit code 0 | 2 | 3 (130: stopped by the user)

Per stage, u = consecutive unproductive segments (progress = a file of the ledger changed on disk OR more checks / acceptance items /
test runs are green than when the segment began):
    u=0 resume         a fresh context for the same stage with the VERIFIED STATE (what the stage runner does anyway)
    u=1 resume_pause   the same, and reading is paused (LookAct.pause) so the model must act; loop signatures are cleared
    u=2 fresh_stage    the stage's acceptance list is reset; it starts from its goal with a note that two segments made no progress + what fails
    u>=3 needs_human   the stage is BLOCKED (reason recorded); the run goes on with the next stage (stages are independent unless the goal
                       says "after stage N" and stage N is blocked)
A productive segment resets u. Global stops: coding.autopilot_max_idle (default 4) unproductive segments in a row (blocking a stage counts
as a change and starts the count again), autopilot_max_hours (0 = none), autopilot_max_segments (200), a model server that does not come
back within autopilot_server_wait_s (1800). A Python exception in a segment is an 'error' stop (traceback in the event log), Ctrl-C saves
and ends with exit 130. While it runs, `systemd-inhibit` keeps the PC from suspending (coding.autopilot_inhibit, default true).
HARD BUDGET per stage (stages.budget_blocked; coding.stage_max_minutes / stage_max_steps / stage_min_edits_per_100_steps): at a segment boundary a stage
that used up its time or step budget without editing at a healthy rate is BLOCKED ('budget: ...', event stage_budget) - not dead: when every other
stage is done or blocked, ONE retry round gives each such stage half a budget and a sharp note; a second budget block is final. A stage whose
scope is satisfied, with no new test failures and no edit in its last 2 segments, is closed by the harness once it used 60 % of its budget.
The report (.scratch/autopilot-report.md) is built from measurements only. coding.autopilot 'off' (default) | 'auto': should_autopilot()."""

from __future__ import annotations

from agent import brand, osport

import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.markup import escape

from agent import llm as llm_mod
from agent.codeagent import edits, intake as intake_mod, serverwait, stages as stages_mod, timing, verified
from agent.codeagent.lookact import is_scratch_path

logger = logging.getLogger(__name__)

MAX_IDLE, MAX_SEGMENTS, SERVER_WAIT_S = 4, 200, 1800
AUTO_MIN_BULLETS, AUTO_MIN_CHARS = 3, 600
SUSPEND_FACTOR, SUSPEND_MIN_S = 10, 30            # wall delta > 10x the monotonic delta (and > 30 s): the PC slept during the segment
GRADES = ("resume", "resume_pause", "fresh_stage")
PREFILL_END_SHARE, PREFILL_WINDOW_S, PREFILL_MIN_STEPS = 0.5, 600, 20    # measured rule: prefill > 50 % of the last 10 min of the segment -> a fresh context
PREFILL_MIN_WALL_S = 180.0                # a couple of cache-miss spikes after an edit are not a trend
FRESH_NOTE = "continuing in a fresh context: the previous one had grown slow to read; see VERIFIED STATE"
REPORT_NAME = "autopilot-report.md"
_DEPENDS = re.compile(r"\bafter\s+(?:stage|stufe|phase|part|teil)\s+(\d+)\b", re.I)
_PRUNE = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_TEST_RUN = re.compile(r"pytest|unittest|\btest\b|nose|tox|cargo test|go test|npm (?:run )?test", re.I)
_BAD_TESTS = ("fail", "timeout", "stopped")
_QUIET_CHECKS = ("pass", "skipped", "unreviewed", "not run", "none", "")


def _coding(cfg: Any) -> dict[str, Any]:
    return (cfg or {}).get("coding") or {}


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def should_autopilot(task: Any, cfg: Any, root: Any = None) -> bool:
    """coding.autopilot == 'auto' and the one-shot task is big: >= 3 top-level list items (outside code fences) or >= 600 chars, or - with a `root` -
    a short task that stages.split() would turn into a 'ranges' plan (it names 1-3 files, one above coding.range_split_lines, and edits them by
    passes: stages.range_job, names and sizes only, no model call)."""
    if str(_coding(cfg).get("autopilot", "auto")).strip().lower() != "auto" or not isinstance(task, str) or not task.strip():
        return False
    if len(task.strip()) >= AUTO_MIN_CHARS:
        return True
    if root is not None and stages_mod.range_job(task.strip(), root, cfg) is not None:
        return True
    lines = task.splitlines()
    free = stages_mod._fence_free(lines)
    return sum(1 for ln, ok in zip(lines, free) if ok and stages_mod._ITEM.match(ln)) >= AUTO_MIN_BULLETS


@dataclass
class SupervisorResult:
    verdict: str                                  # done | needs_human | partial
    exit_code: int                                # 0 done, 2 needs a human, 3 partial, 130 stopped by the user
    report: str
    decisions: list[dict[str, Any]] = field(default_factory=list)       # {at, stage, reason, action, progress, u}
    stages: stages_mod.StageReport | None = None
    reason: str = ""                              # why the run ended when it was not "all done"


def measure_progress(engine: Any) -> tuple[str, int]:
    """(hash of the CONTENT of every project file the ledger recorded as written, number of green things). A rewrite with the same bytes is
    no progress; green = passing checks + confirmed acceptance items + successful test commands in the ledger."""
    h = hashlib.sha1()
    root = Path(engine.root)
    for p in sorted(engine.ctx.ledger.files):
        if is_scratch_path(p):
            continue
        f = Path(p) if Path(p).is_absolute() else root / p
        try:
            h.update(f"{p}\0".encode() + hashlib.sha1(f.read_bytes()).digest())
        except OSError:
            h.update(f"{p}\0-\n".encode())
    green = 0
    try:
        green += sum(c.status() == "pass" for c in engine.checks.all())
    except Exception:  # noqa: BLE001 - a broken checks folder is no progress, not a crash
        pass
    plan = getattr(getattr(engine, "plan", None), "plan", None)
    if plan is not None:
        green += plan.confirmed()
    green += sum(1 for _s, cmd, code, _t in engine.ctx.ledger.runs if code == 0 and _TEST_RUN.search(cmd))
    return h.hexdigest(), green


def red_items(state: Any) -> list[str]:
    """What the measured state says is NOT green: failing / timed-out tests, checks with a verdict other than pass."""
    out: list[str] = []
    sec = getattr(state, "sections", None) or {}
    t = sec.get("TESTS") or {}
    if t.get("status") in _BAD_TESTS or t.get("failed") or t.get("errors"):
        out.append(" ".join(str(t.get("text") or f"tests {t.get('status')}").split())[:160].removeprefix("TESTS: ") or "tests red")
    for cid, status in ((sec.get("CHECKS") or {}).get("results") or {}).items():
        if status not in _QUIET_CHECKS:
            out.append(f"check {cid}: {status}")
    smoke = sec.get("TUI SMOKE") or {}
    if smoke.get("status") == "fail":
        out.append(" ".join(str(smoke.get("text") or "TUI SMOKE: FAIL").splitlines()[0].split()).replace("TUI SMOKE:", "tui smoke:", 1)[:200])
    return out


def _one_line(text: Any, limit: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


class Supervisor:
    """`session` is the CodeEngine. Everything that touches the outside world can be injected (clock, sleep, probe, popen, ...)."""

    def __init__(self, session: Any, *, cfg: dict[str, Any] | None = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep, events: Any = None,
                 save: Callable[[], None] | None = None, runner: Any = None, progress: Callable[[], tuple[str, int]] | None = None,
                 verify: Callable[[], Any] | None = None, probe: Callable[[str], bool] | None = None,
                 which: Callable[[str], str | None] | None = None, popen: Callable[..., Any] | None = None,
                 killpg: Callable[[int, int], None] | None = None, breakdown_fn: Callable[[], timing.Breakdown] | None = None,
                 ask: Callable[[str], str] | None = None) -> None:
        self.session = session
        self.cfg = cfg if cfg is not None else getattr(session, "config", {})
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.events = events if events is not None else session.events
        self._save, self.runner = save, runner
        self.progress = progress or (lambda: measure_progress(session))
        self.verify = verify or (lambda: verified.build(session, run_tests=True))
        self.probe = probe
        self._which = which or shutil.which
        self._popen = popen or subprocess.Popen
        self._killpg = killpg or osport.kill_group
        self.decisions: list[dict[str, Any]] = []
        self._inhibitor: Any = None
        self._t0 = 0.0
        self._segments = 0
        self._final: tuple[int, Any] | None = None           # (segments when measured, state)
        self._breakdown_fn = breakdown_fn                    # injected in tests; default: timing.breakdown over the recent events of this segment
        self._seg_wall = 0.0
        self._ask = ask                                      # reads one line from the human (injected in tests; default: the console, only with a terminal)
        self._loop_stops: dict[int, int] = {}                # 'loop' stops per stage
        self._hints: dict[int, str] = {}                     # the user's hint per stage, for the restart note

    # ── small helpers ───────────────────────────────────────────────────
    def conf(self, key: str, default: float) -> float:
        return _num(_coding(self.cfg).get(key, default), default)

    def persist(self) -> None:
        try:
            if self._save is not None:
                self._save()
        except Exception:  # noqa: BLE001 - a failed save must not end the run; the next boundary saves again
            logger.exception("autopilot save failed")

    def emit(self, kind: str, **data: Any) -> None:
        self.events.emit(kind, **data)

    def slow_prefill(self) -> str | None:
        """The measured rule (coding.prefill_end_share, default 0.5, 0 = off): over the last 10 minutes of the current segment the model
        spent more than that share of the wall time reading its prompt -> a reason (text), else None. Needs >= 5 measured steps."""
        share = self.conf("prefill_end_share", PREFILL_END_SHARE)
        if share <= 0:
            return None
        if self._breakdown_fn is not None:
            b = self._breakdown_fn()
        else:
            since = max(self.wall() - PREFILL_WINDOW_S, self._seg_wall)
            b = timing.breakdown(getattr(self.events, "recent", []), since=since, session=getattr(self.events, "session", None))
        if b.steps < PREFILL_MIN_STEPS or b.wall_s < PREFILL_MIN_WALL_S or b.prefill_share <= share:
            return None
        return (f"prefill is {b.prefill_share * 100:.0f}% of the last {b.wall_s / 60:.0f} min ({b.steps} steps, mean prompt "
                f"{b.mean_prompt / 1000:.0f}k tokens)")

    def _end_check(self) -> str | None:
        try:
            return self.slow_prefill()
        except Exception:  # noqa: BLE001 - a broken measurement must never end or break a segment
            return None

    def say(self, text: str) -> None:
        try:
            self.session.console.print(text, markup=False, highlight=False)
        except Exception:  # noqa: BLE001
            pass

    def decide(self, st: Any, reason: str, action: str, progress: bool, u: int) -> None:
        d = {"at": round(self.clock() - self._t0, 1), "stage": st.id if st is not None else 0, "reason": reason, "action": action,
             "progress": progress, "u": u}
        self.decisions.append(d)
        self.emit("autopilot_decision", **{k: v for k, v in d.items() if k != "at"})
        self.say(f"[autopilot] stage {d['stage']}: {reason} -> {action}" + (" (progress)" if progress else " (no progress)"))

    # ── systemd-inhibit ─────────────────────────────────────────────────
    def _inhibit_start(self) -> None:
        if not _coding(self.cfg).get("autopilot_inhibit", True):
            return
        try:
            exe = self._which("systemd-inhibit")
            if not exe:
                return
            self._inhibitor = self._popen([exe, "--what=sleep:idle", "--who=agent", "--why=agent code autopilot",
                                 "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except Exception:  # noqa: BLE001 - no systemd, no permission: the run is just not protected against suspend
            self._inhibitor = None

    def _inhibit_stop(self) -> None:
        proc, self._inhibitor = self._inhibitor, None
        if proc is None:
            return
        try:
            try:
                self._killpg(proc.pid, signal.SIGTERM)            # its own process group; the `tail --pid` holder also ends by itself when we die (kill -9)
            except (OSError, TypeError, ValueError):
                proc.terminate()
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # ── server ──────────────────────────────────────────────────────────
    def server_ok(self) -> bool:
        llm = getattr(self.session, "llm", None)
        model = getattr(self.session, "model", "")
        url = llm_mod.server_url_of(llm, model)
        if url is None:
            return True                                           # a client without a health endpoint
        probe = self.probe or serverwait._http_probe
        if probe(url):
            return True
        llm_mod.revive_server(llm, model)                         # a dead Vulkan server is started again if there is a start command
        t0 = self.clock()
        limit = self.conf("autopilot_server_wait_s", SERVER_WAIT_S)
        self.say(f"[autopilot] model server not reachable - waiting up to {limit:.0f} s")
        up = serverwait.wait_for_server(url, max_wait_s=limit, sleep=self.sleep, clock=self.clock, probe=probe)
        self.emit("autopilot_server_wait", seconds=round(self.clock() - t0, 1), ok=up, url=url)
        return up

    # ── plan and stage selection ────────────────────────────────────────
    def _plan(self, task: str) -> stages_mod.StagePlan:
        s = self.session
        if s.stage_plan is not None and s.stage_plan.unfinished():
            return s.stage_plan                                   # --resume: the saved plan, done stages stay done
        plan = stages_mod.split(task, llm=getattr(s, "llm", None), cfg=self.cfg, model=getattr(s, "model", None),
                                cache=getattr(s, "split_cache", None), root=getattr(s, "root", None))
        stages_mod.announce(s.console, plan)
        s.stage_plan = plan
        s.ctx.user_requests.append(task)                          # the whole task is the session's first request (title, resume banner)
        return plan

    def _block_dependents(self, plan: stages_mod.StagePlan) -> None:
        blocked = {x.id for x in plan.stages if x.status == "blocked"}
        for st in plan.stages:
            if st.status not in ("pending", "active"):
                continue
            body = st.goal.split("\n\n", 1)[1] if st.goal.startswith("Overall task:") and "\n\n" in st.goal else st.goal
            dep = next((int(n) for n in _DEPENDS.findall(body) if int(n) in blocked and int(n) != st.id), None)
            if dep is not None:
                self.runner.block(st, f"depends on stage {dep}, which is blocked")
                self.decide(st, f"depends on blocked stage {dep}", "needs_human", False, 0)
                blocked.add(st.id)

    def _global_stop(self, idle: int) -> str:
        if self._segments >= int(self.conf("autopilot_max_segments", MAX_SEGMENTS)):
            return "segment cap"
        hours = self.conf("autopilot_max_hours", 0)
        if hours > 0 and self.clock() - self._t0 >= hours * 3600:
            return "time budget"
        if idle >= max(1, int(self.conf("autopilot_max_idle", MAX_IDLE))):
            return "idle"
        return ""

    def _hint(self) -> str:
        """What fails, from the VERIFIED STATE the latest segment started from (measured, no model)."""
        state = getattr(self.runner, "last_state", None)
        sec = getattr(state, "sections", None) or {}
        parts: list[str] = []
        failing = (sec.get("TESTS") or {}).get("failing") or []
        if failing:
            parts.append(f"failing test {failing[0]}")
        parts += [f"check {cid} is {status}" for cid, status in ((sec.get("CHECKS") or {}).get("results") or {}).items()
                  if status not in _QUIET_CHECKS][:2]
        stat = str((sec.get("STATIC") or {}).get("text") or "").splitlines()
        if len(stat) > 1:
            parts.append(stat[1].strip())
        return "; ".join(parts)

    def _ask_hint(self, st: Any) -> str:
        """The 2nd 'loop' stop of a stage, with a human at the terminal: ask ONCE for a hint (Enter = let it continue). '' when nobody answers."""
        if not intake_mod.tty():
            return ""
        items = getattr(getattr(self.session, "dead_ends", None), "items", None) or []
        head = f"{items[-1]['tool']}: {_one_line(items[-1]['cmd'], 80)}" if items else (self._hint() or f"stage {st.id}")
        question = f"I keep failing at {head}. Hint? (Enter to let me continue)"
        try:
            if self._ask is not None:
                return str(self._ask(question) or "").strip()
            self.session.console.print(f"[yellow]?[/yellow] {escape(question)}")
            return str(self.session.console.input("  ▸ ") or "").strip()
        except (EOFError, KeyboardInterrupt):
            return ""
        except Exception:  # noqa: BLE001 - a console that cannot read must not end the run
            return ""

    # ── the stage budget ────────────────────────────────────────────────
    def _budget_event(self, st: Any, action: str, reason: str, decision: str) -> None:
        self.emit("stage_budget", id=st.id, spent_s=round(st.spent_s, 1), steps=st.spent_steps, edits=st.real_edits, action=action)
        self.decide(st, reason, decision, False, 0)

    def _scope_satisfied(self, st: Any) -> bool:
        done = {*st.files, *st.waived}
        if st.scope and getattr(st, "lines", None):               # a ranged part: a change inside its range (cursor ranges), or a waiver
            from agent.codeagent import cursor as cursor_mod

            f = st.scope[0]
            return f in st.waived or cursor_mod.overlaps(st.edited.get(f), st.lines) or (f in done and not cursor_mod.merge(st.edited.get(f)))
        return bool(st.scope) and all(f in done for f in st.scope)

    def _harness_closes(self, st: Any) -> bool:
        """A stage the model never ended: scope changed / waived, no edit in its last 2 segments, >= 60 % of its budget used, and no test
        that is newly red (the runner's stage-end test measurement). Then nothing is left to do: done, by the harness."""
        if not (self._scope_satisfied(st) and st.no_edit_segments >= 2 and stages_mod.budget_share(st, self.cfg) >= stages_mod.BUDGET_CLOSE_SHARE):
            return False
        check = getattr(self.runner, "stage_tests", None)
        try:
            if check is not None and check(st):
                return False
        except Exception:  # noqa: BLE001 - a broken measurement is no reason to close a stage
            return False
        self.runner.close(st, stages_mod.CLOSE_NOTE)
        self.decide(st, "scope satisfied, no edits in 2 segments", "closed_by_harness", True, 0)
        return True

    def _budget_gate(self, st: Any) -> bool:
        """At a segment boundary of an unfinished stage: True when the harness closed it or the budget blocked it (the caller moves on)."""
        if self._harness_closes(st):
            return True
        why = stages_mod.budget_blocked(st, self.cfg)
        if not why:
            return False
        self.runner.block(st, why)
        final = st.budget_round >= 1
        self._budget_event(st, "final" if final else "block", why, "needs_human" if final else "budget_block")
        return True

    def _budget_retry(self, plan: stages_mod.StagePlan, notes: dict[int, str], fresh: dict[int, bool]) -> bool:
        """Everything else is done or blocked: ONE more round for the stages that were blocked by the budget, with half a budget."""
        again = [x for x in plan.stages if x.status == "blocked" and x.reason.startswith("budget:") and x.budget_round == 0]
        for x in again:
            dead = [f"{i['tool']}: {_one_line(i['cmd'], 80)}" for i in (getattr(getattr(self.session, "dead_ends", None), "items", None) or [])[-3:]]
            detail = "; ".join(p for p in (x.summary, ("dead ends: " + " | ".join(dead)) if dead else "") if p) or x.reason
            notes[x.id] = (f"Previous attempt used the whole budget without finishing: {_one_line(detail, 500)}. Decide quickly: finish what "
                           "is left of THIS part, or state with mark_done evidence that nothing is left.")
            fresh[x.id] = True
            numbers = stages_mod.budget_numbers(x)
            self.emit("stage_budget", id=x.id, spent_s=round(x.spent_s, 1), steps=x.spent_steps, edits=x.real_edits, action="retry")
            x.status, x.budget_round, x.budget_scale, x.reason = "pending", 1, 0.5, ""
            x.spent_s, x.spent_steps, x.real_edits, x.no_edit_segments, x.finished_at = 0.0, 0, 0, 0, 0.0
            self.decide(x, f"budget used up ({numbers}); one retry with half a budget", "budget_retry", False, 0)
        if again:
            self.persist()
        return bool(again)

    # ── one segment ─────────────────────────────────────────────────────
    def _segment(self, st: Any, note: str, fresh: bool) -> stages_mod.Segment:
        """The runner's segment; a Python exception is an 'error' stop with its traceback in the event log (KeyboardInterrupt and
        SystemExit are not caught here)."""
        try:
            return self.runner.segment(st, note, fresh_plan=fresh)
        except Exception as exc:  # noqa: BLE001 - crash isolation: this segment is lost, the supervisor and the saved work are not
            logger.exception("autopilot segment crashed")
            self.emit("autopilot_crash", stage=st.id, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            return stages_mod.Segment(types.SimpleNamespace(stopped="error", text=None), "red", f"crash: {type(exc).__name__}: {exc}")

    def _measure_final(self) -> Any:
        if self._final is None or self._final[0] != self._segments:
            try:
                state = self.verify()
            except Exception as exc:  # noqa: BLE001
                state = types.SimpleNamespace(text="", sections={"TESTS": {"text": f"TESTS: could not be measured ({type(exc).__name__})", "status": "error"}})
            self._final = (self._segments, state)
        return self._final[1]

    # ── the loop ────────────────────────────────────────────────────────
    def run(self, task: str) -> SupervisorResult:
        s = self.session
        self._t0 = self.clock()
        plan = self._plan(task)
        if self.runner is None:
            self.runner = stages_mod.make_runner(s, plan, self.persist)
        self.emit("autopilot_start", stages=len(plan.stages), task=_one_line(task, 200))
        self._inhibit_start()
        reason = ""
        try:
            reason = self._loop(plan)
        except KeyboardInterrupt:
            reason = "interrupted"
            self.say(f"[autopilot] interrupted - the session is saved; continue with: {brand.CMD} code --autopilot --resume")
        finally:
            self._inhibit_stop()
            s.stage_cap, s.stage_goal, s.segment_end_check = None, None, None
            self.persist()
        return self._finish(plan, reason)

    def _loop(self, plan: stages_mod.StagePlan) -> str:
        u: dict[int, int] = {}
        notes: dict[int, str] = {}
        fresh: dict[int, bool] = {}
        idle, current = 0, 0
        while True:
            self._block_dependents(plan)
            st = next((x for x in plan.stages if x.status in ("pending", "active")), None)
            if st is None:
                if self._budget_retry(plan, notes, fresh):
                    idle, current = 0, 0                      # the retried stage is activated again (a fresh acceptance list)
                    continue
                return ""
            stop = self._global_stop(idle)
            if stop:
                self.decide(st, stop, "stop", False, u.get(st.id, 0))
                return stop
            if not self.server_ok():
                self.decide(st, "server down", "stop", False, u.get(st.id, 0))
                return "server down"
            if current != st.id:
                fresh[st.id] = self.runner.activate(st) or fresh.get(st.id, False)
                current = st.id
            before = self.progress()
            t0, w0 = self.clock(), self.wall()
            self._seg_wall = w0
            self.session.segment_end_check = self._end_check
            seg = self._segment(st, " ".join(x for x in (notes.pop(st.id, ""), self._hints.pop(st.id, "")) if x), fresh.pop(st.id, False))
            self._segments += 1
            dm, dw = self.clock() - t0, self.wall() - w0
            suspended = dw > SUSPEND_MIN_S and dw > SUSPEND_FACTOR * dm
            after = self.progress()
            progress = after[0] != before[0] or after[1] > before[1]
            self.persist()
            stopped = getattr(seg.res, "stopped", "")
            if seg.kind == "interrupted" or stopped == "interrupted":
                return "interrupted"
            note = ""
            if seg.why == "resplit":
                # the stage was replaced by smaller sub-stages (ids renumbered): that is progress in itself, nothing is counted against them
                self.decide(st, "stage_cap", "resplit", True, u.get(st.id, 0))
                for per_stage in (u, notes, fresh, self._loop_stops, self._hints):
                    per_stage.clear()
                idle, current = 0, 0
                continue
            if stopped == "fresh_context":
                # ended on purpose (slow prefill): the same stage in a fresh context, not an unproductive segment - u and idle stay as they are
                self.decide(st, "slow prefill", "fresh_context", progress, u.get(st.id, 0))
                notes[st.id] = FRESH_NOTE
                if self._budget_gate(st):
                    u.pop(st.id, None)
                    idle = 0
                continue
            if seg.kind == "green":
                reds = red_items(self._measure_final()) if self._last_unfinished(plan, st) else []
                if not reds:
                    self.runner.finish(st, seg.res)
                    self.persist()
                    u[st.id], idle = 0, 0
                    continue
                why = "red"
                note = ("continuing: the answer was accepted but the measured state is still red (" + "; ".join(reds[:3]) + "); fix that first, "
                        "see VERIFIED STATE")
            elif seg.kind == "cap":
                why, note = "stage_cap", stages_mod.CAP_NOTE
            else:
                why = stopped or "red"
                note = f"continuing: the previous segment ended without a green result ({seg.why or why}); see VERIFIED STATE"
            if self._budget_gate(st):                          # the hard per-stage budget (or the harness closes a satisfied stage)
                u.pop(st.id, None)
                idle = 0
                self.persist()
                continue
            if stopped in ("loop", "verify_budget"):          # the verification budget ends a segment like a loop stop (phase.py)
                self._loop_stops[st.id] = self._loop_stops.get(st.id, 0) + 1
                if self._loop_stops[st.id] == 2:
                    hint = self._ask_hint(st)
                    if hint:
                        self._hints[st.id] = f"The user's hint: {hint}"
                        self.emit("autopilot_hint", stage=st.id)
            if suspended and not progress:
                self.decide(st, "suspend", "resume", False, u.get(st.id, 0))        # the PC slept: not the model's fault, not counted
                notes[st.id] = note
                continue
            k = u.get(st.id, 0)
            if progress:
                u[st.id], idle = 0, 0
                self.decide(st, why, "resume", True, 0)
                notes[st.id] = note
                continue
            u[st.id], idle = k + 1, idle + 1
            action = GRADES[k] if k < len(GRADES) else "needs_human"
            self.decide(st, why, action, False, k)
            if action == "resume_pause":
                self.session.lookact.pause()
                self.session._sigs.clear()
                getattr(self.session, "_declined", set()).clear()
                notes[st.id] = note + " Reading is paused for the next steps: make a change or run a command first."
            elif action == "fresh_stage":
                fresh[st.id] = True
                hint = self._hint()
                notes[st.id] = (f"NOTE: the previous {k + 1} segments of this stage made no progress (last stop: {why})"
                                + (f"; what fails: {hint}" if hint else "") + ". Start this stage again from its goal and take a different approach.")
            elif action == "needs_human":
                self.runner.block(st, f"needs human: {k + 1} segments in a row without progress, last stop: {why}"
                                  + (f" ({self._hint()})" if self._hint() else ""))
                u.pop(st.id, None)
                idle = 0
                self.persist()
            else:
                notes[st.id] = note

    @staticmethod
    def _last_unfinished(plan: stages_mod.StagePlan, st: Any) -> bool:
        return not any(x.status in ("pending", "active") and x is not st for x in plan.stages)

    # ── the end ─────────────────────────────────────────────────────────
    def _finish(self, plan: stages_mod.StagePlan, reason: str) -> SupervisorResult:
        states = [x.status for x in plan.stages]
        any_done = "done" in states
        if reason == "interrupted":
            verdict, code = "partial", 130
        elif reason in ("idle", "server down"):
            verdict, code = "needs_human", 2
        elif all(x == "done" for x in states):
            verdict, code = "done", 0
        else:
            budget = any(x.status == "blocked" and x.reason.startswith("budget:") for x in plan.stages)
            verdict, code = ("partial", 3) if any_done or budget else ("needs_human", 2)
        report = self._report(plan, verdict, reason)
        res = SupervisorResult(verdict, code, report, list(self.decisions),
                               stages_mod.StageReport("done" if verdict == "done" else ("partial" if any_done else "failed"), plan.stages), reason)
        self.emit("autopilot_end", verdict=verdict, exit_code=code, reason=reason, segments=self._segments,
                  seconds=round(self.clock() - self._t0, 1), stages=states)
        self.say(report)
        return res

    def _scratch_files(self) -> list[str]:
        root, out = Path(self.session.root), []
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in _PRUNE)
            for name in sorted(files):
                rel = (Path(dirpath) / name).relative_to(root).as_posix()
                if rel != f".scratch/{REPORT_NAME}" and is_scratch_path(rel):
                    out.append(rel)
        return out

    def _speed_lines(self) -> list[str]:
        """The time breakdown and the causes of THIS run's event-log session (timing.py); measured values only."""
        sid = getattr(self.events, "session", None)
        try:
            log = [e for e in timing.read_log(getattr(self.events, "path", None)) if e.get("session") == sid] or list(getattr(self.events, "recent", []))
            return timing.render(timing.breakdown(log)).splitlines()
        except Exception:  # noqa: BLE001 - the report must come out even when the log cannot be read
            return ["not measured (event log unreadable)"]

    def _report(self, plan: stages_mod.StagePlan, verdict: str, reason: str) -> str:
        state = self._measure_final()
        sec = getattr(state, "sections", None) or {}
        tests = str((sec.get("TESTS") or {}).get("text") or "TESTS: not measured").splitlines()[0]
        smoke = str((sec.get("TUI SMOKE") or {}).get("text") or "").splitlines()[:1]
        reds = red_items(state)
        secs = self.clock() - self._t0
        lines = [f"# autopilot report: {verdict}" + (f" ({reason})" if reason else ""), "",
                 f"wall time: {secs / 3600:.2f} h ({secs:.0f} s)   segments: {self._segments}   decisions: {len(self.decisions)}", "", "## stages"]
        for x in plan.stages:
            lines.append(f"- stage {x.id} [{x.status}] {x.title}: {x.attempts} attempt(s), {x.seconds:.0f} s, "
                         + (", ".join(x.files[:8]) + (f" +{len(x.files) - 8}" if len(x.files) > 8 else "") if x.files else "no file changed")
                         + (f" - {x.reason}" if x.reason else ""))
        lines += ["", "## tests", tests, *(["", "## tui smoke (random-play crash gate, cached)", *smoke] if smoke else []), "", "## speed (measured)", *self._speed_lines(), "", "## NOT VERIFIED"]
        todo = [f"stage {x.id} {x.title}: {x.reason or x.status}" + (" (retried once with half a budget)" if x.reason.startswith("budget:") and x.budget_round else "")
                for x in plan.stages if x.status != "done"] + reds
        if reason in ("idle", "segment cap", "time budget", "server down"):
            todo.append(f"run ended early: {reason}")
        lines += [f"- {t}" for t in todo] or ["- none"]
        changed = sorted({*getattr(getattr(self.session.ctx, "ledger", None), "files", {}), *(f for x in plan.stages for f in x.files)})
        scratch = self._scratch_files()
        lines += ["", "## changed files"] + ([f"- {f}" for f in changed[:60]] + ([f"- ... and {len(changed) - 60} more"] if len(changed) > 60 else [])
                                             or ["- none"])
        lines += ["", "## scratch files (not part of the work, safe to delete)"] + ([f"- {f}" for f in scratch[:40]] or ["- none"])
        text = "\n".join(lines) + "\n"
        try:
            out = Path(self.session.root) / ".scratch" / REPORT_NAME
            edits.ensure_scratch_excluded(self.session.root)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        except OSError:
            pass
        return text
