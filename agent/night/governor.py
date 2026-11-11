"""The night governor: walks the prompt queue until the clock runs out, whatever the model does.

    Night(root, items, rules, hours, runner=..., tests=...).run() -> Report

Everything that decides is MEASURED: the autopilot's exit code, the project's own tests (per test, with a flaky re-run), git (what changed, against the
last green checkpoint), and the coding agent's event log. Never what the model says it did.

Per attempt:  health gate -> run (watched subprocess) -> tests -> guard (weakened tests / protected files are reverted) -> analyze (why did it fail) -> lesson
              -> checkpoint if green -> handoff text for the next attempt.
The next strategy comes from the failure KIND (analyzer.next_strategy): resume | fresh | rollback | small (and `start` for the first). A prompt whose time slice
is spent is 'partial'; leftover time goes to the open prompts again. The last RESERVE_S are for the end state: tests red -> roll back to the last green
checkpoint (the red work stays under night/red-N) and write the report."""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from agent.night import vramhogs as vramhogs_mod
from agent.night import analyzer, attemptlog, deliverables, evidence, gitbox, guard, handoff as handoff_mod, history as history_mod, mapper, planfile, testwatch
from agent.night.lessons import Lessons
from agent.night.queue import Item
from agent.night.runner import Attempt, Outcome
from agent.night.handoff import DiffStat, EventSummary
from agent.night.testwatch import TestState

LADDER = ("resume", "fresh", "rollback", "small")
ATTEMPT_MAX_S, STALL_S, RESERVE_S, MIN_SLICE_S, QUICK_FAIL_S, BACKOFF_S, GATE_POLL_S = 90 * 60, 20 * 60, 15 * 60, 20 * 60, 30, 60, 60
CLOSING_MIN_S, CLOSING_MAX_S = 8 * 60, 15 * 60
GRACE_S = 12 * 60                                      # extra time for a prompt that ran out of its slice while still making progress
REVIEW_S, SLOW_TPS, PLAN_S = 25 * 60, 8.0, 8 * 60              # the plan attempt must never eat a prompt's time slice
REVIEW = ("Review the work that was just done for this task: run `git diff {tag}` and compare it with the task above. Look for placeholders, missing edge cases, missing tests and behaviour "
          "that does not match the task. Fix REAL problems only, do not add features, keep every test green, then stop.")

_SMALL = "Pick the ONE most important part of this task that is still missing, finish it completely (code and a test for it), then stop."
_DIFFERENT = "The last approach failed and its work was rolled back to the last good state. Do NOT repeat it: choose a different way to reach the goal."
_BUG = "This is a bug: FIRST write a test that fails because of the bug (run it and see it fail), THEN fix the code until that test passes and no other test breaks."
_REPRO = ("Write ONLY a test that reproduces this bug: it must FAIL now, because of the bug (run it and see it fail). Put it in the project's test folder. "
          "Do NOT fix the bug in this step and do NOT change any other file. Then stop.")
_FIX = "A test that proves the bug already exists and fails: {ids}. Fix the code (not the test) until it passes and no other test breaks."
_REPAIR = ("The work on this task is almost finished, but the tests are not green: {digest} Fix exactly this - in the code, NEVER by deleting, skipping or weakening a test - "
           "make the whole test suite pass, then stop. Do not start anything new.")
_END = ("Work until every part of THIS task is implemented and the project's tests pass. Do not stop to ask: decide yourself. "
        "Never delete, skip or weaken a test to get green. When it is complete, call mark_done for each item with evidence.")


@dataclass
class Result:
    out: Outcome
    after: TestState
    hand: handoff_mod.Handoff
    verdict: analyzer.Verdict
    report: guard.Report
    ev: EventSummary
    strategy: str
    spent: float
    trigger: str = ""                                       # the failure kind that made the governor choose this strategy ("" for the first attempt)
    wall0: float = 0.0                                      # wall-clock start of the attempt (the evidence slices the event log with it)
    task_chars: int = 0


@dataclass
class Report:
    done: list[str]
    partial: list[str]
    tests_ok: bool
    text: str


def as_state(res) -> TestState:
    """A tests callable may return a TestState or the older (ok, tail) pair."""
    if isinstance(res, TestState):
        return res
    ok, tail = res
    return TestState(ok=bool(ok), tail=str(tail), digest=str(tail)[-400:])


class Night:
    def __init__(self, root: Path, items: list[Item], rules: str = "", hours: float = 8.0, *, runner: Callable[[Attempt], Outcome],
                 tests: Callable[[], object], clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print, health=None, events: Path | None = None, wall: Callable[[], float] = time.time,
                 budget: guard.Budget | None = None, verify: Callable[[str], bool] | None = None, map_mode: str = "auto", tournament: int = 3, review: bool = False, env_lines: list[str] | None = None, plan_all: bool = False, use_history: bool = False,
                 heartbeat: Callable[..., None] | None = None, prompts_src: list[str] | None = None) -> None:
        self.root, self.items, self.rules = Path(root), items, rules
        self.runner, self._tests, self.clock, self.sleep, self.say = runner, tests, clock, sleep, log
        self.health, self.events_path, self.wall, self.budget = health, events, wall, budget or guard.Budget()
        self.verify_cmd = verify or self._shell_ok
        self.tournament, self.review = tournament, review     # best-of-N after two equal failures (0/1 = off); a review pass after every finished prompt
        self.beat = heartbeat or (lambda *a, **k: None)   # proof of life for the systemd watchdog (sdnotify.Heartbeat.alive)
        self.prompts_src = prompts_src or []             # where the prompts came from (copied into the evidence archive)
        self.use_history = use_history                    # the strategy priors of earlier nights (history.py); off in tests so they never read the real history
        self.plan_all = plan_all                          # write a checklist plan first for every prompt (a prompt can ask for it with `plan: true`)
        self.env_lines = env_lines or []                  # what the night runs on (envinfo.collect): printed at the top of the report
        self.map_mode = map_mode                          # off | auto (the harness-written map) | full (+ a model pass that adds notes)
        self.night_dir = self.root / ".night"
        self.lessons = Lessons(self.night_dir / "LESSONS.md")
        self.attempts_file = self.night_dir / "attempts.jsonl"
        self.start = self.clock()
        self.end = self.start + hours * 3600
        self.end_wall = self.wall() + hours * 3600          # the monotonic clock stands still while the PC is suspended: the wall clock keeps the promise of `--until 07:30`
        self.events: list[str] = []
        self.base_ok = True
        self.last_state: TestState | None = None          # the tests as they were at the last green checkpoint (the baseline for regressions)
        self.flaky: set[str] = set()
        self.last_digest = ""                             # what the tests said after the latest attempt (for the closing repair round)
        self.tps_hist: list[float] = []                  # decode speed of the attempts, to notice a model that got slow during the night
        self.hogs = vramhogs_mod.hogs                     # who holds graphics memory (injectable)

    # ---- bookkeeping
    def note(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        self.events.append(line)
        self.say(line)
        self.save()

    def save(self) -> None:
        self.night_dir.mkdir(exist_ok=True)
        state = {"left_s": max(0, int(self.remaining())), "end_wall": self.end_wall, "items": [asdict(i) for i in self.items], "events": self.events[-200:], "flaky": sorted(self.flaky)}
        tmp = self.night_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        os.replace(tmp, self.night_dir / "state.json")           # atomic: a crash never leaves half a state file

    def remaining(self) -> float:
        return min(self.end - self.clock(), self.end_wall - self.wall())

    def control(self) -> str:
        """The content of .night/CONTROL ('stop' or 'pause'), '' when absent."""
        try:
            return (self.night_dir / "CONTROL").read_text().strip().lower()
        except OSError:
            return ""

    def _shell_ok(self, cmd: str) -> bool:
        try:
            return subprocess.run(cmd, shell=True, cwd=self.root, capture_output=True, timeout=900).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def run_tests(self) -> TestState:
        """The tests, with a flaky settle: a failure that disappears on a re-run is not held against the attempt."""
        first = as_state(self._tests())
        if first.ok or not first.failed or len(first.failed) > 5:
            return first
        state, flaky = testwatch.settle_flaky(first, lambda: as_state(self._tests()))
        if flaky:
            self.flaky |= set(flaky)
            self.note(f"flaky tests (passed on a re-run): {', '.join(flaky[:4])}")
        return state

    # ---- the text of an attempt
    def _task(self, it: Item, strategy: str, handoff: str) -> str:
        if strategy == "resume":
            return ""
        parts = [self.rules] if self.rules else []
        if (self.night_dir / "KNOWLEDGE.md").exists():
            parts.append("What earlier tasks of this night produced is summarised in .night/KNOWLEDGE.md - read it first.")
        lessons = self.lessons.recent(it.name)
        if lessons:
            parts.append("LESSONS from earlier attempts of this night (measured, not guessed):\n" + lessons)
        parts.append(it.text)
        patch = self.night_dir / "patches" / f"{gitbox.slug(it.name)}.patch"
        if patch.exists() and strategy != "resume":
            parts.append(f"An earlier attempt at this task left unfinished work, saved as `.night/patches/{patch.name}` (a git patch; apply it with `git apply .night/patches/{patch.name}`). "
                         "It leaves tests failing: reuse it and fix those tests, or start over if it is not worth keeping.")
        if planfile.path(self.root, it.name).exists():
            left = planfile.remaining(self.root, it.name)
            parts.append(f"A plan for this task is in .night/plans/{planfile.path(self.root, it.name).name}: follow it. The harness ticks a step only when its check passes."
                         + (f" Still open: {left}." if left else ""))
        if strategy == "repro":
            parts.append(_REPRO)
        elif it.kind == "bug" and it.repro:
            parts.append(_FIX.format(ids=", ".join(it.repro[:4])))
        elif it.kind == "bug":
            parts.append(_BUG)
        if strategy == "repair":
            parts.append(_REPAIR.format(digest=(self.last_digest or "see the failing tests")[:600]))
        if strategy == "small":
            parts.append(_SMALL)
        if strategy == "rollback":
            parts.append(_DIFFERENT)
        if handoff:
            parts.append("HANDOFF from the previous attempt (measured by the harness): " + handoff)
        parts.append(_END)
        return "\n\n".join(parts)

    # ---- waiting for the machine
    def _gate(self) -> None:
        if self.health is None:
            return
        reason = self.health.gate()
        shown = ""
        while reason and self.remaining() > RESERVE_S and self.control() != "stop":
            if reason != shown:
                self.note(f"waiting: {reason}")
                shown = reason
            self.sleep(GATE_POLL_S)
            self.beat("waiting: " + reason)
            reason = self.health.gate()
        if shown and not reason:
            self.note("machine is fit again, going on")

    def _paused(self) -> None:
        while self.control() == "pause" and self.remaining() > RESERVE_S:
            self.sleep(GATE_POLL_S)

    # ---- one attempt = execute (measure) + commit (decide what stays)
    def _execute(self, it: Item, strategy: str, handoff: str, until: float, trigger: str = "") -> Result:
        self._gate()
        base_tag = it.repro_sha or gitbox.last_green(self.root)             # what this prompt's own changes are measured against (a prepared / proven state, else the last green checkpoint)
        wall0, t0 = self.wall(), self.clock()
        log = self.night_dir / f"{it.name}-attempt{it.attempts}.log"
        pidfile = self.night_dir / "attempt.pid"
        task = self._task(it, strategy, handoff)
        self.beat(f"{it.name} attempt {it.attempts} ({strategy})")
        try:
            out = self.runner(Attempt(task, new=strategy != "resume", timeout_s=min(ATTEMPT_MAX_S, max(60.0, until - t0)),
                                      stall_s=STALL_S, log=str(log), on_start=lambda pid: pidfile.write_text(str(pid)), on_poll=self.beat))
        finally:
            pidfile.unlink(missing_ok=True)
        spent = self.clock() - t0
        it.seconds += spent
        if out.reaped:
            self.note(f"[{it.name}] stopped {len(out.reaped)} process(es) the attempt left running: {'; '.join(out.reaped[:3])}")
        after = self.run_tests()
        self.beat(f"{it.name}: tests run")
        if planfile.path(self.root, it.name).exists():
            done, total = planfile.evaluate(self.root, it.name, after)
            self.note(f"[{it.name}] plan: {done}/{total} steps verified")
        allow = bool(guard.REMOVAL_WORDS.search(it.text))                  # a prompt that asks to remove / refactor may delete and shrink code
        report = guard.check(self.root, base_tag, self.budget, allow)
        if report.fatal:
            fixed = guard.revert(self.root, base_tag, report)
            self.note(f"[{it.name}] guard: {report.text()} -> reverted {', '.join(fixed) or 'nothing'}")
            self.lessons.add("broke_tests" if any(f.kind in ("weakened", "deleted_src") for f in report.fatal) else "crash", report.text()[:100], it.name)
            after = self.run_tests()
            report = guard.check(self.root, base_tag, self.budget, allow)
        elif report.findings:
            self.note(f"[{it.name}] guard (warning): {report.text()}")
        ev = handoff_mod.summarize_events(self.events_path, wall0, self.root.name) if self.events_path else handoff_mod.EventSummary()
        if self.events_path:
            handoff_mod.slice_events(self.events_path, wall0, self.root.name, self.night_dir / "events" / f"{it.name}-attempt{it.attempts}.jsonl")
        self._check_drift(it, ev)
        if ev.steps >= 10 and 0 < ev.tps < SLOW_TPS:
            self.note(f"[{it.name}] the model was slow: {ev.tps:.1f} tok/s (prompt reading {ev.prefill_share * 100:.0f}% of the time) - CPU spill, a hot GPU or a game?")
        self.last_digest = "" if after.ok else after.digest
        hand = handoff_mod.build(self.root, base_tag, self.last_state, after, ev, out.code, out.why)
        verdict = analyzer.classify(out, self.last_state, after, hand.diff, ev, _tail(log))
        if verdict.kind == "green" and it.verify and not self.verify_cmd(it.verify):
            verdict = analyzer.Verdict("tests_red", f"the verify command of this prompt fails: {it.verify}")
        if verdict.kind == "green":
            gone = deliverables.missing(it.expected, self.root)
            if gone:
                verdict = analyzer.Verdict("incomplete", "missing: " + ", ".join(gone[:4]))
        if any(f.kind == "syntax" for f in report.findings):
            verdict = analyzer.Verdict("tests_red", report.text())
        return Result(out, after, hand, verdict, report, ev, strategy, spent, trigger, wall0, len(task))

    def _check_drift(self, it: Item, ev: EventSummary) -> None:
        """The same model on the same machine should keep its speed. When the last two attempts run at under 60% of the first two (4+ attempts with 10+ steps), say who holds
        graphics memory and restart the model server once (health.restart, at most every 30 min): a fresh server often gets the layers back on the GPU."""
        if ev.steps < 10 or ev.tps <= 0:
            return
        self.tps_hist.append(ev.tps)
        h = self.tps_hist
        if len(h) < 4:
            return
        start, now = sum(h[:2]) / 2, sum(h[-2:]) / 2
        if now >= 0.6 * start:
            return
        try:
            who = vramhogs_mod.summary(self.hogs())
        except Exception:  # noqa: BLE001 - diagnostics only
            who = "graphics memory not readable"
        restarted = bool(self.health is not None and getattr(self.health, "restart", None) and self.health.restart())
        self.note(f"[{it.name}] speed drifted: {now:.0f} tok/s now, {start:.0f} at the start of the night; {who}" + ("; the model server was restarted" if restarted else ""))

    def _commit(self, it: Item, r: Result) -> tuple[bool, str]:
        """Record the result: the failure kind, a checkpoint when green, a lesson when not. Returns (prompt is done, handoff text)."""
        evidence.record_attempt(self, it, r)                       # the record of what happened, before the governor reacts to it
        it.kinds.append(r.verdict.kind)
        before = self.last_state
        progress = bool(before) and (r.after.passed > before.passed or len(r.after.failed) < len(before.failed)) and not r.hand.diff.empty()
        attemptlog.record(self.attempts_file, prompt=it.name, n=it.attempts, strategy=r.strategy, trigger=r.trigger, kind=r.verdict.kind, green=r.verdict.kind == "green",
                          progress=progress, passed=r.after.passed, failed=len(r.after.failed), added=r.hand.diff.added, removed=r.hand.diff.removed,
                          tps=round(r.ev.tps, 1), minutes=round(r.spent / 60, 1), steps=r.ev.steps, guards=r.ev.limits_hit, blocked=r.ev.look_blocked, think=r.ev.think_steps,
                          aborts=r.ev.loop_aborts, caps=r.ev.stage_caps, masks=r.ev.masks)
        if r.after.ok and r.verdict.kind not in ("tests_red", "incomplete") or r.verdict.kind == "green":
            tag = gitbox.checkpoint(self.root, f"{it.name} attempt {it.attempts}")
            self.last_state = r.after
            self._refresh_map()
            self.note(f"[{it.name}] attempt {it.attempts} ({r.strategy}) {r.verdict.kind}, exit {r.out.code} {r.out.why} -> {tag}")
        else:
            self.note(f"[{it.name}] attempt {it.attempts} ({r.strategy}) {r.verdict.kind}: {r.verdict.detail[:120]}")
        if r.verdict.kind == "green":
            return True, ""
        self.lessons.add(r.verdict.kind, r.verdict.detail, it.name)
        it.note = f"{r.verdict.kind}: {r.verdict.detail[:160]}"
        return False, r.hand.text

    def _attempt(self, it: Item, strategy: str, handoff: str, until: float, trigger: str = "") -> tuple[bool, str]:
        return self._commit(it, self._execute(it, strategy, handoff, until, trigger))

    def _tournament(self, it: Item, handoff: str, until: float) -> tuple[bool, str]:
        """Best-of-N from ONE checkpoint: every candidate starts from the same good state; the best by measurement (green, then tests passing, then the
        smaller diff) is kept and committed, the others are only snapshots under no tag. Only used when two attempts in a row failed the same way."""
        base = gitbox.last_green(self.root)
        cands: list[tuple[tuple, str, Result, int]] = []
        for k in range(self.tournament):
            if self.clock() >= until - 5 * 60 or self.control() == "stop":
                break
            gitbox.reset_to(self.root, base)
            it.attempts += 1
            r = self._execute(it, "fresh" if k % 2 == 0 else "small", handoff, until, it.kinds[-1] if it.kinds else "")
            sha = gitbox.snapshot(self.root, f"{it.name} tournament candidate {k + 1}")
            key = (1 if r.verdict.kind == "green" else 0, testwatch.score(r.after), -(r.hand.diff.added + r.hand.diff.removed))
            cands.append((key, sha, r, it.attempts))
            self.note(f"[{it.name}] tournament {k + 1}/{self.tournament}: {r.verdict.kind}, {r.after.passed} passing, +{r.hand.diff.added}/-{r.hand.diff.removed}")
        if not cands:
            gitbox.reset_to(self.root, base)
            return False, handoff
        best = max(cands, key=lambda c: c[0])
        gitbox.reset_to(self.root, best[1])
        self.note(f"[{it.name}] tournament winner: candidate with {best[2].verdict.kind}")
        return self._commit(it, best[2])

    def _review(self, it: Item) -> None:
        """A second look at finished work: a fresh attempt reads the diff against the task and fixes real problems only. Its result is kept
        only if the tests stay green and the guards are clean - otherwise the work is exactly as it was."""
        base = gitbox.last_green(self.root)
        it.attempts += 1
        r = self._execute(it, "review", REVIEW.format(tag=base), self.clock() + REVIEW_S)
        if r.after.ok and not r.report.fatal:
            tag = gitbox.checkpoint(self.root, f"{it.name} review")
            self.last_state = r.after
            self._refresh_map()
            self.note(f"[{it.name}] review kept ({tag}): +{r.hand.diff.added}/-{r.hand.diff.removed}")
        else:
            gitbox.restore(self.root, base)
            self.note(f"[{it.name}] review discarded: tests were not green afterwards")

    def _repro(self, it: Item, until: float) -> bool:
        """A bug prompt starts with a test that proves the bug: the attempt must leave the tests RED with a NEW failing test and change nothing but test files.
        Measured, not claimed. The test is committed (a roll back of the fix never removes it). After 3 tries without one the prompt runs as a plain task."""
        slug = "".join(c if c.isalnum() else "-" for c in it.name)
        for k in range(3):
            if self.clock() >= until - 5 * 60 or self.control() == "stop":
                return False
            it.attempts += 1
            r = self._execute(it, "repro", "", until)
            new = testwatch.regressions(self.last_state, r.after)
            only_tests = bool(r.hand.diff.files) and all(guard.is_test(path) for path, _, _ in r.hand.diff.files)
            if not r.after.ok and new and only_tests:
                it.repro, it.repro_sha = new, gitbox.snapshot(self.root, f"{it.name}: test that proves the bug")
                gitbox.git(self.root, "tag", "-f", f"night/repro-{slug}", it.repro_sha)
                self.last_state = r.after                    # the baseline of "what broke" is now: this one known failing test
                self.note(f"[{it.name}] the bug is proven by {', '.join(new[:3])}")
                return True
            self.note(f"[{it.name}] repro attempt {k + 1} did not give exactly one new failing test in test files only: undone")
            gitbox.restore(self.root, gitbox.last_green(self.root))
        it.kind = ""
        self.note(f"[{it.name}] no reproducing test after 3 tries: running it as a plain task")
        return False

    def _plan(self, it: Item, until: float) -> bool:
        """The architect step: one attempt that writes ONLY the plan file; kept when it is a usable checklist and nothing else changed, else undone (twice at most)."""
        p = planfile.path(self.root, it.name)
        for k in range(2):
            if self.clock() >= until - 10 * 60 or self.control() == "stop":
                return False
            it.attempts += 1
            task = (self.rules + "\n\n" if self.rules else "") + it.text + "\n\n" + planfile.PLAN_PROMPT.format(path=f".night/plans/{p.name}")
            self.runner(Attempt(task, new=True, timeout_s=min(PLAN_S, max(60.0, until - self.clock())), stall_s=STALL_S,
                                log=str(self.night_dir / f"{it.name}-plan{k + 1}.log")))
            steps = planfile.parse(p.read_text()) if p.exists() else []
            changed = [f for f, _, _ in handoff_mod.diff_since(self.root, gitbox.last_green(self.root)).files]
            if planfile.usable(steps) and not changed:
                done, total = planfile.evaluate(self.root, it.name, self.last_state)
                self.note(f"[{it.name}] plan with {len(steps)} steps ({done} already true)")
                return True
            self.note(f"[{it.name}] plan attempt {k + 1} unusable ({len(steps)} steps, {len(changed)} other files changed): undone")
            if changed:
                gitbox.restore(self.root, gitbox.last_green(self.root))
            p.unlink(missing_ok=True)
        return False

    def store_for(self, it: Item) -> Path:
        """A folder outside the project for what the prepare/score tools of a prompt keep between the two (the model cannot reach it by accident)."""
        import hashlib

        from agent import osport

        d = osport.state_home() / "agent" / "night-eval" / f"{self.root.name}-{hashlib.sha1(str(self.root).encode()).hexdigest()[:8]}" / gitbox.slug(it.name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _run_hook(self, it: Item, cmd: str, what: str) -> tuple[int, str]:
        env = {**os.environ, "NIGHT_STORE": str(self.store_for(it)), "NIGHT_ROOT": str(self.root), "NIGHT_PROMPT": it.name}
        try:
            p = subprocess.run(cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=1800, env=env)
        except (OSError, subprocess.SubprocessError) as e:
            return 1, f"{type(e).__name__}: {e}"
        (self.night_dir / f"{it.name}-{what}.log").write_text(f"$ {cmd}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}\n--- exit {p.returncode}\n")
        return p.returncode, p.stdout

    def _prepare(self, it: Item) -> None:
        """`prepare:` of the prompt (an evaluation prompt injects its bugs or records its golden master here). The state after it is the floor this prompt is measured against and
        rolled back to; the tests are measured again, because the preparation usually makes them red on purpose."""
        code, out = self._run_hook(it, it.prepare, "prepare")
        it.prepared = True
        self.note(f"[{it.name}] prepared (exit {code}): {out.strip().splitlines()[-1][:140] if out.strip() else ''}")
        self.last_state = self.run_tests()
        it.repro_sha = gitbox.snapshot(self.root, f"{it.name}: prepared state")
        self.last_digest = "" if self.last_state.ok else self.last_state.digest

    def _score(self, it: Item) -> None:
        """`score:` of the prompt: the last JSON line its command prints is the measured result of the prompt (kept in the state, the evidence and the report)."""
        code, out = self._run_hook(it, it.score, "score")
        result = None
        for line in reversed(out.strip().splitlines()):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                result = value
                break
        it.scored = result if result is not None else {"error": f"no JSON result (exit {code})"}
        self.note(f"[{it.name}] score: {json.dumps(it.scored)[:200]}")

    def _work(self, it: Item, until: float) -> None:
        try:
            if it.prepare and not it.prepared:
                self._prepare(it)
            self._work_inner(it, until)
        finally:
            if it.score:
                try:
                    self._score(it)
                except Exception as e:  # noqa: BLE001 - a broken scorer must not take the night with it
                    it.scored = {"error": f"{type(e).__name__}: {str(e)[:100]}"}

    def _work_inner(self, it: Item, until: float) -> None:
        it.status = "active"
        if (it.plan or self.plan_all) and not planfile.path(self.root, it.name).exists() and it.kind != "map":
            self._plan(it, until)
        if not it.expected:
            it.expected = deliverables.expected(it.text, self.root)
        if it.kind == "bug" and not it.repro:
            self._repro(it, until)
        first, handoff, quick, n, trigger = it.attempts == 0, it.note, 0, 0, ""
        history = list(it.kinds)
        while self.clock() < until - 60 and self.control() != "stop":
            self._paused()
            n += 1
            if n == 1:
                strategy = "start" if first else "fresh"
            else:
                trigger = history[-1] if history else "tests_red"
                stats = history_mod.merge(attemptlog.rates(attemptlog.rows(self.attempts_file), trigger), history_mod.priors(trigger) if self.use_history else {})
                strategy = analyzer.next_strategy(trigger, history[:-1], stats)        # the kinds BEFORE the latest decide how often it repeated
            stuck = len(history) >= 2 and history[-1] == history[-2] and history[-1] not in ("green", "server_down")
            if self.tournament >= 2 and stuck and until - self.clock() > 45 * 60:
                done, handoff = self._tournament(it, handoff, until)
                history = list(it.kinds)
                if done:
                    return self._finish_prompt(it)
                continue
            it.attempts += 1
            if strategy == "rollback":
                target = it.repro_sha or gitbox.last_green(self.root)         # a bug fix goes back to the proving test, not to before it
                kept = gitbox.restore(self.root, target)
                self.note(f"[{it.name}] rolled back to {target[:12]} (red work kept as {kept})")
            t0 = self.clock()
            done, handoff = self._attempt(it, strategy, handoff, until, trigger)
            history = list(it.kinds)
            if done:
                return self._finish_prompt(it)
            quick = quick + 1 if self.clock() - t0 < QUICK_FAIL_S else 0
            if quick >= 2:                                   # the model server is probably down: do not spin
                self.sleep(BACKOFF_S)
        self._grace(it, history)
        if it.status != "done":
            it.status = "partial"
            self.note(f"[{it.name}] time slice spent: partial")
            if self.last_digest and any(x.status == "pending" and x is not it for x in self.items):      # a red tree: the NEXT prompt must not build on it (one red test would block every later checkpoint)
                self._shelve(it)                                       # (the last open prompt keeps its tree: the revisit and the closing round repair it)

    def _shelve(self, it: Item) -> None:
        """An unfinished prompt with red tests: save its work as a patch against the last green checkpoint and go back to that checkpoint (the red state stays under night/red-N too).
        The prompt gets the patch back when it is revisited, so the work is reusable and nothing is lost."""
        base = it.repro_sha or gitbox.last_green(self.root)
        red = gitbox.restore(self.root, base)
        patch = gitbox.git(self.root, "diff", "--binary", base, red).stdout
        target = self.night_dir / "patches" / f"{gitbox.slug(it.name)}.patch"
        if patch.strip():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch)
            it.note = (it.note + f" | unfinished work saved as .night/patches/{target.name} (tests were red: {self.last_digest[:80]})")[:400]
        self.last_digest = ""
        self.note(f"[{it.name}] the tests are still red: the work is saved as {target.name if patch.strip() else 'no patch'} and the folder is back at {base} so the next prompt starts green")

    def _grace(self, it: Item, history: list[str]) -> None:
        """A prompt whose slice ran out while it was making progress (its last attempt ended in a timeout, red tests or a missing file after a real change) gets ONE more stretch
        of up to GRACE_S: a repair attempt when the tests are red, a fresh continuation when they are green. Moving on with a red working tree would hand the next prompt a broken base."""
        if it.grace_used or not history or history[-1] not in ("timeout", "almost", "tests_red", "incomplete", "broke_tests") or self.control() == "stop":
            return
        if self.remaining() < RESERVE_S + GRACE_S + 5 * 60:
            return
        it.grace_used = True
        strategy = "repair" if self.last_digest else "fresh"
        self.note(f"[{it.name}] grace period: {GRACE_S // 60} more minutes ({strategy}) for work that was making progress")
        it.attempts += 1
        done, _ = self._attempt(it, strategy, it.note, self.clock() + GRACE_S, history[-1])
        if done:
            self._finish_prompt(it)

    def _safe_work(self, it: Item, until: float) -> None:
        """A bug in the governor or a surprise from the machine must cost one prompt at most, never the night."""
        try:
            self._work(it, until)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001
            self.night_dir.mkdir(exist_ok=True)
            with open(self.night_dir / "errors.log", "a") as f:
                f.write(f"--- {time.strftime('%H:%M:%S')} [{it.name}]\n{traceback.format_exc()}\n")
            it.status, it.note = "partial", f"governor error: {type(e).__name__}: {str(e)[:100]}"
            self.note(f"[{it.name}] governor error ({type(e).__name__}): the prompt is set aside, the night goes on (see .night/errors.log)")
            self.sleep(BACKOFF_S)

    def _finish_prompt(self, it: Item) -> None:
        it.status, it.note = "done", ""
        gitbox.mark_done(self.root, it.name)
        self._learn(it)
        if self.review and it.name != "map" and self.remaining() > RESERVE_S + 2 * REVIEW_S:
            self._review(it)

    def _refresh_map(self) -> None:
        if self.map_mode != "off":
            try:
                changed = mapper.refresh(self.root)
                if changed:
                    self.note(f"map refreshed ({len(changed)} files changed)")
            except Exception as e:  # noqa: BLE001 - a map problem must never stop the night
                self.note(f"map refresh failed: {e}")

    def _learn(self, it: Item) -> None:
        report = self.root / ".scratch" / "autopilot-report.md"
        body = "\n".join(report.read_text(errors="replace").splitlines()[:30]) if report.exists() else "done"
        with open(self.night_dir / "KNOWLEDGE.md", "a") as f:
            f.write(f"\n## {it.name} (done)\n{body}\n")

    # ---- the whole night
    def _ready(self, it: Item) -> bool:
        by_name = {i.name: i for i in self.items}
        # a prerequisite that used up its time slice (partial, with attempts) no longer blocks: one unfinished prompt must not end the whole night
        return all(by_name[d].status == "done" or (by_name[d].status == "partial" and by_name[d].attempts > 0) for d in it.depends if d in by_name)

    def _slice(self, it: Item, open_items: list[Item]) -> float:
        total = sum(max(i.weight, 0.1) for i in open_items) or 1.0
        return max(MIN_SLICE_S, (self.remaining() - RESERVE_S) * max(it.weight, 0.1) / total)

    def _kill_orphan(self) -> None:
        """After a hard kill of the governor (SIGKILL, a power cut) the attempt it started may still be working alone in the folder: end it before anything else runs."""
        pidfile = self.night_dir / "attempt.pid"
        try:
            pid = int(pidfile.read_text().strip())
        except (OSError, ValueError):
            return
        pidfile.unlink(missing_ok=True)
        try:
            if os.getsid(pid) == pid:                                          # still alive and the leader of its own session: ours
                from agent.night import reap

                left = reap.reap_session(pid)
                self.note(f"stopped an attempt left over from an earlier run (pid {pid}): {'; '.join(left[:2]) or 'it ended by itself'}")
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def run(self) -> Report:
        self._kill_orphan()
        gitbox.ensure_repo(self.root)
        self.last_state = self.run_tests()
        self.base_ok = self.last_state.ok
        self._refresh_map()
        if self.map_mode == "full" and not any(i.name == "map" for i in self.items):
            self.items.insert(0, Item("map", mapper.MAP_PROMPT, weight=0.4))
        self.note(f"night starts: {len(self.items)} prompts, {self.remaining() / 3600:.1f} h, baseline tests {'green' if self.base_ok else 'RED'}")
        for it in self.items:
            if it.status == "done" or self.remaining() < RESERVE_S + 60 or self.control() == "stop":
                continue
            if not self._ready(it):
                it.status, it.note = "partial", "waiting for " + ", ".join(it.depends)
                continue
            open_items = [x for x in self.items if x.status != "done" and x is not it or x is it]
            self._safe_work(it, min(self.clock() + self._slice(it, [x for x in open_items if x.status != "done"]), self.end - RESERVE_S))
        while self.remaining() > RESERVE_S + 5 * 60 and self.control() != "stop" and any(x.status != "done" and self._ready(x) for x in self.items):
            for it in [x for x in self.items if x.status != "done" and self._ready(x)]:
                if self.remaining() > RESERVE_S + 5 * 60 and self.control() != "stop":
                    self._safe_work(it, self.clock() + max(MIN_SLICE_S, (self.remaining() - RESERVE_S) / 2))
        return self._finish()

    def _closing_round(self) -> bool:
        """The night ends with red tests and time is left (part of the reserve): one last, narrow repair attempt on the prompt that was worked on last - the work is
        usually one failing test away from done. Kept only if the tests are green afterwards (the normal end-state rollback follows otherwise)."""
        if self.control() == "stop" or self.remaining() < CLOSING_MIN_S:
            return False
        worked = [i for i in self.items if i.attempts > 0 and i.status != "done"]
        if not worked:
            return False
        it = worked[-1]
        self.note(f"[{it.name}] closing round: the tests are red and {self.remaining() / 60:.0f} min are left - one repair attempt")
        it.attempts += 1
        r = self._execute(it, "repair", "", self.clock() + min(CLOSING_MAX_S, self.remaining() - 4 * 60), "tests_red")
        done, _ = self._commit(it, r)
        if done:
            it.status, it.note = "done", ""
            gitbox.mark_done(self.root, it.name)
            self._learn(it)
        return r.after.ok

    def _finish(self) -> Report:
        state = self.run_tests()
        if not state.ok:
            try:
                if self._closing_round():
                    state = self.run_tests()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # noqa: BLE001 - the end state must still be written
                self.note(f"closing round failed ({type(e).__name__}): {str(e)[:80]}")
        ok = state.ok
        if not ok and (gitbox.last_green(self.root) != "night/base" or self.base_ok):      # never throw work away for a baseline that was red too
            kept = gitbox.restore(self.root, gitbox.last_green(self.root))
            self.note(f"tests red at the end: rolled back to {gitbox.last_green(self.root)} (red work kept as {kept})")
            ok = self.run_tests().ok
        done = [i.name for i in self.items if i.status == "done"]
        partial = [i.name for i in self.items if i.status != "done"]
        from agent.night import report as report_mod

        text = report_mod.render(self, ok)
        if self.use_history:
            try:
                history_mod.record(attemptlog.rows(self.attempts_file), project=self.root.name, done=done, partial=partial, hours=(self.clock() - self.start) / 3600, tests_ok=ok)
            except OSError:
                pass
        (self.night_dir / "NIGHT_REPORT.md").write_text(text)
        self.save()
        archive = evidence.finalize(self, self.prompts_src)
        if archive:
            self.note(f"evidence archived: {archive}")
        return Report(done, partial, ok, text)


def _tail(path: Path, n: int = 3000) -> str:
    try:
        return path.read_text(errors="replace")[-n:]
    except OSError:
        return ""
