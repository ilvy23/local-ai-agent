"""The verified-state block: what exists, what passes, what fails and what is open - measured by the harness, never remembered.

Seen live: a resumed session re-read the whole code base (60-89 read_file calls in a row, 84% of the time, no edit for 10+
minutes) because the model had lost every detail. A human supervisor used to write a "verified state" fact block by hand; this
module computes it from real measurements so the model can start working at once. NO model call anywhere (the reviewer is never
asked: a rubric check shows its stored verdict).

    verified.build(engine, run_tests=None) -> VerifiedState(text, sections, seconds, dropped)
    verified.render_cli(state) -> str                      the same block for /state and `agent code state`

Used on resume (session.resume_into appends the text to engine.resume_note), by `/state` and `agent code state`, and exported for
the future stage runner: call `verified.build(engine)` when a fresh-context stage starts and put `.text` into its first message.

Sections, highest priority first (the text is capped at MAX_CHARS ~ 3000 tokens; the LOWEST priority sections are dropped first):
  TESTS (plus one `plain pytest: FAILS (import path)` line when plain `pytest` cannot import the project, stdpytest.py), CHECKS, TUI SMOKE (the last crash-gate result of a terminal program, cached; only when the gate ran), STATIC, OPEN ITEMS, STAGE CURSOR (stage files: edited line ranges, where to continue), FILES SO FAR (restarted stage: what it already read / edited), DEAD ENDS (attempts repeated with the same result, loopguard.py; only when there are some), FILES, SYMBOLS, LAST STOP.
Config (coding.*): verified_tests (default true), verified_timeout_s (120), verified_on_resume (true, read by session.py)."""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.codeagent import checks as checks_mod, context, edits, outputs, static, stdpytest, symbols

MAX_CHARS = 10000                 # ~3000 tokens
HEADER = "VERIFIED STATE - measured by the harness just now, not remembered. Trust it; do not re-read files to confirm it."
TEST_TIMEOUT = 120
MAX_FAILURES, MAX_STATIC, MAX_FILES, MAX_STATIC_FILES, MAX_SYMBOL_FILES, MAX_OPEN = 8, 6, 40, 60, 12, 15
SYMBOLS_CHARS = 1500
SECTION_CAP = {"TESTS": 3200, "CHECKS": 2000, "TUI SMOKE": 900, "STATIC": 1500, "OPEN ITEMS": 2800, "STAGE CURSOR": 500, "FILES SO FAR": 700, "DEAD ENDS": 600, "FILES": 3000, "SYMBOLS": SYMBOLS_CHARS, "LAST STOP": 400}
ORDER = list(SECTION_CAP)
RESERVE = 160                     # room for the "dropped for size" line


@dataclass
class VerifiedState:
    text: str                                            # what the model sees
    sections: dict[str, dict[str, Any]]                  # name -> {"text": ..., plus the measured numbers}; for tests and logs
    seconds: float
    dropped: list[str] = field(default_factory=list)     # sections left out of `text` for size, lowest priority first


def _coding(engine: Any) -> dict[str, Any]:
    return (getattr(engine, "config", None) or {}).get("coding") or {}


def _clip(text: Any, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def _cap(lines: list[str], limit: int) -> str:
    text = "\n".join(lines)
    return text if len(text) <= limit else text[: limit - 4].rsplit("\n", 1)[0] + "\n  ..."


# ── TESTS ───────────────────────────────────────────────────────────────────

_SUMMARY = re.compile(r"\b(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b")
_HEAD = re.compile(r"^_{3,} (.+?) _{3,}$")
_FAIL_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+)(?: - (.*))?$")


def _first_assertions(out: str) -> dict[str, str]:
    """pytest section name -> its first `E ` line (the assertion)."""
    found: dict[str, str] = {}
    name = None
    for line in out.splitlines():
        m = _HEAD.match(line.rstrip())
        if m:
            name = m.group(1)
        elif name and name not in found and re.match(r"^E\s", line):
            found[name] = line[1:].strip()
    return found


def _known_red(engine: Any) -> set[str] | None:
    """Failing test ids measured before the running stage (StagePlan.baseline_red); None outside a staged run or in the last stage (strict)."""
    plan = getattr(engine, "stage_plan", None)
    red = getattr(plan, "baseline_red", None)
    if red is None:
        return None
    stages = getattr(plan, "stages", None) or []
    cur = getattr(plan, "current", 0)
    if stages and isinstance(cur, int) and cur >= len(stages) - 1 and _coding(engine).get("stage_tests_strict_last", True):
        return None
    return set(red)


def _tests(engine: Any, run: bool) -> tuple[list[str], dict[str, Any]]:
    if not run:
        return ["TESTS: tests not run"], {"status": "not run"}
    cmd = engine.profile.test_cmd
    if not cmd:
        return ["TESTS: no test command detected"], {"status": "none"}
    if "pytest" in cmd:
        cmd = re.sub(r"\s+", " ", re.sub(r"(?<=\s)-x(?=\s|$)", "", cmd)).strip()     # the default stops at the first failure; list them all
    timeout = max(1, int(_coding(engine).get("verified_timeout_s", TEST_TIMEOUT) or TEST_TIMEOUT))
    code, out, secs = engine.tools._exec(cmd, timeout)          # the engine's own path: process-group kill, output cap
    if code is None:
        if "timed out after" in out:
            return [f"TESTS: tests timed out after {timeout}s"], {"status": "timeout", "seconds": round(secs, 1)}
        return [f"TESTS: stopped, too much output ({secs:.0f}s)"], {"status": "stopped", "seconds": round(secs, 1)}
    counts: dict[str, int] = {}
    for line in reversed(out.splitlines()):                     # the last line that looks like a summary
        found = _SUMMARY.findall(line)
        if found and len(line) < 200:
            counts = {k.rstrip("s") if k.startswith("error") else k: int(n) for n, k in found}
            break
    data: dict[str, Any] = {"status": "pass" if code == 0 else "fail", "exit": code, "seconds": round(secs, 1),
                            "passed": counts.get("passed", 0), "failed": counts.get("failed", 0), "errors": counts.get("error", 0), "failing": []}
    if not counts and "no tests ran" not in out:
        tail = outputs.summarize_pytest(out, 600) if len(out) > 600 else out
        last = [ln.strip() for ln in tail.splitlines() if ln.strip()][-3:]
        return [f"TESTS: exit {code} in {secs:.0f}s, no pass/fail count found"] + [f"  {_clip(ln, 160)}" for ln in last], data
    head = f"TESTS: {data['passed']} passed, {data['failed']} failed" + (f", {data['errors']} errors" if data["errors"] else "") + f" ({secs:.0f}s)"
    if not counts:
        head = f"TESTS: no tests collected (exit {code}, {secs:.0f}s)"
    lines = [head]
    asserts = _first_assertions(out)
    seen: list[str] = []
    for line in out.splitlines():
        m = _FAIL_LINE.match(line.strip())
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
            parts = m.group(1).split("::")
            why = asserts.get(parts[-1]) or asserts.get(".".join(parts[1:])) or m.group(2) or ""
            data["failing"].append(m.group(1))
            if len(seen) <= MAX_FAILURES:
                lines.append(f"  {_clip(m.group(1), 120)}" + (f" - {_clip(why, 170)}" if why else ""))
    if len(seen) > MAX_FAILURES:
        lines.append(f"  ... and {len(seen) - MAX_FAILURES} more failing")
    known = _known_red(engine)
    if known is not None and seen:                                  # a running multi-part plan: failures from before this stage are not this stage's job
        old_ids = [i for i in seen if i in known]
        if len(old_ids) == len(seen):
            lines[0] += f" - all {len(seen)} were already red before this stage; nothing new broke, they are not part of this stage"
            lines[1:] = [ln + " (already red)" if ln.startswith("  ") and "more failing" not in ln else ln for ln in lines[1:]]
        elif old_ids:
            lines.append(f"  ({len(old_ids)} of these were already red before this stage and are not part of it)")
    if code != 0 and len(lines) == 1 and counts:
        lines.append(f"  (exit {code}; no FAILED lines in the output)")
    std = getattr(engine, "std_pytest_now", None)
    try:
        found = std() if callable(std) and "-m pytest" in cmd else None      # the standard way (stdpytest.py): cached per code state
    except Exception:  # noqa: BLE001 - a bonus line, never a crash
        found = None
    if found is not None:
        lines.append(f"  {stdpytest.LINE}: {found.module} is not importable without python -m pytest")
        data["plain_pytest"] = "fails"
    return lines, data


def run_tests(engine: Any) -> tuple[list[str], dict[str, Any]]:
    """The test run of `build` on its own (same command, same coding.verified_timeout_s): (TESTS lines, data). The stage runner uses it."""
    return _tests(engine, True)


# ── CHECKS ──────────────────────────────────────────────────────────────────

def _first_line(res: dict[str, Any]) -> str:
    lines = [ln.strip() for ln in str(res.get("output") or "").splitlines() if ln.strip()]
    for ln in lines:
        if not ln.startswith("$ ") and not re.match(r"^exit (\d+|timeout)\b", ln):
            return _clip(ln, 150)
    return _clip(next((ln for ln in lines if ln.startswith("exit")), ""), 150)


def _checks(engine: Any, run: bool = True) -> tuple[list[str], dict[str, Any]]:
    try:
        every = engine.checks.all()
    except Exception as exc:  # noqa: BLE001 - a broken checks folder must not hide the other facts
        return [f"CHECKS: could not be loaded ({type(exc).__name__})"], {"status": "error"}
    if not every:
        return ["CHECKS: none defined"], {"status": "none", "results": {}}
    runnable = [c for c in every if c.kind != "rubric"]
    if runnable and run:
        try:
            engine.run_checks(runnable, quiet=True)              # the same runner the final gate uses (never the reviewer)
        except Exception:  # noqa: BLE001
            pass
    lines, results = ["CHECKS:"], {}
    for c in every:
        r = c.last_result or {}
        status = str(r.get("status") or "not run")
        results[c.id] = status
        if c.kind == "rubric":
            why = f" - {_first_line(r)}" if status != "pass" and r.get("output") else ""
            lines.append(f"  {c.id}: {status.upper() if r else 'NOT JUDGED YET'} (rubric, stored verdict, not re-judged){why}")
        elif status == "pass":
            lines.append(f"  {c.id}: PASS")
        elif status in ("skipped", "unreviewed", "not run"):
            lines.append(f"  {c.id}: {status.upper()} - {_first_line(r) or c.description}")
        else:
            lines.append(f"  {c.id}: FAIL - {_first_line(r) or status}")
    return lines, {"status": "done", "results": results}


# ── TUI SMOKE ───────────────────────────────────────────────────────────────

def _tui_smoke(engine: Any, run: bool = False) -> tuple[list[str], dict[str, Any]]:
    """The last result of the harness' crash gate for terminal programs (tuigate.py), as the gate measured it: NOT re-run here unless `run`.
    [] when the gate never ran in this session (no terminal program, or nothing edited)."""
    last = getattr(engine, "tui_last", None)
    if run:
        try:
            last = engine.tui_gate_now()
        except Exception:  # noqa: BLE001 - a broken gate must not hide the other facts
            pass
    if last is None:
        return [], {}
    lines = [_clip(last.line, 360)] + ([f"  {_clip(ln, 150)}" for ln in last.detail.splitlines()[:6]] if last.status == "fail" else [])
    return lines, {"status": last.status, "entry": last.entry, "crash_line": last.crash_line}


# ── the files the sections look at ──────────────────────────────────────────

def _files(engine: Any) -> list[tuple[str, str, int, float]]:
    """(relative path, last action, step, mtime) of the files that exist on disk, newest change first. The session ledger when it
    has files, else the code files in the folder (a folder the agent never worked in: `agent code state`)."""
    root = Path(engine.root)
    rows: list[tuple[str, str, int, float]] = []
    ledger = dict(engine.ctx.ledger.files)
    if ledger:
        for rel, e in ledger.items():
            f = Path(rel) if Path(rel).is_absolute() else root / rel
            try:
                if f.is_file():
                    rows.append((rel, e.action, e.step, f.stat().st_mtime))
            except OSError:
                continue
    else:
        for f in edits.iter_files(root, 400):
            try:
                rows.append((str(f.relative_to(root)), "on disk", 0, f.stat().st_mtime))
            except (OSError, ValueError):
                continue
    return sorted(rows, key=lambda r: (-r[3], r[0]))


def _read(root: Path, rel: str) -> str:
    f = Path(rel) if Path(rel).is_absolute() else root / rel
    try:
        return f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ── STATIC ──────────────────────────────────────────────────────────────────

def _static(engine: Any, rows: list[tuple[str, str, int, float]]) -> tuple[list[str], dict[str, Any]]:
    root = Path(engine.root)
    py = [r[0] for r in rows if r[0].endswith(".py")][:MAX_STATIC_FILES]
    problems: list[str] = []
    for rel in py:
        try:
            ast.parse(_read(root, rel), filename=rel)
        except SyntaxError as exc:
            problems.append(f"{rel}:{exc.lineno or 1} SyntaxError: {exc.msg}")
    try:
        found = static.check_files(root, py)
    except Exception:  # noqa: BLE001 - the static pass is a bonus, never a crash
        found = {}
    for rel, probs in found.items():
        for p in probs:
            m = re.match(r"line (\d+): (.*)", p)
            problems.append(f"{rel}:{m.group(1)} {m.group(2)}" if m else f"{rel} {p}")
    data = {"files": len(py), "problems": len(problems)}
    if not py:
        return ["STATIC: no .py files to check"], data
    if not problems:
        return [f"STATIC: {len(py)} .py files checked, no problems"], data
    lines = [f"STATIC: {len(problems)} problems in {len(py)} .py files checked"] + [f"  {_clip(p, 190)}" for p in problems[:MAX_STATIC]]
    if len(problems) > MAX_STATIC:
        lines.append(f"  ... and {len(problems) - MAX_STATIC} more")
    return lines, data


# ── OPEN ACCEPTANCE ITEMS ───────────────────────────────────────────────────

def _open_items(engine: Any) -> tuple[list[str], dict[str, Any]]:
    plan = engine.plan.plan
    if plan is None or not plan.items:
        return ["OPEN ACCEPTANCE ITEMS: no acceptance list"], {"open": [], "confirmed": 0, "total": 0}
    left = plan.open_items()
    head = f"OPEN ACCEPTANCE ITEMS: {plan.confirmed()} of {len(plan.items)} confirmed" + ("" if left else " - none open")
    lines = [head] + [f"  {i.id}  {_clip(i.text, 170)}" for i in left[:MAX_OPEN]]
    if len(left) > MAX_OPEN:
        lines.append(f"  ... and {len(left) - MAX_OPEN} more open")
    return lines, {"open": [i.id for i in left], "confirmed": plan.confirmed(), "total": len(plan.items)}


# ── STAGE CURSOR ────────────────────────────────────────────────────────────

def _stage_cursor(engine: Any) -> tuple[list[str], dict[str, Any]]:
    """Per file of the running stage: the line ranges it edited so far and where to continue (cursor.py). [] outside a stage / without files."""
    from agent.codeagent import cursor

    st = getattr(engine, "stage_obj", None)
    cur = cursor.cursor(st, engine.root) if st is not None else {}
    if not cur:
        return [], {"files": 0}
    rows = ["STAGE CURSOR (line ranges this stage already edited; continue after them, do not re-read them):"] + [f"  {ln}" for ln in cursor.lines(cur)]
    return _cap(rows, SECTION_CAP["STAGE CURSOR"]).split("\n"), {"files": len(cur), "cursor": cur}


def _files_so_far(engine: Any) -> tuple[list[str], dict[str, Any]]:
    """What the running stage already read and edited, per file, from its tool calls (fileseen.py). [] outside a stage / before anything was read."""
    from agent.codeagent import fileseen

    st = getattr(engine, "stage_obj", None)
    lines = fileseen.block(getattr(st, "seen", None) or {}, getattr(st, "edited", None)) if st is not None else []
    return lines, {"files": max(len(lines) - 2, 0)} if lines else {"files": 0}


# ── DEAD ENDS ───────────────────────────────────────────────────────────────

def _dead_ends(engine: Any) -> tuple[list[str], dict[str, Any]]:
    """What the loop guard saw repeated with the same result: a fresh context must not walk into it again. [] when nothing."""
    ledger = getattr(engine, "dead_ends", None)
    lines = ledger.lines() if ledger is not None else []
    return lines, {"count": len(ledger.items) if ledger is not None else 0}


# ── FILES, SYMBOLS ──────────────────────────────────────────────────────────

def _file_list(engine: Any, rows: list[tuple[str, str, int, float]]) -> tuple[list[str], dict[str, Any]]:
    root = Path(engine.root)
    if not rows:
        return ["FILES: none yet"], {"count": 0}
    lines = [f"FILES ({len(rows)}, newest change first):"]
    for rel, action, step, _mt in rows[:MAX_FILES]:
        n = len(_read(root, rel).splitlines())
        lines.append(f"  {_clip(rel, 100)}  {n} lines  {action}" + (f" (step {step})" if step else ""))
    if len(rows) > MAX_FILES:
        lines.append(f"  ... and {len(rows) - MAX_FILES} more")
    return lines, {"count": len(rows), "listed": min(len(rows), MAX_FILES)}


def _symbol_map(engine: Any, rows: list[tuple[str, str, int, float]]) -> tuple[list[str], dict[str, Any]]:
    """Top-level defs and classes of the most recently changed code files, grouped per file (`path  name:line ...`) - compact."""
    root = Path(engine.root)
    lines = ["SYMBOL MAP (name:line):"]
    size, n = len(lines[0]), 0
    for rel, _a, _s, _mt in rows[:MAX_SYMBOL_FILES]:
        if not rel.endswith(".py"):
            continue
        names = [f"{s.name}:{s.start}" for s in symbols.python_spans(_read(root, rel)) if "." not in s.name and s.kind in ("def", "class")]
        if not names:
            continue
        line = f"  {_clip(rel, 70)}  " + " ".join(names[:14]) + (f" +{len(names) - 14}" if len(names) > 14 else "")
        if size + len(line) + 1 > SYMBOLS_CHARS:
            lines.append("  ...")
            break
        lines.append(line)
        size += len(line) + 1
        n += 1
    return (lines if n else []), {"files": n}


# ── LAST STOP ───────────────────────────────────────────────────────────────

def _last_stop(engine: Any) -> tuple[list[str], dict[str, Any]]:
    stopped = str(getattr(engine, "last_stopped", "") or "")
    requests = list(getattr(engine.ctx, "user_requests", []) or [])
    if not requests and not stopped:
        return ["LAST STOP: no earlier prompt in this session"], {"stopped": ""}
    if stopped == "killed":
        why = "the last prompt was cut off mid-step (the process was killed); its last step may be partial"
    elif stopped == "interrupted":
        why = "the last prompt was interrupted before it finished"
    elif stopped:
        why = f"the last prompt ended early ({_clip(stopped, 80)})"
    else:
        why = "the last prompt finished normally"
    task = f"; last request: {_clip(requests[-1], 140)}" if requests else ""
    return [f"LAST STOP: {why}{task}"], {"stopped": stopped}


# ── assembling ──────────────────────────────────────────────────────────────

def fit(sections: list[tuple[str, str]], budget: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Keep the sections (priority order) that fit `budget` characters: the lowest priority ones are dropped first. The names
    dropped come back lowest priority first. The first section always stays (cut to the budget when it alone is too long)."""
    kept, dropped = list(sections), []
    while len(kept) > 1 and sum(len(t) + 1 for _n, t in kept) > budget:
        dropped.append(kept.pop()[0])
    if kept and sum(len(t) + 1 for _n, t in kept) > budget:
        name, text = kept[0]
        kept[0] = (name, text[: max(budget - 4, 0)].rsplit("\n", 1)[0] + "\n  ...")
    return kept, dropped


def _clean(text: str) -> str:
    """Nothing in the block may look like one of the harness' own markers (the model copies what it sees)."""
    for _name, rx in context.MARKERS:
        text = rx.sub("(trimmed)", text)
    return text


def build(engine: Any, *, run_tests: bool | None = None, run_checks: bool = True, run_tui: bool = False) -> VerifiedState:
    """Measure the project now: no model call. `run_tests=None` follows config coding.verified_tests (default true). `run_checks=False` shows
    the checks' last known results without running them (the forced ending of engine.py reuses what is known). The TUI SMOKE line is the gate's
    cached result; `run_tui=True` plays the program again."""
    t0 = time.monotonic()
    if run_tests is None:
        run_tests = bool(_coding(engine).get("verified_tests", True))
    rows = _files(engine)
    parts: dict[str, tuple[list[str], dict[str, Any]]] = {
        "TESTS": _tests(engine, run_tests),
        "CHECKS": _checks(engine, run_checks),
        "TUI SMOKE": _tui_smoke(engine, run_tui),
        "STATIC": _static(engine, rows),
        "OPEN ITEMS": _open_items(engine),
        "STAGE CURSOR": _stage_cursor(engine),
        "FILES SO FAR": _files_so_far(engine),
        "DEAD ENDS": _dead_ends(engine),
        "FILES": _file_list(engine, rows),
        "SYMBOLS": _symbol_map(engine, rows),
        "LAST STOP": _last_stop(engine),
    }
    texts = [(name, _clean(_cap(lines, SECTION_CAP[name]))) for name, (lines, _d) in parts.items() if lines]
    kept, dropped = fit(texts, MAX_CHARS - len(HEADER) - RESERVE)
    body = [HEADER] + [t for _n, t in kept]
    if dropped:
        body.append("(left out for size: " + ", ".join(dropped) + " - use list_dir / find_symbol if you need them)")
    sections = {name: {"text": "\n".join(lines), **data} for name, (lines, data) in parts.items() if lines or name not in ("DEAD ENDS", "TUI SMOKE")}
    return VerifiedState("\n".join(body)[:MAX_CHARS], sections, round(time.monotonic() - t0, 2), dropped)


def render_cli(state: VerifiedState) -> str:
    return f"{state.text}\n(measured in {state.seconds:.1f}s, no model used)"
