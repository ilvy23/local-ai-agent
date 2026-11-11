"""The evidence of a night: everything needed to EVALUATE it afterwards, written as it happens and gathered at the end.

Per attempt (`.night/evidence/<prompt>-attemptN.json`): which strategy ran and why (the failure kind that triggered it), what the harness measured (verdict, tests before and after, the
per-test failures, the guard findings, the diff per file, tool usage, steps, tokens per second, prompt peak, loops / blocked reads / forced thinking), what the model itself said at the end
(the summary box of its log, the evidence it gave to `mark_done`, the project notes it saved), the lessons and handoff it was given, and the commit it left behind. At the end `finalize()` writes
`.night/EVAL.json` and `EVAL.md` (per prompt and for the night) and an archive in the user's state folder (`~/.local/state/agent/night-evidence/<project>-<time>.tar.gz`) with the whole
`.night/` folder, the final patch against the baseline, the git log, the prompts and the unit's exit log - the project folder may be reset, the evidence stays.
Nothing in here decides anything: it is a record. Every function swallows its own errors (the record must never stop a night)."""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Any

from agent import osport
from agent.night import attemptlog, gitbox

_BOX = re.compile(r"^[│┃|]\s?(.*?)\s*[│┃|]?\s*$")
_MARK = re.compile(r"mark_done\s+(\{.*)")
_NOTE = re.compile(r"^\s+note\s{2}(.+)$", re.M)


def parse_log(path: Path, limit: int = 3000) -> dict[str, Any]:
    """What the model itself reported, read from the attempt's log: the last summary box, the evidence of its mark_done calls, the notes it saved, and a few counters."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    lines = text.splitlines()
    box: list[str] = []
    current: list[str] = []
    for ln in lines:
        if ln.startswith(("╭", "┌")):
            current = []
        elif ln.startswith(("╰", "└")):
            if current:
                box = current
        else:
            m = _BOX.match(ln) if ln.startswith(("│", "┃")) else None
            if m is not None:
                current.append(m.group(1))
    marks = [m.group(1)[:400] for ln in lines if (m := _MARK.search(ln))]
    return {"summary": "\n".join(box)[:limit], "mark_done": marks[-12:], "notes": [n.strip()[:200] for n in _NOTE.findall(text)][-20:],
            "not_run": text.count("NOT RUN"), "loop_msgs": text.count("repeated the same call") + text.count("loop detected"), "log_chars": len(text)}


def record_attempt(night: Any, it: Any, r: Any) -> Path | None:
    try:
        ev = r.ev
        data = {
            "prompt": it.name, "attempt": it.attempts, "strategy": r.strategy, "triggered_by": r.trigger or None, "wall_start": getattr(r, "wall0", 0.0), "seconds": round(r.spent, 1),
            "outcome": {"exit_code": r.out.code, "why": r.out.why or None, "left_running": r.out.reaped or []},
            "verdict": {"kind": r.verdict.kind, "detail": r.verdict.detail},
            "tests_after": {"ok": r.after.ok, "passed": r.after.passed, "failed": r.after.failed[:40], "timed_out": getattr(r.after, "timed_out", False), "digest": r.after.digest[:600]},
            "tests_before": ({"ok": night.last_state.ok, "passed": night.last_state.passed, "failed": night.last_state.failed[:40]} if night.last_state else None),
            "guard": [{"kind": f.kind, "path": f.path, "detail": f.detail, "fatal": f.fatal} for f in r.report.findings],
            "diff": {"files": [{"path": p, "added": a, "removed": d} for p, a, d in r.hand.diff.files[:80]], "added": r.hand.diff.added, "removed": r.hand.diff.removed},
            "events": {"steps": ev.steps, "reads": ev.reads, "edits": ev.edits, "tools": ev.tools, "loops": ev.loops, "blocked_reads": ev.look_blocked, "forced_thinking": ev.think_steps,
                       "loop_aborts": ev.loop_aborts, "stage_caps": ev.stage_caps, "prompt_cuts": ev.masks, "model_errors": ev.model_errors, "tokens_generated": ev.gen_tokens,
                       "decode_s": round(ev.decode_s, 1), "prefill_s": round(ev.prefill_s, 1), "tokens_per_second": round(ev.tps, 1), "peak_prompt_tokens": ev.peak_prompt},
            "task_chars": getattr(r, "task_chars", 0), "handoff_to_next": r.hand.text[:1800], "lessons_given": night.lessons.recent(it.name)[:900],
            "model_said": parse_log(night.night_dir / f"{it.name}-attempt{it.attempts}.log"),
            "head_after": gitbox.git(night.root, "rev-parse", "--short", "HEAD").stdout.strip(),
        }
        d = night.night_dir / "evidence"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{gitbox.slug(it.name)}-attempt{it.attempts}.json"
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return f
    except Exception:  # noqa: BLE001
        return None


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def summarize(night: Any) -> dict[str, Any]:
    """The night as one dict: per prompt (status, attempts, minutes, strategies, speed, guards, diff, score) and for the whole night."""
    rows = attemptlog.rows(night.attempts_file)
    ev_dir = night.night_dir / "evidence"
    records = [d for d in (_read_json(f) for f in sorted(ev_dir.glob("*.json"))) if d] if ev_dir.exists() else []
    prompts = []
    for it in night.items:
        mine = [r for r in rows if r.get("prompt") == it.name]
        recs = [d for d in records if d.get("prompt") == it.name]
        tps = [r["tps"] for r in mine if r.get("tps")]
        prompts.append({"name": it.name, "status": it.status, "attempts": it.attempts, "minutes": round(it.seconds / 60, 1), "failure_kinds": it.kinds, "strategies": [r.get("strategy") for r in mine],
                        "green": sum(1 for r in mine if r.get("green")), "tokens_per_second_mean": round(sum(tps) / len(tps), 1) if tps else None,
                        "guard_interventions": sum(r.get("guards", 0) for r in mine), "steps": sum(r.get("steps", 0) for r in mine),
                        "lines_added": sum(d["diff"]["added"] for d in recs), "lines_removed": sum(d["diff"]["removed"] for d in recs),
                        "tests_passed_last": recs[-1]["tests_after"]["passed"] if recs else None, "score": getattr(it, "scored", None), "note": it.note[:200],
                        "model_final_summary": (recs[-1]["model_said"].get("summary", "")[:700] if recs else "")})
    steps = sum(p["steps"] for p in prompts)
    return {"project": night.root.name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "hours_planned": round((night.end - night.start) / 3600, 2) if night.end else None,
            "hours_used": round((night.clock() - night.start) / 3600, 2), "prompts": prompts,
            "totals": {"prompts": len(prompts), "done": sum(1 for p in prompts if p["status"] == "done"), "attempts": sum(p["attempts"] for p in prompts), "steps": steps,
                       "guard_interventions": sum(p["guard_interventions"] for p in prompts), "flaky_tests": sorted(night.flaky)},
            "machine": night.env_lines, "lessons": night.lessons._lines()[-40:], "governor_events": night.events[-120:]}


def render_md(summary: dict[str, Any]) -> str:
    t = summary["totals"]
    lines = [f"# Evaluation of the night: {summary['project']}", "", f"{t['done']}/{t['prompts']} prompts done, {t['attempts']} attempts, {t['steps']} model steps, {summary['hours_used']} h of "
             f"{summary['hours_planned']} h, the guards of the coding agent interfered {t['guard_interventions']} times ({t['guard_interventions'] / max(t['steps'], 1) * 100:.0f}% of the steps).", "",
             "| prompt | status | attempts | min | strategies | tok/s | +/- lines | tests | score |", "|---|---|---|---|---|---|---|---|---|"]
    for p in summary["prompts"]:
        lines.append(f"| {p['name']} | {p['status']} | {p['attempts']} | {p['minutes']} | {', '.join(s or '?' for s in p['strategies'][:6])} | {p['tokens_per_second_mean'] or '-'} | "
                     f"+{p['lines_added']}/-{p['lines_removed']} | {p['tests_passed_last'] if p['tests_passed_last'] is not None else '-'} | {json.dumps(p['score'])[:60] if p['score'] else '-'} |")
    for p in summary["prompts"]:
        if p["model_final_summary"]:
            lines += ["", f"## {p['name']}: what the model said at the end", "", p["model_final_summary"]]
    if summary["lessons"]:
        lines += ["", "## Lessons recorded during the night", ""] + summary["lessons"]
    lines += ["", "## Where the raw evidence is", "", "- `.night/evidence/<prompt>-attemptN.json`: everything measured per attempt, plus what the model said",
              "- `.night/events/<prompt>-attemptN.jsonl`: the coding agent's event log of that attempt", "- `.night/<prompt>-attemptN.log`: its full console output",
              "- `.night/attempts.jsonl`, `LESSONS.md`, `NIGHT_REPORT.md`, `patches/`, and the archive in `~/.local/state/agent/night-evidence/`"]
    return "\n".join(lines) + "\n"


def archive_dir() -> Path:
    return Path(os.environ.get("NIGHT_EVIDENCE_DIR") or osport.state_home() / "agent" / "night-evidence")


def finalize(night: Any, prompts_src: list[str] | None = None) -> Path | None:
    """EVAL.json + EVAL.md in `.night/` and the archive of the whole evidence. Returns the archive path (None when it could not be written)."""
    try:
        summary = summarize(night)
        (night.night_dir / "EVAL.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (night.night_dir / "EVAL.md").write_text(render_md(summary))
        d = archive_dir()
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"{night.root.name}-{time.strftime('%Y%m%d-%H%M')}.tar.gz"
        patch = gitbox.git(night.root, "diff", "--binary", "night/base", "HEAD").stdout
        log = gitbox.git(night.root, "log", "--oneline", "--decorate=short", "night/base..HEAD").stdout
        with tarfile.open(out, "w:gz") as tar:
            for f in night.night_dir.rglob("*"):
                if f.is_file() and f.name not in ("LOCK", "attempt.pid"):
                    tar.add(f, arcname=f"night/{f.relative_to(night.night_dir)}")
            for name, text in (("final.patch", patch), ("git-log.txt", log)):
                data = text.encode("utf-8", "replace")
                info = tarfile.TarInfo(f"night/{name}")
                info.size = len(data)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(data))
            for src in prompts_src or []:
                p = Path(src)
                if p.is_dir():
                    for f in sorted(p.glob("*.md")):
                        tar.add(f, arcname=f"night/prompts/{f.name}")
                elif p.is_file():
                    tar.add(p, arcname=f"night/prompts/{p.name}")
        return out
    except Exception:  # noqa: BLE001
        return None
