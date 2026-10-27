"""Speed trial of the staged autopilot: runs `agent code --autopilot --stages` on a folder with a small stage prompt cap, then prints the
measured time breakdown of THAT run (agent.codeagent.timing) and a PASS / FAIL against thresholds.

    python3 scripts/speed_trial.py ~/Documents/some-project "add a /health route and test it" --cap 16000
    python3 scripts/speed_trial.py --dry-run ~/.local/state/agent/code-events.jsonl [--session ID]     # only parse an events file, run nothing

Real run (it uses the model and the GPU - only start it when they are free): the event log goes to a file of its own (--events, default a
temp file), the cap is passed as AGENT_CODE_STAGE_PROMPT_CAP for that process only. Thresholds (flags): prefill share < 25 %, and
steps/min >= 6 - judged only when the peak prompt stayed <= 30k tokens (a bigger prompt is a different problem, shown as 'not judged').
Exit code: 0 PASS, 1 FAIL, 2 could not run / read."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from agent.codeagent import timing


def build_command(folder: Path | str, task: str, *, cap: int, events: Path | str, base_env: dict[str, str] | None = None) -> tuple[list[str], dict[str, str]]:
    cmd = ["agent", "code", "--autopilot", "--stages", "--folder", str(folder), task]
    env = {**(os.environ if base_env is None else base_env), "AGENT_CODE_EVENTS": str(events), "AGENT_CODE_STAGE_PROMPT_CAP": str(cap)}
    return cmd, env


def verdict(b: timing.Breakdown, max_prefill: float, min_steps_per_min: float, peak_limit: int = 30000) -> tuple[bool, list[str]]:
    """(passed, lines) - every line starts with PASS, FAIL or 'not judged'. A run without a single measured step cannot pass."""
    if b.steps == 0:
        return False, ["FAIL no model steps were measured"]
    lines, ok = [], True
    if b.prefill_share < max_prefill:
        lines.append(f"PASS prefill share {b.prefill_share * 100:.0f}% < {max_prefill * 100:.0f}%")
    else:
        ok = False
        lines.append(f"FAIL prefill share {b.prefill_share * 100:.0f}% >= {max_prefill * 100:.0f}%")
    if b.peak_prompt > peak_limit:
        lines.append(f"not judged: steps/min {b.steps_per_min:.1f} (peak prompt {b.peak_prompt / 1000:.0f}k > {peak_limit / 1000:.0f}k tokens)")
    elif b.steps_per_min >= min_steps_per_min:
        lines.append(f"PASS steps/min {b.steps_per_min:.1f} >= {min_steps_per_min:g}")
    else:
        ok = False
        lines.append(f"FAIL steps/min {b.steps_per_min:.1f} < {min_steps_per_min:g}")
    return ok, lines


def report(path: Path, session: str | None, args: argparse.Namespace) -> int:
    events = timing.select(timing.read_log(path))
    if session:
        events = timing.select(events, session=session)
    elif events:
        latest = events[-1].get("session")
        events = [e for e in events if e.get("session") == latest]
    if not events:
        print(f"no events found in {path}", file=sys.stderr)
        return 2
    b = timing.breakdown(events)
    print(timing.render(b))
    ok, lines = verdict(b, args.max_prefill, args.min_steps_per_min, args.peak_limit)
    print("\n".join(lines))
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="project folder to run in")
    ap.add_argument("task", nargs="?", help="the task text")
    ap.add_argument("--dry-run", metavar="EVENTS_FILE", help="only parse this events file and judge it; run nothing")
    ap.add_argument("--session", help="event-log session (id or prefix); default: the latest one in the file")
    ap.add_argument("--cap", type=int, default=16000, help="coding.stage_prompt_cap for this run (default 16000)")
    ap.add_argument("--events", help="event log file of the run (default: a temp file)")
    ap.add_argument("--max-prefill", type=float, default=0.25, help="PASS needs a prefill share below this (default 0.25)")
    ap.add_argument("--min-steps-per-min", type=float, default=6.0, help="PASS needs at least this many steps/min (default 6)")
    ap.add_argument("--peak-limit", type=int, default=30000, help="steps/min is judged only while the peak prompt is at most this (default 30000)")
    args = ap.parse_args(argv)
    if args.dry_run:
        path = Path(args.dry_run).expanduser()
        if not path.is_file():
            print(f"not a file: {path}", file=sys.stderr)
            return 2
        return report(path, args.session, args)
    if not args.folder or not args.task:
        ap.error("folder and task are required (or use --dry-run EVENTS_FILE)")
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    events = Path(args.events).expanduser() if args.events else Path(tempfile.mkdtemp(prefix="speed-trial-")) / "events.jsonl"
    cmd, env = build_command(folder, args.task, cap=args.cap, events=events)
    print("running:", " ".join(cmd[:6]), f"... (stage_prompt_cap {args.cap}, events {events})", flush=True)
    try:
        code = subprocess.run(cmd, env=env).returncode
    except OSError as exc:
        print(f"could not start the agent: {exc}", file=sys.stderr)
        return 2
    print(f"agent exit code {code} (0 done, 2 needs a human, 3 partial)")
    return report(events, None, args)


if __name__ == "__main__":
    sys.exit(main())
