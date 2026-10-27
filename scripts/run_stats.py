#!/usr/bin/env python3
"""Run statistics for `agent code` (stdlib only, read-only).

Reads the events log (default ~/.local/state/agent/code-events.jsonl) and the
saved sessions (~/.local/state/agent/code-sessions/<folder-hash>/<id>.json +
<id>.meta.json) and prints per-session and overall numbers.

    python scripts/run_stats.py                 # table for every session
    python scripts/run_stats.py --session ced9  # substring match on session name
    python scripts/run_stats.py --since 24h     # only events newer than 24 h (s/m/h/d)
    python scripts/run_stats.py --json          # machine readable

Definitions (all measured from the log, nothing estimated):
  wall      sum of gaps between consecutive events, gaps > --idle-gap (300 s) excluded
  idle      sum of the excluded gaps
  prefill / decode / load   sums of the step fields prefill_s / decode_s / load_s
  tool      sum of tool `seconds`
  compact   model time of `compact_call` events
  other     wall - (the above), floored at 0 (harness overhead, waits shorter than
            the idle gap, untimed model calls)
  The first model call finishes at the first event, so its time is added to wall.
  reads     read_file read_files search list_dir show_symbol repo_map outline where_used
            git_status web_search
  edits     edit_file replace_lines append_file write_file apply_patch delete_file
  runs      run run_tests
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

STATE = Path.home() / ".local" / "state" / "agent"
READS = {"read_file", "read_files", "search", "list_dir", "show_symbol", "repo_map",
         "outline", "where_used", "git_status", "web_search"}
EDITS = {"edit_file", "replace_lines", "append_file", "write_file", "apply_patch", "delete_file"}
RUNS = {"run", "run_tests"}


def parse_since(text: str | None) -> float | None:
    if not text:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", text.strip())
    if not m:
        raise SystemExit(f"bad --since value {text!r} (use e.g. 90m, 24h, 7d)")
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    return time.time() - float(m.group(1)) * mult


def load_events(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and "ts" in d and "session" in d:
                out.append(d)
    out.sort(key=lambda e: e["ts"])
    return out


def load_metas(sessions_dir: Path) -> dict[str, dict]:
    """Saved-session metadata by session id. Adds declined count from the body."""
    metas: dict[str, dict] = {}
    if not sessions_dir.is_dir():
        return metas
    for mp in sessions_dir.glob("*/*.meta.json"):
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sid = meta.get("id") or mp.name.split(".")[0]
        body = mp.with_name(sid + ".json")
        meta["declined"] = None
        meta["messages_saved"] = None
        if body.exists():
            try:
                d = json.loads(body.read_text(encoding="utf-8"))
                es = d.get("engine_state") or {}
                meta["declined"] = len(es.get("declined") or [])
                meta["messages_saved"] = len(d.get("messages") or [])
            except (OSError, ValueError):
                pass
        metas[sid] = meta
    return metas


def meta_for(session: str, metas: dict[str, dict]) -> dict | None:
    tail = session.rsplit("-", 1)[-1]
    return metas.get(tail)


def pct(a: float, b: float) -> float:
    return round(100.0 * a / b, 1) if b > 0 else 0.0


def analyse(session: str, evs: list[dict], metas: dict[str, dict], idle_gap: float) -> dict:
    evs = sorted(evs, key=lambda e: e["ts"])
    wall = idle = 0.0
    idle_gaps = 0
    for a, b in zip(evs, evs[1:]):
        gap = b["ts"] - a["ts"]
        # a long gap that ends in a step/compact_call is mostly that model call, not idle
        own = 0.0
        if b["kind"] in ("step", "compact_call"):
            own = (b.get("prefill_s") or 0) + (b.get("decode_s") or 0) + (b.get("load_s") or 0)
        if gap - own > idle_gap:
            idle += gap - own
            wall += min(own, gap)
            idle_gaps += 1
        else:
            wall += gap
    steps = [e for e in evs if e["kind"] == "step"]
    if steps and evs[0]["kind"] == "step":
        # the first model call finished at the first event; its time precedes it
        wall += (steps[0].get("prefill_s") or 0) + (steps[0].get("decode_s") or 0) + (steps[0].get("load_s") or 0)
    compact_s = sum((e.get("prefill_s") or 0) + (e.get("decode_s") or 0)
                    for e in evs if e["kind"] == "compact_call")
    tools = [e for e in evs if e["kind"] == "tool"]
    prompts = [e["prompt"] for e in steps if isinstance(e.get("prompt"), (int, float))]
    prefill = sum(e.get("prefill_s") or 0 for e in steps)
    decode = sum(e.get("decode_s") or 0 for e in steps)
    load = sum(e.get("load_s") or 0 for e in steps)
    tool_s = sum(e.get("seconds") or 0 for e in tools)
    other = max(0.0, wall - prefill - decode - load - tool_s - compact_s)
    mix = collections.Counter(e.get("name", "?") for e in tools)
    reads = sum(n for t, n in mix.items() if t in READS)
    edits = sum(n for t, n in mix.items() if t in EDITS)
    runs = sum(n for t, n in mix.items() if t in RUNS)
    kinds = collections.Counter(e["kind"] for e in evs)
    loops = collections.Counter(e.get("tool", "?") for e in evs if e["kind"] == "loop")
    aborts = collections.Counter(e.get("tool", "?") for e in evs if e["kind"] == "loop_abort")
    blocked = collections.Counter(e.get("tool", "?") for e in evs if e["kind"] == "look_blocked")
    end = [e for e in evs if e["kind"] == "autopilot_end"]
    meta = meta_for(session, metas)
    stage_done = kinds["stage_done"]
    if end:
        outcome = "finished" if end[-1].get("verdict") == "done" else f"ended:{end[-1].get('verdict')}"
    elif aborts:
        outcome = "interrupted (loop_abort)"
    elif meta and meta.get("state") == "finished":
        outcome = "finished"
    elif meta and meta.get("state") == "interrupted":
        outcome = f"interrupted ({meta.get('stopped') or '?'})"
    elif meta and meta.get("state") == "in progress":
        outcome = f"unfinished ({meta.get('stopped') or 'in progress'})"
    else:
        outcome = "unknown (no end event, no saved session)"
    return {
        "session": session,
        "title": (meta or {}).get("title"),
        "model": (meta or {}).get("model"),
        "start": evs[0]["ts"],
        "end": evs[-1]["ts"],
        "events": len(evs),
        "wall_s": round(wall, 1),
        "idle_s": round(idle, 1),
        "idle_gaps": idle_gaps,
        "steps": len(steps),
        "steps_per_min": round(len(steps) / (wall / 60), 2) if wall > 0 else 0.0,
        "prompt_mean": round(sum(prompts) / len(prompts)) if prompts else 0,
        "prompt_peak": max(prompts) if prompts else 0,
        "gen_tokens": sum(e.get("gen") or 0 for e in steps),
        "think_steps": sum(1 for e in steps if e.get("think")),
        "prefill_s": round(prefill, 1),
        "decode_s": round(decode, 1),
        "load_s": round(load, 1),
        "tool_s": round(tool_s, 1),
        "compact_s": round(compact_s, 1),
        "other_s": round(other, 1),
        "share_pct": {
            "prefill": pct(prefill, wall), "decode": pct(decode, wall), "load": pct(load, wall),
            "tool": pct(tool_s, wall), "compact": pct(compact_s, wall), "other": pct(other, wall),
            "idle_of_elapsed": pct(idle, wall + idle),
        },
        "tool_calls": len(tools),
        "tool_failed": sum(1 for e in tools if e.get("ok") is False),
        "tool_mix": dict(mix.most_common()),
        "reads": reads, "edits": edits, "runs": runs,
        "read_edit_ratio": round(reads / edits, 2) if edits else None,
        "loops": sum(loops.values()), "loops_by_tool": dict(loops),
        "loop_aborts": sum(aborts.values()), "aborts_by_tool": dict(aborts),
        "look_blocked": sum(blocked.values()), "blocked_by_tool": dict(blocked),
        "declined_saved": (meta or {}).get("declined"),
        "wander": kinds["wander"], "paging": kinds["paging"], "edit_gap": kinds["edit_gap"],
        "collapse": kinds["collapse"], "compact": kinds["compact"],
        "fresh_context": kinds["fresh_context"],
        "malformed": kinds["malformed_call"], "salvaged": kinds["salvaged_call"],
        "model_error": kinds["model_error"], "cut_off": kinds["cut_off"],
        "degenerate": kinds["degenerate"],
        "stages_started": kinds["stage_start"], "stage_done": stage_done,
        "stage_cap": kinds["stage_cap"], "stage_resplit": kinds["stage_resplit"],
        "autopilot_decisions": dict(collections.Counter(
            f'{e.get("reason")}->{e.get("action")}' for e in evs if e["kind"] == "autopilot_decision")),
        "outcome": outcome,
        "saved_state": (meta or {}).get("state"),
    }


def overall(rows: list[dict]) -> dict:
    tot = lambda k: sum(r[k] for r in rows)  # noqa: E731
    wall = tot("wall_s")
    mix: collections.Counter = collections.Counter()
    loops: collections.Counter = collections.Counter()
    aborts: collections.Counter = collections.Counter()
    for r in rows:
        mix.update(r["tool_mix"])
        loops.update(r["loops_by_tool"])
        aborts.update(r["aborts_by_tool"])
    reads, edits, runs = tot("reads"), tot("edits"), tot("runs")
    return {
        "sessions": len(rows), "wall_s": round(wall, 1), "idle_s": round(tot("idle_s"), 1),
        "steps": tot("steps"),
        "steps_per_min": round(tot("steps") / (wall / 60), 2) if wall else 0.0,
        "prefill_pct": pct(tot("prefill_s"), wall), "decode_pct": pct(tot("decode_s"), wall),
        "load_pct": pct(tot("load_s"), wall), "tool_pct": pct(tot("tool_s"), wall),
        "compact_pct": pct(tot("compact_s"), wall), "other_pct": pct(tot("other_s"), wall),
        "reads": reads, "edits": edits, "runs": runs,
        "read_edit_ratio": round(reads / edits, 2) if edits else None,
        "tool_mix": dict(mix.most_common()),
        "loops": tot("loops"), "loops_by_tool": dict(loops.most_common()),
        "loop_aborts": tot("loop_aborts"), "aborts_by_tool": dict(aborts.most_common()),
        "look_blocked": tot("look_blocked"),
        "stage_cap": tot("stage_cap"), "stage_resplit": tot("stage_resplit"),
        "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
    }


def fmt_dur(s: float) -> str:
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def render(rows: list[dict], tot: dict) -> str:
    out = []
    out.append(f"{len(rows)} session(s)\n")
    for r in rows:
        sh = r["share_pct"]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["start"]))
        out.append(f"== {r['session']}  [{when}]  {r['outcome']}")
        if r["title"]:
            out.append(f"   {r['title']}")
        out.append(
            f"   wall {fmt_dur(r['wall_s'])} (idle excluded {fmt_dur(r['idle_s'])} in {r['idle_gaps']} gaps) | "
            f"steps {r['steps']} ({r['steps_per_min']}/min) | prompt mean {r['prompt_mean']} peak {r['prompt_peak']}")
        out.append(
            f"   time share: prefill {sh['prefill']}%  decode {sh['decode']}%  load {sh['load']}%  "
            f"tool {sh['tool']}%  compact {sh['compact']}%  other {sh['other']}%  | idle {sh['idle_of_elapsed']}% of elapsed")
        out.append(
            f"   tools {r['tool_calls']} (failed {r['tool_failed']}): reads {r['reads']} edits {r['edits']} "
            f"runs {r['runs']} read:edit {r['read_edit_ratio']}")
        out.append("   mix: " + ", ".join(f"{k} {v}" for k, v in r["tool_mix"].items()))
        out.append(
            f"   loops {r['loops']} {r['loops_by_tool']} | aborts {r['loop_aborts']} {r['aborts_by_tool']} | "
            f"look_blocked {r['look_blocked']} {r['blocked_by_tool']} | declined(saved) {r['declined_saved']}")
        out.append(
            f"   wander {r['wander']} paging {r['paging']} edit_gap {r['edit_gap']} collapse {r['collapse']} "
            f"compact {r['compact']} fresh_ctx {r['fresh_context']} | stages {r['stage_done']}/{r['stages_started']} done, "
            f"cap {r['stage_cap']}, resplit {r['stage_resplit']}")
        out.append("")
    out.append("== OVERALL")
    out.append(
        f"   wall {fmt_dur(tot['wall_s'])} | steps {tot['steps']} ({tot['steps_per_min']}/min) | "
        f"prefill {tot['prefill_pct']}% decode {tot['decode_pct']}% load {tot['load_pct']}% "
        f"tool {tot['tool_pct']}% compact {tot['compact_pct']}% other {tot['other_pct']}%")
    out.append(f"   reads {tot['reads']} edits {tot['edits']} runs {tot['runs']} read:edit {tot['read_edit_ratio']}")
    out.append("   mix: " + ", ".join(f"{k} {v}" for k, v in tot["tool_mix"].items()))
    out.append(f"   loops {tot['loops']} {tot['loops_by_tool']} | aborts {tot['loop_aborts']} {tot['aborts_by_tool']} | "
               f"look_blocked {tot['look_blocked']} | stage_cap {tot['stage_cap']} resplit {tot['stage_resplit']}")
    out.append(f"   outcomes: {tot['outcomes']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--events", type=Path, default=STATE / "code-events.jsonl")
    ap.add_argument("--sessions-dir", type=Path, default=STATE / "code-sessions")
    ap.add_argument("--session", help="substring of the session name (e.g. ced91526)")
    ap.add_argument("--since", help="only events newer than this age: 30m, 24h, 7d")
    ap.add_argument("--idle-gap", type=float, default=300.0, help="seconds; larger gaps count as idle")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    events = load_events(args.events)
    cutoff = parse_since(args.since)
    if cutoff is not None:
        events = [e for e in events if e["ts"] >= cutoff]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for e in events:
        if args.session and args.session not in e["session"]:
            continue
        by[e["session"]].append(e)
    metas = load_metas(args.sessions_dir)
    rows = [analyse(s, v, metas, args.idle_gap) for s, v in by.items()]
    rows.sort(key=lambda r: r["start"])
    tot = overall(rows)
    if args.json:
        json.dump({"sessions": rows, "overall": tot}, sys.stdout, indent=2)
        print()
    elif not rows:
        print("no matching events")
    else:
        print(render(rows, tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
