"""Where the wall time of a coding run went - measured from the event log (eventlog.py), never guessed.

    breakdown(events, since=None, session=None) -> Breakdown   prefill / generation / tools / waiting / other, steps per minute, prompt sizes
    why_slow(breakdown) -> list[str]                          up to 3 rule-based causes WITH the measured numbers, or 'no obvious bottleneck'
    time_cli(session, since)                                  `agent code time [--session ID] [--since 2h]`; the REPL has `/why-slow`

Sources (events the engine already writes): `step` (prompt tokens, prefill_s, decode_s, load_s, think) and `compact_call` (model time),
`tool` (seconds, name, cmd of a run) and `check` (seconds), `server_wait` / `autopilot_server_wait` (seconds), `stage_start`, `segment_start`.
The wall clock is first event minus its own duration .. last event; whatever no event accounts for is 'other' (the model's own overhead, the
harness between steps, a suspended PC). A field that an older log does not have counts as 0 / is skipped."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from agent.codeagent import eventlog
from agent.codeagent.lookact import LOOK_TOOLS, readonly_shell

PREFILL_SHARE, BIG_PROMPT = 0.25, 30000        # prefill above 25 % of the wall with a mean prompt above 30k tokens: the stage cap is the lever
MIN_STEPS_PER_MIN = 6.0                        # measured on a healthy staged run: 7-9
MIN_STEPS_FOR_RATE = 3                         # fewer steps say nothing about a rate
LOOK_SHARE, MIN_TOOL_CALLS = 0.5, 10
TOOL_SHARE, TEST_SHARE = 0.3, 0.5
THINK_SHARE, WAIT_SHARE = 0.3, 0.15
_TEST_RUN = re.compile(r"pytest|unittest|\btest\b|nose|tox|cargo test|go test|npm (?:run )?test", re.I)
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


@dataclass
class Breakdown:
    wall_s: float = 0.0
    prefill_s: float = 0.0
    gen_s: float = 0.0
    tool_s: float = 0.0                # tool execution incl. the harness's own checks
    wait_s: float = 0.0                # model server not reachable + model (re)loading
    test_s: float = 0.0                # the part of tool_s spent in test runs
    steps: int = 0
    mean_prompt: float = 0.0
    peak_prompt: int = 0
    segments: int = 0
    stages: int = 0
    tool_calls: int = 0
    look_calls: int = 0
    gen_tokens: int = 0
    think_tokens: int = 0

    def _share(self, part: float) -> float:
        return part / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def other_s(self) -> float:
        return max(self.wall_s - self.prefill_s - self.gen_s - self.tool_s - self.wait_s, 0.0)

    prefill_share = property(lambda self: self._share(self.prefill_s))
    gen_share = property(lambda self: self._share(self.gen_s))
    tool_share = property(lambda self: self._share(self.tool_s))
    wait_share = property(lambda self: self._share(self.wait_s))
    other_share = property(lambda self: self._share(self.other_s))
    steps_per_min = property(lambda self: self.steps / (self.wall_s / 60) if self.wall_s > 0 else 0.0)
    look_share = property(lambda self: self.look_calls / self.tool_calls if self.tool_calls else 0.0)
    think_share = property(lambda self: self.think_tokens / self.gen_tokens if self.gen_tokens else 0.0)

    def table(self) -> str:
        def row(name: str, secs: float, share: float, extra: str = "") -> str:
            return f"  {name:<11}{secs:>9.1f} s  {share * 100:>3.0f}%{extra}"

        return "\n".join([
            f"wall {self.wall_s:.1f} s   {self.steps} steps ({self.steps_per_min:.1f} steps/min)   {self.segments} segments, {self.stages} stages",
            row("prefill", self.prefill_s, self.prefill_share),
            row("generation", self.gen_s, self.gen_share),
            row("tools", self.tool_s, self.tool_share, f"   (test runs {self.test_s:.1f} s)" if self.test_s else ""),
            row("waiting", self.wait_s, self.wait_share),
            row("other", self.other_s, self.other_share),
            f"mean prompt {_k(self.mean_prompt)} tokens   peak prompt {_k(self.peak_prompt)} tokens",
        ])


def _k(tokens: float) -> str:
    return f"{tokens / 1000:.0f}k" if tokens >= 1000 else f"{tokens:.0f}"


def _pct(share: float) -> str:
    return f"{share * 100:.0f}%"


def select(events: list[dict[str, Any]], *, since: float | None = None, session: str | None = None) -> list[dict[str, Any]]:
    """The usable events (dicts with a numeric ts) of `session` (exact id or prefix) at or after epoch second `since`."""
    out = []
    for e in events:
        if not isinstance(e, dict) or isinstance(e.get("ts"), bool) or not isinstance(e.get("ts"), (int, float)):
            continue
        if since is not None and e["ts"] < since:
            continue
        if session and not str(e.get("session", "")).startswith(session):
            continue
        out.append(e)
    return out


def breakdown(events: list[dict[str, Any]], *, since: float | None = None, session: str | None = None) -> Breakdown:
    ev = select(events, since=since, session=session)
    b = Breakdown()
    if not ev:
        return b
    start, end, prompts = float("inf"), 0.0, []
    for e in ev:
        kind, own = e.get("kind"), 0.0
        if kind in ("step", "compact_call"):
            pre, dec = _num(e.get("prefill_s")), _num(e.get("decode_s"))
            load = _num(e.get("load_s"))
            b.prefill_s += pre
            b.gen_s += dec
            b.wait_s += load
            own = pre + dec + load
            if kind == "step":
                b.steps += 1
                prompt = int(_num(e.get("prompt")))
                prompts.append(prompt)
                b.peak_prompt = max(b.peak_prompt, prompt)
                gen = int(_num(e.get("gen")))
                b.gen_tokens += gen
                if e.get("think") is True:
                    b.think_tokens += gen
        elif kind == "tool":
            own = _num(e.get("seconds"))
            b.tool_s += own
            b.tool_calls += 1
            name, cmd = e.get("name"), e.get("cmd")
            if name in LOOK_TOOLS or (name == "run" and isinstance(cmd, str) and readonly_shell(cmd)):
                b.look_calls += 1
            if name == "run_tests" or (name == "run" and isinstance(cmd, str) and _TEST_RUN.search(cmd)):
                b.test_s += own
        elif kind == "check":
            own = _num(e.get("seconds"))
            b.tool_s += own
        elif kind in ("server_wait", "autopilot_server_wait"):
            own = _num(e.get("seconds"))
            b.wait_s += own
        elif kind == "segment_start":
            b.segments += 1
        elif kind == "stage_start":
            b.stages += 1
        start, end = min(start, e["ts"] - own), max(end, e["ts"])
    b.wall_s = max(end - start, 0.0)
    b.mean_prompt = sum(prompts) / len(prompts) if prompts else 0.0
    return b


def why_slow(b: Breakdown) -> list[str]:
    """Up to 3 causes in a fixed order of how much each one costs and how directly it can be fixed; every number is from `b`."""
    if b.steps == 0:
        return ["no model steps in the selected events - nothing to measure"]
    causes: list[str] = []
    if b.prefill_share > PREFILL_SHARE:
        if b.mean_prompt > BIG_PROMPT:
            causes.append(f"prefill is {_pct(b.prefill_share)} of the wall time ({b.prefill_s:.0f} s) with a mean prompt of {_k(b.mean_prompt)} tokens "
                          f"(peak {_k(b.peak_prompt)}): lower coding.stage_prompt_cap or enable stages (--stages)")
        else:
            causes.append(f"prefill is {_pct(b.prefill_share)} of the wall time ({b.prefill_s:.0f} s) although the mean prompt is only "
                          f"{_k(b.mean_prompt)} tokens: the server re-reads the prompt often (masks, compactions, a changed prompt head) - see /events")
    if b.steps >= MIN_STEPS_FOR_RATE and b.steps_per_min < MIN_STEPS_PER_MIN:
        causes.append(f"only {b.steps_per_min:.1f} steps/min ({b.steps} steps in {b.wall_s:.0f} s; a healthy staged run does 7-9)")
    if b.tool_calls >= MIN_TOOL_CALLS and b.look_share > LOOK_SHARE:
        causes.append(f"{_pct(b.look_share)} of the {b.tool_calls} tool calls were read-only (look steps): the model reads more than it changes")
    if b.tool_share > TOOL_SHARE and b.tool_s > 0 and b.test_s / b.tool_s > TEST_SHARE:
        causes.append(f"tool time is {_pct(b.tool_share)} of the wall time and {_pct(b.test_s / b.tool_s)} of it is test runs ({b.test_s:.0f} s): "
                      "run a narrower test command while iterating")
    if b.gen_tokens and b.think_share > THINK_SHARE:
        causes.append(f"thinking steps produced {_pct(b.think_share)} of all generated tokens ({b.think_tokens} of {b.gen_tokens})")
    if b.wait_share > WAIT_SHARE:
        causes.append(f"waiting on the model server (not reachable / loading the model) is {_pct(b.wait_share)} of the wall time ({b.wait_s:.0f} s)")
    return causes[:3] or ["no obvious bottleneck"]


def render(b: Breakdown) -> str:
    return b.table() + "\ncauses:\n" + "\n".join(f"  - {c}" for c in why_slow(b))


# ── reading the log, the command ────────────────────────────────────────────

def read_log(path: Path | None = None) -> list[dict[str, Any]]:
    import json

    out: list[dict[str, Any]] = []
    try:
        with (path or eventlog.default_path()).open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        pass
    return out


def parse_since(text: str) -> float:
    """'2h' / '90m' / '30s' / '1d' -> seconds; a bare number is minutes. ValueError on anything else."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", str(text))
    if not m:
        raise ValueError(f"cannot read {text!r} as a time span (use 30m, 2h, 1d)")
    return float(m.group(1)) * (_UNITS[m.group(2)] if m.group(2) else 60)


def time_cli(session: str | None, since: str | None, *, path: Path | None = None, console: Console | None = None) -> int:
    """`agent code time [--session ID] [--since 2h]`. Without --session: the latest session of the log (or, with --since, every session
    since then). No model, no lock - it only reads the event log."""
    console = console or Console()
    try:
        after = time.time() - parse_since(since) if since else None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    events = select(read_log(path), since=after)
    if session:
        events = select(events, session=session)
        label = f"session {session}*"
    elif since:
        label = f"all sessions, last {since}"
    else:
        latest = events[-1]["session"] if events and events[-1].get("session") else None
        events = [e for e in events if e.get("session") == latest] if latest else []
        label = f"session {latest}"
    if not events:
        console.print("[red]no events found[/red] [dim](" + str(path or eventlog.default_path()) + ")[/dim]")
        return 1
    console.print(f"[bold]{label}[/bold]")
    console.print(render(breakdown(events)), markup=False, highlight=False, soft_wrap=True)
    return 0
