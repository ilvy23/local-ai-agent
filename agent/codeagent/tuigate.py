"""The crash gate for terminal programs / games: the HARNESS plays the program with seeded random keys on a real pty before an answer is accepted.

Measured: a model-built curses snake crashed every time the snake died (the GAME OVER banner was wider than the grid, `curses.error`) and neither the
model nor its own tests noticed, because its scripts drove the game LOGIC, never the real screen; the later "fix" silently skipped drawing the banner.
`tui_smoke` finds both in seconds, but the model does not reliably call it - so the engine's final gate calls it (no model call anywhere).

    find_entry(root, edited, override)          the program to play: a top-level .py with a main guard that uses a terminal library
    key_sequence(seed, seconds, small)          the seeded random keys (+ `wait:` gaps) and the closing keys Esc, q, Ctrl-C
    run_gate(entry, runner, ...) -> GateResult  2 runs at 80x24 and 1 at 40x12; fails on a traceback / crash message, a non-zero exit after the
                                                closing keys, or a program that was killed without ever printing; a program that is still running
                                                after the closing keys is only a WARNING
    banner_words(changes, extra)                words like GAME OVER in string literals an edit added -> a NOTE when they never reach a screen

The engine owns the policy (jail, protect restore - the runner is `CodeTools._smoke`), the nudge counting and the cache (engine._tui_gate).
Config (coding.*): tui_gate (true), tui_entry, tui_gate_seconds (8), tui_banner_words."""

from __future__ import annotations

import ast
import hashlib
import math
import random
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from agent.codeagent import tuismoke

KEYS = ["Up", "Down", "Left", "Right", "w", "a", "s", "d", "Space", "Enter", "p", "r"]
CLOSING = ["Esc", "wait:120", "q", "wait:120", "Ctrl-C"]
DEFAULT_SECONDS = 8
MAX_KEYS, SMALL_KEYS = 30, 12
MIN_WAIT, MAX_WAIT = 60, 200                     # ms between two random keys
SETTLE_S = 0.08                                  # what one key costs besides its gap (the program's quiet time after it)
SIZES = ("80x24", "80x24", "40x12")
BANNER_WORDS = ("GAME OVER", "Game over", "You died", "Press any key", "Victory")
MAX_DETAIL = 1400

_MAIN = re.compile(r"^if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", re.M)
_IMPORT_NAME = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", re.M)


@dataclass
class GateResult:
    status: str                          # "pass" | "fail" | "skipped"
    line: str                            # the compact result: 'TUI SMOKE: pass (3 runs, 18 s)' / 'TUI SMOKE: FAIL - <crash line>'
    detail: str = ""                     # a failure: how to reproduce, the transcript tail and the screen (what the model reads below the line)
    crash_line: str = ""                 # one line that names the failure (the exception line, the exit code ...)
    entry: str = ""
    runs: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.line + ("\n" + self.detail if self.detail else "")


# ── what to play ────────────────────────────────────────────────────────────

def _read(path: Path, limit: int = 65536) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def find_entry(root: Path | str, edited: Iterable[Path | str] = (), override: str = "") -> str | None:
    """The program entry, relative to `root`: `override` (coding.tui_entry) when that file exists, else the top-level .py files that have an
    `if __name__ == "__main__"` guard and use a terminal library (themselves or through one top-level module they import); a file edited in this
    prompt is preferred, then the newest by modification time. None when there is none."""
    base = Path(root)
    name = str(override or "").strip()
    if name and not name.startswith(("/", "~")) and ".." not in Path(name).parts and (base / name).is_file():
        return name
    try:
        files = sorted(base.glob("*.py"))[:60]
    except OSError:
        return None
    texts = {f.stem: _read(f) for f in files}
    uses = {stem for stem, t in texts.items() if tuismoke._TUI_IMPORT.search(t)}
    mine = {Path(p).resolve() for p in edited}
    found: list[tuple[bool, float, str]] = []
    for f in files:
        t = texts[f.stem]
        if not _MAIN.search(t):
            continue
        if f.stem not in uses and not any(n in uses and n != f.stem for n in _IMPORT_NAME.findall(t)):
            continue
        try:
            found.append((f.resolve() in mine, f.stat().st_mtime, f.name))
        except OSError:
            continue
    return max(found)[2] if found else None


# ── the random play ─────────────────────────────────────────────────────────

def seed_for(text: str) -> int:
    """The same file content always plays the same game."""
    return int.from_bytes(hashlib.sha1(text.encode("utf-8", errors="replace")).digest()[:6], "big")


def estimated_seconds(keys: list[str]) -> float:
    return sum(int(k[5:]) / 1000 if k.startswith("wait:") else SETTLE_S for k in keys)


def key_sequence(seed: int, seconds: float = DEFAULT_SECONDS, small: bool = False) -> list[str]:
    """Up to 30 random keys (12 for the small terminal) with a `wait:60..200` gap after each, as many as fit into `seconds` (half of it for the
    small run), then Esc, q and Ctrl-C."""
    rng = random.Random(seed)
    budget = max(0.5, float(seconds)) * (0.5 if small else 1.0)
    keys: list[str] = []
    used = 0.0
    for _ in range(SMALL_KEYS if small else MAX_KEYS):
        wait = rng.randint(MIN_WAIT, MAX_WAIT)
        if used + wait / 1000 + SETTLE_S > budget and len(keys) >= 6:
            break
        keys += [rng.choice(KEYS), f"wait:{wait}"]
        used += wait / 1000 + SETTLE_S
    return keys + CLOSING


# ── the verdict of one run ──────────────────────────────────────────────────

def _last_line(res: Any) -> str:
    lines = [ln.strip() for ln in (list(res.tail or []) or str(res.last_frame or res.screen or "").splitlines()) if ln.strip()]
    return lines[-1][:200] if lines else ""


def judge(res: Any) -> tuple[str, str]:
    """("ok" | "warn" | "fail", reason) for one SmokeResult. Ctrl-C is a closing key: the KeyboardInterrupt it raises and the exit it causes are no
    crash. A program still running after the closing keys is a warning - unless it never printed anything."""
    last = _last_line(res)
    if res.traceback and not last.startswith("KeyboardInterrupt"):
        return "fail", last or "a traceback / crash message was printed"
    if res.status == "exited":
        interrupted = res.signal == "SIGINT" or res.exit_code in (130, -2) or (res.traceback and last.startswith("KeyboardInterrupt"))
        if res.exit_code not in (0, None) and not interrupted:
            return "fail", f"exit code {res.exit_code}" + (f" ({res.signal})" if res.signal else "") + " after Esc, q and Ctrl-C" + (f": {last}" if last else "")
        return "ok", ""
    if res.status == "output_cap":
        return "warn", "printed more than 2 MB (the harness stopped it)"
    if not res.startup_bytes:
        return "fail", "no output: the program printed nothing at startup, ignored the closing keys and had to be killed"
    return "warn", "still running after Esc, q and Ctrl-C (the harness stopped it)"


def _excerpt(res: Any, size: str, keys: list[str]) -> str:
    sent = [k for k in keys[: max(res.keys_sent, 1)] if not k.startswith("wait:")]
    rows = [ln.rstrip()[:90] for ln in str(res.last_frame or res.screen or "").splitlines() if ln.strip()][:10]
    tail = [ln.rstrip()[:110] for ln in (res.tail or []) if ln.strip()]
    hist = str(res.history or "")
    start = hist.rfind("Traceback (most recent call last)")
    if start >= 0:                                           # the traceback from its first line (the tail of 15 lines may have lost it): head + last frames
        trace = [ln.rstrip()[:110] for ln in hist[start:].splitlines() if ln.strip()]
        tail = trace if len(trace) <= 9 else [*trace[:2], "  ...", *trace[-6:]]
    parts = [f"run at {size}, keys: {' '.join(sent[:40])} (gaps of {MIN_WAIT}-{MAX_WAIT} ms between them)"]
    if tail:
        parts.append("transcript tail:\n" + "\n".join(tail[-10:]))
    if rows:
        parts.append("screen:\n" + "\n".join(rows))
    text = "\n".join(parts)
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 4].rsplit("\n", 1)[0] + "\n..."


def run_gate(entry: str, runner: Callable[[str, list[str], str, int], Any], *, seconds: float = DEFAULT_SECONDS, source: str = "",
             banners: Iterable[str] = (), command: str | None = None) -> GateResult:
    """Play `entry`: `runner(command, keys, size, timeout)` -> SmokeResult (the engine passes CodeTools._smoke, so the protect snapshot applies).
    The first failing run ends the gate. `source`: the entry's text (the seed); `banners`: words that must have been seen on some screen."""
    cmd = command or f"python3 {shlex.quote(entry)}"
    seed = seed_for(source)
    warns: dict[str, list[str]] = {}
    history, total, done = [], 0.0, 0
    for i, size in enumerate(SIZES):
        keys = key_sequence(seed + i, seconds, small=size != SIZES[0])
        res = runner(cmd, keys, size, math.ceil(estimated_seconds(keys)) + 6)
        if res.error:
            return GateResult("skipped", f"TUI SMOKE: skipped ({res.error})", entry=entry, runs=done, seconds=total)
        done += 1
        total += res.seconds
        history.append(res.history)
        kind, why = judge(res)
        if kind == "fail":
            return GateResult("fail", f"TUI SMOKE: FAIL - {why}", _excerpt(res, size, keys), why, entry, done, total)
        if kind == "warn":
            warns.setdefault(why, []).append(size)
    seen = "\n".join(history).lower()
    notes = [f'The text "{w}" exists in the code but never appeared on screen in the random-play runs - check that the player can actually see it.'
             for w in banners if w.lower() not in seen]
    warnings = [f"{why} ({', '.join(dict.fromkeys(sizes))})" for why, sizes in warns.items()]
    line = f"TUI SMOKE: pass ({done} runs, {total:.0f} s)" + "".join(f" - WARNING: {w}" for w in warnings) + "".join(f" - NOTE: {n}" for n in notes)
    return GateResult("pass", line, entry=entry, runs=done, seconds=total, warnings=warnings, notes=notes)


# ── the visible-text angle ──────────────────────────────────────────────────

def _literals(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def banner_words(changed: dict[str, tuple[str | None, str]], extra: Iterable[str] = ()) -> list[str]:
    """The banner words (BANNER_WORDS + coding.tui_banner_words, case-insensitive) that occur in a string literal this prompt's edits ADDED to a .py
    file, as they are written there; one per word."""
    words = [w for w in (*BANNER_WORDS, *(str(x).strip() for x in extra or ())) if w]
    rx = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)), re.I) if words else None
    found: dict[str, str] = {}
    for path, (before, after) in sorted(changed.items()):
        if rx is None or not str(path).endswith(".py") or after is None or before == after:
            continue
        for lit in sorted(_literals(after) - _literals(before or ""), key=after.find):
            for m in rx.finditer(lit):
                found.setdefault(m.group(0).lower(), m.group(0))
    return list(found.values())
