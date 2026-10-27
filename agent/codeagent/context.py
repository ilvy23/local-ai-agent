"""Context management for long coding runs: keep the prompt prefix stable, trim rarely and in bulk, compact with ground truth.

What the earlier agent did wrong (measured on a real session): 88% of its own output was whole-file `write_file`
payloads that stayed in the history after the file was on disk; it re-read files it had just written; masking rewrote
earlier messages every step, which throws away the server's prompt cache (a 25k-token prompt costs ~35 s to re-read
at ~700 tok/s); the work log claimed unfinished things and forgot the user's constraints; and nobody could tell afterwards
when any of this happened. This module fixes those:

  * calls are stored in compact form (the content is on disk) and the tool result carries an outline instead,
  * old tool output is masked ONCE in bulk when the window is ~42% full, then not again until it has grown by 15%,
  * the work log is built from a harness LEDGER verified against the disk (files, sizes, syntax, commands and exit
    codes) plus the model's own DONE/DECISIONS/PITFALLS, and always carries the user's requests verbatim,
  * every mask / compaction / fold is written to the event log with before/after sizes.
"""

from __future__ import annotations

import re
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.codeagent import edits
from agent.codeagent.eventlog import EventLog
from agent.memory import worklog

logger = logging.getLogger(__name__)

# Measured on a live run (31 min, 279 steps): 13 masks at 50% each freed only 20-35k tokens, the window refilled in ~12 steps, and every mask
# re-read the whole prompt (23-79 s): 14 steps = 10.5 of the 22 minutes spent on prefill. Later and bigger is cheaper per step.
MASK_AT = 0.72
MASK_REARM = 0.15
MASK_MIN_GAIN = 0.15             # a mask re-reads the whole prompt after its first change (measured: 50-77 s for ~50k tokens); one
                                 # that frees less than this share of the window is not worth it - compaction takes over at COMPACT_AT
COMPACT_AT = 0.80
EMERGENCY_AT = 0.92              # still this full after compaction: model-free hard trim
FOLD_AT = 0.72                   # at the start of a prompt: fold old turns once the history is this full. Was 0.45: a resumed session folded at once and the model re-read the whole code base (74 read_file in a row, no edit)
KEEP_TAIL = 6
PINNED_PREFIXES = ("BRIEF CARD", "STAGE GOAL")    # system messages that are never folded, compacted or masked (brief.py, stages.py)
MIN_MASK_SAVING_CHARS = 8000
# Inside a stage the window that matters is the stage cap (stages.py, 28k), not num_ctx: the old line .72 x 96k = 69k was never reached, so a
# stage never masked and died at the cap (measured: 6 caps in 11 min, 82 reads vs 14 edits, every restart a full prompt re-read).
STAGE_MASK_AT = 0.70             # mask in ONE batch when the prompt passes this share of min(num_ctx, stage cap)
STAGE_MASK_GAP = 0.40            # ... and not again before the prompt has grown by this share of the cap since that batch (Anthropic's 'clear_at_least':
                                 # every mutation of the prefix costs the server's prompt cache from the first changed message on). Was 0.25
STAGE_MASK_MIN_GAIN = 0.25       # a TRIGGER batch (0.70 x cap) must free at least this share of the PROMPT, else it is skipped: it costs a ~26k-token
                                 # prefill (measured 600-700 tok/s = ~40 s). Was 0.08 of the cap. A forced batch at the cap keeps "any gain".
STAGE_MASK_TARGET = 0.50         # a batch masks old results, oldest first, until the estimated prompt is <= this share of the stage cap
MASK_KEEP_TOKENS = 0.20          # ... but keeps the newest results (the working set) up to this share of the cap; the newest result is ALWAYS kept
MASK_KEEP_TURNS = 2              # floor: the newest assistant/tool turns a stage batch never touches (was 6: those turns held the big reads, a batch
                                 # freed ~3k tokens, the next step masked again and every mask re-read ~26k tokens: measured 11 big prefills = 135 s in 8 min)
STAGE_MASK_WINDOW = 8            # thrash guard: two fruitless batches within this many steps end the segment ("stage_cap") ...
STAGE_MASK_RETRY_STEPS = 6       # ... and so does a batch needed again within this many steps of the previous one
STAGE_MASK_FRUITLESS = 0.20      # a batch that freed less than this share of the prompt is fruitless
# results a stage batch may stub: everything the model can fetch again. Never edit/write results, never system messages.
STAGE_MASK_TOOLS = frozenset({"read_file", "read_files", "outline", "search", "list_dir", "repo_map", "show_symbol", "where_used",
                              "run", "run_tests", "lint", "tui_smoke", "web_search", "web_fetch"})
CHARS_PER_TOKEN = 3.2

CODE_WORKLOG_SYSTEM = (
    "You maintain the work log of an autonomous coding agent whose conversation is being compacted. Rewrite the "
    "log so the agent can continue from it alone. The harness appends a verified LEDGER of files and commands "
    "(ground truth, checked against the disk) - do not repeat it. Output ONLY these sections as terse bullets:\n"
    "DONE: what was built or changed and why it matters (one line each; refer to files by path, never paste code).\n"
    "STATE: facts that matter now - what passes/fails, versions, environment quirks, the failing error text.\n"
    "DECISIONS: design and naming choices, APIs the code relies on (exact function/class names), so they are "
    "not re-litigated and later code is written against the real names.\n"
    "OPEN: every requirement of the user's requests (given verbatim) and acceptance items that the LEDGER does not "
    "prove, next step first. Anything not proven by a successful command or an existing file stays OPEN - claiming "
    "unfinished work is done is the worst mistake.\n"
    "PITFALLS: errors hit, fixes, and approaches NOT to retry.\n"
    "Keep exact identifiers, paths, commands and error text. Drop pleasantries, repeated reads and raw file "
    "contents. If an earlier log is given, merge it: keep still-true items, update changed ones, never lose open items."
)


IN_PLACE_PROMPT = (
    "[harness] The context window is nearly full. The harness will now replace the older part of this conversation with a "
    "work log that YOU write in this reply (your system prompt, my requests, the acceptance list and your last few steps stay "
    "visible as they are; a verified LEDGER of files and commands is appended automatically - do not repeat it). Do NOT call a "
    "tool now. Reply with the log only, as terse bullets under exactly these headings:\n"
    "DONE: what was built or changed (file paths, one line each, never code).\n"
    "STATE: what passes/fails right now, the exact failing error text, environment quirks.\n"
    "DECISIONS: design and naming choices, the exact function/class names and signatures later code relies on.\n"
    "OPEN: every requirement of my requests and every acceptance item not yet proven by a successful command, in order, the "
    "very next step first (be concrete: file and function). Unfinished work must stay OPEN.\n"
    "PITFALLS: errors hit, their fixes, approaches not to retry.\n"
    "If an earlier WORK LOG is above, merge it: keep what is still true, never drop open items."
)


def est_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


@dataclass
class FileEntry:
    action: str
    step: int
    lines: int
    size: int
    names: list[str] = field(default_factory=list)


class Ledger:
    """Exact, harness-recorded facts. Rendered against the disk at compaction time, so it can't drift from reality."""

    def __init__(self) -> None:
        self.files: dict[str, FileEntry] = {}
        self.runs: list[tuple[int, str, int | None, str]] = []
        self.check_cmd: str | None = None      # coding.check_cmd, set by the ContextManager

    def note_file(self, path: str, text: str, step: int, action: str) -> None:
        names = [o.split(":", 1)[1].strip().split("(")[0].replace("def ", "").replace("class ", "")
                 for o in edits.outline(path, text, 14) if ":" in o]
        self.files[path] = FileEntry(action, step, text.count("\n") + 1, len(text), names[:12])

    def note_run(self, step: int, command: str, code: int | None, tail: str) -> None:
        self.runs.append((step, command[:160], code, tail[:160]))
        del self.runs[:-40]

    def render(self, root: Path) -> str:
        lines = ["LEDGER (recorded by the harness, verified against the disk just now):"]
        for path, e in sorted(self.files.items()):
            p = root / path if not Path(path).is_absolute() else Path(path)
            if not p.exists():
                lines.append(f"- {path}: MISSING on disk (was {e.action} at step {e.step})")
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                lines.append(f"- {path}: unreadable")
                continue
            err = edits.check_syntax(str(p), text, self.check_cmd)
            names = ", ".join(e.names[:10])
            lines.append(f"- {path}: {len(text.splitlines())} lines, {len(text) // 1000 + 1}k, "
                         f"{'SYNTAX ERROR' if err else 'syntax ok'}, {e.action} at step {e.step}" + (f"; defines {names}" if names else ""))
        if self.runs:
            lines.append("commands (newest last):")
            for step, cmd, code, tail in self.runs[-14:]:
                lines.append(f"- step {step}: `{cmd}` -> exit {code if code is not None else '?'}" + (f" ({tail})" if code else ""))
        return "\n".join(lines)


def pinned(message: dict[str, Any]) -> bool:
    """The user's brief card and a stage's goal stay byte-identical for the whole task: compaction moves them, never rewrites them."""
    return message.get("role") == "system" and str(message.get("content", "")).startswith(PINNED_PREFIXES)


def _parse_call(content: str) -> tuple[str, dict[str, Any]] | None:
    try:
        data = json.loads(content)
    except ValueError:
        return None
    if isinstance(data, dict) and "tool" in data:
        return str(data["tool"]), data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
    return None


def mask_observations(messages: list[dict[str, Any]], root: Path, keep_last: int = 10, min_chars: int = 700, *,
                      keep_turns: int | None = None, only_tools: frozenset[str] | None = None) -> tuple[list[dict[str, Any]], int]:
    """Shrink OLD tool results without making the model forget what it learned. A bare "[output omitted]" stub made the model
    re-read the same files again and again (read -> mask -> forget -> read ...: 54 steps, nothing written). Instead:
      * a file read keeps the file's OUTLINE (names, signatures, line numbers: ~10-15% of its size) so the model still knows
        where everything is and can ask for exactly the lines it needs,
      * search results keep their first lines, command output keeps its head and tail,
      * everything else keeps its first line.
    `keep_turns` (instead of `keep_last` messages) keeps the newest N assistant turns and what followed them; `only_tools` limits the
    stubbed results to those tools (the stage batch: never an edit result)."""
    out, saved, _ = _mask(messages, root, keep_last, min_chars, keep_turns, only_tools)
    return out, saved


def _seen_ranges(messages: list[dict[str, Any]]) -> dict[str, list[str]]:
    """{path: ["1-250", "300-400", ...]} of the file reads in the history (newest last), from the calls and the 'lines a-b' of their results."""
    seen: dict[str, list[str]] = {}
    pending: tuple[str, dict[str, Any]] | None = None
    for m in messages:
        content = str(m.get("content", ""))
        if m.get("role") == "assistant":
            pending = _parse_call(content)
        elif m.get("role") == "tool":
            if pending and pending[0] == "read_file" and pending[1].get("path") and not content.startswith("[old "):
                hit = _RANGE.search(content)
                rng = f"{hit.group(1)}-{hit.group(2)}" if hit else "all"
                lst = seen.setdefault(str(pending[1]["path"]), [])
                if rng not in lst:
                    lst.append(rng)
            pending = None
    return seen


_RANGE = re.compile(r"^--- lines (\d+)-(\d+) ---$", re.M)


def _mask(messages: list[dict[str, Any]], root: Path, keep_last: int, min_chars: int, keep_turns: int | None,
          only_tools: frozenset[str] | None, only_idx: set[int] | frozenset[int] | None = None) -> tuple[list[dict[str, Any]], int, int]:
    if keep_turns is not None:
        turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        cut = turns[-keep_turns] if len(turns) >= keep_turns > 0 else (len(messages) if keep_turns <= 0 else 0)
    else:
        cut = max(len(messages) - keep_last, 0)
    out: list[dict[str, Any]] = []
    saved = stubbed = 0
    pending: tuple[str, dict[str, Any]] | None = None
    seen = _seen_ranges(messages)
    for i, m in enumerate(messages):
        role, content = m.get("role"), str(m.get("content", ""))
        if role == "assistant":
            pending = _parse_call(content)
            if i < cut and pending and pending[0] in ("write_file", "append_file", "replace_lines", "insert_after") and len(content) >= min_chars:
                small = json.dumps(compact_call(*pending), ensure_ascii=False)
                if len(small) < len(content):
                    saved += len(content) - len(small)
                    out.append({**m, "content": small})
                    continue
        if (i in only_idx if only_idx is not None else i < cut) and role == "tool" and len(content) >= min_chars and "[old " not in content[:12] \
                and (only_tools is None or (pending is not None and pending[0] in only_tools)):
            name, args = pending or ("", {})
            stub = _stub_for(name, args, content, Path(root), seen.get(str(args.get("path")), []))
            if len(stub) < len(content):
                out.append({**m, "content": stub})
                saved += len(content) - len(stub)
                stubbed += 1
                pending = None
                continue
        if role == "tool":
            pending = None
        out.append(m)
    return out, saved, stubbed


def _plan_stage_mask(messages: list[dict[str, Any]], root: Path, need_tokens: float, keep_tokens: float, keep_turns: int) -> frozenset[int]:
    """Which tool results a stage batch stubs: the OLD ones, oldest first, until `need_tokens` are freed. Never masked: the newest tool result,
    the working set (the newest results up to `keep_tokens` in total, walking back from the newest and stopping at the first one that does
    not fit - a single result larger than the budget is then the only one kept), anything inside the newest `keep_turns` turns, edit/write
    results (not in STAGE_MASK_TOOLS), system messages, an already stubbed result.
    Order matters for the server's KV cache: the first CHANGED message decides how much of the cached prefix survives. Masking a contiguous
    OLD prefix (oldest first, nothing unmasked before the masked ones) keeps the cache valid up to the earliest change and makes the
    unavoidable re-read start as late as it can; masking results in the middle while older ones stay unmasked would invalidate from the
    same or an earlier point for less gain."""
    if need_tokens <= 0:
        return frozenset()
    turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    cut = turns[-keep_turns] if len(turns) >= keep_turns > 0 else (len(messages) if keep_turns <= 0 else 0)
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    limit = cut
    if tool_idx:
        total, lowest = 0.0, tool_idx[-1]                    # the newest result is always kept, however large
        for n, i in enumerate(reversed(tool_idx)):
            size = len(str(messages[i].get("content", ""))) / CHARS_PER_TOKEN
            if n and total + size > keep_tokens:
                break
            total, lowest = total + size, i
        limit = min(cut, lowest)
    seen = _seen_ranges(messages)
    pending: tuple[str, dict[str, Any]] | None = None
    chosen: set[int] = set()
    freed = 0.0
    for i, m in enumerate(messages):
        if i >= limit or freed >= need_tokens:
            break
        content = str(m.get("content", ""))
        if m.get("role") == "assistant":
            pending = _parse_call(content)
        elif m.get("role") == "tool":
            if len(content) >= 700 and "[old " not in content[:12] and pending is not None and pending[0] in STAGE_MASK_TOOLS:
                name, args = pending
                stub = _stub_for(name, args, content, Path(root), seen.get(str(args.get("path")), []))
                if len(stub) < len(content):
                    chosen.add(i)
                    freed += (len(content) - len(stub)) / CHARS_PER_TOKEN
            pending = None
    return frozenset(chosen)


_SPILL_LINE = re.compile(r"^full output: \.scratch/spill/.*$", re.M)


def _stub_for(name: str, args: dict[str, Any], content: str, root: Path, seen: list[str] | None = None) -> str:
    """The stub of an old tool result; a spilled result keeps its one locator line (spill.py), the way back to the whole text."""
    stub = _stub_body(name, args, content, root, seen)
    loc = _SPILL_LINE.findall(content)
    return stub + ("\n" + loc[-1] if loc and loc[-1] not in stub else "")


def _stub_body(name: str, args: dict[str, Any], content: str, root: Path, seen: list[str] | None = None) -> str:
    # every "[old ...]" stub and "[... N more lines omitted]" below is a harness marker: see MARKERS
    lines = content.splitlines()
    head = next((ln.strip() for ln in lines if ln.strip()), "")[:140]
    if name in ("read_file", "read_files", "outline") and args.get("path"):
        p = Path(str(args["path"]))
        p = p if p.is_absolute() else root / p
        try:
            outline = edits.outline(str(p), p.read_text(encoding="utf-8", errors="replace"), 45)
        except OSError:
            outline = []
        rng = f" lines {args.get('start')}-{args.get('end')}" if args.get("start") or args.get("end") else ""
        body = ("\n".join(outline) if outline else "(no definitions)")
        hit = _RANGE.search(content)
        got = f"{args.get('start') or hit.group(1)}-{args.get('end') or hit.group(2)}" if hit else None
        again = (f"read_file(path=\"{args['path']}\", start={got.split('-')[0]}, end={got.split('-')[1]})" if got
                 else f"read_file(path=\"{args['path']}\")")
        ranges = f"\nRanges of this file you have read: {', '.join(seen[-8:])}." if seen else ""
        return (f"[old read of {args['path']}{rng} omitted to save context ({len(content)} chars). Its structure, so you do not need to re-read it:\n"
                f"{body}{ranges}\nTo get exactly this back: {again}. Read only the exact line range you need.]")
    if name == "search":
        return "[old search results trimmed]\n" + "\n".join(lines[:8]) + (f"\n[... {len(lines) - 8} more lines omitted]" if len(lines) > 8 else "")
    if name in ("run", "run_tests", "lint", "tui_smoke"):
        return "[old command output trimmed]\n" + "\n".join(lines[:4]) + ("\n[...]\n" + "\n".join(lines[-6:]) if len(lines) > 10 else "")
    return f"[old tool output omitted: {head} ({len(content)} chars) - run the tool again if you need it]"


def _clip_text(text: Any, limit: int) -> Any:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "...[cut]"       # harness marker: see MARKERS


PLACEHOLDER = re.compile(r"^[ \t]*\[\d+ lines?, \d+ chars - now on disk\][ \t]*$\n?", re.M)   # what the history shows instead of written content

# EVERY text the harness inserts into what the model sees (the model copies what it sees - three real incidents: the on-disk placeholder
# sent as file content, `...[cut]` written into a source file, and the read pause bypassed with `run sed`). Each emitter has a comment
# pointing here; tools._commit_write refuses a write that adds one, tools.run refuses a writing command that contains one.
MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("on_disk", re.compile(r"\[\d+ lines?, \d+ chars - now on disk\]")),               # compact_call / mask_observations
    ("cut", re.compile(r"\.\.\.\[cut\]")),                                            # _clip_text
    ("chars_omitted", re.compile(r"\[… \d+ chars omitted …\]")),                      # tools.cap_output, reviewer._clip
    ("bytes_omitted", re.compile(r"\[… \d+ bytes of output omitted …\]")),            # tools._read_capped
    ("truncated", re.compile(r"\[truncated - use a smaller range\]")),                  # tools.read_file
    ("more_matches", re.compile(r"\[… \d+ more matches")),                             # tools.search
    ("old_read", re.compile(r"\[old read of ")),                                       # mask stubs (_stub_for)
    ("old_output", re.compile(r"\[old tool output omitted")),
    ("old_command", re.compile(r"\[old command output trimmed\]")),
    ("old_search", re.compile(r"\[old search results trimmed\]")),
    ("more_lines", re.compile(r"\[\.\.\. \d+ more lines omitted\]")),
    ("no_result", re.compile(r"\[no result recorded")),                                # sessionstore.NO_RESULT
    ("spill_gap", re.compile(r"\(\d+ of \d+ lines not shown: lines \d+-\d+\)")),         # spill.Spill.clip
    ("spill_locator", re.compile(r"full output: \.scratch/spill/")),                    # spill.Spill.locator (clip / note)
]


def new_markers(before: str | None, after: str) -> list[str]:
    """Names of the markers that occur more often in `after` than in `before` (None = empty): what a write would ADD."""
    before = before or ""
    return [name for name, rx in MARKERS if len(rx.findall(after)) > len(rx.findall(before))]


def compact_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """The form in which a tool call stays in the history. The model imitates its own earlier calls, so the SHAPE must stay
    exactly right (a measured failure: `edits` was stored as the text "[1 edit(s) applied]", after which the model sent `edits`
    as a string every time; a patch stored as "[12 lines applied]" was copied as the patch). Only the bulk is cut:
      * write_file/append_file: the content is on disk (a short marker string - it is a string either way),
      * edit_file: still a list of {search, replace} objects, each text clipped,
      * apply_patch: still a real diff, clipped after the first lines."""
    if name in ("write_file", "append_file") and isinstance(args.get("content"), str):
        text = args["content"]
        # the on-disk placeholder is a harness marker (MARKERS "on_disk")
        return {"tool": name, "arguments": {**{k: v for k, v in args.items() if k != "content"},
                                            "content": f"[{text.count(chr(10)) + 1} lines, {len(text)} chars - now on disk]"}}
    if name == "edit_file":
        slim = {k: v for k, v in args.items() if k not in ("edits", "search", "replace")}
        if isinstance(args.get("edits"), list):
            slim["edits"] = [{"search": _clip_text(e.get("search", ""), 160), "replace": _clip_text(e.get("replace", ""), 160)}
                             for e in args["edits"][:6] if isinstance(e, dict)]
        elif "search" in args:
            slim["search"], slim["replace"] = _clip_text(args.get("search", ""), 160), _clip_text(args.get("replace", ""), 160)
        return {"tool": name, "arguments": slim}
    if name in ("replace_lines", "insert_after") and isinstance(args.get("new_text"), str):
        # NOT clipped with "...[cut]": the model copied that marker into the next call and broke a file (seen live, twice). Like write_file the
        # new text stays in the clear while it is recent (engine._run_call) and becomes the on-disk placeholder once it is old (mask_observations).
        text = args["new_text"]
        return {"tool": name, "arguments": {**{k: v for k, v in args.items() if k != "new_text"},
                                            "new_text": f"[{text.count(chr(10)) + 1} lines, {len(text)} chars - now on disk]"}}
    if name == "apply_patch" and isinstance(args.get("patch"), str):
        return {"tool": name, "arguments": {**{k: v for k, v in args.items() if k != "patch"}, "patch": _clip_text(args["patch"], 500)}}
    return {"tool": name, "arguments": args}


class ContextManager:
    def __init__(self, llm: Any, config: dict[str, Any], num_ctx: int, events: EventLog, root: Path) -> None:
        self.llm, self.config, self.num_ctx, self.events, self.root = llm, config, num_ctx, events, Path(root)
        self.messages: list[dict[str, Any]] = []
        self.ledger = Ledger()
        self.ledger.check_cmd = (config.get("coding") or {}).get("check_cmd")
        self.user_requests: list[str] = []
        self.last_prompt = 0
        self.last_eval = 0
        self._since = 0                    # chars appended since the last model call
        self.mask_floor = 0.0              # no new mask pass before the estimate passes this
        self.on_clear_cache: Any = None    # called when old observations were dropped (read cache must be reset)
        self.step = 0
        self.last_tps = 0.0
        # set by the engine: write the work log with the conversation model on the conversation itself (its prompt is already
        # in the server cache - only the log is generated), and extra state for every log (live acceptance list, notes)
        self.summarize_in_place: Any = None
        self.extras: Any = None
        self.model_fn: Any = None          # set by the engine: the current main model (the fallback summary must not load a second one)
        self._compact_retry_at = 0.0
        self.stage_cap_fn: Any = None      # set by the engine: the current stage's prompt cap (tokens) or None
        self.stage_mask_log: list[tuple[int, float]] = []   # (step, share of the prompt freed) of the stage batches of this segment
        self.stage_mask_thrash = False     # the thrash guard fired: no more batches this segment, the engine ends it with "stage_cap"
        self.stage_mask_floor = 0.0        # no stage batch before the estimate passes this (after a batch: its result + the growth gap).
                                           # Saved with the session; a missing value reads as 0 = "not masked since", which at worst masks once more

    # ── messages ────────────────────────────────────────────────────────
    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._since += len(str(message.get("content", "")))

    def add_user(self, text: str) -> None:
        self.user_requests.append(text)
        self.add({"role": "user", "content": text})

    def estimate(self) -> int:
        return self.last_prompt + self.last_eval + int(self._since / CHARS_PER_TOKEN)

    def note_usage(self, message: dict[str, Any], think: bool = False) -> None:
        self.last_prompt = int(message.get("_prompt_eval_count", 0) or 0)
        self.last_eval = int(message.get("_eval_count", 0) or 0)
        self._since = 0
        ev_ns = int(message.get("_eval_ns", 0) or 0)
        if ev_ns > 3e8 and self.last_eval > 10:
            self.last_tps = self.last_eval / (ev_ns / 1e9)
        load_ns = message.get("_load_ns")
        self.events.emit("step", n=self.step, prompt=self.last_prompt, gen=self.last_eval,
                         prefill_s=round(int(message.get("_prompt_eval_ns", 0) or 0) / 1e9, 2),
                         decode_s=round(int(message.get("_eval_ns", 0) or 0) / 1e9, 2),
                         load_s=round(load_ns / 1e9, 2) if isinstance(load_ns, (int, float)) else None, think=think)

    def fill(self) -> float:
        return self.estimate() / max(self.num_ctx, 1)

    # ── maintenance ─────────────────────────────────────────────────────
    def _coding_float(self, key: str, default: float) -> float:
        try:
            return float((self.config.get("coding") or {}).get(key, default))
        except (TypeError, ValueError):
            return default

    def stage_budget(self) -> int | None:
        """Inside a stage the context decisions are made against min(window, stage cap); None outside a stage (the old thresholds apply)."""
        cap = self.stage_cap_fn() if self.stage_cap_fn else None
        return min(self.num_ctx, int(cap)) if cap else None

    def stage_mask(self, force: bool = False) -> str | None:
        """ONE batch over the old tool results: oldest first, until the estimated prompt is <= `stage_mask_target` x budget, keeping the newest
        result and a working set of `mask_keep_tokens` x budget (never edit results, never a system message; `mask_keep_turns` is only a floor).
        A mutation of the prefix costs the server's prompt cache from the first changed message on, so it happens rarely and in bulk: at
        `stage_mask_at` x budget (only when it frees >= `STAGE_MASK_MIN_GAIN` of the prompt), then not again before the prompt has grown by
        `stage_mask_gap` x budget. `force` (the prompt is over the cap) skips the trigger level and the minimum gain, not the growth gap.
        Thrash guard: a batch needed again within 6 steps, or two fruitless ones within `stage_mask_window` steps, stops masking for this
        segment (`stage_mask_thrash`; the engine ends it with "stage_cap": one clean restart is cheaper than repeated cache invalidations).
        Returns a UI note when it masked."""
        budget = self.stage_budget()
        if not budget or self.stage_mask_thrash:
            return None
        used = self.estimate()
        if used < self.stage_mask_floor:
            return None
        if not force and used < self._coding_float("stage_mask_at", STAGE_MASK_AT) * budget:
            return None
        retry = int(self._coding_float("stage_mask_retry_steps", STAGE_MASK_RETRY_STEPS))
        if self.stage_mask_log and self.step - self.stage_mask_log[-1][0] < retry:
            return self._stage_mask_thrash("again_within_steps", used)
        target = self._coding_float("stage_mask_target", STAGE_MASK_TARGET) * budget
        keep_tokens = self._coding_float("mask_keep_tokens", MASK_KEEP_TOKENS) * budget
        keep = max(int(self._coding_float("mask_keep_turns", MASK_KEEP_TURNS)), 0)
        idx = _plan_stage_mask(self.messages, self.root, used - target, keep_tokens, keep)
        masked, saved, n = _mask(self.messages, self.root, 0, 700, keep, STAGE_MASK_TOOLS, idx)
        tokens = int(saved / CHARS_PER_TOKEN)
        if not n or (not force and tokens < self._coding_float("stage_mask_min_gain", STAGE_MASK_MIN_GAIN) * used):
            if not force:                                     # not worth a cache miss yet: look again after some growth
                self.stage_mask_floor = max(self.stage_mask_floor, used + self._coding_float("stage_mask_gap", STAGE_MASK_GAP) * budget)
                self.events.emit("stage_mask_skipped", reason="low_gain", used=used, saved_chars=saved)
            return None
        self.messages = masked
        after = max(used - tokens, 0)
        self.last_prompt = max(0, self.last_prompt - tokens)  # the next measure replaces this estimate
        self.stage_mask_floor = after + self._coding_float("stage_mask_gap", STAGE_MASK_GAP) * budget
        self._cache_reset()
        self.stage_mask_log.append((self.step, tokens / max(used, 1)))
        self.events.emit("stage_mask", before=used, after=after, masked_results=n)
        window = int(self._coding_float("stage_mask_window", STAGE_MASK_WINDOW))
        if len(self.stage_mask_log) >= 2 and all(f < STAGE_MASK_FRUITLESS for _, f in self.stage_mask_log[-2:]) \
                and self.stage_mask_log[-1][0] - self.stage_mask_log[-2][0] <= window:
            self.stage_mask_thrash = True                     # the batch above is applied; but no third one - end the segment cleanly
            self.events.emit("stage_mask_skipped", reason="two_fruitless_batches", used=after)
        return f"stage context trimmed  {used:,} → ~{after:,} tokens ({n} old results)".replace(",", ".")

    def _stage_mask_thrash(self, reason: str, used: int) -> None:
        self.stage_mask_thrash = True
        self.events.emit("stage_mask_skipped", reason=reason, used=used)
        return None

    def maintain(self) -> list[str]:
        """Run after each step. Returns short notes for the UI. Order: mask once in bulk, then compact."""
        notes: list[str] = []
        used = self.estimate()
        if self.stage_budget():
            note = self.stage_mask()
            if note:
                notes.append(note)
                used = self.estimate()
        elif used >= MASK_AT * self.num_ctx and used >= self.mask_floor:
            masked, saved = mask_observations(self.messages, self.root, keep_last=10)
            if saved < max(MIN_MASK_SAVING_CHARS, MASK_MIN_GAIN * self.num_ctx * CHARS_PER_TOKEN):
                # Too little to gain for a full re-read of the prompt (one run paid four such masks in 59 steps, ~250 s of prefill,
                # each freeing ~12k tokens): leave the prompt alone until compaction, which frees far more for about the same price.
                self.mask_floor = max(COMPACT_AT * self.num_ctx, used + 0.05 * self.num_ctx)
                self.events.emit("mask_skipped", used=used, saved_chars=saved)
            else:
                self.messages = masked
                after = used - int(saved / CHARS_PER_TOKEN)
                self.mask_floor = after + MASK_REARM * self.num_ctx
                self._cache_reset()
                self.events.emit("mask", before=used, after=after, saved_chars=saved)
                notes.append(f"context trimmed  -{int(saved / CHARS_PER_TOKEN):,} tokens of old tool output".replace(",", "."))
                used = after
        if used >= COMPACT_AT * self.num_ctx and used >= self._compact_retry_at:
            note = self.compact()
            if note:
                notes.append(note)
                used = self.estimate()
            else:                                            # failed (a model call): not again every step, only after 5% more
                self._compact_retry_at = used + 0.05 * self.num_ctx
        if used >= EMERGENCY_AT * self.num_ctx:
            # compaction failed (it is a model call) or one huge result filled the window: trim harder without a model, so the
            # next prompt still fits - an overflowing prompt is cut by the server from the front, silently losing the task.
            # Each trim re-reads the prompt, so a small one is only worth it right at the edge (seen live: 500-char trims every
            # step, each one a cache miss).
            masked, saved = mask_observations(self.messages, self.root, keep_last=4, min_chars=300)
            if saved >= MIN_MASK_SAVING_CHARS or (saved and used >= 0.95 * self.num_ctx):
                self.messages = masked
                self.last_prompt = max(0, self.last_prompt - int(saved / CHARS_PER_TOKEN))
                self._cache_reset()
                self.events.emit("emergency_mask", before=used, saved_chars=saved)
                notes.append(f"context nearly full - trimmed {int(saved / CHARS_PER_TOKEN):,} tokens of old tool output".replace(",", "."))
        return notes

    def force_pass(self) -> list[str]:
        """ONE manual compaction pass (the user asked for it): compact the middle of the conversation with the normal compactor; when there is
        too little to fold, mask old tool output instead. Same events as the automatic passes (compact / mask). Returns UI notes."""
        note = self.compact()
        if note:
            return [note]
        before = self.estimate()
        masked, saved = mask_observations(self.messages, self.root, keep_last=10)
        if not saved:
            return []
        self.messages = masked
        after = max(before - int(saved / CHARS_PER_TOKEN), 0)
        self.mask_floor = after + MASK_REARM * self.num_ctx
        self._cache_reset()
        self.events.emit("mask", before=before, after=after, saved_chars=saved, manual=True)
        return [f"context trimmed  -{int(saved / CHARS_PER_TOKEN):,} tokens of old tool output".replace(",", ".")]

    def fit(self, fraction: float = 0.8) -> bool:
        """The window just shrank (out-of-memory retry): get the history under `fraction` of it BEFORE it is sent - the server cuts an
        overflowing prompt from the FRONT, which drops the system prompt and the task. Mask old output, compact, then trim hard.
        True when it fits (also when it already did)."""
        limit = fraction * self.num_ctx
        if self.estimate() <= limit:
            return True
        self.mask_floor, self._compact_retry_at = 0.0, 0.0
        before = self.estimate()

        def mask(keep_last: int, min_chars: int):
            masked, saved = mask_observations(self.messages, self.root, keep_last=keep_last, min_chars=min_chars)
            if saved:
                self.messages = masked
                self._cache_reset()

        for shrink in (lambda: mask(10, 700), self.compact, lambda: mask(4, 300), lambda: mask(2, 100)):
            shrink()
            self.last_prompt = self.last_eval = 0           # the token counts of the last reply describe the old history
            self._since = sum(len(str(m.get("content", ""))) for m in self.messages)
            if self.estimate() <= limit:
                break
        after = self.estimate()
        self.events.emit("ctx_fit", before=before, after=after, num_ctx=self.num_ctx, fits=after <= limit)
        return after <= limit

    def _cache_reset(self) -> None:
        if self.on_clear_cache:
            try:
                self.on_clear_cache()
            except Exception:  # noqa: BLE001
                logger.debug("cache reset failed", exc_info=True)

    def _head_end(self) -> int:
        last_user = max((i for i, m in enumerate(self.messages) if m.get("role") == "user"), default=-1)
        return last_user + 1

    def compact(self, force_head: int | None = None) -> str | None:
        """Fold the middle of the conversation into one work-log message. Head (up to the newest user message) and
        the last few messages stay verbatim. Returns a UI note, or None when there was nothing to do / it failed."""
        msgs = self.messages
        head_end = self._head_end() if force_head is None else force_head
        cut = max(len(msgs) - KEEP_TAIL, head_end)
        while cut < len(msgs) and msgs[cut].get("role") != "assistant":
            cut += 1
        middle = msgs[head_end:cut]
        keep = [m for m in middle if pinned(m)]                     # the brief card / stage goal: kept verbatim, right after the head
        middle = [m for m in middle if not pinned(m)]
        fresh = [m for m in middle if not worklog._is_worklog(m)]
        if len(fresh) < worklog.MIN_MIDDLE_MESSAGES:
            return None
        before = self.estimate()
        # A log that a history fold put into the head (before the newest request) would otherwise stay there for good next to
        # every later log (seen live: two logs of ~11k chars each in a 16k window, compaction could not get below 68%). Merge it.
        stuck = [i for i in range(head_end) if worklog._is_worklog(msgs[i])] if force_head is None else []
        logs = [msgs[i] for i in stuck] + [m for m in middle if worklog._is_worklog(m)]
        existing = "\n\n".join(str(m["content"]).split("\n", 1)[-1] for m in logs) or None
        if existing:
            existing, _ = worklog.split_tool_log(existing)
            existing = "\n\n".join(part.split("\nLEDGER", 1)[0].rstrip() for part in existing.split("\n\n" + worklog.WORKLOG_MARKER)) \
                if len(logs) > 1 else existing.split("\nLEDGER", 1)[0].rstrip()
        extras = self._extras()
        t0 = time.monotonic()
        log, via = None, "in_place"
        if self.summarize_in_place is not None:
            # The conversation model writes the log on the conversation as it is: that prompt is already in the server's cache, so
            # only the log is generated. The separate summary call below re-reads a 60k-token transcript first (~90 s) and evicts
            # the conversation from the cache.
            try:
                log = self.summarize_in_place(IN_PLACE_PROMPT + (f"\n\nHarness state to keep:\n{extras}" if extras else ""))
            except Exception as exc:  # noqa: BLE001 - fall back to the separate summary call
                logger.warning("in-place compaction failed: %s", exc)
                log = None
        if not log:
            via = "background"
            text = worklog.render(fresh)
            if existing:
                text = f"EARLIER LOG:\n{existing}\n\nNEW ACTIVITY TO FOLD IN:\n{text}"
            requests = "\n---\n".join(worklog._clip(r, 1500) for r in self.user_requests[-6:])
            text = f"USER REQUESTS (verbatim, oldest first):\n{requests}\n\n{text}" + (f"\n\n{extras}" if extras else "")
            try:
                log = "".join(self.llm.chat(
                    messages=[{"role": "system", "content": CODE_WORKLOG_SYSTEM}, {"role": "user", "content": text}],
                    model=self._compact_model(), stream=False, background=True)).strip()
            except Exception as exc:  # noqa: BLE001 - compaction must never break a run
                logger.warning("compaction failed: %s", exc)
                self.events.emit("compact_failed", error=str(exc)[:200])
                return None
        if not log:
            return None
        ledger = self.ledger.render(self.root)
        head = [m for i, m in enumerate(msgs[:head_end]) if i not in stuck]
        kept = {str(m.get("content", "")) for m in [*head, *msgs[cut:]] if m.get("role") == "user"}
        # requests still visible verbatim in the head need no second copy (it cost ~1-5k chars in every log)
        asked = "\n---\n".join(worklog._clip(r, 1200) for r in self.user_requests[-4:] if r not in kept)
        note = {"role": "system",
                "content": f"{worklog.WORKLOG_MARKER} (auto-compacted from {len(middle)} earlier steps; the LEDGER is verified "
                           f"against the disk, files are recoverable with read_file/outline):\n{log}\n\n{ledger}"
                           + (f"\n\n{extras}" if extras else "")
                           + (f"\n\nEARLIER REQUESTS OF THE USER (verbatim, newest last):\n{asked}" if asked else "")}
        self.messages = [*head, *keep, note, *msgs[cut:]]
        self.last_prompt = 0
        self.last_eval = 0
        self._since = sum(len(str(m.get("content", ""))) for m in self.messages)
        self.mask_floor = self.stage_mask_floor = 0.0
        self.stage_mask_log, self.stage_mask_thrash = [], False
        self._compact_retry_at = 0.0
        self._cache_reset()
        after = self.estimate()
        self.events.emit("compact", before=before, after=after, folded=len(middle), seconds=round(time.monotonic() - t0, 1), via=via)
        return f"context compacted  {before:,} → ~{after:,} tokens".replace(",", ".")

    def _compact_model(self) -> str:
        """A second model on the GPU makes Ollama evict the main one (a reload of 10-60 s, and the prompt cache is gone): the fallback
        summary is written by the main model unless coding.compact_model names another."""
        explicit = (self.config.get("coding") or {}).get("compact_model")
        return explicit or (self.model_fn() if self.model_fn else None) or self.config["models"]["background"]

    def _extras(self) -> str:
        if self.extras is None:
            return ""
        try:
            return str(self.extras() or "").strip()
        except Exception:  # noqa: BLE001 - extra state is a bonus
            logger.debug("compaction extras failed", exc_info=True)
            return ""

    def fold_history(self) -> str | None:
        """At the start of a new prompt: if earlier turns already fill the window, fold them (head excludes the new
        user message, which has just been added)."""
        if self.fill() < FOLD_AT:
            return None
        lead = 0
        while lead < len(self.messages) and self.messages[lead].get("role") == "system" and not worklog._is_worklog(self.messages[lead]):
            lead += 1
        last_user = self._head_end() - 1
        if last_user <= lead:
            return None
        # fold everything between the leading system messages and the newest user message
        old = self.messages
        before = self.estimate()
        saved_tail = old[last_user:]
        saved_lead = old[:lead]
        self.messages = old[:last_user]
        self._since = 0
        note = self.compact(force_head=lead)
        if note is None:
            self.messages = old
            return None
        self.messages = [*self.messages, *saved_tail]
        self.events.emit("fold", before=before, after=self.estimate())
        return note.replace("compacted", "history folded")
