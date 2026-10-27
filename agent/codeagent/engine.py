"""The coding engine: one prompt in, a verified piece of work out.

The loop is the chat agent's, rebuilt around what went wrong in a real 3-hour session:
  * `think=False` on tool steps (no hidden reasoning pass per step); thinking is switched on only when stuck,
  * several tool calls per step are honoured, reads can be batched, so a step buys more,
  * the context manager keeps the prompt prefix stable (see context.py) and logs every mask/compaction,
  * the harness - not the model - runs the checks before it accepts "done" (a bounded number of times, then it
    accepts and says what is unproven),
  * the reply is streamed, so a generation that degenerates into repeating itself is stopped early instead of
    running to the token cap,
  * repeated failures trigger a web lookup (autonomy.StuckWatcher) and a thinking step,
  * everything goes through the per-repo policy (ask / medium / auto, allow/deny lists, network, installs).
"""

from __future__ import annotations

from agent import brand

import difflib
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from rich.markup import escape

from agent import autonomy, plan as plan_mod, toolargs, undo, verify
from agent.codeagent import stepbudget, brief as brief_mod, prose as prose_mod, checks as checks_mod, cursor as cursor_mod, edits, fileseen, intake as intake_mod, loopguard, phase as phase_mod, policy as policy_mod, profile as profile_mod, reviewer, spill as spill_mod, sprawl as sprawl_mod, stdpytest, tuigate, tuismoke
from agent.codeagent.context import ContextManager, _parse_call, compact_call
from agent.codeagent.eventlog import EventLog
from agent.codeagent.livectl import LiveControl, clip as _live_clip
from agent.codeagent.serverwait import DEFAULT_WAIT_S, wait_for_server
from agent.codeagent.lookact import (  # noqa: F401 - re-exported: sessionstore and tests import them from here
    LOOK_BLOCK_AT, LOOK_TOOLS, READ_VOLUME_NUDGE, TEST_STREAK_NUDGE, LookAct, Reaction, classify, is_scratch_path, readonly_shell, real_edit, run_changed)
from agent.codeagent.tools import KINDS, PROTECT_MARK, READ, SHELL, WRITE, CodeTools, cap_output, filter_split
from agent.llm import GenerationAborted, revive_server, server_url_of, streaming_chat
from agent.safety import RiskLevel, block_reason, classify_command

logger = logging.getLogger(__name__)

DENIED = "The user declined this action, so it was NOT performed. Do not claim it happened; offer an alternative or ask. Scratch experiments belong in .scratch/ inside the repo, not in /tmp or another folder."
MAX_GATE_NUDGES = 12               # HARD ceiling of "not done yet" messages per finish attempt (reset by progress), all gates together: nothing can loop forever
GATE_CONSECUTIVE = 4               # the real cap: "not done yet" messages in a row WITHOUT a tool call in between (a real call resets it); then the answer is accepted - or, with open acceptance items, ends as "gate"
SAMPLING = {"temperature": 0.25, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.0}
THINK_PREDICT = 4000                # hard cap of a reasoning pass. Measured live: one forced pass ran 8.5k tokens = 4.7 min and changed nothing; 12000 was up to 6.7 min
NUM_PREDICT = 8192                  # ~27 KB of code in one tool call; bigger files go in chunks (append_file). Measured: runaway generations of 16k and 32k tokens (14 min each, no result), and one 16k generation in a real run cost ~5 min at 54 tok/s - 8k loses at most ~2.5 min
REPEAT_SPAN, REPEAT_MAX_PERIOD = 8000, 2000     # degenerate = the last 8000 chars are one block repeated 4+ times
FORCED_STOPS = frozenset({"loop", "verify_budget", "budget", "announce", "stage_cap"})     # stops without a model answer that still get the one tool-less summary turn (+ error/'malformed')
WEB_TOOLS = ("web_search", "web_fetch")
SPILL_OWN = frozenset({"read_file", "read_files", "outline", "show_symbol", "run", "run_tests", "lint", "search"})   # not spilled again here: a repo file has its own line ranges, the others spill inside CodeTools
WEB_WORDS = ("http", "www", "search", "online", "docs", "documentation", "latest", "current version", "look up", "google")
WEB_NOT_NEEDED = "Web lookup not needed yet: write the code first; use web only for an unfamiliar library/API or after the same error happened twice."
LOOP_NOTE, LOOP_HINT, LOOP_ABORT, LOOP_WINDOW = 2, 3, 4, 16   # similar call+result within the last 16 calls: a note at 2, the nudge + one reasoning pass at 3, the segment ends at 4
AUTO_EXTEND_MAX = 15               # step budget windows added without asking in a row, counted from the last real progress (60 steps each by default)
MALFORMED_ABORT = 5                # consecutive replies that are neither a readable tool call nor an answer (garbled call, cut off, empty, thinking only): the segment ends as error/'malformed'
SERVER_ATTEMPTS = 3                # tries per model call when the server is down or breaks the reply (each one after a wait)
SPRAWL = "sprawl"                  # GenerationAborted reason of sprawl.SprawlGuard
_SERVER_TEXT = re.compile(r"unexpected eof|connection refused|runner|terminated")
OUTSIDE_READ_TOOLS = frozenset({"read_file", "read_files", "outline", "show_symbol"})        # reads that name a file: outside a ranged part they count double
OUT_OF_SCOPE_READ_WEIGHT = 2        # coding.out_of_scope_read_weight: look-streak steps one such read counts
STRAY_TOOLS = frozenset({"search", "run", "list_dir"})       # investigation of a ranged part: its first STRAY_LOOKS count STRAY_LOOK_WEIGHT steps (also during the free steps)
STRAY_LOOKS, STRAY_LOOK_WEIGHT = 8, 2                        # coding.stray_looks / coding.stray_look_weight
ANNOUNCE_MAX = 8                   # "you announced a next step but stopped" continuations per finish attempt (reset by progress)
COMPACT_PREDICT = 3072             # the work log written in place
CLAIM_WORDS = re.compile(r"(?i)\b(verified|tested|reachable|all (?:endings|tests|features|requirements|items)|every|everything|works?|working|passes|passing|alle|gepr\w+|getestet|funktioniert)\b")
_TIMING = re.compile(r"\(\d+(?:\.\d+)?s\)")
_WEAK_CHECKS = ("python3 -m py_compile", "ruff check --select", "node --check", "bash -n", "gofmt -e")      # syntax-only checks prove nothing ran

SYSTEM = """You are a senior software engineer working autonomously in the user's repository (your working directory). \
Do the task end to end: explore, change, run, verify.

How to work:
- Start from the PROJECT block and repo map below. Explore with `repo_map`, `outline`, `search` and ranged `read_file` (`read_files` batches small files); do NOT read big files whole, and do not re-read a file you already have.
- Change existing code with small edits; never rewrite a large existing file. Two ways: `edit_file` (search/replace, when you have the exact old text) or - without quoting anything - `show_symbol(name, path)` (one function/class/dict with line numbers and a #hash) then `replace_lines(path, start, end, new_text, expect=#hash)` / `insert_after(path, new_text, symbol=...)`. `where_used(name)` finds every caller.
- Big new content goes into a NEW file (`write_file`), wired in with small edits - the default for a new feature, a new character, a data table. A file over ~250 lines: write a skeleton first (imports, signatures, docstrings, bodies `...`), then fill one function per call with replace_lines.
- A syntax gate runs on every write and tells you the exact error line - fix it with a small `edit_file`.
- A command run costs real seconds: never repeat a run whose output you already have; change something first. A long tool result is kept whole in .scratch/spill/ and ends with a `full output:` line naming the file and the range to read - `read_file` that range (or search the file); never re-run a command to see the middle of its output.
- Prove it: run the program or the tests with `run` / `run_tests` (feed scripted input with `stdin`). Nothing counts as done until you ran it and saw it work.
- Interactive or slow programs (menus, REPLs, anything that prints slowly): test the logic with a small script that imports the module, calls the functions and prints or asserts the state - not by replaying long sessions. Pass no_sleep=true to `run` when the program pauses or types text slowly.
- If the same error repeats, change approach: read the actual error, check the real API with `outline`/`search`, or look it up (`web_search`).
- Decide every code question yourself. Ask the user (`ask_user`) only about real product decisions or when you are blocked.
- Throwaway copies and test runs (anything that writes data files): use ./.scratch/ inside the repo - create it, copy files into it, delete it; never let a test read or modify the user's real data files.
- Stay inside the repo and use relative paths (the root is in the PROJECT block). Obey the user's constraints exactly (directories, scope, "plan first", language).
- If a call is refused or "NOT RUN", never repeat it: change it.
- Be honest: if something is not done or not verified, say so plainly. Final answer: what changed, how you verified it, what is left. Short."""


def repeating(text: str, span: int = REPEAT_SPAN, max_period: int = REPEAT_MAX_PERIOD) -> bool:
    """True when the last `span` chars of `text` are one block repeated over and over (a degenerate generation).
    Legitimate code - even a long data table - is never exactly periodic over 8000 chars; a model stuck in a loop is."""
    if len(text) < span:
        return False
    tail = text[-span:]
    pos = span - 64
    probe = tail[pos:]
    end = pos + 63                                        # rfind window that excludes the probe itself
    for _ in range(24):                                   # earlier occurrences of the last 64 chars are the period candidates
        i = tail.rfind(probe, 0, end)
        if i < 0 or pos - i > max_period:
            return False
        period = pos - i
        if tail[period:] == tail[:-period]:
            return True
        end = i + 63
    return False


@dataclass
class TurnResult:
    text: str | None
    steps: int = 0
    files: list[str] = field(default_factory=list)
    reason: str = ""                       # why a stopped="error" segment ended ("malformed": too many unusable replies in a row)
    stopped: str = ""                      # "", "budget", "loop", "verify_budget" (phase.py: the post-edit verification budget), "interrupted", "error", "stage_cap" (stages.py), "fresh_context" (the supervisor's measured slow-prefill rule), "announce" (announce limit),
                                           # "gate" (the nudge budget is spent and acceptance items are still open; the text is labelled NOT VERIFIED)
    verified: bool | None = None
    not_verified: list[str] = field(default_factory=list)     # checks that did not pass when the answer was accepted (checks.py)


class _Working:
    """Status text that updates itself: Ollama sends nothing while a long tool call is being generated (the whole file
    arrives at once), so the screen would look frozen. Shows seconds, estimated tokens and context fill."""

    def __init__(self, engine: "CodeEngine", label: str) -> None:
        self.engine, self.label, self.t0 = engine, label, time.monotonic()

    def __rich_console__(self, console: Any, options: Any):
        from rich.text import Text

        e = self.engine
        secs = time.monotonic() - self.t0
        tps = e.avg_tps()
        made = f"{e.live_pieces:,} tokens" if e.live_pieces else f"~{int(secs * tps):,} tokens at ~{tps:.0f} tok/s"     # streamed text is counted, a native tool call is estimated
        yield Text(f"{self.label}…  {secs:.0f}s  {made}  ·  context {e.ctx.fill() * 100:.0f}% of {e.num_ctx // 1024}k".replace(",", "."),
                   style="dim")


def _one_line(text: str, limit: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


class CodeEngine:
    def __init__(
        self, root: Path | str, config: dict[str, Any], llm: Any, console: Any, *, ask_fn: Callable[[str], str] | None = None,
        approve_fn: Callable[[str, str], bool] | None = None, web_tools: list | None = None, notes: Any = None,
        note_key: str | None = None, events: EventLog | None = None, interactive: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.config, self.llm, self.console = config, llm, console
        self.interactive = interactive
        self.pol = policy_mod.for_repo(self.root)
        self.model = self.pol.model or (config.get("code") or {}).get("model") or config["models"]["chat"]
        raw_ctx = self.pol.ctx or int((config.get("models") or {}).get("chat_num_ctx", 16384))
        self.num_ctx = raw_ctx
        self.events = events or EventLog(session=f"{self.root.name}-{int(time.time())}")
        prof = profile_mod.detect(self.root, self.pol.test_cmd)
        prof._override = bool(self.pol.test_cmd)
        self.ctx = ContextManager(llm, config, self.num_ctx, self.events, self.root)
        self.approve_fn = approve_fn or self._ask_yes_no
        self._explicit_approve = approve_fn is not None
        policy_mod.configure(config)                           # coding.deny_paths / allow_paths (extend-only)
        self._noed: dict[tuple[str, str], str] = {}            # jail answers by call: a repeat gets the identical NO without policy work
        self._declined: set[tuple[str, str]] = set()           # (tool, sha1 of the whole call) the user's absence already refused
        self.undo = undo.UndoStack()
        self.plan = plan_mod.Holder()
        self.notes, self.note_key = notes, note_key
        note_tool = self._make_note_tool() if notes is not None and note_key else None
        plan_tool = self._gated_mark_done(plan_mod.make_tool(self.plan, console))
        self.checks = checks_mod.CheckSet(self.root, lambda: self.session_id)
        self._check_returns: dict[str, int] = {}               # final-gate returns per failing check in this prompt
        self._gate_streak = 0                                  # gate nudges in a row with no tool call from the model in between
        self.tui_last: tuigate.GateResult | None = None        # the last crash-gate result (tuigate.py): shown by verified.build, never re-run for the same code
        self._tui_stamp = ""                                   # content stamp of the edited .py files that result was measured on
        self.std_last: stdpytest.Finding | None = None        # the last 'plain pytest' finding (stdpytest.py), None = fine / not applicable
        self._std_stamp: str | None = None                     # content stamp it was measured on (None = never measured)
        self.std_runner: Any = None                            # test hook: replaces the engine's _exec for the check
        self.tools = CodeTools(self.root, prof, self.ctx, console, web_tools=web_tools, plan_tool=plan_tool,
                               note_tool=note_tool, ask=ask_fn, run_timeout=int((config.get("coding") or {}).get("run_timeout", 60)),
                               check_cmd=(config.get("coding") or {}).get("check_cmd"),
                               protect=(config.get("coding") or {}).get("protect"),
                               syntax_reject=str((config.get("coding") or {}).get("syntax_reject", True)).strip().lower() not in ("false", "off", "0", "no"),
                               read_max_chars=self._limit("read_max_chars", 0),
                               smoke_timeout=max(1, self._limit("tui_smoke_timeout", tuismoke.DEFAULT_TIMEOUT)),
                               smoke_max_chars=max(500, self._limit("tui_smoke_max_chars", tuismoke.MAX_CHARS)))
        self.tools.smoke_allow = self._allowed_without_asking     # the automatic tui_smoke after a `run` of a terminal program (tuihint.py) obeys the policy too
        self.registry = self.tools.registry()
        self.registry.register(self._make_check_tool())
        self.ask_fn = ask_fn                       # reads one line from the human (None: the console's input); only used while a terminal is attached
        self.registry.register(self._make_ask_tool())
        self.prose_calls = 0                       # write_prose calls in this prompt (coding.prose_max_calls)
        if prose_mod.enabled(config):              # off by default: no tool, no prompt sentence
            self.registry.register(self._make_prose_tool())
        self.schema = self.registry.to_ollama_schema()
        self.ctx.summarize_in_place = self._summarize_in_place
        self.ctx.model_fn = lambda: self.model
        self.ctx.extras = self._compaction_extras
        self.ctx.stage_cap_fn = lambda: self.stage_cap
        self._in_place_failures = 0
        self.watcher = autonomy.StuckWatcher(max_lookups=int((config.get("autonomy") or {}).get("max_lookups_per_turn", 3)))
        self.think_steps = 0
        self.on_step: Any = None                   # session.py hooks the autosave in here (a killed run keeps its conversation)
        self.live = LiveControl()                  # live-UI control state (livectl.py): steer queue, pause / stop / compact requests, extra step hooks, "now"
        self.capture_thinking = False              # forward the model's reasoning text to subscribers (emit_live "thinking"); never persisted
        self._turn_t0 = 0.0
        self._sleep, self._clock, self._probe = time.sleep, time.monotonic, None     # the server wait; tests replace them
        self.live_pieces = 0                       # streamed pieces (~tokens) of the reply being generated
        # loop guard history: (tool, call hash, result hash, errorish, run stem hash, call text, result text, outcome line). The texts allow
        # the fuzzy "nearly the same" match; an entry restored from a saved session has only the hashes (None texts) and matches exactly.
        self._sigs: list[tuple[str, str, str, bool, str, str | None, str | None, str | None]] = []
        self.dead_ends = loopguard.DeadEnds()      # attempts repeated with the same result (loopguard.py); shown to every fresh context
        self._interactive_hinted = False           # the "drive it from a script" hint is given once per session
        self._loop_note_at = -99
        self.family = loopguard.FamilyGuard()          # same-kind-of-command / read-only run spam counters (loopguard.py)
        self.pattern = loopguard.PatternGuard()        # same tool with changing arguments / alternating calls / re-reads in a window (loopguard.py)
        self._abort_loop = 0
        self.phase = phase_mod.VerifyPhase()           # the verification-spiral phase of the current prompt (phase.py)
        self._guard_spoke = False                      # the family guard said something on the current call
        # look vs. act (reading paused after a long read streak, wander / edit-gap notes): one module, see lookact.py
        self.lookact = LookAct(read_lines=lambda: self.tools.read_lines, reset_lines=lambda: setattr(self.tools, "read_lines", 0),
                               limit_fn=lambda: self._limit("look_block_at", LOOK_BLOCK_AT),
                               chunk_fn=self._chunk_limit, hint_fn=self._chunk_hint)
        self._started = False
        self._head: list[str] | None = None        # system prompt, project memory and repo map of this session (start): fixed so the prompt prefix stays cached
        self.real_writes = 0                       # successful writes of real project files (not scratch): the stage budget's edit count
        self.step_ceiling = self.pol.max_steps      # upper bound of the step budget (the user's value when steps_set)
        self.budget_now = self.pol.max_steps        # the step budget of the running prompt (meters)
        self.budget_source = "user" if self.pol.steps_set else "fixed" if not self.config.get("coding", {}).get("adaptive_steps", True) else "auto"
        self._progress = 0                         # successful edits of real project files + confirmed plan items (budget auto-extension)
        self._announce_continues = 0
        self._announce_limit = False
        self._announce_at = -99
        self._calls_run = 0                        # tool calls run so far (an announcement is not repeated without a tool step in between)
        self._announce_calls = -1
        self._ok_calls = 0                         # tool calls whose result was not an error (a successful call ends a run of unreadable replies)
        # saved-session bookkeeping (sessionstore.py): id, start, what the last prompt ended with, a note for the model after a resume
        self.session_id = ""
        self.session_created = time.time()
        self.in_turn = False
        self.last_stopped = ""
        self.resume_note: str | None = None
        self.session_notes: list[str] = []         # project notes saved during this session: they ride along in every compaction
        self.asks = 0                              # ask_user answers given in this prompt / stage (cap intake.MAX_ASKS)
        self._task_text = ""                       # the user's last real prompt (not an AUTOPILOT / [harness] continuation): the web check reads it
        self._prompt_fails = 0                     # failed tool runs since that prompt (non-zero exit / error shape): 2 allow a web lookup
        self._ask_goal: str | None = None
        self.intake_seen: set[str] = set()         # sha1 of task texts the intake (intake.py) already looked at: never asked twice
        self.split_cache: dict[str, Any] = {}      # sha1 of task text -> the model's stage split (stages.py): a repeat does not ask again
        self.briefs: list[str] = []                # brief cards of this session's big tasks (brief.py), newest last; pinned in the context
        # stages (stages.py): all off by default, so a plain prompt runs exactly as before
        self.stage_plan: Any = None                # the StagePlan of a staged task (saved with the session)
        self.stage_goal: str | None = None         # set while a stage's fresh context is in use (its brief card is the whole task's)
        self.stage_scope: list[str] = []           # the files the running sub-stage must handle (stages.py sets it for one segment; scope gate)
        self.stage_changed0: set[str] = set()      # scope files changed in earlier segments of this stage
        self.stage_marks: dict[str, tuple] = {}    # ledger marks at the start of the segment: what differs afterwards was changed in it
        self.stage_waived: set[str] = set()        # scope files the model said need no change (mark_done / project_note naming them)
        self.stage_obj: Any = None                 # the Stage of the running segment (stages.py; verified.build shows its cursor)
        self.stage_edited: dict[str, list[list[int]]] | None = None    # that stage's per-file edited line ranges: filled by every write while it runs
        self._stray_looks = 0                      # search / run / list_dir looks of the current ranged part that were weighted (STRAY_LOOKS)
        self._outside_noted = False                # the one 'Outside this part' note of a ranged stage was given in this segment
        self.stage_chunk = False                   # the running stage is an editing-by-passes stage (cursor.py): reading is paused earlier
        self.stage_seen: dict[str, dict[str, Any]] | None = None       # that stage's per-file read record (fileseen.py): filled by every read while it runs
        self.tools.on_write = self._record_write
        self.tools.on_read = self._record_read
        self.tools.next_hint = self._next_unchanged
        self.stage_cap: int | None = None          # absolute prompt cap in tokens: a step that measured more ends the prompt with "stage_cap"
        self.prompt_peak = 0                       # largest measured prompt since the stage runner reset it
        self.segment_end_check: Callable[[], str | None] | None = None    # the supervisor's measured rule: a reason ends the segment at a step boundary

    @property
    def profile(self) -> profile_mod.Profile:
        return self.tools.profile

    # the look/act counters, under the names the session store and older tests use
    _look_streak = property(lambda self: self.lookact.look_streak, lambda self, v: setattr(self.lookact, "look_streak", v))
    _wander_nudges = property(lambda self: self.lookact.wander_nudges, lambda self, v: setattr(self.lookact, "wander_nudges", v))
    _edit_gap = property(lambda self: self.lookact.edit_gap, lambda self, v: setattr(self.lookact, "edit_gap", v))
    _edit_nudges = property(lambda self: self.lookact.edit_nudges, lambda self, v: setattr(self.lookact, "edit_nudges", v))

    # ── setup ───────────────────────────────────────────────────────────
    def avg_tps(self) -> float:
        """Decode speed of the recent steps (falls back to a typical 38 before anything was measured)."""
        rates = [r["gen"] / r["decode_s"] for r in self.events.of("step") if r.get("decode_s", 0) > 0.5 and r.get("gen", 0) > 30]
        return sum(rates[-6:]) / len(rates[-6:]) if rates else 38.0

    # ── live-UI API (thread-safe; takes effect at the next step boundary unless noted) ──────────────
    @property
    def now(self) -> str:
        """One line saying what the engine is doing right now (for a status line)."""
        return self.live.now

    def _set_now(self, text: str) -> None:
        if text != self.live.now:
            self.live.now = text
            self.events.emit_live("now", text=text)

    def add_step_hook(self, fn: Callable[[], None]) -> None:
        """Call `fn()` before every model call, in addition to the legacy `on_step` slot. Exceptions in a hook are swallowed."""
        self.live.add_hook(fn)

    def remove_step_hook(self, fn: Callable[[], None]) -> None:
        self.live.remove_hook(fn)

    def queue_user_message(self, text: str) -> None:
        """Steer a running turn: the text is added as a labelled user message before the next model call."""
        self.live.queue_message(text)

    def request_compact(self) -> None:
        """Force ONE compaction (or, when there is too little to fold, one masking pass) at the next step boundary."""
        self.live.request_compact()

    def request_pause(self) -> None:
        self.live.pause()

    def request_resume(self) -> None:
        self.live.resume()

    def request_stop(self) -> None:
        """End the running turn like Ctrl-C does ('interrupted', work so far kept) at the next step boundary, or between two streamed pieces
        of a reply being generated. Ignored while no turn runs."""
        if self.in_turn:
            self.live.stop()

    def _boundary(self) -> None:
        """Top of every loop pass: honour stop / pause, inject steering messages, run a requested compaction."""
        live = self.live
        if live.stopping:
            raise KeyboardInterrupt
        if live.paused:
            self.events.emit_live("pause", paused=True)
            live.wait_while_paused(lambda: self._set_now("paused"))
            self.events.emit_live("pause", paused=False)
            if live.stopping:
                raise KeyboardInterrupt
        for text in live.drain_messages():
            self.ctx.add_user(f"[user message during the run] {text}")
            self.events.emit("steer", chars=len(text))
            self.events.emit_live("steer", text=_live_clip(text))
        if live.take_compact():
            self._set_now("compacting the context")
            notes = self.ctx.force_pass()
            self.events.emit("compact_requested", done=bool(notes))
            for note in notes or ["nothing to compact yet"]:
                self.console.print(f"[dim]  {note}[/dim]")

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._head is None:
            # Built once per session and then reused by every fresh stage context: the model server re-reads a prompt from the first byte that
            # differs, and the old code rebuilt the PROJECT MEMORY (new notes) and the REPO MAP (new files) in front of the stage goal at every
            # stage start. What changed since is carried by the VERIFIED STATE (FILES, SYMBOLS) and the PROJECT NOTES SAVED IN THIS SESSION.
            extra = "\n" + prose_mod.PROMPT_SENTENCE if prose_mod.enabled(self.config) else ""
            if tuismoke.repo_uses_tui(self.root):               # two short sentences, only for a repo that imports curses / textual / urwid / pygame / prompt_toolkit / blessed
                extra += "\n" + tuismoke.PROMPT_SENTENCE
            head = [SYSTEM + extra + "\n\n" + self.profile.text()]
            if self.notes is not None and self.note_key:
                msg = self.notes.system_message(self.note_key)
                if msg:
                    head.append(msg)
            head.append("REPO MAP (snapshot at session start; new files show up in tool results):\n" + edits.repo_map(self.root, 3500))
            self._head = head
        for text in self._head:
            self.ctx.add({"role": "system", "content": text})

    def _make_note_tool(self):
        from agent.tools.registry import Tool

        def project_note(note: str = "", **_: Any) -> str:
            self._waive_from(note)
            if self.notes.add(self.note_key, note):
                self.session_notes.append(_one_line(note, 200))
                self.console.print(f"[dim]  note  {_one_line(note, 90)}[/dim]")
                return "Saved; it will be in context in future sessions of this project."
            return "Not saved (too short, or already known)."

        return Tool(name="project_note", description="Save a durable fact about THIS project (how to run/test it, a convention, a gotcha, a decision). One short sentence.",
                    parameters={"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}, handler=project_note, risk="safe")

    def _make_ask_tool(self):
        from agent.tools.registry import Tool

        def ask_user(question: str = "", **_: Any) -> str:
            q = _one_line(question, 300)
            if not q:
                return "error: ask_user needs question."
            if not intake_mod.tty():
                return intake_mod.NO_HUMAN                     # headless / autopilot without a terminal: never blocks
            if self.asks >= intake_mod.MAX_ASKS:
                return intake_mod.LIMIT_REACHED
            self.asks += 1
            self.console.print(f"  [yellow]?[/yellow] {escape(q)}")
            try:
                ans = str((self.ask_fn or (lambda _q: self.console.input("  ▸ ")))(q) or "").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            return ans or intake_mod.NO_ANSWER

        return Tool(name="ask_user", description="Ask the human ONE short question when you are blocked or a choice would change the result. "
                    "Not for confirmations of routine steps.",
                    parameters={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
                    handler=ask_user, risk="safe")

    def _make_prose_tool(self):
        from agent.tools.registry import Tool

        def write_prose(request: str = "", style_hint: str = "", max_words: Any = prose_mod.DEFAULT_WORDS, **_: Any) -> str:
            if self.prose_calls >= self._limit("prose_max_calls", prose_mod.DEFAULT_MAX_CALLS):
                return prose_mod.LIMIT.format(n=self.prose_calls)
            self.prose_calls += 1
            return prose_mod.write_prose(self.llm, self.config, self.root, request, style_hint, max_words, events=self.events)

        S = {"type": "string"}
        return Tool(name="write_prose", description=prose_mod.TOOL_DESCRIPTION,
                    parameters={"type": "object", "properties": {"request": S, "style_hint": S, "max_words": {"type": "integer"}}, "required": ["request"]},
                    handler=write_prose, risk="safe")

    def _approve(self, description: str, reason: str) -> bool | None:
        """The one place that asks the user. None = nobody can be asked (no terminal on stdin): the caller treats that
        as a refusal at once, so a run started from a script or another program can never block on input."""
        if not self._explicit_approve and not sys.stdin.isatty():
            return None
        self.live.waiting = "approval"
        self._set_now(f"waiting for your approval: {description}"[:200])
        try:
            return self.approve_fn(description, reason)
        finally:
            self.live.waiting = ""

    def _ask_yes_no(self, description: str, reason: str) -> bool:
        self.console.print(f"  [yellow]?[/yellow] {description}" + (f"  [dim]({reason})[/dim]" if reason else ""))
        try:
            ans = self.console.input("    run it? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return False
        return ans in ("y", "yes", "j", "ja")

    # ── display ─────────────────────────────────────────────────────────
    def describe(self, name: str, args: dict[str, Any]) -> str:
        a = args or {}
        p = a.get("path") or a.get("src") or ""
        if name == "run":
            return f"run  {_one_line(a.get('command', ''), 100)}"
        if name == "tui_smoke":
            return f"tui  {_one_line(a.get('command', ''), 90)}  keys {len(a.get('keys') or [])}"
        if name in ("write_file", "append_file"):
            return f"write  {p}"
        if name in ("edit_file", "apply_patch", "delete_file", "move_file"):
            return f"{name.replace('_file', '')}  {p or 'patch'}"
        if name == "replace_lines":
            return f"edit  {p} {a.get('start', '')}-{a.get('end', '')}"
        if name == "insert_after":
            return f"insert  {p} after {a.get('symbol') or a.get('line', '')}"
        if name in ("show_symbol", "where_used"):
            return f"{'symbol' if name == 'show_symbol' else 'uses'}  {a.get('name', '')}" + (f" in {p}" if p else "")
        if name in ("read_file", "outline", "list_dir"):
            rng = f" {a.get('start', '')}-{a.get('end', '')}" if a.get("start") or a.get("end") else ""
            return f"read  {p or '.'}{rng}"
        if name == "read_files":
            return f"read  {len(a.get('paths') or [])} files"
        if name == "search":
            return f"search  {_one_line(a.get('pattern', ''), 60)}"
        if name in ("web_search", "web_fetch"):
            return f"web  {_one_line(a.get('query') or a.get('url') or '', 70)}"
        return f"{name}  {_one_line(json.dumps(a, ensure_ascii=False), 70)}"

    def _result_line(self, name: str, result: str) -> str:
        first = next((ln for ln in str(result).splitlines() if ln.strip()), "")
        if name in ("run", "run_tests", "lint", "tui_smoke"):
            return first
        if name in ("write_file", "edit_file", "apply_patch", "append_file", "replace_lines", "insert_after"):
            return first + ("  ✗ syntax" if "SYNTAX GATE FAILED" in result else "")
        return _one_line(first, 110)

    # ── gates ───────────────────────────────────────────────────────────
    def decide(self, name: str, args: dict[str, Any]) -> policy_mod.Decision:
        kind = KINDS.get(name)
        if kind == READ:
            given = [args.get("path"), args.get("src"), *(args.get("paths") if isinstance(args.get("paths"), list) else [])]
            worst = policy_mod.Decision("allow")
            for p in (str(x) for x in given if x):
                d = policy_mod.check_path(self.pol, self.root, p, write=False)
                if d.verdict == "block" and d.reason.startswith("NO:"):
                    return d
                if not policy_mod.inside(self.tools.resolve(p), self.root) and d.verdict != "allow":
                    worst = policy_mod.Decision("ask", "outside the repo")
            return worst
        if kind == WRITE:
            if name == "apply_patch":
                paths = [f[0] for f in edits.parse_patch(str(args.get("patch", "")))] or ["."]
            elif name == "move_file":
                paths = [str(args.get("src", ".")), str(args.get("dst", "."))]
            else:
                paths = [str(args.get("path", "."))]
            worst = policy_mod.Decision("allow")
            for pth in paths:
                d = policy_mod.check_path(self.pol, self.root, pth, write=True)
                if d.verdict == "block":
                    return d
                if d.verdict == "ask":
                    worst = d
            return worst
        if kind == SHELL:
            if name == "git_commit":
                return policy_mod.Decision({"never": "block", "ask": "ask", "auto": "allow"}[self.pol.commit], "git commit policy: " + self.pol.commit)
            command = str(args.get("command", "")) if name in ("run", "tui_smoke") else (self.profile.test_cmd or "") if name == "run_tests" else "ruff check"
            jailed = policy_mod.forbidden_in_command(command, self.root) if command else None
            if jailed:
                return policy_mod.Decision("block", jailed)
            level = classify_command(command, self.config) if command else RiskLevel.SAFE
            if name in ("run", "tui_smoke") and level is RiskLevel.BLOCKED:
                return policy_mod.Decision("block", block_reason(command, self.config))
            return policy_mod.check_command(self.pol, self.root, command, level.name, owned=self.tools.owned)
        return policy_mod.Decision("allow")

    def execute(self, name: str, args: dict[str, Any]) -> str:
        tool = self.registry.get(name)
        if tool is None:
            return f"Error: no such tool '{name}'. Tools: {', '.join(sorted(t['function']['name'] for t in self.schema))}"
        args, bad = toolargs.validate(tool, args)
        if bad:
            return bad
        if name in WEB_TOOLS and not self._web_allowed():
            self.events.emit("web_refused", tool=name)
            return WEB_NOT_NEEDED
        no_key = (name, _sha(_call_key(args)))
        if no_key in self._noed:
            self.console.print(f"  [red]blocked[/red] {self.describe(name, args)}  [dim](same call as before)[/dim]")
            return self._noed[no_key]
        paused = self.lookact.blocked(name, args)
        if paused:
            # Offering fewer tools would change the prompt head (a full re-read by the server), so the call is simply not run (lookact.py)
            self.events.emit("look_blocked", tool=name, streak=self._look_streak)
            return paused
        try:
            verdict = self.decide(name, args)
        except Exception as exc:  # noqa: BLE001 - an absurd path (too long, a symlink loop) must not end the turn
            verdict = policy_mod.Decision("block", f"could not check this call ({type(exc).__name__}: {str(exc)[:120]})")
        if verdict.verdict == "block":
            self.console.print(f"  [red]blocked[/red] {self.describe(name, args)}  [dim]{verdict.reason}[/dim]")
            if verdict.reason.startswith("NO:"):
                self._noed[no_key] = verdict.reason            # the jail: never a question, the same short answer every time
                return verdict.reason
            return f"BLOCKED: {verdict.reason}. Choose another approach."
        if verdict.verdict == "ask":
            key = (name, _sha(json.dumps(args, sort_keys=True, default=str)))
            if key in self._declined:
                return ("NOT RUN (asked before): this exact call was already refused. Do not send it again - change it (relative paths inside the repo, "
                        "no cd elsewhere) or choose another approach.")
            answer = self._approve(self.describe(name, args), verdict.reason)
            if answer is None:
                self._declined.add(key)
                self.console.print(f"  [dim](not run: {verdict.reason}; no terminal to ask)[/dim]")
                return f"NOT RUN: {verdict.reason}. Nobody is available to approve it. {self._not_run_advice(name, args, verdict.reason)}"
            if not answer:
                self.console.print("  [dim](declined)[/dim]")            # a real user may say yes next time: only auto-denials are remembered
                return DENIED
        t0 = time.monotonic()
        try:
            result = tool.handler(**args)
        except Exception as exc:  # noqa: BLE001 - handler errors go back to the model
            logger.debug("tool error", exc_info=True)
            result = f"Error running {name}: {exc}"
        result = str(result)
        if name not in SPILL_OWN and KINDS.get(name) != WRITE and len(result) > spill_mod.SPILL_MIN:
            big = len(result)
            result = self.tools.spill.clip(name, result)          # kept whole in .scratch/spill/, the model gets head+tail and where the rest is
            if len(result) < big:
                self.events.emit("spill", tool=name, chars=big)
        if PROTECT_MARK in result and result.startswith(("error:", "apply_patch failed:")):
            self._noed[no_key] = result                    # the player's data: the same call gets the identical answer, no work
        extra = {"cmd": str(args.get("command", ""))[:120]} if name in ("run", "tui_smoke") else {}
        self.events.emit("tool", name=name, seconds=round(time.monotonic() - t0, 2), ok=not result.startswith(("Error", "error")), **extra)
        return result

    def _web_allowed(self) -> bool:
        """The model's own web lookups: the task asked for one (a URL, docs, "latest" ...) or two runs failed since the prompt.
        (The automatic 'stuck: same problem 2x' lookup calls the tool directly and does not come through here.)"""
        text = f"{self._task_text}\n{self.stage_goal or ''}".lower()
        return self._prompt_fails >= 2 or any(w in text for w in WEB_WORDS)

    def _not_run_advice(self, name: str, args: dict[str, Any], reason: str) -> str:
        """What to change, matched to WHY it was not run. Seen live: `rm -rf .scratch && mkdir ... | tail` (relative paths, inside the
        repo) was told to "rewrite it with relative paths inside the repo" - it sent the same thing again."""
        command = str(args.get("command", "")) if name in ("run", "tui_smoke") else ""
        if command and re.search(r"(^|[;&|]\s*)rm\s", command):
            alone = (" `rm` inside .scratch/ runs without asking only when it comes first and its paths are plain (a `$VAR` set earlier in the same"
                     " command is fine, `$(...)` and pipes into rm are not); put other commands after it, not before."
                     if self.pol.mode == policy_mod.AUTO else "")
            return (f"Deleting with rm needs the user.{alone} To delete a repo file use delete_file (recoverable); to refresh a copy, "
                    "just copy over it (cp -f).")
        if "outside" in reason:
            return f"Rewrite the call so it stays inside the repo ({self.root}) with relative paths, or skip it."
        return "Do it another way (a different command or one of the file tools), or skip it."

    # ── the turn ────────────────────────────────────────────────────────
    def turn(self, user_text: str) -> TurnResult:
        self.start()
        self.tools.changes.clear()
        self.tools.evidence.clear()
        self.tools.dirty.clear()
        self.in_turn, self.last_stopped = True, ""
        self._turn_t0 = time.monotonic()
        self._announce_continues, self._announce_at, self._announce_limit, self._announce_calls = 0, -99, False, -1
        self._check_returns = {}
        self._gate_streak = 0
        self.phase.reset()
        if self.stage_goal is None:
            self.prose_calls = 0
            self.asks = 0                                      # a stage keeps its count over its segments (begin_stage)
        if self.resume_note:
            # once, right before the request: a system message after an assistant answer is invisible to the model
            self.ctx.add({"role": "system", "content": self.resume_note})
            self.resume_note = None
        if not user_text.startswith(intake_mod._DIRECTIVES):          # a continuation keeps the task and the failures of its prompt
            self._task_text, self._prompt_fails = user_text, 0
        self.ctx.add_user(user_text)
        folded = self.ctx.fold_history()
        if folded:
            self.console.print(f"[dim]  {folded}[/dim]")
        card = brief_mod.make(user_text) if self.stage_goal is None else None
        if card and card not in self.briefs:
            self._pin_brief(card)
        done_plan = self.plan.plan is not None and not self.plan.plan.open_items()
        if plan_mod.should_create(user_text) and (self.plan.plan is None or done_plan):
            # a finished list does not cover a new big task in the same session (seen live: the next four stages ran without one)
            with self.console.status("[dim]making an acceptance list…[/dim]", spinner="dots"):
                self.plan.plan = plan_mod.generate(self.llm, self.config, user_text)
            if self.plan.plan:
                self._insert_plan()
                for line in self.plan.plan.lines():
                    self.console.print(f"[dim]  plan {line.strip()}[/dim]")
        result = TurnResult(None)
        self._size_budget()
        try:
            self._loop(result)
            if self._forced_applies(result):
                self._forced_ending(result)
        except KeyboardInterrupt:
            result.stopped = "interrupted"
            self.console.print("\n[dim]interrupted - the work so far is kept (/undo reverts it)[/dim]")
        except Exception as exc:  # noqa: BLE001 - a harness bug ends this prompt, not the session (the conversation is kept)
            logger.exception("code turn failed")
            result.stopped = "error"
            self.console.print(f"[red]internal error: {type(exc).__name__}: {exc}[/red] [dim](work so far is kept; try again or /new)[/dim]")
        finally:
            self.in_turn, self.last_stopped = False, result.stopped
            self.live.clear_stop()
            self.live.resume()
            self._set_now("idle")
            changed = self.tools.changes.changed()
            self.undo.push(changed, self.tools.changes.checkpoint)
            result.files = [self.tools.rel(Path(p)) for p in changed]
            self.tools.changes.print_summary(self.console, self.root)
        return result

    def begin_stage(self, goal: str, verified_text: str, notes: str = "", *, reset_plan: bool = True) -> None:
        """A fresh context for a stage: system prompt, repo map, the brief card, the pinned STAGE GOAL, the earlier stages (`notes`) and the
        VERIFIED STATE as the first user turn. Ledger, change log, checks, look/act state and the project notes live on in the engine."""
        c = self.ctx
        c.messages = []
        c.last_prompt = c.last_eval = c._since = 0
        self.tools.spill.prune()                                    # the spill folder keeps its newest files only
        c.mask_floor, c._compact_retry_at, c.stage_mask_floor = 0.0, 0.0, 0.0
        c.stage_mask_log, c.stage_mask_thrash = [], False
        self._started = False
        self.start()
        if self.briefs:
            c.add({"role": "system", "content": self.briefs[-1]})
        c.add({"role": "system", "content": "STAGE GOAL (pinned by the harness; work on this stage only):\n" + goal})
        dead = self.dead_ends.text() if loopguard.DEAD_ENDS_HEADER not in verified_text else ""      # the VERIFIED STATE usually carries it already
        extra = [("EARLIER STAGES (carried by the harness):\n" + notes) if notes else "", dead,
                 ("PROJECT NOTES SAVED IN THIS SESSION:\n" + "\n".join(f"- {n}" for n in self.session_notes[-12:])) if self.session_notes else ""]
        for text in filter(None, extra):
            c.add({"role": "system", "content": text})
        if reset_plan:
            self.plan.plan = None
        elif self.plan.plan is not None:
            c.add({"role": "system", "content": self.plan.plan.system_message()})
        c.add({"role": "user", "content": verified_text})
        if goal != self._ask_goal:
            self.asks, self._ask_goal = 0, goal
        self.stage_goal, self.resume_note, self.think_steps = goal, None, 0
        self.lookact.begin_stage()
        self._outside_noted = False
        self._stray_looks = 0
        self._sigs.clear()
        self.family.reset_stage()
        self.pattern.reset_stage()
        self.tools.read_cache.clear()

    def _pin_brief(self, card: str) -> None:
        """The user's requirements, verbatim (brief.py), right before the request; the context manager never rewrites it. A session keeps
        the newest MAX_CARDS cards (an older one is dropped from the context when a new big task comes)."""
        self.briefs = [*self.briefs, card][-brief_mod.MAX_CARDS:]
        cards = [i for i, m in enumerate(self.ctx.messages) if brief_mod.is_card(m)]
        if len(cards) >= brief_mod.MAX_CARDS:
            drop = set(cards[:len(cards) - brief_mod.MAX_CARDS + 1])
            self.ctx.messages = [m for i, m in enumerate(self.ctx.messages) if i not in drop]
        self._insert_before_request({"role": "system", "content": card})
        self.events.emit("brief", items=card.count("\n"), chars=len(card))

    def brief(self) -> str:
        """The current task's card ('' when the task got none)."""
        return self.briefs[-1] if self.briefs else ""

    def _insert_plan(self) -> None:
        self._insert_before_request({"role": "system", "content": self.plan.plan.system_message()})

    def _insert_before_request(self, msg: dict[str, Any]) -> None:
        msgs = self.ctx.messages
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                self.ctx.messages = [*msgs[:i], msg, *msgs[i:]]
                self.ctx._since += len(str(msg.get("content", "")))
                return
        self.ctx.add(msg)

    def _call_model(self, think: bool) -> dict[str, Any]:
        opts = {**SAMPLING, "num_predict": self._limit("num_predict", NUM_PREDICT), **((self.config.get("coding") or {}).get("sampling") or {})}
        if think:
            opts.update({"temperature": 0.6, "top_p": 0.95, "num_predict": self._limit("think_predict", THINK_PREDICT)})
        self.live_pieces = 0
        stream = streaming_chat(self.llm)
        if stream is None:
            message = self.llm.chat_with_tools(self.ctx.messages, self.model, self.schema, num_ctx=self.num_ctx, think=think, extra_options=opts)
            if self.capture_thinking and isinstance(message, dict) and message.get("thinking"):
                self.events.emit_live("thinking", text=str(message["thinking"]), pieces=0)
            return message
        # Streamed so the reply can be watched while it is generated: the model writes its calls as JSON text (it imitates the
        # history), so even a big write_file arrives piece by piece. A reply that turns into the same block over and over is
        # stopped at ~8k chars instead of running to the 32k-token cap (minutes of a blocked GPU). Native tool calls arrive
        # in one piece at the end (Ollama buffers them), so for those only the token cap applies.
        buf: dict[str, list[str]] = {"content": [], "thinking": []}
        size = {"content": 0, "thinking": 0, "checked": 0}
        sprawl = sprawl_mod.SprawlGuard(self._limit("sprawl_tokens", sprawl_mod.SPRAWL_TOKENS), self._limit("sprawl_quiet", sprawl_mod.SPRAWL_QUIET))
        chant: dict[str, Any] = {}                            # early stop of a chanting reply or thought (loopguard.ChantDetector), one per kind
        if (self.config.get("coding") or {}).get("chant_detect", True):
            chant = {"content": loopguard.ChantDetector("content", self._limit("chant_hits", 0)),
                     "thinking": loopguard.ChantDetector("thinking", self._limit("chant_think_hits", 0))}

        def watch(kind: str):
            def on_piece(piece: str) -> None:
                self.live_pieces += 1
                if self.live.stopping:
                    raise KeyboardInterrupt                       # request_stop: end the turn between two pieces, like Ctrl-C
                if kind == "thinking" and self.capture_thinking:
                    self.events.emit_live("thinking", text=piece, pieces=self.live_pieces)
                buf[kind].append(piece)
                size[kind] += len(piece)
                if chant and chant[kind].feed(piece):
                    raise GenerationAborted(f"{kind} repeats itself")
                if sprawl.feed(piece, kind):                  # thousands of tokens of prose / thinking, no tool call open: stop it, do not wait for the cap
                    raise GenerationAborted(f"{SPRAWL} after {sprawl.n} tokens")
                total = size["content"] + size["thinking"]
                if total - size["checked"] >= 1500:
                    size["checked"] = total
                    if repeating("".join(buf[kind])):
                        raise GenerationAborted(f"{kind} repeats itself")
            return on_piece

        return stream(self.ctx.messages, self.model, self.schema, num_ctx=self.num_ctx, on_content=watch("content"),
                      on_thinking=watch("thinking"), think=think, extra_options=opts)

    def _think_now(self) -> bool:
        if self.pol.think == "on":
            return True
        if self.pol.think == "off":
            return False
        if self.think_steps > 0:
            self.think_steps -= 1
            return True
        return False

    def _model_failed(self, exc: Exception, retried: bool) -> str:
        """What to do after a failed model call: "backoff" (smaller window, retry), "wait" (the server is down or broke the
        reply: wait for it, resend the same request), "retry" (once, same state) or "stop".
        An HTTP 500 used to always mean "out of memory" and shrank the window - which reloads the model (8-10 s) and
        re-reads the whole prompt. Only the error text decides that now."""
        body = ""
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.text[:300]
            except Exception:  # noqa: BLE001 - an unread stream has no text
                body = ""
        text = f"{exc} {body}".lower()
        oom = re.search(r"out of memory|\boom\b|alloc|insufficient memory|vram", text)
        if oom and self.num_ctx > 16384:
            return "backoff"
        status = response.status_code if response is not None else 0
        if 400 <= status < 500 and not oom:
            return "stop"                                       # the request itself is wrong: waiting cannot fix it
        if not oom and self._server_wait_s() > 0 and (
                isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError, httpx.TimeoutException, KeyError, ValueError))
                or status >= 500 or _SERVER_TEXT.search(text)):
            return "wait"                                       # a killed server / half-written body (KeyError, bad JSON) after a crash
        return "stop" if retried else "retry"

    def _limit(self, key: str, default: int) -> int:
        """An integer coding.* key; the module constant is the default (a missing or broken value falls back to it)."""
        try:
            return int((self.config.get("coding") or {}).get(key, default))
        except (TypeError, ValueError):
            return default

    def _server_wait_s(self) -> float:
        return float((self.config.get("coding") or {}).get("server_wait_s", DEFAULT_WAIT_S))

    def _await_server(self) -> bool:
        """Block until the model server answers again (it was SIGKILLed, or the PC was suspended). True: resend the request."""
        url = server_url_of(self.llm, self.model)
        if url is None:
            return True                                         # a client without a health endpoint: just resend
        limit = self._server_wait_s()
        self.console.print(f"[dim]  server not reachable - waiting up to {limit:.0f} s[/dim]")
        revive_server(self.llm, self.model)                     # a dead Vulkan server is started again if there is a start command
        t0 = self._clock()
        self.live.waiting = "server"
        self._set_now("waiting for the model server")
        try:
            up = wait_for_server(url, max_wait_s=limit, sleep=self._sleep, clock=self._clock, probe=self._probe)
        finally:
            self.live.waiting = ""
        self.events.emit("server_wait", seconds=round(self._clock() - t0, 1), ok=up, url=url)
        return up

    def _size_budget(self) -> None:
        """Start of a prompt: an automatic budget is sized from the task (stepbudget.estimate); a user value stays the cap."""
        self.step_ceiling = self.pol.max_steps
        cfg = self.config.get("coding") or {}
        if self.pol.steps_set or not cfg.get("adaptive_steps", True) or self.stage_goal is not None:      # a stage keeps the budget its runner is tuned for
            self.budget_source, self.budget_now = "user" if self.pol.steps_set else "fixed", self.pol.max_steps
            return
        self.budget_source = "auto"
        est = stepbudget.estimate(self._task_text or "", files=stepbudget.repo_files(self.root),
                                  effort=str((self.config.get("build") or {}).get("effort", "mid")), diagnose=bool(cfg.get("diagnose_first")))
        self.budget_now = stepbudget.clamp(est, self.step_ceiling)
        self.events.emit("budget_set", steps=self.budget_now, estimate=est, ceiling=self.step_ceiling)

    def _loop(self, result: TurnResult) -> None:
        nudges = {"syntax": 0, "static": 0, "verify": 0, "plan": 0, "claims": 0, "scope": 0, "tui": 0, "std": 0}
        blanks = server_fails = 0
        bad, reminded = 0, False                            # the streak of unusable replies (malformed_abort) and whether it got its one reminder
        bad_limit = max(2, self._limit("malformed_abort", MALFORMED_ABORT))
        retried = False
        steps = 0
        budget = self.budget_now
        auto = self.budget_source == "auto"
        grown = False
        extended, progress_mark = 0, self._progress
        attempt = self._measure()                           # what "progress" is measured against (supervisor.measure_progress); the limits below reset on it
        ext_mark = attempt
        while True:
            self._boundary()
            if steps >= budget and auto:
                now = self._measure()
                if not grown and budget < self.step_ceiling and self._progress > progress_mark and self._advanced(ext_mark, now):
                    grown, new = True, stepbudget.grow(budget, self.step_ceiling)    # demonstrable progress: one bounded extension
                    self.events.emit("budget_extend", steps=steps, progress=self._progress - progress_mark, n=1, to=new)
                    self.console.print(f"[dim]  {steps} steps, files/tests moved - budget {budget} -> {new} (ceiling {self.step_ceiling})[/dim]")
                    budget = self.budget_now = new
                else:
                    result.stopped = "budget"
                    self.console.print(f"[yellow]stopped after {steps} steps (automatic budget {budget}, ceiling {self.step_ceiling}) - "
                                       f"say 'continue' to go on (or later: {brand.CMD} code --resume continue)[/yellow]")
                    return
            if steps >= budget:
                # A long autonomous run must not need babysitting: while the last window made progress (a real file written or an
                # acceptance item confirmed) the budget grows by itself, bounded; only a window without progress asks (or stops).
                if self._progress > progress_mark:
                    now = self._measure()
                    if self._advanced(ext_mark, now):       # real progress (changed bytes, a new green check/test/item) starts the count again
                        extended, ext_mark = 0, now
                max_ext = self._limit("auto_extend_max", AUTO_EXTEND_MAX)
                if self._progress > progress_mark and extended < max_ext:
                    extended += 1
                    self.events.emit("budget_extend", steps=steps, progress=self._progress - progress_mark, n=extended)
                    self.console.print(f"[dim]  {steps} steps, still making progress - continuing (+{self.pol.max_steps} steps)[/dim]")
                    budget += self.pol.max_steps
                    self.budget_now = budget
                    progress_mark = self._progress
                elif self.interactive and self._more_steps(steps):
                    budget += self.pol.max_steps
                    self.budget_now = budget
                    progress_mark = self._progress
                else:
                    result.stopped = "budget"
                    why = "without progress in the last window" if extended < max_ext else "(the automatic extensions are used up)"
                    self.console.print(f"[yellow]stopped after {steps} steps {why} - say 'continue' to go on "
                                       f"(or later: {brand.CMD} code --resume continue)[/yellow]")
                    return
            steps += 1
            result.steps = steps
            self.ctx.step += 1
            if self.on_step:
                self.on_step()
            self.live.fire_hooks()
            think = self._think_now()
            label = "thinking" if think else "working"
            self._set_now(f"waiting for the model (step {self.ctx.step}{', reasoning' if think else ''})")
            try:
                with self.console.status(_Working(self, label), spinner="dots"):
                    message = self._call_model(think)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                what = self._model_failed(exc, retried)
                detail = f"{type(exc).__name__}: {_one_line(str(exc), 200)}"
                self.events.emit("model_error", error=detail, action=what)
                if what == "wait":
                    server_fails += 1
                    if server_fails < SERVER_ATTEMPTS and self._await_server():
                        steps -= 1
                        continue
                    result.stopped = "error"
                    self.console.print(f"[red]model call failed after {server_fails} attempt{'s' if server_fails != 1 else ''}: {detail}[/red] [dim](the conversation and all "
                                       "edits are kept; check the model server, then say 'continue')[/dim]")
                    return
                if what == "backoff":
                    self.num_ctx = max(16384, self.num_ctx * 3 // 4)
                    self.ctx.num_ctx = self.num_ctx
                    self.console.print(f"[yellow]model runner ran out of memory - context now {self.num_ctx // 1024}k, retrying[/yellow]")
                    self.events.emit("ctx_backoff", num_ctx=self.num_ctx)
                    if not self.ctx.fit():                  # resending the old history would be cut from the front by the server
                        result.stopped = "error"
                        self.console.print(f"[red]the conversation does not fit the smaller window ({self.num_ctx // 1024}k) even after compaction - "
                                           "stopping instead of sending a truncated prompt[/red] [dim](nothing is lost; try /new, or raise the "
                                           "memory budget and say 'continue')[/dim]")
                        return
                    steps -= 1
                    continue
                if what == "retry":
                    retried = True
                    self.console.print(f"[yellow]model call failed ({detail}) - retrying once[/yellow]")
                    steps -= 1
                    continue
                result.stopped = "error"
                self.console.print(f"[red]model call failed twice: {detail}[/red] [dim](the conversation and all edits are kept; "
                                   "check the model server, then say 'continue')[/dim]")
                return
            retried, server_fails = False, 0
            self.ctx.note_usage(message, think)
            self.prompt_peak = max(self.prompt_peak, self.ctx.last_prompt)
            if str(message.get("_aborted", "")).startswith(SPRAWL):
                blanks += 1                                         # a sprawl counts like an empty reply: bounded the same way (6 -> error)
                if think:
                    self.think_steps = 0
                self.events.emit("gen_sprawl", pieces=self.live_pieces, n=blanks)
                self.console.print(f"[yellow]  the reply ran {self.live_pieces:,} tokens without a tool call - stopped, asking for smaller steps[/yellow]".replace(",", "."))
                if blanks >= 6:
                    result.stopped = "error"
                    self.console.print("[red]the model keeps writing without acting - stopping (nothing is lost; try again or /new)[/red]")
                    return
                self.ctx.add({"role": "system", "content": _cut_off_note(self.live_pieces)})
                continue
            if message.get("_aborted"):
                blanks += 1                                         # degenerate output counts like an empty reply: bounded the same way
                if think:
                    self.think_steps = 0                            # a reasoning pass that loops: act without it
                self.events.emit("degenerate", reason=message["_aborted"], pieces=self.live_pieces, n=blanks)
                self.console.print(f"[yellow]  the reply started repeating itself ({message['_aborted']}) - stopped after {self.live_pieces:,} tokens[/yellow]".replace(",", "."))
                if blanks >= 6:
                    result.stopped = "error"
                    self.console.print("[red]the model keeps degenerating - stopping (nothing is lost; try again or /new)[/red]")
                    return
                if blanks >= 2:
                    self.ctx.add({"role": "system", "content": "Your last reply degenerated into repeating the same text and was discarded. "
                                  "Do the next step differently: one small tool call; write a big file in parts (append_file)."})
                continue
            cut = message.get("_done_reason") == "length"
            calls = [] if cut else self._extract_calls(message)     # a reply that ran into the output limit never runs: its last call is half-written
            if not calls:
                content = _strip_think(message.get("content", ""))
                reminder: str | None = None                         # what this reply is, when it is unusable (no readable call, no answer)
                if message.get("_done_reason") == "length":
                    self.events.emit("cut_off", gen=message.get("_eval_count", 0))
                    self.console.print("[yellow]  the reply hit the output limit and was cut off - asking for smaller steps[/yellow]")
                    reminder = _cut_off_note(message.get("_eval_count", 0))
                elif not content.strip():
                    blanks += 1                                     # empty, or only thinking: nothing to run and nothing to say
                    self.events.emit("blank_reply", n=blanks)
                    if think:
                        self.think_steps = 0                        # the reasoning used its whole budget without an answer: act without it
                    reminder = "Your last reply was empty. Continue with the next tool call, or give your final answer."
                    if bad == 0 and not reminded:
                        reminder = ""                               # a first blank is a stochastic blip: re-roll with the same context
                elif _TOOLISH.search(content):
                    blanks = 0
                    self.events.emit("malformed_call", n=bad + 1)
                    self.console.print("[dim]  unreadable tool call - asking again[/dim]")
                    reminder = ("Your last message looked like a tool call but could not be parsed. Send it again as ONE proper tool call "
                                "(a single JSON object with the tool name and its arguments), without extra text.")
                if reminder is not None:
                    bad += 1
                    if bad >= bad_limit:
                        result.stopped, result.reason = "error", "malformed"
                        self.events.emit("malformed_abort", n=bad)
                        try:
                            self.dead_ends.record("model", "produced unusable replies", f"{bad} in a row", bad, error=True, files=[], rsha="malformed")
                        except Exception:  # noqa: BLE001 - the ledger is a hint
                            logger.debug("dead end not recorded", exc_info=True)
                        self.console.print(f"[red]the model produced {bad} unusable replies in a row (cut off, empty or unreadable calls) - stopping this prompt "
                                           "(nothing is lost; the supervisor restarts it clean, or say 'continue')[/red]")
                        return
                    if reminder and not reminded:
                        reminded = True                             # one short format reminder per streak, not one per bad reply
                        self.ctx.add({"role": "system", "content": reminder})
                    continue
                blanks = 0
                bad, reminded = 0, False                            # a real answer: the streak is over
                attempt = self._renew(attempt, nudges)
                announced = self._announced_next(content, steps)
                if announced:
                    self.ctx.add({"role": "assistant", "content": content})
                    self.ctx.add({"role": "system", "content": announced})
                    continue
                if self._announce_limit:
                    result.stopped = "announce"
                    self.events.emit("announce_limit", n=self._announce_continues)
                    self.console.print(f"[yellow]the model announced a next step {self._announce_continues} times without doing it - stopping this prompt "
                                       "(work so far is kept)[/yellow]")
                    return
                nudge = self._final_gate(content, nudges, result)
                if nudge is None:
                    shown = content
                    if result.not_verified:
                        # the answer goes out, but labelled from measured data: a check that did not pass is never reported as done (D3).
                        # The label is for the user only: the stored message stays unprefixed (the model imitated such headers in later turns)
                        shown = "NOT VERIFIED: " + "; ".join(result.not_verified) + "\n\n" + content
                    if result.stopped == "gate":
                        self.console.print("[yellow]the nudges are spent and acceptance items are still open - ending this prompt as NOT VERIFIED "
                                           "(work so far is kept; say 'continue' to go on)[/yellow]")
                    self._print_final(shown, message, stored=content)
                    result.text = shown
                    return
                self._gate_streak += 1
                self.ctx.add({"role": "assistant", "content": content})
                self.ctx.add({"role": "system", "content": nudge})
                continue
            blanks = 0
            bad, reminded = 0, False                                # a readable call: the streak is over
            for call in calls[:4]:
                self._run_call(call)
                if classify(call["name"], call.get("arguments")) != "neutral":
                    self._gate_streak = 0                        # the model acted after the nudge: the gate cap counts only blocks answered with text
                if self._abort_loop or self.phase.ended:
                    break
            if len(calls) > 4 and not self._abort_loop and not self.phase.ended:
                self.ctx.add({"role": "system", "content": f"Only the first 4 of your {len(calls)} tool calls were run. Send the others again if they are still needed."})
            if self._abort_loop:
                n, self._abort_loop = self._abort_loop, 0
                result.stopped = "loop"
                self.console.print(f"[red]the model repeated the same call {n} times - stopping this prompt (work so far is kept; rephrase or /new)[/red]")
                return
            if self.phase.ended:
                result.stopped = "verify_budget"
                self.events.emit("verify_budget", calls=self.phase.clean)
                self.console.print(f"[yellow]the files are written and the last {self.phase.clean} calls changed and broke nothing - ending the tool use "
                                   "(work so far is kept; say 'continue' to go on)[/yellow]")
                return
            if self.stage_cap and self.ctx.last_prompt > self.stage_cap:
                # the cap is the hard limit, but a restart re-reads the whole prompt and the files again: first try ONE batch mask of the
                # old tool results and re-measure by estimate; only when the prompt is still over the cap the stage goes on in a fresh context
                note = self.ctx.stage_mask(force=True)
                if note:
                    self.console.print(f"[dim]  {note}[/dim]")
                if self.ctx.last_prompt > self.stage_cap:
                    # the calls of this step have run; the stage goes on in a fresh context (stages.py) instead of a bigger prompt
                    result.stopped = "stage_cap"
                    return
            if self.stage_cap and self.ctx.stage_mask_thrash:
                # the thrash guard (context.stage_mask): repeated fruitless batches each cost a full prefill; ONE clean restart is cheaper
                result.stopped = "stage_cap"
                return
            why = self.segment_end_check() if self.segment_end_check is not None else None
            if why:
                # measured to be slow (timing.py via the supervisor): the same stage goes on in a fresh, small context
                self.events.emit("fresh_context", reason=why)
                result.stopped = "fresh_context"
                return
            for note in self.ctx.maintain():
                self.console.print(f"[dim]  {note}[/dim]")

    # ── the honest forced ending (phase.py) ─────────────────────────────
    def _forced_applies(self, result: TurnResult) -> bool:
        """A stop WITHOUT a model answer in a one-shot or REPL prompt gets one tool-less summary turn. Not inside a stage / the autopilot (stage_goal
        is set there: the supervisor restarts the segment), not for an interruption or a dead model server, not with coding.forced_summary false."""
        if result.text is not None or self.stage_goal is not None or not (self.config.get("coding") or {}).get("forced_summary", True):
            return False
        return result.stopped in FORCED_STOPS or (result.stopped == "error" and result.reason == "malformed")

    def _forced_ending(self, result: TurnResult) -> None:
        """ONE call with tools disabled, thinking off and <= 400 tokens, on the harness-measured facts; its text is shown behind a factual harness
        line, the stored message stays clean. A failed call (or an unusable answer) falls back to the mechanical summary. Nothing is re-run."""
        try:
            facts = phase_mod.facts(self, result, phase_mod.rel_files(self))
        except Exception:  # noqa: BLE001 - the ending must still produce something
            logger.debug("forced ending: facts failed", exc_info=True)
            facts = {"stop": phase_mod.reason_text(result.stopped, result.reason), "files": [], "commands": [], "tests": [], "open": [],
                     "not_verified": ["the harness could not measure the state"], "since_edit": 0}
        result.not_verified = list(facts["not_verified"])
        text, error = None, ""
        try:
            text = self._summary_call(facts)
        except Exception as exc:  # noqa: BLE001 - any failure of the call: the mechanical summary below
            error = f"{type(exc).__name__}: {_one_line(str(exc), 160)}"
            logger.debug("forced summary failed", exc_info=True)
        self.events.emit("forced_summary", ok=text is not None, stopped=result.stopped, error=error)
        text = text or phase_mod.mechanical(facts)
        shown = phase_mod.prefix(facts) + "\n\n" + text
        self._print_final(shown, {}, stored=text)
        result.text = shown

    def _summary_call(self, facts: dict[str, Any]) -> str | None:
        """The tool-less turn: the conversation so far (or, when it leaves no room, only the task) plus the facts and the fixed instruction.
        None when the model answers with a call or nothing usable."""
        ask = phase_mod.message(self._task_text, facts)
        if self.ctx.estimate() + len(ask) / 3.2 + 1500 <= self.num_ctx:
            msgs = [*self.ctx.messages, {"role": "user", "content": ask}]
        else:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}]
        opts = {**SAMPLING, **((self.config.get("coding") or {}).get("sampling") or {}), "num_predict": phase_mod.SUMMARY_PREDICT}
        with self.console.status("[dim]writing the final answer from the measured facts…[/dim]", spinner="dots"):
            message = self.llm.chat_with_tools(msgs, self.model, [], num_ctx=self.num_ctx, think=False, extra_options=opts)
        text = _strip_think(str(message.get("content", ""))).strip()
        if message.get("tool_calls") or not text or text.startswith("{") or _TOOLISH.search(text):
            return None
        return text

    def _measure(self) -> tuple[str, int] | None:
        """The engine's measure of progress (supervisor.measure_progress: content of the written project files, number of green checks /
        confirmed items / passing test runs). None when it cannot be measured - then nothing counts as progress."""
        from agent.codeagent.supervisor import measure_progress      # late: the supervisor imports the stages, which import this module

        try:
            return measure_progress(self)
        except Exception:  # noqa: BLE001 - a measuring problem is no progress, not a crash
            return None

    @staticmethod
    def _advanced(before: tuple[str, int] | None, now: tuple[str, int] | None) -> bool:
        return before is not None and now is not None and (now[0] != before[0] or now[1] > before[1])

    def _renew(self, mark: tuple[str, int] | None, nudges: dict[str, int]) -> tuple[str, int] | None:
        """A new finish attempt: when real progress happened since `mark`, the gate nudges, the check returns and the announce
        continuations start from zero (a limit spent early in a long run must not silence a new, real problem much later)."""
        now = self._measure()
        if not self._advanced(mark, now):
            return mark
        nudges.update({k: 0 for k in nudges})
        self._announce_continues = 0
        self._check_returns = {}
        self.events.emit("limits_reset", progress=True)
        return now

    def _open_work(self) -> list[str] | None:
        """What the harness knows is still open: open acceptance items and - in a staged run - syntax / static errors or a failed command
        since the last edit. None when there is neither a plan nor an unfinished stage plan to tell (the wording heuristic decides).
        Red checks are not listed: the checks gate sends the model back for them with the failing output, bounded and labelled."""
        plan = self.plan.plan
        staged = self.stage_plan is not None and self.stage_plan.unfinished()
        if plan is None and not staged:
            return None
        t = self.tools
        out = [f"acceptance item {i.id} is not confirmed ({i.text})" for i in (plan.open_items()[:3] if plan is not None else [])]
        if staged:
            out += [f"{p} has errors" for p in sorted({*t.syntax_bad, *t.static_bad})[:3]]
            if t.evidence and t.evidence[-1][2] not in (0, None):
                out.append(f"the last command failed: {_one_line(t.evidence[-1][1], 60)}")
        return out

    def _announced_next(self, content: str, steps: int) -> str | None:
        """Seen live: the model ends a turn with "Now update main.py ..." - an announced next step of its own that it never did, and
        the driver took that as the final answer. Free and deterministic. The decision is structural: a pure-text answer while the
        acceptance list has open items, a check is red (or a staged run has errors) is an announcement whatever its wording. Only without
        any plan / check / stage information the wording decides (announced_step: an own next action, or a trailing ":" / "...").
        Bounded: at most ANNOUNCE_MAX per finish attempt (progress starts the count again), never twice without a tool step in between."""
        if self._calls_run == self._announce_calls:
            return None                                          # no tool step since the last continuation
        sentence = announced_step(content)
        open_work = self._open_work()
        if open_work is None:
            if not sentence or not re.search(r"(:|\.\.\.|…)\s*$", sentence):
                return None
        elif not open_work:
            return None
        if self._announce_continues >= self._limit("announce_max", ANNOUNCE_MAX):
            self._announce_limit = self.stage_goal is not None  # staged run: the limit is used up and it announces again -> the prompt ends
            return None
        self._announce_continues += 1
        self._announce_at, self._announce_calls = steps, self._calls_run
        said = sentence or _one_line(content.strip().splitlines()[-1] if content.strip() else "", 120)
        self.events.emit("announce_continue", n=self._announce_continues, text=said[:120], structural=bool(open_work))
        self.console.print(f"[dim]  stopped with work open - continuing ({_one_line(said, 70)})[/dim]")
        plan = self.plan.plan
        left = plan.open_items() if plan is not None else []
        nxt = f" Next open acceptance item: {left[0].id}. {left[0].text}" if left else ""
        if not sentence:
            return (f"You answered without a tool call, but work is still open ({'; '.join(open_work[:3])}) - nothing ran. Do that work now "
                    f"with tool calls instead of describing it.{nxt} Give the final answer only when the work is done (or say plainly "
                    "what is blocked and why).")
        return (f"You wrote \"{_one_line(sentence, 160)}\" and stopped without doing it - nothing ran. Do that step now with tool "
                f"calls instead of describing it, then go on with the rest of the task.{nxt} Give the final answer only when the work is "
                "done (or say plainly what is blocked and why).")

    def _compaction_extras(self) -> str:
        """State the work log must carry verbatim through every compaction: the live acceptance list (the copy in the prompt head is
        a snapshot from when it was made) and the project notes saved in this session."""
        out = []
        plan = self.plan.plan
        if plan is not None:
            out.append(f"ACCEPTANCE LIST NOW ({plan.confirmed()} of {len(plan.items)} confirmed; confirm open ones with mark_done):\n"
                       + "\n".join(f"{i.id}. [{'x' if i.done else ' '}] {i.text}" for i in plan.items))
        if self.session_notes:
            out.append("PROJECT NOTES SAVED IN THIS SESSION:\n" + "\n".join(f"- {n}" for n in self.session_notes[-12:]))
        return "\n\n".join(out)

    def _summarize_in_place(self, instruction: str) -> str | None:
        """Compaction with the conversation model on the conversation itself: the same messages, tools and window as the last step,
        so the server reuses its cached prompt and only the log is generated. None (-> the separate summary call) when it answers
        with a tool call or something that is not a log; after two such misses it is not tried again in this session."""
        if self._in_place_failures >= 2:
            return None
        if self.ctx.estimate() + len(instruction) / 3.2 + 1500 > self.num_ctx:
            return None                     # no room left to write the log (seen live: a resumed run at 97% got 475 tokens, cut off)
        opts = {**SAMPLING, **((self.config.get("coding") or {}).get("sampling") or {}), "num_predict": COMPACT_PREDICT}
        msgs = [*self.ctx.messages, {"role": "user", "content": instruction}]
        self._set_now("compacting the context (writing the work log)")
        with self.console.status("[dim]writing the work log (compacting the context)…[/dim]", spinner="dots"):
            message = self.llm.chat_with_tools(msgs, self.model, self.schema, num_ctx=self.num_ctx, think=False, extra_options=opts)
        text = _strip_think(str(message.get("content", ""))).strip()
        heads = sum(1 for h in ("DONE", "STATE", "DECISIONS", "OPEN", "PITFALLS") if re.search(rf"(?m)^\W*{h}\b", text))
        ok = not message.get("tool_calls") and not text.startswith("{") and heads >= 3
        self.events.emit("compact_call", ok=ok, prompt=int(message.get("_prompt_eval_count", 0) or 0), gen=int(message.get("_eval_count", 0) or 0),
                         prefill_s=round(int(message.get("_prompt_eval_ns", 0) or 0) / 1e9, 2),
                         decode_s=round(int(message.get("_eval_ns", 0) or 0) / 1e9, 2))
        if not ok:
            self._in_place_failures += 1
            return None
        return text

    def _more_steps(self, done: int) -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            return self.console.input(f"[yellow]{done} steps so far. Keep going? [y/N] [/yellow]").strip().lower() in ("y", "yes", "j", "ja")
        except (EOFError, KeyboardInterrupt):
            return False

    # ── tool calls ──────────────────────────────────────────────────────
    def _extract_calls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        from agent.tui import _extract_tool_call, _normalize_arguments

        out = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name"):
                out.append({"name": fn["name"], "arguments": _normalize_arguments(fn.get("arguments", {}))})
        if not out:
            one = _extract_tool_call({"content": message.get("content", "")})
            if one and one.get("name"):
                out.append(one)
        if not out:
            rescued = salvage_call(str(message.get("content", "")), {t["function"]["name"] for t in self.schema}, self.schema)
            if rescued:
                self.events.emit("salvaged_call", name=rescued["name"])
                out.append(rescued)
        return out

    def _run_call(self, call: dict[str, Any]) -> None:
        name, args = call["name"], call["arguments"]
        shown = self.describe(name, args)
        self.console.print(f"  [cyan]›[/cyan] {shown}")
        self._set_now(shown[:200])
        t_call = time.monotonic()
        self.events.emit_live("tool_start", name=name, summary=_one_line(shown, 200), args=_live_clip(json.dumps(args, ensure_ascii=False, default=str)),
                              tool_kind=KINDS.get(name), risk=self._risk_of(name, args), step=self.ctx.step)
        result = self.execute(name, args)
        self.events.emit_live("tool_result", name=name, summary=_one_line(shown, 200), result=_live_clip(result), chars=len(result),
                              seconds=round(time.monotonic() - t_call, 2), ok=not _errorish(result), step=self.ctx.step)
        # What the model sees of its own recent calls is what it copies next. write_file/append_file keep their real text until the bulk
        # mask (context.mask_observations) shrinks the OLD ones: with a placeholder stored right away it sent the placeholder as the
        # content, 94 times (seen live). The other tools keep their shape with clipped text.
        shown = {"tool": name, "arguments": args} if name in ("write_file", "append_file", "replace_lines", "insert_after") else compact_call(name, args)
        stored_call = {"role": "assistant", "content": json.dumps(shown, ensure_ascii=False)}
        stored_result = {"role": "tool", "content": cap_output(result, 1200, 5200) if name not in ("read_file", "read_files") else result}
        self.ctx.add(stored_call)
        self.ctx.add(stored_result)
        self.console.print(f"    [dim]{self._result_line(name, result)}[/dim]")
        self._calls_run += 1
        if not _errorish(result):
            self._ok_calls += 1
        if _failed_run(result):
            self._prompt_fails += 1
        if KINDS.get(name) == WRITE:
            # a successful edit may have made earlier cached reads of that file stale
            self.tools.read_cache.clear()
        if _real_write(name, args, result):
            self.real_writes += 1
        if _real_write(name, args, result) or (name == "mark_done" and " confirmed (" in result):
            self._progress += 1
        verdict = self.watcher.observe(name, args, result, denied=result == DENIED)
        if verdict is not None and verdict.kind == "lookup":          # repeated calls are the loop guard's job
            self._react_to_stuck(verdict)
        if name == "run" and result.startswith("exit code 0"):
            m = re.match(r"\s*(python3?\s+[\w./-]+\.py)\s*$", str(args.get("command", "")))
            if m:
                self._remember("run", m.group(1))
        self._loop_guard(name, args, result)
        free = self.lookact.free_steps
        outside = self._outside_read(name, args)
        for reaction in self.lookact.observe(name, args, result):
            self._react(reaction)
        if outside:
            self._note_outside_read(outside, free <= 0)
        self._weigh_stray_look(name, args, result, free)

    def _risk_of(self, name: str, args: dict[str, Any]) -> str | None:
        """SAFE / CAUTION / DANGEROUS / BLOCKED for a shell command (the safety classifier); None for the other tools."""
        try:
            if name in ("run", "tui_smoke") and args.get("command"):
                return classify_command(str(args["command"]), self.config).name
        except Exception:  # noqa: BLE001 - display only
            logger.debug("risk not classified", exc_info=True)
        return None

    def _weigh_stray_look(self, name: str, args: dict[str, Any], result: str, free: int) -> None:
        """A ranged part (stages.py) has its lines to rewrite: its first `coding.stray_looks` (8) search / read-only run / list_dir calls count
        `coding.stray_look_weight` (2) steps toward the look pause, also during the free orientation steps (measured: 9 searches, 7 reads and 12 runs
        and no edit on a 687-line file). Reading the range itself and every edit are not touched."""
        st = self.stage_obj
        if st is None or self.stage_goal is None or not getattr(st, "lines", None) or name not in STRAY_TOOLS or classify(name, args, result) != "look":
            return
        if self._stray_looks >= self._limit("stray_looks", STRAY_LOOKS):
            return
        self._stray_looks += 1
        counted = 0 if free > 0 else 1                         # observe() already counted this look once, unless it was a free step
        self.lookact.look_streak += max(0, self._limit("stray_look_weight", STRAY_LOOK_WEIGHT) - counted)

    def _outside_read(self, name: str, args: dict[str, Any]) -> str:
        """The first file a read of a RANGED stage (stages.py: 'dialogue.py lines 1-150') touches that is not the stage's file, else ''."""
        st = self.stage_obj
        if st is None or self.stage_goal is None or not getattr(st, "lines", None) or name not in OUTSIDE_READ_TOOLS:
            return ""
        raw = args.get("paths") if name == "read_files" else args.get("path")
        items = re.split(r"[\s,]+", raw) if isinstance(raw, str) else raw if isinstance(raw, (list, tuple)) else []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                continue
            try:
                rel = self.tools.rel(self.tools.resolve(item.strip()))
            except Exception:  # noqa: BLE001 - an absurd path is the tool's error, not ours
                continue
            if rel not in st.scope:
                return rel
        return ""

    def _note_outside_read(self, rel: str, counted: bool) -> None:
        """A read outside a ranged part: one factual note per segment; the read counts `coding.out_of_scope_read_weight` (default 2) steps toward
        the look pause (the orientation reads at the start of a segment stay free)."""
        st = self.stage_obj
        self.events.emit("out_of_scope_read", path=rel, stage=getattr(st, "id", 0))
        if not self._outside_noted:
            self._outside_noted = True
            a, b = st.lines
            self.ctx.add({"role": "system", "content": f"Outside this part: only {st.scope[0]} lines {a}-{b} are yours ({rel} is not part of this stage)."})
        if counted:
            self.lookact.look_streak += max(0, self._limit("out_of_scope_read_weight", OUT_OF_SCOPE_READ_WEIGHT) - 1)

    def _loop_guard(self, name: str, args: dict[str, Any], result: str) -> None:
        """Going in circles, as seen in real runs: a search repeated 14 times, a 6.6 s scripted playthrough re-sent 8+ times
        with the order of two inputs swapped, edited `python3 -c "..."` variants with one output, a 3-step cycle (run -> search -> cat
        -> run ...), read -> failing edit -> read. So the guard counts, within the last 16 calls (not only back to back), calls that are
        the same - for `run` / `search` also nearly the same (similar text, or an equal token set within the last 10 calls, loopguard.py) -
        AND had the same outcome (the same result, timings ignored; an error with the same first error line also counts). A successful
        write starts a fresh count: the results may legitimately differ now. The ladder (coding.loop_note / loop_repeat / loop_abort):
        2: one factual line appended to the tool result; 3: the nudge with the last attempts, the earlier copies are removed from the
        context and ONE reasoning pass; 4: the segment ends (stopped='loop') so the supervisor restarts clean. Every detection above the
        note is also kept in the DEAD ENDS ledger. In the verification phase (phase.py: files written, nothing failing) the notes ask for the final
        answer instead of a different approach, and no reasoning pass is forced."""
        self._phase_update(name, args, result)
        if KINDS.get(name) == WRITE and str(result).startswith(("wrote ", "edited ", "appended ", "patched ", "deleted ", "moved ")) \
                and not any(m in str(result) for m in ("✗ syntax", "SYNTAX GATE FAILED", "SyntaxError")) \
                and not is_scratch_path(str((args or {}).get("path") or (args or {}).get("dst") or (args or {}).get("dest") or "")):      # a write that left a syntax error is not progress (94 identical appends, seen live); neither is rewriting a scratch script between identical runs (14 repeats, seen live)
            self._sigs.clear()
            self.family.edited()
            self.pattern.edited()
            self.dead_ends.edited(str(args.get("path") or args.get("dst") or args.get("dest") or ""))
            return
        self._guard_spoke = False
        if self._family_guard(name, args, result):
            return
        spoke = self._repeat_ladder(name, args, result)
        if self._abort_loop:
            return
        # one voice per streak: a family / run-spam note already spoke for `run` and `search`
        self._pattern_guard(name, args, result, quiet=spoke or self._guard_spoke or (name in ("run", "search") and bool(self.family.noted)))

    def _phase_active(self) -> bool:
        return self.phase.active(self._limit("verify_phase_min", phase_mod.VERIFY_PHASE_MIN))

    def _phase_update(self, name: str, args: dict[str, Any], result: str) -> None:
        """Feed the verification-phase counters (phase.py) with one finished call. A real edit starts the phase's clock; a failure signal, a
        changed file (also a scratch one), any other write or a non-look tool starts the count again; a clean `run` / read-only call counts."""
        ph = self.phase
        if _real_write(name, args, result):
            ph.edit()
            return
        kind = classify(name, args, result)
        if kind == "neutral":
            return                                                   # bookkeeping (mark_done, project_note ...) neither counts nor resets
        bad = phase_mod.failure_signal(name, result, _failed_run(result))
        looked = kind == "look" or name in ("run", "run_tests", "lint", "tui_smoke")
        if bad or not looked or KINDS.get(name) == WRITE or (name == "run" and run_changed(result)):
            ph.restart()
            return
        was = ph.ended
        ph.clean_call(max(0, self._limit("verify_phase_budget", phase_mod.VERIFY_PHASE_BUDGET)))
        if ph.ended and not was:
            self.events.emit("verify_phase_end", calls=ph.clean, tool=name)

    def _repeat_ladder(self, name: str, args: dict[str, Any], result: str) -> bool:
        """The repeat ladder of `_loop_guard` (same or near-identical call with the same outcome). True when it said something (a note,
        the nudge, a paging hint or the stop), so the pattern guard does not add a second voice to the same call."""
        stem = _run_stem(args) if name == "run" else ""
        ckey, rkey = _call_key(args), _result_key(result)
        errorish = _errorish(result)
        ctext = loopguard.call_text(name, args)
        oline = loopguard.outcome_line(result) if errorish else ""
        entry = (name, _sha(ckey), _sha(rkey), errorish, _sha(stem) if stem else "", ckey, rkey, oline)
        hist = self._sigs
        hist.append(entry)
        del hist[:-LOOP_WINDOW]
        self.dead_ends.attempt(name, ctext, errorish, entry[2])
        recent = len(hist) - loopguard.NEAR_WINDOW
        def same_outcome(h: tuple) -> bool:
            if h[2] == entry[2]:
                return True
            if h[3] and errorish and oline and h[7]:           # two errors: the first error line decides (a failing test id, an exception)
                return h[7] == oline
            return h[6] is not None and _same_result(h[6], rkey)

        same = [(i, h) for i, h in enumerate(hist) if h[0] == name and same_outcome(h)]
        matches = [h for i, h in same if h[1] == entry[1] or (h[5] is not None and _same_call(name, h[5], ckey))
                   or (i >= recent and loopguard.near_calls(name, loopguard.text_of_key(h[5]), ctext))]
        near = len(matches)
        hint = max(1, self._limit("loop_repeat", LOOP_HINT))
        note_at = self._limit("loop_note", LOOP_NOTE)
        abort = max(hint, self._limit("loop_abort", LOOP_ABORT))
        added: list[str] = []
        if near == note_at and near < hint and name not in loopguard.CONTENT_TOOLS:
            added.append(loopguard.SAME_RESULT_NOTE)
        if near == 2 and errorish and name == "run":                  # the second identical error, never the first
            line = loopguard.error_hint(result, self.root)
            if line:
                added.append(line)
        if near >= 3 and not self._interactive_hinted and loopguard.piped_prompt_run(args, result):
            self._interactive_hinted = True
            added.append(loopguard.INTERACTIVE_HINT)
        self._append_to_result(added)
        if near >= abort:
            self._abort_loop = near
            self._dead_end(name, args, ctext, result, matches, errorish, entry[2], near)
            self.events.emit("loop_abort", tool=name, repeats=self._abort_loop)
            return True
        if stem and self.ctx.step - self._loop_note_at >= 4 and self._paging(stem, result, hint):
            return True
        if near < hint or self.ctx.step - self._loop_note_at < 4:
            return bool(added)
        self._loop_note_at = self.ctx.step
        errorish = any(h[3] for h in hist[-6:])
        self._dead_end(name, args, ctext, result, matches, bool(entry[3]), entry[2], near)
        self.events.emit("loop", tool=name, shape="error" if errorish else "same", repeats=near, phase=self._phase_active())
        if self._phase_active():
            self.console.print("[dim]  verification spiral - asking for the final answer[/dim]")
            self._collapse_repeats()
            self.ctx.add({"role": "system", "content": phase_mod.PHASE_NOTE})
            return True                                              # no reasoning pass: the answer is a plain text turn
        self.console.print("[dim]  loop detected - asking for a different approach[/dim]")
        if not errorish and name == "run":
            text = (f"You have sent this command (or a nearly identical one) {near} times and the output is the same every time, so it is not doing "
                    "what you expect. Compare the output with what you intended: piped input is probably being consumed by the wrong prompt or ignored. "
                    "Stop re-sending it. Instead call the code directly from a small python script (import the module, build the state, call the function, "
                    "print the result) or run one short input at a time, and fix what the output shows.")
        elif not errorish:
            text = (f"You have made the identical `{name}` call {near} times and it returns the same result every time. That information is "
                    "already in your context above - do not ask again. Use it: decide what to change and write it now, or search for something different.")
        else:
            text = (f"You are repeating yourself: `{name}` keeps returning the same result ({rkey[:70]!r}). Do not send that call again. "
                    "Read the error text carefully and change the call itself: check the argument types (a list must be a real JSON array, "
                    "not a quoted string), use apply_patch or write_file for a new full version, or run a command to see the real state.")
        rows = [(loopguard.text_of_key(h[5]) or ctext, (h[7] if len(h) > 7 and h[7] else "") or loopguard.outcome_line(h[6] or "")) for h in matches[-3:]]
        text += (f"\nYour last {len(rows)} attempts:\n{loopguard.attempts_text(rows)}\nDo something different: read the code at the failing "
                 "line (read_file / show_symbol), or change your hypothesis about the cause; do not run another variant of the same command.")
        self._collapse_repeats()
        self.ctx.add({"role": "system", "content": text})
        # Measured on a real stuck context: without reasoning the model re-sent the identical call 2 of 2 times; with it, 2 of 2 times
        # it changed course, in ~4 s and ~300 characters of thinking. One bounded step (THINK_PREDICT caps it), only when the guard fires.
        self.think_steps = max(self.think_steps, 1)
        return True

    def _family_guard(self, name: str, args: dict[str, Any], result: str) -> bool:
        """The family / read-only-run-spam counters (loopguard.FamilyGuard). A real edit (also a shell one) resets them. A note is one
        line appended to the result; the abort ends the segment (stopped='loop') and records a DEAD END. Never forces thinking.
        Returns True when the segment is to end."""
        if _real_write(name, args, result):
            self.family.edited()
            self.pattern.edited()
            return False
        if KINDS.get(name) == WRITE:
            return False
        changed = bool(name in ("run", "tui_smoke") and run_changed(result))
        events = self.family.observe(name, args, result, changed=changed,
                                     note_at=max(1, self._limit("family_note", loopguard.FAMILY_NOTE)), abort_at=self._limit("family_abort", loopguard.FAMILY_ABORT),
                                     spam_note=max(1, self._limit("run_spam_note", loopguard.RUN_SPAM_NOTE)), spam_abort=self._limit("run_spam_abort", loopguard.RUN_SPAM_ABORT))
        self._guard_spoke = bool(events)
        return self._apply_guard_events(name, args, result, events)

    def _pattern_guard(self, name: str, args: dict[str, Any], result: str, quiet: bool) -> bool:
        """The pattern counters (loopguard.PatternGuard): the same tool with changing arguments, alternating calls, re-reads in a window.
        `quiet`: the family guard or the repeat ladder already spoke on this call (notes dropped, a stop still counts). Never forces thinking.
        Returns True when the segment is to end."""
        changed = bool(name in ("run", "tui_smoke") and run_changed(result))
        cfg = lambda key, default: max(1, self._limit(key, default))  # noqa: E731
        events = self.pattern.observe(name, args, result, changed=changed, quiet=quiet, look_only=name == "run" and readonly_shell(args.get("command")),
                                      same_note=cfg("same_tool_note", loopguard.SAME_TOOL_NOTE), same_abort=self._limit("same_tool_abort", loopguard.SAME_TOOL_ABORT),
                                      alt_note=cfg("alt_note", loopguard.ALT_NOTE), alt_abort=self._limit("alt_abort", loopguard.ALT_ABORT),
                                      read_note=cfg("read_repeat_note", loopguard.READ_NOTE), read_abort=self._limit("read_repeat_abort", loopguard.READ_ABORT))
        return self._apply_guard_events(name, args, result, events)

    def _apply_guard_events(self, name: str, args: dict[str, Any], result: str, events: list[tuple[str, str, str, int, str]]) -> bool:
        """Notes go onto the tool result (in the verification phase: the one 'give your final answer' line instead); the first abort ends the
        segment (stopped='loop') and records a DEAD END. Returns True on an abort."""
        notes = [text for act, _s, _f, _n, text in events if act == "note"]
        self._append_to_result([phase_mod.PHASE_NOTE] if notes and self._phase_active() else notes)
        for act, shape, fam, n, _text in events:
            if act != "abort":
                continue
            self._abort_loop = n
            try:
                self.dead_ends.record(name, fam, "investigated without editing", n, error=False, files=loopguard.files_of(name, args, result, self.root), rsha="family")
            except Exception:  # noqa: BLE001 - the ledger is a hint
                logger.debug("dead end not recorded", exc_info=True)
            self.events.emit("loop_abort", tool=name, repeats=n, shape=shape, family=fam)
            return True
        return False

    def _dead_end(self, name: str, args: dict[str, Any], ctext: str, result: str, matches: list[Any], errorish: bool, rsha: str, times: int) -> None:
        """Put the repeated attempt into the DEAD ENDS ledger (loopguard.DeadEnds)."""
        try:
            out = loopguard.outcome_line(result)
            self.dead_ends.record(name, ctext, out, times, error=errorish, files=loopguard.files_of(name, args, result, self.root), rsha=rsha)
        except Exception:  # noqa: BLE001 - the ledger is a hint; it must never break the guard
            logger.debug("dead end not recorded", exc_info=True)

    def _append_to_result(self, lines: list[str]) -> None:
        """Add one-line facts to the tool result the model is about to read (the last tool message of the context)."""
        msgs = self.ctx.messages
        if lines and msgs and msgs[-1].get("role") == "tool":
            lines = [x for x in lines if x not in str(msgs[-1].get("content", ""))]          # a fixed note is said once per result
            if not lines:
                return
            msgs[-1] = {**msgs[-1], "content": str(msgs[-1].get("content", "")).rstrip("\n") + "\n" + "\n".join(lines)}
            self.ctx._since += sum(len(x) + 1 for x in lines)

    def _paging(self, stem: str, result: str, hint: int = LOOP_HINT) -> bool:
        """Seen live (twice in one run, 34 runs of 17-26 s): the same program re-run with only the filter changed - `| sed -n '1,15p'`,
        '1,20p', '1,25p' ... - to page through output it already had. Every output differs, so the repeat check above never matches.
        Same program (the command before its trailing filters) 3 times since the last write: one hint naming the saved output, and
        one reasoning step (non-thinking steps ignore hints). Again at 6, 9 ... within the window. Never blocks."""
        runs = [h for h in self._sigs if h[0] == "run" and h[4] == _sha(stem)]
        if len(runs) < hint or len(runs) % hint or len({h[1] for h in runs}) < 2:
            return False
        self._loop_note_at = self.ctx.step
        m = _TIMING.search(str(result))
        took = f" (each run takes {m.group(0)[1:-1]})" if m else ""
        self.events.emit("paging", repeats=len(runs))
        self.console.print("[dim]  same program re-run to page through its output - pointing at the saved output[/dim]")
        kept = self.tools.spill.last.get("run")
        saved = (f"Its unfiltered output is saved in {kept}: read_file it with start/end, or search it."
                 if kept and (self.root / kept).is_file() else "Its output is already in your context above.")
        self.ctx.add({"role": "system", "content": (
            f"You have run `{_one_line(stem, 100)}` {len(runs)} times since your last edit, changing only the filter after it{took}. Nothing "
            f"in the code changed in between, so the output is the same as before. {saved} Run the program again only after you change the code.")})
        self.think_steps = max(self.think_steps, 1)
        return True

    def _collapse_repeats(self) -> None:
        """The model imitates its own history: a context holding the same call+result five times is the strongest possible
        signal to send it a sixth time, and no hint at the end outweighs it. Keep only the latest copy of the repeated pair
        (same matching as the loop guard, on the stored assistant-JSON / tool-result messages)."""
        msgs = self.ctx.messages
        i = len(msgs) - 1
        while i > 0 and msgs[i].get("role") != "tool":
            i -= 1
        if i < 1 or msgs[i - 1].get("role") != "assistant":
            return
        cur = _parse_call(str(msgs[i - 1].get("content", "")))
        if cur is None:
            return
        name, key, res = cur[0], _call_key(cur[1]), _result_key(loopguard.strip_notes(msgs[i].get("content", "")))
        drop: set[int] = set()
        for j in range(i - 3, -1, -1):
            if msgs[j].get("role") != "assistant" or msgs[j + 1].get("role") != "tool":
                continue
            old = _parse_call(str(msgs[j].get("content", "")))
            if old and old[0] == name and _same_call(name, _call_key(old[1]), key) and _same_result(_result_key(loopguard.strip_notes(msgs[j + 1].get("content", ""))), res):
                drop.update((j, j + 1))
        if drop:
            self.ctx.messages = [m for k, m in enumerate(msgs) if k not in drop]
            self.events.emit("collapse", dropped=len(drop) // 2)

    def _remember(self, kind: str, command: str | None) -> None:
        """Save a command that provably worked as a project note, once per session (next session starts knowing it)."""
        if not command or self.notes is None or not self.note_key:
            return
        seen = self.__dict__.setdefault("_noted", set())
        if (kind, command) in seen:
            return
        seen.add((kind, command))
        try:
            if self.notes.add(self.note_key, f"{kind}: {command}  (verified working)", kind="command"):
                self.session_notes.append(f"{kind}: {command}  (verified working)")
        except Exception:  # noqa: BLE001 - memory is a bonus
            logger.debug("note failed", exc_info=True)

    def _react(self, reaction: Reaction | None) -> None:
        """A look/act reaction (lookact.py): log it, tell the model once, maybe one bounded reasoning step."""
        if reaction is None:
            return
        self.events.emit(reaction.event, **reaction.data)
        self.ctx.add({"role": "system", "content": reaction.text})
        if reaction.think:
            self.think_steps = max(self.think_steps, 1)

    def _wander_guard(self, name: str) -> None:
        """The look-streak half of lookact.observe on its own (kept for callers and tests that drive it directly)."""
        self._react(self.lookact._wander(classify(name, {}, "")))

    def _edit_gap_guard(self, name: str, args: dict[str, Any], result: str) -> None:
        """The edit-gap half of lookact.observe on its own (kept for callers and tests that drive it directly)."""
        self._react(self.lookact._edit_gap(name, args, result, classify(name, args, result)))

    def _react_to_stuck(self, verdict: Any) -> None:
        web = self.registry.get("web_search")
        if web is None or not autonomy.enabled(self.config):
            self.ctx.add({"role": "system", "content": "The same error keeps coming back. Read it carefully, check the real API with outline/search, and try a different approach."})
            return
        self.console.print(f"[dim]  stuck: same problem {verdict.count}x - looking it up[/dim]")
        try:
            found = web.handler(query=verdict.query, num_results=3)
        except Exception:  # noqa: BLE001
            logger.warning("automatic lookup failed", exc_info=True)
            return
        self.events.emit("lookup", query=verdict.query)
        self.ctx.add({"role": "system", "content": autonomy.lookup_message(verdict.query, verdict.count, cap_output(str(found), 1500, 1500))})

    # ── "done" gates ────────────────────────────────────────────────────
    def _final_gate(self, content: str, nudges: dict[str, int], result: TurnResult) -> str | None:
        """None = accept the answer, else the system message to continue with: the classic gates (syntax, the project's checks,
        the acceptance list, unsupported claims), then the CHECKS (checks.py), each bounded on its own."""
        result.stopped, result.not_verified = "", []             # decided again on every attempt to finish
        nudge = self._classic_gates(content, nudges, result)
        nudge = nudge if nudge is not None else self._checks_gate(result, spent=self._gate_spent(nudges))
        if nudge is not None:
            result.stopped, result.not_verified = "", []         # the model goes on: nothing is final yet
        return nudge

    def _gate_spent(self, nudges: dict[str, int]) -> bool:
        """The gate budget is used up: `gate_consecutive` nudges in a row that the model answered with text only (Claude Code's Stop-hook
        rule: the cap counts blocks with no tool call in between), or the hard per-attempt ceiling `gate_nudges`."""
        return (self._gate_streak >= self._limit("gate_consecutive", GATE_CONSECUTIVE)
                or sum(nudges.values()) >= self._limit("gate_nudges", MAX_GATE_NUDGES))

    def _classic_gates(self, content: str, nudges: dict[str, int], result: TurnResult) -> str | None:
        """None = accept the answer, else the system message to continue with. Every gate is bounded, and all of them
        together send the model back at most GATE_CONSECUTIVE times in a row without a tool call (a real call resets the count) and
        MAX_GATE_NUDGES times per finish attempt (progress starts a new one); after that the
        answer is accepted and `verified` says honestly whether the checks passed - except with open acceptance items: then the run ends
        as stopped="gate" and the text is labelled NOT VERIFIED (_open_items_stop)."""
        t = self.tools
        if self._gate_spent(nudges):
            if t.syntax_bad or t.dirty:
                result.verified = False
                self.console.print("[yellow]  accepting the answer - the checks did not pass (see above)[/yellow]")
            self._tui_unverified(result)
            self._std_unverified(result)
            self._duplicate_unverified(result)
            self._open_items_stop(result)
            return None
        if t.syntax_bad and nudges["syntax"] < 2:
            nudges["syntax"] += 1
            bad = "\n".join(f"{p}: {e.splitlines()[0]}" for p, e in list(t.syntax_bad.items())[:5])
            return f"You are not done: these files still have syntax errors:\n{bad}\nFix them with edit_file, then verify."
        self._static_sweep()
        if t.static_bad and nudges["static"] < 2:
            nudges["static"] += 1
            result.verified = False
            bad = "\n".join(f"{p}: {'; '.join(probs[:3])}" for p, probs in list(t.static_bad.items())[:6])
            self.events.emit("static_failed", files=len(t.static_bad))
            return f"You are not done: the static check finds errors that fail at run time (even if no test reaches them):\n{bad}\nFix them, then verify."
        if t.dirty and nudges["verify"] < 2:
            outcome = self._verify()
            if outcome is not None:
                nudges["verify"] += 1
                self.think_steps = max(self.think_steps, 1)             # diagnose the failure properly before editing
                result.verified = False
                return outcome
            result.verified = bool(t.evidence) and all(code == 0 for _s, _c, code in t.evidence)
        self._duplicate_unverified(result)                  # the static nudges are spent: a dead duplicate is not hidden behind a green test run
        crashed = self._tui_gate(nudges, result)
        if crashed is not None:
            return crashed
        not_standard = self._std_gate(nudges, result)
        if not_standard is not None:
            return not_standard
        plan = self.plan.plan
        if plan is not None and plan.open_items() and nudges["plan"] < 1:
            nudges["plan"] += 1
            self.console.print(f"[dim]  plan {plan.confirmed()} of {len(plan.items)} confirmed - back to work[/dim]")
            return plan.reminder()
        if (t.changes.changed() and nudges["claims"] < 1 and (self.config.get("coding") or {}).get("claims_check", True)
                and CLAIM_WORDS.search(content) and not any(code == 0 and not c.startswith(_WEAK_CHECKS) for _s, c, code in t.evidence)):
            # Free and deterministic: the answer says it works, but nothing that exercises the code ran since the last edit (a
            # compile check is not "tested"). This replaced a second model call, which also evicted the conversation from the
            # server's prompt cache (a 27k-token prompt costs ~38 s to read again).
            nudges["claims"] += 1
            self.console.print("[dim]  the answer claims it works, but nothing ran since the last edit - verifying[/dim]")
            return ("Your answer claims things the evidence does not support: you changed files but ran nothing that exercises them since the last edit. "
                    "Run it now (use `run` with stdin for interactive programs) and report what you saw, or remove the claim and say it is untested.")
        missing = self.scope_missing() if nudges["scope"] < 2 else []
        if missing:
            nudges["scope"] += 1
            self.events.emit("scope_gate", missing=missing)
            if getattr(self.stage_obj, "lines", None):
                return (f"You are not done: this part covers {missing[0]} and no line in that range was changed yet. Change lines inside it, or - if the range "
                        "really needs no change - call mark_done (or project_note) with evidence that names the file and says that range needs no change.")
            return (f"You are not done: this stage covers {', '.join(self.stage_scope)}, and these files are not handled yet: {', '.join(missing)}. "
                    "Change each of them, or - if a file really needs no change - call mark_done (or project_note) with evidence that names the "
                    "file and says it needs no change.")
        self._open_items_stop(result)
        return None

    # ── the crash gate for terminal programs (tuigate.py) ───────────────
    def _tui_files(self) -> list[str]:
        """The project .py files really edited in this prompt (repo-relative): no scratch files, no tests."""
        out = []
        for p in self.tools.changes.changed():
            rel = self.tools.rel(Path(p))
            name = rel.rsplit("/", 1)[-1]
            if (rel.endswith(".py") and not rel.startswith(("/", "..")) and not is_scratch_path(rel) and not name.startswith(("test_", "conftest"))
                    and not name.endswith("_test.py") and "tests" not in rel.split("/")[:-1]):
                out.append(rel)
        return sorted(out)

    def _tui_gate(self, nudges: dict[str, int], result: TurnResult) -> str | None:
        """Terminal program / game with an edited .py file: play it with seeded random keys on a real pty (tuigate.py, no model call). A crash sends the
        model back (at most twice, then the answer goes out as NOT VERIFIED); the same code is never played twice; a pass, a warning or a note only
        shows in the result line. Skipped silently without an entry, with coding.tui_gate false, or for a repo that is no terminal program."""
        cfg = self.config.get("coding") or {}
        if not cfg.get("tui_gate", True) or not tuismoke.repo_uses_tui(self.root):
            return None
        edited = self._tui_files()
        if not edited:
            return None
        res = self._tui_play(edited, force=False)
        if res is None or res.status != "fail":
            return None
        result.verified = False
        if nudges.get("tui", 0) >= 2:                            # the cap is spent: the answer goes out, labelled
            result.not_verified.append(f"program crashes under random play: {res.crash_line}")
            return None
        nudges["tui"] = nudges.get("tui", 0) + 1
        self.events.emit("tui_gate_nudge", entry=res.entry, n=nudges["tui"])
        self.console.print("[dim]  the program crashed under random play - back to work[/dim]")
        return ("Your program crashed when played with random keys. Reproduce with tui_smoke and fix the cause (not by hiding the error).\n" + res.text)

    def _tui_play(self, edited: list[str], force: bool) -> tuigate.GateResult | None:
        """The result for the current code: cached when nothing was edited since it was measured, else a fresh gate run on the entry."""
        cfg = self.config.get("coding") or {}
        entry = tuigate.find_entry(self.root, [self.root / r for r in edited], str(cfg.get("tui_entry") or ""))
        if entry is None:
            return None
        stamp = self._tui_content_stamp([*edited, entry])
        if not force and self.tui_last is not None and stamp == self._tui_stamp and self.tui_last.entry == entry:
            self.events.emit("tui_gate", status=self.tui_last.status, entry=entry, cached=True)
            return self.tui_last
        command = f"python3 {entry}"
        if not self._allowed_without_asking(command):
            res = tuigate.GateResult("skipped", f"TUI SMOKE: skipped (the repo policy would ask before running `{command}`)", entry=entry)
        else:
            self.console.print(f"  [cyan]✓?[/cyan] [dim]tui smoke  {escape(entry)}  (random keys, 3 runs)[/dim]")
            try:
                seconds = float(cfg.get("tui_gate_seconds", tuigate.DEFAULT_SECONDS))
            except (TypeError, ValueError):
                seconds = float(tuigate.DEFAULT_SECONDS)
            words = tuigate.banner_words(self.tools.changes.changed(), cfg.get("tui_banner_words") or [])
            try:
                res = tuigate.run_gate(entry, lambda c, k, size, t: self.tools._smoke(c, k, size, None, t, 50)[0], seconds=seconds,
                                       source=(self.root / entry).read_text(encoding="utf-8", errors="replace"), banners=words, command=command)
            except Exception as exc:  # noqa: BLE001 - the gate is a bonus: a problem of its own is never a crash of the engine
                logger.warning("tui gate failed", exc_info=True)
                res = tuigate.GateResult("skipped", f"TUI SMOKE: skipped ({type(exc).__name__})", entry=entry)
        self.tui_last, self._tui_stamp = res, stamp
        self.events.emit("tui_gate", status=res.status, entry=entry, seconds=round(res.seconds, 1), runs=res.runs, cached=False)
        color = "red" if res.status == "fail" else "yellow" if res.status == "skipped" or res.warnings or res.notes else "green"
        self.console.print(f"    [{color}]{escape(res.line)}[/{color}]")
        return res

    def _tui_content_stamp(self, rels: list[str]) -> str:
        import hashlib

        h = hashlib.sha1()
        for rel in sorted(set(rels)):
            try:
                h.update(rel.encode() + b"\0" + (self.root / rel).read_bytes() + b"\0")
            except OSError:
                h.update(rel.encode() + b"\0gone\0")
        return h.hexdigest()

    def tui_gate_now(self) -> tuigate.GateResult | None:
        """Play the program again now (verified.build(run_tui=True)); None when the repo has no terminal program entry."""
        if not tuismoke.repo_uses_tui(self.root):
            return self.tui_last
        return self._tui_play(self._tui_files(), force=True) or self.tui_last

    def _tui_unverified(self, result: TurnResult) -> None:
        """The answer goes out while the last crash-gate result is still a failure on the current code: say so (NOT VERIFIED)."""
        last = self.tui_last
        if last is None or last.status != "fail":
            return
        edited = self._tui_files()
        if edited and self._tui_content_stamp([*edited, last.entry]) == self._tui_stamp:
            result.verified = False
            result.not_verified.append(f"program crashes under random play: {last.crash_line}")

    # ── tests the standard way (stdpytest.py) ───────────────────────────
    def _std_check(self, force: bool = False) -> stdpytest.Finding | None:
        """The plain-`pytest` finding for the current code: cached on a content stamp (one run per real edit), None when the check does not apply."""
        cfg = self.config.get("coding") or {}
        cmd = self.profile.test_cmd
        if str(cfg.get("standard_pytest", True)).strip().lower() in ("false", "off", "0", "no") or not stdpytest.applies(self.root, cmd):
            self.std_last, self._std_stamp = None, None            # not (any more) applicable, e.g. a conftest.py was added: nothing stays flagged
            return None
        stamp = stdpytest.stamp(self.root)
        if not force and stamp == self._std_stamp:
            return self.std_last
        runner = self.std_runner or (lambda c, timeout: self.tools._exec(c, timeout))
        bare = stdpytest.standard_command(str(cmd))
        if bare is None or (self.std_runner is None and not self._allowed_without_asking(str(cmd))):
            return None
        self.console.print("  [cyan]✓?[/cyan] [dim]tests the standard way  (plain pytest, collection only)[/dim]")
        try:
            res = stdpytest.check(self.root, cmd, runner, timeout=int(cfg.get("verified_timeout_s", stdpytest.TIMEOUT) or stdpytest.TIMEOUT))
        except Exception:  # noqa: BLE001 - a bonus check: a problem of its own is never a crash of the engine
            logger.warning("standard pytest check failed", exc_info=True)
            return None
        self.std_last, self._std_stamp = res, stamp
        self.events.emit("std_pytest", fails=res is not None, module=res.module if res else "")
        return res

    def _std_gate(self, nudges: dict[str, int], result: TurnResult) -> str | None:
        """A python file was edited in this prompt / stage and the tests need `python -m pytest` to import the project: one nudge, then NOT VERIFIED."""
        if not stdpytest.touched(self.tools.changes.changed()):
            return None
        res = self._std_check()
        if res is None:
            return None
        result.verified = False
        if nudges.get("std", 0) >= 1:
            result.not_verified.append(res.not_verified)
            return None
        nudges["std"] = nudges.get("std", 0) + 1
        self.events.emit("std_pytest_nudge", module=res.module)
        self.console.print("[dim]  plain pytest cannot import the project - back to work[/dim]")
        return res.nudge

    def _std_unverified(self, result: TurnResult) -> None:
        """The gates are spent: a cached finding on the current code is labelled."""
        if self.std_last is not None and self._std_stamp == stdpytest.stamp(self.root):
            result.verified = False
            result.not_verified.append(self.std_last.not_verified)

    def std_pytest_now(self) -> stdpytest.Finding | None:
        """The finding for verified.build (cached per code state)."""
        return self._std_check()

    def _record_write(self, rel: str, before: str | None, after: str) -> None:
        """Every file-tool write of a running stage goes into the stage's edited line ranges (cursor.py). Every write is also announced live."""
        self._emit_edit(rel, before, after)
        if self.stage_edited is None or self.stage_goal is None:
            return
        old = None if before is None else before.replace("\r\n", "\n").split("\n")
        new = after.replace("\r\n", "\n").split("\n")
        for rows in (old, new):
            if rows and rows[-1] == "":
                rows.pop()
        self.stage_edited[rel] = cursor_mod.apply_change(self.stage_edited.get(rel), old, new)
        if self.stage_seen is not None:
            fileseen.note_edit(self.stage_seen, rel)

    def _emit_edit(self, rel: str, before: str | None, after: str) -> None:
        try:
            old = [] if before is None else before.replace("\r\n", "\n").splitlines()
            new = after.replace("\r\n", "\n").splitlines()
            diff = list(difflib.unified_diff(old, new, "/dev/null" if before is None else f"a/{rel}", f"b/{rel}", lineterm="", n=3))
            added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
            removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
            self.events.emit_live("edit", path=rel, added=added, removed=removed, created=before is None, diff=_live_clip("\n".join(diff)))
        except Exception:  # noqa: BLE001 - display only
            logger.debug("edit event failed", exc_info=True)

    def _next_unchanged(self, rel: str) -> int | None:
        """The next line of the running stage's window of `rel` that is not edited yet (tools: ALREADY APPLIED); -1 when the window is done,
        None outside a stage or for a file the stage does not name."""
        st = self.stage_obj
        if st is None or self.stage_goal is None:
            return None
        c = cursor_mod.cursor(st, self.root).get(rel)
        if not c:
            return None
        return -1 if c["total"] and c["next"] > c["total"] else int(c["next"])

    def _record_read(self, rel: str, a: int | None, b: int | None) -> None:
        """Every read of a running stage goes into the stage's FILES SO FAR record (fileseen.py)."""
        if self.stage_seen is not None and self.stage_goal is not None:
            fileseen.note_read(self.stage_seen, rel, a, b)

    def _chunk_limit(self) -> int | None:
        """The look-pause threshold of an editing-by-passes stage (coding.chunk_look_block), None for every other stage."""
        if not self.stage_chunk or self.stage_goal is None:
            return None
        return min(self._limit("chunk_look_block", cursor_mod.CHUNK_LOOK_BLOCK), self._limit("look_block_at", LOOK_BLOCK_AT))

    def _chunk_hint(self) -> str:
        if not self.stage_chunk or self.stage_obj is None:
            return ""
        return cursor_mod.next_chunk(cursor_mod.cursor(self.stage_obj, self.root))

    def scope_missing(self) -> list[str]:
        """The files of the running sub-stage (stage_scope) that were neither changed in the stage (ledger delta) nor waived; [] outside a stage."""
        if not self.stage_scope or self.stage_goal is None:
            return []
        files = self.ctx.ledger.files
        done = {*self.stage_changed0, *(p for p, e in files.items() if self.stage_marks.get(p) != (e.action, e.step, e.size))}
        st = self.stage_obj
        out = []
        for f in self.stage_scope:
            if f in self.stage_waived:
                continue
            if st is not None and getattr(st, "lines", None) and list(st.scope[:1]) == [f]:
                # a ranged part: a change INSIDE its range (the cursor's edited ranges, shifted by line count changes) or a waiver; a change the
                # cursor never saw (a shell write) counts when nothing at all was recorded for the file
                cursor_mod.sync_range(st, self.root)
                if cursor_mod.overlaps(st.edited.get(f), st.lines) or (f in done and not cursor_mod.merge(st.edited.get(f))):
                    continue
                out.append(f"{f} lines {st.lines[0]}-{st.lines[1]}")
            elif f not in done:
                out.append(f)
        return out

    def _waive_from(self, text: Any) -> None:
        """A scope file whose basename appears in a mark_done evidence / project_note counts as handled ('it needs no change')."""
        body = str(text or "")
        for f in self.stage_scope:
            if re.search(r"(?<![\w./-])" + re.escape(f.rsplit("/", 1)[-1]) + r"(?![\w])", body, re.I):
                self.stage_waived.add(f)

    def _open_items_stop(self, result: TurnResult) -> None:
        """The gates are spent and the answer would be accepted: with acceptance items still open that is not "done" - the run ends as
        "gate" and the delivered text names the open items (NOT VERIFIED), instead of silently counting as finished."""
        plan = self.plan.plan
        left = plan.open_items() if plan is not None else []
        missing = self.scope_missing()
        if missing:
            result.stopped, result.verified = "gate", False
            result.not_verified.append("files not handled: " + ", ".join(missing))
            self.events.emit("gate_stop", missing=missing)
        if not left:
            return
        result.stopped, result.verified = "gate", False
        result.not_verified += [f"acceptance item {i.id} not confirmed" for i in left[:6]] + ([f"{len(left) - 6} more items"] if len(left) > 6 else [])
        self.events.emit("gate_stop", open=[i.id for i in left])

    def _verify(self) -> str | None:
        """Run the project's checks. Returns a failure message for the model, or None when everything passed
        (or there is nothing to run)."""
        cmds = self.tools.check_command_list()
        if not cmds:
            return None
        for cmd in cmds:
            if any(c == cmd and code == 0 for _s, c, code in self.tools.evidence):
                continue                                                # the model already ran exactly this after its last edit
            self.console.print(f"  [cyan]✓?[/cyan] [dim]verify  {_one_line(cmd, 90)}[/dim]")
            level = classify_command(cmd, self.config)
            verdict = policy_mod.check_command(self.pol, self.root, cmd, level.name, owned=self.tools.owned)
            if verdict.verdict == "block":
                continue
            if verdict.verdict == "ask" and not self._approve(f"run  {_one_line(cmd, 100)}", "verification"):
                self.console.print("    [dim]skipped[/dim]")
                return None
            code, out, secs = self.tools._exec(cmd, 300)
            self.ctx.ledger.note_run(self.ctx.step, cmd, code, out.strip().splitlines()[-1] if out.strip() else "")
            self.tools.evidence.append((self.ctx.step, cmd, code))
            if code != 0:
                self.console.print(f"    [red]failed ({secs:.1f}s)[/red]")
                return verify.failure_message(cmd, f"exit code {code}\n{cap_output(out.strip(), 600, 1800)}")
            self.console.print(f"    [green]passed ({secs:.1f}s)[/green]")
        self._remember("tests", self.profile.test_cmd)
        self.tools.dirty.clear()
        return None

    def _duplicate_unverified(self, result: TurnResult) -> None:
        """A touched file still defines a name twice (static.py: def / class / module-level data block) and the static nudges are spent: the
        answer goes out labelled NOT VERIFIED. Seen live: a 141-line dict block existed twice after a replace_lines with a too small end line."""
        if any(x.startswith("duplicate definition") for x in result.not_verified):
            return
        found = []
        for rel, probs in self.tools.static_bad.items():
            for prob in probs:
                m = re.search(r"(\S+?) is defined twice", prob)
                if m:
                    found.append(f"{rel} ({m.group(1).strip(chr(39))})")
        if found:
            result.verified = False
            result.not_verified.append("duplicate definition: " + ", ".join(found[:3]))

    def _static_sweep(self) -> None:
        """Re-run the static checks on every Python file changed in this prompt (an edit elsewhere can break or fix a name)."""
        from agent.codeagent import static

        changed = [self.tools.rel(Path(p)) for p in self.tools.changes.changed()]
        fresh = static.check_files(self.root, changed)
        for rel in changed:
            if rel in fresh:
                self.tools.static_bad[rel] = fresh[rel]
            else:
                self.tools.static_bad.pop(rel, None)

    # ── checks (checks.py) ──────────────────────────────────────────────
    def check_runner(self) -> checks_mod.Runner:
        return checks_mod.Runner(self.root, exec=lambda cmd, timeout: self.tools._exec(cmd, timeout), allow=self._allowed_without_asking,
                                 review=self._review, tui=self.tools.tui_check)

    def _allowed_without_asking(self, command: str) -> bool:
        """A model-written check command runs only if the repo policy would run it without asking (a check never prompts)."""
        level = classify_command(command, self.config)
        if level is RiskLevel.BLOCKED:
            return False
        return policy_mod.check_command(self.pol, self.root, command, level.name, owned=self.tools.owned).verdict == "allow"

    def _review(self, criterion: str, output: str) -> dict[str, Any]:
        with self.console.status("[dim]reviewer (fresh context) judging the output…[/dim]", spinner="dots"):
            return reviewer.review(self.llm, self.model, self.brief(), criterion, output, num_ctx=self.num_ctx, events=self.events)

    def run_checks(self, which: list[checks_mod.Check] | None = None, quiet: bool = False) -> list[checks_mod.Check]:
        """Run checks (all by default). A rubric check is judged again only after the code changed (a review costs a model call)."""
        todo = self.checks.all() if which is None else which
        runner = self.check_runner()
        for c in todo:
            last = c.last_result or {}
            # fresh = nothing changed since: the progress counter AND the stamp of the ledger's files on disk must match (the counter
            # alone misses an edit made outside the tools, or by an older session that did not save it)
            if c.kind == "rubric" and last.get("at") == self._progress and last.get("status") in ("pass", "fail") \
                    and last.get("stamp") == checks_mod.content_stamp(self.root, self.ctx.ledger.files):
                continue
            self._set_now(f"running check {c.id} ({c.kind})")
            if not quiet:
                self.console.print(f"  [cyan]✓?[/cyan] [dim]check  {c.id}  ({c.kind})[/dim]")
            res = checks_mod.run(c, runner)
            res["at"] = self._progress
            res["stamp"] = checks_mod.content_stamp(self.root, self.ctx.ledger.files)
            self.checks.remember(c)
            self.events.emit("check", id=c.id, check_kind=c.kind, status=res["status"], seconds=res["seconds"], source=c.source)
            if not quiet:
                color = "green" if res["status"] == "pass" else "yellow" if res["status"] in ("skipped", "unreviewed") else "red"
                self.console.print(f"    [{color}]{res['status']} ({res['seconds']:.1f}s)[/{color}]")
        return todo

    def _checks_gate(self, result: TurnResult, spent: bool = False) -> str | None:
        """The final gate over ALL checks: failing ones (fail / error) send the model back with their short output and the brief card,
        at most CHECK_RETURNS times each; then the answer goes out and every check that did not pass is listed as NOT VERIFIED."""
        checks = self.checks.all()
        if not checks:
            return None
        self.run_checks(checks)
        failing = [c for c in checks if c.status() in checks_mod.FIXABLE]
        retry = [c for c in failing if self._check_returns.get(c.id, 0) < checks_mod.CHECK_RETURNS] if not spent else []
        if retry:
            for c in retry:
                self._check_returns[c.id] = self._check_returns.get(c.id, 0) + 1
            result.verified = False
            self.events.emit("checks_failed", ids=[c.id for c in retry])
            self.console.print(f"[dim]  {len(retry)} check(s) failed - back to work[/dim]")
            return checks_mod.failure_block(retry, self.brief())
        open_ = [c for c in checks if c.status() not in checks_mod.PASSING]
        result.not_verified += [f"{c.id} ({c.status()})" for c in open_]
        if open_:
            result.verified = False
        return None

    def _make_check_tool(self):
        from agent.tools.registry import Tool

        def add_check(id: str = "", description: str = "", kind: str = "", items: Any = None, **kw: Any) -> str:  # noqa: A002 - tool argument
            cid = re.sub(r"[^\w.-]+", "-", str(id or "").strip())[:60]
            if not cid:
                return "error: add_check needs id (short, unique), description, kind and the kind's fields."
            kind = str(kind or "").strip().lower()
            keys = {"command": ("command", "timeout"), "python": ("path", "timeout"), "banned": ("patterns", "regex", "files", "objects"),
                    "rubric": ("criterion", "scenario", "timeout"),
                    "tui": ("command", "keys", "expect", "size", "timeout", "settle_ms", "exit_code")}.get(kind, ())
            spec = {k: kw[k] for k in keys if kw.get(k) not in (None, "", [])}
            for k in ("patterns", "files", "objects", "scenario", "expect"):
                if isinstance(spec.get(k), str) and k != "scenario":
                    spec[k] = [x.strip() for x in re.split(r"[\n,]", spec[k]) if x.strip()]
            ids = items if isinstance(items, list) else ([] if items in (None, "") else [items])
            check = checks_mod.Check(cid, str(description or cid), kind, spec, "model", [int(i) for i in ids if str(i).strip().isdigit()])
            why = self.checks.add(check)
            if why:
                return f"error: {why}"
            self.events.emit("check_added", id=cid, check_kind=kind)
            if kind == "rubric":
                return f"check {cid} added (rubric: the reviewer judges it at the final gate on the scenario output)."
            res = self.run_checks([check], quiet=True)[0].last_result or {}
            return f"check {cid} added; it runs at every final gate" + (f" and gates mark_done for item(s) {check.items}" if check.items else "") \
                + f". Ran it now: {res.get('status', '?').upper()}\n{checks_mod._clip(res.get('output', ''), 200, 600)}"

        S = {"type": "string"}
        tui_first = "For terminal programs prefer kind=tui. " if tuismoke.repo_uses_tui(self.root) else ""
        return Tool(name="add_check", description=tui_first + (
            "Add a check the harness runs itself at the final gate (and before mark_done of the items it names): kind=command (command; pass = "
            "exit 0), python (path of a script you wrote under .scratch/checks/; pass = exit 0 - `from agent.codeagent import textcheck` has "
            "sentence-level duplicate and banned-phrase helpers), banned (patterns that must not occur in files=[globs] / objects=[\"module:NAME\"]), "
            "rubric (criterion + scenario commands; an independent reviewer judges the output), tui (command + keys + expect [+ size, exit_code]: "
            "a curses / terminal program is run in a real terminal by the harness like tui_smoke; expect = what the player must SEE, e.g. the game over banner; "
            "exit_code=0 requires a clean exit). "
            "Make checks strict: they decide what counts as done."),
            parameters={"type": "object", "properties": {
                "id": S, "description": S, "kind": {"type": "string", "enum": list(checks_mod.KINDS)}, "command": S, "path": S,
                "patterns": {"type": "array", "items": S}, "regex": {"type": "boolean"}, "files": {"type": "array", "items": S},
                "objects": {"type": "array", "items": S}, "criterion": S, "scenario": {"type": "array", "items": S},
                "keys": {"type": "array", "items": S}, "expect": {"type": "array", "items": S}, "size": S, "exit_code": {"type": "integer"},
                "settle_ms": {"type": "integer"},
                "items": {"type": "array", "items": {"type": "integer"}}, "timeout": {"type": "integer"}},
                "required": ["id", "kind"]}, handler=add_check, risk="safe")

    def _gated_mark_done(self, tool: Any) -> Any:
        """mark_done for an item that has checks is accepted only when the harness ran them and they pass; the model's evidence is ignored."""
        inner = tool.handler

        def mark_done(id: Any = None, evidence: str = "", ids: Any = None, **kw: Any) -> str:  # noqa: A002 - the tool's argument name
            self._waive_from(evidence)
            wanted = list(ids) if isinstance(ids, (list, tuple)) else ([] if ids in (None, "") else [ids])
            if id not in (None, ""):
                wanted.insert(0, id)
            ok, refused = [], []
            for raw in wanted:
                try:
                    item = int(raw)
                except (TypeError, ValueError):
                    ok.append(raw)
                    continue
                linked = self.checks.for_item(item)
                if not linked:
                    ok.append(item)
                    continue
                self.run_checks(linked, quiet=True)
                bad = [c for c in linked if c.status() not in checks_mod.PASSING]
                if bad:
                    refused.append(f"item {item} NOT confirmed - its checks did not pass: "
                                   + "; ".join(f"{c.id}: {c.status()} {checks_mod._clip((c.last_result or {}).get('output', ''), 80, 300)}" for c in bad))
                else:
                    ok.append(item)
                    evidence = f"{evidence} [harness: checks {', '.join(c.id for c in linked)} passed]".strip()
            out = [inner(ids=ok, evidence=evidence if len(evidence) >= 12 else f"{evidence} (harness)".strip(), **kw)] if ok else []
            return "\n".join(out + refused) or inner(id=id, evidence=evidence, ids=ids, **kw)

        tool.handler = mark_done
        return tool

    # ── output ──────────────────────────────────────────────────────────
    def _print_final(self, content: str, message: dict[str, Any], stored: str | None = None) -> None:
        from rich.markdown import Markdown
        from rich.panel import Panel

        used = self.ctx.last_prompt + self.ctx.last_eval
        bar = f"{used:,} / {self.num_ctx:,} tokens".replace(",", ".") + (f" · {self.ctx.last_tps:.0f} tok/s" if self.ctx.last_tps else "")
        self.console.print(Panel(Markdown(content), border_style="cyan", title="[bold cyan]code[/bold cyan]", subtitle=f"[dim]{bar}[/dim]", padding=(0, 1)))
        self.ctx.add({"role": "assistant", "content": content if stored is None else stored})


_WROTE = ("wrote ", "edited ", "appended ", "patched ", "deleted ", "moved ")


def _real_write(name: str, args: dict[str, Any], result: str) -> bool:
    """A successful write to a project file (not a scratch copy, not one that left a syntax error)."""
    if name == "run":                                  # a shell edit, noted by tools.run
        return real_edit(name, args, result) and "SYNTAX GATE FAILED" not in str(result).split("NOTE: this command changed", 1)[-1]
    if KINDS.get(name) != WRITE or not str(result).startswith(_WROTE) or "SYNTAX GATE FAILED" in str(result):
        return False
    target = str((args or {}).get("path") or (args or {}).get("dst") or (args or {}).get("dest") or "")
    return not is_scratch_path(target)


_OPENER = re.compile(
    r"(?i)^(?:(?:ok(?:ay)?|good|great|done|perfect|alright|excellent|gut|fertig|so|super)\s*[,.!:]?\s+)?"
    r"(now|next|then|after that|let me|let's|i'll|i will|i'm going to|i am going to|i need to|moving on|proceeding|"
    r"jetzt|nun|als n[äa]chstes|anschlie(?:ß|ss)end|danach|ich werde|weiter mit)\b[,:]?\s*(.*)$")
_SELF = {"let me", "let's", "i'll", "i will", "i'm going to", "i am going to", "i need to", "moving on", "proceeding", "ich werde", "weiter mit"}
_ACTION = re.compile(
    r"(?i)^(?:(?:i(?:'ll| will| need to| am going to|'m going to)?|let(?:'s| me)|we(?:'ll| will)?|the next step is(?: to)?|step \d+\W*|"
    r"to|on to|ich|wir|i'll now|also|now|jetzt)\s+)*"
    r"(update|add|write|create|implement|fix|run|test|wire|build|move|edit|refactor|finish|integrate|make|change|replace|extend|"
    r"complete|continue|proceed|start|handle|modify|hook|register|define|split|verify|check|read|look|tackle|set up|put|remove|"
    r"rename|clean|generate|apply|connect|adjust|document|work on|port|aktualisier|füge|schreib|erstell|bau|teste|implementier|"
    r"passe|änder|mach|ergänz|prüf|start)\w*\b")
_NOT_MINE = re.compile(r"(?i)\b(you|your|sie|du|dir|euch|could|would|might|optional\w*|suggest\w*|recommend\w*|cannot|can't|unable|"
                       r"blocked|permission|approv\w*|stop|leave it|pause|wait for|kann nicht|könnten?|würde)\b|\?")


def announced_step(content: str) -> str | None:
    """The last sentence of `content` when it announces the writer's OWN next action ("Now update main.py ...", "Next, I'll add the
    tests:", "Jetzt passe ich main.py an"), else None. Suggestions to the user, questions and "blocked/cannot" are not announcements."""
    lines = [ln.strip() for ln in str(content or "").strip().splitlines() if ln.strip() and not ln.strip().startswith("```")]
    if not lines:
        return None
    last = re.sub(r"^(?:[-*>#]+|\d+[.)])\s*", "", lines[-1]).replace("**", "").replace("__", "").strip()
    sentence = re.split(r"(?<=[.!])\s+(?=[A-ZÄÖÜ])", last)[-1].strip()
    m = _OPENER.match(sentence)
    if not m or _NOT_MINE.search(sentence) or len(sentence) > 240:
        return None
    if m.group(1).lower() not in _SELF and not _ACTION.match(m.group(2)):
        return None
    return sentence


def _run_stem(args: dict[str, Any]) -> str:
    """The program part of a `run` command whose output is piped into filters (`prog 2>&1 | sed -n '1,45p'` -> `prog 2>&1`), else ''."""
    command = str((args or {}).get("command", ""))
    cut = filter_split(command)
    return " ".join(command[:cut].split()) if cut is not None else ""


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def _call_key(args: Any) -> str:
    return " ".join(json.dumps(args, sort_keys=True, default=str, ensure_ascii=False).split())


def _result_key(result: Any) -> str:
    """A tool result without its timing; long output keeps head and tail (a game prints the same banner every run)."""
    r = " ".join(_TIMING.sub("", str(result)).split())
    return r if len(r) <= 1600 else r[:800] + " … " + r[-800:]


def _failed_run(result: Any) -> bool:
    """A failed tool run: an error shape or a command with a non-zero exit code."""
    return _errorish(result) or bool(re.match(r"\s*exit code [1-9]", str(result), re.I))


def _errorish(result: Any) -> bool:
    return str(result).lstrip().lower().startswith(("error", "edit_file failed", "refused", "blocked", "no such", "exit code 1", "exit code 2",
                                                    "not run", "the user declined", "apply_patch failed"))


def _ratio_at_least(a: str, b: str, ratio: float) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    m = difflib.SequenceMatcher(None, a, b)
    return m.real_quick_ratio() >= ratio and m.quick_ratio() >= ratio and m.ratio() >= ratio


_NUMBER_VALUE = re.compile(r"(?<=: )-?\d+(?=\s*[,}])")   # a numeric argument VALUE; digits inside strings (f1.py) are not touched


def _same_call(name: str, a: str, b: str) -> bool:
    """Exact, or equal once numbers are ignored (a search re-sent with `context` 9, 10, 11 ... gave the same result 24 times); for `run` a
    nearly identical command counts too (the same playthrough with two inputs swapped). Callers also require the same RESULT."""
    return a == b or _NUMBER_VALUE.sub("#", a) == _NUMBER_VALUE.sub("#", b) or (name == "run" and _ratio_at_least(a, b, 0.85))


def _same_result(a: str, b: str) -> bool:
    return _ratio_at_least(a, b, 0.95)


_TOOLISH = re.compile(r'"tool"\s*:|<parameter=|<function=|<tool_call>', re.I)


def _cut_off_note(tokens: int) -> str:
    return (f"Your reply was cut off after {tokens} tokens and nothing ran: write the file in several smaller write_file/append_file calls of at most "
            "~150 lines each (or split the work into modules), and do the next step as ONE short tool call without long explanations.")


def _coerce_param(value: str, spec: dict[str, Any]) -> Any:
    """An XML parameter value is always text; give it the JSON type its schema asks for (integer, boolean, array, object)."""
    want = spec.get("type")
    try:
        if want == "integer":
            return int(value.strip())
        if want == "number":
            return float(value.strip())
        if want == "boolean" and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if want in ("array", "object"):
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list if want == "array" else dict) else value
    except ValueError:
        pass
    return value


def salvage_call(content: str, tool_names: set[str], schema: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Recover a tool call from garbled text. Seen from the 27B: `{"tool": "read_file",\n<parameter=arguments>\n{...}\n</parameter>`
    (JSON and XML mixed) and the qwen-coder style `<function=NAME><parameter=KEY>VALUE</parameter></function>`; also the Hermes shape
    `{"name": NAME, "arguments": {...}}` (bare, fenced or inside <tool_call>) that Qwen's templates document, with `arguments` possibly a JSON
    string. With `schema` (the Ollama tools list) XML parameter values get their declared type (a `start` of "3" becomes 3)."""
    from agent.jsonx import extract_json_value

    hermes = extract_json_value(content, "{", "}")
    if isinstance(hermes, dict) and hermes.get("name") in tool_names and "tool" not in hermes:
        from agent.tui import _normalize_arguments

        raw = hermes.get("arguments", hermes.get("parameters", {}))
        args = _normalize_arguments(raw)
        if args or raw in ({}, None):
            return {"name": hermes["name"], "arguments": args}
    m = re.search(r'"tool"\s*:\s*"([A-Za-z_]\w*)"', content) or re.search(r"<function=([A-Za-z_]\w*)>", content)
    if not m or m.group(1) not in tool_names:
        return None
    name, rest = m.group(1), content[m.end():]
    params = re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", rest, re.S)
    if params and not (len(params) == 1 and params[0][0] == "arguments"):
        props = next((t["function"].get("parameters", {}).get("properties", {}) for t in schema or [] if t["function"]["name"] == name), {})
        return {"name": name, "arguments": {k: _coerce_param(v, props.get(k, {})) for k, v in params}}
    args = extract_json_value(rest, "{", "}")
    if isinstance(args, dict):
        if set(args) == {"arguments"} and isinstance(args["arguments"], dict):
            args = args["arguments"]
        return {"name": name, "arguments": args}
    return {"name": name, "arguments": {}} if not rest.strip(" \n,}<>/parameter=arguments") else None


def _strip_think(content: str) -> str:
    import re

    content = re.sub(r"<think>.*?</think>\s*", "", content or "", flags=re.DOTALL)
    return re.sub(r"^.*?</think>\s*", "", content, flags=re.DOTALL) if "</think>" in content else content
