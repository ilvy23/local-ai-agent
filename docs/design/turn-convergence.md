# Turn convergence (P7): terminal loop vs. web engine

Status: step 1 of 2. Step 1 moves code and adds `agent/chat/helpers.py` + `agent/chat/gates.py` without changing
behaviour; step 2 wires the gates into `agent/chat/turn.py::run_turn` and the web executor.

Two loops exist today:

* terminal: `tui._run_agent_turn_core` (+ `_execute_tool`, `_run_agent_turn` wrapper, pre/post work in `run_repl` and
  `ui/chat_adapter.run_chat_turn`);
* web: `chat/turn.py::run_turn` with the executor from `web/agent_runtime.py` (suspends on `ApprovalNeeded`).

## Inventory

"Web has it" = present in run_turn/executor today. Home: H = `chat/helpers.py` (pure), G = `chat/gates.py` (stateful).

| Feature | Where (terminal) | State touched | Web has it | Web-relevant | Home |
|---|---|---|---|---|---|
| Tool-call extraction / arg repair | `_extract_tool_call`, `_normalize_arguments` | - | yes (imports) | yes | H (done) |
| Result framing, denial framing | `_frame_tool_result`, denial literal | - | yes (own copy of denial) | yes | H (done, one copy) |
| Risk classification | `_classify_call`, `_static_risk_level` | taint (read) | yes | yes | H (done) |
| `<think>` stripping, num_ctx | `_strip_thinking_artifacts`, `_num_ctx` | gpumode (read) | yes | yes | H (done) |
| Stuck watcher reaction | `_react_to_stuck` | `StuckWatcher` per turn | yes, but only with `console`, fixed `count=2`, denial not passed | yes | H (done); step 2: fix call site |
| remember/take_note/save_skill tools | `_make_*_tool` | store | yes (deps) | yes | H (done) |
| Verify gate (run project check before "done", retry <= N) | core: `gate.due()` -> `_run_verification`; `gate.note_edit` after clean edit | `VerifyGate` per turn, `_PROJECT_HOOK` (module global, set in `setup_chat`) | NO | yes (needs run_command through the executor) | G `make_verify_gate`, `note_clean_edit`, `run_verification` |
| Plan nudge (<= 2 nudges, then "unconfirmed") | core: after final answer | `plan_holder.plan.nudges`, reset in `_run_agent_turn` | NO | yes | G `PlanGate` |
| Plan creation (pre-turn) | `run_repl`, `run_chat_turn`: `should_create` + config + `AGENT_PLAN` | `plan_holder.plan` | NO | yes | G `plan_creation_enabled` |
| Goal restatement every N tool calls | core, after each tool | `goal_state` dict (total_calls/every_n/original_task) | NO | yes | G `GoalReminder` |
| Step extension (ask for more steps when budget used) | core + `_ask_more_steps` (console.input) | `budget` local | NO (hard stop + error) | yes, as approval-style event | G `StepBudget` (callback) |
| Per-prompt change log + git checkpoint | `_execute_tool` (checkpoint_tried, record) | `_CHANGES` | NO | yes (diff/edit events, Undo) | G `ChangeTracker` |
| Undo stack, `_LAST_EDITED` (active project) | `_run_agent_turn` finally; `/undo` | `_UNDO`, `_LAST_EDITED` | NO | undo: yes later | G `ChangeTracker`, shared singletons |
| Closing `changes` / `done` / `error` events, `print_summary` | `_run_agent_turn` | `_CHANGES` | partial (`done` only) | yes | G `ChangeTracker.finish` + event build in step 2 |
| Repo access / taint check before asking | `_execute_tool`: `taint.copied` + `repoaccess.allows` | taint, repoaccess grants | NO (web always asks) | yes | G `decide_access` (pure) |
| BLOCKED handling, audit log, edit events (diff) | `_execute_tool` | store | partial (no diff, no timing) | yes | stays in executor, step 2 aligns |
| Safe-read printing (`read >`), tool status, search evidence event | `_execute_tool`, `_emit_search_evidence` | `turn_evidence` | NO sources event on tool path | yes (`sources`) | stays; step 2 emits via executor |
| Forced `/web` pre-step, correction nudge | `run_repl`, `run_chat_turn` | `turn_evidence`, store | NO | maybe | later (not a gate; pre-turn step) |
| Context compaction (mask + compact, forced `/compact`) | `tui._maybe_compact` | `ctl.compact` | duplicate in turn.py (no force) | yes | later; one implementation in step 2 |
| Empty-content re-roll, interrupted handling, streaming sink | core | - | re-roll yes; interruption yes | yes | already shared in shape |
| Per-turn singleton resets | `taint.clear()`, `turn_evidence.clear()` in run_repl/adapter | taint, evidence | NO | yes | step 2 (turn start) |

## Gate APIs (contract for step 2)

All in `agent/chat/gates.py`, none prints or reads stdin; the front end decides how to show things.

* `make_verify_gate(config, registry) -> VerifyGate | None`; `note_clean_edit(gate, call, result)`;
  `run_verification(gate, run_command, messages, answer, *, say, event, width, on_pass) -> list | None`
  (`run_command(cmd) -> result` goes through the front end's executor; returns messages to retry with, or None).
* `PlanGate.reset(holder)`; `PlanGate.check(holder, content) -> PlanDecision | None` (`action` "nudge"|"unconfirmed",
  `line` console markup, `event` dict, `retry_messages(messages, content)`); `plan_creation_enabled(text, config)`.
* `GoalReminder(goal_state).tick() -> GoalDecision | None` (`log_text`, `message`).
* `StepBudget(max_iterations, ask_more=None)`; `.may_continue(iteration) -> bool` (extends by `max_iterations` when the
  callback says yes; no callback = stop).
* `ChangeTracker(log, undo_stack, last_edited)`; `ChangeTracker.shared()`; `.begin()`, `.checkpoint_before(path)`,
  `.record(path, before, after)`, `.finish() -> dict`, `.totals()`, `.change_events()`, `.undo_lines(list_only)`.
* `decide_access(name, arguments, level) -> AccessDecision(from_web, auto_allowed)`.

Singletons `CHANGES`, `UNDO`, `LAST_EDITED` now live in gates.py; `tui._CHANGES/_UNDO/_LAST_EDITED` are aliases of the same
objects (`tui._PROJECT_HOOK` stays in tui: it is rebound by `setup_chat` and handed to the gate as a callable).
Not re-entrant yet (one turn per process), as before.

## Step 2 (done): one engine

`agent/chat/turn.py::run_turn(..., gates=TurnGates(...))` is now the engine of the web chat and of the new terminal chat
(`ui/chat_adapter.run_chat_turn` -> `run_agent_turn_engine`). The classic `chat --classic` loop
(`tui.run_repl` -> `_run_agent_turn_core` / `_execute_tool`) is unchanged legacy; tests/test_chat_parity.py runs the same
scripted scenarios through both and compares event kinds/order, the payload fields the UI reads, the persisted messages and
the exact prompts the model saw.

* `agent/chat/executor.py`: the ONE tool executor (`ToolExecutor`). Approval is a strategy: `BlockingApprover` (terminal,
  `approve_fn`, honours folder grants except for web-copied text) and `SuspendApprover` (web: pending approval + `ApprovalNeeded`,
  never auto-allows). `ExecutorStyle` keeps the two event dialects (terminal: `tool_call` first, `risk`, `status`; web: silent
  for unknown tools / bad arguments, `result_preview`).
* `agent/chat/compaction.py`: the one compaction implementation (incl. the forced `/compact` pass); `tui._maybe_compact` delegates.
* Web: `agent/web/agent_runtime.py` keeps a `WebTurnRuntime` per session (change log, verify gate, goal count, plan holder,
  queued `edit` events) so the gates survive the suspension for an approval. Config: `chat.gates.{plan,goal,step_extension,changes,verify}`
  (web defaults: all on except `step_extension`).
* Fixed on the way: `TurnControl.event("stuck", kind=...)` raised `TypeError` (the first parameter was also called `kind`), so a stuck
  reaction crashed any terminal-UI turn on the legacy loop; the web reported a made-up `count=2` / empty query.
* Corrections to the inventory: the web already received `sources`/`verdict` (the web tools emit them through the console), so the executor
  reports search evidence only in the terminal style. Plan creation and the `mark_done` tool are wired into the web route/registry.

Left on the legacy loop: `chat --classic` (`run_repl`) only. Not shared yet: the pre-turn forced `/web` step and the correction nudge
(they stay in `run_chat_turn`), step extension on the web (needs a GUI event).
