# Handoff: nightage (ilvy23/local-ai-agent)

## Current status (read this first)
- `main` = `claude/hopeful-allen-7l6x9m` = commit 40c5f58. Everything below marked done is merged and pushed.
- User setup: NixOS, AMD RX 9070 XT 16 GB + RTX 3060 Ti 8 GB, two Ollama servers:
  - the normal one (:11434, AMD only);
  - `ollama-vk` (Vulkan, :11436, both cards, "dual" mode).
- Models: gpt-oss:20b and a custom `qwen3.8:27b-text` (only on the Vulkan server).
- The user installs with `curl -LsSf https://raw.githubusercontent.com/ilvy23/local-ai-agent/main/install.sh | sh`.
  Config: ~/.local/share/nightage/config.yaml.

### Done (rounds 1-3)
- Phases 0-11 and 13 of the plan below.
- From 12/14: GPU-detect cache, English GPU question, tools check off the main thread, one db_path, per-turn web
  taint/evidence.
- Round 2: enum aliases (`30d` -> month) in agent/toolargs.py; two-GPU context label on the home screen.
- Round 3, long prompts:
  - agent/mentions.py: `@file` inlines the whole file in chat and `nightage code`. `as_task=True` for code makes the
    file text the task, plus a TASK_NOTE naming the file.
  - GoalReminder capped at 1,500 chars (agent/chat/gates.py).
  - Stage planner reads up to 60k chars, and each stage goal names the spec file (`stages._overall`).
  - A show_symbol #hash vouches for its sub-ranges in replace_lines.
  - No web lookup for the agent's own tool errors (`engine._OWN_TOOL_ERROR`).
  - EOF hint when a program waiting for input runs without stdin.
- modelpick.installed() merges the Vulkan server's /api/tags in a GPU mode (a model that is only on ollama-vk is no
  longer "missing").

### Still open (priority order)
1. The home screen does not reload or resolve the config after Settings (it shows stale models until a restart).
   Fix in `main._home()`.
2. Invented sources: check "according to X" claims in a research answer against this turn's evidence
   (agent/research/evidence.take).
3. agent/web/routes/chat.py: move blocking work to asyncio.to_thread.
4. One shared Ollama snapshot for home / ollama_local / modelpick.
5. Ask the GPU question after role resolution; show resolved roles on the home screen.
6. Smaller items:
   - cap read_file/run_command results sent to the model;
   - write-protect config.yaml/greenlist;
   - quoted Windows paths in the /auto scope check;
   - code_admin stdout swap;
   - /api/status double game check and nvidia-smi on every poll.
7. Model-side issues, not fixed:
   - gpt-oss invents tool names (e.g. `run_file`) -> maybe tool-name aliases;
   - Ollama "error parsing tool call" on large file writes -> suggest write_file in chunks or append_file;
   - pytest missing on NixOS -> report it as "not run", not "failed".
- Deferred: native tool-call turns for gpt-oss, task-ranked repo map, sqlite-vec kind partition, `bash -c` vs `-lc`.

### Rules for working on it
- Run the targeted tests, then `uv run --frozen pytest -q`. The 19 env-only failures in the cloud sandbox (installer
  tests, Windows import) are known.
- The conftest isolates the real config.yaml. Never write the user's config from tests.

---
# Full original plan (history)

# Plan: full bug sweep + high-impact improvements (model pipeline, coding agent, framework)

## Context
The user runs nightage on NixOS: AMD 16 GB + RTX 3060 Ti 8 GB, a second "Vulkan" Ollama on :11436 spanning both
cards, main model gpt-oss:20b. They saw wrong models after `/model`, gpt-oss suddenly 100 % on the CPU (9 vs 85 tok/s,
4-min tool turns), ~10 s waits, the context choice not applied, confusing texts. First fixes are pushed on
`claude/hopeful-allen-7l6x9m` (follow_chat, `_foreground_models`, cap_ctx bypass, `_slow_note`, warm-up, prose GPU).
A read-only audit (3 investigations + design review, key claims re-verified by hand) found ~60 more real issues,
including critical ones:
- `safety.classify_command("ls\nrm -rf ~")` == 0 (SAFE, runs without asking); `rm -rf ~` alone == 3. Same for `&`,
  `<(…)`, `find -fprint`, `env -S`, `rg --pre`, `git --output`.
- coding agent: `apply_patch` policy check ignores `path=` (writes to `~/.ssh/...` allowed in medium/auto);
  `apply_hunks` silently deletes the line after a change (blank context trimmed for matching, not for the splice);
  non-UTF-8 files corrupted on edit (decode `errors="replace"`, write UTF-8); read-only protection off for root.
- my `/model` change made the test suite write `background: m:2` into the real `config.yaml` (gitignored).
- the CPU problem is only half fixed: the memory catch-up 1.5 s after start runs a background call with `num_gpu 0`
  before the first chat call, and Ollama then keeps reusing that CPU runner (a request without `num_gpu` matches it).
  Three helper calls in `tui.py` send no `num_ctx` -> reload at default context, then reload again.
- memory facts/notes/summary sit in the system prompt and the memory worker runs after every turn on the same model
  -> Ollama's cached conversation is thrown away and the next reply re-reads the whole chat.
Baseline (3e18559): 23 test failures, all from this sandbox (root; fastapi missing). 4 of them are real root bugs
(read-only/home jail) and must pass after Phase 3; the other 19 (18 installer, 1 Windows import) stay env-only.

## User decisions
All phases, one commit + push per phase (stoppable at any point) · memory extraction **only when idle** (30 s
default, never before the first reply of a session) · **unload on quit stays** · gpt-oss reasoning **low for helper
calls, medium for chat/code** + a setting for that effort. Experiments (native tool calls, ranked repo map) deferred.

## Phase 0 – commit pending work
Commit + push the tested uncommitted change (`check_model_loaded` tag matching, `wants_vulkan` DUAL skips embedders).
Reset the sandbox `config.yaml` `models.background` to `auto`.

## Phase 1 – test isolation (stop tests writing the real config)
- `config.load_config(config_path=None, data_dir=None)`: resolve defaults at call time.
- autouse fixture in `tests/conftest.py` pointing `agent.config.DEFAULT_CONFIG_PATH`, `agent.settings.DEFAULT_CONFIG_PATH`
  (and any module copy equal to the real path) + a fresh `config.LAST_LOAD` at a tmp file; session guard hashing the
  real `config.yaml` that fails the run if it changed.
- one writer for `/model`: `settings.switch_chat_model(config, new)` (chat + follow_chat roles in one write), used by
  `tui._handle_model_switch`, `ui/handlers.cmd_model`, `web/routes/models.select_model`.

## Phase 2 – shell classifier (critical security)
- new pure `agent/shellparse.py`: `split(cmd) -> list[Segment] | None` (None = unparseable -> DANGEROUS). One lexer
  for '…', "…" (escapes), `$'…'`, backslash/line continuation, nesting `$( <( >( `` ( ) { }`; top-level separators
  `; ;; \n && || | |& &`; not separators `&> &>> >&N <&N N>&M >|`; heredoc bodies kept apart from command text.
- `agent/safety.py`: `_split_segments`/`_block_reason_recursive` use it; redirect tokens `&> &>> >| <>`; `/dev/null`,
  `/dev/stdout|stderr` targets no longer demote SAFE; `<(`/`>(` + substitutions in unquoted heredocs classified;
  shell keywords stripped (`if/then/do/while/!/{`), `case`/`select` DANGEROUS; `env -S/--split-string`, `xargs`,
  `eval`, `watch`, `flock`, `su/runuser/script -c` classify their command; command names with `$`, backtick, glob or
  brace expansion DANGEROUS (try `codeagent.policy.resolve_vars` first); heredoc into sh/bash/zsh DANGEROUS;
  `find -fprint*/-fls` (reuse `codeagent/lookact._FIND_ACTS`), `rg --pre`, `git --output` treated as writes.
- tests: new tables in `tests/test_safety.py` (every bypass above BLOCKED/DANGEROUS; `ls 2>&1`, `ls &>/dev/null`,
  `grep -r x . 2>/dev/null | head` SAFE; heredoc file write CAUTION); all existing cases unchanged.
- UX: `2>/dev/null` commands stop asking; heredoc file writes ask y/N instead of a typed "yes".

## Phase 3 – coding-agent gates
- `edits.patch_paths(patch, path)` used by both `engine.decide` (~:497) and `tools.apply_patch` (~:1091).
- `tools.is_read_only(p, st)` = owner-write bit clear OR (foreign owner and not `os.access`); used in `_protected`
  (~:871) and `snapshot_tree` (~:177); fix the `os.access` assertion in `tests/test_codeagent_runtrack.py:114`.
- `codeagent/policy.py`: `_under` with normcase + both separators; foreign homes = {/home,/Users,/var/home,/root} +
  pwd home parents minus own home (replaces the `dirname(home)=="/home"` test); SAFE commands touching paths outside
  the repo ask unless `repoaccess._shell_in_scope`/`_scope_with_allow` allow (as the docstring promises) – UX change.
- `_verify` returns passed/failed/not-run; blocked, declined or exit 127 => NOT VERIFIED, never "verified working".
- tests: header-less patch to `~/.ssh/...` blocked; read-only kept when `os.access` is always true; `/home/other`
  jailed with home `/root`; outside read asks; blocked check stays unverified. The 4 root baseline tests pass.

## Phase 4 – edit engine + encodings (data loss)
- `edits.apply_hunks`: drop blank context at hunk edges on both sides first, then match untrimmed with the ladder
  exact -> `_norm` -> `_typo_norm` -> `_anchor_blocks`, splice exactly the matched lines, report `_not_found_hint`.
- `_reindent`: tab-aware, maps indent levels between the model's and the file's unit (unchanged when units equal).
- new `agent/textfile.py`: `decode(raw)` (BOM -> utf-8-sig/utf-16; strict utf-8; cp1252; latin-1), `encode` raising
  `UnencodableText` (nothing written), `write_atomic` (moved from `tools.atomic_write`); used by code tools, chat
  `tools/files.py` writers (now atomic) and undo. CRLF and BOM preserved.
- undo: deletions recorded (`FileChangeLog.undo_items()`), move = delete + create, CRLF-safe reads, restores deletes.
- one line splitter `edits.lines()` for read_file, show_symbol, numbered, outline, replace_lines.
- tests: patch ending in a blank line keeps the next line; latin-1/cp1252 byte-exact after edit; unencodable refused;
  tabs kept (Go, Makefile); undo of CRLF/delete/move; form-feed numbering consistent.

## Phase 5 – chat file + shell tools
- `tools/files.read_file`: regular files only, bounded read (`max_chars*4+4096` bytes), real size reported.
- `EXCLUDED_DIRS`: drop `"data"`, skip only the resolved agent data dir.
- new `agent/procrun.py` (extracted from `CodeTools._exec`): own process group, kill the group on timeout
  (`osport.kill_group`), output head 2.5k + tail 7k; used by `tools/shell.run_command`.
- tests: FIFO and `/dev/zero` refused; `data/` folders searchable; output tail kept; grandchildren killed.

## Phase 6 – GPU placement + helper calls (the user's CPU / reload problem)
- `llm.py`: replace `_foreground_models` by a protection API: `mark_foreground(models, num_ctx)`,
  `protect_roles(config)` (chat + `code.model`, ctx from `chat.helpers._num_ctx`), `is_protected(model)`.
  Called from `ollama_local.begin()`, `ui/chat_app._warm_up_model`, `CodeEngine.__init__`, `switch_chat_model`,
  web `deps.get_config`, `preload()`. `chat(background=True)`: a protected model always gets its own ctx and never
  `num_gpu`; another model gets `num_gpu 0` only if `/api/ps` (5 s cache) shows a different model holding VRAM on
  that server; probe failure never forces CPU; background calls never overwrite the remembered ctx.
  `codeagent/prose.py` switches to `is_protected`.
- self-repair: warm-up checks `/api/ps`; chat model with `size_vram == 0` on a local GPU box -> unload + preload once,
  notice "was on the CPU, reloaded on the GPU" / "still on the CPU: VRAM full"; same check in `nightage doctor`.
- `llm.utility_chat(...) -> (text, done_reason)` (always background, cancellable) for `_rewrite_search_query`,
  `_save_learned_facts`, distill, `worklog.summarize`, `plan.generate`; `_is_factual_correction` only calls a model
  when the background model is not the chat model, else uses the existing `_looks_like_disagreement` nudge.
- `speculative_options`: default 2 draft tokens (the code's own measurement), only for qwen3.5+, never gpt-oss;
  `preload` sends the same load options as chat. `codeagent/session.free_gpu` uses the tag-aware name match.
- tests (`tests/test_llm_vulkan.py` + new): protected model never `num_gpu 0`, even before the first chat; other
  model CPU only when VRAM is taken; failed probe never CPU; preload protects; warm-up repairs a CPU-only load;
  no draft tokens for gpt-oss (existing assertion 4 -> 2); memory catch-up after `protect_roles` sends no `num_gpu`.

## Phase 7 – stable prompt + idle memory work (reply latency)
- `memory/recall.build_context`: system prompt = frozen per-session head (persona, guidance, tools, env, facts
  snapshot) + "original task"/compaction summary that only change when the window slides. History = messages after
  the saved watermark (`store.get/set_session_compaction`); slides in one chunk to <= 60 % of the budget, never
  starting on a tool message. Recalled memories, skill hints, working notes and facts new since the snapshot go into
  the existing note before the newest user message (`_build_recall_note` slot).
- compaction off the reply path: `recall.after_turn()` on a thread right after the reply is saved (called from
  `ui/chat_adapter.run_chat_turn`, the classic tui loop, `web/routes/chat.py`); `build_context` only compacts inline
  above 1.25x budget.
- memory worker (`memory/queue.py`): no work before the first reply of a session; then only after `memory.idle_seconds`
  (default 30) without input; `busy(True)` cancels a running step (the cancel event closes the stream, the batch stays
  queued); after work on the chat model, **re-prime** the chat cache by replaying the last foreground payload with
  `num_predict 1` (new `llm.reprime(model)`, payload stored per model in `_stream_chat`/`_post_chat`) so the next reply
  starts without re-reading the conversation.
- truncated helper replies: `distill_batch` does not advance the mark on unparseable / `done_reason=="length"`
  (retry with half the batch, give up after 3); `worklog.summarize` rejects truncated output.
- tests: head byte-identical after a fact is added (fact in the end note); window fixed until a chunk slide; no model
  call inside `build_context` below 1.25x; worker idle-gated and cancelled by `busy(True)`; reprime payload equals the
  last chat payload except `num_predict`; unparseable reply keeps the mark.

## Phase 8 – context sizes, reasoning effort, errors
- `llm.configure()` keeps `models.overrides.<m>.num_ctx` + `max_ctx_ceiling` so every `cap_ctx` honours them;
  `_num_ctx` clamped to the model's native max (cached `/api/show`); `CodeEngine.num_ctx = pol.ctx or _num_ctx(...)`
  (same window as chat -> no reload when switching); history and read budgets from the same value; `num_predict`
  clamped to the room left in the window (`engine._call_model`, `chat/turn._model_call`); drop hard-coded 8192 in
  `web/routes/apps.py` and promptlab.
- `llm.think_value(model, want)`: qwen3.5+ bool as today; gpt-oss levels - background/helper calls `"low"`, chat and
  code main steps the new setting `models.reasoning_effort` (low|medium|high, default medium = today's behaviour,
  editable in `settings.EDITABLE`).
- `chat/turn.py`: at most 2 empty-reply retries; on `done_reason=="length"` one retry with a note, then a clear stop.
- errors: `OllamaStreamError` for `{"error"}` lines in `_stream_chat` and `chat()` (no more "refused (200)"); read the
  body before `raise_for_status`; Vulkan crash after success resets `_vulkan_ok`, one revive, then `VulkanServerDown`
  naming :11436 and the restart command; failed probe cached 30 s; footer shows the real route (`route_of`).
- tests: override honoured; chat == code window; gpt-oss levels; bounded retries; error texts; probe cache.

## Phase 9 – config + settings
- new `agent/configio.py`: `read_user(path)` (empty file = {}), `set_keys(path, {dotted: value})` – comment-keeping
  text edit (generalised from `modelpick.set_models`), else dump of the user's own dict, atomic write. All writers go
  through it: `settings.set_value`, `coding/models.set_model` (also writes `models.coding` -> `code-model` works),
  `wa_shared`, `wa_public`; nobody dumps `load_config()` again.
- `follow_chat` also moves `code.model` equal to the old chat model; `main._home()` reloads + resolves the config after
  Settings/doctor; `config.resolved()` resolves auto roles for web, memory search, prompt, reembed; `_mood` uses the
  session config.
- tests: comments + `auto` survive; empty file; `code-model` effective; profile overlay never saved; settings live.

## Phase 10 – coding agent verification
- `profile._python_runner(root)`: `uv.lock` -> `uv run --frozen --with pytest python -m pytest -q -x`; `.venv`/`venv`
  with pytest -> that interpreter; poetry/pdm when lock + tool exist; else today's command. ruff from `.venv`, PATH or
  `uv run`; configured-but-missing ruff -> no lint command + a project note.
- `_verify`: targeted tests for edited modules first (`xref.read_module`, `static._module_file`), then the full
  command, through a new `CodeTools.run_check` that reuses `run()` (protect snapshot, condensed output, traceback
  snippets); exit 127 = not run. `outputs.summarize_pytest` keeps the "short test summary info" block + final line.
- tests: runners for uv/venv/poetry; missing ruff; missing tool = not run; targeted first; summary tail kept.

## Phase 11 – coding agent context + progress
- `read_files` total capped at 2x `read_max_chars` (rest: "use read_file"); `plan.generate(model=)` gets the engine's
  model; aborted streams report unknown prompt tokens and `note_usage` keeps the estimate; progress = unseen content
  states or more passing checks (A->B->A no longer extends the budget), stage budget checked every step.
- tool set decided once per session: `tui_smoke` only if `tuismoke.repo_uses_tui`, `git_commit` only if the policy
  allows; shorten the longest descriptions (`run`, `tui_smoke`, `edit_file`) once – stable, smaller cached prompt.
- tests: read_files cap; A->B->A stops; plan uses the engine model; tool list stable across steps.

## Phase 12 – start-up
- cache `gpumode.detect_gpus()` (10 s, also for `governor`); one shared Ollama snapshot (`modelpick.installed` +
  2 s cache) for `doctor/legacy`, `ui/home`, `ollama_local`; `is_embedding` remembered by digest.
- `supports_tools` via `/api/show` on the default server, moved into the warm-up thread (no 90 s Vulkan start before
  the UI); warm-up also loads the embedder.
- GPU question after `begin()` + role resolution, chat row found by name, English text (`gpumode.py:301,390,436`,
  `tui._ask_more_steps`); home screen shows resolved roles.

## Phase 13 – memory store
- migration v17: indexes `messages(session_id,id)`, `memory_items(kind,ref_id)`, `facts(active,kind)`;
  `synchronous=NORMAL` with WAL (update `tests/test_store.py` version 16 -> 17).
- `VectorIndex.search` widens candidates (x4 … x64) until k hits of the wanted kinds; 32-entry embed cache so the
  user message is embedded once per turn.

## Phase 14 – web
- `config.db_path(config)` (resolved against the project root) for `web/deps.py`, `memory/background.py`,
  `web/routes/code_runs.py`, `main.py`, `ui/chat_app.py`; auto roles resolved in `deps.get_config`.
- web taint + evidence per turn via `contextvars` (security: sessions no longer reset each other's taint);
  blocking work in `routes/chat.py` moved to `asyncio.to_thread`.

## Status after round 1 (branch claude/hopeful-allen-7l6x9m, last commit 029d2cf)
Done + pushed: phases 0-11, 13, and from 12/14: GPU-detect cache, English GPU question, tools check off the main thread,
memory indexes / vector widening / embed cache, one db_path for web+CLI, per-turn web taint/evidence.
User test on the real machine (NixOS, gpt-oss:20b, dual GPU): 82-86 tok/s incl. a 3-tool-call turn (was 9), ctx 131k applied,
footer "Vulkan dual", second reply 1.2 s.

## Round 2 (next session)
Seen in the user's test run:
1. Home screen says "Context 44.4k tokens (vram)" while the GPU question then offers/chooses 131,072 (dual, q8_0): the home line
   (`agent/ui/home.py:99,187` via `doctor/probes.ctx_budget` -> `ctxbudget.explain`) ignores the dual option and the remembered
   choice. Show the remembered GPU mode/context (gpumode.last_mode / chat_ctx_override) or label it "one GPU estimate".
2. A tool call failed once on an enum: `web_search(freshness="30d")` -> "must be one of any/day/week/month/year"; the model
   retried. Normalise common aliases before validation (30d/1m->month, 7d/1w->week, 24h/1d->day, 365d/1y->year) where the
   argument validator lives (find "must be one of" under agent/chat or agent/tools), generic for every enum parameter.
3. The research answer cited "German Hospital Association, 2025 data" that no fetched source contained (an invented source).
   Consider a light check in the chat finish path: a "source:"/"according to X" claim whose X is not among this turn's evidence
   sources gets an "(not in the sources read)" note - reuse agent/research/evidence.take() and the code agent's claims-check idea.
Remaining from round 1: blocking work in agent/web/routes/chat.py -> asyncio.to_thread; one shared Ollama snapshot for home /
ollama_local / modelpick; GPU question after role resolution; home screen shows resolved roles; smaller findings
(read_file/run_command tool results capped for the model, config.yaml/greenlist write protection, quoted Windows paths in the
/auto scope check, code_admin stdout swap, /api/status double game check + nvidia-smi per poll).
Verification as before: full `uv run --frozen pytest -q` vs the baseline list, plus the user's checks (ollama ps 100% GPU, tok/s,
footer ctx, quick start after a 40 s pause).

## Round 3 – long prompts (pasted or as a .md file) — CURRENT
Context: the user wants to hand gpt-oss a ~25 kB multi-section spec ("Kawaii Osaka" master prompt) either pasted into the
chat or as a .md file, and have it read whole and followed.
Findings (read-only check):
- Paste works: the dock uses bracketed paste (`tests/test_ui_dock.py:148`), and `recall.build_context` always appends the
  user message whole (budget ctx x 1.6 chars; 131k ctx ~ 210k chars).
- Gap 1: `@file` only completes the path (`agent/ui/dock.py:221 _files`). The model sees the literal "@prompt.md" and must
  call read_file, whose default cap is 20,000 chars at <=16k ctx (`agent/tools/files.default_read_cap`), so a 25 kB spec is cut.
- Gap 2 (bug): `chat/gates.GoalReminder` appends the FULL original task as one more system message every 10 tool calls
  (`chat/turn.py:402`, `tui.py:1162`). With a 25 kB spec that adds +25 kB per 10 calls, accumulating, so the context overflows.
- Gap 3: `nightage code "task"` needs the spec as a shell argument (quoting 25 kB); there is no file input.
Changes:
1. New `agent/mentions.py`: `expand(text, cwd=None, limit=None) -> Expansion(text, files, notes)`.
   - It matches `@path` at the start or after whitespace (also `@"with spaces"`), expands `~`, and resolves relative to cwd.
   - Only existing regular text files are used (`agent/textfile.read` / `looks_binary`); everything else (emails, unknown
     names) stays untouched.
   - Each file is appended as `--- file: NAME (N chars) ---\n...\n--- end of NAME ---`. A message that is ONLY `@file`
     says "The user's message is the file below; follow it as their request."
   - Total limit: 60 % of `recall.history_char_budget(config)` (at least 20,000). Beyond it the head is kept with
     "cut at N of M chars - read the rest with read_file".
2. Wire it in:
   - `ui/chat_adapter.run_chat_turn` after `_split_web_trigger`. The expanded text is what gets stored, so later turns keep
     it; the title comes from the typed text; a log event "added NAME (N chars)".
   - The classic `tui` loop (~:1600).
   - `main.code` for the task argument (`nightage code @prompt.md`) and `ui/code_app._turn` for follow-ups.
   - The dock help text: "@file adds the file".
3. `GoalReminder`: restate at most 1,500 chars of the task (+ "... full task: the first user message"), and replace the
   earlier reminder instead of adding another (both loops).
4. Tests:
   - new `tests/test_mentions.py`: inline, only-mention, missing file and email untouched, binary refused, cap note,
     quoted path;
   - dock: bracketed paste of a real ~25 kB markdown spec gives one byte-identical submission;
   - `run_chat_turn` with a fake client: the last user message holds the whole file;
   - `GoalReminder` is capped and does not accumulate.
Verification: targeted tests, then the full suite vs the baseline; commit + push to claude/hopeful-allen-7l6x9m.
Note for the user: delete the trailing "CRITICAL: Respond with TEXT ONLY..." block from the spec. It is not part of it and
would stop gpt-oss from using tools. For a project this size, `nightage code --stages --plan-first @prompt.md` in an empty
folder fits better than the chat.

## Deferred (with reason)
Native tool-call turns for gpt-oss (needs eval runs on real models) · task-ranked repo map (prompt change, measure
first) · sqlite-vec kind partition (needs re-embedding) · `bash -c` instead of `bash -lc` (PATH/nvm risk) ·
low-impact web nits (stdout swap in code_admin, double game check in `/api/status`).

## Verification
- Every phase: targeted test files, then `uv run --frozen pytest -q`; failures compared to the baseline list
  (`/tmp/claude-0/base.txt`). Target after Phase 3: only the 19 env failures remain; no new failure in any phase.
- Security: ad-hoc script printing `classify_command` for every bypass string (all BLOCKED/DANGEROUS) next to the
  new test tables; patch/encoding fixes checked on real temp files (byte compare).
- Model pipeline: httpx `MockTransport` tests asserting the exact payloads (no `num_gpu` for protected models, equal
  `num_ctx` across preload/chat/background/reprime, gpt-oss `think` levels, draft options).
- Real hardware (user, after pulling): start `nightage`, then `ollama ps` and `OLLAMA_HOST=127.0.0.1:11436 ollama ps`
  -> gpt-oss "100% GPU"; a tool question runs at the normal ~85 tok/s; footer ctx matches the chosen 131072; pause
  40 s, send a message -> answer starts without a long "reading the prompt"; `nightage doctor` shows the placement.
- Each phase ends with a commit (attribution lines from the system reminder) and `git push -u origin
  claude/hopeful-allen-7l6x9m`.
