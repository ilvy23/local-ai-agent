# `nightage night` - the overnight coding run (private edition)

You give it a queue of prompts and a time budget (default 8 hours). It works through them with the local model while you sleep, **measures** everything itself
(tests, git, the event log) and never believes the model's own "done". It does not stop because the model is stuck: it escalates, rolls back, retries, and
at the end leaves a consistent project, a report, and every piece of discarded work under a git tag.

## The short version

```bash
nightage night doctor                      # can this machine run a night right now?
nightage night init prompts                # prompt templates + shared rules.md to fill in
nightage night plan prompts                # what WOULD happen: order, minutes per prompt, warnings (starts nothing)
nightage night run prompts --until 07:30   # go (asks once to confirm the command charter)
nightage night watch                       # live view from another terminal
nightage night stop                        # end gracefully; `pause` / `go` hold and continue
# in the morning
nightage night report                      # the measured report
nightage night nights                      # the nights this machine has run (history)
nightage night review                      # what changed, what was rolled back, how to keep or drop it
nightage night squash                      # all of it as ONE commit on top of the baseline
nightage night undo                        # or: back to the state before the night (the night stays under a tag)
```

## Prompts
A prompt is a `.md`/`.txt` file, a folder of them (natural order: `prompt2` before `prompt10`) or inline text. **Only one prompt is visible to the model at a time**;
the next one appears when the current one is done or its time slice is spent. `rules.md` in a folder is shared: it is put in front of every prompt. Optional front matter:

```
---
name: characters
weight: 2                # share of the night, relative to the others
verify: pytest tests/test_characters.py -q     # must also exit 0 for the prompt to count as done
depends: map             # waits for these prompts
kind: bug                # a failing test is written first (and must really fail), then the fix
plan: true               # a checklist plan is written first; the harness ticks its steps by measurement
---
```
`nightage night split big-prompt.md folder` cuts one big prompt (`Stage 1:` ... or `##` headings) into one chained prompt per part.
Before the start a prompt check warns about prompts no machine can verify (no test, no "must", a question to you, files that do not exist).

## What it checks (and what it never trusts)
| measured | what it decides |
|---|---|
| the project's own tests, per test, with a flaky re-run | done / red / "broke these tests that passed before" |
| the project's build command (lint is advice only) | a project that does not build is not done |
| `git diff` against the last green checkpoint | an attempt that changed nothing, how much it changed |
| guards | weakened / skipped / deleted tests and protected files (CI, lockfiles) are **reverted**; syntax errors; placeholder code (warning) |
| the prompt's `create \`file.py\`` | a named file that does not exist keeps the prompt open |
| the coding agent's event log | loops, reads without writes, model errors, tokens per second |

## What it does when something fails
The failure is classified (`server_down`, `timeout`, `stalled`, `crash`, `no_edits`, `loop`, `broke_tests`, `tests_red`, `incomplete`, `partial`) and the next strategy follows from
the kind: resume the same session, a fresh context with a **machine-written handoff** (what changed, which tests broke, what looped), a rollback to the last green checkpoint,
or small steps. After two equal failures a **tournament** runs 3 candidates from the same checkpoint and keeps the best by measurement. Which strategy worked is logged per night
(`.night/attempts.jsonl`) and, with enough evidence, overrules the default plan. A bug in the governor itself costs one prompt, never the night.

## Safety
* The **charter**: the commands the run may use are settled at the start (the project's own tooling + what the model proposes, validated) and confirmed by you once. In auto mode
  those run without asking **inside the project folder**; the safety classifier's BLOCKED tier, your deny patterns and anything outside the folder stay refused. Your previous policy
  of the folder is restored when the night ends.
* Git checkpoints only when the tests are green; nothing is ever lost (`night/red-N`, `night/undone-N`, `night/before-squash-N`).
* Stray processes an attempt leaves behind (a dev server) are stopped when it ends - only those of that attempt, never yours.
* The PC is kept awake for the whole night; the run waits while a game is running, the model server is down (it is restarted), the disk is nearly full or a GPU is hot.
* Nothing leaves the machine. Opt-in only: `NIGHT_NTFY_URL` (your own ntfy topic) gets one line when the night ends.

## Speed (measured on two GPUs, 16 GB + 8 GB, qwen3.8:27b)
* The biggest lever was `coding.stage_prompt_cap` (28000 -> 64000): the night sets it for its attempts only (`--stage-cap`), your config is not touched.
* 128k context runs at 100% on the GPUs and ~36 tok/s **when the graphics memory is free**; with a game or browser holding 1-3 GB it spilled to the CPU and fell to 15-21.
  `nightage night doctor` shows who holds VRAM. 192k and up spilled in every run.
* `nightage night tune` shows the two Ollama service settings that helped (`LLAMA_ARG_FIT_TARGET=64`, optional P-core pinning) and writes them as a separate drop-in only with `--apply`.

## Defaults in config.yaml
```yaml
night:
  hours: 8
  tournament: 3
  review: false
  map: auto            # off | auto | full
  stage_cap: 64000
  ctx: 0
  pause_on_game: true
```
Command-line options always win. Everything the night keeps is in `<project>/.night/` (git-ignored): `state.json`, `NIGHT_REPORT.md`, `attempts.jsonl`, `LESSONS.md`, `KNOWLEDGE.md`, `MAP.md`, `FACTS.md`, `plans/`, the attempt logs.

## Continuing anything, from anywhere (`nightage continue`)

One list of everything resumable on this machine: every saved coding session (all folders) and every overnight run.

    nightage continue            # numbered table, newest first (kind, id, folder, status, age, what); asks which one when run in a terminal
    nightage continue 2          # by number, or by id / id prefix: it continues IN ITS OWN FOLDER, no cd needed
    nightage continue --list     # only print, never ask
    nightage continue --prune    # remove entries whose folder or night state is gone (also coding sessions of deleted folders)
    nightage night runs          # the night runs only

(`nightage resume` is the chat command, so this one is called `continue`.) A coding session continues exactly like `nightage code --resume ID`.
A night continues like `nightage night run <its prompts> --folder <folder> --resume --yes` with the options it was started with, and the time that was left.

Where it reads from: coding sessions from `~/.local/state/agent/code-sessions/`; nights from the index `~/.local/state/agent/night-runs.jsonl` (one line per folder, written
when `night run` starts: folder, prompt files or inline text, options, times) plus a scan of every known folder for `.night/state.json`, so nights that started before the
index existed are found too (their prompts are rebuilt from the state into `.night/resume-prompts/`; the same happens when the saved prompt files have moved).
Ids of nights look like `n3fa91c2`, coding ids are the usual 8 hex characters.

Status: `running` (a live `.night/LOCK` pid, or a held session lock: continuing is refused, stop it first with `nightage night stop --folder DIR`), `interrupted`
(Ctrl+C, crash, reboot: continuable), `finished` / `partial (night over)` (the night ended: read `night report`, start a new one), `folder gone`, `state gone`.
Ctrl+C or a crash during a night saves the state and prints the `nightage continue ID` line instead of a traceback; a damaged `state.json` is reported, not crashed on.

## Unattended operation (the first real night)
```bash
nightage night install-unit PROMPTS --folder PROJECT --write --extra "--until 10:00 --no-canary --then EVAL_PROMPTS"   # writes a systemd user unit, enables nothing
systemctl --user start nightage-night-<project>.service         # go (it ends by itself); `enable --now` also starts it after a login
```
* The unit is `Type=notify` with a watchdog: the governor sends a heartbeat only while its main loop shows life (an attempt being polled, a gate wait, a test run). A governor that HANGS is
  restarted within about an hour; one that crashes (exit 70) is restarted after 60 s and resumed from its saved state; the remaining time comes from the wall clock (`end_wall`), not
  from the last saved value. Exit 3 (unfinished prompts) is a normal end, exit 1 (a refusal: another night running, bad folder) and 130 (stopped) are never restarted, and more than
  10 restarts in an hour stop the unit. Every stop is logged in `.night/unit-exits.log`.
* A prompt that runs out of its slice while it is making progress gets one grace stretch (12 min); a prompt that is still red afterwards is shelved as `.night/patches/<name>.patch` and
  the folder goes back to the last green checkpoint, so the next prompt starts green. Red tests at the very end get one closing repair round before the end-state rollback.
* `git` calls have a time limit; an attempt left behind by a hard-killed governor is stopped on resume (`.night/attempt.pid`); a game running on the PC pauses the night.

## Evidence (for evaluating a long run)
Written while the night runs, gathered at the end, and archived outside the project (`~/.local/state/agent/night-evidence/<project>-<time>.tar.gz`):
* `.night/evidence/<prompt>-attemptN.json`: per attempt the strategy and the failure kind that triggered it, the verdict, tests before and after (per test), guard findings, the diff per
  file, tool usage, steps, tokens per second, peak prompt, loops / blocked reads / forced thinking, the lessons and handoff it got, and WHAT THE MODEL SAID (final summary, `mark_done`
  evidence, saved notes); `.night/events/` holds the coding agent's raw event log of each attempt, `.night/<prompt>-attemptN.log` its console output.
* `.night/EVAL.json` / `EVAL.md` (`nightage night eval`): per prompt and for the night; `NIGHT_REPORT.md`; `attempts.jsonl`; `LESSONS.md`; `patches/`; the archive also holds `final.patch`
  (all changes against the baseline), `git-log.txt` and a copy of the prompts.

## Evaluation phase (`--then`)
When the main prompts are done and at least `--then-min-hours` (default 2) are left before the end time, the prompts of the `--then` folder run on a COPY of the result (`<project>-eval`), so
the main result stays intact. Their measurements do not rely on the model: front matter `prepare:` runs before the prompt (inject bugs, record a golden master) and `score:` after it
(the last JSON line it prints is stored). Tools in `agent/night/evaltools/`: `mutate` (inject N bugs that turn the tests red, then score how many were fixed and whether tests were edited),
`golden` (record the game's output for scripted sessions, then compare: behaviour preserved?) and `fuzz` (hostile input sessions: crashes and hangs). `.night/DONE` (which ends the
unit) is written only after the evaluation phase.
