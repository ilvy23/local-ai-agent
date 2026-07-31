"""Greenfield builds: generate the whole program in ONE shot, then repair.

A local 7B loses coherence when forced to build incrementally through many small
tool calls — it writes a stub, forgets to fill it in, thrashes for 30 steps. But
it writes a complete, correct file in a SINGLE generation (measured: a clean,
runnable 73-line snake game in 10s). So a from-scratch goal plays to that: ask
for every file at once, apply, verify, and loop only to FIX whole files against
real check output — the Aider pattern, which is why it works with weak models
where tool-loop agents don't.
"""

from __future__ import annotations

import httpx

from agent.coding import config as coding_config
from agent.coding.agent_loop import _usage_note
from agent.coding.edit.apply import apply_edits
from agent.coding.edit.format import parse_edits
from agent.coding.session import Outcome, _diagnose, _generate
from agent.coding.verify import guards
from agent.coding.verify.checks import reset_install_budget, run_ladder
from agent.coding.workspace import Workspace
from agent.events import NULL, Events

_EFFORT = {
    "min": "SCOPE: MINIMAL. Exactly ONE program file + ONE small test file. The fewest "
           "lines that fully work — no extra features, no config, no niceties. Test only "
           "the one or two most important behaviours.",
    "mid": "SCOPE: BALANCED. ONE program file + ONE thorough test file. Implement the core "
           "behaviour well and test it properly; skip nice-to-haves.",
    "max": "SCOPE: THOROUGH. You MAY split into a few files if it genuinely helps (e.g. a "
           "logic module, the entry point, and tests). Handle edge cases and cover them "
           "with tests. Still keep every file COMPLETE and imports consistent.",
}

_GEN = """You are an expert engineer. Build this, complete and working:

{goal}

{effort_clause}

Output EVERY file needed. Each file MUST use this exact format:

### FILE: path/to/file.py
```python
<the complete file contents>
```

Rules:
- Write COMPLETE, runnable files — no placeholders, no TODO, no `...`.
- Follow the SCOPE above for how many files. Default to the program file (e.g.
  `app.py`) + ONE test file (`test_app.py`) with all classes/constants in the one
  program file; only split further when SCOPE says THOROUGH. The test imports what
  it needs from the program: `from app import thing, OTHER`.
- Separate the core LOGIC (the rules/computation/state) from the I/O (printing,
  GUI, network, file, keyboard). Put the logic in plain functions/classes that
  need no screen/socket/input, so tests can call them directly. Any entry point
  (`main()` under `if __name__ == "__main__":`, a CLI, a render loop) just wires
  that logic to the outside world.
- The test file must IMPORT every name it uses from the program module — never
  reference a constant, class, or function without importing it first. Missing
  imports are the most common failure; check each name you use.
- The test must PROVE the behaviour with concrete assertions that would FAIL if
  the logic were subtly wrong. Assert exact values/state, never just `True` or
  that something "runs". Cover the smallest/first case (empty input, a single
  element, the first step) and the boundaries the goal implies.
- When you test something that happens partway through a step/update, respect the
  ORDER of operations. Trace what the code actually does in sequence and set up the
  state so the event you're testing genuinely fires — don't assume a check happens
  before the mutation that precedes it.
- Use normal imports; third-party packages are installed for you automatically.
- Stay INSIDE the project directory. Never write, copy, install, or move files to
  system locations — no /usr/local/bin, /usr/bin, /etc, ~/.local/bin, PATH dirs,
  cron, autostart, or service files. "Portable / works on other systems" means
  writing code that RUNS anywhere (relative paths, standard library), NOT installing
  anything system-wide. Installing a global command is a step the USER does, not you.

Output the files now, nothing else."""

_FIX = """You are fixing a project so its automated checks pass. The goal was:

{goal}

Current files:
{files}

The checks FAILED:
{error}
{diagnosis}
You own BOTH the code and the tests here. Fix whichever is actually WRONG:
- If the implementation is wrong, fix it.
- If a TEST's setup makes its expectation impossible (wrong assumptions about the
  order of operations, the state, or the expected values), correct the TEST. Never
  weaken a correct test just to pass — the change must keep proving the behaviour.

Rewrite whatever files are needed. Output COMPLETE files in the same
`### FILE:` + ```python``` format. Never leave anything out or use placeholders."""


def _dump_files(ws, paths) -> str:
    blocks = []
    for p in sorted(paths):
        try:
            blocks.append(f"### FILE: {p}\n```python\n{ws.read(p)}\n```")
        except OSError:
            pass
    return "\n\n".join(blocks)


def _gen(llm, cfg, prompt, events):
    """One generation, with graceful context backoff if the runner OOM-crashes."""
    ctx = cfg["executor"]["num_ctx"]
    while True:
        try:
            return _generate(llm, cfg, [{"role": "user", "content": prompt}], events, None)
        except httpx.HTTPStatusError as exc:
            if getattr(exc.response, "status_code", 0) == 500 and ctx > 8192:
                ctx = max(8192, ctx // 2)
                cfg = {**cfg, "executor": {**cfg["executor"], "num_ctx": ctx}}
                events.emit("ctx_backoff", num_ctx=ctx)
                continue
            raise


def build(repo_root, goal: str, config: dict | None = None, llm=None,
          events: Events = NULL, max_rounds: int = 8) -> Outcome:
    cfg = coding_config.load(config)
    if llm is None:
        from agent.llm import OllamaClient
        llm = OllamaClient()
    reset_install_budget()

    seen: set[str] = set()
    error: str | None = None

    with Workspace(repo_root) as ws:
        for rnd in range(max_rounds):
            phase = "generate" if rnd == 0 else "repair"
            events.emit("build_round", n=rnd + 1, of=max_rounds, phase=phase)
            if rnd == 0:
                effort = cfg.get("build", {}).get("effort", "mid")
                prompt = _GEN.format(goal=goal, effort_clause=_EFFORT.get(effort, _EFFORT["mid"]))
            else:
                # Diagnose the failure first (separate reasoning pass) so the model
                # fixes the root cause instead of thrashing. Works for any bug —
                # game logic, CLI arg parsing, off-by-one, whatever the check caught.
                diag = ""
                if cfg.get("diagnose_first", True) and seen:
                    d = _diagnose(llm, cfg, goal, sorted(seen), ws, error or "", events,
                                  seed=rnd, tests_editable=True)
                    diag = f"\nRoot-cause diagnosis (implement this fix):\n{d}\n" if d else ""
                prompt = _FIX.format(goal=goal, files=_dump_files(ws, seen),
                                     error=error or "", diagnosis=diag)
            try:
                raw = _gen(llm, cfg, prompt, events)
            except httpx.HTTPError as exc:
                # Persistent server error even after context backoff — almost always
                # the model doesn't fit in VRAM (e.g. the 30B on an 8GB GPU OOMs).
                # Fail cleanly with actionable advice instead of a raw traceback.
                diff = ws.diff()
                if diff and cfg.get("auto_commit", True):
                    ws.commit(f"agent code WIP: {goal[:50]}")
                    ws.keep = True
                events.emit("finished", ok=False)
                return Outcome(
                    ok=False, stage="model-error", diff=diff,
                    branch=ws.branch if ws.keep else "", attempts=rnd + 1,
                    error=f"Ollama error ({type(exc).__name__}): the model likely "
                          "doesn't fit in your GPU's VRAM and ran out of memory. "
                          "Swap to a smaller model that fits (menu ▸ 'Swap the coding "
                          "model', pick the 7B) — it runs fully on the GPU and won't OOM.")

            default = next(iter(seen)) if len(seen) == 1 else "main.py"
            parsed = parse_edits(raw, default_path=default)
            if not parsed.edits:
                error = ("No files parsed. Output each file as '### FILE: name.py' "
                         "followed by a ```python code block.")
                continue

            changed, errs = apply_edits(ws, parsed.edits)
            seen.update(changed)
            for c in changed:
                events.emit("edit", path=c)
            if errs:
                error = "; ".join(errs)
                continue

            res = run_ladder(ws.path, [ws.path / c for c in changed], events=events)
            if not res.ok:
                error = f"{res.stage}: {res.error}"
                continue

            # Passed the ladder. Final guards: coverage (unless a smoke-run — a
            # game has no meaningful coverage) then the reviewer.
            g = cfg.get("guards", {})
            if g.get("coverage_gate", True) and "smoke:" not in (res.note or ""):
                cov = guards.coverage_gate(ws, list(seen))
                if not cov.ok:
                    error = f"coverage: {cov.error}"
                    continue
            if g.get("llm_judge", True):
                verdict = guards.judge_diff(llm, cfg, goal, ws.diff())
                if not verdict.ok:
                    error = f"reviewer flagged the code as gaming the check: {verdict.reason}"
                    continue

            diff = ws.diff()
            usage = _usage_note(llm, cfg["executor"]["model"], diff)
            if cfg.get("auto_commit", True):
                ws.commit(f"agent code: {goal[:60]}")
                ws.keep = True
            events.emit("finished", ok=True, usage=usage)
            return Outcome(ok=True, stage="built", diff=diff,
                           branch=ws.branch if ws.keep else "", attempts=rnd + 1)

        diff = ws.diff()
        if diff and cfg.get("auto_commit", True):
            ws.commit(f"agent code WIP: {goal[:50]}")
            ws.keep = True
        events.emit("finished", ok=False)
        return Outcome(ok=False, stage="stuck", diff=diff,
                       branch=ws.branch if ws.keep else "",
                       error=error or "did not converge in the round budget",
                       attempts=max_rounds)
