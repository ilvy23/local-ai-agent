"""`agent code`: the interactive coding session and the one-shot runner around CodeEngine.

    agent code                      interactive session in the current project folder
    agent code "add a /health route"   one task, then exit
    agent code --resume [ID]        continue a saved session (the folder's latest, or the one with that id prefix)
    agent code sessions             every saved session of every folder (--here: this folder; --delete ID; --prune DAYS)
    agent code-policy ...           show / change what the agent may do in a repo (green repos)

Inside a session: /help lists the slash commands (mode, allow/deny, network, installs, commit, steps, think, model,
ctx, plan, undo, diff, status, map, context, compact, events, note, new, sessions, resume, delete).
Sessions are saved after every step under ~/.local/state/agent/code-sessions/ (see sessionstore.py) and survive a reboot.
"""

from __future__ import annotations

from agent import brand

import dataclasses
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agent import repoaccess
from agent.llm import READ_TIMEOUT_S, loaded_models_of, set_read_timeout, unload_model
from agent.codeagent import intake as intake_mod, policy as policy_mod, sessionstore, stages as stages_mod, timing, verified
from agent.codeagent.engine import CodeEngine, TurnResult
from agent.codeagent.eventlog import EventLog

HELP = """\
  /mode ask|medium|auto   how much I may do here without asking (remembered for this folder)
  /green                  same as /mode auto  - this repo is trusted
  /allow <pattern>        a command pattern that never asks (e.g. /allow pytest*)   /unallow <pattern>
  /deny <pattern>         a command pattern that never runs                         /undeny <pattern>
  /network on|off         may commands use the network?       /installs on|off   may they install packages?
  /commit never|ask|auto  may I git commit?                    /steps N   tool steps per prompt
  /think auto|on|off      reasoning on tool steps (auto = only when stuck)
  /model [name]  /ctx [N] model and context size for this session
  /policy                 everything I may do here             /plan [clear]   acceptance list
  /undo                   revert the last prompt's edits       /diff   /status   /map   /files
  /context                how full the window is               /compact   fold the history now
  /events                 what the engine did to its context and how long steps took
  /why-slow               where this session's time went (prefill, generation, tools, waiting) and the likely causes
  /stages [run]           the stages of a staged task (id, status, attempts, title); `run` continues the unfinished ones
  /state                  the verified state, measured now: tests, checks, static problems, open items, files
                          (no model; also `agent code state [--folder PATH]`, and added to a resumed session)
  /note <text>            save a fact about this project       /new   start fresh   /quit
  free text               a task for me"""


def state_dir() -> Path:
    return sessionstore.state_root()


_held_locks: list[Any] = []


def lock_file(root: Path) -> Path:
    return state_dir().parent / "code-locks" / (hashlib.sha1(str(root).encode()).hexdigest()[:16] + ".lock")


def acquire_lock(root: Path) -> tuple[bool, str]:
    """One coding session per folder: two agents editing the same files at once corrupt each other's work (that happened
    in a real test run). The lock is a kernel flock, so it disappears with the process - no stale locks."""
    from agent import osport

    f = lock_file(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    fh = open(f, "a+")
    if not osport.try_lock(fh):
        try:
            fh.seek(0)
            owner = fh.read().strip() or "unknown"
        except OSError:                                    # Windows: the lock holder's byte range cannot be read by us
            owner = "unknown"
        fh.close()
        return False, owner
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    _held_locks.append((str(root), fh))
    return True, ""


def holds_lock(root: Path) -> bool:
    return any(r == str(root) for r, _fh in _held_locks)


def find_root(start: Path) -> Path:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=start, capture_output=True, text=True)
    top = Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else start
    return top.resolve()


def save_session(engine: CodeEngine) -> None:
    """Runs before every model call and after every prompt: a killed run keeps its conversation. Atomic (temp file +
    rename), so a kill in the middle of the save leaves the previous save. A session without a request is not kept."""
    if not engine.ctx.user_requests:
        return
    if not engine.session_id:
        engine.session_id = sessionstore.new_id()
    try:
        sessionstore.save(sessionstore.snapshot(engine))
    except (OSError, TypeError, ValueError):
        pass


def reset_engine(engine: CodeEngine) -> None:
    """A fresh session in the same engine (the old one stays saved)."""
    engine.ctx.messages.clear()
    engine.ctx.user_requests.clear()
    engine.ctx.ledger.files.clear()
    engine.ctx.ledger.runs.clear()
    engine.ctx.step = 0
    engine.tools.read_cache.clear()                     # "unchanged since step N" would point into a history that is gone
    engine.ctx.last_prompt = engine.ctx.last_eval = engine.ctx._since = 0
    engine.ctx.mask_floor = engine.ctx.stage_mask_floor = 0.0
    engine.ctx.stage_mask_log, engine.ctx.stage_mask_thrash = [], False
    engine.plan.plan = None
    engine.stage_plan = engine.stage_goal = None
    engine.session_id, engine.session_created = sessionstore.new_id(), time.time()
    engine.last_stopped, engine.resume_note, engine.session_notes = "", None, []
    engine.dead_ends.clear()
    engine._interactive_hinted = False
    engine._started = False
    engine._head = None                                 # a new session gets a fresh repo map and project memory
    engine.start()


def resume_into(engine: CodeEngine, meta: sessionstore.Meta) -> dict[str, Any] | None:
    """Load a saved session into `engine`. None (and nothing changed) when the file is damaged."""
    try:
        data = sessionstore.load(meta.path)
    except ValueError as exc:
        engine.console.print(f"[yellow]can't resume {meta.id}: {exc}[/yellow]")
        return None
    engine.ctx.messages, engine.ctx.user_requests = [], []
    info = sessionstore.restore(engine, data)
    staged = engine.stage_plan is not None and engine.stage_plan.unfinished()      # begin_stage puts the VERIFIED STATE in: not twice
    if (getattr(engine, "config", None) or {}).get("coding", {}).get("verified_on_resume", True) and not staged:
        append_verified(engine)
    return info


def append_verified(engine: CodeEngine) -> None:
    """After a resume the model has forgotten the code base and re-reads it (60-89 reads in a row, seen live): measure the project now
    (verified.py, no model call) and add the block to the resume note, which the next turn sends once."""
    engine.console.print("[dim]computing verified state ...[/dim]", end="")
    try:
        st = verified.build(engine)
    except Exception as exc:  # noqa: BLE001 - the resume must work even when the measurement does not
        engine.console.print(f"[yellow] failed ({type(exc).__name__}: {escape(str(exc)[:80])})[/yellow]")
        return
    note = (engine.resume_note or "").replace(engine.dead_ends.text(), "").strip() if engine.dead_ends else (engine.resume_note or "")
    engine.resume_note = (note + "\n\n" if note else "") + st.text      # DEAD ENDS: once, inside the measured block
    engine.console.print(f"[dim] {st.seconds:.1f}s[/dim]")


def stage_mode(config: dict[str, Any]) -> str:
    """coding.stages: 'off'  | 'auto' (staged when the task splits into >= 2 stages, or is a 'ranges' plan: stages.range_plan) | 'on'."""
    mode = str((config.get("coding") or {}).get("stages", "auto")).strip().lower()
    return mode if mode in ("auto", "on") else "off"


def with_stages(config: dict[str, Any], on: bool) -> dict[str, Any]:
    """`agent code --stages`: sets coding.stages to 'on' for this run."""
    return {**config, "coding": {**(config.get("coding") or {}), "stages": "on"}} if on else config


def run_task_staged(engine: CodeEngine, task: str, *, mode: str | None = None, max_stages: int | None = None) -> stages_mod.StageReport | None:
    """Run `task` as stages in fresh small contexts (stages.py). None = not staged (mode off, or 'auto' and the task is one stage): the
    caller runs `engine.turn(task)` as before. A saved unfinished stage plan (--resume) is continued; `task` is then ignored.
    The entry the future supervisor calls."""
    mode = mode or stage_mode(engine.config)
    if mode == "off":
        return None
    if engine.stage_plan is not None and engine.stage_plan.unfinished():
        plan = engine.stage_plan
    else:
        plan = stages_mod.split(task, llm=engine.llm, cfg=engine.config, model=engine.model, cache=engine.split_cache, root=engine.root)
        if mode == "auto" and len(plan.stages) < 2 and plan.source != "ranges":
            return None
        stages_mod.announce(engine.console, plan)
        engine.stage_plan = plan
        engine.ctx.user_requests.append(task)               # the whole task is the session's first request (title, resume banner)
    if not engine.session_id:
        engine.session_id = sessionstore.new_id()
    return stages_mod.run_stages(engine, plan, save=lambda: save_session(engine), max_stages=max_stages)


def plan_first_gate(console: Console, engine: CodeEngine, task: str, cfg: dict[str, Any],
                    confirm: Callable[[], bool] | None = None) -> bool:
    """`agent code --plan-first`: build the stage plan `run_task_staged`/`Supervisor` would build anyway (same `stages.split`,
    same cache), show it (planview.py) and ask before anything runs. True = proceed (the plan is pinned to `engine.stage_plan`
    so the staged runner / supervisor reuse it rather than splitting again); False = declined, nothing was run."""
    from agent.codeagent import planview

    plan = stages_mod.split(task, llm=engine.llm, cfg=cfg, model=engine.model, cache=engine.split_cache, root=engine.root)
    view = planview.from_parts(plan, engine.plan.plan, len(engine.dead_ends.items), cfg)
    console.print(planview.table(view))
    proceed = confirm() if confirm is not None else console.input(
        "[bold]Proceed with this plan? [/bold][y/N] ").strip().lower() in ("y", "yes", "j", "ja")
    if not proceed:
        console.print("[yellow]Plan declined - nothing was run.[/yellow]")
        return False
    engine.stage_plan = plan
    engine.ctx.user_requests.append(task)          # the whole task is the session's first request (title, resume banner)
    return True


def with_autopilot(config: dict[str, Any]) -> dict[str, Any]:
    """`agent code --autopilot`: stages are always on (the supervisor drives the stage runner) and the model is not unloaded while it runs
    (the normal unload at the very end uses the ORIGINAL config)."""
    return {**config, "coding": {**(config.get("coding") or {}), "stages": "on"}, "ollama": {**(config.get("ollama") or {}), "unload_on_exit": False}}


def autopilot_mode(engine: CodeEngine, mode: str | None) -> str:
    """The policy mode of a run that nobody is there to answer: the user's `--mode` wins; else 'ask' (a new folder, nobody to ask - it
    would refuse everything) becomes 'medium'; any other saved mode stays. In memory only: nothing is written to the repo policy."""
    pol = engine.pol
    want = mode or (policy_mod.MEDIUM if pol.mode == policy_mod.ASK else pol.mode)
    if want != pol.mode:
        problem = repoaccess.folder_problem(engine.root) if want != policy_mod.ASK else None
        if problem:
            engine.console.print(f"[yellow]can't grant {want} here: {problem}[/yellow]")
            return pol.mode
        engine.pol = dataclasses.replace(pol, mode=want)
        engine.console.print(f"[dim]mode {want} for this run (not saved)[/dim]")
    return engine.pol.mode


def run_autopilot(engine: CodeEngine, task: str, *, mode: str | None = None) -> int:
    """`agent code --autopilot`: the supervisor (supervisor.py) runs the task in stages until done or it needs a human. The exit code of
    the process: 0 done, 2 needs a human, 3 partial, 130 stopped by the user. An empty `task` continues the saved stage plan (--resume)."""
    from agent.codeagent.supervisor import Supervisor

    if not task and not (engine.stage_plan is not None and engine.stage_plan.unfinished()):
        engine.console.print("[yellow]nothing to continue: this session has no unfinished stages (give the task)[/yellow]")
        return 1
    autopilot_mode(engine, mode)
    return Supervisor(engine, save=lambda: save_session(engine)).run(task).exit_code


def resume_banner(console: Console, engine: CodeEngine, info: dict[str, Any]) -> None:
    """What the user needs to pick up again: which chat, how old, how far it got, what was going on."""
    plan = engine.plan.plan
    state = "cut off mid-step (killed)" if info.get("state") == "in progress" else (
        f"stopped early ({info['stopped']})" if info.get("stopped") else "finished")
    console.print(f"[bold]resumed[/bold] {engine.session_id}  [cyan]{escape(info['title'])}[/cyan]  [dim]· {sessionstore.age_text(info['age'])} ago · "
                  f"{info['steps']} steps · {info['files']} files touched · last prompt {state}[/dim]")
    if info.get("stopped") or info.get("state") == "in progress":
        console.print(f"[dim]  in progress: {escape(_short(info.get('task') or '', 110))}   (say 'continue' to pick it up)[/dim]")
    if plan is not None:
        console.print(f"[dim]  acceptance list: {plan.confirmed()} of {len(plan.items)} confirmed[/dim]")
    for c in info.get("changes") or []:
        console.print(f"[yellow]  changed since: {c}[/yellow]")
    for r in info.get("repaired") or []:
        console.print(f"[dim]  repaired: {r}[/dim]")


def _short(text: str, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def sessions_table(metas: list[sessionstore.Meta], current: str = "", show_folder: bool = True) -> Table:
    home = str(Path.home())
    t = Table("id", *(["folder"] if show_folder else []), "title", "age", "steps", "state", box=None)
    now = time.time()
    for m in metas:
        folder = m.root.replace(home, "~", 1) if m.root.startswith(home) else m.root
        state = "this one" if m.id == current else m.shown_state() if Path(m.root).is_dir() else "folder gone"
        t.add_row(m.id + (" *" if m.id == current else ""), *([escape(_short(folder, 38))] if show_folder else []), escape(_short(m.title, 52)),
                  sessionstore.age_text(now - m.updated), str(m.steps), state)
    return t


def ask_access(console: Console, root: Path) -> None:
    """First time in a folder: ask how much freedom to give (and remember it)."""
    problem = repoaccess.folder_problem(root)
    if problem:
        console.print(f"[yellow]{root}: {problem}.[/yellow] I will ask before every edit and command here.")
        return
    console.print(f"[bold]New project folder:[/bold] {root}\nHow much may I do here without asking?\n"
                  "  [cyan]1[/cyan] ask      ask before every edit and command (safest)\n"
                  "  [cyan]2[/cyan] medium   edit files freely, ask before running commands\n"
                  "  [cyan]3[/cyan] auto     edit and run commands freely inside this folder (a \"green repo\")")
    try:
        ans = console.input("  choose [1/2/3, default 1] ▸ ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = "1"
    mode = {"2": "medium", "3": "auto"}.get(ans, "ask")
    pol = policy_mod.save_repo(root, mode=mode)
    console.print(f"[dim]saved: {pol.mode} mode for {root}  (change anytime with /mode, /allow, /network ...)[/dim]")


def open_engine(console: Console, config: dict[str, Any], root: Path, interactive: bool, test: str | None = None):
    from agent import projectmem
    from agent.main import _open_llm, _open_store
    from agent.tools.web import make_web_tools

    llm = _open_llm()
    apply_client_config(llm, config)
    store = _open_store(config).__enter__()
    notes = projectmem.ProjectMemory(store.conn)
    key = projectmem.project_key(root) or str(root)
    web = []
    try:
        web = make_web_tools(console)
    except Exception:  # noqa: BLE001 - the web tools are optional
        pass

    def ask(question: str) -> str:
        console.print(f"[yellow]?[/yellow] {question}")
        try:
            return console.input("  ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    engine = CodeEngine(root, config, llm, console, ask_fn=ask if interactive else None, web_tools=web, notes=notes, note_key=key,
                        events=EventLog(f"{root.name}-{int(time.time())}"), interactive=interactive)
    if test:
        engine.profile.test_cmd = test
    return engine, llm, store


def apply_client_config(llm: Any, config: dict[str, Any]) -> None:
    """coding.read_timeout_s (default 1800): how long the client waits for the first byte of a reply (a cold prefill)."""
    set_read_timeout(llm, (config.get("coding") or {}).get("read_timeout_s", READ_TIMEOUT_S))


def with_keep_model(config: dict[str, Any], keep: bool) -> dict[str, Any]:
    """`agent code --keep-model`: sets coding.keep_model_loaded for this run."""
    return {**config, "coding": {**(config.get("coding") or {}), "keep_model_loaded": True}} if keep else config


def release_gpu_on_exit(console: Console, llm: Any, config: dict[str, Any]) -> None:
    """Unload the models when the session ends (ollama.unload_on_exit) - unless coding.keep_model_loaded: then the next
    `agent code` finds the model warm instead of reloading it."""
    from agent.tui import _unload_models_on_exit

    if not (config.get("coding") or {}).get("keep_model_loaded"):
        _unload_models_on_exit(console, llm, config)


def free_gpu(console: Console, llm: Any, model: str) -> None:
    """Unload every other model so the coding model has the VRAM to itself."""
    try:
        base = model.split(":")[0]
        others = [m for m in loaded_models_of(llm) if m.split(":")[0] != base and "embed" not in m and "bge" not in m]
        for m in others:
            unload_model(llm, m)
        if others:
            console.print(f"[dim]freed VRAM: unloaded {', '.join(others)}[/dim]")
    except Exception:  # noqa: BLE001
        pass


def banner(console: Console, engine: CodeEngine, resumed: bool) -> None:
    pol = engine.pol
    console.print(f"[bold cyan]{brand.CMD} code[/bold cyan]  {engine.root}  [dim]· {engine.model} · {engine.num_ctx // 1024}k · {pol.mode} mode"
                  f"{' · network' if pol.network else ''}{' · installs' if pol.installs else ''}[/dim]")
    console.print(f"[dim]{engine.profile.text().splitlines()[0]}  · session {engine.session_id}{'' if resumed else ' (new)'}   (/help)[/dim]")


def show_events(console: Console, engine: CodeEngine) -> None:
    ev = engine.events.recent
    steps = [e for e in ev if e["kind"] == "step"]
    if not steps:
        console.print("[dim]no steps yet[/dim]")
        return
    pf = [e["prefill_s"] for e in steps]
    dc = [(e["gen"] / e["decode_s"]) for e in steps if e["decode_s"] > 0.2 and e["gen"] > 10]
    console.print(f"steps {len(steps)}   prefill avg {sum(pf) / len(pf):.1f}s (max {max(pf):.1f}s)   "
                  f"decode avg {sum(dc) / max(len(dc), 1):.0f} tok/s   last prompt {steps[-1]['prompt']:,} tokens".replace(",", "."))
    for kind in ("mask", "compact", "fold", "lookup", "ctx_backoff", "model_error", "degenerate", "loop", "collapse"):
        for e in [x for x in ev if x["kind"] == kind][-4:]:
            detail = {k: v for k, v in e.items() if k not in ("ts", "session", "kind")}
            console.print(f"[dim]{kind:8} {detail}[/dim]")


def why_slow_cmd(console: Console, engine: CodeEngine) -> None:
    """`/why-slow`: the measured time breakdown of this session (event log, timing.py) and up to 3 causes."""
    log = [e for e in timing.read_log(engine.events.path) if e.get("session") == engine.events.session] or list(engine.events.recent)
    console.print(timing.render(timing.breakdown(log)), markup=False, highlight=False, soft_wrap=True)


def handle_command(console: Console, engine: CodeEngine, line: str) -> bool:
    """Returns True when `line` was a slash command (handled)."""
    if not line.startswith("/"):
        return False
    cmd, _, arg = line[1:].partition(" ")
    cmd, arg = cmd.lower(), arg.strip()
    root = engine.root
    pol = engine.pol

    def persist(**kw: Any) -> None:
        engine.pol = policy_mod.save_repo(root, **kw)
        engine.tools.run_timeout = engine.tools.run_timeout
        console.print(f"[dim]saved for {root.name}[/dim]")

    if cmd in ("help", "?"):
        console.print(HELP)
    elif cmd == "stages":
        plan = engine.stage_plan
        if plan is None:
            console.print(f"[dim]no staged task in this session ({brand.CMD} code --stages, or coding.stages: auto)[/dim]")
        elif arg == "run":
            if plan.unfinished():
                run_task_staged(engine, plan.task, mode="on")
            else:
                console.print("[dim]every stage is finished[/dim]")
        else:
            console.print(f"stages ({plan.source}):\n" + "\n".join(plan.table()), markup=False, highlight=False)
    elif cmd == "state":
        console.print(verified.render_cli(verified.build(engine)), markup=False, highlight=False, soft_wrap=True)
    elif cmd in ("mode", "green"):
        mode = "auto" if cmd == "green" else arg.lower()
        problem = mode_problem(engine, mode)
        if problem:
            console.print(f"[yellow]{escape(problem)}[/yellow]")
        else:
            persist(mode=mode)
            console.print(f"mode {mode}")
    elif cmd in ("allow", "deny", "unallow", "undeny"):
        key = "allow" if cmd in ("allow", "unallow") else "deny"
        cur = list(getattr(pol, key))
        if not arg:
            console.print(f"{key}: {', '.join(cur) or '-'}")
        else:
            cur = [c for c in cur if c != arg] if cmd.startswith("un") else (cur if arg in cur else [*cur, arg])
            persist(**{key: cur})
    elif cmd in ("network", "installs"):
        flag = policy_mod.parse_flag(arg)
        if flag is None:
            console.print(f"{cmd} is {'on' if getattr(pol, cmd) else 'off'}  (/{cmd} on|off)")
        else:
            persist(**{cmd: flag})
    elif cmd == "commit":
        if arg in ("never", "ask", "auto"):
            persist(commit=arg)
        else:
            console.print(f"commit is {pol.commit}  (/commit never|ask|auto)")
    elif cmd == "steps":
        persist(max_steps=int(arg)) if arg.isdigit() else console.print(f"steps per prompt: {pol.max_steps}")
    elif cmd == "think":
        persist(think=arg) if arg in ("auto", "on", "off") else console.print(f"think is {pol.think}  (/think auto|on|off)")
    elif cmd == "model":
        if arg:
            engine.model = arg
            persist(model=arg)
        console.print(f"model {engine.model}")
    elif cmd == "ctx":
        if arg.isdigit():
            n = max(4096, int(arg) * (1024 if int(arg) < 1024 else 1))
            engine.num_ctx = engine.ctx.num_ctx = n
            persist(ctx=n)
        console.print(f"context {engine.num_ctx:,} tokens".replace(",", "."))
    elif cmd == "policy":
        console.print(policy_mod.describe(root))
    elif cmd == "plan":
        holder = engine.plan
        if arg == "clear":
            holder.plan = None
            console.print("plan cleared")
        elif holder.plan is None:
            console.print("[dim]no acceptance list (big tasks get one automatically)[/dim]")
        else:
            console.print("\n".join(holder.plan.lines()))
    elif cmd == "undo":
        for ln in engine.undo.undo():
            console.print(f"  {ln}")
    elif cmd == "diff":
        console.print(engine.tools.git_diff())
    elif cmd == "status":
        console.print(engine.tools.git_status())
        console.print(f"[dim]context {engine.ctx.estimate():,} / {engine.num_ctx:,} tokens ({engine.ctx.fill() * 100:.0f}%)[/dim]".replace(",", "."))
    elif cmd == "map":
        console.print(engine.tools.repo_map())
    elif cmd == "files":
        led = engine.ctx.ledger
        console.print("\n".join(f"  {p}  {e.lines} lines  ({e.action} at step {e.step})" for p, e in sorted(led.files.items())) or "no files touched yet")
    elif cmd == "context":
        c = engine.ctx
        sizes = {}
        for m in c.messages:
            sizes[m["role"]] = sizes.get(m["role"], 0) + len(str(m.get("content", "")))
        console.print(f"{c.estimate():,} of {engine.num_ctx:,} tokens ({c.fill() * 100:.0f}%)   messages {len(c.messages)}   ".replace(",", ".")
                      + "   ".join(f"{r} {n // 1000}k chars" for r, n in sizes.items()))
        from agent.codeagent import context as ctx_mod

        console.print(f"[dim]mask at {ctx_mod.MASK_AT:.0%}, compact at {ctx_mod.COMPACT_AT:.0%}, history fold at {ctx_mod.FOLD_AT:.0%} of the window[/dim]")
    elif cmd == "compact":
        note = engine.ctx.compact()
        console.print(note or "nothing to compact yet")
    elif cmd == "events":
        show_events(console, engine)
    elif cmd == "why-slow":
        why_slow_cmd(console, engine)
    elif cmd == "note":
        if engine.notes is not None and arg and engine.notes.add(engine.note_key, arg):
            console.print("[dim]saved[/dim]")
        else:
            console.print("[dim]not saved[/dim]")
    elif cmd in ("new", "clear"):
        save_session(engine)
        old = engine.session_id if engine.ctx.user_requests else ""
        reset_engine(engine)
        console.print(f"[dim]fresh session {engine.session_id}" + (f" (the last one stays saved: /resume {old})" if old else "") + "[/dim]")
    elif cmd == "sessions":
        metas = sessionstore.list_sessions(root)
        console.print(sessions_table(metas, engine.session_id, show_folder=False) if metas else "[dim]no saved sessions in this folder yet[/dim]")
        console.print(f"[dim]/resume <id> switches · other folders: {brand.CMD} code sessions[/dim]")
    elif cmd == "resume":
        meta, why = sessionstore.find(arg, root) if arg else (sessionstore.latest(root), "")
        if meta is None:
            console.print(f"[yellow]{why or 'no saved session in this folder'}[/yellow]")
        elif meta.id == engine.session_id:
            console.print("[dim]that is this session[/dim]")
        elif Path(meta.root) != root:
            console.print(f"[yellow]{meta.id} belongs to {meta.root}[/yellow] - one folder per process: run `{brand.CMD} code --resume {meta.id}` there")
        else:
            save_session(engine)
            info = resume_into(engine, meta)
            if info is not None:
                resume_banner(console, engine, info)
    elif cmd == "delete":
        meta, why = sessionstore.find(arg, root) if arg else (None, "give the id: /delete <id> (see /sessions)")
        if meta is None:
            console.print(f"[yellow]{why}[/yellow]")
        elif meta.id == engine.session_id:
            console.print("[yellow]that is the session you are in - /new first, then delete it[/yellow]")
        else:
            console.print(f"deleted {meta.id}  {meta.title}" if sessionstore.delete(meta) else f"[yellow]could not delete {meta.id}[/yellow]")
    else:
        reply = repoaccess.handle_command(line)
        if reply:
            console.print(reply[1])
            engine.pol = policy_mod.for_repo(root)
        else:
            console.print(f"[yellow]unknown command /{cmd}[/yellow]  (/help)")
    return True


def pick_session(console: Console, root: Path, resume: str | None, new: bool) -> sessionstore.Meta | None:
    """Which saved session to continue: `--resume ID`, `--resume` (the folder's latest, any age), or by default the
    folder's latest when it is less than 12 h old (the old behaviour). `--new` always starts a fresh one."""
    if new:
        return None
    if resume and resume != "latest":
        meta, why = sessionstore.find(resume, root)
        if meta is None:
            console.print(f"[yellow]{why}[/yellow]")
        return meta
    warn: list[str] = []
    sessionstore.list_sessions(root, warn)
    for w in warn:
        console.print(f"[yellow]{w}[/yellow]")
    meta = sessionstore.latest(root, None if resume == "latest" else sessionstore.DEFAULT_RESUME_H)
    if meta is None and resume == "latest":
        console.print("[dim]no saved session in this folder yet - starting a new one[/dim]")
    return meta


def persist_policy(engine: CodeEngine, **changes: Any) -> None:
    """Save per-folder policy changes and make the engine use them (what /mode, /allow, /network ... do). No output."""
    engine.pol = policy_mod.save_repo(engine.root, **changes)


def mode_problem(engine: CodeEngine, mode: str) -> str | None:
    """Why `mode` can not be set in this folder (unknown name, or the folder is not safe for it); None = fine."""
    if mode not in policy_mod.MODES:
        return f"mode is one of: {', '.join(policy_mod.MODES)}  (now {engine.pol.mode})"
    problem = repoaccess.folder_problem(engine.root) if mode != "ask" else None
    return f"can't grant that here: {problem}" if problem else None


def resolve_target(console: Console, folder: str | None, resume: str | None) -> tuple[sessionstore.Meta | None, Path | None]:
    """Which saved session (`--resume ID`) and which project root a run starts in. root None = refused, the reason is printed.
    A session resumes in its own folder."""
    chosen: sessionstore.Meta | None = None
    if resume and resume != "latest":
        here = Path(folder).expanduser() if folder else Path.cwd()
        chosen, why = sessionstore.find(resume, find_root(here) if here.is_dir() else None)
        if chosen is None:
            console.print(f"[red]{why}[/red]")
            return None, None
        if not Path(chosen.root).is_dir():
            console.print(f"[red]session {chosen.id} belongs to {chosen.root}, which no longer exists.[/red] It stays saved "
                          f"(`{brand.CMD} code sessions --delete {chosen.id}` removes it).")
            return None, None
        folder = chosen.root                                   # a session resumes in its own folder
    root = find_root(Path(folder).expanduser() if folder else Path.cwd())
    if root == Path.home() or len(root.parts) < 3:
        console.print("[red]Run this inside a project folder, not your home or a top-level folder.[/red]")
        return None, None
    return chosen, root


def prepare_engine(console: Console, engine: CodeEngine, root: Path, chosen: sessionstore.Meta | None, resume: str | None, new: bool,
                   *, mode: str | None = None, autopilot: bool = False, steps: int | None = None,
                   banner_fn: Callable[[dict[str, Any] | None], None] | None = None) -> dict[str, Any] | None:
    """The start of a session once the engine exists: resume the chosen / latest saved session or begin a new one, name the event
    session, hook the per-step save, show the banner (`banner_fn(info)`), then apply this run's --mode / --steps (not saved).
    Returns the resume info (None = a new session)."""
    meta = chosen or pick_session(console, root, resume, new)
    info = resume_into(engine, meta) if meta is not None else None
    if info is None:
        engine.session_id = sessionstore.new_id()
        engine.start()
    engine.events.session = f"{root.name}-{engine.session_id}"
    engine.on_step = lambda: save_session(engine)
    if banner_fn is not None:
        banner_fn(info)
    if mode and not autopilot:
        autopilot_mode(engine, mode)
    if steps:
        engine.pol = dataclasses.replace(engine.pol, max_steps=steps, steps_set=True)     # this run only (not saved)
    return info


def run_session(folder: str | None = None, goal: str | None = None, new: bool = False, files: list[str] | None = None,
                test: str | None = None, resume: str | None = None, keep_model: bool = False, stages: bool = False,
                autopilot: bool = False, mode: str | None = None, steps: int | None = None, plan_first: bool = False,
                confirm: Callable[[], bool] | None = None) -> int:
    """`plan_first` (agent code --plan-first): before a fresh goal executes, build its stage plan (stages.split, the same split a
    staged/autopilot run would use) and show it (planview.py) instead of starting at once; `confirm` decides whether to proceed
    (default: ask on the console), tests inject a stub. Declining runs nothing. Forces `stages` on (there is nothing to show
    otherwise) and, once confirmed, hands the SAME plan to the existing stage runner / supervisor - no second execution path."""
    from agent import foreground
    from agent.codeagent.supervisor import should_autopilot
    from agent.config import load_config

    stages = stages or plan_first
    console = Console()
    config = with_stages(with_keep_model(load_config(), keep_model), stages)
    if goal is None and autopilot and resume:
        goal = ""                                                  # `--autopilot --resume`: no new task, continue the saved stage plan
    autopilot = goal is not None and (autopilot or should_autopilot(goal, config))      # never in the REPL
    chosen, root = resolve_target(console, folder, resume)
    if root is None:
        return 1
    autopilot = autopilot or (goal is not None and should_autopilot(goal, config, root))       # a short task that names a big file needs the root
    run_config = with_autopilot(config) if autopilot else config
    interactive = goal is None
    ok, owner = acquire_lock(root)
    if not ok:
        console.print(f"[red]Another coding session is already working in {root} (pid {owner}).[/red] Two agents in one folder overwrite "
                      "each other - finish or stop that one first (`kill " + owner + "`).")
        return 1
    known = policy_mod.all_repos()
    if str(root) not in known and policy_mod.for_repo(root).mode == "ask" and interactive and not mode:
        ask_access(console, root)
    with foreground.session(config):
        engine, llm, store = open_engine(console, run_config, root, interactive, test)
        try:
            free_gpu(console, llm, engine.model)
            def show_banner(info: dict[str, Any] | None) -> None:
                banner(console, engine, info is not None)
                if info is not None:
                    resume_banner(console, engine, info)

            info = prepare_engine(console, engine, root, chosen, resume, new, mode=mode, autopilot=autopilot, steps=steps, banner_fn=show_banner)
            if goal is not None:
                task = goal if not files or not goal else f"{goal}\n\nFocus on these files: {', '.join(files)}."
                if info is None:
                    task = intake_mod.run(engine, task)
                    if plan_first and not plan_first_gate(console, engine, task, run_config, confirm):
                        return 1
                if autopilot:
                    return run_autopilot(engine, task, mode=mode)
                report = run_task_staged(engine, task)
                if report is not None:
                    return 0 if report.verdict == "done" else 1
                res = engine.turn(task)
                save_session(engine)
                return 0 if res.stopped == "" and res.text is not None else 1
            while True:
                try:
                    from agent.textio import read_multiline

                    line = read_multiline(console, "[bold green]code>[/bold green] ")     # a pasted multi-line task stays whole
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break
                if not line:
                    continue
                if line in ("/quit", "/exit", "/q"):
                    break
                if handle_command(console, engine, line):
                    continue
                big = not line.startswith("/") and should_autopilot(line, config, root)          # judged on what the user typed, not on the answers added below
                text = intake_mod.run(engine, line) if not line.startswith("/") else line
                if big:                                                  # a big prompt runs under the supervisor here too; the mode stays the user's
                    run_autopilot(engine, text, mode=engine.pol.mode)
                    save_session(engine)
                    continue
                if stage_mode(config) != "off" and not line.startswith("/") and run_task_staged(engine, text) is not None:
                    continue
                engine.turn(text)
                save_session(engine)
        finally:
            save_session(engine)
            release_gpu_on_exit(console, llm, config)
    return 0


class _NoModel:
    """The engine of a read-only command needs an llm object but must never call one."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"no model in this command (tried {name})")


def state_cli(folder: str | None = None) -> int:
    """`agent code state [--folder PATH]`: the verified state of a folder, with its latest saved session if there is one. No model, no
    lock (it only reads, apart from running the project's own tests and checks)."""
    from agent.config import load_config

    console = Console()
    start = Path(folder).expanduser() if folder else Path.cwd()
    if not start.is_dir():
        console.print(f"[red]{start} is not a folder[/red]")
        return 1
    root = find_root(start)
    engine = CodeEngine(root, load_config(), _NoModel(), console, interactive=False, events=EventLog(f"{root.name}-state-{int(time.time())}"))
    meta = sessionstore.latest(root)
    if meta is not None:
        try:
            sessionstore.restore(engine, sessionstore.load(meta.path))
        except (ValueError, OSError, KeyError, TypeError):
            console.print(f"[yellow]the saved session {meta.id} is damaged; showing the folder without it[/yellow]")
    console.print(verified.render_cli(verified.build(engine)), markup=False, highlight=False, soft_wrap=True)
    return 0


def sessions_cli(here: bool = False, folder: str | None = None, delete: str | None = None, prune: float | None = None) -> int:
    """`agent code sessions`: every saved session of every folder, newest first (--here / --folder: one folder)."""
    console = Console()
    if delete:
        meta, why = sessionstore.find(delete)
        if meta is None:
            console.print(f"[red]{why}[/red]")
            return 1
        if meta.shown_state() == "in progress":
            console.print(f"[red]{meta.id} is running right now in {meta.root}[/red]")
            return 1
        console.print(f"deleted {meta.id}  {meta.title}" if sessionstore.delete(meta) else f"[red]could not delete {meta.id}[/red]")
        return 0
    if prune is not None:
        gone = sessionstore.prune(prune)
        console.print(f"deleted {len(gone)} session(s) untouched for more than {prune:g} days")
        return 0
    root = find_root(Path(folder).expanduser() if folder else Path.cwd()) if (here or folder) else None
    warn: list[str] = []
    metas = sessionstore.list_sessions(root, warn)
    for w in warn:
        console.print(f"[yellow]{w}[/yellow]")
    if not metas:
        console.print("[dim]no saved coding sessions" + (f" for {root}" if root else "") + " yet[/dim]")
        return 0
    console.print(sessions_table(metas, show_folder=root is None))
    console.print(f"[dim]resume: {brand.CMD} code --resume <id>   (it opens in its own folder)[/dim]")
    return 0


# ── agent code-policy ───────────────────────────────────────────────────────

def policy_cli(args: list[str]) -> int:
    console = Console()
    if not args or args[0] == "list":
        repos = policy_mod.all_repos()
        t = Table("repo", "mode", "commit", "network", "installs", "allow", "deny", "steps", box=None)
        for path, p in sorted(repos.items()):
            t.add_row(path, p.mode, p.commit, "on" if p.network else "of", "on" if p.installs else "off", ", ".join(p.allow) or "-",
                      ", ".join(p.deny) or "-", str(p.max_steps))
        console.print(t if repos else f"[dim]no repos configured yet - run `{brand.CMD} code` in a project, or: {brand.CMD} code-policy set <dir> mode=auto[/dim]")
        gl = repoaccess.greenlist()
        if gl:
            console.print("[dim]also fully trusted via the chat /greenlist: " + ", ".join(str(g) for g in gl) + "[/dim]")
        return 0
    if args[0] == "show" and len(args) > 1:
        console.print(policy_mod.describe(Path(args[1]).expanduser()))
        return 0
    if args[0] == "forget" and len(args) > 1:
        console.print("removed" if policy_mod.forget_repo(Path(args[1]).expanduser()) else "not configured")
        return 0
    if args[0] == "set" and len(args) > 2:
        folder = Path(args[1]).expanduser()
        fixed, note = repoaccess.resolve_folder(str(folder))
        if fixed is None:
            console.print(f"[red]{note}[/red]")
            return 1
        changes: dict[str, Any] = {}
        for kv in args[2:]:
            k, _, v = kv.partition("=")
            if k in ("network", "installs"):
                flag = policy_mod.parse_flag(v)
                if flag is None:
                    console.print(f"[red]{k} must be on/off[/red]")
                    return 1
                changes[k] = flag
            elif k in ("allow", "deny"):
                changes[k] = [x for x in v.split(",") if x]
            elif k in ("max_steps", "ctx"):
                changes[k] = int(v)
            elif k in ("mode", "commit", "think", "model", "test_cmd"):
                changes[k] = v
            else:
                console.print(f"[red]unknown setting {k}[/red]")
                return 1
        console.print(policy_mod.describe(fixed) if policy_mod.save_repo(fixed, **changes) else "")
        return 0
    console.print(f"usage: {brand.CMD} code-policy [list | show <dir> | forget <dir> | set <dir> mode=auto network=on allow=pytest*,make* ...]")
    return 1
