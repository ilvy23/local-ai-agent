"""Slash commands of the terminal code UI, bound to the engine.

`build_registry()` is the command table of code mode: the shared REGISTRY commands that make sense while coding plus the classic code-REPL
commands (agent/codeagent/session.py HELP) with their `when` (instant / step / deferred) set for "a run is going on right now".
`make_handlers(ctx)` returns {name: fn(args, app)} for NightageApp. `ctx` is the CodeSession (code_app.py): it only needs `.engine`,
`.runner` (CodeRunner), `.app` and `.start_prompt(text)`.

Reuse, not a copy: every classic command goes through `session.handle_command` with a capturing console, so /mode, /allow, /network, /undo,
/resume ... do exactly what they do in `agent code`; the output comes back as text for the renderer. Nothing here reads stdin.
`CodeControl` is the NightageApp `control` object (step-boundary commands while a run is active).
"""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Callable
from typing import Any

from rich.console import Console

from agent.ui import commands as C
from agent.ui.commands import REGISTRY, Command, Registry

Handler = Callable[[str, Any], Any]

# shared commands that make sense in code mode (name -> alias names to drop because code mode gives them another meaning)
_KEEP: dict[str, tuple[str, ...]] = {
    "say": (), "thinking": ("think",), "model": (), "steps": (), "mode": (), "plan": (), "undo": (), "context": ("ctx",), "compact": (),
    "status": (), "tasks": (), "queue": (), "log": (), "events": (), "usage": (), "transcript": (), "why-slow": (), "export": (),
    "help": (), "stop": (), "pause": (), "resume": (), "clear": (), "new": (), "quit": (),
}


def _extra() -> list[Command]:
    D, I = C.DEFERRED, C.INSTANT
    return [
        Command("green", "Trust this repo: same as /mode auto", "/green", "Safety & access", D),
        Command("allow", "A command pattern that never asks (e.g. pytest*)", "/allow [pattern]", "Safety & access", D, sub_when={"": I}),
        Command("unallow", "Remove an allowed command pattern", "/unallow <pattern>", "Safety & access", D),
        Command("deny", "A command pattern that never runs", "/deny [pattern]", "Safety & access", D, sub_when={"": I}),
        Command("undeny", "Remove a denied command pattern", "/undeny <pattern>", "Safety & access", D),
        Command("network", "May commands use the network?", "/network [on|off]", "Safety & access", D, sub_when={"": I}),
        Command("installs", "May commands install packages?", "/installs [on|off]", "Safety & access", D, sub_when={"": I}),
        Command("commit", "May the agent git commit?", "/commit [never|ask|auto]", "Safety & access", D, sub_when={"": I}),
        Command("policy", "Everything the agent may do in this folder", "/policy", "Safety & access", I),
        Command("think", "Reasoning on tool steps (auto = only when stuck)", "/think [auto|on|off]", "Code", D, sub_when={"": I}),
        Command("ctx", "Show or set the context window size", "/ctx [N]", "Context & memory", D, sub_when={"": I}),
        Command("diff", "Git diff, or the diff of one edited file of this run", "/diff [path]", "Code", I),
        Command("files", "Files this run changed", "/files", "Code", I),
        Command("map", "The repo map the agent sees", "/map", "Code", I),
        Command("state", "Verified state measured now (runs the tests)", "/state", "Code", D),
        Command("stages", "Stages of a staged task (`run` continues them)", "/stages [run]", "Code", I, sub_when={"run": D}),
        Command("note", "Save a fact about this project", "/note <text>", "Context & memory", I),
        Command("sessions", "Saved sessions of this folder", "/sessions", "Session", I),
        Command("delete", "Delete a saved session", "/delete <id>", "Session", I),
    ]


def build_registry(base: Registry = REGISTRY) -> Registry:
    """The command table of code mode (a copy: the shared REGISTRY is not touched)."""
    r = Registry()
    for cmd in base.commands():
        if cmd.name in _KEEP:
            drop = _KEEP[cmd.name]
            if cmd.name == "resume":
                cmd = dataclasses.replace(cmd, summary="Resume a paused run, or switch to a saved session: /resume <id>", usage="/resume [id]")
            r.register(dataclasses.replace(cmd, aliases=tuple(a for a in cmd.aliases if a not in drop), handler=None))
    # `undo list` is read-only; `plan clear` / `new` change the session
    for cmd in _extra():
        r.register(cmd)
    return r


# ── capture of the classic command output ──────────────────────────────────

def capture_console(width: int = 100) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=max(40, width), force_terminal=False, color_system=None, highlight=False, legacy_windows=False), buf


def classic(engine: Any, line: str, width: int = 100) -> str:
    """Run one classic code-REPL command (`/network on`) with the engine and return what it printed, as plain text."""
    from agent.codeagent import session as S

    con, buf = capture_console(width)
    S.handle_command(con, engine, line)
    return buf.getvalue().rstrip("\n")


def _line(name: str, args: str) -> str:
    return f"/{name} {args}".rstrip()


# ── step-boundary control (NightageApp `control`) ──────────────────────────

class CodeControl:
    """Used by the app while a run is active. Everything takes effect at the engine's next step boundary."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def request_compact(self) -> None:
        self.ctx.runner.compact()

    def queue_user_message(self, text: str) -> None:
        self.ctx.runner.say(text)

    def request_pause(self) -> None:
        self.ctx.runner.pause()

    def request_resume(self) -> None:
        self.ctx.runner.resume()

    def request_stop(self) -> None:
        self.ctx.runner.stop()

    def request_steps(self, n: int) -> None:
        from agent.codeagent import session as S

        n = max(5, min(int(n), 500))
        S.persist_policy(self.ctx.engine, max_steps=n)
        self.ctx.app.renderer.info(f"steps per prompt: {n} (saved for this folder)")

    def request_mode(self, name: str) -> None:
        from agent.codeagent import session as S

        mode = name.strip().lower()
        problem = S.mode_problem(self.ctx.engine, mode)
        if problem:
            self.ctx.app.renderer.notice(problem)
            return
        S.persist_policy(self.ctx.engine, mode=mode)
        self.ctx.app.renderer.info(f"mode {mode} (saved for this folder)")


# ── handlers ───────────────────────────────────────────────────────────────

BRIDGED = ("mode", "green", "allow", "unallow", "deny", "undeny", "network", "installs", "commit", "steps", "think", "model", "ctx", "policy",
           "undo", "compact", "events", "why-slow", "note", "new", "clear", "sessions", "delete", "state")


def make_handlers(ctx: Any) -> dict[str, Handler]:
    """{command name: fn(args, app)} for NightageApp(handlers=...). Output is returned as text (or a renderable)."""

    def width(app: Any) -> int:
        return int(getattr(app.console, "width", 100) or 100)

    def bridged(name: str) -> Handler:
        def run(args: str, app: Any) -> Any:
            if name in ("new", "clear"):
                out = classic(ctx.engine, _line(name, args), width(app))
                ctx.reset_run_view()
                return out
            return classic(ctx.engine, _line(name, args), width(app))
        return run

    handlers: dict[str, Handler] = {n: bridged(n) for n in BRIDGED}

    def say(args: str, app: Any) -> Any:
        if not args.strip():
            return "usage: /say <text>   (a message for the running agent; without a run it simply starts one)"
        return ctx.start_prompt(args.strip())

    def undo(args: str, app: Any) -> Any:
        if args.strip().lower() == "list":
            rows = ctx.runner.snapshot().get("files") or []
            if not rows:
                return "nothing to undo yet"
            return ctx.runner.panel.files(width=width(app))
        return classic(ctx.engine, _line("undo", args), width(app))

    def diff(args: str, app: Any) -> Any:
        path = args.strip()
        if path:
            panel = ctx.runner.panel
            hit = path if path in panel.diffs else next((p for p in panel.diffs if p.endswith(path)), None)
            if hit is None:
                known = ", ".join(sorted(panel.diffs)[:6]) or "none yet"
                return f"no recorded edit for {path} in this run (edited: {known})"
            return panel.diff(hit, width=width(app))
        return str(ctx.engine.tools.git_diff() or "").rstrip() or "no changes in the working tree"      # not via rich markup: a diff is full of [...]

    def repo_map(args: str, app: Any) -> Any:
        return str(ctx.engine.tools.repo_map() or "").rstrip() or "(empty repo map)"

    def files(args: str, app: Any) -> Any:
        ctx.runner.snapshot()
        if ctx.runner.panel.file_rows():
            return ctx.runner.panel.files(width=width(app))
        return classic(ctx.engine, "/files", width(app))

    def plan(args: str, app: Any) -> Any:
        if args.strip().lower() == "clear":
            return classic(ctx.engine, "/plan clear", width(app))
        snap = ctx.runner.snapshot()
        if snap.get("plan"):
            return ctx.runner.panel.plan(width=width(app))
        return classic(ctx.engine, "/plan", width(app))

    def stages(args: str, app: Any) -> Any:
        from agent.codeagent import session as S

        if args.strip().lower() != "run":
            return classic(ctx.engine, "/stages", width(app))
        plan_ = ctx.engine.stage_plan
        if plan_ is None or not plan_.unfinished():
            return "no unfinished stages in this session"
        if S.stage_mode(ctx.engine.config) == "off":
            ctx.engine.config = S.with_stages(ctx.engine.config, True)
        ctx.start_prompt(plan_.task)                    # the runner continues the unfinished stage plan (the text is ignored)
        return None

    def resume(args: str, app: Any) -> Any:
        if not args.strip():                            # no id: the paused run (the app's own /resume, which also restarts a paused queue)
            return app._local["resume"](args, app)
        if app.worker.busy:
            return "a run is going on: /stop it first, then /resume <id>"
        out = classic(ctx.engine, _line("resume", args), width(app))
        ctx.reset_run_view()
        return out

    handlers.update({"map": repo_map, "resume": resume, "say": say, "undo": undo, "diff": diff, "files": files, "plan": plan, "stages": stages})
    return handlers


def context_text(ctx: Any, width: int = 100) -> str:
    """`/context` for the app's context_provider: the classic numbers (window fill, message sizes, thresholds)."""
    return classic(ctx.engine, "/context", width)


def status_rows(ctx: Any) -> dict[str, str]:
    """Extra `/status` rows from the live engine snapshot."""
    s = ctx.runner.snapshot()
    rows: dict[str, str] = {}
    eng = ctx.engine
    rows["session"] = str(getattr(eng, "session_id", "") or "-")
    step = f"{s.get('step', 0)}/{s['budget_now']}" if s.get("budget_now") else str(s.get("step", 0))
    if s.get("budget_source"):
        step += f" ({s['budget_source']})"
    if s.get("step_ceiling") and s.get("step_ceiling") != s.get("budget_now"):
        step += f", ceiling {s['step_ceiling']}"
    rows["steps"] = step
    rows["policy"] = ctx.policy_text()
    if s.get("files_changed"):
        rows["files"] = f"{s['files_changed']} changed (+{s['lines_added']} -{s['lines_removed']}), /files /diff /undo"
    plan = s.get("plan") or {}
    if plan.get("stages"):
        done = sum(1 for x in plan["stages"] if x.get("status") == "done")
        rows["stages"] = f"{done}/{len(plan['stages'])} done"
    if plan.get("checklist_total"):
        rows["acceptance"] = f"{plan.get('checklist_confirmed', 0)}/{plan['checklist_total']} confirmed"
    if s.get("queued"):
        rows["steer"] = f"{s['queued']} message(s) waiting for the next step"
    return rows
