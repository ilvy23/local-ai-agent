"""`nightage night run|status`: the command line of the night run (see __init__.py)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import typer
from rich.console import Console

from agent.night import evidence as evidence_mod, polish as polish_mod, sdnotify, selftest as selftest_mod, canary as canary_mod, charter as charter_mod, history as history_mod, unit as unit_mod, clock as clock_mod, doctor as doctor_mod, envinfo, gates as gates_mod, gitbox, nightconf, inhibit, lock as lock_mod, notify as notify_mod, overlay as overlay_mod, plan as plan_mod, preflight, probe as probe_mod, queue, splitter, templates, watch as watch_mod
from agent.night.governor import Night
from agent.night.runner import subprocess_runner

night_app = typer.Typer(help="Unattended coding run for the night: a queue of prompts, a command charter, checkpoints, 8 hours.", no_args_is_help=True)
console = Console()
TEST_TIMEOUT_S = 15 * 60


def _say(text: str) -> None:
    """A log line as it is: `[prompt-name]` in it is not Rich markup."""
    console.print(text, markup=False, highlight=False)


def _tests(root: Path, timeout_s: float = TEST_TIMEOUT_S):
    from agent.codeagent import profile as profile_mod
    from agent.night import testwatch

    prof = profile_mod.detect(root)
    cmd = prof.test_cmd
    if not cmd:
        return lambda: testwatch.TestState(ok=True, tail="no test command", digest="no test command")
    extra = gates_mod.from_profile(prof)

    def run() -> testwatch.TestState:
        try:
            p = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return testwatch.TestState(ok=False, tail="the tests timed out", digest=f"the tests did not finish within {timeout_s:.0f} s", timed_out=True)
        state = testwatch.parse(p.stdout + p.stderr, p.returncode)
        return gates_mod.apply(state, gates_mod.run(root, extra)) if state.ok and extra else state
    return run


def _ask_model():
    from agent import llm as llm_mod
    from agent.codeagent.session import apply_client_config
    from agent.config import load_config, resolved
    from agent.main import _open_llm

    config = resolved(load_config())
    client = _open_llm()
    apply_client_config(client, config)
    model = (config.get("code") or {}).get("model") or (config.get("models") or {}).get("coding") or config["models"]["chat"]
    return lambda q: llm_mod.utility_chat(client, [{"role": "user", "content": q}], model, json_mode=True, num_predict=1500)[0]


HEARTBEAT = sdnotify.Heartbeat()          # replaced by a fresh one for every `night run` (see _crash_guard); proof of life for the systemd watchdog
CRASH_EXIT = 70                      # EX_SOFTWARE: an unexpected error in the governor. Not 1 (a deliberate refusal): the systemd unit restarts a crash, never a refusal


def _crash_guard(fn):
    """An unexpected exception in `night run` ends with CRASH_EXIT after the traceback went to .night/errors.log and the lock and the policy were put right: so a bug in the governor
    is restarted (and resumed) by the unit instead of being mistaken for a refusal."""
    import functools
    import traceback

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global HEARTBEAT
        HEARTBEAT = sdnotify.Heartbeat()
        try:
            with HEARTBEAT:
                return fn(*args, **kwargs)
        except (typer.Exit, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001
            root = Path(str(kwargs.get("folder") or ".")).expanduser().resolve()
            try:
                (root / ".night").mkdir(exist_ok=True)
                with open(root / ".night" / "errors.log", "a") as f:
                    f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} crash of `night run`\n{traceback.format_exc()}\n")
                charter_mod.restore_policy(root)
                lock_mod.release(root)
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass
            console.print(f"[red]night run crashed ({type(e).__name__}: {str(e)[:120]}); traceback in {root / '.night' / 'errors.log'}[/red]")
            raise typer.Exit(CRASH_EXIT)
    return wrapper


@night_app.command("run")
@_crash_guard
def run(prompts: list[str] = typer.Argument(..., help="prompt files, folders of prompts (natural order, rules.md is shared) or inline text; revealed one at a time"),
        folder: str = typer.Option(".", "--folder", help="the project folder (it becomes a git repo if it is none)"),
        hours: float = typer.Option(None, "--hours", help="how long the night lasts (config night.hours, default 8)"),
        until: str = typer.Option("", "--until", help="HH:MM: end at this time of day instead of after --hours (the next time the clock shows it)"),
        start_at: str = typer.Option("", "--start-at", help="HH:MM: wait until then before starting (the charter is still asked now)"),
        tournament: int = typer.Option(None, "--tournament", help="candidates per round when two attempts in a row failed the same way (0/1 = off; config night.tournament, default 3)"),
        plan_first: bool = typer.Option(False, "--plan", help="every prompt starts with a checklist plan whose steps the harness ticks by measurement (a prompt can ask for it with `plan: true`)"),
        review: bool = typer.Option(None, "--review/--no-review", help="after each finished prompt, a fresh attempt reviews the diff and fixes real problems (kept only if tests stay green)"),
        yes: bool = typer.Option(False, "--yes", help="accept the proposed command charter without asking"),
        no_model: bool = typer.Option(False, "--no-model", help="charter from the project alone, no model call"),
        resume: bool = typer.Option(False, "--resume", help="continue an interrupted night (keeps its charter, statuses and the time that was left); `nightage continue` does this from any folder"),
        map_mode: str = typer.Option(None, "--map", help="off | auto (a map of the project written from the code, kept fresh) | full (+ a first model pass that adds notes)"),
        pause_on_game: bool = typer.Option(None, "--pause-on-game/--ignore-games", help="wait while a game is running (config night.pause_on_game, default on)"),
        stage_cap: int = typer.Option(None, "--stage-cap", help="prompt cap per stage for the attempts (tokens; 0 = leave your config). 64000 was the best of the measured runs"),
        ctx: int = typer.Option(None, "--ctx", help="context window for the attempts (tokens; 0 = your config)"),
        canary: bool = typer.Option(True, "--canary/--no-canary", help="before the night: one tiny task through the whole chain (model, runner, file, git) must work"),
        then: str = typer.Option("", "--then", help="prompts to run on a COPY of the result afterwards, in the time that is left, when the main prompts are done and at least --then-min-hours remain"),
        then_min_hours: float = typer.Option(2.0, "--then-min-hours", help="how much time must be left for the --then phase to start"),
        default_limits: bool = typer.Option(False, "--default-limits", help="keep the coding agent's normal guard limits instead of the relaxed night profile (overlay.RELAXED)")) -> None:
    from agent import repoaccess
    from agent.config import load_config

    conf = nightconf.from_config(load_config())                                  # the night: section of config.yaml; options given on the command line win
    hours = conf["hours"] if hours is None else hours
    tournament = conf["tournament"] if tournament is None else tournament
    review = conf["review"] if review is None else review
    map_mode = conf["map"] if map_mode is None else map_mode
    pause_on_game = conf["pause_on_game"] if pause_on_game is None else pause_on_game
    stage_cap = conf["stage_cap"] if stage_cap is None else stage_cap
    ctx = conf["ctx"] if ctx is None else ctx
    root = Path(folder).expanduser().resolve()
    problem = repoaccess.folder_problem(root)
    if problem:
        console.print(f"[red]{problem}[/red]")
        raise typer.Exit(1)
    items, rules = queue.load(prompts)
    if not items:
        console.print("[red]no prompts found[/red]")
        raise typer.Exit(1)
    from datetime import datetime

    try:
        if until:
            hours = clock_mod.hours_until(until, datetime.now()) - clock_mod.seconds_to_start(start_at, datetime.now()) / 3600
            hours = max(hours, 0.25)
        clock_mod.seconds_to_start(start_at, datetime.now())
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    nd = root / ".night"
    nd.mkdir(exist_ok=True)
    try:
        state = json.loads((nd / "state.json").read_text()) if resume and (nd / "state.json").exists() else None
        if state is not None:
            {i["name"]: i for i in state["items"]}, state["left_s"]
    except (OSError, ValueError, KeyError, TypeError):
        console.print(f"[red]{nd / 'state.json'} is damaged, so this night cannot be continued.[/red] Start it again without --resume (the old state is kept as it is).")
        raise typer.Exit(1)
    if state:
        old = {i["name"]: i for i in state["items"]}
        for it in items:
            if it.name in old:
                it.status, it.attempts, it.seconds, it.note = ("pending" if old[it.name]["status"] == "active" else old[it.name]["status"]), old[it.name]["attempts"], old[it.name]["seconds"], old[it.name]["note"]
        hours = state["left_s"] / 3600
        if state.get("end_wall"):                                                  # the wall clock, not the last saved value: a crash and a restart took time too
            hours = (state["end_wall"] - time.time()) / 3600
            if hours <= 0.05:
                console.print("[yellow]the time of this night is over (see .night/NIGHT_REPORT.md)[/yellow]")
                raise typer.Exit(3)
    if state and then and all(i["status"] == "done" for i in state["items"]) and hours > then_min_hours:
        console.print("the main prompts are done: continuing with the evaluation phase")
        code = _then_phase(root, then, state["end_wall"] if state.get("end_wall") else time.time() + hours * 3600)
        notify_mod.notify(root, f"the evaluation phase of {root.name} is over (see {root.name}-eval/.night/EVAL.md and NIGHT_REPORT.md)", marker=True)
        raise typer.Exit(code)
    warnings = preflight.check(items, root)
    for w in warnings:
        console.print(f"[yellow]prompt check:[/yellow] {w}")
    if warnings and not yes and not resume and not typer.confirm("Start anyway?", default=True):
        raise typer.Exit(1)
    if resume and (nd / "charter.json").exists():
        ch = charter_mod.Charter(**json.loads((nd / "charter.json").read_text()))
    else:
        console.print("[dim]settling the command charter ...[/dim]")
        ch = charter_mod.propose(root, [i.text for i in items], None if no_model else _try(_ask_model))
        console.print(f"[bold]Charter for {root}[/bold]\n{ch.show()}\n[dim]Edits inside this folder and these commands run without asking; BLOCKED commands, deny patterns "
                      "and anything outside the folder stay refused.[/dim]")
        if not yes and not typer.confirm("Start the night with this charter?", default=False):
            raise typer.Exit(1)
        (nd / "charter.json").write_text(ch.to_json())
    tests = _tests(root)
    from agent.codeagent import profile as profile_mod

    facts = probe_mod.probe(profile_mod.detect(root).test_cmd, tests)
    probe_mod.write(root, facts)
    tests = _tests(root, probe_mod.test_timeout(facts))                          # a hang is noticed in minutes, not after a fixed quarter hour
    console.print(f"[dim]{facts.text()}[/dim]")
    if not facts.usable and not yes and not typer.confirm("The test command does not work, so nothing can be verified. Start anyway?", default=False):
        raise typer.Exit(1)
    charter_mod.apply(root, ch)
    from agent.codeagent import eventlog

    if not lock_mod.acquire(root):
        console.print(f"[red]another night (pid {lock_mod.owner(root)}) is already running in {root}[/red]")
        raise typer.Exit(1)
    from agent import resume as resume_mod

    resume_mod.record_night(root, prompts, dict(plan_first=plan_first, review=review, map_mode=map_mode, tournament=tournament, pause_on_game=pause_on_game,
                                                stage_cap=stage_cap, ctx=ctx, default_limits=default_limits, no_model=no_model), resumed=resume)
    try:
        from agent.config import load_config

        env_lines = envinfo.collect(root, load_config(), json.loads(overlay_mod.env(stage_cap, ctx, relaxed=not default_limits)["NIGHT_OVERLAY"]))
    except Exception:  # noqa: BLE001 - the machine description is a bonus
        env_lines = []
    runner = subprocess_runner(root, eventlog.default_path(), env=overlay_mod.env(stage_cap, ctx, relaxed=not default_limits), progress=lambda: gitbox.work_token(root))
    health_obj = _health(root, pause_on_game)

    def make_night(h: float) -> Night:
        return Night(root, items, rules, h, runner=runner, tests=tests, log=_say, health=health_obj, events=eventlog.default_path(), map_mode=map_mode, tournament=tournament,
                     review=review, env_lines=env_lines, plan_all=plan_first, use_history=True, heartbeat=HEARTBEAT.alive, prompts_src=[str(Path(p).expanduser()) for p in prompts])

    night = make_night(hours)
    (nd / "CONTROL").unlink(missing_ok=True)
    (nd / "DONE").unlink(missing_ok=True)                                          # a marker of an earlier night must not stop this one
    if canary and not resume:
        console.print("[dim]canary: one tiny task through the whole chain ...[/dim]")
        ok, detail = canary_mod.run(root, night.runner, timeout_s=canary_mod.timeout_for(facts.seconds))
        console.print(f"canary: {detail}")
        if not ok:
            charter_mod.restore_policy(root)
            lock_mod.release(root)
            raise typer.Exit(1)
    wait = clock_mod.seconds_to_start(start_at, datetime.now())
    if wait > 0:
        console.print(f"waiting until {start_at} ({wait / 3600:.1f} h) ...")
        end = time.time() + wait
        while time.time() < end:                                           # in slices: the watchdog must see that this process is alive while it waits
            HEARTBEAT.alive(f"waiting for the start time {start_at}")
            time.sleep(min(60.0, max(end - time.time(), 0.0)))
        night = make_night(hours)                                          # the clock of the night starts now
    import signal

    def _graceful(signum, frame):                                       # kill, systemd stop, a closed terminal: same clean end as Ctrl+C (state saved, policy restored, lock released)
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _graceful)
        except (ValueError, OSError):
            pass
    try:
        with inhibit.Inhibitor():
            report = night.run()
    except KeyboardInterrupt:
        night.note("the night was stopped (Ctrl+C / SIGTERM): this is NOT an end state; continue with `nightage continue`")
        try:                                                              # what exists so far is still worth reading: the report and the evaluation of the attempts made
            from agent.night import report as report_mod

            (nd / "NIGHT_REPORT.md").write_text(report_mod.render(night, bool(night.last_state and night.last_state.ok)))
            evidence_mod.finalize(night, night.prompts_src)
        except Exception:  # noqa: BLE001 - the stop must always work
            pass
        night.save()
        console.print(f"[yellow]stopped - the state is saved. Continue with `nightage continue {resume_mod.night_id(root)}` (from any folder)[/yellow]")
        raise typer.Exit(130)
    except Exception as e:  # noqa: BLE001 - a crash must leave a saved state and a clear message, not a traceback
        import traceback

        night.save()
        with open(nd / "errors.log", "a") as f:
            f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} crash of the governor\n{traceback.format_exc()}\n")
        console.print(f"[red]the night stopped with an error ({type(e).__name__}: {str(e)[:200]}).[/red] The state is saved: continue with `nightage continue {resume_mod.night_id(root)}`.", markup=True)
        raise typer.Exit(CRASH_EXIT)                                        # not 1: the unit restarts a crash and resumes, it never restarts a refusal
    finally:
        charter_mod.restore_policy(root)
        lock_mod.release(root)
    left_h = (night.end_wall - time.time()) / 3600
    more = bool(then) and left_h >= then_min_hours
    notify_mod.notify(root, notify_mod.headline(report.done, report.partial, report.tests_ok) + (" - the evaluation phase follows" if more else ""), marker=not more)
    console.print(f"[bold]night over[/bold]  done {len(report.done)}  not finished {len(report.partial)}  tests {'green' if report.tests_ok else 'RED'}  report: {nd / 'NIGHT_REPORT.md'}")
    code = 0 if not report.partial else 3
    if more:
        code = max(code, _then_phase(root, then, night.end_wall))
        notify_mod.notify(root, f"the evaluation phase of {root.name} is over (see {root.name}-eval/.night/EVAL.md and NIGHT_REPORT.md)", marker=True)
    raise typer.Exit(code)


def eval_folder(root: Path) -> Path:
    return root.parent / f"{root.name}-eval"


def _then_phase(root: Path, then_dir: str, end_wall: float) -> int:
    """The evaluation phase: the --then prompts run on a COPY of the finished project (<name>-eval) until the end time of the night. The copy keeps the main result intact; a
    restart resumes the copy's own night (its state is in its own .night/)."""
    import shutil
    import sys

    dest = eval_folder(root)
    if not dest.exists():
        shutil.copytree(root, dest, symlinks=True, ignore=shutil.ignore_patterns(".night", "__pycache__", ".pytest_cache"))
        for tag in gitbox.git(dest, "tag", "-l", "night/*").stdout.split():
            gitbox.git(dest, "tag", "-d", tag)
        gitbox.git(dest, "reset", "-q", "--hard", "HEAD")
    until = time.strftime("%H:%M", time.localtime(end_wall))
    console.print(f"[bold]evaluation phase[/bold] on {dest} until {until}")
    cmd = [sys.executable, "-c", "from agent.main import app; app()", "night", "run", then_dir, "--folder", str(dest), "--yes", "--no-model", "--no-canary", "--resume", "--until", until]
    return subprocess.run(cmd).returncode


def _health(root: Path, pause_on_game: bool):
    from agent.night.health import Health

    vk = subprocess.run(["systemctl", "--user", "cat", "ollama-vk"], capture_output=True).returncode == 0       # the Vulkan server of the dual-GPU setup
    return Health(root, urls=["http://127.0.0.1:11436"] if vk else ["http://127.0.0.1:11434"], restart_cmds=[["systemctl", "--user", "restart", "ollama-vk"]] if vk else [],
                  pause_on_game=pause_on_game)


def _try(factory):
    try:
        return factory()
    except Exception:  # noqa: BLE001 - no model reachable: the baseline charter still works
        return None


@night_app.command("status")
def status(folder: str = typer.Option(".", "--folder")) -> None:
    f = Path(folder).expanduser().resolve() / ".night" / "state.json"
    if not f.exists():
        console.print("no night run in this folder")
        raise typer.Exit(1)
    s = json.loads(f.read_text())
    if _charter_left(f.parent.parent):
        console.print("[yellow]the command charter of an earlier night is still in this folder's policy (the night did not end cleanly): "
                      "`nightage night restore-policy` puts your policy back[/yellow]")
    console.print(f"time left: {s['left_s'] / 3600:.1f} h")
    for i in s["items"]:
        console.print(f"  {i['status']:8s} {i['name']}  attempts {i['attempts']}  {i['seconds'] / 60:.0f} min  {i['note'][:80]}")
    console.print("\n".join(s["events"][-8:]))
    console.print(f"[dim]checkpoints:\n{gitbox.head_summary(f.parent.parent, 8)}[/dim]")


def _control(folder: str, word: str | None) -> None:
    f = Path(folder).expanduser().resolve() / ".night" / "CONTROL"
    if not f.parent.exists():
        console.print("no night run in this folder")
        raise typer.Exit(1)
    if word:
        f.write_text(word + "\n")
    else:
        f.unlink(missing_ok=True)


@night_app.command("stop")
def stop(folder: str = typer.Option(".", "--folder")) -> None:
    """End the night gracefully: the running attempt finishes its slice, then the end state and the report are written."""
    _control(folder, "stop")
    console.print("stop requested")


@night_app.command("pause")
def pause(folder: str = typer.Option(".", "--folder")) -> None:
    """Hold between attempts (the running one finishes) until `nightage night go`."""
    _control(folder, "pause")
    console.print("pause requested")


@night_app.command("go")
def go(folder: str = typer.Option(".", "--folder")) -> None:
    _control(folder, None)
    console.print("running again")


@night_app.command("report")
def report(folder: str = typer.Option(".", "--folder")) -> None:
    """Print the morning report of the last night in this folder."""
    f = Path(folder).expanduser().resolve() / ".night" / "NIGHT_REPORT.md"
    if not f.exists():
        console.print("no report yet")
        raise typer.Exit(1)
    console.print(f.read_text(), markup=False)


@night_app.command("init")
def init(folder: str = typer.Argument("night-prompts", help="where to write the prompt folder"),
         only: list[str] = typer.Option([], "--only", help="template names to write (feature, bug, coverage, refactor, docs); default all")) -> None:
    """Write rules.md and prompt templates (with front matter) to fill in; existing files are never overwritten."""
    files = templates.init(Path(folder).expanduser(), only or None)
    console.print("\n".join(f"wrote {f}" for f in files) or "nothing written (the files already exist)")


@night_app.command("split")
def split(prompt: str = typer.Argument(..., help="one big prompt file"), folder: str = typer.Argument("night-prompts"),
          independent: bool = typer.Option(False, "--independent", help="do not make each part depend on the one before")) -> None:
    """Split one big prompt (Stage 1: ... Stage 2: ... or ## headings) into one prompt file per part."""
    files = splitter.write(Path(prompt).expanduser().read_text(errors="replace"), Path(folder).expanduser(), chain=not independent)
    if not files:
        console.print("[yellow]found fewer than two stages or headings: nothing to split[/yellow]")
        raise typer.Exit(1)
    console.print("\n".join(f"wrote {f}" for f in files))


@night_app.command("plan")
def plan(prompts: list[str] = typer.Argument(...), folder: str = typer.Option(".", "--folder"), hours: float = typer.Option(8.0, "--hours"),
         no_model: bool = typer.Option(True, "--no-model/--model", help="charter from the project alone (default) or ask the model too")) -> None:
    """Show what a night would do - order, minutes per prompt, prompt check, charter, test command, machine - without starting anything."""
    from agent.codeagent import profile as profile_mod

    root = Path(folder).expanduser().resolve()
    items, _ = queue.load(prompts)
    if not items:
        console.print("[red]no prompts found[/red]")
        raise typer.Exit(1)
    ch = charter_mod.propose(root, [i.text for i in items], None if no_model else _try(_ask_model))
    facts = probe_mod.probe(profile_mod.detect(root).test_cmd, _tests(root))
    console.print(plan_mod.render(items, root, hours, ch.show(), facts.text(), _health(root, True).gate()), markup=False)


@night_app.command("watch")
def watch(folder: str = typer.Option(".", "--folder"), every: float = typer.Option(5.0, "--every", help="seconds between redraws")) -> None:
    """A live view of a running night (Ctrl+C leaves it running)."""
    from rich.live import Live

    from agent.night import health

    root = Path(folder).expanduser().resolve()
    if not watch_mod.load(root):
        starting = (root / ".night" / "LOCK").exists() or (root / ".night" / "charter.json").exists()
        console.print("a night is starting up (prompt check, baseline tests ~1-2 min); run watch again shortly" if starting else "no night run in this folder")
        raise typer.Exit(1)
    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                try:
                    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5).stdout.splitlines()[1:2]
                except (OSError, subprocess.SubprocessError):
                    ps = []
                control = ""
                try:
                    control = (root / ".night" / "CONTROL").read_text().strip()
                except OSError:
                    pass
                live.update(watch_mod.render(watch_mod.load(root), " ".join(ps[0].split()[:6]) if ps else "", health.gpu_temps(), control))
                time.sleep(every)
    except KeyboardInterrupt:
        pass


@night_app.command("doctor")
def doctor(folder: str = typer.Option(".", "--folder")) -> None:
    """Can this machine run a night right now? Checks git, the coding model, the model server, the test command, disk, games, GPU heat, sleep inhibit."""
    from agent import llm as llm_mod
    from agent.codeagent import profile as profile_mod
    from agent.config import load_config, resolved

    root = Path(folder).expanduser().resolve()
    config = resolved(load_config())
    models = config.get("models") or {}
    model = (config.get("code") or {}).get("model") or models.get("coding") or models.get("chat")
    try:
        import httpx

        installed = [m["name"] for u in ("http://127.0.0.1:11434", "http://127.0.0.1:11436") for m in httpx.get(u + "/api/tags", timeout=3).json().get("models", [])]
    except Exception:  # noqa: BLE001
        installed = None
    facts = probe_mod.probe(profile_mod.detect(root).test_cmd, _tests(root))
    checks = doctor_mod.run(root, coding_model=model, installed=installed, urls=["http://127.0.0.1:11436", "http://127.0.0.1:11434"], test_facts=facts)
    out = doctor_mod.text(checks)
    console.print(out, markup=False)
    raise typer.Exit(0 if out.endswith("ready") and "not ready" not in out else 1)


@night_app.command("review")
def review(folder: str = typer.Option(".", "--folder")) -> None:
    """What the night changed, what was rolled back (and where it is kept), and how to keep or drop it."""
    root = Path(folder).expanduser().resolve()
    if not gitbox.tags(root, "night/base"):
        console.print("no night run in this folder")
        raise typer.Exit(1)
    console.print(gitbox.review_text(root), markup=False)


@night_app.command("undo")
def undo(folder: str = typer.Option(".", "--folder"), yes: bool = typer.Option(False, "--yes")) -> None:
    """Put the project back to where it was before the night. Nothing is lost: the night stays under the tag night/undone-N."""
    root = Path(folder).expanduser().resolve()
    if not gitbox.tags(root, "night/base"):
        console.print("no night run in this folder")
        raise typer.Exit(1)
    if lock_mod.owner(root) is not None:
        console.print("[red]a night is still running here: `nightage night stop` first[/red]")
        raise typer.Exit(1)
    if not yes and not typer.confirm("Go back to the state before the night?", default=False):
        raise typer.Exit(1)
    console.print(f"back to the baseline; the night is kept as {gitbox.undo(root)}")


@night_app.command("squash")
def squash(folder: str = typer.Option(".", "--folder"), message: str = typer.Option("nightage night: overnight work", "--message", "-m")) -> None:
    """Make all of the night's commits one commit on top of the baseline."""
    root = Path(folder).expanduser().resolve()
    if lock_mod.owner(root) is not None:
        console.print("[red]a night is still running here: `nightage night stop` first[/red]")
        raise typer.Exit(1)
    sha = gitbox.squash(root, message) if gitbox.tags(root, "night/base") else None
    console.print(f"one commit: {sha}" if sha else "nothing to squash")
    raise typer.Exit(0 if sha else 1)


@night_app.command("tune")
def tune(fit: int = typer.Option(64, "--fit", help="LLAMA_ARG_FIT_TARGET in MB: headroom per GPU the model server keeps (your tune.conf has 256; 64 was measured faster at 128k)"),
         pin: bool = typer.Option(False, "--pin", help="also pin the model server to one thread per P-core (helps only if part of the model sits on the CPU)"),
         apply_it: bool = typer.Option(False, "--apply", help="write the drop-in (nothing is written without this)"),
         restart: bool = typer.Option(False, "--restart", help="with --apply/--revert: restart ollama-vk now (this unloads the model)"),
         revert_it: bool = typer.Option(False, "--revert", help="remove the drop-in this command wrote")) -> None:
    """Show (or with --apply, write) the Ollama service settings measured to matter at 128k. Your own unit and tune.conf are never changed."""
    from agent.night import tune as tune_mod, vramhogs

    if revert_it:
        console.print("removed" if tune_mod.revert(restart=restart) else "nothing to remove", markup=False)
        return
    plan = tune_mod.Plan(fit, tune_mod.p_core_threads() if pin else [])
    console.print(f"drop-in: {tune_mod.dropin_path()}\n{plan.text()}", markup=False)
    console.print("graphics memory: " + vramhogs.summary(vramhogs.hogs()), markup=False)
    if apply_it:
        console.print(f"written: {tune_mod.apply(plan, restart=restart)}" + ("" if restart else "  (takes effect at the next restart of ollama-vk)"), markup=False)
    else:
        console.print("not written (add --apply)", markup=False)


@night_app.command("install-unit")
def install_unit(prompts: list[str] = typer.Argument(..., help="the same prompts as for `run`"), folder: str = typer.Option(".", "--folder"),
                 write: bool = typer.Option(False, "--write", help="write the unit file (default: only print it)"),
                 extra: str = typer.Option("", "--extra", help="more options of `night run`, e.g. \"--start-at 01:00 --until 08:00 --tournament 3 --review\"")) -> None:
    """A systemd user service that resumes the night after a crash or reboot and ends when the night is done. Nothing is enabled for you."""
    import sys

    root = Path(folder).expanduser().resolve()
    import os
    import shlex
    import shutil

    exe = shutil.which("nightage") or (sys.argv[0] if sys.argv and sys.argv[0].endswith("nightage") else "nightage")        # systemd has no login PATH: the full path
    text = unit_mod.render(root, [str(Path(p).expanduser().resolve()) if Path(p).expanduser().exists() else p for p in prompts], exe, extra_args=shlex.split(extra), path_env=os.environ.get("PATH", ""))
    console.print(text, markup=False)
    if write:
        target = unit_mod.unit_path(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        console.print(f"written: {target}\n{unit_mod.instructions(root)}", markup=False)
    else:
        console.print("(not written: add --write)", markup=False)


@night_app.command("nights")
def nights(last: int = typer.Option(10, "--last")) -> None:
    """The nights this machine has run: when, which project, how much got done."""
    rows = history_mod.rows()[-last:]
    if not rows:
        console.print("no nights recorded yet")
        return
    for r in rows:
        console.print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts']))}  {r['project']:<24} done {r['done']}  open {r['partial']}  {r['hours']} h  tests {'green' if r['tests_ok'] else 'RED'}", markup=False)


@night_app.command("runs")
def runs() -> None:
    """Every night run this machine knows (index + folders found by scanning), with its status and how to continue it. Same as `nightage continue --list` for nights."""
    from agent import resume as resume_mod

    raise typer.Exit(resume_mod.continue_cli(list_only=True, kind="night"))


@night_app.command("selftest")
def selftest(hours: float = typer.Option(0.75, "--hours", help="time budget of the toy night"), keep: bool = typer.Option(False, "--keep", help="keep the temp folder")) -> None:
    """Run a whole night on a toy project and check the result by measurement. Needs the model server: run it when the PC is free."""
    import shutil
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="night-selftest-")) / "toy"
    base.mkdir(parents=True)
    prompts = selftest_mod.build(base)
    console.print(f"toy project: {base}", markup=False)
    res = subprocess.run(selftest_mod.command(base, prompts, hours))
    checks = selftest_mod.verify(base)
    console.print(selftest_mod.text(checks), markup=False)
    if not keep:
        shutil.rmtree(base.parent, ignore_errors=True)
        shutil.rmtree(prompts, ignore_errors=True)
    raise typer.Exit(0 if all(c.ok for c in checks) and res.returncode == 0 else 1)


@night_app.command("polish")
def polish(prompt: str = typer.Argument(..., help="a prompt file"), folder: str = typer.Option(".", "--folder")) -> None:
    """Ask the model to rewrite a vague prompt into stages with checks. Writes PROMPT.polished.md next to it, never touches the original, and only if the prompt check likes it better."""
    src = Path(prompt).expanduser()
    root = Path(folder).expanduser().resolve()
    text = src.read_text(errors="replace")
    better, before, after = polish_mod.polish(text, root, _ask_model())
    console.print(f"prompt check before: {len(before)} problem(s)" + "".join(f"\n  - {w}" for w in before), markup=False)
    if better is None:
        console.print("no better version was produced (the original stays)", markup=False)
        raise typer.Exit(1)
    out = polish_mod.target(src)
    out.write_text(better + "\n")
    console.print(f"written: {out}  (prompt check after: {len(after)} problem(s)); the original is untouched - read it before you use it", markup=False)


def _charter_left(root: Path) -> bool:
    return (root / ".night" / "policy-before.json").exists() and lock_mod.owner(root) is None


@night_app.command("restore-policy")
def restore_policy(folder: str = typer.Option(".", "--folder")) -> None:
    """Put the folder's coding policy back to what it was before a night (a night that was killed hard leaves its command charter behind)."""
    root = Path(folder).expanduser().resolve()
    if lock_mod.owner(root) is not None:
        console.print("[red]a night is running here: `nightage night stop` first[/red]")
        raise typer.Exit(1)
    console.print("policy restored" if charter_mod.restore_policy(root) else "nothing to restore: no night charter is left in this folder", markup=False)


@night_app.command("eval")
def eval_cmd(folder: str = typer.Option(".", "--folder")) -> None:
    """The evaluation of a night (what was measured per attempt and per prompt, what the model said, where the raw evidence and its archive are)."""
    f = Path(folder).expanduser().resolve() / ".night" / "EVAL.md"
    if not f.exists():
        console.print("no evaluation yet (it is written when the night ends)")
        raise typer.Exit(1)
    console.print(f.read_text(), markup=False)
