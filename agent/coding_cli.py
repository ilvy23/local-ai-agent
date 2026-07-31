"""The print layer for `agent code`.

Subscribes to the coding event bus and renders; the coding core never prints.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

from agent.coding import repair
from agent.coding.verify import runner
from agent.events import Events


def _git_root(start: Path) -> Path | None:
    r = runner.run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    return Path(r.stdout.strip()) if r.ok and r.stdout.strip() else None


def _too_broad(root: Path) -> str | None:
    """Reason this directory must NOT be auto-committed, or None if it's fine.

    Guards the disaster where someone runs `agent code` in ~ and we'd snapshot
    their whole home folder. Home/system dirs are refused outright; anything with
    2000+ files is treated as 'not a single project'."""
    root = Path(root)
    if root == Path.home():
        return "your home directory"
    if root == Path("/") or root in Path.home().parents:
        return "a system directory"
    r = runner.run(["bash", "-lc",
                    "find . -type f -not -path './.git/*' 2>/dev/null | head -2001 | wc -l"],
                   cwd=root, timeout=30)
    try:
        n = int(r.stdout.strip())
    except ValueError:
        n = 0
    return f"a very large directory ({n}+ files)" if n > 2000 else None


def _baseline_commit(console, root: Path) -> Path | None:
    """Give a repo an initial commit so the worktree sandbox has a HEAD."""
    reason = _too_broad(root)
    if reason:
        console.print(f"[red]Refusing to auto-commit — {root} is {reason}, not a project.[/]")
        console.print("[dim]cd into the specific project folder and run `agent code` there. "
                      "If it really is your project, commit it yourself: git add -A && git commit -m init[/]")
        return None
    runner.run(["git", "add", "-A"], cwd=root)

    def _commit():
        return runner.run(["git", "commit", "-q", "-m", "initial commit (agent code)"], cwd=root)

    r = _commit()
    if not r.ok:  # no git identity → set a local one and retry
        runner.run(["git", "config", "user.email", "agent@localhost"], cwd=root)
        runner.run(["git", "config", "user.name", "agent"], cwd=root)
        r = _commit()
    if not r.ok:  # empty directory → empty baseline so HEAD exists
        runner.run(["git", "commit", "-q", "--allow-empty", "-m", "initial commit (agent code)"], cwd=root)
    console.print("[green]✓ committed a baseline[/]")
    return Path(root)


def _ensure_repo(console) -> Path | None:
    """Return a usable repo root: a git repo WITH a commit. Sets one up if safe.

    Cases: (1) cwd is inside a usable repo → return it; (2) cwd is inside a repo
    that's too-broad (home/huge) → IGNORE the outer repo and treat cwd as
    project root (offer to init HERE); (3) cwd is inside a repo with no commits
    → baseline it; (4) not in a repo → offer to init in cwd. Every init/commit
    path is guarded, and the guard fires BEFORE `git init` so a refusal never
    leaves a stray .git behind."""
    cwd = Path.cwd()
    root = _git_root(cwd)

    # Case 1 + 2: already inside a repo
    if root:
        if _too_broad(root) and root != cwd:
            # e.g. cwd is ~/Desktop/Test and a stray ~/.git was found. Ignore
            # the outer repo — the user clearly means to work in cwd.
            console.print(f"[dim]ignoring outer git repo at {root} (too broad); "
                          f"using {cwd} as the project.[/]")
            root = None
        elif runner.run(["git", "rev-parse", "HEAD"], cwd=root).ok:
            return root  # ready to go
        else:
            # repo exists but has no commits
            if _too_broad(root):
                console.print(f"[red]{root} is a git repo with no commits, and it's your home/"
                              "a huge dir — refusing to auto-commit it.[/]")
                console.print(f"[dim]remove it with:  rm -rf {root}/.git   "
                              "then run agent code from inside your project folder.[/]")
                return None
            console.print(f"[yellow]{root} is a git repo with no commits yet[/] — making a baseline.")
            return _baseline_commit(console, root)

    # Case 4: not in a repo — guard cwd BEFORE we ever touch git init
    reason = _too_broad(cwd)
    if reason:
        console.print(f"[red]Refusing — {cwd} is {reason}, not a project.[/]")
        console.print("[dim]cd into the specific project folder and run `agent code` there.[/]")
        return None
    console.print(f"[yellow]{cwd} isn't a git repository[/] — the agent needs one for its sandbox.")
    if console.input("Initialize a git repo here and continue? [Y/n] ").strip().lower() in ("n", "no"):
        console.print("[dim]cd into the project you want to work on, or run: git init[/]")
        return None
    runner.run(["git", "init", "-q"], cwd=cwd)
    return _baseline_commit(console, cwd)


def _prep_model(console):
    """Free the GPU for the coding model: unload every OTHER resident model so
    only this one occupies VRAM. Returns (llm, model) for later release."""
    from agent.coding import config as coding_config
    from agent.config import load_config
    from agent.llm import OllamaClient
    llm = OllamaClient()
    model = coding_config.load(load_config())["executor"]["model"]
    base = model.split(":")[0]
    others = [m for m in llm.loaded_models() if m.split(":")[0] != base]
    for m in others:
        llm.unload(m)
    if others:
        console.print(f"[dim]freed the GPU — unloaded: {', '.join(others)}[/]")
    return llm, model


def _release_model(llm, model, console):
    """Unload the coding model so it stops occupying VRAM after the run/stop/exit."""
    try:
        if llm.unload(model):
            console.print(f"[dim]unloaded {model} — GPU freed[/]")
    except Exception:  # noqa: BLE001 - cleanup must never raise
        pass


def _stream_token(console, text: str) -> None:
    """Print raw generation tokens live so the user watches the code being written."""
    import sys
    sys.stdout.write(text)
    sys.stdout.flush()


def run_goal_cli(goal: str) -> int:
    """Greenfield mode: generate the whole program in one shot, then repair it.

    A 7B loses coherence building incrementally through tools, but writes a
    complete file in one generation — so we play to that and only loop to fix."""
    from agent.coding.build import build

    console = Console()
    root = _ensure_repo(console)
    if root is None:
        return 1

    ev = Events()

    def on(e):
        k, d = e.kind, e.data
        if k == "build_round":
            label = "writing the whole program" if d["phase"] == "generate" else "fixing checks"
            console.print(f"\n[cyan]▸ round {d['n']}/{d['of']}[/] [dim]({label})[/]")
        elif k == "generate_start":
            console.print("[dim]  ┄ writing (live):[/]")
        elif k == "token":
            _stream_token(console, d["text"])
        elif k == "generate_end":
            console.print()  # newline after the live stream
        elif k == "diagnose_end":
            console.print(f"    [blue]🔍 diagnosis:[/] [dim]{d['text'][:200]}[/]")
        elif k == "edit":
            console.print(f"    [green]✎ {d['path']}[/]")
        elif k == "auto_install":
            console.print(f"    [magenta]📦 auto-installing: {' '.join(d['packages'])}[/]")
        elif k == "check_passed":
            console.print("    [green]✓ checks pass[/]")
        elif k == "check_failed":
            console.print(f"    [yellow]✗ {d['stage']}[/]")
        elif k == "ctx_backoff":
            console.print(f"    [dim]context lowered to {d['num_ctx']} (VRAM headroom)[/]")
        elif k == "finished" and d.get("ok") and d.get("usage"):
            console.print(f"\n[bold]How to use it[/]\n{d['usage']}")
    ev.subscribe(on)

    from agent.config import load_config
    llm, model = _prep_model(console)
    console.print(f"[bold]agent code[/] · goal: {goal}  [dim]· {model}[/]")
    try:
        out = build(root, goal, config=load_config(), events=ev)
    except KeyboardInterrupt:
        console.print("\n[yellow]■ stopped[/]")
        return 130
    finally:
        _release_model(llm, model, console)
    console.print()
    if out.ok:
        console.print(f"[bold green]✓ built[/] in {out.attempts} round(s) · branch [cyan]{out.branch}[/]")
        if out.diff:
            console.print(Syntax(out.diff, "diff", theme="ansi_dark", word_wrap=True))
        console.print(f"[dim]bring it in:[/] git merge {out.branch}")
        return 0
    console.print(f"[bold red]✗ {out.stage}[/] after {out.attempts} round(s). {out.error[:500]}")
    if out.branch:
        console.print(f"[dim]partial work saved to branch[/] [cyan]{out.branch}[/] "
                      f"[dim](git checkout {out.branch} to inspect / cherry-pick)[/]")
    if out.diff:
        console.print(out.diff[:1500])
    return 1


def run_code(task: str, files: list[str], test: str | None) -> int:
    console = Console()
    root = _ensure_repo(console)
    if root is None:
        return 1
    if not files:
        console.print("[red]Give at least one file to edit with -f/--file.[/red]")
        return 1
    for f in files:
        if not (root / f).exists():
            console.print(f"[red]No such file:[/red] {f}")
            return 1

    ev = Events()

    def on(e):
        k, d = e.kind, e.data
        if k == "attempt":
            console.print(f"[cyan]▸ attempt {d['n']}/{d['of']}[/] [dim]{d['model']}[/]")
        elif k == "diagnose_end":
            console.print(f"  [blue]🔍 diagnosis:[/] [dim]{d['text'][:200]}[/]")
        elif k == "generate_start":
            console.print("  [dim]┄ writing (live):[/]")
        elif k == "token":
            _stream_token(console, d["text"])
        elif k == "generate_end":
            console.print()
        elif k == "check":
            console.print(f"    [dim]check: {d['stage']}…[/]")
        elif k == "check_failed":
            console.print(f"    [yellow]✗ {d['stage']}[/]")
        elif k == "check_passed":
            console.print("    [green]✓ tests pass[/]")
        elif k == "guard":
            mark = "[green]ok[/]" if d["ok"] else "[yellow]blocked[/]"
            console.print(f"    guard · {d['name']}: {mark}")
        elif k == "guard_reject":
            console.print(f"    [yellow]✗ rejected edit to test file(s): {d['paths']}[/]")
        elif k == "attempt_failed":
            console.print(f"  [yellow]✗ attempt {d['n']} failed ({d['stage']})[/]")
        elif k == "stall":
            console.print("  [magenta]stalled — identical output twice[/]")
        elif k == "escalate":
            console.print(f"  [magenta]⚡ escalating to {d['to']}[/]")

    ev.subscribe(on)

    from agent.config import load_config
    llm, model = _prep_model(console)
    console.print(f"[bold]agent code[/] · {task}  [dim]· {model}[/]")
    try:
        outcome = repair.run(root, task, files, config=load_config(), test_target=test, events=ev)
    except KeyboardInterrupt:
        console.print("\n[yellow]■ stopped[/]")
        return 130
    finally:
        _release_model(llm, model, console)
    console.print()

    if outcome.ok and outcome.stage == "already-passing":
        console.print("[green]Nothing to do — the checks already pass.[/]")
        return 0
    if outcome.ok:
        extra = f", {outcome.malformed_count} re-ask(s)" if outcome.malformed_count else ""
        console.print(f"[bold green]✓ fixed[/] in {outcome.attempts} attempt(s){extra}")
        console.print(f"[dim]committed to branch[/] [cyan]{outcome.branch}[/]")
        if outcome.diff:
            console.print(Syntax(outcome.diff, "diff", theme="ansi_dark", word_wrap=True))
        console.print(
            f"[dim]bring it into your tree:[/] git merge {outcome.branch}  "
            f"[dim]or[/]  git cherry-pick {outcome.branch}"
        )
        return 0

    console.print(f"[bold red]✗ stuck[/] after {outcome.attempts} attempt(s). Full trail:")
    for line in outcome.trail:
        console.print(f"  [dim]{line}[/]")
    console.print(f"\n[dim]last error:[/]\n{outcome.error[:1000]}")
    return 1
