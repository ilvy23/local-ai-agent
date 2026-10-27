"""`nightage update [--check] [--yes]`: show what is new, show exactly what will run, ask, update, say 'restart'.

    from agent.update_cli import run_update, register
    code = run_update(check_only=True)            # exit code
    register(app)                                  # adds the `update` command to a typer app (main.py wiring)

How each install kind updates (user-level only; never sudo; every command is shown before it runs):
  git checkout   `git fetch`, then `git pull --ff-only` (refused on local changes to tracked files, on a diverged history and on a
                 detached HEAD), then `uv sync` (or `pip install -e .`) when pyproject.toml / uv.lock changed
  uv tool        `uv tool upgrade agent`
  pip            `python -m pip install --upgrade git+<repo>@<tag>`
  editable copy / unknown   it says what to do by hand (there is nothing safe to automate)
Afterwards a small doctor subset runs in a fresh process (new code) and you are told to restart. Guide: docs/UPDATING.md.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import brand, update
from agent.update import InstallInfo, UpdateReport

FORBIDDEN = frozenset({"sudo", "su", "doas", "pkexec", "sh", "bash", "zsh", "dash", "env", "eval", "rm", "systemctl"})
DEP_FILES = ("pyproject.toml", "uv.lock")
POST_CHECK_GROUPS = "python,packages,config,database"


@dataclass
class Step:
    argv: list[str]
    why: str = ""
    cwd: Path | None = None

    def shown(self) -> str:
        import shlex

        return " ".join(shlex.quote(a) for a in self.argv)


@dataclass
class Plan:
    kind: str
    steps: list[Step] = field(default_factory=list)
    changelog: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    refusal: str = ""
    manual: str = ""
    up_to_date: bool = False


class Refused(Exception):
    pass


def _check_argv(argv: list[str]) -> None:
    prog = os.path.basename(argv[0]) if argv else ""
    if not prog or prog in FORBIDDEN:
        raise Refused(f"{prog or '(empty)'} is never run by the updater")
    if prog not in ("git", "uv") and not prog.startswith("python"):
        raise Refused(f"{prog} is not an updater command")


def _lines(text: str, limit: int) -> list[str]:
    ls = [ln.rstrip() for ln in text.strip().splitlines()]
    return ls[:limit] + ([f"... ({len(ls) - limit} more lines)"] if len(ls) > limit else [])


# ----------------------------------------------------------------------------------------------- planning


def plan_git(rep: UpdateReport, install: InstallInfo, run: update.Runner, which: Callable[[str], str | None]) -> Plan:
    root = Path(install.root)
    p = Plan("git")
    if not install.has_remote:
        p.refusal = "This checkout has no git remote (`git remote add origin <url>`), so there is nowhere to update from."
        return p
    if not install.branch:
        p.refusal = ("HEAD is detached (you are on a tag or a commit, not a branch). Switch to a branch first, e.g. "
                     "`git switch main`, then run the update again.")
        return p
    rc, out = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, timeout=10)
    if rc != 0:
        p.refusal = "`git status` failed: " + (out.strip().splitlines() or ["unknown error"])[-1][:120]
        return p
    if out.strip():
        files = [ln[3:] for ln in out.splitlines() if ln.strip()][:6]
        p.refusal = ("You have uncommitted changes to tracked files (" + ", ".join(files) + "). Updating could overwrite them: "
                     "commit them (`git commit -a`) or put them aside (`git stash`), then run the update again.")
        return p
    rc, up = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=root, timeout=5)
    remote, branch = "origin", install.branch
    if rc == 0 and "/" in up.strip():
        remote, _, branch = up.strip().partition("/")
    rc, out = run(["git", "fetch", remote, branch], cwd=root, timeout=60)              # changes only remote-tracking refs
    if rc != 0:
        p.refusal = "Could not reach the remote (offline?): " + (out.strip().splitlines() or ["no answer"])[-1][:120]
        return p
    ref = "FETCH_HEAD"
    rc, cnt = run(["git", "rev-list", "--left-right", "--count", f"HEAD...{ref}"], cwd=root, timeout=10)
    try:
        ahead, behind = (int(x) for x in cnt.split())
    except ValueError:
        p.refusal = "Could not compare your checkout with the remote."
        return p
    if behind == 0:
        p.up_to_date = True
        p.notes.append(f"Your checkout already has everything on {remote}/{branch}" + (f" (and is {ahead} commit(s) ahead)" if ahead else "") + ".")
        return p
    if ahead:
        p.refusal = (f"Your branch and {remote}/{branch} have diverged ({ahead} commit(s) of yours, {behind} new upstream). A fast-forward "
                     "is impossible. Rebase or merge by hand (`git pull --rebase`), or ask for help; nothing was changed.")
        return p
    rc, log = run(["git", "log", "--oneline", "--no-decorate", f"HEAD..{ref}"], cwd=root, timeout=10)
    p.changelog = _lines(log, 30) if rc == 0 else []
    rc, names = run(["git", "diff", "--name-only", "HEAD", ref], cwd=root, timeout=10)
    changed = set(names.split()) if rc == 0 else set()
    p.steps.append(Step(["git", "pull", "--ff-only", remote, branch], f"fast-forward to the {behind} new commit(s)", root))
    if changed & set(DEP_FILES) or any(c.startswith("requirements") for c in changed):
        if which("uv") and (root / "uv.lock").exists():
            p.steps.append(Step(["uv", "sync"], "dependencies changed (pyproject.toml / uv.lock)", root))
        elif install.editable:
            p.steps.append(Step([sys.executable, "-m", "pip", "install", "-e", str(root)], "dependencies changed", root))
        else:
            p.notes.append("pyproject.toml / uv.lock changed: refresh the dependencies yourself (`uv sync` or `pip install -e .`).")
    return p


def plan_framework(rep: UpdateReport, install: InstallInfo, run: update.Runner, config: dict | None = None,
                   which: Callable[[str], str | None] | None = None) -> Plan:
    import shutil

    which = which or shutil.which
    fw = rep.framework
    if install.kind == "git":
        p = plan_git(rep, install, run, which)
        if fw.source == "release" and fw.notes:
            p.notes.insert(0, f"Release {fw.tag or fw.latest} notes: " + (fw.notes_url or ""))
            p.changelog = p.changelog or _lines(fw.notes, 25)
        return p
    p = Plan(install.kind)
    p.changelog = _lines(fw.notes, 25) if fw.notes else []
    if install.kind == "uv-tool":
        if not which("uv"):
            p.manual = "uv is not on PATH; run `uv tool upgrade agent` where uv is installed."
        else:
            p.steps.append(Step(["uv", "tool", "upgrade", update.DIST_NAME], "upgrade the uv-managed tool"))
        return p
    if install.kind == "pip":
        slug = update.repo_slug(None, install) or update.parse_slug(fw.repo)
        base = install.vcs_url.split("@")[0] if install.vcs_url else (f"git+https://github.com/{slug[0]}/{slug[1]}.git" if slug else "")
        if base and not base.startswith("git+"):
            base = "git+" + base
        tag = fw.tag or (f"v{fw.latest}" if fw.latest else "")
        if not base or not tag:
            p.manual = "No release tag is known; reinstall from the project page with `pip install --upgrade git+<repo-url>@<tag>`."
        else:
            p.steps.append(Step([sys.executable, "-m", "pip", "install", "--upgrade", f"{base}@{tag}"], f"install release {tag}"))
        return p
    if install.kind == "editable":
        p.manual = ("This is an editable install of a folder that is not a git checkout, so there is nothing to pull. Get the new "
                    "release into the folder, then run `pip install -e .` there.")
        return p
    p.manual = "nightage cannot tell how it was installed. Reinstall it the way you installed it the first time (see the README)."
    return p


# ----------------------------------------------------------------------------------------------- the command


def _post_check(run: update.Runner, console: Any) -> None:
    rc, out = run([sys.executable, "-m", "agent.doctor", "--only", POST_CHECK_GROUPS, "--no-web"], timeout=90)
    console.print("[bold]Quick check of the updated install[/bold]")
    for ln in (out.strip().splitlines() or ["(no output)"])[-25:]:
        console.print("  " + _esc(ln))
    if rc not in (0, 2):
        console.print("[yellow]The quick check reported problems: run `nightage doctor` for details.[/yellow]")


def _esc(s: str) -> str:
    from rich.markup import escape

    return escape(s)


def run_update(*, check_only: bool = False, yes: bool = False, config: dict | None = None, console: Any = None,
               confirm: Callable[[str], bool] | None = None, run: update.Runner | None = None, http: update.HttpGet | None = None,
               install: InstallInfo | None = None, root: Path | None = None, which: Callable[[str], str | None] | None = None,
               post_check: Callable[[], None] | None = None, cache: Path | None = None) -> int:
    """The command. 0 = fine / up to date / updated; 1 = refused, failed or cancelled."""
    from rich.console import Console

    con = console or Console()
    run = run or update.default_runner
    if confirm is None:
        def confirm(q: str) -> bool:
            import typer

            return bool(typer.confirm(q, default=False))
    install = install or update.detect_install(root=root, run=run)
    rep = update.check_for_updates(config, force=True, http=http, run=run, install=install, cache=cache, root=root)
    fw = rep.framework
    con.print(f"{brand.NAME} {_esc(fw.current)}   install: {_esc(install.describe())}")
    if fw.no_source and not fw.available and install.kind != "git":
        con.print("No update source: set `update.repo: owner/name` in config.yaml (the project page is not configured yet).")
    elif rep.offline:
        con.print("Could not reach GitHub (offline?). Nothing was changed.")
    if fw.latest:
        con.print(f"Latest release: {_esc(fw.latest)}" + (f"  {_esc(fw.notes_url)}" if fw.notes_url else ""))
    elif fw.repo and not rep.offline:
        con.print(f"{_esc(fw.repo)} has no releases yet.")
    if rep.plugin_updates:
        con.print(f"{len(rep.plugin_updates)} plugin update(s): " + ", ".join(p.name for p in rep.plugin_updates) +
                  f"  - `{brand.CMD} plugin update <name>`")

    if check_only:
        if fw.available:
            n = f"{fw.latest} (you have {fw.current})" if fw.source == "release" else f"{fw.behind or 'new'} commit(s) on {fw.branch}"
            con.print(f"[bold]Update available:[/bold] {_esc(n)}. Run `{brand.CMD} update` to install it.")
        elif not rep.offline:
            con.print("You are up to date.")
        return 0

    plan = plan_framework(rep, install, run, config, which)
    if plan.refusal:
        con.print(f"[red]Not updating:[/red] {_esc(plan.refusal)}")
        return 1
    if plan.up_to_date:
        for n in plan.notes:
            con.print(_esc(n))
        if fw.available and fw.source == "release":
            con.print(f"Release {_esc(fw.latest)} exists, but your branch has no newer commits to pull yet.")
        else:
            con.print("You are up to date.")
        return 0
    if not plan.steps and not plan.manual:
        con.print("You are up to date.")
        return 0
    if plan.changelog:
        con.print("[bold]What is new[/bold]")
        for ln in plan.changelog:
            con.print("  " + _esc(ln))
    for n in plan.notes:
        con.print(_esc(n))
    if plan.manual:
        con.print(f"[yellow]Manual step needed:[/yellow] {_esc(plan.manual)}")
        return 1
    con.print("[bold]This will run (as you, no sudo):[/bold]")
    for s in plan.steps:
        _check_argv(s.argv)
        con.print(f"  $ {_esc(s.shown())}" + (f"    # {s.why}" if s.why else ""))
    if not yes and not confirm("Update now?"):
        con.print("Not updated.")
        return 1
    for s in plan.steps:
        _check_argv(s.argv)
        rc, out = run(s.argv, cwd=s.cwd, timeout=900)
        if rc != 0:
            con.print(f"[red]Failed:[/red] {_esc(s.shown())}")
            for ln in _lines(out, 12):
                con.print("  " + _esc(ln))
            con.print("Your install was left as it is. Fix the problem above and run `nightage update` again.")
            return 1
    con.print("[green]Updated.[/green]")
    (post_check or (lambda: _post_check(run, con)))()
    con.print(f"[bold]Restart {brand.CMD}[/bold] to use the new version (anything running now keeps the old code).")
    return 0


def update_command(check: bool = False, yes: bool = False) -> None:
    import typer

    from agent.config import load_config

    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001
        cfg = {}
    raise typer.Exit(run_update(check_only=check, yes=yes, config=cfg))


def register(app: Any) -> None:
    """Add `nightage update [--check] [--yes]` to a typer app."""
    import typer

    @app.command("update")
    def update_(check: bool = typer.Option(False, "--check", help="Only look for updates and print what is new."),
                yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before updating (for scripts).")) -> None:
        """Update nightage: shows what is new and exactly what will run, asks, updates, then tells you to restart."""
        update_command(check, yes)
