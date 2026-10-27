"""Updating plugins that were installed from a git URL (`nightage plugin install <url>`), and the trust model around it.

An install record `.source.json` sits in the plugin's folder: {"type": "git", "url", "ref", "branch", "commit", "installed_at"}.
It is written by `plugin_host.install`; plugins installed from a local path have none ("no update source"). A plugin's manifest
may also name `repository` (or `update_url`) so a copied plugin can still be updated.

Flow of `nightage plugin update <name>` (agent/plugin_cli.py calls `run_plugin_update`):
  1. `git clone --depth 50` of the source into a TEMPORARY folder (nothing in your plugin folder changes yet),
  2. show the changelog (`git log --oneline old..new`) and a FILE-LEVEL SUMMARY (added / removed / changed files with line counts),
  3. ask; on yes: the old folder is moved to `<plugin folder>/.backups/`, the new files take its place, the record is rewritten,
  4. updated code is NEW code: the plugin is switched OFF and only the normal enable flow (the new fingerprint is shown, the warning
     is repeated, you must confirm) turns it on again. `--yes` skips the update question only, never the enable question.

Everything that runs git goes through an injectable `run`, so the tests use throw-away local repositories.
Guide: docs/UPDATING.md.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent import plugin_host
from agent.plugin_api import PluginError

SOURCE_FILE = ".source.json"
BACKUP_DIR = ".backups"
IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache"}
Runner = Callable[..., "tuple[int, str]"]


def _runner(run: Runner | None) -> Runner:
    if run is not None:
        return run
    from agent.update import default_runner

    return default_runner


# ----------------------------------------------------------------------------------------------- the install record


def read_git_head(repo: Path) -> tuple[str, str]:
    """(branch, commit) of a git checkout read straight from .git (no subprocess); ('', '') when unreadable."""
    try:
        gitdir = repo / ".git"
        if gitdir.is_file():                                       # worktree / submodule pointer
            gitdir = repo / gitdir.read_text().split(":", 1)[1].strip()
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return "", head if re.fullmatch(r"[0-9a-f]{40}", head) else ""
        ref = head.split(":", 1)[1].strip()
        branch = ref.removeprefix("refs/heads/")
        f = gitdir / ref
        if f.is_file():
            return branch, f.read_text().strip()
        packed = gitdir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return branch, line.split()[0]
    except (OSError, IndexError):
        pass
    return "", ""


def write_source(dest: Path, *, url: str, ref: str = "", repo: Path | None = None, commit: str = "", branch: str = "") -> dict[str, Any]:
    """Write `.source.json` into the installed plugin folder `dest` (best effort; never raises)."""
    b, c = read_git_head(repo) if repo else ("", "")
    rec = {"type": "git", "url": url, "ref": ref, "branch": branch or b, "commit": commit or c,
           "installed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        (dest / SOURCE_FILE).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    except OSError:
        pass
    return rec


def read_source(folder: Path) -> dict[str, Any] | None:
    try:
        d = json.loads((folder / SOURCE_FILE).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and d.get("url") else None
    except (OSError, ValueError):
        return None


def manifest_source(folder: Path) -> str:
    """`repository` / `update_url` from plugin.toml ('' when absent)."""
    try:
        raw = tomllib.loads((folder / plugin_host.MANIFEST).read_text(encoding="utf-8"))
        return str(raw.get("update_url") or raw.get("repository") or "")
    except Exception:  # noqa: BLE001
        return ""


def installed_commits() -> dict[str, str]:
    out = {}
    for m in plugin_host.discover():
        rec = read_source(m.path) if m.path.is_dir() else None
        if rec:
            out[m.id] = str(rec.get("commit") or "")
    return out


# ----------------------------------------------------------------------------------------------- checking (cheap, no clone)


def remote_commit(url: str, ref: str, run: Runner) -> tuple[str, str]:
    """(commit, error) of what `url` points at for `ref` (default HEAD), via `git ls-remote`."""
    patterns = [f"refs/heads/{ref}", f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"] if ref else ["HEAD"]
    rc, out = run(["git", "ls-remote", url, *patterns], timeout=10)
    if rc != 0:
        return "", f"offline ({(out.strip().splitlines() or ['no answer'])[-1][:60]})"
    peeled, plain = "", ""
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            if parts[1].endswith("^{}"):
                peeled = peeled or parts[0]
            else:
                plain = plain or parts[0]
    return (peeled or plain), ("" if (peeled or plain) else f"{ref or 'HEAD'} not found at the source")


def check_all(config: dict | None, *, run: Runner | None = None) -> list[Any]:
    from agent.update import PluginUpdate

    run = _runner(run)
    out = []
    for m in plugin_host.discover():
        pu = PluginUpdate(name=m.id, current=m.version)
        rec = read_source(m.path) if m.path.is_dir() else None
        if not rec:
            src = manifest_source(m.path) if m.path.is_dir() else ""
            pu.note = (f"update source {src} named in plugin.toml, but the installed commit was not recorded; "
                       "`nightage plugin update` can still compare") if src else "no update source (installed from a local path)"
            pu.url = src
            out.append(pu)
            continue
        pu.url, pu.ref, pu.installed_commit = str(rec["url"]), str(rec.get("ref") or ""), str(rec.get("commit") or "")
        commit, err = remote_commit(pu.url, pu.ref, run)
        if err:
            pu.note = err
        elif not pu.installed_commit:
            pu.remote_commit, pu.note = commit, "installed commit unknown"
        else:
            pu.remote_commit, pu.latest = commit, commit[:8]
            pu.available = commit != pu.installed_commit
        out.append(pu)
    return out


# ----------------------------------------------------------------------------------------------- planning and applying


@dataclass
class FileChange:
    status: str            # added | removed | changed
    path: str
    plus: int = 0
    minus: int = 0
    binary: bool = False


@dataclass
class Plan:
    name: str
    url: str
    ref: str
    old_commit: str
    new_commit: str
    old_version: str
    new_version: str
    changelog: list[str] = field(default_factory=list)
    changes: list[FileChange] = field(default_factory=list)
    tmp: Path | None = None
    repo: Path | None = None
    problem: str = ""

    @property
    def unchanged(self) -> bool:
        return not self.changes and self.old_commit == self.new_commit

    def summary_lines(self, width: int = 100) -> list[str]:
        out = []
        for c in self.changes:
            tag = {"added": "+", "removed": "-", "changed": "~"}[c.status]
            tail = "(binary)" if c.binary else f"+{c.plus} -{c.minus}" if c.status == "changed" else f"{c.plus or c.minus} line(s)"
            out.append(f"  {tag} {c.path}  {tail}")
        return out

    def cleanup(self) -> None:
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None


def _files(root: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part in IGNORE_DIRS for part in p.relative_to(root).parts) and p.name != SOURCE_FILE \
                and not p.name.endswith(".pyc"):
            out[p.relative_to(root).as_posix()] = p
    return out


def _lines(p: Path) -> list[str] | None:
    try:
        return p.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return None


def diff_trees(old: Path, new: Path) -> list[FileChange]:
    """File-level summary of what an update changes: added / removed / changed files and how many lines."""
    a, b = _files(old), _files(new)
    out = []
    for rel in sorted(set(a) | set(b)):
        if rel not in b:
            la = _lines(a[rel])
            out.append(FileChange("removed", rel, minus=len(la or []), binary=la is None))
        elif rel not in a:
            lb = _lines(b[rel])
            out.append(FileChange("added", rel, plus=len(lb or []), binary=lb is None))
        elif a[rel].read_bytes() != b[rel].read_bytes():
            la, lb = _lines(a[rel]), _lines(b[rel])
            if la is None or lb is None:
                out.append(FileChange("changed", rel, binary=True))
                continue
            plus = minus = 0
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes():
                if tag in ("replace", "delete"):
                    minus += i2 - i1
                if tag in ("replace", "insert"):
                    plus += j2 - j1
            out.append(FileChange("changed", rel, plus=plus, minus=minus))
    return out


def source_for(name: str) -> tuple[Path, dict[str, Any]]:
    """The installed plugin folder and its source record (from .source.json, else the manifest's repository / update_url)."""
    m = plugin_host.find(name)
    if m is None or not m.path.is_dir():
        raise PluginError(f"no installed plugin folder called {name!r} (see `nightage plugin list`)")
    rec = read_source(m.path)
    if rec:
        return m.path, rec
    url = manifest_source(m.path)
    if url:
        return m.path, {"type": "git", "url": url, "ref": "", "branch": "", "commit": ""}
    raise PluginError(f"{name} has no update source: it was installed from a local folder. Reinstall it from its git URL "
                      f"(`nightage plugin install <url> --force`) to make it updatable")


def plan_update(name: str, *, run: Runner | None = None) -> Plan:
    """Clone the source into a temp folder and work out what would change. The plugin folder is not touched."""
    run = _runner(run)
    folder, rec = source_for(name)
    old = plugin_host.read_manifest(folder)
    url, ref = str(rec["url"]), str(rec.get("ref") or "")
    tmp = Path(tempfile.mkdtemp(prefix="plugin-update-"))
    repo = tmp / "repo"
    argv = ["git", "clone", "--depth", "50", *(["--branch", ref] if ref else []), url, str(repo)]
    plan = Plan(name, url, ref, str(rec.get("commit") or ""), "", old.version, "", tmp=tmp, repo=repo)
    rc, out = run(argv, timeout=180)
    if rc != 0:
        plan.problem = "git clone failed: " + ((out.strip().splitlines() or ["unknown error"])[-1][:160])
        return plan
    rc, head = run(["git", "rev-parse", "HEAD"], cwd=repo, timeout=10)
    plan.new_commit = head.strip() if rc == 0 else read_git_head(repo)[1]
    new = plugin_host.read_manifest(repo)
    if not (repo / plugin_host.MANIFEST).is_file():
        plan.problem = f"the source no longer has a {plugin_host.MANIFEST}: it is not a nightage plugin any more"
        return plan
    if new.error:
        plan.problem = new.error
        return plan
    if new.id != old.id:
        plan.problem = f"the source now calls itself {new.id!r}, not {old.id!r}; refusing to replace one plugin with another"
        return plan
    why = plugin_host.check_compat(new)
    if why:
        plan.problem = f"the new version cannot run here: {why}"
        return plan
    plan.new_version = new.version
    if plan.old_commit:
        rc, log = run(["git", "log", "--oneline", f"{plan.old_commit}..HEAD"], cwd=repo, timeout=10)
        if rc == 0:
            plan.changelog = log.strip().splitlines()
        else:
            rc, log = run(["git", "log", "--oneline", "-10"], cwd=repo, timeout=10)
            plan.changelog = ["(the installed commit is not in the last 50 commits; latest ones:)", *log.strip().splitlines()] if rc == 0 else []
    else:
        rc, log = run(["git", "log", "--oneline", "-10"], cwd=repo, timeout=10)
        plan.changelog = log.strip().splitlines() if rc == 0 else []
    plan.changes = diff_trees(folder, repo)
    return plan


def apply_plan(plan: Plan) -> tuple[Path, bool]:
    """Back up the old folder, put the new files in, rewrite the record, switch the plugin OFF. Returns (backup, was_enabled)."""
    folder, _rec = source_for(plan.name)
    assert plan.repo is not None
    was_enabled = plugin_host.is_enabled(plan.name)
    backups = plugin_host.plugin_dir() / BACKUP_DIR
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = backups / f"{plan.name}-{(plan.old_commit or 'unknown')[:8]}-{stamp}"
    shutil.move(str(folder), str(backup))
    try:
        shutil.copytree(plan.repo, folder, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        shutil.move(str(backup), str(folder))               # put the old plugin back; nothing half-installed
        raise
    write_source(folder, url=plan.url, ref=plan.ref, commit=plan.new_commit,
                 branch=(read_git_head(plan.repo)[0] if not plan.ref else ""))
    plugin_host.set_enabled(plan.name, False)               # new code is not trusted until the user confirms it again
    return backup, was_enabled


# ----------------------------------------------------------------------------------------------- the command


def _default_enable(name: str) -> bool:
    """The normal `plugin enable` flow (shows the new fingerprint and the warning, asks). True when it ended enabled."""
    import typer

    from agent import plugin_cli

    try:
        plugin_cli.enable(name, yes=False)
    except typer.Exit:
        return False
    return plugin_host.is_enabled(name)


def run_plugin_update(names: list[str] | None = None, *, all_plugins: bool = False, check_only: bool = False, yes: bool = False,
                      console: Any = None, confirm: Callable[[str], bool] | None = None, run: Runner | None = None,
                      config: dict | None = None, enable_fn: Callable[[str], bool] | None = None) -> int:
    """`nightage plugin update`. Returns the exit code (0 fine, 1 something failed or was refused)."""
    from rich.console import Console
    from rich.markup import escape

    con = console or Console()
    run = _runner(run)
    confirm = confirm or _ask
    enable_fn = enable_fn or _default_enable
    updatable = [m for m in plugin_host.discover()]
    if not names and not all_plugins and not check_only:
        con.print("Name a plugin (`nightage plugin update my-plugin`), or use --all, or --check to just look.")
        return 1
    if names:
        missing = [n for n in names if plugin_host.find(n) is None]
        if missing:
            con.print(f"[red]No plugin called {escape(', '.join(missing))}.[/red] See `nightage plugin list`.")
            return 1
        wanted = [plugin_host.find(n).id for n in names]                    # type: ignore[union-attr]
    else:
        wanted = [m.id for m in updatable]
    if not wanted:
        con.print("No plugins installed.")
        return 0
    from agent.update import PluginUpdate

    status = {p.name: p for p in check_all(config, run=run)}
    rc = 0
    updated = 0
    for name in wanted:
        pu: PluginUpdate = status.get(name) or PluginUpdate(name=name)
        if check_only:
            if pu.available:
                con.print(f"[yellow]{escape(name)}[/yellow]: update available ({pu.installed_commit[:8]} -> {pu.remote_commit[:8]}) from {escape(pu.url)}")
            else:
                con.print(f"{escape(name)}: {escape(pu.note or 'up to date')}")
            continue
        if not pu.available and (pu.note.startswith("no update source") or pu.note.startswith("offline")):
            con.print(f"{escape(name)}: {escape(pu.note)}")
            rc = rc or (1 if names else 0)
            continue
        if not pu.available and not pu.note and pu.installed_commit:
            con.print(f"{escape(name)}: already up to date ({pu.installed_commit[:8]}).")
            continue
        plan = plan_update(name, run=run)
        try:
            if plan.problem:
                con.print(f"[red]{escape(name)}: cannot update:[/red] {escape(plan.problem)}")
                rc = 1
                continue
            if plan.unchanged:
                con.print(f"{escape(name)}: already up to date.")
                continue
            con.print(f"[bold]{escape(name)}[/bold]  {escape(plan.old_version)} -> {escape(plan.new_version)}   "
                      f"{plan.old_commit[:8] or '?'} -> {plan.new_commit[:8]}   from {escape(plan.url)}")
            if plan.changelog:
                con.print("Changes:")
                for line in plan.changelog[:20]:
                    con.print("  " + escape(line))
                if len(plan.changelog) > 20:
                    con.print(f"  ... and {len(plan.changelog) - 20} more")
            con.print("Files:" if plan.changes else "Files: no differences in the plugin files")
            for line in plan.summary_lines()[:60]:
                con.print(escape(line))
            if not yes and not confirm(f"Replace {name} with this version?"):
                con.print("Not updated.")
                continue
            backup, was_enabled = apply_plan(plan)
            updated += 1
            con.print(f"Updated {escape(name)}. The old files are kept in {escape(str(backup))}.")
            m = plugin_host.find(name)
            fp = plugin_host.fingerprint(m) if m else "?"
            con.print(f"[yellow]The plugin is now switched OFF: updated code is new code.[/yellow] New files fingerprint: {fp}.")
            if was_enabled:
                if yes:
                    con.print(f"With --yes the plugin stays off. Look at the new code, then: nightage plugin enable {escape(name)}")
                elif enable_fn(name):
                    con.print(f"{escape(name)} is enabled again (loads the next time nightage starts).")
                else:
                    con.print(f"{escape(name)} stays off. Enable it when you have looked at the code: nightage plugin enable {escape(name)}")
            else:
                con.print(f"It was not enabled before and stays that way (nightage plugin enable {escape(name)}).")
        finally:
            plan.cleanup()
    if check_only:
        return 0
    if updated:
        con.print("Restart nightage for plugin changes to take effect.")
    return rc


def _ask(question: str) -> bool:
    import typer

    return bool(typer.confirm(question, default=False))
