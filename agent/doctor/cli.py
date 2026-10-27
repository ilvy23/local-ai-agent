"""`nightage doctor`: options, grouped output, --fix, --json, --list, --report.

    from agent.doctor.cli import run, doctor_command
    code = run(config, only="ollama,gpu", deep=True)          # -> exit code (0 ok, 1 any problem, 2 warnings with strict)
    app.command("doctor")(doctor_command)                      # the typer command with every option (main.py wiring)

Exit codes: 0 everything fine (warnings allowed), 1 at least one problem, 2 only warnings and `--strict` was given,
3 `--report` refused because the self-check found something that was not anonymised.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from agent.doctor import core
from agent.doctor import report as rpt
from agent.doctor.anonymise import Identity, anonymise, leak_check
from agent.doctor.core import FAIL, INFO, WARN, Context
from agent.doctor.probes import Probes
from agent.ui import style


def _split(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if not value:
        return []
    items = [value] if isinstance(value, str) else list(value)
    return [t.strip().lower() for chunk in items for t in str(chunk).split(",") if t.strip()]


def build_meta(ctx: Context) -> dict[str, Any]:
    """Facts for the report header. Never raises."""
    m: dict[str, Any] = {}
    try:
        from agent import update

        m["version"] = update.current_version()
        m["install"] = update.detect_install().describe()
    except Exception:  # noqa: BLE001
        pass
    try:
        si, pi = ctx.sysinfo, ctx.p.platform_info()
        m["python"] = f"{pi.get('implementation', 'Python')} {pi.get('python', '?')}"
        m["system"] = si.summary() + (" (WSL)" if "wsl" in ctx.tags else "") + (f", {si.distro_name}" if si.distro_name else "")
        m["machine"] = f"{pi.get('system', '')} {pi.get('release', '')} {pi.get('machine', '')}".strip()
        m["cpu"] = f"{si.cpu_count} cores"
        m["ram"] = f"{si.ram_total_mb // 1024} GB total, {si.ram_free_mb // 1024} GB free" if si.ram_total_mb else "unknown"
        m["gpu"] = ", ".join(f"{g.name}" for g in ctx.gpus) or "none"
        m["profile"] = str(ctx.config.get("profile") or "default")
    except Exception:  # noqa: BLE001
        pass
    try:
        o = ctx.ollama
        m["ollama"] = (f"{o.version or '?'} ({'remote' if o.url and not ('localhost' in o.url or '127.0.0.1' in o.url) else 'local'})"
                       if o.reachable else "not reachable")
        t = ctx.p.terminal()
        m["terminal"] = f"{t.get('width')} columns, {'colour' if t.get('color') else 'no colour'}, {'unicode' if t.get('unicode') else 'ASCII'}"
    except Exception:  # noqa: BLE001
        pass
    return m


def collect(config: dict | None = None, *, probes: Any = None, only: Any = (), skip: Any = (), deep: bool = False,
            no_web: bool = False, verbose: bool = False) -> tuple[rpt.Report, Context, list[core.CheckSpec]]:
    ctx = Context(config, probes or Probes(), deep=deep, no_web=no_web, verbose=verbose)
    specs = core.select(ctx, only=_split(only), skip_groups=_split(skip))
    results = core.run_checks(ctx, specs)
    for mod, err in core.LOAD_ERRORS:
        results.append(core.warn(f"could not load {mod}: {err}", id="doctor.load", group="system", title="Doctor module"))
    rep = rpt.Report(results, build_meta(ctx))
    try:
        from agent import update

        rep.update_note = update.notice_line(ctx.config) or ""
    except Exception:  # noqa: BLE001
        pass
    return rep, ctx, specs


def default_confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        import typer

        return bool(typer.confirm(question, default=False))
    except Exception:  # noqa: BLE001 - Abort, EOF, ...
        return False


def apply_fixes(rep: rpt.Report, ctx: Context, specs: list[core.CheckSpec], console: Any, confirm: Callable[[str], bool]) -> int:
    """Offer every fixable finding, one by one, default NO. Returns how many ran."""
    from rich.markup import escape

    todo = [r for r in rep.results if r.fix is not None and r.status in (WARN, FAIL, INFO)]
    if not todo:
        console.print("Nothing here has a one-command fix.")
        return 0
    ran, touched = 0, set()
    for r in todo:
        console.print(f"\n[bold]{escape(r.title)}[/bold]: {escape(r.detail)}")
        console.print(f"  command: [bold]{escape(r.command or '(see hint)')}[/bold]   (user-level only; never sudo)")
        if not confirm("Run it?"):
            console.print("  skipped.")
            continue
        try:
            msg = r.fix()           # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            msg = f"failed: {type(exc).__name__}: {exc}"
        ran += 1
        touched.add(r.spec_id)
        console.print(f"  {escape(str(msg))}")
    if touched:
        ctx._cache.clear()
        again = core.run_checks(ctx, [s for s in specs if s.id in touched])
        ids = {x.id for x in again}
        rep.results = [x for x in rep.results if x.id not in ids] + again
        rep.results.sort(key=lambda x: (core._group_rank(x.group)))
        console.print(f"\nAfter the fixes: {rep.summary()}")
    return ran


def run(config: dict | None = None, *, only: Any = (), skip: Any = (), deep: bool = False, fix: bool = False, as_json: bool = False,
        list_checks: bool = False, verbose: bool = False, no_web: bool = False, make_report: bool = False, strict: bool = False,
        no_hints: bool = False, probes: Any = None, console: Any = None, plain: bool | None = None, width: int | None = None,
        confirm: Callable[[str], bool] | None = None, out: Any = None, err: Any = None) -> int:
    """The whole command. `out`/`err` are text streams (default stdout/stderr); `console` overrides the rich console."""
    out = out or (console.file if console is not None and getattr(console, "file", None) else sys.stdout)
    err = err or sys.stderr
    console = console or style.make_console(width=width, plain=plain, file=out)

    if list_checks:
        ctx = Context(config, probes or Probes())
        specs = core.all_specs()
        here = {s.id for s in core.select(ctx, specs=specs)} | {s.id for s in specs if s.deep and s.applies(ctx.tags)}
        if as_json:
            rpt.render_list(specs, console, as_json=True)
        else:
            rpt.render_list([s for s in specs], console, plain=plain, width=width)
            console.print(f"\n{len(specs)} checks; deep ones need --deep. Checks for other systems are skipped automatically ({len(here)} apply here).")
        return 0

    if deep and not as_json and not make_report:
        console.print("Deep tests use your model (a few tokens each) and can take a minute.")
    rep, ctx, specs = collect(config, probes=probes, only=only, skip=skip, deep=deep, no_web=no_web, verbose=verbose)

    if make_report:
        ident = Identity.current()
        text = anonymise(rpt.to_markdown(rep), ident)
        problems = leak_check(text, ident)
        if problems:
            err.write("Report NOT printed: the anonymisation self-check still found " + ", ".join(sorted(set(problems))) +
                      ". Please report this as a bug (without pasting the output).\n")
            return 3
        out.write(text)
        err.write("Report anonymised and self-checked. Read it once before you post it.\n")
        return rep.exit_code(strict)

    if as_json:
        out.write(rpt.to_json(rep, hints=not no_hints))
        if fix:
            err.write("--fix is ignored with --json.\n")
        return rep.exit_code(strict)

    rpt.render(rep, console, plain=plain, width=width, verbose=verbose, hints=not no_hints)
    if fix:
        console.print("\n[bold]Fixes[/bold]" if not style.plain_mode(stream=out) and plain is not True else "\nFixes")
        apply_fixes(rep, ctx, specs, console, confirm or default_confirm)
    return rep.exit_code(strict)


def doctor_command(
    only: str | None = None,
    skip: str | None = None,
    deep: bool = False,
    fix: bool = False,
    json_out: bool = False,
    list_checks: bool = False,
    verbose: bool = False,
    no_web: bool = False,
    report: bool = False,
    strict: bool = False,
    no_fix: bool = False,
) -> None:
    """Check this machine and setup. See docs/DOCTOR.md. (The typer options are attached in `register`.)"""
    import typer

    from agent.config import load_config

    code = run(load_config(), only=only or "", skip=skip or "", deep=deep, fix=fix, as_json=json_out, list_checks=list_checks,
               verbose=verbose, no_web=no_web, make_report=report, strict=strict, no_hints=no_fix)
    raise typer.Exit(code)


def register(app: Any) -> None:
    """Add the `doctor` command with all its options to a typer app (replaces the old one in main.py)."""
    import typer

    @app.command("doctor")
    def doctor(
        only: str = typer.Option("", "--only", help="Only these groups or check ids, comma separated (see --list)."),
        skip: str = typer.Option("", "--skip", help="Leave these groups or check ids out, comma separated."),
        deep: bool = typer.Option(False, "--deep", help="Also run the tests that use the model (speed, embedding, tool call) and recommend a model."),
        fix: bool = typer.Option(False, "--fix", help="Offer the one-command fixes, asking before each (user-level only, never sudo)."),
        json_out: bool = typer.Option(False, "--json", help="Print the report as JSON."),
        list_checks: bool = typer.Option(False, "--list", help="List every check and which systems it applies to."),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show check ids, timings and extra rows."),
        no_web: bool = typer.Option(False, "--no-web", help="Skip every check that needs the internet."),
        report: bool = typer.Option(False, "--report", help="Print an anonymised markdown report for a GitHub issue."),
        strict: bool = typer.Option(False, "--strict", help="Exit with 2 when there are warnings (and no problems)."),
        no_fix: bool = typer.Option(False, "--no-fix", help="Leave out the fix hints."),
    ) -> None:
        """Check Python, Ollama, models, GPU, memory, disk, network, config, security, plugins and this operating system."""
        doctor_command(only, skip, deep, fix, json_out, list_checks, verbose, no_web, report, strict, no_fix)

