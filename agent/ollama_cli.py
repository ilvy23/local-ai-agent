"""`nightage ollama [status|start|stop|list|pull|rm]` and the shortcut `nightage pull <model>`.

The progress bar is the Nightage one (agent/ui/style.py glyphs; plain ASCII when the terminal cannot show blocks) and adapts to the
terminal width on every update: the line is never longer than width-1, so it does not wrap on a 40-column phone terminal."""

from __future__ import annotations

import time
from typing import IO, Any

import httpx
import typer
from rich.console import Console

from agent import brand, ollama_local
from agent.ui import style

ollama_app = typer.Typer(add_completion=False, invoke_without_command=True,
                         help="Run and manage the Ollama on this machine: status, start, stop, list, pull, rm.")

BAR_MIN = 10


# ------------------------------------------------------------------------------------------------------------ rendering


def fmt_bytes(n: float) -> str:
    n = float(n)
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("kB", 1024)):
        if n >= div:
            return f"{n / div:.1f} {unit}" if unit != "kB" else f"{n / div:.0f} {unit}"
    return f"{int(n)} B"


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "ETA ?"
    s = int(max(0, seconds))
    if s < 60:
        return f"ETA {s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"ETA {m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"ETA {h}h {m:02d}m"


def render_line(p: ollama_local.Progress, width: int, g: style.Glyphs | None = None) -> str:
    """`[████████░░░░░░░░]  50%  1.2/2.4 GB  12.3 MB/s  ETA 3m 12s`, shortened to fit width-1 columns (bar >= 10 when possible)."""
    g = g or style.UNICODE
    limit = max(16, int(width) - 1)
    pct = f"{min(100.0, max(0.0, p.pct)):3.0f}%"
    prio = ["size", "speed", "eta"]                     # dropped in this order when the line is too wide
    parts = {"size": f"{fmt_bytes(p.completed)}/{fmt_bytes(p.total)}" if p.total else "", "speed": f"{fmt_bytes(p.speed)}/s" if p.speed > 0 else "",
             "eta": fmt_eta(p.eta) if (p.speed > 0 or p.done) else ""}
    while True:
        tail = "  ".join(parts[k] for k in ("size", "speed", "eta") if parts[k])
        fixed = 3 + len(pct) + (2 + len(tail) if tail else 0)          # "[" "]" " " pct "  " tail
        room = limit - fixed
        if room >= BAR_MIN or not any(parts.values()):
            break
        for k in prio:
            if parts[k]:
                parts[k] = ""
                break
    bar_w = max(4, min(room, 40))
    return f"[{style.bar(p.pct, bar_w, g)}] {pct}" + (f"  {tail}" if tail else "")


class PullView:
    """Draws progress: one rewritten line on a terminal, a line per 10 % otherwise (pipes, logs, plain mode)."""

    def __init__(self, console: Console, plain: bool | None = None) -> None:
        self.console = console
        self.plain = style.plain_mode(stream=console.file) if plain is None else plain
        self.g = style.glyphs(self.plain)
        self.live = console.is_terminal and not self.plain
        self.tty_line = console.is_terminal                  # \r rewriting also works in plain (ASCII) mode on a real terminal
        self._last_step = -1
        self._status = ""
        self._last_draw = 0.0

    def update(self, p: ollama_local.Progress) -> None:
        f: IO[str] = self.console.file
        if self.tty_line:
            now = time.monotonic()
            if p.status != self._status or now - self._last_draw >= 0.1 or p.done:
                if p.status != self._status and not p.total:
                    f.write("\r\x1b[2K" + style.trunc(p.status, self.console.width - 1, self.g) + "\n")
                    f.flush()
                    self._status = p.status
                if p.total:
                    f.write("\r\x1b[2K" + render_line(p, self.console.width, self.g))
                    f.flush()
                self._last_draw = now
            return
        step = int(p.pct // 10)
        if p.total and step != self._last_step:
            self._last_step = step
            f.write(render_line(p, self.console.width if self.console.width else 80, self.g) + "\n")
            f.flush()

    def finish(self) -> None:
        if self.tty_line:
            self.console.file.write("\n")
            self.console.file.flush()


# ------------------------------------------------------------------------------------------------------------ commands


def _cfg() -> dict[str, Any]:
    from agent.config import load_config

    return load_config()


def _status_lines(console: Console, config: dict) -> ollama_local.Status:
    st = ollama_local.status(config)
    console.print(st.line())
    if st.reachable and st.models is not None:
        console.print(f"[dim]{st.models} model(s) installed. Address: {st.url}[/dim]")
    fix = st.fix()
    if fix:
        console.print(f"[yellow]{fix}[/yellow]")
    if st.local:
        console.print(f"[dim]Server log: {st.log}[/dim]")
    return st


@ollama_app.callback()
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _status_lines(style.make_console(), _cfg())


@ollama_app.command("status")
def status_cmd() -> None:
    """Is Ollama running here, which address, installed models and the exact fix when it is not."""
    st = _status_lines(style.make_console(), _cfg())
    raise typer.Exit(0 if st.reachable else 1)


@ollama_app.command("start")
def start_cmd() -> None:
    """Start Ollama in the background (only on this machine) and wait until it answers."""
    console = style.make_console()
    res = ollama_local.start(_cfg())
    console.print(res.message, markup=False)
    raise typer.Exit(0 if res.ok else 1)


@ollama_app.command("stop")
def stop_cmd() -> None:
    """Stop the Ollama that Nightage started (never one you or the system started)."""
    console = style.make_console()
    res = ollama_local.stop(_cfg())
    console.print(res.message, markup=False)
    raise typer.Exit(0 if res.ok else 1)


@ollama_app.command("list")
def list_cmd() -> None:
    """The installed models with their sizes."""
    console = style.make_console()
    config = _cfg()
    url = ollama_local.base_url(config)
    try:
        models = httpx.get(url + "/api/tags", timeout=5).json().get("models", [])
    except (httpx.HTTPError, ValueError, OSError):
        console.print(f"Ollama does not answer at {url}. Try: {brand.CMD} ollama start", markup=False)
        raise typer.Exit(1) from None
    if not models:
        console.print(f"No models installed. Try: {brand.CMD} pull qwen3.5:4b", markup=False)
        return
    for m in sorted(models, key=lambda m: m.get("name", "")):
        console.print(f"{m.get('name', '?'):<34} {fmt_bytes(m.get('size', 0)):>9}", markup=False)


@ollama_app.command("rm")
def rm_cmd(model: str = typer.Argument(..., help="Model to delete from this machine's Ollama.")) -> None:
    """Delete an installed model (frees disk space)."""
    console = style.make_console()
    url = ollama_local.base_url(_cfg())
    try:
        r = httpx.request("DELETE", url + "/api/delete", json={"model": model}, timeout=15)
    except httpx.HTTPError:
        console.print(f"Ollama does not answer at {url}. Try: {brand.CMD} ollama start", markup=False)
        raise typer.Exit(1) from None
    if r.status_code == 404:
        console.print(f"{model} is not installed.", markup=False)
        raise typer.Exit(1)
    r.raise_for_status()
    console.print(f"Removed {model}.", markup=False)


def do_pull(model: str, *, force: bool = False, console: Console | None = None, config: dict | None = None) -> int:
    """The pull flow shared by `nightage pull` and `nightage ollama pull`. Returns an exit code."""
    console = console or style.make_console()
    config = config if config is not None else _cfg()
    url = ollama_local.base_url(config)
    started_here = False
    if ollama_local.is_local(url):
        res = ollama_local.ensure(config)
        if not res.ok:
            console.print(res.message, markup=False)
            return 1
        started_here = res.action == "started"
        if started_here:
            console.print("Started Ollama in the background for this download.", markup=False)
        verdict = _fit(model, console)
        if verdict == "fail" and not force:
            console.print("Not downloading. Pick a smaller model, free space, or add --force if you know better.", markup=False)
            return 2
    view = PullView(console)
    console.print(f"Downloading {model}", markup=False, highlight=False)
    code = 0
    try:
        ollama_local.pull(model, view.update, config=config)
        view.finish()
        console.print(f"{model} is ready.", markup=False)
    except KeyboardInterrupt:
        view.finish()
        console.print(f"Stopped. Nothing is lost: run  {brand.CMD} pull {model}  again to continue where it stopped.", markup=False)
        code = 130
    except ollama_local.PullError as exc:
        view.finish()
        console.print(f"Could not download {model}: {exc}", markup=False)
        code = 1
    finally:
        if started_here and not ollama_local.other_clients():
            ollama_local.stop(config)
    return code


def _fit(model: str, console: Console) -> str:
    try:
        gpus: list = []
        try:
            from agent.doctor.probes import Probes

            gpus = list(Probes().gpus())
        except Exception:  # noqa: BLE001 - GPU probing is optional
            gpus = []
        fit = ollama_local.check_fit(model, gpus=gpus, disk_free=ollama_local.disk_free_bytes())
    except Exception:  # noqa: BLE001 - advice only; never block a download on a probe failure
        return "unknown"
    if fit.verdict in ("warn", "fail"):
        size = f" ({fit.size / 1024 ** 3:.1f} GB, {'estimated' if fit.source in ('tag',) else 'from ' + fit.source})" if fit.size else ""
        console.print(f"{'Too big for this machine' if fit.verdict == 'fail' else 'Warning'}: {model}{size} {fit.text}.", markup=False)
    return fit.verdict


@ollama_app.command("pull")
def pull_cmd(model: str = typer.Argument(..., help="Model to download, e.g. qwen3.5:4b"),
             force: bool = typer.Option(False, "--force", help="Download even when the model looks too big for this machine.")) -> None:
    """Download a model with a percent bar (speed, ETA). Ctrl+C is safe: run it again to resume."""
    raise typer.Exit(do_pull(model, force=force))


def register(app: typer.Typer) -> None:
    app.add_typer(ollama_app, name="ollama")

    @app.command("pull")
    def pull_shortcut(model: str = typer.Argument(..., help="Model to download, e.g. qwen3.5:4b"),
                      force: bool = typer.Option(False, "--force", help="Download even when the model looks too big for this machine.")) -> None:
        """Shortcut for `nightage ollama pull <model>`."""
        raise typer.Exit(do_pull(model, force=force))

