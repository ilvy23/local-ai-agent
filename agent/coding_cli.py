from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.coding.session import CodingSession, SessionResult
from agent.coding.workspace import Workspace, create_workspace
from agent.events import ActivityEvent, EventBus

_TIER_NAMES = {
    0: "tier 0 exerciser",
    1: "tier 1 (accumulating)",
    2: "tier 2 (fresh context)",
    3: "tier 3 (escalated)",
    4: "handback",
}


def format_event(event: ActivityEvent) -> str | None:
    kind = event.kind
    if kind == "attempt":
        tier = event.data.get("tier", 1)
        model = event.data.get("model", "")
        return f"[cyan]▶ {_TIER_NAMES.get(tier, f'tier {tier}')}[/cyan] · {model}"
    if kind == "malformed":
        return f"[yellow]  malformed response ({event.data.get('count')}), re-asking[/yellow]"
    if kind == "check":
        status = event.data.get("status")
        if status == "failed" or event.data.get("ok") is False:
            return f"[red]  ✗ {event.message}[/red]"
        if status == "passed":
            return f"[green]  ✓ {event.data.get('name')}[/green]"
        return None
    if kind == "escalate":
        return f"[magenta]⚡ escalating: {event.message}[/magenta]"
    if kind == "success":
        return "[bold green]✓ tests passing[/bold green]"
    if kind == "handback":
        return "[bold red]✗ handing back — see trail[/bold red]"
    return None


def read_files(paths: list[str], base: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for p in paths:
        path = Path(p)
        abs_path = path if path.is_absolute() else base / path
        try:
            rel = str(abs_path.resolve().relative_to(base.resolve()))
        except ValueError:
            rel = path.name
        try:
            files[rel] = abs_path.read_text(encoding="utf-8")
        except OSError:
            continue
    return files


def write_back(workspace: Workspace, changed: tuple[str, ...], base: Path) -> list[str]:
    written = []
    for rel in changed:
        try:
            content = workspace.read(rel)
        except OSError:
            continue
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)
    return written


def run_code(
    client: Any,
    config: dict[str, Any],
    base: Path,
    task: str,
    file_paths: list[str],
    *,
    apply: bool,
    emit: Any,
) -> SessionResult:
    bus = EventBus()
    bus.subscribe(emit)
    files = read_files(file_paths, base)

    workspace = create_workspace(base)
    try:
        session = CodingSession(client, config, workspace, bus=bus)
        result = session.run(task, files=files or None)
        if result.success and apply:
            write_back(workspace, result.changed_files, base)
    finally:
        workspace.close()
    return result
