"""Turning results into output: the grouped terminal view, JSON, and the shareable markdown report."""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

from agent import brand
from agent.doctor import core
from agent.doctor.core import FAIL, INFO, OK, SKIP, WARN, Result
from agent.ui import style


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    update_note: str = ""

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    def get(self, rid: str) -> Result | None:
        return next((r for r in self.results if r.id == rid), None)

    def exit_code(self, strict: bool = False) -> int:
        if self.count(FAIL):
            return 1
        return 2 if strict and self.count(WARN) else 0

    def groups(self) -> list[tuple[str, list[Result]]]:
        order: dict[str, list[Result]] = {}
        for r in self.results:
            order.setdefault(r.group, []).append(r)
        keys = sorted(order, key=core._group_rank)
        return [(g, order[g]) for g in keys]

    def summary(self) -> str:
        parts = [f"{self.count(OK)} ok", f"{self.count(WARN)} warning(s)", f"{self.count(FAIL)} problem(s)"]
        if self.count(INFO):
            parts.append(f"{self.count(INFO)} note(s)")
        if self.count(SKIP):
            parts.append(f"{self.count(SKIP)} skipped")
        return ", ".join(parts)

    def to_dict(self, hints: bool = True) -> dict[str, Any]:
        worst = FAIL if self.count(FAIL) else WARN if self.count(WARN) else OK
        return {"product": brand.NAME, "status": worst, "ok": not self.count(FAIL),
                "summary": {s: self.count(s) for s in core.STATUSES}, "meta": self.meta,
                "checks": [r.to_dict(hints) for r in self.results]}


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, max(10, width), break_long_words=True, break_on_hyphens=False) or [""]


def render(rep: Report, console: Any = None, *, plain: bool | None = None, width: int | None = None, verbose: bool = False,
           hints: bool = True) -> None:
    from rich.text import Text

    console = console or style.make_console(width=width, plain=plain)
    plain = style.plain_mode() if plain is None else plain
    g, pal = style.glyphs(plain), style.palette()
    w = max(30, console.width or width or 80)
    mark = {OK: (g.check, pal.ok), WARN: ("!", pal.warn), FAIL: (g.cross, pal.error), INFO: ("i", pal.dim), SKIP: ("-", pal.dim)}

    def out(line: str, st: str = "") -> None:
        console.print(Text(line, style="" if plain else st), overflow="fold", soft_wrap=False)

    out(f"{brand.NAME} doctor", f"bold {pal.accent}")
    titles = core.group_titles([gr for gr, _ in rep.groups()])
    for group, rows in rep.groups():
        out("")
        out(f"{titles[group]}", "bold")
        for r in rows:
            sym, col = mark[r.status]
            t = Text()
            t.append(f" {sym} ", style="" if plain else col)
            title = r.title + (f"  [{r.id}, {r.ms} ms]" if verbose else "")
            t.append(title, style="" if plain else ("bold" if r.status not in (INFO, SKIP) else pal.dim))
            console.print(t, overflow="fold", soft_wrap=False)
            if r.detail:
                for line in _wrap(r.detail, w - 5):
                    out("     " + line)
            for extra in r.lines:
                for line in _wrap(extra, w - 7):
                    out("       " + line)
            if hints and r.hint and r.status in (WARN, FAIL, INFO):
                for i, line in enumerate(_wrap(r.hint, w - 10)):
                    out(("     fix: " if i == 0 else "          ") + line, pal.dim)
            if hints and r.command and r.status in (WARN, FAIL, INFO) and r.fix is not None and (verbose or r.status != INFO):
                for i, line in enumerate(_wrap(r.command, w - 12)):
                    out(("     run: " if i == 0 else "          ") + line, pal.dim)
    out("")
    out(f" {rep.summary()}", "bold")
    if rep.update_note:
        for line in _wrap(rep.update_note, w - 2):
            out(" " + line)
    if rep.count(WARN) or rep.count(FAIL):
        fixable = sum(1 for r in rep.results if r.fix is not None and r.status in (WARN, FAIL, INFO))
        if fixable:
            out(f" {fixable} finding(s) have a one-command fix: run `{brand.CMD} doctor --fix` (it asks first).")


def to_json(rep: Report, hints: bool = True) -> str:
    return json.dumps(rep.to_dict(hints), indent=2) + "\n"


def render_list(specs: list[core.CheckSpec], console: Any = None, *, plain: bool | None = None, width: int | None = None,
                as_json: bool = False) -> None:
    from rich.text import Text

    console = console or style.make_console(width=width, plain=plain)
    if as_json:
        data = [{"id": s.id, "group": s.group, "title": s.title, "systems": list(s.systems or []), "deep": s.deep,
                 "network": s.network, "needs_ollama": s.needs_ollama, "core": s.core} for s in specs]
        console.file.write(json.dumps(data, indent=2) + "\n")
        return
    w = max(30, console.width or width or 80)
    cur = None
    titles = core.group_titles({s.group for s in specs})
    for s in specs:
        if s.group != cur:
            cur = s.group
            console.print(Text(f"\n{titles[s.group]}  ({s.group})", style="bold"))
        flags = [f for f, on in (("deep", s.deep), ("network", s.network), ("needs ollama", s.needs_ollama)) if on]
        where = ",".join(s.systems) if s.systems else "all systems"
        first = f"  {s.id}"
        meta = f"{where}" + (f"; {', '.join(flags)}" if flags else "")
        line = f"{first:<28} {meta}"
        for part in _wrap(line, w - 2) if len(line) > w - 2 else [line]:
            console.print(Text(part), overflow="fold", soft_wrap=False)
        console.print(Text("      " + s.title[: w - 8]), overflow="fold", soft_wrap=False)


# ----------------------------------------------------------------------------------------------- markdown report


def to_markdown(rep: Report, config_summary: dict[str, Any] | None = None) -> str:
    """The report BEFORE anonymisation: facts about the machine and every check. No chat content, no config values except
    the model names and the profile name."""
    m = rep.meta
    lines = [f"# {brand.NAME} doctor report", "",
             "Anonymised by `nightage doctor --report`: user name, host name, home folder, IP addresses and tokens are replaced. "
             "Read it once before posting.", "", "## Environment", ""]
    for k in ("version", "install", "python", "system", "machine", "cpu", "ram", "gpu", "ollama", "profile", "terminal"):
        if m.get(k):
            lines.append(f"- **{k}**: {m[k]}")
    if config_summary:
        for k, v in config_summary.items():
            lines.append(f"- **{k}**: {v}")
    lines += ["", f"## Summary: {rep.summary()}", ""]
    for group, rows in rep.groups():
        lines += [f"### {core.GROUP_TITLES.get(group, group)}", "", "| status | check | detail |", "|---|---|---|"]
        for r in rows:
            d = r.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.status} | `{r.id}` | {d} |")
            for extra in r.lines:
                lines.append(f"|  |  | {extra.replace('|', chr(92) + '|')} |")
        lines.append("")
    bad = [r for r in rep.results if r.status in (WARN, FAIL) and r.hint]
    if bad:
        lines += ["### Suggested fixes", ""] + [f"- `{r.id}`: {r.hint}" for r in bad] + [""]
    lines.append(f"<sub>Generated by {brand.NAME} {m.get('version', '')}</sub>")
    return "\n".join(lines) + "\n"


def stdout_is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False
