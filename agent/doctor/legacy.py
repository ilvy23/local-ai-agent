"""The pre-modular doctor API, kept working: `Check` / `DoctorReport` (ok | warn | fail only), `collect_checks`, `run_doctor`,
`needs_onboarding`, `run_onboarding`. `agent/ui/doctor.py` re-exports these. `collect_checks` runs the cheap `core=True` checks of
the new engine (agent/doctor/checks) and converts them, so the home screen, onboarding and the old tests behave as before while the
full doctor (`nightage doctor`, agent/doctor/cli.py) has many more checks. Onboarding stays fast: core checks only, no network."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from agent import brand
from agent.doctor import core
from agent.doctor.probes import Probes, model_installed
from agent.ui import style

OK, WARN, FAIL = "ok", "warn", "fail"
REQUIRED_PACKAGES = (
    ("rich", "rich", True), ("typer", "typer", True), ("prompt_toolkit", "prompt_toolkit", True),
    ("httpx", "httpx", True), ("yaml", "pyyaml", True), ("sqlite_vec", "sqlite-vec", False),
)
_EMBED_HINTS = ("embed", "bge", "nomic", "minilm", "e5", "gte-")
EXAMPLE_PROMPTS = (
    "Summarise what is in this folder and suggest what to work on first.",
    "Explain the difference between a process and a thread, with a short example.",
    "Research: what changed in the latest stable Python release?",
)
LEGACY_ORDER = ("python", "packages", "platform", "profile", "storage", "ollama", "remote_latency", "model_chat", "model_coding",
                "model_embed", "gpu", "context", "data", "config", "web", "terminal")


@dataclass
class Check:
    id: str
    title: str
    status: str            # ok | warn | fail
    detail: str = ""
    fix: str = ""

    def to_dict(self, fix_hints: bool = True) -> dict[str, str]:
        d = {"id": self.id, "title": self.title, "status": self.status, "detail": self.detail}
        if fix_hints:
            d["fix"] = self.fix
        return d


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    suggestion: str = ""                      # filled by onboarding
    examples: list[str] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    @property
    def ok(self) -> bool:
        return self.count(FAIL) == 0

    @property
    def status(self) -> str:
        return FAIL if self.count(FAIL) else WARN if self.count(WARN) else OK

    @property
    def exit_code(self) -> int:
        return 1 if self.count(FAIL) else 0

    def get(self, check_id: str) -> Check | None:
        return next((c for c in self.checks if c.id == check_id), None)

    def to_dict(self, fix_hints: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "product": brand.NAME, "status": self.status, "ok": self.ok,
            "summary": {"ok": self.count(OK), "warn": self.count(WARN), "fail": self.count(FAIL)},
            "checks": [c.to_dict(fix_hints) for c in self.checks],
        }
        if self.suggestion:
            d["suggestion"] = self.suggestion
        if self.examples:
            d["examples"] = list(self.examples)
        return d


def _models(config: dict | None) -> dict[str, Any]:
    return (config or {}).get("models") or {}


def suggest_model(installed: dict[str, int], config: dict | None) -> str:
    """The model to start with: the configured chat model if present, else the largest non-embedding model that
    is not huge, else the smallest one. '' when nothing is installed."""
    want = _models(config).get("chat") or ""
    cands = {n: s for n, s in installed.items() if not any(h in n.lower() for h in _EMBED_HINTS)}
    if want and model_installed(want, cands):
        return want
    if not cands:
        return ""
    limit = 20 * 1024 ** 3
    fitting = {n: s for n, s in cands.items() if s <= limit}
    if fitting:
        return max(fitting, key=lambda n: (fitting[n], n))
    return min(cands, key=lambda n: (cands[n], n))


def collect_checks(config: dict | None = None, *, probes: Probes | None = None, check_web: bool = True,
                   web_timeout: float = 2.0) -> DoctorReport:
    """Run the cheap core checks (same ids and meanings as always) and return them as an ok/warn/fail report."""
    ctx = core.Context(config, probes or Probes(), no_web=not check_web)
    specs = core.select(ctx, specs=[s for s in core.all_specs() if s.core])
    results = core.run_checks(ctx, specs)
    rank = {cid: i for i, cid in enumerate(LEGACY_ORDER)}
    rep = DoctorReport()
    for r in sorted(results, key=lambda r: rank.get(r.id, 99)):
        rep.checks.append(_to_check(r))
    return rep


def _to_check(r: core.Result) -> Check:
    status = r.status
    if status == core.INFO:
        status = OK
    elif status == core.SKIP:
        status = OK if r.detail.startswith("skipped") else WARN
    return Check(r.id, r.title, status, r.detail, r.hint)


# ----------------------------------------------------------------------------- rendering


def render_report(rep: DoctorReport, console: Any = None, *, fix_hints: bool = True, plain: bool | None = None,
                  width: int | None = None) -> None:
    from rich.text import Text

    console = console or style.make_console(width=width, plain=plain)
    plain = style.plain_mode() if plain is None else plain
    g, pal = style.glyphs(plain), style.palette()
    w = max(30, console.width or width or 80)
    mark = {OK: (g.check, pal.ok), WARN: ("!", pal.warn), FAIL: (g.cross, pal.error)}
    console.print(Text(f"{brand.NAME} doctor", style=f"bold {pal.accent}"))
    for c in rep.checks:
        sym, col = mark[c.status]
        t = Text()
        t.append(f" {sym} ", style=col if not plain else "")
        t.append(c.title, style="bold" if not plain else "")
        console.print(t)
        if c.detail:
            for line in _wrap(c.detail, w - 5):
                console.print(Text("     " + line))
        if fix_hints and c.fix and c.status != OK:
            for i, line in enumerate(_wrap(c.fix, w - 10)):
                console.print(Text(("     fix: " if i == 0 else "          ") + line, style="" if plain else pal.dim))
    n = rep.count
    console.print()
    console.print(Text(f" {n(OK)} ok, {n(WARN)} warning(s), {n(FAIL)} problem(s)", style="" if plain else "bold"))
    if rep.suggestion:
        console.print(Text(f" Suggested model: {rep.suggestion}"))
    if rep.examples:
        console.print(Text(" Try one of these:"))
        for e in rep.examples:
            for i, line in enumerate(_wrap(e, w - 8)):
                console.print(Text(("   - " if i == 0 else "     ") + line))


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, max(10, width)) or [""]


def run_doctor(config: dict | None = None, *, fix_hints: bool = True, as_json: bool = False, probes: Probes | None = None,
               check_web: bool = True, print_report: bool = True, console: Any = None, plain: bool | None = None) -> DoctorReport:
    """Run all checks. Prints the pretty report (or JSON when `as_json`) unless `print_report=False`.
    Callers should exit with `report.exit_code` (1 when any check failed)."""
    rep = collect_checks(config, probes=probes, check_web=check_web)
    if print_report:
        if as_json:
            out = (console.file if console is not None and getattr(console, "file", None) else sys.stdout)
            out.write(json.dumps(rep.to_dict(fix_hints), indent=2) + "\n")
        else:
            render_report(rep, console, fix_hints=fix_hints, plain=plain)
    return rep


# ----------------------------------------------------------------------------- onboarding


def needs_onboarding(config: dict | None = None, *, probes: Probes | None = None) -> bool:
    """First run: no database yet, or Ollama is up but has no models at all."""
    p = probes or Probes()
    _, db = p.data_paths(config)
    if not db.exists():
        return True
    oll = p.ollama()
    return bool(oll.reachable and not oll.models)


def run_onboarding(config: dict | None = None, *, probes: Probes | None = None, check_web: bool = False,
                   print_report: bool = True, console: Any = None, plain: bool | None = None) -> DoctorReport:
    """Doctor + a model suggestion from what is installed + three example prompts. Meant for the first start."""
    p = probes or Probes()
    rep = collect_checks(config, probes=p, check_web=check_web)
    oll = p.ollama()
    sug = suggest_model(oll.models, config) if oll.reachable else ""
    rep.suggestion = sug
    rep.examples = list(EXAMPLE_PROMPTS)
    if not sug and oll.reachable:
        want = _models(config).get("chat") or "a chat model"
        rep.suggestion = ""
        rep.examples = [f"First install a model: `ollama pull {want}`", *EXAMPLE_PROMPTS]
    if print_report:
        from rich.text import Text
        console = console or style.make_console(plain=plain)
        console.print(Text(f"Welcome to {brand.NAME}. {brand.TAGLINE}"))
        console.print()
        render_report(rep, console, plain=plain)
    return rep
