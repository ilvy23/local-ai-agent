"""The doctor engine: results, the check registry, the context checks share, and the isolating runner.

A check is a function `fn(ctx) -> Result | list[Result] | None` registered with `@check(...)`. The runner

  * runs every check inside its own try/except and its own time limit (a crash or a hang becomes a 'could not run'
    warning, never a broken doctor),
  * runs network checks in parallel threads and everything else one after the other,
  * skips checks that do not apply to this system (`systems=`), need a running Ollama (`needs_ollama=True`), or
    depend (`requires=`) on another check that failed,
  * leaves `deep=True` checks out unless the context asks for them.

Nothing here prints; `agent.doctor.report` renders, `agent.doctor.cli` is the command. Guide: docs/DOCTOR.md.
"""

from __future__ import annotations

import importlib
import itertools
import pkgutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OK, WARN, FAIL, INFO, SKIP = "ok", "warn", "fail", "info", "skip"
STATUSES = (OK, WARN, FAIL, INFO, SKIP)

# Display order of the groups (unknown groups, e.g. from plugins, come after these, alphabetically).
GROUP_ORDER = ("system", "python", "packages", "ollama", "models", "gpu", "memory", "disk", "network", "ports",
               "terminal", "filesystem", "database", "config", "security", "plugins", "update", "deep")
GROUP_TITLES = {
    "system": "System", "python": "Python", "packages": "Packages and tools", "ollama": "Ollama", "models": "Models",
    "gpu": "GPU", "memory": "Memory", "disk": "Disk", "network": "Network", "ports": "Ports", "terminal": "Terminal",
    "filesystem": "Files and folders", "database": "Database", "config": "Configuration", "security": "Security",
    "plugins": "Plugins", "update": "Updates", "deep": "Deep tests (use the model)",
}


@dataclass
class Result:
    """One finding. `hint` is the plain-language fix; `command` the exact command `--fix` would run for it, and `fix`
    the callable that runs it (only ever user-level commands, see agent/doctor/fixes.py)."""

    status: str
    detail: str = ""
    hint: str = ""
    command: str = ""
    fix: Callable[[], str] | None = field(default=None, repr=False)
    id: str = ""
    group: str = ""
    title: str = ""
    lines: list[str] = field(default_factory=list)      # extra rows (tables); always printed
    ms: int = 0
    deep: bool = False
    spec_id: str = ""

    def to_dict(self, hints: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "group": self.group, "title": self.title, "status": self.status,
                             "detail": self.detail}
        if hints:
            d["hint"] = self.hint
            d["command"] = self.command
        if self.lines:
            d["lines"] = list(self.lines)
        d["ms"] = self.ms
        return d


def ok(detail: str = "", hint: str = "", **kw: Any) -> Result:
    return Result(OK, detail, hint, **kw)


def warn(detail: str, hint: str = "", **kw: Any) -> Result:
    return Result(WARN, detail, hint, **kw)


def fail(detail: str, hint: str = "", **kw: Any) -> Result:
    return Result(FAIL, detail, hint, **kw)


def info(detail: str, hint: str = "", **kw: Any) -> Result:
    return Result(INFO, detail, hint, **kw)


def skip(reason: str, **kw: Any) -> Result:
    return Result(SKIP, reason, **kw)


# ----------------------------------------------------------------------------------------------- context


class Context:
    """What every check receives: the config, the probes (everything that touches the machine), the options and a few
    lazily computed shared facts. Tests pass a fake `probes`; checks never touch the machine except through it."""

    def __init__(self, config: dict | None = None, probes: Any = None, *, deep: bool = False, no_web: bool = False,
                 verbose: bool = False) -> None:
        from agent.doctor.probes import Probes

        if config is None:
            from agent.config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self.config: dict = config
        self.p = probes or Probes()
        self.deep, self.no_web, self.verbose = deep, no_web, verbose
        self.results: dict[str, Result] = {}
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._cache:
                try:
                    self._cache[key] = fn()
                except Exception as exc:  # noqa: BLE001 - a failing probe is cached as "unknown", not retried
                    self._cache[key] = exc
            v = self._cache[key]
        if isinstance(v, Exception):
            raise v
        return v

    # --- shared facts
    @property
    def sysinfo(self) -> Any:
        return self.cached("sysinfo", self.p.sysinfo)

    @property
    def ollama(self) -> Any:
        return self.cached("ollama", self.p.ollama)

    @property
    def gpus(self) -> list[Any]:
        try:
            return list(self.cached("gpus", self.p.gpus))
        except Exception:  # noqa: BLE001
            return []

    @property
    def env(self) -> Mapping[str, str]:
        return self.p.environ()

    @property
    def raw_config(self) -> tuple[Path, dict | None, str]:
        return self.cached("config_file", self.p.config_file)

    def section(self, name: str) -> dict:
        v = (self.config or {}).get(name)
        return v if isinstance(v, dict) else {}

    @property
    def tags(self) -> frozenset[str]:
        """What kind of system this is: linux / macos / windows / termux / android / nixos (distro id) / wsl / systemd."""
        def build() -> frozenset[str]:
            try:
                si = self.sysinfo
            except Exception:  # noqa: BLE001 - unknown system: only the checks that apply everywhere run
                return frozenset()
            t = {si.os_family}
            if si.os_family == "android-termux":
                t |= {"termux", "android"}
            if si.distro:
                t.add(si.distro)
            if si.init_system == "systemd":
                t.add("systemd")
            if si.os_family == "linux" and self.p.is_wsl():
                t.add("wsl")
            return frozenset(t)
        return self.cached("tags", build)

    def chat_model(self) -> str:
        return str(self.section("models").get("chat") or "")


# ----------------------------------------------------------------------------------------------- registry


_counter = itertools.count()


@dataclass
class CheckSpec:
    id: str
    group: str
    title: str
    fn: Callable[[Context], Any]
    systems: tuple[str, ...] | None = None
    deep: bool = False
    needs_ollama: bool = False
    network: bool = False          # skipped by --no-web and run in a parallel thread
    parallel: bool = False
    timeout: float = 8.0
    requires: tuple[str, ...] = ()
    core: bool = False             # part of the cheap set used by onboarding and the old `collect_checks`
    order: int = 0
    module: str = ""

    def applies(self, tags: Iterable[str]) -> bool:
        return self.systems is None or bool(set(self.systems) & set(tags))


REGISTRY: dict[str, CheckSpec] = {}
LOAD_ERRORS: list[tuple[str, str]] = []


def check(id: str, group: str, title: str, *, systems: Iterable[str] | None = None, deep: bool = False,
          needs_ollama: bool = False, network: bool = False, parallel: bool = False, timeout: float = 8.0,
          requires: Iterable[str] = (), core: bool = False) -> Callable[[Callable[[Context], Any]], Callable[[Context], Any]]:
    """Register a check. `id` is its family id; the Results it returns may carry their own ids (`id="model_chat"`)."""
    def deco(fn: Callable[[Context], Any]) -> Callable[[Context], Any]:
        REGISTRY[id] = CheckSpec(id, group, title, fn, tuple(systems) if systems else None, deep, needs_ollama, network,
                                 parallel or network, timeout, tuple(requires), core, next(_counter), fn.__module__)
        return fn
    return deco


_discovered = False


def discover() -> None:
    """Import every module under agent/doctor/checks (one per area; checks/system/ holds the per-OS ones)."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    from agent.doctor import checks

    for mod in pkgutil.walk_packages(checks.__path__, checks.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - a broken check module must not break the doctor
            LOAD_ERRORS.append((mod.name, f"{type(exc).__name__}: {exc}"))


def all_specs() -> list[CheckSpec]:
    discover()
    return sorted(REGISTRY.values(), key=_spec_key)


def _group_rank(group: str) -> tuple[int, str]:
    return (GROUP_ORDER.index(group), "") if group in GROUP_ORDER else (len(GROUP_ORDER), group)


def _spec_key(s: CheckSpec) -> tuple:
    return (*_group_rank(s.group), s.order)


def group_titles(groups: Iterable[str]) -> dict[str, str]:
    return {g: GROUP_TITLES.get(g, g.replace("_", " ").title()) for g in groups}


def matches(spec: CheckSpec, tokens: Iterable[str]) -> bool:
    toks = {t.strip().lower() for t in tokens if t.strip()}
    return spec.group in toks or spec.id.lower() in toks or any(spec.id.lower().startswith(t + ".") for t in toks)


def select(ctx: Context, *, only: Iterable[str] = (), skip_groups: Iterable[str] = (),
           specs: Iterable[CheckSpec] | None = None) -> list[CheckSpec]:
    """The specs that will run: applicable to this system, not deep unless asked, `only`/`skip` filters applied."""
    only, skip_groups = list(only), list(skip_groups)
    chosen = []
    for s in (list(specs) if specs is not None else all_specs()):
        if not s.applies(ctx.tags):
            continue
        if s.deep and not ctx.deep:
            continue
        if only and not matches(s, only):
            continue
        if skip_groups and matches(s, skip_groups):
            continue
        chosen.append(s)
    return chosen


# ----------------------------------------------------------------------------------------------- runner


def _toposort(specs: list[CheckSpec]) -> list[CheckSpec]:
    by_id = {s.id: s for s in specs}
    done: set[str] = set()
    out: list[CheckSpec] = []

    def visit(s: CheckSpec, stack: tuple[str, ...]) -> None:
        if s.id in done or s.id in stack:
            return
        for r in s.requires:
            if r in by_id:
                visit(by_id[r], (*stack, s.id))
        done.add(s.id)
        out.append(s)

    for s in specs:
        visit(s, ())
    return out


def _normalise(spec: CheckSpec, raw: Any) -> list[Result]:
    items = [] if raw is None else (raw if isinstance(raw, (list, tuple)) else [raw])
    out = []
    for r in items:
        if not isinstance(r, Result):
            r = Result(WARN, f"check returned {type(r).__name__}, not a Result")
        r.id = r.id or spec.id
        r.group = r.group or spec.group
        r.title = r.title or spec.title
        r.deep = spec.deep
        r.spec_id = spec.id
        out.append(r)
    return out


class _Job:
    """One check running in a daemon thread so a hang cannot freeze the doctor."""

    def __init__(self, spec: CheckSpec, ctx: Context) -> None:
        self.spec, self.ctx = spec, ctx
        self.raw: Any = None
        self.error: BaseException | None = None
        self.t0 = time.monotonic()
        self.thread = threading.Thread(target=self._run, name=f"doctor-{spec.id}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        s = self.spec
        try:
            if s.needs_ollama and not self.ctx.ollama.reachable:
                self.raw = skip("needs a running Ollama, and it is not reachable (see the Ollama group)")
            else:
                self.raw = s.fn(self.ctx)
        except BaseException as exc:  # noqa: BLE001 - includes SystemExit from a plugin
            self.error = exc

    def finish(self) -> list[Result]:
        s = self.spec
        self.thread.join(max(0.01, s.timeout - (time.monotonic() - self.t0)))
        ms = int((time.monotonic() - self.t0) * 1000)
        if self.thread.is_alive():
            res = [warn(f"could not finish within {s.timeout:g} s, so it was skipped", "Run it alone with --only "
                        f"{s.id} to see if it is just slow.")]
        elif self.error is not None:
            e = self.error
            res = [warn(f"could not run: {type(e).__name__}: {str(e)[:160]}".rstrip(": "),
                        "This is a bug in the check, not a problem with your setup; `nightage doctor --report` helps us fix it.")]
        else:
            res = _normalise(s, self.raw)
            for r in res:
                r.ms = ms
            return res
        res = _normalise(s, res)
        for r in res:
            r.ms = ms
        return res


def run_checks(ctx: Context, specs: list[CheckSpec], *, progress: Callable[[CheckSpec], None] | None = None) -> list[Result]:
    """Run `specs` (already selected). Returns the results sorted by group and registration order."""
    ordered = _toposort(specs)
    results: dict[str, list[Result]] = {}
    jobs: dict[str, _Job] = {}
    # parallel (network) checks start right away, in their own threads
    for s in ordered:
        if s.parallel and not s.requires and not (s.network and ctx.no_web):
            jobs[s.id] = _Job(s, ctx)
    for s in ordered:
        if progress:
            progress(s)
        if s.network and ctx.no_web:
            res = _normalise(s, skip("skipped (--no-web)"))
        else:
            blocked = [r for r in s.requires if _status_of(ctx, r) in (FAIL, SKIP, None)]
            if blocked:
                res = _normalise(s, skip(f"skipped because {', '.join(blocked)} did not pass"))
            else:
                job = jobs.get(s.id) or _Job(s, ctx)
                res = job.finish()
        results[s.id] = res
        for r in res:
            ctx.results[r.id] = r
    flat = [r for s in sorted(specs, key=_spec_key) for r in results.get(s.id, [])]
    return flat


def _status_of(ctx: Context, check_id: str) -> str | None:
    r = ctx.results.get(check_id)
    return r.status if r else None


def run_isolated(fn: Callable[[], Any], timeout: float) -> tuple[Any, str]:
    """(value, error) of `fn()` run in a daemon thread with a time limit. Used for plugin checks."""
    box: dict[str, Any] = {}

    def go() -> None:
        try:
            box["v"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["e"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, f"did not finish within {timeout:g} s"
    return box.get("v"), box.get("e", "")


def coerce_plugin_result(value: Any) -> Result:
    """What a plugin check may return: a string (ok), (status, detail[, hint]), a dict, or a Result."""
    if isinstance(value, Result):
        return value
    if isinstance(value, str):
        return ok(value)
    if isinstance(value, (tuple, list)) and value:
        st = str(value[0]).lower()
        st = st if st in STATUSES else WARN
        return Result(st, str(value[1]) if len(value) > 1 else "", str(value[2]) if len(value) > 2 else "")
    if isinstance(value, Mapping):
        st = str(value.get("status", OK)).lower()
        return Result(st if st in STATUSES else WARN, str(value.get("detail", "")), str(value.get("hint", "")))
    return warn(f"the check returned {type(value).__name__}; return a string or (status, detail, hint)")
