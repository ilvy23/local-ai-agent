"""The check ladder: ast.parse → ruff → pytest. Cheapest first, fail fast.

Each rung returns a model-legible error so a 7B can act on it. Lint gates only on
real bugs (pyflakes F + syntax E9), never style — a fix loop must not thrash on
line length. `ruff --fix` runs first so the model never spends a generation call
on an unused import.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from agent.coding.verify import runner
from agent.events import NULL, Events


@dataclass
class CheckResult:
    ok: bool
    stage: str          # "syntax" | "lint" | "tests" | "all"
    error: str = ""     # enriched, model-legible; empty when ok
    note: str = ""      # side-info the model should know (e.g. auto-installed pkgs)


def check_syntax(files: list[str | Path]) -> CheckResult:
    for f in files:
        p = Path(f)
        if p.suffix != ".py" or not p.exists():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(src, filename=str(p))
        except SyntaxError as exc:
            lines = src.splitlines()
            offending = lines[exc.lineno - 1] if exc.lineno and exc.lineno <= len(lines) else ""
            caret = " " * (max(0, (exc.offset or 1) - 1)) + "^"
            return CheckResult(
                False, "syntax",
                f"SyntaxError in {p.name}, line {exc.lineno}: {exc.msg}\n"
                f"    {offending.strip()}\n    {caret}",
            )
    return CheckResult(True, "syntax")


# Common stdlib modules the model forgets to import. Names it uses without `import X`
# get auto-fixed — a well-known model failure mode, especially on longer files.
_AUTO_IMPORT_STDLIB = frozenset({
    "random", "time", "math", "json", "os", "sys", "re", "datetime", "collections",
    "itertools", "functools", "typing", "pathlib", "string", "io", "logging",
    "abc", "enum", "dataclasses", "copy", "warnings", "socket", "threading", "queue",
    "asyncio", "subprocess", "tempfile", "shutil", "argparse", "pickle", "unittest",
})


def _auto_import_stdlib(issues: list[dict], cwd: str | Path) -> list[str]:
    """Add missing `import X` for stdlib names the model used but forgot. Returns
    names actually added (for the CheckResult.note)."""
    added: list[str] = []
    by_file: dict[str, set[str]] = {}
    for i in issues:
        if i.get("code") != "F821":
            continue
        # "Undefined name `random`" → "random"
        msg = i.get("message", "")
        m = re.search(r"`([\w.]+)`", msg)
        if not m:
            continue
        name = m.group(1).split(".")[0]
        if name not in _AUTO_IMPORT_STDLIB:
            continue
        by_file.setdefault(i["filename"], set()).add(name)
    for fname, names in by_file.items():
        try:
            src = Path(fname).read_text(encoding="utf-8")
        except OSError:
            continue
        new_imports = [f"import {n}" for n in sorted(names) if f"\nimport {n}" not in "\n" + src]
        if not new_imports:
            continue
        Path(fname).write_text("\n".join(new_imports) + "\n" + src, encoding="utf-8")
        added.extend(names)
    return sorted(set(added))


def check_lint(cwd: str | Path) -> CheckResult:
    # Auto-fix the cosmetic stuff first, silently.
    runner.run(["ruff", "check", "--fix", "--quiet", "."], cwd=cwd, timeout=60)
    # Gate only on real bugs: F = pyflakes (undefined names, unused, bad calls),
    # E9 = runtime/syntax errors. Style rules never fail the ladder.
    r = runner.run(
        ["ruff", "check", "--select", "F,E9", "--output-format", "json", "."],
        cwd=cwd, timeout=60,
    )
    if r.ok:
        return CheckResult(True, "lint")
    # Auto-inject missing stdlib imports (random/time/math/…) and retry once.
    try:
        issues_raw = json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        issues_raw = []
    added = _auto_import_stdlib(issues_raw, cwd)
    if added:
        r = runner.run(
            ["ruff", "check", "--select", "F,E9", "--output-format", "json", "."],
            cwd=cwd, timeout=60,
        )
        if r.ok:
            return CheckResult(True, "lint",
                               note=f"auto-added imports: {', '.join(added)}")
    try:
        issues = json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        return CheckResult(False, "lint", r.output[:2000] or "ruff failed")
    if not issues:  # ruff errored for a non-lint reason; don't block on it
        return CheckResult(True, "lint")
    lines = [
        f"{Path(i['filename']).name}:{i['location']['row']}:{i['location']['column']}: "
        f"{i['code']} {i['message']}"
        for i in issues[:10]
    ]
    # Specific hint for the biggest small-model failure mode: undefined names
    # that aren't stdlib (so we couldn't auto-import them).
    undef = [re.search(r"`([\w.]+)`", i.get("message", "")) for i in issues if i.get("code") == "F821"]
    undef_names = sorted({m.group(1) for m in undef if m})
    hint = ""
    if undef_names:
        hint = ("\n\nFIX: for each undefined name above, either add `<name> = <value>` "
                "at the top of the file (for constants like colours/dimensions) or import "
                "it. Do this in ONE edit — read the file, then write_file with the "
                f"complete corrected version. Missing: {', '.join(undef_names[:8])}")
    return CheckResult(False, "lint", "Lint errors (likely real bugs):\n" + "\n".join(lines) + hint)


_INSTALL_BUDGET = 6         # per-process cap so a runaway import loop can't hammer pip
_installs_used = 0
_installed_this_session: set[str] = set()


def check_tests(cwd: str | Path, target: str | None = None,
                timeout: int = 120, events=None) -> CheckResult:
    global _installs_used
    # No `-x`: run ALL tests and report EVERY failure. The model does whole-file
    # rewrites, so it must see all failures at once — stopping at the first turns
    # a coherent rewrite into multi-round whack-a-mole (fixing #1 breaks #5).
    cmd = [sys.executable, "-m", "pytest", "-q", "-l", "--tb=short",
           "--no-header", "-p", "no:cacheprovider"]
    if target:
        cmd.append(target)
    # PYTHONDONTWRITEBYTECODE: never let a stale .pyc from a previous repair
    # attempt shadow the model's freshly-written source.
    # SDL_VIDEODRIVER/SDL_AUDIODRIVER=dummy: pygame et al. call display.set_mode
    # at import time in game code — without headless drivers they crash and the
    # model reads a scary traceback and edits forever.
    env = {**os.environ,
           "PYTHONDONTWRITEBYTECODE": "1",
           "SDL_VIDEODRIVER": "dummy",
           "SDL_AUDIODRIVER": "dummy"}
    r = runner.run(cmd, cwd=cwd, timeout=timeout, env=env)
    if r.ok:
        return CheckResult(True, "tests")
    if r.timed_out:
        return CheckResult(False, "tests", r.stderr.strip() or "tests timed out")

    # Auto-install missing packages and retry ONCE. Kills the whole class of
    # 'ImportError → model edits forever' failures for every third-party lib,
    # not just pygame. Budget-capped so a bogus import can't loop.
    pkgs = _installable(r.output)
    pkgs = [p for p in pkgs if p not in _installed_this_session]  # don't reinstall failures
    if pkgs and _installs_used < _INSTALL_BUDGET:
        _installs_used += 1
        _installed_this_session.update(pkgs)
        if events is not None:
            events.emit("auto_install", packages=pkgs)
        ok, out = _pip_install(pkgs, cwd)
        if ok:
            r = runner.run(cmd, cwd=cwd, timeout=timeout, env=env)  # retry
            if r.ok:
                return CheckResult(True, "tests",
                                   note=f"auto-installed: {', '.join(pkgs)}")
        else:
            return CheckResult(False, "tests",
                f"Auto-install failed for {', '.join(pkgs)}. Pip said:\n{out[-800:]}\n"
                "The package name may be wrong — check spelling or PyPI.")
    # Still failing after (maybe) install → give a clean actionable message
    still_missing = _installable(r.output)
    if still_missing:
        return CheckResult(False, "tests",
            f"Still missing after install attempt: {', '.join(still_missing)}. "
            "Likely a naming issue (pip package name ≠ import name for some libs, e.g. "
            "`pip install pillow` gives `import PIL`). Use web_docs to find the correct pip name.")
    if r.rc == 5:  # pytest: no tests collected — try SMOKE run for scripts/games
        smoke = _smoke_run(cwd, env, timeout=4)
        if smoke is not None:
            ok, note = smoke
            if ok:
                # No unit test possible (game loop / interactive script), but the
                # code runs cleanly under headless drivers — that's proof enough.
                return CheckResult(True, "tests", note=f"smoke: {note}")
            return CheckResult(False, "tests",
                f"No tests to run, and the script crashes on import/start:\n{note}")
        has_py = any(Path(cwd).rglob("*.py"))
        if not has_py:
            msg = ("No Python code exists yet. Write your implementation FIRST "
                   "(write_file some_module.py with the actual code) — editing "
                   "requirements.txt / config files won't create tests. Then add a "
                   "test_<name>.py with real assertions.")
        else:
            msg = ("You have code but no test file was collected. Either create "
                   "test_<name>.py (functions named `test_*` that `assert` results), "
                   "OR make one of your .py files runnable by adding "
                   "`if __name__ == '__main__': main()` — for a runnable script, "
                   "running cleanly counts as a passing check.")
        return CheckResult(False, "tests", msg)
    return CheckResult(False, "tests", _trim_pytest(r.output))


def _smoke_run(cwd: str | Path, env: dict, timeout: int = 4) -> tuple[bool, str] | None:
    """Try to run a runnable script for a few seconds. Returns (ok, note) or None
    if no runnable script exists. Timeout counts as success for game loops."""
    scripts = [p for p in Path(cwd).rglob("*.py")
               if not (p.name.startswith("test_") or p.name.endswith("_test.py"))
               and "if __name__" in p.read_text(encoding="utf-8", errors="replace")]
    if not scripts:
        return None
    # Pick the shortest path — usually the entry point (snake.py, main.py, app.py).
    script = min(scripts, key=lambda p: (len(str(p.relative_to(cwd))), str(p)))
    rel = script.relative_to(cwd)
    r = runner.run([sys.executable, str(rel)], cwd=cwd, timeout=timeout, env=env)
    if r.timed_out:  # game loop — running for `timeout`s without crashing = works
        return True, f"{rel} ran for {timeout}s without crashing (game loop)"
    if r.ok:
        return True, f"{rel} ran and exited cleanly"
    return False, f"{rel} crashed:\n{(r.output)[-1500:]}"


_MISSING_MOD = re.compile(r"ModuleNotFoundError: No module named ['\"]([\w.]+)['\"]")
# Package-name shape must be strict — a typo or garbage token could otherwise
# name a random PyPI package and we'd install it. Alphanum + underscores/hyphens,
# reasonable length, no path separators or leading dots.
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")
# Never try to pip install these — either they're stdlib or they'd resolve to
# something dangerous. (stdlib rarely reaches this path anyway; belt-and-braces.)
_NEVER_INSTALL = frozenset({
    "os", "sys", "json", "re", "math", "time", "typing", "pathlib", "subprocess",
    "collections", "itertools", "functools", "dataclasses", "asyncio", "unittest",
    "logging", "datetime", "random", "string", "io", "shutil", "tempfile",
    "abc", "enum", "argparse", "copy", "pickle", "socket", "threading", "queue",
})


def _installable(output: str) -> list[str]:
    """Missing packages from `output` that are safe to `pip install`."""
    raw = sorted({m.split(".")[0] for m in _MISSING_MOD.findall(output)})
    return [m for m in raw if m not in _NEVER_INSTALL and _SAFE_NAME.match(m)]


# Known-good fallbacks for packages that often fail to build from source without
# system libs. The import name is the same, so the model's code needs no change.
_PIP_FALLBACK = {
    "pygame": "pygame-ce",   # community edition ships prebuilt wheels for more Pythons
}


def _pip_install(pkgs: list[str], cwd: str | Path, timeout: int = 180) -> tuple[bool, str]:
    """Install packages into the same Python that runs the tests. Returns (ok, output).
    On failure, retries each failed package with a known-good fallback (e.g.
    pygame → pygame-ce) so the model doesn't burn steps on unfixable build errors."""
    r = runner.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--disable-pip-version-check", *pkgs],
                   cwd=str(cwd), timeout=timeout)
    if r.ok:
        return True, r.output
    fallbacks = [_PIP_FALLBACK[p] for p in pkgs if p in _PIP_FALLBACK]
    if fallbacks:
        r2 = runner.run([sys.executable, "-m", "pip", "install", "--quiet",
                         "--disable-pip-version-check", *fallbacks],
                        cwd=str(cwd), timeout=timeout)
        if r2.ok:
            return True, f"used fallback: {' '.join(fallbacks)}"
    return False, r.output


def _trim_pytest(out: str, max_chars: int = 4000) -> str:
    """Keep the failures section — the part after the last 'FAILURES' banner."""
    marker = out.rfind("= FAILURES =")
    if marker == -1:
        marker = out.rfind("FAILURES")
    body = out[marker:] if marker != -1 else out
    return body[:max_chars]


def reset_install_budget() -> None:
    """Fresh budget for each agent-code session (called at loop start)."""
    global _installs_used
    _installs_used = 0
    _installed_this_session.clear()


def run_ladder(cwd: str | Path, changed_files: list[str | Path],
               test_target: str | None = None, events: Events = NULL) -> CheckResult:
    """Run syntax → lint → tests, stopping at the first failure."""
    for stage, fn in (
        ("syntax", lambda: check_syntax(changed_files)),
        ("lint", lambda: check_lint(cwd)),
        ("tests", lambda: check_tests(cwd, test_target, events=events)),
    ):
        events.emit("check", stage=stage)
        result = fn()
        if not result.ok:
            events.emit("check_failed", stage=stage, error=result.error)
            return result
    events.emit("check_passed")
    return CheckResult(True, "all")


if __name__ == "__main__":  # self-check with a real temp repo, no model
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "mod.py").write_text("def add(a, b):\n    return a - b\n")  # bug: - not +
        (root / "test_mod.py").write_text(
            "from mod import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        )
        res = run_ladder(root, [root / "mod.py"])
        assert not res.ok and res.stage == "tests", f"expected test failure, got {res}"
        assert "add" in res.error.lower() or "assert" in res.error.lower(), res.error

        # syntax error is caught before tests even run
        (root / "mod.py").write_text("def add(a, b)\n    return a + b\n")  # missing colon
        res = run_ladder(root, [root / "mod.py"])
        assert not res.ok and res.stage == "syntax", f"expected syntax failure, got {res}"

        # fix it → whole ladder passes
        (root / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        res = run_ladder(root, [root / "mod.py"])
        assert res.ok and res.stage == "all", f"expected pass, got {res}"

    print("check-ladder self-check passed ✓")
