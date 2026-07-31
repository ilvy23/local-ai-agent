"""The coding agent's toolset, bound to a sandbox worktree.

Everything runs inside the throwaway git worktree. The `run` tool is fully
autonomous — it executes without prompting — except the BLOCKED tier (fork
bombs, mkfs, rm -rf /, sudo-to-root-destructive), which is refused and fed back
to the model. Test files are read-only. GUI/game code runs headless.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from agent.coding.verify import guards, runner
from agent.events import NULL, Events
from agent.safety import RiskLevel, classify_command
from agent.tools.registry import Tool, ToolRegistry

_MAX_OUT = 4000       # chars of tool output fed back to the model

# Writes to these locations mean "install system-wide" — never allowed from the
# sandbox. Reads (cat /etc/os-release to detect the distro) are fine, so we only
# match WRITE intent: a redirect or a copy/move/install/link INTO a system dir.
_SYS_DIRS = (r"(?:/usr/local/bin|/usr/local/sbin|/usr/bin|/usr/sbin|/sbin|/bin|/etc|"
             r"/opt|/usr/lib|/usr/local/lib|/root|~/\.local/bin|\$HOME/\.local/bin|"
             r"/var/spool/cron|/etc/systemd|/etc/init\.d|/etc/cron)")
_SYSTEM_PATH_RE = re.compile(
    rf">>?\s*{_SYS_DIRS}"                                    # > /usr/local/bin/x
    rf"|\b(?:cp|mv|install|tee|ln|rsync|dd)\b[^\n]*?\s{_SYS_DIRS}"  # cp x /usr/bin
    rf"|\bcrontab\b|\bupdate-rc\.d\b|\bsystemctl\s+enable\b",       # persistence
    re.I)
_MAX_READ = 12000     # chars returned by read_file


def _rel_changed_py(root: Path) -> list[str]:
    """Changed/added .py files in the worktree, relative paths."""
    runner.run(["git", "add", "-A"], cwd=root)
    out = runner.run(["git", "diff", "--cached", "--name-only"], cwd=root).stdout
    return [ln for ln in out.splitlines() if ln.endswith(".py")]


def build_coding_tools(ws, config: dict, events: Events = NULL, llm=None,
                       ask_fn=None) -> ToolRegistry:
    root = Path(ws.path)
    timeout = config.get("sandbox", {}).get("timeout_seconds", 30)
    mem = config.get("sandbox", {}).get("memory_mb", 2048)

    def _safe(path: str) -> Path:
        target = (root / path).resolve()
        if target != root.resolve() and root.resolve() not in target.parents:
            raise ValueError("path escapes the workspace")
        return target

    def read_file(path: str, **_) -> str:
        try:
            return _safe(path).read_text(encoding="utf-8", errors="replace")[:_MAX_READ]
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    def list_dir(path: str = ".", **_) -> str:
        try:
            d = _safe(path)
            items = sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir()
                           if x.name not in (".git",))
            return "\n".join(items) or "(empty)"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    def search_files(pattern: str, path: str = ".", **_) -> str:
        r = runner.run(["grep", "-rn", "--include=*.py", "-e", pattern, str(_safe(path))],
                       cwd=root, timeout=20)
        return (r.stdout or "no matches")[:_MAX_OUT]

    # Track the last check failure — if the SAME error persists across attempts
    # while the model keeps editing the impl, the bug is likely in the TEST itself.
    _last_fail_key: list[str | None] = [None]
    _same_fail_count: list[int] = [0]

    def _check_after_edit(action: str, path: str) -> str:
        """Ground truth after any edit: run the ladder and report. This kills the
        'edit-hope-edit' loop by making it impossible for the model to assume its
        change worked — every edit's return value carries the real check result."""
        from agent.coding.verify.checks import run_ladder
        changed = _rel_changed_py(root)
        if not changed:
            return f"{action}"
        res = run_ladder(root, [root / c for c in changed], events=events)
        note = f" [{res.note}]" if res.note else ""  # surface auto-installs to the model
        if res.ok:
            _last_fail_key[0] = None
            _same_fail_count[0] = 0
            return f"{action}. Checks now PASS.{note} If the goal is met, call done."

        # Same failure again? nudge that the TEST itself may be wrong.
        key = f"{res.stage}:{res.error[:200]}"
        if key == _last_fail_key[0]:
            _same_fail_count[0] += 1
        else:
            _last_fail_key[0] = key
            _same_fail_count[0] = 1
        hint = ""
        if _same_fail_count[0] >= 2 and res.stage == "tests":
            hint = ("\nHINT: this same test error hasn't budged. The TEST itself is "
                    "likely wrong (e.g. `assert 'x' in locals()`, wrong expected value, "
                    "importing something that doesn't exist). Read the test file, "
                    "identify what it's actually asserting, and rewrite it if it's buggy.")
        return f"{action}. Checks FAIL at {res.stage}:{note} {res.error[:600]}{hint}"

    def write_file(path: str, content: str, **_) -> str:
        # Allow creating NEW test files (the agent should write tests), but never
        # let it edit an EXISTING test to make failing code pass.
        if (guards.is_test_file(path)
                and config.get("guards", {}).get("test_files_readonly", True)
                and _safe(path).exists()):
            return f"refused: {path} is an existing test file (read-only). Fix the code instead."
        t = _safe(path)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
        events.emit("edit", path=path, bytes=len(content))
        return _check_after_edit(f"wrote {path} ({len(content)} chars)", path)

    def replace_in_file(path: str, search: str, replace: str, **_) -> str:
        if guards.is_test_file(path) and config.get("guards", {}).get("test_files_readonly", True):
            return f"refused: {path} is a test file (read-only)."
        t = _safe(path)
        try:
            src = t.read_text(encoding="utf-8")
        except OSError:
            return f"error: {path} does not exist"
        if search not in src:
            return "error: SEARCH text not found exactly — read_file first and copy it byte for byte."
        t.write_text(src.replace(search, replace, 1), encoding="utf-8")
        events.emit("edit", path=path)
        return _check_after_edit(f"edited {path}", path)

    def run(command: str, **_) -> str:
        level = classify_command(command, config)
        # Autonomous means "don't ask about safe stuff", NOT "let sudo run freely".
        # Anything DANGEROUS or above (sudo, dd, systemctl, mkfs, rm -rf /) is
        # refused unconditionally in the sandbox — the model has no legitimate
        # reason to escalate privileges, and letting sudo prompt hijacks the
        # user's terminal for a password.
        if level >= RiskLevel.DANGEROUS:
            events.emit("run_blocked", command=command)
            return (f"refused: '{command}' is a privileged/destructive command "
                    f"(risk={level.name}). It will never run. Never use sudo — "
                    "the sandbox has no root. For missing Python packages, just "
                    "`import` them and they auto-install.")
        # Keep the agent inside the project. Refuse commands that touch system
        # locations (installing a global command, writing to PATH/autostart/etc.).
        # "Works on other systems" = portable code, never a system install.
        if _SYSTEM_PATH_RE.search(command):
            events.emit("run_blocked", command=command)
            return ("refused: that writes outside the project (a system path like "
                    "/usr/local/bin, /etc, or ~/.local/bin). Keep everything inside "
                    "the project directory — do NOT install anything system-wide. "
                    "'Portable / works on other systems' means writing code that runs "
                    "anywhere, not installing it. The user installs it themselves.")
        # Dedupe: if the model asks to `pip install X` and X was already installed
        # this session (either by us or by an earlier `run`), skip.
        m = re.match(r"\s*(?:python\s+-m\s+)?pip\s+install\s+(?:--\S+\s+)*(\S+)\s*$", command)
        if m:
            from agent.coding.verify import checks as _c
            pkg = m.group(1).split("==")[0].strip()
            if pkg in _c._installed_this_session:
                return f"skipped: {pkg} was already installed this session. exit=0"
            _c._installed_this_session.add(pkg)  # record so future dupes short-circuit
        events.emit("run", command=command)
        env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
               "PYTHONDONTWRITEBYTECODE": "1"}
        r = runner.run(["bash", "-lc", command], cwd=root, timeout=timeout, mem_mb=mem, env=env)
        events.emit("run_done", command=command, rc=r.rc, timed_out=r.timed_out)
        tail = r.output[-_MAX_OUT:]
        return f"exit={r.rc}{' [timed out]' if r.timed_out else ''}\n{tail}"

    def run_checks(**_) -> str:
        from agent.coding.verify.checks import run_ladder
        changed = _rel_changed_py(root)
        res = run_ladder(root, [root / c for c in changed], events=events)
        note = f" [{res.note}]" if res.note else ""
        return f"checks PASS{note}" if res.ok else f"checks FAILED at {res.stage}:{note}\n{res.error}"

    def web_docs(query: str, **_) -> str:
        from agent.coding import webdocs
        allow = tuple(config.get("coding_web_allowlist", webdocs.ALLOWLIST))
        return webdocs.search_docs(query, events=events, allowlist=allow)

    _asked: dict[str, str] = {}
    def ask_user(question: str, **_) -> str:
        from agent.coding.ask import is_product_question
        model = config.get("executor", {}).get("model")
        if not is_product_question(question, llm=llm, model=model):
            return ("That's a code/technical/permission question — decide it yourself. "
                    "Just DO the action. Only ask me about product/design forks "
                    "(colours, wording, UX, scope).")
        key = " ".join(question.lower().split())[:200]
        if key in _asked:  # same question twice = the model isn't listening
            return (f"You already asked that. The answer was: {_asked[key]}. "
                    "Proceed based on that answer — do not ask again.")
        if ask_fn is None:
            return "No user is available; pick a sensible default and continue."
        events.emit("ask_user", question=question)
        answer = ask_fn(question)
        _asked[key] = answer
        return f"The user answered: {answer}"

    def _tool(name, desc, props, required, handler, risk="safe"):
        return Tool(name=name, description=desc, risk=risk,
                    parameters={"type": "object", "properties": props, "required": required},
                    handler=handler)

    reg = ToolRegistry()
    reg.register(_tool("read_file", "Read a file's contents.",
                       {"path": {"type": "string"}}, ["path"], read_file))
    reg.register(_tool("list_dir", "List a directory (default the repo root).",
                       {"path": {"type": "string"}}, [], list_dir))
    reg.register(_tool("search_files", "grep for a pattern across .py files.",
                       {"pattern": {"type": "string"}, "path": {"type": "string"}},
                       ["pattern"], search_files))
    reg.register(_tool("write_file", "Write the COMPLETE contents of a file (creates or overwrites).",
                       {"path": {"type": "string"}, "content": {"type": "string"}},
                       ["path", "content"], write_file))
    reg.register(_tool("replace_in_file", "Replace an exact snippet in a file (for large files).",
                       {"path": {"type": "string"}, "search": {"type": "string"},
                        "replace": {"type": "string"}}, ["path", "search", "replace"], replace_in_file))
    reg.register(_tool("run", "Run a shell command in the repo (build/run/test). Auto-runs; "
                       "destructive commands are refused.",
                       {"command": {"type": "string"}}, ["command"], run))
    reg.register(_tool("run_checks", "Run the check ladder (syntax, lint, tests) on your changes.",
                       {}, [], run_checks))
    reg.register(_tool("web_docs", "Search coding sites (StackOverflow, official docs, MDN, "
                       "GitHub…) and read the pages. Use when stuck on an API, error, or library.",
                       {"query": {"type": "string"}}, ["query"], web_docs))
    reg.register(_tool("ask_user", "Ask the human a PRODUCT/design decision only (colours, "
                       "wording, UX, scope). Never ask code questions — decide those yourself.",
                       {"question": {"type": "string"}}, ["question"], ask_user))
    reg.register(_tool("done", "Call when the goal is met AND your tests pass. Triggers final verification.",
                       {"summary": {"type": "string"}}, [], lambda **_: "finalizing"))
    return reg
