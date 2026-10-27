"""Intake: ask the human a few clarifying questions BEFORE a non-trivial task starts, and give the model a way to ask later.

A plain prompt like "make a snake game with a menu, high scores and sound" hides decisions that change what gets built (language, platform,
what "etc." covers). One non-thinking call on the MAIN model (the one the engine uses: switching to the small background model would evict the
main model from the GPU) looks at the task plus the repo profile and returns 0-3 short questions, or none for a clear task. The answers are
appended to the task text ("Clarifications from the user:") and, because the text then carries them, the brief card (brief.py) pins them for
the whole task: compaction and stage contexts keep them.

    clarify(engine, task, cfg=...) -> list[str]     the model call; strict validation, any failure -> []
    run(engine, task, read=None) -> str             the whole intake: should it ask, ask, compose; returns the task text to work on

Config (coding.*): intake 'auto' (default: tasks of intake_min_chars or more) | 'off' | 'always', intake_min_chars (default 12).
A short vague prompt ("make a todo app") is exactly the ambiguous kind, so the length bar is low and the MODEL decides (it returns [] for a clear
task). To stay cheap and quiet, `looks_specific` skips a task that names an existing file / symbol / error in a repo that already has code;
an empty repo or greenfield wording ("make / build / create / write me a ...") always runs it.
Only on an interactive terminal; never twice for the same task text in a session; never for a resumed stage plan or a slash command.
The same module holds the texts of the `ask_user` tool (engine.py) so that both ways of asking agree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Any, Callable

from rich.markup import escape

from agent.codeagent.brief import CLARIFY_HEADER as HEADER
YOUR_CALL = "your call (state your assumption in the summary)"
NO_HUMAN = "No human available now. Decide yourself, state the assumption in your final summary."
LIMIT_REACHED = "Question limit reached; decide yourself."
NO_ANSWER = "(no answer: decide yourself and state the assumption in your final summary)"
MAX_QUESTIONS, MAX_QUESTION_CHARS = 3, 200
MAX_ASKS = 3                                   # ask_user calls per prompt (per stage in a staged run)
DEFAULT_MIN_CHARS = 12
NUM_PREDICT = 300
SKIP_WORDS = frozenset({"", "s", "skip"})
GO_WORDS = frozenset({"go"})
_DIRECTIVES = ("/", "AUTOPILOT", "[harness]")

SYSTEM = """You prepare a coding task. Decide whether the developer must answer something BEFORE the work starts.
Ask ONLY about details that would change what gets built and that neither the task nor the repository answers: the name, language or \
framework when the repo does not fix it, the target platform, the scope of an open-ended "etc.", conflicting requirements.
For a greenfield task ("make a todo app" in an empty folder) ask about the 1-3 decisions that most change the result: the language / stack \
(unless the folder or the task implies it), the interface (CLI, GUI or web), whether data must persist (and where), and the scope (the \
must-have features). Skip any of these that the task already states.
Return [] (no questions) for a clear task, for a bug fix with a clear symptom, for a small change, and for anything the repository answers. \
Never ask about style, naming of internals, tests, or whether to proceed.
Output ONLY JSON: {"questions": ["<short question>", ...]} with 0 to 3 questions, each under 200 characters."""


def tty() -> bool:
    """Is a human at a terminal? (The one place tests replace.)"""
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _coding(cfg: Any) -> dict[str, Any]:
    return (cfg or {}).get("coding") or {}


def mode(cfg: Any) -> str:
    m = str(_coding(cfg).get("intake", "auto")).strip().lower()
    return m if m in ("off", "always") else "auto"


def min_chars(cfg: Any) -> int:
    try:
        return max(0, int(_coding(cfg).get("intake_min_chars", DEFAULT_MIN_CHARS)))
    except (TypeError, ValueError):
        return DEFAULT_MIN_CHARS


def wanted(task: Any, cfg: Any) -> bool:
    """Config and text only: is this prompt one that gets intake? (A terminal and 'not asked before' are checked by `run`.)"""
    t = str(task or "").strip()
    m = mode(cfg)
    if m == "off" or not t or t.startswith(_DIRECTIVES):
        return False
    return m == "always" or len(t) >= min_chars(cfg)


_GREENFIELD = re.compile(r"\s*(?:(?:please|pls|can you|could you|would you|i want you to|i need you to|let'?s|go ahead and)\s+)*"
                         r"(?:make|build|create|write|generate|develop|code|program|design)\s+(?:me\s+|us\s+)?(?:a|an|my|some|the)?\s*\w", re.I)
_CODE_EXTS = frozenset({"py", "js", "ts", "tsx", "jsx", "rs", "go", "c", "h", "cpp", "hpp", "cs", "java", "kt", "rb", "php", "sh", "lua", "nix", "toml",
                        "json", "yaml", "yml", "html", "css", "md", "txt", "sql", "ini", "cfg", "lock", "swift", "zig", "ex", "exs", "hs", "ml", "pl", "r"})
_PATHISH = re.compile(r"[\w./-]*\w\.([A-Za-z]{1,5})\b")
_SPECIFIC = re.compile(r"(?i:traceback|exception|\b\w*error\b|\bfix(?:es|ed|ing)?\b|\bbugs?\b|\bcrash(?:es|ed)?\b|\bregression\b)|\w\(\)|`[^`\n]+`|"
                       r"\b[a-z]+_[a-z_0-9]+\b|\b[A-Z][a-z0-9]+[A-Z]\w*\b")
_SKIP_DIRS = frozenset({"node_modules", "__pycache__", "venv", "target", "dist", "build"})


def has_code(engine: Any) -> bool:
    """Does the repo already have code? The profile's kinds, else a shallow look for source files."""
    if getattr(getattr(engine, "profile", None), "kinds", None):
        return True
    root = getattr(engine, "root", None)
    if root is None:
        return False
    base = str(root).rstrip(os.sep).count(os.sep)
    try:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS]
            if dirpath.count(os.sep) - base >= 2:
                dirs[:] = []
            if any(f.rsplit(".", 1)[-1].lower() in _CODE_EXTS - {"md", "txt", "lock"} for f in files if "." in f):
                return True
    except OSError:
        pass
    return False


def looks_specific(task: str, engine: Any) -> bool:
    """True when the intake is not worth a model call: the repo already has code AND the task names something concrete in it (a file with
    an extension, a `symbol` / name() / snake_case / CamelCase, an error or traceback, or says fix / bug). An empty repo and greenfield wording
    ("make / build / create / write me a ...") are never specific: that is where the open decisions are."""
    t = str(task or "").strip()
    if _GREENFIELD.match(t) or not has_code(engine):
        return False
    if any(m.group(1).lower() in _CODE_EXTS for m in _PATHISH.finditer(t)):
        return True
    return bool(_SPECIFIC.search(t))


def key_of(task: str) -> str:
    return hashlib.sha1(" ".join(str(task).split()).encode("utf-8")).hexdigest()


def parse(reply: str) -> list[str]:
    """The questions of a model reply. Strict: a JSON object whose `questions` is a list of at most 3 non-empty strings of at most 200
    characters each; anything else is []."""
    try:
        text = str(reply)
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        items = data["questions"]
        if not isinstance(items, list) or len(items) > MAX_QUESTIONS:
            return []
        out = [i.strip() for i in items if isinstance(i, str)]
        if len(out) != len(items) or not all(out) or any(len(q) > MAX_QUESTION_CHARS for q in out):
            return []
        return out
    except Exception:  # noqa: BLE001 - no questions is always a safe answer
        return []


def _context(engine: Any) -> str:
    """The repo profile (language, tools), the top-level files and a short repo map: what the model needs to know what the repo already answers."""
    parts: list[str] = []
    try:
        parts.append(engine.profile.text())
    except Exception:  # noqa: BLE001
        pass
    try:
        names = sorted(n for n in os.listdir(engine.root) if not n.startswith("."))[:40]
        parts.append("FILES: " + (", ".join(names) or "(empty folder)"))
    except OSError:
        pass
    try:
        from agent.codeagent import edits

        parts.append("REPO MAP:\n" + edits.repo_map(engine.root, 1200))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(parts)


def main_model(target: Any, cfg: Any) -> str:
    models = (cfg or {}).get("models") or {}
    return getattr(target, "model", None) or ((cfg or {}).get("code") or {}).get("model") or models.get("chat") or models.get("background") or ""


def clarify(target: Any, task: str, *, cfg: Any = None, model: str | None = None) -> list[str]:
    """0-3 questions for `task`. `target` is the CodeEngine (its llm, model, config and repo profile are used) or a bare llm client."""
    try:
        llm = getattr(target, "llm", target)
        cfg = cfg if cfg is not None else getattr(target, "config", {})
        user = f"TASK:\n{str(task)[:6000]}\n\n{_context(target) if hasattr(target, 'root') else ''}".strip()
        reply = "".join(llm.chat(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                 model=model or main_model(target, cfg), stream=False, background=True,
                                 extra_options={"num_predict": NUM_PREDICT}))
        return parse(reply)
    except Exception:  # noqa: BLE001 - a missing question must never block the task
        return []


def compose(task: str, answers: list[tuple[str, str]]) -> str:
    """The task text plus the numbered clarifications (unanswered ones say that the choice is the model's)."""
    if not answers:
        return task
    lines = [f"{n}. {q} -> {a or YOUR_CALL}" for n, (q, a) in enumerate(answers, 1)]
    return f"{task.rstrip()}\n\n{HEADER}\n" + "\n".join(lines)


def ask_all(console: Any, questions: list[str], read: Callable[[str], str]) -> list[tuple[str, str]]:
    """Print the questions numbered (dim), then read ONE line per question: Enter or 's' skips it, 'go' stops asking (the rest is skipped)."""
    console.print("[dim]A few questions before I start (Enter or s = skip, go = stop asking):[/dim]")
    for n, q in enumerate(questions, 1):
        console.print(f"[dim]  {n}. {escape(q)}[/dim]")
    out: list[tuple[str, str]] = []
    stopped = False
    for n, q in enumerate(questions, 1):
        ans = ""
        if not stopped:
            try:
                raw = str(read(f"  {n} ▸ ")).strip()
            except (EOFError, KeyboardInterrupt):
                raw, stopped = "go", True
            if raw.lower() in GO_WORDS:
                stopped = True
            elif raw.lower() not in SKIP_WORDS:
                ans = " ".join(raw.split())
        out.append((q, ans))
    return out


def run(engine: Any, task: str, *, read: Callable[[str], str] | None = None) -> str:
    """The intake for one new task. Returns the text to work on: `task` itself when nothing was asked, else `task` + the clarifications."""
    cfg = engine.config
    if not wanted(task, cfg) or not tty():
        return task
    if mode(cfg) != "always" and looks_specific(task, engine):
        return task                                            # names a file / symbol / error in a repo with code: nothing open to ask
    plan = getattr(engine, "stage_plan", None)
    if plan is not None and plan.unfinished():
        return task                                            # a resumed stage plan: the task text is ignored anyway
    key = key_of(task)
    seen = engine.__dict__.setdefault("intake_seen", set())
    if key in seen:
        return task
    seen.add(key)
    with engine.console.status("[dim]reading your request...[/dim]", spinner="dots"):
        questions = clarify(engine, task, cfg=cfg)
    if not questions:
        return task
    answers = ask_all(engine.console, questions, read or (lambda prompt: engine.console.input(prompt)))
    engine.events.emit("intake", questions=len(questions), answered=sum(1 for _q, a in answers if a))
    return compose(task, answers)
