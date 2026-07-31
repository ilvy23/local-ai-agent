"""The autonomous coding loop.

Give it a goal; it works in a throwaway git worktree with a toolset (read/write
files, run commands, run the check ladder), iterating until it calls `done` and
the guards pass — or a step budget runs out. `done` is not trusted on its own:
final verification (ladder + coverage + LLM-judge) must pass, or the loop is told
to keep going. On success the work is auto-committed to a branch.
"""

from __future__ import annotations

import httpx

from agent.coding import config as coding_config
from agent.coding.session import Outcome
from agent.coding.tools import build_coding_tools
from agent.coding.verify import guards
from agent.coding.verify.checks import CheckResult, run_ladder
from agent.coding.workspace import Workspace
from agent.events import NULL, Events


_USAGE_PROMPT = """Someone just built this in a single-command autonomous run. Read the diff \
and write a short user-facing note (max 8 lines, plain text, no code fences) covering:
- what was built
- exactly how to run it (a shell command or `python file.py`)
- what to expect / example input

Diff:
{diff}"""


def _usage_note(llm, model: str, diff: str) -> str:
    try:
        out = "".join(llm.chat(
            [{"role": "user", "content": _USAGE_PROMPT.format(diff=diff[:8000])}],
            model=model, temperature=0.2))
        return out.strip() or ""
    except Exception:  # noqa: BLE001 - a broken usage note must not fail the run
        return ""
from agent.jsonx import extract_json_value

_SYSTEM = """You are an autonomous software engineer working inside a git repository \
(your current directory). You have tools to read/list/search files, write and edit \
files, run shell commands, and run the check ladder.

Work toward the goal on your own:
- Do NOT ask setup questions ("should we start?", "should I create the file?"). Just start.
  Only use ask_user for real product/design forks (colours, wording, UX scope).
- Explore first (list_dir, read_file) before changing anything.
- Make changes with write_file / replace_in_file. Write across as many files as needed.
- Prove it works. For a library/module → write test_<name>.py with real assertions.
  For a runnable script/GUI/game → put your logic under `def main()` and call it from
  `if __name__ == '__main__': main()`; the checks will run your script for a few
  seconds under headless SDL and pass if it doesn't crash.
- Missing third-party packages (pygame, requests, numpy, …) are installed for you
  automatically when tests fail with ModuleNotFoundError. Just `import` them and
  keep going — do NOT run `pip install` yourself unless a check tells you to.
- Debug from real output: run the program, read the error, fix, re-run.
- Decide every CODE question yourself — libraries, syntax, structure, naming.
- Stay INSIDE the project directory. Never write, copy, or install files to system
  locations (/usr/local/bin, /usr/bin, /etc, ~/.local/bin, PATH dirs, cron,
  autostart, services). "Portable / works on other systems" means code that RUNS
  anywhere, NOT installing it system-wide — that's the user's step, not yours.
- Call done ONLY when the goal is met and your tests pass.

Use one tool per step. Do not explain — act."""


import re as _re

_BASH_FENCE = _re.compile(r"```(?:bash|sh|shell)\s*\n(?P<cmd>.+?)\n```", _re.DOTALL)


def _extract_call(message: dict) -> dict | None:
    """Get a tool call from Ollama's structured field, or from what the model
    printed. Local coders vary: some use `tool_calls`, some print
    `{"name":..., "arguments":...}`, some `{"tool":...}`, and some (e.g.
    Qwen3-Coder-30B on Ollama) emit shell commands as a ```bash``` fence."""
    calls = message.get("tool_calls")
    if calls:
        fn = calls[0].get("function", {})
        return {"name": fn.get("name"), "arguments": fn.get("arguments", {}) or {}}

    content = message.get("content", "") or ""
    obj = extract_json_value(content, "{", "}")
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("tool")
        if name:
            args = obj.get("arguments", obj.get("args", {}))
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"name": name, "arguments": args if isinstance(args, dict) else {}}

    # Bash-fence fallback: a single ```bash``` block with no other tool call
    # gets treated as a `run` invocation. Only fires when there's exactly one
    # fence, so a `write_file` payload that happens to contain a bash example
    # isn't mistaken for a shell command.
    fences = _BASH_FENCE.findall(content)
    if len(fences) == 1:
        return {"name": "run", "arguments": {"command": fences[0].strip()}}
    return None


def _finalize(ws, cfg: dict, goal: str, llm, events: Events) -> CheckResult:
    from agent.coding.tools import _rel_changed_py

    changed = _rel_changed_py(ws.path)
    if not changed:
        return CheckResult(False, "empty", "No code was changed yet.")
    res = run_ladder(ws.path, [ws.path / c for c in changed], events=events)
    if not res.ok:
        return res
    g = cfg.get("guards", {})
    # If smoke-run (not pytest) validated the code, coverage is meaningless: the
    # 'tests' are stubs and the real proof is that the script runs. Skip it.
    smoke_validated = "smoke:" in (res.note or "")
    if g.get("coverage_gate", True) and not smoke_validated:
        cov = guards.coverage_gate(ws, changed)
        if not cov.ok:
            return cov
    if g.get("llm_judge", True):
        v = guards.judge_diff(llm, cfg, goal, ws.diff())
        if not v.ok:
            return CheckResult(False, "judge", v.reason)
    return CheckResult(True, "all")


def run_goal(repo_root, goal: str, config: dict | None = None, llm=None,
             events: Events = NULL, max_steps: int = 40, ask_fn=None) -> Outcome:
    cfg = coding_config.load(config)
    if llm is None:
        from agent.llm import OllamaClient
        llm = OllamaClient()
    ex = cfg["executor"]

    from agent.coding.verify.checks import reset_install_budget
    reset_install_budget()  # fresh per-session pip-install budget

    with Workspace(repo_root) as ws:
        registry = build_coding_tools(ws, cfg, events, llm=llm, ask_fn=ask_fn)
        schema = registry.to_ollama_schema()
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Goal: {goal}"},
        ]
        import json as _json
        seen: dict[str, int] = {}   # repeated identical actions = the model is stuck
        trail: list[str] = []
        ctx = ex["num_ctx"]

        for step in range(max_steps):
            events.emit("step", n=step + 1, of=max_steps)
            try:
                message = llm.chat_with_tools(messages, model=ex["model"], tools=schema, num_ctx=ctx)
            except httpx.HTTPStatusError as exc:
                # 500 = the runner OOM-crashed (KV cache too big for VRAM). Halve
                # the context and retry rather than aborting the whole run.
                if getattr(exc.response, "status_code", 0) == 500 and ctx > 8192:
                    ctx = max(8192, ctx // 2)
                    events.emit("ctx_backoff", num_ctx=ctx)
                    continue
                events.emit("finished", ok=False)
                return Outcome(ok=False, stage="crash", diff=ws.diff(), attempts=step + 1,
                               error=f"model runner error: {exc}", trail=trail)
            messages.append(message if isinstance(message, dict)
                            else {"role": "assistant", "content": str(message)})
            call = _extract_call(message if isinstance(message, dict) else {"content": str(message)})

            if call is None:
                messages.append({"role": "user",
                                 "content": "Call a tool, or call done() if the goal is met."})
                continue

            name, args = call["name"], call["arguments"]
            events.emit("tool_call", name=name, args=args)
            trail.append(name)

            # Stall guard: two triggers so a wandering model can't dodge it forever.
            # (a) identical sig 7×  → deep rut on one exact call
            # (b) same tool 8× within the last 10 steps → thrashing on one operation
            #     (this is what the game-test attempt hit — many different edits to
            #     the same file, each with new args)
            sig = name + _json.dumps(args, sort_keys=True, default=str)[:300]
            seen[sig] = seen.get(sig, 0) + 1
            window = trail[-10:]
            same_tool_in_window = window.count(name)
            if seen[sig] == 4 or same_tool_in_window == 6:
                messages.append({"role": "user", "content":
                    "You have made no progress. Try a DIFFERENT approach: read the actual "
                    "error from run_checks, use web_docs, or restructure the code so it's "
                    "testable (e.g. make input a function argument you can pass in tests)."})
            elif seen[sig] >= 7 or same_tool_in_window >= 8:
                events.emit("stall", action=name)
                diff = ws.diff()
                # Keep the WIP branch on stall too — the model often built usable
                # code before churning on tests, and losing it is worse than a
                # slightly noisier branch list. Commit under a WIP prefix.
                if diff and cfg.get("auto_commit", True):
                    ws.commit(f"agent code WIP (stalled): {goal[:50]}")
                    ws.keep = True
                return Outcome(ok=False, stage="stuck", diff=diff, attempts=step + 1,
                               branch=ws.branch if ws.keep else "",
                               error=f"stalled — {name} ×{same_tool_in_window} in last 10 steps",
                               trail=trail)

            if name == "done":
                res = _finalize(ws, cfg, goal, llm, events)
                if res.ok:
                    diff = ws.diff()  # capture before committing (commit clears the staged diff)
                    usage = _usage_note(llm, ex["model"], diff)
                    if cfg.get("auto_commit", True):
                        ws.commit(f"agent code: {goal[:60]}")
                        ws.keep = True
                    events.emit("finished", ok=True, usage=usage)
                    return Outcome(ok=True, stage="done", diff=diff,
                                   branch=ws.branch if ws.keep else "", attempts=step + 1,
                                   trail=[*trail, "usage: " + usage[:200]] if usage else trail)
                messages.append({"role": "tool",
                                 "content": f"Not done — verification failed at {res.stage}:\n"
                                            f"{res.error}\nKeep working."})
                events.emit("done_rejected", stage=res.stage)
                continue

            tool = registry.get(name)
            if tool is None:
                result = f"no such tool '{name}'"
            else:
                try:
                    result = tool.handler(**args)
                except Exception as exc:  # noqa: BLE001 - feed tool errors back to the model
                    result = f"tool error: {exc}"
            events.emit("tool_result", name=name, result=str(result)[:160])
            messages.append({"role": "tool", "content": str(result)[:6000]})

        events.emit("finished", ok=False)
        return Outcome(ok=False, stage="budget", error="step budget exhausted",
                       diff=ws.diff(), attempts=max_steps, trail=trail)
