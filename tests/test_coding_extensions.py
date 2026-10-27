"""P8: Coding mode extensions — budget, skill persistence, trajectory.

Tests verify that the three systems wired in Priority 8 are actually called:
- Budget: BudgetExhausted stops the agent loop and returns a failed Outcome
- Skills: find_similar_skill results appear in the LLM system prompt; save_skill
  is called after a successful run_once
- Trajectory: log_action is called for every tool executed in the coding loop
"""

from __future__ import annotations

import subprocess

import pytest


# ── shared fixture ────────────────────────────────────────────────────────────

def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "e@e"], tmp_path)
    _git(["config", "user.name", "e"], tmp_path)
    (tmp_path / "README.md").write_text("# proj\n")
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-q", "-m", "init"], tmp_path)
    return tmp_path


# ── budget exhaustion ─────────────────────────────────────────────────────────

class _InfiniteLLM:
    """Always emits list_dir so the loop keeps running until budget fires."""

    def chat_with_tools(self, messages, model, tools, num_ctx=None):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "list_dir", "arguments": {}}}],
        }

    def chat(self, messages, model, **kwargs):
        yield "VERDICT: SOLVES"


def test_budget_exhausted_stops_loop(repo):
    """When the action budget is exceeded the loop returns ok=False, stage='budget'."""
    from agent.coding.agent_loop import run_goal

    # max_actions=2 means the 3rd tool call raises BudgetExhausted
    cfg = {"coding": {"budget": {"max_actions": 2, "max_tokens": 999_999, "max_seconds": 9999}}}
    out = run_goal(repo, "test goal", config=cfg, llm=_InfiniteLLM(), max_steps=100)

    assert not out.ok
    assert out.stage == "budget"
    assert "action" in out.error.lower()


def test_budget_error_message_references_limit(repo):
    """The error message names the limit that was exceeded."""
    from agent.coding.agent_loop import run_goal

    cfg = {"coding": {"budget": {"max_actions": 1, "max_tokens": 999_999, "max_seconds": 9999}}}
    out = run_goal(repo, "test", config=cfg, llm=_InfiniteLLM(), max_steps=50)

    assert not out.ok
    # BudgetExhausted message: "action limit exceeded: 2/1"
    assert "/" in out.error


# ── skill lookup ──────────────────────────────────────────────────────────────

def test_skill_context_passed_to_attempt(monkeypatch, repo):
    """When find_similar_skill returns matches they are passed as skill_context to attempt()."""
    import agent.coding.session as sess_mod
    from agent.coding.session import Attempt
    from agent.coding.verify.checks import CheckResult
    from unittest.mock import MagicMock, patch

    fake_skill = {
        "title": "CSV Reader",
        "problem_description": "read csv files into dicts",
        "solution_steps": ["open file", "use DictReader"],
        "tags": ["csv"],
    }
    monkeypatch.setattr(sess_mod, "find_similar_skill", lambda store, q, **kw: [fake_skill])
    monkeypatch.setattr(sess_mod, "save_skill", lambda *a, **kw: 1)

    captured_context: list[str] = []

    def fake_attempt(ws, llm, cfg, task, files, test_target, prior_error,
                     events, seed=None, skill_context=""):
        # Capture the skill_context that run_once constructed and passed
        captured_context.append(skill_context)
        return Attempt(ok=False, result=CheckResult(False, "test", "fails"))

    mock_store = MagicMock()
    # Commit target.py so the worktree at HEAD has it
    (repo / "target.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "add target"], repo)

    with patch("agent.coding.session.run_ladder") as mock_ladder, \
         patch("agent.coding.session.attempt", fake_attempt):
        mock_ladder.return_value = CheckResult(False, "test", "tests fail")
        sess_mod.run_once(
            repo,
            "read csv files",
            ["target.py"],
            store=mock_store,
        )

    assert captured_context, "attempt() was never called"
    assert "CSV Reader" in captured_context[0], (
        f"Skill title not found in skill_context passed to attempt: {captured_context[0]!r}"
    )


def test_skill_context_contains_problem_and_steps(monkeypatch, repo):
    """The skill_context string includes the problem description and solution steps."""
    import agent.coding.session as sess_mod
    from agent.coding.session import Attempt
    from agent.coding.verify.checks import CheckResult
    from unittest.mock import MagicMock, patch

    fake_skill = {
        "title": "My Skill",
        "problem_description": "complicated-unique-problem",
        "solution_steps": ["unique-step-one", "unique-step-two"],
        "tags": [],
    }
    monkeypatch.setattr(sess_mod, "find_similar_skill", lambda *a, **kw: [fake_skill])
    monkeypatch.setattr(sess_mod, "save_skill", lambda *a, **kw: 1)

    captured_context: list[str] = []

    def fake_attempt(ws, llm, cfg, task, files, test_target, prior_error,
                     events, seed=None, skill_context=""):
        captured_context.append(skill_context)
        return Attempt(ok=False, result=CheckResult(False, "test", "fails"))

    (repo / "target.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "add target"], repo)

    with patch("agent.coding.session.run_ladder") as mock_ladder, \
         patch("agent.coding.session.attempt", fake_attempt):
        mock_ladder.return_value = CheckResult(False, "test", "tests fail")
        sess_mod.run_once(
            repo, "some task", ["target.py"],
            store=MagicMock(),
        )

    ctx = captured_context[0]
    assert "complicated-unique-problem" in ctx
    assert "unique-step-one" in ctx


# ── skill save ────────────────────────────────────────────────────────────────

def test_skill_saved_after_successful_run_once(monkeypatch, repo):
    """save_skill is called with the task as title/problem after a successful session."""
    import agent.coding.session as sess_mod
    from agent.coding.session import Attempt
    from agent.coding.verify.checks import CheckResult
    from unittest.mock import MagicMock, patch

    saved: list[dict] = []

    def _capture_save(store, title, problem, steps, tools=None, tags=None):
        saved.append({"title": title, "problem": problem, "tools": tools, "tags": tags})
        return 1

    monkeypatch.setattr(sess_mod, "find_similar_skill", lambda *a, **kw: [])
    monkeypatch.setattr(sess_mod, "save_skill", _capture_save)

    def fake_attempt(ws, llm, cfg, task, files, test_target, prior_error,
                     events, seed=None, skill_context=""):
        return Attempt(ok=True, result=CheckResult(True, "tests"), changed=list(files))

    mock_store = MagicMock()
    (repo / "target.py").write_text("x = 1\n")

    with patch("agent.coding.session.run_ladder") as mock_ladder, \
         patch("agent.coding.session.attempt", fake_attempt):
        mock_ladder.return_value = CheckResult(False, "test", "tests fail")
        out = sess_mod.run_once(
            repo,
            "read csv files into dict",
            ["target.py"],
            config={"coding": {"auto_commit": False}},
            store=mock_store,
        )

    assert out.ok, f"Expected success, got: {out}"
    assert len(saved) == 1, f"save_skill called {len(saved)} time(s), expected 1"
    assert "read csv files" in saved[0]["title"]
    assert saved[0]["tags"] == ["coding"]


def test_skill_not_saved_on_failure(monkeypatch, repo):
    """save_skill is NOT called when the session fails."""
    import agent.coding.session as sess_mod
    from agent.coding.session import Attempt
    from agent.coding.verify.checks import CheckResult
    from unittest.mock import MagicMock, patch

    saved: list[dict] = []
    monkeypatch.setattr(sess_mod, "find_similar_skill", lambda *a, **kw: [])
    monkeypatch.setattr(sess_mod, "save_skill", lambda *a, **kw: saved.append(kw) or 1)

    def fake_attempt_fail(ws, llm, cfg, task, files, test_target, prior_error,
                          events, seed=None, skill_context=""):
        return Attempt(ok=False, result=CheckResult(False, "tests", "still failing"))

    mock_store = MagicMock()
    (repo / "target.py").write_text("x = 1\n")

    with patch("agent.coding.session.run_ladder") as mock_ladder, \
         patch("agent.coding.session.attempt", fake_attempt_fail):
        mock_ladder.return_value = CheckResult(False, "test", "tests fail")
        out = sess_mod.run_once(
            repo, "do something", ["target.py"],
            config={"coding": {"auto_commit": False}},
            store=mock_store,
        )

    assert not out.ok
    assert saved == [], "save_skill should not be called on failure"


# ── trajectory ────────────────────────────────────────────────────────────────

def test_trajectory_logged_per_tool_call(monkeypatch, repo):
    """log_action is called for every non-done tool executed in the coding loop."""
    import agent.coding.agent_loop as loop_mod

    logged: list[dict] = []

    def _fake_log(action, tool, input_summary, output_summary, session_id=None):
        logged.append({"action": action, "tool": tool})

    monkeypatch.setattr(loop_mod, "log_action", _fake_log)

    class ScriptedLLM:
        def __init__(self, script):
            self._script = list(script)

        def chat_with_tools(self, messages, model, tools, num_ctx=None):
            name, args = self._script.pop(0) if self._script else ("done", {})
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}],
            }

        def chat(self, messages, model, **kwargs):
            yield "VERDICT: SOLVES"

    llm = ScriptedLLM([("list_dir", {}), ("done", {})])
    from agent.coding.agent_loop import run_goal
    run_goal(repo, "test trajectory logging", llm=llm, max_steps=10)

    # list_dir should have been logged; 'done' is a guard call, not a tool execution
    assert any(r["tool"] == "list_dir" for r in logged), (
        f"Expected list_dir in trajectory, got: {logged}"
    )
    assert all(r["action"] == "tool_call" for r in logged)


def test_trajectory_not_logged_for_done(monkeypatch, repo):
    """The 'done' pseudo-tool is NOT logged via trajectory (it goes through _finalize)."""
    import agent.coding.agent_loop as loop_mod

    logged: list[dict] = []
    monkeypatch.setattr(loop_mod, "log_action",
                        lambda action, tool, input_summary, output_summary, session_id=None:
                        logged.append(tool))

    class DoneImmediatelyLLM:
        def chat_with_tools(self, messages, model, tools, num_ctx=None):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "done", "arguments": {}}}],
            }

        def chat(self, messages, model, **kwargs):
            yield "VERDICT: SOLVES"

    from agent.coding.agent_loop import run_goal
    run_goal(repo, "done only", llm=DoneImmediatelyLLM(), max_steps=5)

    assert "done" not in logged, "'done' must not appear in trajectory log"


# ── budget: token axis ────────────────────────────────────────────────────────

class _TokenHungryLLM:
    """Reports Ollama-style token counts on every reply, like _post_chat does."""

    def __init__(self, prompt_tokens: int = 400, eval_tokens: int = 100) -> None:
        self._p, self._e = prompt_tokens, eval_tokens

    def chat_with_tools(self, messages, model, tools, num_ctx=None):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "list_dir", "arguments": {}}}],
            "_prompt_eval_count": self._p,
            "_eval_count": self._e,
        }

    def chat(self, messages, model, **kwargs):
        yield "VERDICT: SOLVES"


def test_token_budget_stops_the_loop(repo):
    """coding.budget.max_tokens was inert: nothing ever fed tick() any tokens."""
    from agent.coding.agent_loop import run_goal

    cfg = {"coding": {"budget": {"max_actions": 999, "max_tokens": 1200,
                                 "max_seconds": 9999}}}
    out = run_goal(repo, "burn tokens", config=cfg, llm=_TokenHungryLLM(), max_steps=100)

    assert not out.ok
    assert out.stage == "budget"
    assert "token" in out.error.lower()


def test_token_counts_are_optional(repo):
    """A client that reports no counts must not break the loop or the budget."""
    from agent.coding.agent_loop import run_goal

    cfg = {"coding": {"budget": {"max_actions": 3, "max_tokens": 10, "max_seconds": 9999}}}
    out = run_goal(repo, "no counts", config=cfg, llm=_InfiniteLLM(), max_steps=50)

    assert not out.ok
    assert "action" in out.error.lower()  # actions ran out, not tokens


def test_message_tokens_handles_junk_values():
    from agent.coding.agent_loop import _message_tokens

    assert _message_tokens({"_prompt_eval_count": 5, "_eval_count": 7}) == 12
    assert _message_tokens({}) == 0
    assert _message_tokens("not a dict") == 0
    assert _message_tokens({"_prompt_eval_count": None, "_eval_count": None}) == 0
    assert _message_tokens({"_prompt_eval_count": "abc", "_eval_count": 4}) == 4


# ── trajectory: secrets never reach the log ──────────────────────────────────

def test_tool_args_are_redacted_in_the_trajectory_log(monkeypatch, repo, tmp_path):
    """Tool arguments carrying a credential must be scrubbed before they hit disk."""
    import agent.trajectory as traj_mod
    from agent.coding.agent_loop import run_goal

    log_path = tmp_path / "trajectory.jsonl"
    monkeypatch.setattr(traj_mod, "_TRAJECTORY_PATH", log_path)

    secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"

    class SecretLeakingLLM:
        def __init__(self):
            self._done = False

        def chat_with_tools(self, messages, model, tools, num_ctx=None):
            name = "done" if self._done else "run"
            self._done = True
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name,
                                             "arguments": {"command": f"echo {secret}"}}}],
            }

        def chat(self, messages, model, **kwargs):
            yield "VERDICT: SOLVES"

    run_goal(repo, "leak a secret", llm=SecretLeakingLLM(), max_steps=3)

    written = log_path.read_text() if log_path.exists() else ""
    assert written, "expected the coding loop to write a trajectory record"
    assert secret not in written
