"""P-B: the autonomous agent loop + sandbox tool safety (fake LLM, no Ollama)."""

from __future__ import annotations

import subprocess

import pytest

from agent.coding.agent_loop import run_goal
from agent.coding.tools import build_coding_tools
from agent.coding.workspace import Workspace


class FakeToolLLM:
    """Emits a scripted sequence of tool calls; `chat` returns a judge verdict."""

    def __init__(self, script, verdict="VERDICT: SOLVES"):
        self.script = list(script)
        self.verdict = verdict

    def chat_with_tools(self, messages, model, tools, num_ctx=None):
        name, args = self.script.pop(0) if self.script else ("done", {})
        return {"role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}]}

    def chat(self, messages, model, num_ctx=None, num_gpu=None, temperature=None, extra_options=None):
        yield self.verdict


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


def test_agent_builds_writes_tests_and_verifies(repo):
    script = [
        ("list_dir", {}),
        ("write_file", {"path": "mod.py", "content": "def multiply(a, b):\n    return a * b\n"}),
        ("write_file", {"path": "test_mod.py",
                        "content": "from mod import multiply\n\n\ndef test_m():\n    assert multiply(2, 3) == 6\n"}),
        ("done", {}),
    ]
    out = run_goal(repo, "add a multiply function with a test", llm=FakeToolLLM(script))
    assert out.ok and out.stage == "done", out
    assert out.branch
    branches = subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout
    assert out.branch in branches


def test_done_rejected_when_change_is_hollow(repo):
    # Writes code but no test that exercises it → coverage gate blocks the `done`.
    script = [
        ("write_file", {"path": "mod.py", "content": "def unused():\n    return 42\n"}),
        ("write_file", {"path": "test_mod.py", "content": "def test_trivial():\n    assert True\n"}),
        ("done", {}),
    ]
    # only 4 steps allowed; after done is rejected it runs out → not ok
    out = run_goal(repo, "add unused()", llm=FakeToolLLM(script), max_steps=4)
    assert not out.ok


def test_run_tool_refuses_blocked_command(repo):
    with Workspace(repo) as ws:
        reg = build_coding_tools(ws, {"sandbox": {"timeout_seconds": 10}})
        result = reg.get("run").handler(command="rm -rf /")
        assert "refused" in result.lower()
        # a safe command still runs
        ok = reg.get("run").handler(command="echo hello")
        assert "hello" in ok and "exit=0" in ok


def test_run_tool_refuses_sudo_and_never_lets_it_prompt(repo):
    """The observed security bug: sudo apt-get ran and prompted the user for
    their password. sudo is DANGEROUS (not BLOCKED) so the old code let it
    through. It must be refused outright — the sandbox has no root."""
    with Workspace(repo) as ws:
        reg = build_coding_tools(ws, {"sandbox": {"timeout_seconds": 10}})
        for cmd in ("sudo apt-get update", "sudo -H pip install foo", "sudo rm x"):
            result = reg.get("run").handler(command=cmd)
            assert "refused" in result.lower(), f"{cmd!r} was NOT refused: {result}"
            assert "sudo" in result.lower()


def test_run_refuses_system_install_but_allows_reads(repo):
    """'make it work on Ubuntu systems' must NOT become a system install. Writes to
    /usr/local/bin etc. are refused; reading /etc to detect the distro is fine."""
    with Workspace(repo) as ws:
        reg = build_coding_tools(ws, {"sandbox": {"timeout_seconds": 10}})
        run = reg.get("run").handler
        for bad in ("cp app.py /usr/local/bin/foo.py",
                    "mv script /usr/bin/tool",
                    "echo x > /etc/rc.local",
                    "install -m755 app /usr/local/bin/app",
                    "crontab -l"):
            out = run(command=bad)
            assert "refused" in out.lower(), f"{bad!r} not refused: {out}"
        # a read of /etc still works (needed for 'which distro am I on')
        ok = run(command="cat /etc/hostname")
        assert "refused" not in ok.lower()


def test_runner_closes_stdin(tmp_path):
    """stdin=DEVNULL so a command trying to prompt can't hijack the terminal."""
    from agent.coding.verify.runner import run
    # `cat` reads stdin — with DEVNULL it hits EOF immediately and exits clean.
    r = run(["cat"], cwd=tmp_path, timeout=5)
    assert r.rc == 0 and not r.timed_out


class _Stub:
    def __init__(self, answer): self.answer = answer
    def print(self, *a, **k): pass
    def input(self, *a, **k): return self.answer


def test_ensure_repo_inits_when_missing(tmp_path, monkeypatch):
    from agent import coding_cli
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n")
    root = coding_cli._ensure_repo(_Stub("y"))
    assert root == tmp_path
    assert (tmp_path / ".git").exists()
    tracked = subprocess.run(["git", "ls-files"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "app.py" in tracked


def test_ensure_repo_declined_returns_none(tmp_path, monkeypatch):
    from agent import coding_cli
    monkeypatch.chdir(tmp_path)
    assert coding_cli._ensure_repo(_Stub("n")) is None
    assert not (tmp_path / ".git").exists()


def test_ensure_repo_commits_when_repo_has_no_commits(tmp_path, monkeypatch):
    from agent import coding_cli
    _git(["init", "-q"], tmp_path)  # a repo, but no HEAD yet — the original failure
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    root = coding_cli._ensure_repo(_Stub("y"))
    assert root == tmp_path
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                          capture_output=True).returncode == 0  # now has a commit


def test_too_broad_refuses_home_and_huge(monkeypatch, tmp_path):
    from agent import coding_cli
    from pathlib import Path
    assert coding_cli._too_broad(Path.home()) == "your home directory"
    assert coding_cli._too_broad(tmp_path) is None


def test_ignores_outer_broad_repo(tmp_path, monkeypatch):
    """The bug from the transcript: stray ~/.git makes agent code fail from a
    subfolder. Fix: ignore the outer too-broad repo and treat cwd as the project."""
    from agent import coding_cli
    _git(["init", "-q"], tmp_path)                    # simulate a stray outer .git
    monkeypatch.setattr(coding_cli, "_too_broad",
                        lambda p: "stray outer" if p == tmp_path else None)
    project = tmp_path / "myproj"
    project.mkdir()
    (project / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(project)
    root = coding_cli._ensure_repo(_Stub("y"))
    assert root == project                             # inited HERE, not walked up
    assert (project / ".git").exists()


def test_permission_questions_are_deflected():
    """The observed snake failure: model asked permission for every step. These
    are code/mechanical decisions the agent must make itself."""
    from agent.coding.ask import is_product_question
    for bad in [
        "Should I create the file `snake_game.py`?",
        "Should I install 'pytest' to run the checks?",
        "Do you want me to write a test?",
        "Can I add a game loop?",
        "Shall I use pygame or tkinter?",  # library choice = code decision
        "What should be the name of the test file?",  # naming = code decision
        "What should I call the module?",
        "What should the directory structure be?",
        "Should we write a more comprehensive test for the game logic?",  # the exact one from the log
        "Let's add a helper function, ok?",
        "Do we want to refactor this into a class?",
    ]:
        assert not is_product_question(bad), f"should be deflected: {bad!r}"
    for good in [
        "Should the button be red or blue?",
        "What text should appear on the title screen?",
        "Do you want scores saved between runs?",
    ]:
        # These are only deflected if the LLM classifier says CODE. Without a
        # classifier they pass; that's fine — the real user picks.
        assert is_product_question(good), f"should reach user: {good!r}"


def test_pip_install_dedupes_already_installed(tmp_path):
    """If the model runs `pip install X` after auto-install already got X, skip it —
    the model shouldn't spend 30s on a no-op."""
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "e@e"], tmp_path)
    _git(["config", "user.name", "e"], tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-q", "-m", "i"], tmp_path)
    from agent.coding.tools import build_coding_tools
    from agent.coding.verify import checks as C
    from agent.coding.workspace import Workspace
    C._installed_this_session.add("pygame")
    try:
        with Workspace(tmp_path) as ws:
            reg = build_coding_tools(ws, {"sandbox": {"timeout_seconds": 10}})
            result = reg.get("run").handler(command="pip install pygame")
            assert "skipped" in result.lower() and "pygame" in result
    finally:
        C._installed_this_session.discard("pygame")


def test_extract_bash_fence_as_run(tmp_path):
    """Qwen3-Coder-30B on Ollama emits shell commands as ```bash fences.
    A single bash fence with no other tool call must map to `run(command=…)`."""
    from agent.coding.agent_loop import _extract_call
    msg = {"content": "I'll list the files.\n\n```bash\nls -la\n```"}
    assert _extract_call(msg) == {"name": "run", "arguments": {"command": "ls -la"}}
    # ambiguity guard: two fences → don't guess
    msg2 = {"content": "```bash\nls\n```\n\n```bash\ncat x\n```"}
    assert _extract_call(msg2) is None


def test_no_stray_git_when_refusing(tmp_path, monkeypatch):
    """Guard fires BEFORE git init, so refusal leaves nothing behind."""
    from agent import coding_cli
    monkeypatch.setattr(coding_cli, "_too_broad", lambda p: "test-refusal" if p == tmp_path else None)
    monkeypatch.chdir(tmp_path)
    assert coding_cli._ensure_repo(_Stub("y")) is None
    assert not (tmp_path / ".git").exists()            # no stray .git left


def test_write_refuses_editing_existing_test(repo):
    (repo / "test_x.py").write_text("def test_x():\n    assert True\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "t"], repo)
    with Workspace(repo) as ws:
        reg = build_coding_tools(ws, {"guards": {"test_files_readonly": True}})
        result = reg.get("write_file").handler(path="test_x.py", content="def test_x():\n    assert False\n")
        assert "refused" in result.lower()
