"""Step 3+4: guards and the repair loop, driven by a fake LLM (no Ollama)."""

from __future__ import annotations

import subprocess

import pytest

from agent.coding import repair
from agent.coding.verify import guards
from agent.coding.workspace import Workspace


class FakeLLM:
    """Returns `edit_reply` for edit requests, and a judge verdict for judge calls."""

    def __init__(self, edit_reply: str, verdict: str = "VERDICT: SOLVES"):
        self.edit_reply = edit_reply
        self.verdict = verdict

    def chat(self, messages, model, num_ctx=None, num_gpu=None, temperature=None, extra_options=None):
        prompt = messages[-1]["content"]
        yield self.verdict if "VERDICT:" in prompt else self.edit_reply


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "t@t"], tmp_path)
    _git(["config", "user.name", "t"], tmp_path)
    (tmp_path / "mod.py").write_text("def add(a, b):\n    return a - b\n")  # bug
    (tmp_path / "test_mod.py").write_text(
        "from mod import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-q", "-m", "init"], tmp_path)
    return tmp_path


GOOD_FIX = "### FILE: mod.py\n```python\ndef add(a, b):\n    return a + b\n```"


def test_repair_fixes_and_guards_pass(repo):
    out = repair.run(repo, "fix add", ["mod.py"], llm=FakeLLM(GOOD_FIX))
    assert out.ok and out.stage == "fixed", out
    assert out.branch  # kept for the user
    branches = subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout
    assert out.branch in branches


def test_guard_rejects_editing_the_test(repo):
    cheat = "### FILE: test_mod.py\n```python\ndef test_add():\n    assert True\n```"
    out = repair.run(repo, "fix add", ["mod.py"], llm=FakeLLM(cheat))
    assert not out.ok  # never let it win by neutering the test
    assert any("test" in t.lower() and "reject" in t.lower() for t in out.trail), out.trail


def test_judge_blocks_gaming(repo):
    # A "fix" that passes the test but the judge condemns.
    gamed = "### FILE: mod.py\n```python\ndef add(a, b):\n    return 5  # hardcoded\n```"
    out = repair.run(repo, "fix add", ["mod.py"],
                     llm=FakeLLM(gamed, verdict="VERDICT: GAMES — hardcoded 5"))
    assert not out.ok
    assert any("judge" in t.lower() for t in out.trail), out.trail


def test_coverage_gate_flags_unexecuted_change(tmp_path):
    # A correct, tested baseline; then add a function no test calls. Its body is
    # never executed, so the gate must flag the change as hollow.
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "t@t"], tmp_path)
    _git(["config", "user.name", "t"], tmp_path)
    (tmp_path / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-q", "-m", "init"], tmp_path)

    with Workspace(tmp_path) as ws:
        ws.write("mod.py", "def add(a, b):\n    return a + b\n\n\ndef helper():\n    return 42\n")
        res = guards.coverage_gate(ws, ["mod.py"])
        assert not res.ok and "mod.py" in res.error  # helper's body is never executed
