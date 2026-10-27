"""Greenfield build flow: whole-program generation + repair (fake LLM)."""

from __future__ import annotations

import subprocess

import pytest

from agent.coding.build import build


class FakeLLM:
    """Yields scripted whole-program generations; `chat` answers the judge."""

    def __init__(self, rounds, verdict="VERDICT: SOLVES"):
        self.rounds = list(rounds)
        self.verdict = verdict

    def chat(self, messages, model, num_ctx=None, num_gpu=None, temperature=None, extra_options=None):
        content = messages[-1]["content"]
        if "VERDICT:" in content or "gaming" in content.lower() or "cheating" in content.lower():
            yield self.verdict
            return
        if "how to run" in content.lower() or "user-facing note" in content.lower():
            yield "Run it with: python app.py"
            return
        if "do not write code yet" in content.lower():  # the diagnosis pass
            yield "The multiply used + instead of *. Fix: use *."  # canned diagnosis
            return
        yield self.rounds.pop(0) if self.rounds else self.rounds and "" or ""


def _git(a, c):
    subprocess.run(["git", *a], cwd=c, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "e@e"], tmp_path)
    _git(["config", "user.name", "e"], tmp_path)
    (tmp_path / "README.md").write_text("# p\n")
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-q", "-m", "i"], tmp_path)
    return tmp_path


def test_build_one_shot_success(repo):
    gen = (
        "### FILE: mymath.py\n```python\ndef square(n):\n    return n * n\n```\n\n"
        "### FILE: test_mymath.py\n```python\nfrom mymath import square\n\n\n"
        "def test_sq():\n    assert square(4) == 16\n```"
    )
    out = build(repo, "a square(n) function with a test", llm=FakeLLM([gen]))
    assert out.ok and out.stage == "built", out
    assert out.branch


def test_build_repairs_on_second_round(repo):
    bad = "### FILE: mymath.py\n```python\ndef square(n):\n    return n + n  # bug\n```\n\n" \
          "### FILE: test_mymath.py\n```python\nfrom mymath import square\n\n\n" \
          "def test_sq():\n    assert square(4) == 16\n```"
    good = "### FILE: mymath.py\n```python\ndef square(n):\n    return n * n\n```"
    out = build(repo, "square with test", llm=FakeLLM([bad, good]))
    assert out.ok and out.attempts == 2, out


def test_build_runnable_script_via_smoke(repo):
    # No real test — validated by running it (smoke). No pytest gate needed.
    gen = ("### FILE: app.py\n```python\ndef main():\n    print('hi')\n\n"
           "if __name__ == '__main__':\n    main()\n```")
    out = build(repo, "a hello script runnable with python app.py", llm=FakeLLM([gen]))
    assert out.ok, out
