import copy
import subprocess

from agent.coding.repair import Tier
from agent.coding.session import CodingSession
from agent.coding.workspace import create_workspace
from agent.config import DEFAULT_CONFIG


class FakeClient:
    def __init__(self, responses, judge='{"verdict": "solves", "reason": "ok"}'):
        self.responses = list(responses)
        self.judge = judge
        self.models = []

    def chat(self, messages, model, stream=False, num_ctx=None, temperature=None, options=None):
        self.models.append(model)
        if len(messages) == 1 and "strict code reviewer" in messages[0]["content"]:
            return iter([self.judge])
        return iter([self.responses.pop(0)])


def _init_repo(path, buggy="def add(a, b):\n    return a - b\n"):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "mod.py").write_text(buggy)
    (path / "test_mod.py").write_text(
        "from mod import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=path, check=True)


def _config(**coding_overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for section, values in coding_overrides.items():
        cfg["coding"][section].update(values)
    return cfg


FIX = "<<<< FILE: mod.py\ndef add(a, b):\n    return a + b\n>>>>"
STILL_WRONG = "<<<< FILE: mod.py\ndef add(a, b):\n    return a - b\n>>>>"


def test_first_try_success(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        client = FakeClient([FIX])
        result = CodingSession(client, _config(), ws).run("make add() correct")
        assert result.success
        assert result.tier is Tier.ACCUMULATING
        assert "mod.py" in result.changed_files
        assert result.attempts == 1
    finally:
        ws.close()


def test_repair_then_success(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        client = FakeClient([STILL_WRONG, FIX])
        result = CodingSession(client, _config(), ws).run("fix add()")
        assert result.success
        assert result.attempts == 2
    finally:
        ws.close()


def test_malformed_then_success(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        client = FakeClient(["I will now fix the bug for you.", FIX])
        result = CodingSession(client, _config(), ws).run("fix add()")
        assert result.success
        assert result.malformed == 1
    finally:
        ws.close()


def test_too_many_malformed_hands_back(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        client = FakeClient(["no edits"] * 5)
        result = CodingSession(client, _config(), ws).run("fix add()")
        assert not result.success
        assert "malformed" in result.reason
    finally:
        ws.close()


def test_stall_escalates_to_handback_when_tier3_disabled(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        client = FakeClient([STILL_WRONG] * 6)
        cfg = _config(escalation={"tier3_enabled": False})
        result = CodingSession(client, cfg, ws).run("fix add()")
        assert not result.success
        assert result.tier in (Tier.ESCALATED, Tier.HANDBACK)
    finally:
        ws.close()


def test_stall_escalates_to_tier3_model(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        client = FakeClient([STILL_WRONG, STILL_WRONG, STILL_WRONG, FIX])
        cfg = _config(escalation={"tier3_enabled": True, "tier3_model": "big-model"})
        result = CodingSession(client, cfg, ws).run("fix add()")
        assert result.success
        assert "big-model" in client.models
    finally:
        ws.close()


def test_test_file_edit_rejected_then_fixed(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        edit_test = "<<<< FILE: test_mod.py\ndef test_add():\n    assert True\n>>>>"
        client = FakeClient([edit_test, FIX])
        result = CodingSession(client, _config(), ws).run("fix add()")
        assert result.success
        assert "test_mod.py" not in result.changed_files
    finally:
        ws.close()


def test_llm_judge_blocks_gaming(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        client = FakeClient(
            [FIX] * 5,
            judge='{"verdict": "games", "reason": "hardcoded"}',
        )
        cfg = _config(escalation={"tier3_enabled": False})
        result = CodingSession(client, cfg, ws).run("fix add()")
        assert not result.success
    finally:
        ws.close()


def test_events_emitted(tmp_path):
    _init_repo(tmp_path)
    ws = create_workspace(tmp_path)
    try:
        from agent.events import EventBus

        bus = EventBus()
        seen = []
        bus.subscribe(lambda e: seen.append(e.kind))
        client = FakeClient([FIX])
        CodingSession(client, _config(), ws, bus=bus).run("fix add()")
        assert "attempt" in seen
        assert "success" in seen
    finally:
        ws.close()
