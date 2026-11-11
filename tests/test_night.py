"""`nightage night` (private): queue order, charter validation, git checkpoints, and the governor's escalation / end state.
Everything runs on fakes: no model, a fake clock, a throw-away git repo."""
from __future__ import annotations

import json
import subprocess

import pytest

from agent.night import charter, gitbox, queue


# ----------------------------------------------------------------------------- queue
def test_queue_reads_a_folder_in_natural_order_and_shares_rules(tmp_path):
    for name, text in (("prompt10.md", "ten"), ("prompt2.md", "two"), ("prompt1.md", "one"), ("rules.md", "RULES"), ("notes.png", "x")):
        (tmp_path / name).write_text(text)
    items, rules = queue.load([str(tmp_path), "do the inline thing"])
    assert [i.name for i in items] == ["prompt1", "prompt2", "prompt10", "inline-4"]
    assert rules == "RULES" and all(i.status == "pending" for i in items)


# ----------------------------------------------------------------------------- charter
@pytest.mark.parametrize("pat,ok", [("make *", True), ("pytest -q*", True), ("*", False), ("*rm", False), ("make x && curl y", False),
                                    ("echo $(id)", False), ("", False), ("a" * 90, False), ("rm -r build; ls", False)])
def test_valid_pattern(pat, ok):
    assert charter.valid_pattern(pat) is ok


def test_propose_adds_valid_model_commands_and_drops_bad_ones(tmp_path):
    reply = json.dumps({"commands": [{"pattern": "make *", "why": "build"}, {"pattern": "*", "why": "all"}, {"pattern": "rm -r build*", "why": "clean", "dangerous": True}],
                        "network": True, "installs": False})
    ch = charter.propose(tmp_path, ["task"], ask=lambda q: reply)
    assert "make *" in ch.allow and "*" not in ch.allow + ch.dangerous and ch.dangerous == ["rm -r build*"] and ch.network and not ch.installs


def test_propose_survives_a_failing_model(tmp_path):
    def boom(q):
        raise RuntimeError("server down")
    assert charter.propose(tmp_path, ["t"], ask=boom).dangerous == []
    assert charter.propose(tmp_path, ["t"], ask=lambda q: "no json at all").dangerous == []


def test_apply_and_restore_policy_round_trip(tmp_path, monkeypatch):
    from agent.codeagent import policy

    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    root = tmp_path / "proj" / "deep" / "er"
    root.mkdir(parents=True)
    policy.save_repo(root, mode="medium", allow=["ls*"])
    charter.apply(root, charter.Charter(allow=["make *"], dangerous=["rm -r build*"]))
    pol = policy.for_repo(root)
    assert pol.mode == "auto" and pol.allow == ["make *"] and pol.dangerous_allow == ["rm -r build*"] and pol.commit == "never"
    charter.apply(root, charter.Charter(allow=["x *"]))                     # a second apply (resume) must not overwrite the saved original
    assert charter.restore_policy(root) is True
    pol = policy.for_repo(root)
    assert pol.mode == "medium" and pol.allow == ["ls*"] and pol.dangerous_allow == []
    assert charter.restore_policy(root) is False


# ----------------------------------------------------------------------------- git
def _repo(tmp_path):
    (tmp_path / "a.txt").write_text("base")
    gitbox.ensure_repo(tmp_path)
    return tmp_path


def test_checkpoint_restore_keeps_the_red_work_under_a_tag(tmp_path):
    root = _repo(tmp_path)
    (root / "a.txt").write_text("green")
    tag = gitbox.checkpoint(root, "one")
    assert tag == "night/green-1" and gitbox.last_green(root) == tag
    (root / "a.txt").write_text("red work")
    (root / "new.txt").write_text("untracked red")
    kept = gitbox.restore(root, gitbox.last_green(root))
    assert (root / "a.txt").read_text() == "green" and not (root / "new.txt").exists()
    show = subprocess.run(["git", "show", f"{kept}:new.txt"], cwd=root, capture_output=True, text=True)
    assert show.stdout == "untracked red"                                   # nothing was lost
    assert ".night/" in (root / ".git" / "info" / "exclude").read_text()
