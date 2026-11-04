"""Phase-3 gate fixes: the patch policy check sees the same file the tool writes, read-only files stay protected for root, the home
jail knows /root, /Users and /var/home, read-only commands that look outside the repo ask, and a check that did not run proves nothing."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agent.codeagent import edits, policy as P, tools as T
from tests.test_codeagent_engine import FakeLLM, engine


def test_patch_paths_uses_the_default_path_of_a_headerless_patch():
    assert edits.patch_paths("@@\n+key\n", "~/.ssh/authorized_keys") == ["~/.ssh/authorized_keys"]
    assert edits.patch_paths("--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n", "ignored.py") == ["x.py"]
    assert edits.patch_paths("garbage", None) == ["."]


@pytest.mark.parametrize("mode", ["medium", "auto"])
def test_a_headerless_patch_cannot_write_outside_the_repo(tmp_path, mode):
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode=mode)
    target = Path.home() / ".ssh" / "authorized_keys"
    before = target.read_bytes() if target.exists() else None
    msg = eng.execute("apply_patch", {"patch": "@@\n+ssh-ed25519 AAAA attacker\n", "path": "~/.ssh/authorized_keys"})
    assert "NO:" in msg or "blocked" in msg.lower() or "private" in msg.lower(), msg
    assert (target.read_bytes() if target.exists() else None) == before


def test_read_only_detection_uses_the_mode_bits_even_when_access_says_writable(tmp_path, monkeypatch):
    f = tmp_path / "save.json"
    f.write_text("{}")
    f.chmod(0o444)
    try:
        monkeypatch.setattr(T.os, "access", lambda *a, **k: True)          # what os.access answers for root
        assert T.is_read_only(f)
        f.chmod(0o644)
        assert not T.is_read_only(f)
    finally:
        f.chmod(0o644)


def test_read_only_file_is_refused_by_the_write_tools_for_root_too(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    save = repo / "notes.txt"                       # (not a name the default `coding.protect` patterns cover)
    save.write_text('{"room": 1}\n')
    save.chmod(0o444)
    try:
        monkeypatch.setattr(T.os, "access", lambda *a, **k: True)
        msg = eng.execute("write_file", {"path": "notes.txt", "content": "{}\n"})
        assert "read-only" in msg and "NOT changed" in msg
        assert save.read_text() == '{"room": 1}\n'
    finally:
        save.chmod(0o644)


@pytest.mark.parametrize("home, other", [("/root", "/home/bob/x"), ("/Users/me", "/Users/bob/x"), ("/var/home/me", "/var/home/bob/x"),
                                         ("/home/me", "/root/.bashrc")])
def test_other_users_homes_are_jailed_wherever_homes_live(monkeypatch, tmp_path, home, other):
    monkeypatch.setattr(P, "_home", lambda: home)
    assert P.forbidden_path(other, tmp_path) is not None


def test_a_safe_command_reading_outside_the_repo_asks_but_inside_is_free(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    for mode in ("ask", "medium", "auto"):
        pol = P.RepoPolicy(mode=mode)
        assert P.check_command(pol, repo, "cat a.py", "SAFE").verdict == "allow"
        assert P.check_command(pol, repo, "cat /srv/other/data.txt", "SAFE").verdict == "ask", mode
        assert P.check_command(pol, repo, "ls 2>/dev/null", "SAFE").verdict == "allow"


def test_a_blocked_verification_is_not_reported_as_verified(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="auto")
    monkeypatch.setattr(eng.tools, "check_command_list", lambda: ["pytest -q"])
    monkeypatch.setattr(P, "check_command", lambda *a, **k: P.Decision("block", "denied by you"))
    eng.tools.dirty.add(str(repo / "a.py"))
    assert eng._verify() is None
    assert eng.verify_not_run and "blocked" in eng.verify_not_run[0]
    assert eng.tools.dirty                                       # still unverified: nothing was proven


def test_a_missing_check_tool_is_not_run_rather_than_failed(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="auto")
    monkeypatch.setattr(eng.tools, "check_command_list", lambda: ["ruff check ."])
    monkeypatch.setattr(eng.tools, "_exec", lambda cmd, timeout: (127, "bash: ruff: command not found", 0.01))
    eng.tools.dirty.add(str(repo / "a.py"))
    assert eng._verify() is None                                  # no "VERIFICATION FAILED" sent to the model
    assert eng.verify_not_run == ["ruff check . (command not found)"] and eng.tools.dirty
