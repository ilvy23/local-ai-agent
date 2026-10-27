"""Temp folders are a hard NO for the agent: no question, no approval prompt - the model is told to use .scratch/ instead."""

from pathlib import Path

from agent.codeagent import policy as P


def pol():
    return P.RepoPolicy() if hasattr(P, "RepoPolicy") else None


ROOT = Path("/home/someone/project")


def test_commands_touching_temp_are_blocked_at_once():
    for cmd in ("cd /tmp && rm -rf dbg", "cp a.py /tmp/dbg/", "python3 x.py > /tmp/out.txt", "ls /var/tmp", "cat /dev/shm/x",
                "d=$(mktemp -d)", "echo hi > $TMPDIR/x", "echo hi > ${TMPDIR}/x"):
        d = P.check_command(pol(), ROOT, cmd, "SAFE")
        assert d.verdict == "block" and ".scratch" in d.reason, cmd


def test_normal_commands_and_lookalikes_are_not_blocked():
    for cmd in ("ls .scratch", "python3 tmp_helper.py", "cat src/tmp/x.py", "python3 -m pytest -q", "echo /tmpfoo", "cat my.tmp"):
        assert P.check_command(pol(), ROOT, cmd, "SAFE").verdict != "block", cmd


def test_paths_in_temp_are_blocked_but_a_repo_that_lives_in_temp_still_works(tmp_path):
    assert P.check_path(pol(), ROOT, "/tmp/dbg/x.py", True).verdict == "block"
    assert P.check_path(pol(), ROOT, "/etc/hosts", False).verdict == "ask"             # other outside paths keep asking
    repo = tmp_path / "repo"
    repo.mkdir()
    assert P.check_command(pol(), repo, f"cat {repo}/a.py", "SAFE").verdict == "allow"
    assert P.check_path(pol(), repo, "a.py", True).verdict != "block"
