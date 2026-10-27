"""The path jail: private / system folders are a hard NO (verdict 'block') in file tools and shell commands - no question, no prompt."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.codeagent import policy as P
from tests.test_codeagent_engine import CFG, FakeLLM, call, engine, say

HOME = "/home/tester"
ROOT = Path(HOME) / "work" / "repo"


@pytest.fixture(autouse=True)
def _jail(monkeypatch):
    monkeypatch.setattr(P, "_home", lambda: HOME)
    P.configure({})
    yield
    P.configure({})


def pol(mode=P.AUTO):
    return P.RepoPolicy(mode=mode)


FORBIDDEN_PATHS = [
    "~/.ssh/id_ed25519", "~/.ssh", f"{HOME}/.ssh/id_rsa", "$HOME/.ssh/id_rsa", "${HOME}/.ssh/config", "~/.gnupg/private-keys-v1.d/x",
    "~/.aws/credentials", "~/.kube/config", "~/.docker/config.json", "~/.password-store/mail.gpg", "~/.config/anything", "~/.config",
    "~/.config/google-chrome/Default/Cookies", "~/.local/share/keyrings/login.keyring", "~/.local/share", "~/.claude/settings.json",
    "~/.mozilla/firefox/x.default/logins.json", "~/.netrc", "~/.git-credentials", "~/.bash_history", "~/.zsh_history", "~/.bashrc",
    "/etc/shadow", "/etc/sudoers", "/etc/sudoers.d/x", "/etc/ssh/ssh_host_rsa_key", "/root/.bashrc", "/boot/vmlinuz", "/sys/kernel/x",
    "/proc/self/environ", "/proc/1/cmdline", "/dev/sda", "/home/other/notes.txt", "/home/other", "/home", HOME, "../../..", "../../.ssh/id_rsa",
    "sub/../../../.ssh/id_rsa",
]

ALLOWED_PATHS = [
    "a.py", "src/main.py", ".scratch/x.txt", "./src", ".env", ".git/config", "/usr/bin/python3", "/usr/lib/python3/site-packages/x.py",
    "/nix/store/abc-python3/bin/python3", "/bin/ls", "/lib/x.so", "/etc/os-release", "/etc/hosts", "/dev/null", "/dev/zero", "/dev/urandom",
    "/dev/stdin", "/dev/stdout", "/dev/stderr", "/proc/cpuinfo", "/proc/meminfo", f"{HOME}/.local/share/uv/python/x", f"{HOME}/Documents/other-project",
    f"{HOME}/work/sibling/x.py", f"{ROOT}/a.py",
]


@pytest.mark.parametrize("path", FORBIDDEN_PATHS)
def test_forbidden_path_function(path):
    why = P.forbidden_path(path, ROOT)
    assert why and why.startswith("NO: ") and "work only inside the repo" in why and ".scratch/" in why


@pytest.mark.parametrize("path", FORBIDDEN_PATHS)
@pytest.mark.parametrize("write", [False, True])
def test_check_path_blocks_forbidden(path, write):
    d = P.check_path(pol(), ROOT, path, write)
    assert d.verdict == "block" and d.reason.startswith("NO: "), (path, d)


@pytest.mark.parametrize("path", ALLOWED_PATHS)
def test_check_path_does_not_block_ordinary_paths(path):
    assert P.forbidden_path(path, ROOT) is None
    assert P.check_path(pol(), ROOT, path, False).verdict != "block"


def test_sibling_projects_still_ask_not_block():
    d = P.check_path(pol(), ROOT, f"{HOME}/work/sibling/x.py", False)
    assert d.verdict == "ask"
    assert P.check_path(pol(), ROOT, "../sibling/x.py", True).verdict == "ask"


FORBIDDEN_CMDS = [
    "cat ~/.ssh/id_ed25519", "cat ~/.ssh/id_rsa", "cat $HOME/.ssh/id_rsa", "cat ${HOME}/.ssh/id_rsa", 'cat "$HOME"/.ssh/id_rsa',
    "cp x ~/.config/y", "cp ~/.config/y .", "echo hi > ~/.bashrc", "echo hi >> ~/.bashrc", "echo hi >~/.zshrc", "tee ~/.bashrc < a.py",
    "cat /etc/shadow", "ls /etc/ssh", "grep -r token ~/.aws", "grep -r pw /home/other", "ls /home", "ls /root", "cat /proc/self/environ",
    "ls /boot", "cat ~/.bash_history", "cat ~/.zsh_history", "cd ~/.ssh && ls", "cd ~ && ls", "ls ~", "cd .. && ls ../..", "ls ../../..",
    "find ~/.config -name '*.json'", "rm ~/.netrc", "mv a.py ~/.local/share/x", "python3 -c \"print(open('/etc/shadow').read())\"",
    "python3 -c \"import os; os.listdir('~/.ssh')\"", "cat --file=/etc/shadow", "ls -la /dev/sda", "cat ~/.git-credentials",
    "echo $(cat ~/.ssh/id_rsa)", "cat `echo ~/.ssh/id_rsa`", "ls /home/other/.ssh", "cat ~/.claude/.credentials.json", "sort < ~/.bash_history",
]

ALLOWED_CMDS = [
    "python3 -m pytest", "python3 -m pytest -q tests", "grep -r foo .", "ls ./src", "ls", "cat a.py", "ls .scratch", "/usr/bin/python3 a.py",
    "/nix/store/abc-python3/bin/python3 a.py", "python3 a.py > /dev/null 2>&1", "cat /etc/os-release", "cat /etc/hosts", "cat /proc/cpuinfo",
    "ls /usr/lib/python3", "sed -n 's/a/b/p' a.py", "git status", "git log --oneline -5", 'git commit -m "fix parser"', "cargo test", "npm run build",
    "echo hello | wc -c", "head -n 3 src/main.py", "cd src && ls", "ls ../sibling", "cat .env", f"cat {ROOT}/a.py", "python3 x.py 2>&1 | tail -5",
    "ls $HOME/work/sibling", "ls ~/Documents", "echo a/b/c", "rm -rf .scratch/out",
]


@pytest.mark.parametrize("cmd", FORBIDDEN_CMDS)
def test_check_command_blocks_forbidden(cmd):
    d = P.check_command(pol(), ROOT, cmd, "SAFE")
    assert d.verdict == "block" and d.reason.startswith("NO: "), (cmd, d)


@pytest.mark.parametrize("cmd", FORBIDDEN_CMDS[:12])
@pytest.mark.parametrize("mode", [P.ASK, P.MEDIUM, P.AUTO])
def test_forbidden_commands_block_in_every_mode_and_risk(cmd, mode):
    for risk in ("SAFE", "CAUTION", "DANGEROUS"):
        assert P.check_command(pol(mode), ROOT, cmd, risk).verdict == "block"


@pytest.mark.parametrize("cmd", ALLOWED_CMDS)
def test_check_command_does_not_block_ordinary_commands(cmd):
    assert P.check_command(pol(), ROOT, cmd, "SAFE").verdict != "block", cmd


def test_symlink_inside_repo_to_private_folder_is_blocked(tmp_path, monkeypatch):
    home = tmp_path / "home" / "tester"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("secret")
    repo = home / "work" / "repo"
    repo.mkdir(parents=True)
    (repo / "link").symlink_to(home / ".ssh")
    (repo / "keyfile").symlink_to(home / ".ssh" / "id_rsa")
    monkeypatch.setattr(P, "_home", lambda: str(home))
    for path in ("link", "link/id_rsa", "keyfile", "./link/../.ssh/id_rsa"):
        assert P.check_path(pol(), repo, path, False).verdict == "block", path
    assert P.check_command(pol(), repo, "cat link/id_rsa", "SAFE").verdict == "block"
    assert P.check_command(pol(), repo, "cat keyfile", "SAFE").verdict == "block"
    (repo / "real.py").write_text("x = 1\n")
    assert P.check_path(pol(), repo, "real.py", False).verdict == "allow"


def test_repo_in_a_temp_dir_still_works(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    assert P.check_command(pol(), repo, f"cat {repo}/a.py", "SAFE").verdict == "allow"
    assert P.check_path(pol(), repo, "a.py", True).verdict == "allow"
    assert P.check_path(pol(), repo, f"{repo}/sub/b.py", True).verdict == "allow"


def test_repo_inside_a_denied_zone_is_its_own_business(monkeypatch):
    root = Path(HOME) / ".config" / "myproj"
    assert P.forbidden_path("a.py", root) is None
    assert P.forbidden_path(f"{root}/src/a.py", root) is None
    assert P.forbidden_path("~/.ssh/id_rsa", root)


# ── config: extend-only ─────────────────────────────────────────────────────


def test_config_deny_paths_extend_the_builtin_list():
    assert P.forbidden_path(f"{HOME}/secrets/x", ROOT) is None
    P.configure({"coding": {"deny_paths": ["~/secrets", "/data/private"]}})
    assert P.check_path(pol(), ROOT, f"{HOME}/secrets/x", False).verdict == "block"
    assert P.check_path(pol(), ROOT, "/data/private/db", False).verdict == "block"
    assert P.check_command(pol(), ROOT, "cat ~/secrets/x", "SAFE").verdict == "block"
    assert P.forbidden_path("~/.ssh/id_rsa", ROOT)                       # the built-ins are still there


def test_config_cannot_remove_or_unlock_builtin_denies():
    P.configure({"coding": {"allow_paths": ["~/.ssh", "/etc/shadow", "~/.config", HOME, "/"], "deny_paths": []}})
    for p in ("~/.ssh/id_rsa", "/etc/shadow", "~/.config/x", "/home/other/x", "~/.bash_history"):
        assert P.check_path(pol(), ROOT, p, False).verdict == "block", p
    assert P.check_command(pol(), ROOT, "cat ~/.ssh/id_rsa", "SAFE").verdict == "block"


def test_config_deny_beats_allow():
    P.configure({"coding": {"allow_paths": ["/data"], "deny_paths": ["/data/private"]}})
    assert P.check_path(pol(), ROOT, "/data/public/x", False).verdict == "allow"
    assert P.check_path(pol(), ROOT, "/data/private/x", False).verdict == "block"


def test_config_allow_paths_adds_a_read_path():
    assert P.check_path(pol(), ROOT, "/data/corpus/x.txt", False).verdict == "ask"
    assert P.check_command(pol(), ROOT, "cat /data/corpus/x.txt", "CAUTION").verdict == "ask"
    P.configure({"coding": {"allow_paths": ["/data/corpus", "~/refs"]}})
    assert P.check_path(pol(), ROOT, "/data/corpus/x.txt", False).verdict == "allow"
    assert P.check_path(pol(), ROOT, f"{HOME}/refs/a.md", False).verdict == "allow"
    assert P.check_path(pol(), ROOT, "/data/corpus/x.txt", True).verdict == "ask"          # read only: writing still asks
    assert P.check_path(pol(), ROOT, "/data/other/x.txt", False).verdict == "ask"
    assert P.check_command(pol(), ROOT, "cat /data/corpus/x.txt", "CAUTION").verdict == "allow"


def test_config_garbage_is_ignored():
    P.configure({"coding": {"deny_paths": "not-a-list", "allow_paths": [None, 3, "", "relative/dir"]}})
    assert P.forbidden_path("~/.ssh/id", ROOT)
    assert P.check_path(pol(), ROOT, "a.py", False).verdict == "allow"


# ── engine ──────────────────────────────────────────────────────────────────


def test_engine_answers_a_repeated_forbidden_call_with_the_same_no_and_never_asks(tmp_path):
    asked = []
    llm = FakeLLM([call("run", command="cat ~/.ssh/id_rsa")] * 3 + [say("I cannot read that, giving up.")])
    eng, repo, out = engine(tmp_path, llm, mode="ask", approve=lambda d, r: asked.append(d) or True)
    answers, real = [], eng.execute
    eng.execute = lambda name, args: answers.append(real(name, args)) or answers[-1]
    eng.turn("show me my ssh key")
    no = [a for a in answers if a.startswith("NO: ")]
    assert len(no) == 3 and len(set(no)) == 1, answers
    assert ".ssh/id_rsa" in no[0] and "work only inside the repo" in no[0] and len(no[0]) < 200
    assert asked == []                                                             # nobody was ever asked
    assert not eng._declined                                                       # not a 'declined by user' that invites a retry
    assert len(eng._sigs) == 3 and len({s[1] for s in eng._sigs}) == 1             # all three count toward the loop guard
    assert eng._noed                                                               # and the NO is remembered


def test_engine_repeat_does_not_rerun_policy(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="ask", approve=lambda d, r: pytest.fail("asked"))
    first = eng.execute("read_file", {"path": "~/.ssh/id_rsa"})
    assert first.startswith("NO: ")
    monkeypatch.setattr(eng, "decide", lambda *a, **k: pytest.fail("policy ran again"))
    assert eng.execute("read_file", {"path": "~/.ssh/id_rsa"}) == first


def test_engine_file_tools_block_forbidden_reads_and_writes(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="auto", approve=lambda d, r: pytest.fail("asked"))
    for name, args in (("read_file", {"path": "~/.ssh/id_rsa"}), ("read_files", {"paths": ["a.py", "~/.aws/credentials"]}),
                       ("list_dir", {"path": "~/.config"}), ("search", {"pattern": "x", "path": "/etc/ssh"}),
                       ("write_file", {"path": "~/.bashrc", "content": "x\n"}), ("delete_file", {"path": "/home/other/x"})):
        assert eng.execute(name, args).startswith("NO: "), name
