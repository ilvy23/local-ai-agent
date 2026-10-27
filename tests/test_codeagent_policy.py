from __future__ import annotations

from pathlib import Path

import pytest

from agent.codeagent import policy as P


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def test_default_is_ask_and_save_roundtrip(tmp_path):
    repo = tmp_path / "game"
    repo.mkdir()
    assert P.for_repo(repo).mode == "ask"
    pol = P.save_repo(repo, mode="auto", allow=["pytest*", "make *"], network=True, max_steps=999, commit="bogus")
    assert pol.mode == "auto" and pol.allow == ["pytest*", "make *"] and pol.network is True
    assert pol.max_steps == 500 and pol.commit == "ask"                 # clamped / sanitized
    assert P.for_repo(repo / "sub" / "dir").mode == "auto"               # subfolders inherit
    assert P.forget_repo(repo) and P.for_repo(repo).mode == "ask"


def test_most_specific_repo_wins_and_defaults_apply(tmp_path):
    outer, inner = tmp_path / "work", tmp_path / "work" / "risky"
    inner.mkdir(parents=True)
    P.policy_file().write_text('{"defaults": {"commit": "never"}, "repos": {}}')
    P.save_repo(outer, mode="auto")
    P.save_repo(inner, mode="ask")
    assert P.for_repo(outer).mode == "auto" and P.for_repo(inner).mode == "ask"
    assert P.for_repo(outer).commit == "never"


def test_paths_stay_inside_the_repo(tmp_path):
    root = tmp_path
    auto = P.RepoPolicy(mode="auto")
    assert P.check_path(auto, root, "src/a.py", write=True).verdict == "allow"
    assert P.check_path(auto, root, "/etc/passwd", write=True).verdict == "ask"
    assert P.check_path(auto, root, "../outside.py", write=True).verdict == "ask"
    assert P.check_path(auto, root, ".git/config", write=True).verdict == "block"
    assert P.check_path(P.RepoPolicy(mode="ask"), root, "a.py", write=True).verdict == "ask"
    assert P.check_path(P.RepoPolicy(mode="ask"), root, "a.py", write=False).verdict == "allow"
    assert P.check_path(P.RepoPolicy(mode="medium"), root, "a.py", write=True).verdict == "allow"


def test_commands(tmp_path):
    root = tmp_path
    auto = P.RepoPolicy(mode="auto", deny=["rm *"])
    assert P.check_command(auto, root, "pytest -q", "SAFE").verdict == "allow"
    assert P.check_command(auto, root, "python3 game.py", "CAUTION").verdict == "allow"
    assert P.check_command(auto, root, "rm notes.txt", "CAUTION").verdict == "block"
    assert P.check_command(auto, root, "anything", "BLOCKED").verdict == "block"
    assert P.check_command(auto, root, "sudo make install", "DANGEROUS").verdict == "ask"
    assert P.check_command(auto, root, "cat /srv/someone/secret", "CAUTION").verdict == "ask"        # outside the repo
    assert P.check_command(auto, root, "pip install requests", "CAUTION").verdict == "ask"            # installs off
    assert P.check_command(P.RepoPolicy(mode="auto", installs=True), root, "pip install requests", "CAUTION").verdict == "allow"
    assert P.check_command(auto, root, "curl https://example.com", "CAUTION").verdict == "ask"
    assert P.check_command(P.RepoPolicy(mode="auto", network=True), root, "git push origin main", "CAUTION").verdict == "ask"
    assert P.check_command(P.RepoPolicy(mode="auto", allow=["git push*"]), root, "git push origin main", "CAUTION").verdict == "allow"
    med = P.RepoPolicy(mode="medium", allow=["npm run *"])
    assert P.check_command(med, root, "npm run build", "CAUTION").verdict == "allow"
    assert P.check_command(med, root, "make all", "CAUTION").verdict == "ask"
    assert P.check_command(P.RepoPolicy(), root, "ls", "SAFE").verdict == "allow"


def test_chat_greenlist_counts_as_auto(tmp_path, monkeypatch):
    from agent import repoaccess

    repo = tmp_path / "proj"
    repo.mkdir()
    monkeypatch.setattr(repoaccess, "greenlist", lambda: [repo])
    assert P.for_repo(repo).mode == "auto"
    assert "mode   auto" in P.describe(repo)


def test_housekeeping_in_a_trusted_repo_is_allowed_but_real_danger_still_asks(tmp_path):
    root = tmp_path
    (root / "game").mkdir()
    auto = P.RepoPolicy(mode="auto")
    ok = [
        "rm -f .scratch/save.json", "rm -rf .scratch", "rm -rf .tmp_test", "rm -rf build/ dist/",
        "mkdir -p /tmp/harbour_test && cp engine.py main.py /tmp/harbour_test/", "cp -r game .scratch/game", "rm -rf /tmp/harbour_test",
        "mkdir -p .scratch && cp engine.py .scratch/ && cd .scratch", "touch .scratch/x", "cp engine.py engine_copy.py",
        "mv .scratch/out.txt results.txt", 'rm -r "build"',
    ]
    for cmd in ok:
        assert P.check_command(auto, root, cmd, "DANGEROUS").verdict == "allow", cmd
    bad = [
        "rm -rf /", "rm -rf game", "rm -rf *", "rm *.py", "rm -rf ..", "rm ../x", "rm -rf /tmp", "rm -rf .git", "rm .git/config",
        "cp /etc/passwd .scratch/", "cp engine.py /srv/someone/", "sudo rm x", "rm x | cat", "rm $(echo x)", "echo hi > /tmp/x",
        "rm -rf /tmp/../etc", "mkdir /etc/x", "chmod -R 777 .",
        "rm notes.txt", "mv old.py new.py",                                   # real repo files: delete_file / move_file (recoverable)
    ]
    for cmd in bad:
        assert P.check_command(auto, root, cmd, "DANGEROUS").verdict == "ask", cmd
    for cmd in ("rm -rf ~", "cp engine.py /home/someone/"):                # the home folder / other homes are the jail's NO, not a question
        assert P.check_command(auto, root, cmd, "DANGEROUS").verdict == "block", cmd
    med = P.RepoPolicy(mode="medium")
    assert P.check_command(med, root, "rm -f .scratch/x", "DANGEROUS").verdict == "allow"             # scratch housekeeping: medium too
    assert P.check_command(med, root, "rm -rf .scratch/a .scratch/b", "DANGEROUS").verdict == "allow"
    for cmd in ("rm -rf build", "rm notes.txt", "rm -rf .scratch/x | cat", "cp -r game .scratch/game", "rm -rf /tmp/harbour_test"):
        assert P.check_command(med, root, cmd, "DANGEROUS").verdict == "ask", cmd                           # only .scratch/ itself, only rm / mkdir
    assert P.check_command(P.RepoPolicy(mode="auto", deny=["rm *"]), root, "rm -f .scratch/x", "DANGEROUS").verdict == "block"


def test_scratch_ok_closes_the_bypasses(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / ".scratch").mkdir(parents=True)
    (root / "save.json").write_text("{}")
    (root / "main.py").write_text("x = 1\n")
    outside = tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    (root / ".scratch" / "link").symlink_to(outside)
    monkeypatch.setattr(P, "_TMP", tmp_path / "no-tmp-here")                # pytest's tmp_path is itself under /tmp
    bad = [
        "cp main.py $HOME/.bashrc", "cp main.py ${HOME}/x", "mkdir .scratch\nrm -rf src", "touch .scratch/a & rm -rf src",
        "cd .scratch/nothere ; rm main.py", "cp main.py save.json", "cp -f main.py save.json", "mv main.py /tmp/x/",
        "rm -rf .scratch/link/sub", "cp -t /etc main.py", "rm --no-preserve-root -rf .scratch", "rm -- .scratch/x",
        "cd /tmp && rm -rf foo", "rm -rf ~/x", "cd - && rm -rf build", "rm -rf .scratch\\x", "cp main.py ~user/x",
        "mkdir -p .scratch/n ; cd .scratch/n ; rm main.py",                    # if mkdir failed, `;` would run rm in the repo root
    ]
    for cmd in bad:
        assert not P.scratch_ok(cmd, root), cmd
    assert P.scratch_ok("cd .scratch && rm main.py", root)                    # relative to the cd: .scratch/main.py
    assert P.scratch_ok("mkdir -p .scratch/n && cd .scratch/n && touch a", root)
    mine = tmp_path / "tmpdir"
    mine.mkdir()
    monkeypatch.setattr(P, "_TMP", tmp_path)
    assert P.scratch_ok(f"rm -rf {mine}", root)
    real_uid = __import__("os").getuid()
    monkeypatch.setattr(P.os, "getuid", lambda: real_uid + 1)                # someone else's folder in /tmp (like /tmp/.X11-unix)
    assert not P.scratch_ok(f"rm -rf {mine}", root)
    assert P.scratch_ok(f"rm -rf {tmp_path}/does-not-exist-yet", root)


# ── simple shell variables in scratch housekeeping ──────────────────────────

TODO_CMD = "mkdir -p .scratch && export TODO_FILE=.scratch/todos.json && rm -f $TODO_FILE && python3 todo.py add x"


def _vroot(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / ".scratch").mkdir(parents=True)
    (root / "todo.py").write_text("x = 1\n")
    (root / "save.json").write_text("{}")
    monkeypatch.setattr(P, "_TMP", tmp_path / "no-tmp-here")
    return root


def test_resolve_vars_substitutes_simple_assignments_only():
    assert P.resolve_vars("ls -la") == "ls -la"
    assert P.resolve_vars("export A=.scratch/x && rm -f $A") == "rm -f .scratch/x"
    assert P.resolve_vars("A=.scratch; B=$A/y; rm ${A}/z $B") == "rm .scratch/z .scratch/y"
    assert P.resolve_vars("A=.scratch/t.json python3 todo.py add x") == "python3 todo.py add x"       # prefix form: only for that command
    assert P.resolve_vars("A=.s && B=1 ; echo hi") == "echo hi"
    assert P.resolve_vars("echo a ; X=1 && echo b") == "echo a ; echo b"                              # an assignment never makes the next step 'chained'
    for bad in ("rm -f $UNKNOWN", "rm $(echo x)", "A=$(pwd); rm $A", "A=`pwd`; rm $A", "rm $HOME/x", "A='x y'; rm $A", "A='*'; rm $A",
                "IFS=/; rm a", "PATH=.scratch ls", "A=.s; rm ${A:-x}", "export A; rm $A", "A=.s; A2=$B; rm $A2", "X=1 &&"):
        assert P.resolve_vars(bad) is None, bad


def test_scratch_ok_resolves_variables_and_still_refuses_the_rest(tmp_path, monkeypatch):
    root = _vroot(tmp_path, monkeypatch)
    assert P.scratch_ok("export F=.scratch/todos.json && rm -f $F", root)
    assert P.scratch_ok("F=.scratch/todos.json; rm -f ${F}", root)
    assert P.scratch_ok("D=.scratch && mkdir -p $D/sub && rm -rf $D/sub", root)
    for cmd in ("rm -f $F", "F=$(echo main.py); rm $F", "F=todo.py; rm $F", "F=save.json; rm -f $F", "F=.scratch/../todo.py; rm $F",
                "F=~/.ssh; rm -rf $F", "F=/etc; rm -rf $F", "F='.scratch/a b'; rm $F", "F=.scratch/*; rm $F", "F=; rm -f $F"):
        assert not P.scratch_ok(cmd, root), cmd


def test_mixed_command_with_a_scratch_rm_is_allowed_in_auto_but_not_in_medium(tmp_path, monkeypatch):
    root = _vroot(tmp_path, monkeypatch)
    auto, med = P.RepoPolicy(mode="auto"), P.RepoPolicy(mode="medium")
    assert P.check_command(auto, root, TODO_CMD, "DANGEROUS").verdict == "allow"
    assert P.check_command(auto, root, "TODO_FILE=.scratch/t.json python3 todo.py add x", "CAUTION").verdict == "allow"
    assert P.check_command(med, root, TODO_CMD, "DANGEROUS").verdict == "ask"                          # the python part still needs approval there
    for cmd in (
        "mkdir -p .scratch && F=todo.py && rm -f $F && python3 todo.py",                              # a real repo file
        "F=.scratch/x && rm -f $F && python3 todo.py | tee out.txt && rm -rf src",                    # second rm outside
        "python3 todo.py && rm -rf .scratch/x",                                                       # rm after an unknown command: the files may have changed
        "rm -f $NOPE && python3 todo.py", "F=.scratch/x; rm -f $F; rm -rf $(echo src)",
        "F=.scratch/x; rm -f $F; bash -c 'rm -rf src'",
    ):
        assert P.check_command(auto, root, cmd, "DANGEROUS").verdict == "ask", cmd


def test_variables_cannot_point_the_rm_into_the_jail(tmp_path, monkeypatch):
    root = _vroot(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "TEMP_DIRS", ())                                                           # a real repo is not under /tmp: temp rule applies
    for mode in ("auto", "medium", "ask"):
        pol = P.RepoPolicy(mode=mode)
        for cmd in ("X=~/.ssh; rm -rf $X", "export X=~; rm -rf $X/.ssh", "export D=/tmp/x; rm -rf $D", "H=~; rm -rf ${H}/.config/x",
                    "A=~ && B=$A/.ssh && rm -rf $B", "X=/etc; rm -rf $X/shadow"):
            assert P.check_command(pol, root, cmd, "DANGEROUS").verdict == "block", (mode, cmd)
    assert P.forbidden_in_command("X=~; cat $X/.ssh/id_rsa", root)
    assert P.forbidden_in_command("X=.scratch; cat $X/a", root) is None


# ── agent-owned data files and `cd` tracking ──────────────────────────────────────────────────────

OWNED_CMD = 'rm -f todos.json && python main.py add "x"'
CD_CMD = "mkdir -p .scratch && cd .scratch && rm -f todos.json && python ../main.py add x"


def test_an_rm_of_a_file_the_run_created_is_housekeeping_in_auto_and_medium(tmp_path, monkeypatch):
    root = _vroot(tmp_path, monkeypatch)
    (root / "todos.json").write_text("[]")
    (root / "main.py").write_text("x = 1\n")
    auto, med = P.RepoPolicy(mode="auto"), P.RepoPolicy(mode="medium")
    owned = {"todos.json": "run"}
    assert P.check_command(auto, root, OWNED_CMD, "DANGEROUS", owned=owned).verdict == "allow"
    assert P.check_command(auto, root, "rm -f todos.json", "DANGEROUS", owned=owned).verdict == "allow"
    assert P.check_command(auto, root, "unlink todos.json", "DANGEROUS", owned=owned).verdict == "allow"
    assert P.check_command(med, root, "rm -f todos.json", "DANGEROUS", owned=owned).verdict == "allow"
    # the user's own pre-existing file is not ours
    assert P.check_command(auto, root, OWNED_CMD, "DANGEROUS", owned={}).verdict == "ask"
    assert P.check_command(auto, root, OWNED_CMD, "DANGEROUS").verdict == "ask"
    assert P.check_command(med, root, "rm -f todos.json", "DANGEROUS", owned={}).verdict == "ask"
    # source the model wrote: auto yes, medium still asks
    own_src = {"main.py": "write"}
    assert P.check_command(auto, root, "rm -f main.py", "DANGEROUS", owned=own_src).verdict == "allow"
    assert P.check_command(med, root, "rm -f main.py", "DANGEROUS", owned=own_src).verdict == "ask"


def test_owned_does_not_widen_recursive_globs_dirs_or_outside_paths(tmp_path, monkeypatch):
    root = _vroot(tmp_path, monkeypatch)
    (root / "src").mkdir()
    (root / "todos.json").write_text("[]")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("")
    auto = P.RepoPolicy(mode="auto")
    owned = {"todos.json": "run", "src": "run", ".git/config": "run", "../outside.txt": "run", "cd": "run"}
    for cmd in ("rm -rf src", "rm -r src", "rm src", "rm -f src", "rm -f *.json", "rm -f todos.json src", "rm -f .git/config",
                "rm -rf todos.json", "cd .. && rm -f x", "rm -f ../outside.txt", "rm -f todos.json | cat", "rm -f $(echo todos.json)"):
        assert P.check_command(auto, root, cmd, "DANGEROUS", owned=owned).verdict == "ask", cmd
    (root / "link.json").symlink_to(tmp_path / "elsewhere.txt")
    assert P.check_command(auto, root, "rm -f link.json", "DANGEROUS", owned={"link.json": "run"}).verdict == "ask"


def test_the_jail_still_wins_over_owned(tmp_path, monkeypatch):
    root = _vroot(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "TEMP_DIRS", ())
    for mode in ("auto", "medium"):
        pol = P.RepoPolicy(mode=mode)
        assert P.check_command(pol, root, "rm -f ~/.ssh/x", "DANGEROUS", owned={"~/.ssh/x": "run"}).verdict == "block"


def test_cd_into_a_repo_subfolder_and_back_up_stays_inside(tmp_path, monkeypatch):
    root = _vroot(tmp_path, monkeypatch)
    auto = P.RepoPolicy(mode="auto")
    assert P.check_command(auto, root, CD_CMD, "DANGEROUS").verdict == "allow"
    assert P.check_command(auto, root, CD_CMD, "CAUTION").verdict == "allow"
    assert P.check_command(auto, root, "cd .scratch && python ../todo.py add x", "CAUTION").verdict == "allow"
    for cmd in ("cd .. && python3 todo.py", "cd .scratch && python ../../x.py", "cd .scratch && cd ../.. && python3 todo.py",
                "cd $SOMEWHERE && python3 todo.py", "cd $(pwd)/.. && python3 todo.py", "cd .scratch; cd .. ; python ../x.py", "python ../x.py"):
        assert P.check_command(auto, root, cmd, "CAUTION").verdict == "ask", cmd


def test_the_suite_never_touches_the_real_policy_file(tmp_path):
    """conftest redirects the policy file for every test (the real one once collected 12,000 pytest temp dirs)."""
    import os
    from pathlib import Path

    from agent.codeagent import policy as P

    real = Path.home() / ".config" / "agent" / "code-policy.json"
    assert os.environ.get("AGENT_CODE_POLICY") and Path(os.environ["AGENT_CODE_POLICY"]) != real
    assert P.policy_file() != real
