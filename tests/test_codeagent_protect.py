"""coding.protect: the player's own data (save.json ...) is independent of file modes - tool writes are refused, a shell command
that changes it is undone byte-for-byte (content, mode, mtime). Reads stay allowed; .scratch/ copies stay writable."""

from __future__ import annotations

import io
import os
import stat

import pytest
from rich.console import Console

from agent.codeagent import policy as P
from agent.codeagent import lookact, tools as T
from agent.codeagent.engine import CodeEngine
from agent.codeagent.eventlog import EventLog

from tests.test_codeagent_engine import CFG, FakeLLM, engine


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


SAVE = '{"gold": 5}\n'


def _eng(tmp_path):
    eng, repo, _out = engine(tmp_path, FakeLLM([]))
    (repo / "save.json").write_text(SAVE)
    return eng, repo


def _is_refusal(res: str) -> bool:
    return "save.json is protected (the player's own data), so it was NOT changed. Do not retry this" in res and ".scratch/" in res


def test_every_write_tool_refuses_a_protected_file(tmp_path):
    eng, repo = _eng(tmp_path)
    t = eng.tools
    results = [
        t.write_file("save.json", "{}\n"),
        t.append_file("save.json", "x" * 400),
        t.edit_file("save.json", search="5", replace="6"),
        t.replace_lines("save.json", 1, 1, "{}"),
        t.insert_after("save.json", "{}", line=1),
        t.apply_patch("--- a/save.json\n+++ b/save.json\n@@ -1 +1 @@\n-{\"gold\": 5}\n+{\"gold\": 6}\n"),
        t.move_file("save.json", "other.json"),
        t.delete_file("save.json"),
    ]
    for r in results:
        assert _is_refusal(r) or "save.json is protected" in r, r
    assert (repo / "save.json").read_text() == SAVE and not (repo / "other.json").exists()
    assert "save.json" not in eng.ctx.ledger.files and "save.json" not in t.dirty


def test_creating_or_moving_onto_a_protected_name_is_refused(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "a.txt").write_text("a\n")
    assert "protected" in eng.tools.write_file("slot1.sav", "x\n")
    assert "protected" in eng.tools.move_file("a.txt", "sub/save.json")
    assert not (repo / "slot1.sav").exists() and (repo / "a.txt").exists()


def test_a_folder_holding_a_save_cannot_be_moved(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "data").mkdir()
    (repo / "data" / "x.save").write_text("s\n")
    assert "protected" in eng.tools.move_file("data", "data2")
    assert (repo / "data" / "x.save").exists()


def test_a_refusal_is_neutral_and_repeats_identically(tmp_path):
    res = "error: save.json is protected (the player's own data), so it was NOT changed. Do not retry this; ..."
    assert lookact.classify("write_file", {"path": "save.json"}, res) == "neutral"
    assert lookact.classify("write_file", {"path": "a.py"}, "wrote a.py") == "act"
    eng, repo = _eng(tmp_path)
    a = eng.execute("write_file", {"path": "save.json", "content": "{}"})
    b = eng.execute("write_file", {"path": "save.json", "content": "{}"})
    assert a == b and _is_refusal(a)


def test_reads_stay_allowed(tmp_path):
    eng, repo = _eng(tmp_path)
    assert "gold" in eng.tools.read_file("save.json")


@pytest.mark.parametrize("cmd", [
    "python3 -c \"open('save.json','w').write('{\\\"gold\\\": 99}')\"",
    "echo hacked > save.json",
    "printf 'import json\\njson.dump({\"flag\": 1}, open(\"save.json\", \"w\"))\\n' > game.py && python3 game.py",
    "rm save.json",
])
def test_a_shell_command_that_rewrites_the_save_is_undone(tmp_path, cmd):
    eng, repo = _eng(tmp_path)
    f = repo / "save.json"
    os.utime(f, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    mode, mtime = stat.S_IMODE(f.stat().st_mode), f.stat().st_mtime_ns
    res = eng.tools.run(cmd)
    assert f.read_bytes() == SAVE.encode()
    assert stat.S_IMODE(f.stat().st_mode) == mode and f.stat().st_mtime_ns == mtime
    assert ("NOTE: save.json is protected; this command changed it and the change was undone. "
            "Run the game or script on a copy in .scratch/ instead (copy the folder, run there).") in res
    assert "save.json" not in eng.tools.dirty and "save.json" not in eng.ctx.ledger.files


def test_the_file_stays_writable_for_the_human(tmp_path):
    eng, repo = _eng(tmp_path)
    eng.tools.run("echo x > save.json")
    assert os.access(repo / "save.json", os.W_OK)
    (repo / "save.json").write_text("human\n")                 # nothing locks him out


def test_a_command_that_only_reads_it_is_untouched(tmp_path):
    eng, repo = _eng(tmp_path)
    f = repo / "save.json"
    before = f.stat().st_mtime_ns
    res = eng.tools.run("cat save.json > /dev/null; python3 -c \"open('save.json').read()\"")
    assert "NOTE" not in res and f.stat().st_mtime_ns == before


def test_a_protected_file_the_command_created_is_deleted(tmp_path):
    eng, repo = _eng(tmp_path)
    res = eng.tools.run("echo s > slot2.sav; echo ok > fine.txt")
    assert not (repo / "slot2.sav").exists() and (repo / "fine.txt").exists()
    assert "NOTE: slot2.sav is protected; this command created it and it was deleted." in res
    assert "slot2.sav" not in eng.ctx.ledger.files and "slot2.sav" not in eng.tools.dirty


def test_a_nested_save_is_protected_by_its_basename(tmp_path):
    eng, repo = _eng(tmp_path)
    (repo / "data").mkdir()
    (repo / "data" / "save.json").write_text("N\n")
    res = eng.tools.run("echo z > data/save.json")
    assert (repo / "data" / "save.json").read_text() == "N\n" and "data/save.json is protected" in res


def test_custom_patterns_extend_and_defaults_cannot_be_removed(tmp_path):
    cfg = {**CFG, "coding": {**CFG["coding"], "protect": ["profile.dat", "slots/*"]}}
    repo = tmp_path / "repo"
    repo.mkdir()
    P.save_repo(repo, mode="auto")
    eng = CodeEngine(repo, cfg, FakeLLM([]), Console(file=io.StringIO(), force_terminal=False), interactive=False,
                     events=EventLog("t", tmp_path / "ev.jsonl"))
    (repo / "profile.dat").write_text("P\n")
    (repo / "save.json").write_text(SAVE)
    assert "profile.dat is protected" in eng.tools.write_file("profile.dat", "x\n")
    assert "protected" in eng.tools.write_file("slots/one.txt", "x\n")
    assert "save.json is protected" in eng.tools.write_file("save.json", "x\n")           # default still there
    assert "NOTE: profile.dat is protected" in eng.tools.run("echo q > profile.dat")
    assert T.protect_patterns([]) == T.DEFAULT_PROTECT and T.protect_patterns("junk") == T.DEFAULT_PROTECT
    assert T.protect_patterns(["x", "save.json"]) == (*T.DEFAULT_PROTECT, "x")


def test_a_read_only_and_protected_file_still_works_in_a_run(tmp_path):
    eng, repo = _eng(tmp_path)
    f = repo / "save.json"
    f.chmod(0o444)
    res = eng.tools.run("python3 -c \"import os;open('x.tmp','w').write('z');os.replace('x.tmp','save.json')\"; echo fine")
    assert f.read_text() == SAVE and stat.S_IMODE(f.stat().st_mode) == 0o444
    assert "save.json is protected; this command changed it" in res and "fine" in res
    assert "protected" in eng.tools.write_file("save.json", "q\n")


def test_scratch_copies_stay_writable(tmp_path):
    eng, repo = _eng(tmp_path)
    assert not T.is_protected_rel(".scratch/x/save.json", T.DEFAULT_PROTECT)
    assert T.is_protected_rel("save.json", T.DEFAULT_PROTECT) and T.is_protected_rel("a/b/slot.sav", T.DEFAULT_PROTECT)
    assert not T.is_protected_rel("main.py", T.DEFAULT_PROTECT)
    eng.tools.write_file(".scratch/x/save.json", '{"gold": 1}\n')
    assert (repo / ".scratch/x/save.json").read_text() == '{"gold": 1}\n'
    res = eng.tools.run("mkdir -p .scratch/y && echo '{}' > .scratch/y/save.json")
    assert (repo / ".scratch/y/save.json").read_text() == "{}\n" and "protected" not in res
