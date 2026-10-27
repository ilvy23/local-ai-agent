import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from agent import repoaccess as ra
from agent import tui


@pytest.fixture(autouse=True)
def _clean():
    ra.reset_session()
    yield
    ra.reset_session()


@pytest.fixture
def proj(tmp_path):
    p = tmp_path / "a" / "b" / "Proj"
    (p / "sub").mkdir(parents=True)
    return p.resolve()


def test_parse_work_in_and_commands():
    assert ra.parse_work_in("work in this repo: /home/alice/downloads/proj please") == "/home/alice/downloads/proj"
    assert ra.parse_work_in("arbeite in ~/code/x") == "~/code/x"
    assert ra.parse_work_in("hello there") is None
    m = ra.COMMAND_RE.match("/auto-/home/alice/downloads/proj")
    assert m.group(1) == "auto" and m.group(2) == "/home/alice/downloads/proj"
    assert ra.COMMAND_RE.match("/greenlist").group(2) is None


def test_resolve_folder_fixes_case_and_refuses_home_and_missing(proj, tmp_path):
    folder, note = ra.resolve_folder(str(proj).replace("Proj", "proj"))
    assert folder == proj and "using" in note
    assert ra.resolve_folder(str(tmp_path / "nope"))[0] is None
    assert ra.resolve_folder(str(Path.home()))[0] is None
    assert ra.resolve_folder("/etc")[0] is None
    assert ra.resolve_folder("relative/dir")[0] is None


def test_medium_allows_edits_in_folder_only(proj, tmp_path):
    ra.handle_command(f"/medium-{proj}")
    inside = {"path": str(proj / "sub" / "x.py"), "content": "1"}
    outside = {"path": str(proj.parent / "other.py"), "content": "1"}
    assert ra.allows("write_file", inside, True)
    assert ra.allows("replace_in_file", inside, True)
    assert not ra.allows("write_file", outside, True)
    assert not ra.allows("write_file", inside, False)                 # classifier said not CAUTION
    assert not ra.allows("run_command", {"command": f"cd {proj} && ls"}, True)   # shell needs auto
    assert not ra.allows("wa_send_message", inside, True)


def test_auto_allows_scoped_shell_commands(proj, monkeypatch):
    ra.handle_command(f"/auto-{proj}")
    ok = lambda c: ra.allows("run_command", {"command": c}, True)
    assert ok(f"cd {proj} && uv run --with pytest pytest -q")
    assert ok(f"cd {proj} && python3 {proj}/main.py > /dev/null")
    assert ok(f"cd {proj} && ls /usr/bin")
    assert not ok(f"cd {proj.parent} && ls")                          # runs outside the folder
    assert not ok(f"cd {proj} && cat /etc/passwd")                    # outside path
    assert not ok(f"cd {proj} && rm -rf ../x")                        # traversal
    assert not ok("ls")                                               # cwd is not inside the folder
    monkeypatch.chdir(proj)
    assert ok("ls -la")
    ra.handle_command(f"/ask-{proj}")
    assert not ok("ls -la")


def test_greenlist_persists_lists_and_removes(proj):
    kind, msg = ra.handle_command(f"/greenlist-{proj}")
    assert kind == "greenlist" and "greenlisted" in msg
    assert json.loads(ra._greenlist_file().read_text())["paths"] == [str(proj)]
    ra.reset_session()                                                # a "new session"
    assert ra.allows("write_file", {"path": str(proj / "x.py")}, True)
    assert str(proj) in ra.handle_command("/greenlist")[1]
    ra.handle_command(f"/ungreenlist-{proj}")
    assert not ra.allows("write_file", {"path": str(proj / "x.py")}, True)
    assert ra.greenlist() == []


def test_bad_folder_is_reported_not_granted(tmp_path):
    kind, msg = ra.handle_command("/auto-/etc")
    assert kind == "auto" and "can't" in msg or "top-level" in msg
    assert ra.handle_command("hello") is None


def test_question_lists_the_ready_to_paste_commands(proj):
    q = ra.question(proj)
    assert f"/auto-{proj}" in q and f"/medium-{proj}" in q and f"/greenlist-{proj}" in q


def test_repl_work_in_repo_asks_then_continues_after_the_level_is_chosen(proj, tmp_path):
    from agent.memory.store import Store
    from tests.test_tui import CONFIG, ScriptedToolClient, _registry_with_echo, _ScriptedConsole

    store = Store(tmp_path / "agent.db")
    task = f"work in this repo: {proj} and add a README"
    client = ScriptedToolClient([{"role": "assistant", "content": "On it."}])
    console = _ScriptedConsole([task, f"/auto-{proj}", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=_registry_with_echo())
    out = " ".join(console.file.getvalue().split())      # the console wraps long paths
    assert "How much may I do there without asking?" in out
    assert "full access" in out
    # the model got the original message only after the level was chosen
    assert len(client.chat_with_tools_calls) == 1
    assert any(task in str(m.get("content")) for m in client.chat_with_tools_calls[0])
    assert ra.mode_for(proj) == ra.AUTO
    store.close()


def test_shell_scope_follows_simple_cd_inside_the_folder(tmp_path):
    from agent import repoaccess

    root = tmp_path / "repo"
    (root / "sub" / "deep").mkdir(parents=True)
    ok = ("cd sub && python ../main.py", "cd sub/deep && cd .. && ls ../sub", "mkdir -p sub && cd sub; python ../main.py", "cd sub/.. && ls", "ls sub/../sub")
    for cmd in ok:
        assert repoaccess._shell_in_scope(cmd, root, cwd=root), cmd
    bad = ("cd .. && ls", "cd sub && ls ../..", "cd sub && cd ../.. && ls", "cd $HOME && ls", "cd $(pwd)/.. && ls", "cd `pwd`/.. && ls",
           "cd sub deep && ls", "cd -", "ls ..", "cd /etc && ls", "cd sub && ls /etc/passwd")
    for cmd in bad:
        assert not repoaccess._shell_in_scope(cmd, root, cwd=root), cmd
