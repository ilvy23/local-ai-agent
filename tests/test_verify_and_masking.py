import json
from pathlib import Path

from agent import tui, verify
from agent.memory import worklog
from agent.memory.store import Store
from agent.tools.registry import Tool, ToolRegistry


# ------------------------------------------------------------------------------ masking / tool log

def _call(name, **args):
    return {"role": "assistant", "content": json.dumps({"tool": name, "arguments": args})}


def _res(text):
    return {"role": "tool", "content": text}


def test_mask_old_observations_stubs_old_results_and_written_code_but_not_recent_ones():
    big = "x" * 5000
    msgs = [{"role": "system", "content": "persona"}, {"role": "user", "content": "task"},
            _call("read_file", path="/p/a.py"), _res("$ cat a\n" + big),
            _call("write_file", path="/p/b.py", content="y" * 4000), _res("wrote b.py"),
            _call("run_command", command="pytest"), _res("exit code 0\n" + big)]
    out, saved = worklog.mask_old_observations(msgs, keep_last=2)
    assert saved > 8000
    assert out[3]["content"].startswith("[old tool output omitted") and "5008 chars" in out[3]["content"]
    assert json.loads(out[4]["content"])["arguments"]["content"] == "[4000 chars - now on disk]"
    assert json.loads(out[4]["content"])["arguments"]["path"] == "/p/b.py"
    assert out[7]["content"] == msgs[7]["content"]                     # inside the kept tail
    assert out[0] == msgs[0] and out[1] == msgs[1]
    again, saved2 = worklog.mask_old_observations(out, keep_last=2)
    assert saved2 == 0 and again == out                                  # idempotent


def test_mask_thresholds(monkeypatch):
    monkeypatch.delenv("AGENT_MASK_AT", raising=False)
    assert not worklog.should_mask(30000, 65000) and worklog.should_mask(33000, 65000)
    assert worklog.mask_threshold() < worklog.threshold()


def test_tool_log_is_exact_and_survives_recompaction():
    chunk = [_call("write_file", path="/p/a.py", content="1"), _res("ok"),
             _call("run_command", command="pytest -q"), _res("$ pytest -q\n(in /p, exit code 1)\nFAILED"),
             _call("replace_in_file", path="/p/a.py", search="a", replace="b"), _res("ok"),
             _call("run_command", command="pytest -q"), _res("$ pytest -q\n(in /p, exit code 0)\n1 passed")]
    assert worklog.tool_log(chunk) == ["- wrote /p/a.py", "- ran `pytest -q` -> exit 1", "- edited /p/a.py",
                                       "- ran `pytest -q` -> exit 0"]
    body, lines = worklog.split_tool_log(f"DONE: x\n\n{worklog.TOOL_LOG_MARKER}\n- wrote /p/a.py\n- edited /p/b.py")
    assert body == "DONE: x" and lines == ["- wrote /p/a.py", "- edited /p/b.py"]


class FakeLLM:
    def chat(self, messages, model, stream=True, **kw):
        return iter(["DONE: things"])


def test_compaction_appends_the_harness_tool_log():
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "task"}]
    for i in range(8):
        msgs += [_call("write_file", path=f"/p/f{i}.py", content="c"), _res("ok")]
    out = worklog.compact_turn_messages(FakeLLM(), {"models": {"background": "b"}}, msgs)
    note = out[2]["content"]
    assert "DONE: things" in note and worklog.TOOL_LOG_MARKER in note and "- wrote /p/f0.py" in note
    again = worklog.compact_turn_messages(FakeLLM(), {"models": {"background": "b"}}, out + msgs[2:])
    assert again[2]["content"].count(worklog.TOOL_LOG_MARKER) == 1 and "- wrote /p/f0.py" in again[2]["content"]


# ----------------------------------------------------------------------------------- verify module

def test_project_root_and_check_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    proj = tmp_path / "w" / "proj"
    (proj / "pkg").mkdir(parents=True)
    f = str(proj / "pkg" / "a.py")
    assert verify.project_root(f) is None                                # no marker yet
    (proj / "tests").mkdir()
    assert verify.project_root(f) == proj
    assert verify.check_command(proj, [f]).endswith("python3 -m pytest -q -x") and f"cd {proj}" in verify.check_command(proj, [f])
    (proj / "tests").rmdir()
    (proj / ".git").mkdir()
    cmd = verify.check_command(proj, [f])
    assert "py_compile pkg/a.py" in cmd                                  # no tests: syntax check only
    (proj / "Cargo.toml").write_text("[package]")
    assert verify.check_command(proj, [f]).endswith("cargo test -q")
    (proj / "Cargo.toml").unlink()
    (proj / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
    assert verify.check_command(proj, [str(proj / "x.js")]).endswith("npm test --silent")
    (proj / "package.json").write_text(json.dumps({"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}))
    assert verify.check_command(proj, [str(proj / "x.js")]) is None
    assert verify.project_root(str(Path.home() / "x.py")) is None


def test_gate_bookkeeping():
    g = verify.VerifyGate(retries=2)
    g.note_edit("/p/readme.md")
    assert not g.due()                                                   # docs don't need a check
    g.note_edit("/p/a.py")
    assert g.due() and g.failed() and g.failed() and not g.failed()
    g.give_up()
    assert not g.due()
    assert verify.exit_code("$ x\n(in /p, exit code 3)\nboom") == 3 and verify.exit_code("nothing") is None


def test_enabled_switches(monkeypatch):
    monkeypatch.delenv("AGENT_VERIFY", raising=False)
    assert verify.enabled({}) and not verify.enabled({"autonomy": {"verify": False}})
    monkeypatch.setenv("AGENT_VERIFY", "0")
    assert not verify.enabled({})


# ---------------------------------------------------------------------------------- gate in the loop

def _env(tmp_path, run_results):
    from tests.test_tui import ScriptedToolClient  # noqa: F401

    proj = tmp_path / "w" / "proj"
    (proj / "tests").mkdir(parents=True)
    runs = []
    registry = ToolRegistry()
    def fake_write(path, content="", **k):
        Path(path).write_text(content)
        return f"Wrote {path}"

    registry.register(Tool(name="write_file", description="w", parameters={"type": "object", "properties": {}},
                           handler=fake_write, risk="safe"))

    def fake_run(command, **k):
        runs.append(command)
        return run_results.pop(0)

    registry.register(Tool(name="run_command", description="r", parameters={"type": "object", "properties": {}},
                           handler=fake_run, risk="safe"))
    return proj, registry, runs


def _write(proj, n):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_file",
            "arguments": {"path": str(proj / "app.py"), "content": f"x = {n}\n"}}}]}


def test_failed_check_sends_the_model_back_to_work_until_it_passes(tmp_path):
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    fail = "$ pytest\n(in /p, exit code 1)\nFAILED tests/test_a.py::test_x - assert 1 == 2"
    ok = "$ pytest\n(in /p, exit code 0)\n1 passed"
    proj, registry, runs = _env(tmp_path, [fail, ok])
    client = ScriptedToolClient([_write(proj, 1), {"role": "assistant", "content": "Done, all green!"},
                                 _write(proj, 2), {"role": "assistant", "content": "Fixed for real."}])
    store = Store(tmp_path / "a.db")
    console = _ScriptedConsole(["build it", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=registry)
    assert len(runs) == 2 and "pytest" in runs[0]
    retry_messages = client.chat_with_tools_calls[2]
    assert any("VERIFICATION FAILED" in str(m.get("content")) and "test_x" in str(m.get("content")) for m in retry_messages)
    out = console.file.getvalue()
    assert "verify  failed - back to work (1/2)" in out and "verify  passed" in out and "Fixed for real." in out
    store.close()


def test_still_failing_after_the_retries_is_reported_honestly(tmp_path):
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    fail = "$ pytest\n(in /p, exit code 1)\nFAILED"
    proj, registry, runs = _env(tmp_path, [fail, fail, fail])
    msgs = [_write(proj, 1), {"role": "assistant", "content": "done 1"}, {"role": "assistant", "content": "done 2"},
            {"role": "assistant", "content": "done 3"}]
    client = ScriptedToolClient(msgs)
    store = Store(tmp_path / "a.db")
    console = _ScriptedConsole(["build it", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=registry)
    out = console.file.getvalue()
    assert len(runs) == 3 and "still failing after 2 fixes" in out and "done 3" in out
    store.close()


def test_no_check_for_non_code_or_unknown_projects(tmp_path):
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    proj, registry, runs = _env(tmp_path, [])
    note = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_file",
            "arguments": {"path": str(proj / "notes.md"), "content": "hi"}}}]}
    client = ScriptedToolClient([note, {"role": "assistant", "content": "wrote notes"}])
    store = Store(tmp_path / "a.db")
    tui.run_repl(client, CONFIG, console=_ScriptedConsole(["write notes", "/quit"]), store=store, tool_registry=registry)
    assert runs == []
    store.close()
