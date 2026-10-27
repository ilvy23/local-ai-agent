import json
import getpass

from agent import autonomy as au
from agent import tui
from agent.memory.store import Store
from agent.tools.registry import Tool, ToolRegistry


def _cmd_result(code, output, command="python3 app.py"):
    return f"$ {command}\n(in /home/x/proj, exit code {code})\n{output}"


TB = ("Traceback (most recent call last):\n  File \"/home/alice/proj/app.py\", line 12, in <module>\n"
      "    import rich_tools\nModuleNotFoundError: No module named 'rich_tools'")


def test_sanitize_strips_private_things():
    user = getpass.getuser()
    text = (f"PermissionError: [Errno 13] Permission denied: '/home/{user}/secret/key.pem' at 192.168.1.20:8080 "
            "contact me@example.com token ghp_abcdefghijklmnopqrstuvwx12345 0x7ffd1234 https://u:p@host.io/x line 42")
    out = au.sanitize(text)
    for private in ("/home/", user, "192.168", "me@example.com", "ghp_", "0x7ffd", "host.io", "42", "key.pem"):
        assert private not in out, (private, out)
    assert "PermissionError" in out and "Permission denied" in out


def test_failure_signature_for_commands_and_tools():
    assert au.failure_signature("run_command", {}, _cmd_result(0, "ok")) is None
    assert au.failure_signature("run_command", {}, _cmd_result(1, TB)).endswith("No module named 'rich_tools'")
    assert au.failure_signature("run_command", {}, _cmd_result(2, "something odd happened")) == "something odd happened"
    assert au.failure_signature("read_file", {}, "Error: file not found: x") == "Error: file not found: x"
    assert au.failure_signature("echo", {}, "echoed: hi") is None


def test_error_query_is_clean_and_has_language_context():
    q = au.error_query("ModuleNotFoundError: No module named 'rich_tools'", "cd ~/p && python3 -m pytest -q")
    assert q == "python ModuleNotFoundError: No module named rich_tools" or q.startswith("python ModuleNotFoundError")
    assert "/" not in q and len(q) <= au.MAX_QUERY_CHARS + 10
    assert au.error_query("error: linker failed", "cargo build").startswith("rust ")


def test_same_error_twice_triggers_one_lookup_then_stays_quiet():
    w = au.StuckWatcher()
    fail = _cmd_result(1, TB)
    assert w.observe("run_command", {"command": "python3 app.py"}, fail) is None
    v = w.observe("run_command", {"command": "python3 app.py --x"}, fail)
    assert v.kind == "lookup" and "ModuleNotFoundError" in v.query and v.count == 2
    assert w.observe("run_command", {"command": "python3 app.py --y"}, fail) is None      # already looked up


def test_three_different_failures_in_a_row_trigger_a_lookup_and_success_resets():
    w = au.StuckWatcher()
    assert w.observe("run_command", {"command": "a"}, _cmd_result(1, "error: first thing broke")) is None
    assert w.observe("run_command", {"command": "b"}, _cmd_result(1, "error: second thing broke")) is None
    assert w.observe("echo", {}, "echoed: fine") is None                                  # success: streak reset
    assert w.observe("run_command", {"command": "c"}, _cmd_result(1, "error: third thing broke")) is None
    assert w.observe("run_command", {"command": "d"}, _cmd_result(1, "error: fourth thing broke")) is None
    v = w.observe("run_command", {"command": "e"}, _cmd_result(1, "error: fifth thing broke"))
    assert v and v.kind == "lookup"


def test_identical_call_three_times_gets_a_change_of_approach_nudge():
    w = au.StuckWatcher(max_lookups=0)                                                    # lookups exhausted
    args = {"path": "/x"}
    assert w.observe("read_file", args, "Error: no such file") is None
    assert w.observe("read_file", args, "Error: no such file") is None
    v = w.observe("read_file", args, "Error: no such file")
    assert v.kind == "repeat" and "identical read_file call" in v.message


def test_lookup_budget_and_denied_calls():
    w = au.StuckWatcher(max_lookups=1)
    for i in range(2):
        v = w.observe("run_command", {"command": f"x{i}"}, _cmd_result(1, "error: alpha"))
    assert v.kind == "lookup"
    for i in range(2):
        v = w.observe("run_command", {"command": f"y{i}"}, _cmd_result(1, "error: beta is different"))
    assert v is None                                                                       # budget used up
    assert w.observe("run_command", {}, "User denied execution.", denied=True) is None


def test_enabled_switches(monkeypatch):
    monkeypatch.delenv("AGENT_AUTO_LOOKUP", raising=False)
    assert au.enabled({}) and not au.enabled({"autonomy": {"web_lookup": False}})
    monkeypatch.setenv("AGENT_AUTO_LOOKUP", "0")
    assert not au.enabled({})


def test_agent_turn_looks_the_error_up_when_stuck(tmp_path, monkeypatch):
    from agent.research.fetch import Fetched
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    searches = []                                  # run_repl registers the real web tools: fake the network
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: searches.append(q) or [
        {"url": "https://docs.example.org/e42", "title": "E42 docs", "snippet": "turn E42 off"}])
    monkeypatch.setattr("agent.tools.web._http_fetch", lambda url: Fetched(
        url, html="<html><body><article>" + "<p>To fix error E42 set the E42 option to off in the backend config file.</p>" * 12 + "</article></body></html>"))
    registry = ToolRegistry()
    registry.register(Tool(name="flaky", description="always fails", parameters={"type": "object", "properties": {"n": {"type": "integer"}}},
                           handler=lambda n=0: "Error: backend exploded with code E42"))
    call = lambda n: {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "flaky", "arguments": {"n": n}}}]}
    client = ScriptedToolClient([call(1), call(2), {"role": "assistant", "content": "fixed it"}])
    store = Store(tmp_path / "a.db")
    console = _ScriptedConsole(["do the thing", "/quit"])
    tui.run_repl(client, CONFIG, console=console, store=store, tool_registry=registry)
    assert len(searches) == 1 and "backend exploded" in searches[0] and "E42" in searches[0]
    third_call_messages = client.chat_with_tools_calls[2]
    assert any("AUTOMATIC WEB LOOKUP" in str(m.get("content")) and "set the E42 option to off" in str(m.get("content"))
               for m in third_call_messages)
    assert "looking it up" in console.file.getvalue()
    store.close()


def test_switch_off_means_no_lookup(tmp_path, monkeypatch):
    from tests.test_tui import CONFIG, ScriptedToolClient, _ScriptedConsole

    searches = []
    monkeypatch.setattr("agent.tools.web._search_ddg", lambda q, k, df=None: searches.append(q) or [])
    monkeypatch.setenv("AGENT_AUTO_LOOKUP", "0")
    registry = ToolRegistry()
    registry.register(Tool(name="flaky", description="x", parameters={"type": "object", "properties": {}},
                           handler=lambda **k: "Error: backend exploded"))
    call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "flaky", "arguments": {}}}]}
    client = ScriptedToolClient([call, call, {"role": "assistant", "content": "done"}])
    store = Store(tmp_path / "a.db")
    tui.run_repl(client, CONFIG, console=_ScriptedConsole(["go", "/quit"]), store=store, tool_registry=registry)
    assert searches == []
    store.close()


def test_watcher_state_round_trips_and_tolerates_junk():
    w = au.StuckWatcher(max_lookups=5)
    w.observe("run_command", {"command": "a"}, _cmd_result(1, TB))
    w.observe("run_command", {"command": "a"}, _cmd_result(1, TB))
    w2 = au.StuckWatcher(max_lookups=5)
    w2.load(json.loads(json.dumps(w.state())))
    assert w2.state() == w.state() and w2.last_call == w.last_call and w2.looked_up == w.looked_up and w2.lookups == 1
    w2.load({"lookups": "x", "seen": [], "last_call": [1]})                        # damaged: defaults, no crash
    assert (w2.lookups, w2.seen, w2.last_call) == (0, {}, None)
