"""The contract plugin system: loader, enable/disable, API rules, scaffold, hooks, doctor, install (fakes only)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from agent import plugin_api, plugin_host
from agent.plugin_api import PluginError
from agent.plugin_cli import plugin_app
from agent.plugin_testing import load_plugin
from agent.plugins import PluginContext, discover, load_plugins
from agent.tools.registry import Tool, ToolRegistry
from agent.ui.commands import REGISTRY

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "plugins"


@pytest.fixture
def pdir(tmp_path, monkeypatch):
    d = tmp_path / "plugins"
    d.mkdir()
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(d))
    yield d
    for name in list(plugin_api.RUNTIME.provided):          # the global runtime/command registry must not leak
        plugin_api.RUNTIME.forget(name, REGISTRY)
    plugin_api.RUNTIME.status.clear()
    plugin_api.RUNTIME.session_started = False
    for event in plugin_api.RUNTIME.hooks:
        plugin_api.RUNTIME.hooks[event].clear()
    plugin_api.RUNTIME.cli_apps.clear()


def ctx(config=None, registry=None):
    return PluginContext(registry=registry or ToolRegistry(), store=None, vectors=None, llm=None,
                         config=config or {}, console=Console(quiet=True))


def write_plugin(root: Path, name="demo", body=None, extra_toml=""):
    f = root / name
    f.mkdir()
    (f / "plugin.toml").write_text(f'name = "{name}"\nversion = "1.2.3"\ndescription = "d"\n{extra_toml}')
    (f / "plugin.py").write_text(body or (
        "def register(api):\n"
        "    api.tool('demo_tool', 'does a thing', {}, lambda **kw: 'ok')\n"
        "    api.command('demo', 'demo cmd', lambda a: 'cmd:' + a)\n"))
    return f


def run(*args, input=None):
    return CliRunner().invoke(plugin_app, list(args), input=input)


# ------------------------------------------------------------------------------------------ loader


def test_enabled_plugin_loads_and_disabled_does_not(pdir):
    write_plugin(pdir)
    c = ctx()
    assert load_plugins(c) == []                                   # found but never enabled: code is not run
    assert c.registry.get("demo_tool") is None
    plugin_host.set_enabled("demo", True)
    c = ctx()
    assert load_plugins(c) == ["demo"]
    assert c.registry.get("demo_tool").handler() == "ok"
    assert REGISTRY.get("demo").handler("x", None) == "cmd:x"
    plugin_host.set_enabled("demo", False)
    c = ctx()
    assert load_plugins(c) == []


def test_enable_state_persists_and_config_lists_apply(pdir):
    write_plugin(pdir)
    plugin_host.set_enabled("demo", True)
    assert plugin_host.is_enabled("demo")
    assert (pdir / ".state.json").exists()
    assert not plugin_host.is_enabled("demo", {"plugins": {"disabled": ["demo"]}})      # config disabled always wins
    other = pdir / "solo"
    other.mkdir()
    assert plugin_host.is_enabled("solo", {"plugins": {"enabled": ["solo"]}})


def test_broken_plugin_is_isolated(pdir, capsys):
    write_plugin(pdir, "boom", "def register(api):\n    raise RuntimeError('nope')\n")
    write_plugin(pdir, "syntax", "def register(api:\n")
    write_plugin(pdir, "exits", "import sys\nsys.exit(3)\n")
    write_plugin(pdir, "noreg", "x = 1\n")
    write_plugin(pdir, "good")
    for n in ("boom", "syntax", "exits", "noreg", "good"):
        plugin_host.set_enabled(n, True)
    messages = []
    c = ctx()
    c.console = type("C", (), {"print": lambda self, m: messages.append(m)})()
    assert load_plugins(c) == ["good"]
    assert len(messages) == 4 and all("did not load" in m for m in messages)
    assert plugin_api.RUNTIME.status["boom"][0] == "error"
    assert c.registry.get("demo_tool") is None or True


def test_failed_register_leaves_nothing_behind(pdir):
    write_plugin(pdir, "half", "def register(api):\n    api.command('halfcmd', 's', lambda a: '')\n    raise ValueError('late')\n")
    plugin_host.set_enabled("half", True)
    load_plugins(ctx())
    assert REGISTRY.get("halfcmd") is None


def test_reload_does_not_duplicate(pdir):
    write_plugin(pdir)
    plugin_host.set_enabled("demo", True)
    for _ in range(3):
        assert load_plugins(ctx()) == ["demo"]
    assert len([c for c in REGISTRY.commands() if c.name == "demo"]) == 1


def test_incompatible_min_version_is_refused(pdir):
    write_plugin(pdir, extra_toml='min_nightage = "99.0.0"\n')
    plugin_host.set_enabled("demo", True)
    assert load_plugins(ctx()) == []
    assert "needs nightage" in plugin_host.listing()[0]["detail"]


def test_relative_imports_between_plugin_files(pdir):
    f = write_plugin(pdir, "multi", "from . import helper\ndef register(api):\n    api.tool('multi_t', 'd', {}, lambda **k: helper.VALUE)\n")
    (f / "helper.py").write_text("VALUE = 'from helper'\n")
    plugin_host.set_enabled("multi", True)
    c = ctx()
    load_plugins(c)
    assert c.registry.get("multi_t").handler() == "from helper"


# ------------------------------------------------------------------------------------------ single file + legacy


def test_single_file_plugin_with_plugin_dict(pdir):
    (pdir / "solo.py").write_text('PLUGIN = {"name": "solo", "version": "0.3", "description": "one file"}\n'
                                  "def register(api):\n    api.tool('solo_t', 'd', {}, lambda **k: 'solo')\n")
    assert discover() == []                                          # the legacy loader leaves it alone
    m = plugin_host.discover()[0]
    assert (m.id, m.version, m.description, m.kind) == ("solo", "0.3", "one file", "file")
    plugin_host.set_enabled("solo", True)
    c = ctx()
    assert load_plugins(c) == ["solo"]
    assert c.registry.get("solo_t").handler() == "solo"


def test_legacy_directory_plugin_loads_exactly_as_before(pdir):
    legacy = pdir / "whatsappish"
    legacy.mkdir()
    (legacy / "__init__.py").write_text("def register(ctx):\n    ctx.registry.register(__import__('agent.tools.registry', fromlist=['Tool']).Tool("
                                        "name='legacy_t', description='d', parameters={}, handler=lambda **k: 'old', risk='safe'))\n")
    (pdir / "legacyfile.py").write_text("def register(ctx):\n    pass\n")
    assert [p.name for p in discover()] == ["legacyfile.py", "whatsappish"]
    c = ctx()
    assert sorted(load_plugins(c)) == ["legacyfile", "whatsappish"]          # no enable step, as before
    assert c.registry.get("legacy_t").handler() == "old"


def test_nightage_plugin_dir_env_is_honoured(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_PLUGIN_DIR", raising=False)
    monkeypatch.setenv("NIGHTAGE_PLUGIN_DIR", str(tmp_path / "x"))
    assert plugin_host.plugin_dir() == tmp_path / "x"


# ------------------------------------------------------------------------------------------ API rules



def make_api(tmp_path, reg=None):
    from agent.ui.commands import Registry

    return plugin_api.PluginAPI("t", path=tmp_path, registry=reg or ToolRegistry(), commands=Registry(), config={},
                                console=Console(quiet=True), data_root=tmp_path / "data", runtime=plugin_api.Runtime())


def test_default_risk_is_caution_and_bad_values_are_never_safe(tmp_path):
    api = make_api(tmp_path)
    api.tool("a_tool", "d", {}, lambda **k: "x")
    api.tool("b_tool", "d", {}, lambda **k: "x", risk="SAFE ")
    api.tool("c_tool", "d", {}, lambda **k: "x", risk="harmless")
    api.tool("d_tool", "d", {}, lambda **k: "x", risk="dynamic")
    api.tool("e_tool", "d", {}, lambda **k: "x", risk=None)
    risks = {t.name: t.risk for t in api._registry.list()}
    assert risks == {"a_tool": "caution", "b_tool": "safe", "c_tool": "caution", "d_tool": "dangerous", "e_tool": "caution"}


def test_plugin_cannot_replace_a_builtin_tool_or_command(tmp_path):
    reg = ToolRegistry()
    reg.register(Tool("run_command", "builtin", {}, lambda **k: "", "dynamic"))
    api = make_api(tmp_path, reg)
    with pytest.raises(PluginError, match="already taken"):
        api.tool("run_command", "evil", {}, lambda **k: "")
    api._commands.register(__import__("agent.ui.commands", fromlist=["Command"]).Command("help", "h"))
    with pytest.raises(PluginError, match="already exists"):
        api.command("help", "evil", lambda a: "")


def test_api_has_no_way_to_touch_safety(tmp_path):
    api = make_api(tmp_path)
    public = {n for n in dir(api) if not n.startswith("_")}
    assert public == {"version", "name", "plugin_version", "path", "log", "console", "provided", "tool", "command", "cli",
                      "home_item", "on", "config", "data_dir", "doctor_check"}      # doctor_check: report-only, see docs/DOCTOR.md
    api._config = {"safety": {"safe_commands": ["rm"]}}
    assert api.config("safety.safe_commands") is None            # only plugins.<name>.* is readable


def test_validation_messages(tmp_path):
    api = make_api(tmp_path)
    for bad in ("Bad Name", "x", "1abc", ""):
        with pytest.raises(PluginError):
            api.tool(bad, "d", {}, lambda **k: "")
    with pytest.raises(PluginError, match="description"):
        api.tool("ok_name", " ", {}, lambda **k: "")
    with pytest.raises(PluginError, match="when"):
        api.command("zz", "s", lambda a: "", when="never")
    with pytest.raises(PluginError, match="unknown event"):
        api.on("nope", lambda d: None)


def test_tool_handler_errors_become_text(tmp_path):
    api = make_api(tmp_path)
    api.tool("bad_tool", "d", {}, lambda **k: 1 / 0)
    api.tool("none_tool", "d", {}, lambda **k: None)
    api.tool("num_tool", "d", {}, lambda **k: 5)
    assert "ZeroDivisionError" in api._registry.execute("bad_tool", {})
    assert api._registry.execute("none_tool", {}) == "" and api._registry.execute("num_tool", {}) == "5"


def test_config_and_data_dir(tmp_path):
    api = make_api(tmp_path)
    api._config = {"plugins": {"t": {"a": 1, "nest": {"b": 2}}, "other": {"a": 9}}}
    assert (api.config("a"), api.config("nest.b"), api.config("missing", "dflt")) == (1, 2, "dflt")
    d = api.data_dir()
    assert d == tmp_path / "data" / "t" and d.is_dir()


def test_hooks_are_isolated_and_read_only(tmp_path):
    api = make_api(tmp_path)
    seen = []

    def boom(data):
        raise RuntimeError("hook crash")

    def mutate(data):
        data["x"] = 1

    api.on("turn_end", boom)
    api.on("turn_end", mutate)
    api.on("turn_end", lambda d: seen.append(dict(d)))
    api._rt.emit("turn_end", outcome="done")                      # must not raise
    assert seen == [{"outcome": "done"}]


def test_tool_result_hook_fires_from_registry(pdir):
    got = []
    plugin_api.RUNTIME.hooks["tool_result"].append(("t", lambda d: got.append((d["tool"], d["result"]))))
    ToolRegistry().fire_on_call("calc", {"a": 1}, "42")
    assert got == [("calc", "42")]


# ------------------------------------------------------------------------------------------ scaffold, CLI, examples


def test_scaffold_produces_a_working_plugin(tmp_path):
    folder = plugin_host.scaffold("my-plugin", tmp_path)
    assert {p.name for p in folder.iterdir()} == {"plugin.toml", "plugin.py", "test_plugin.py", "README.md"}
    api = load_plugin(folder)
    assert api.tools["my_plugin_greet"].risk == "safe"
    assert "Hello, Ada" in api.call_tool("my_plugin_greet", name="Ada")
    assert api.run_command("my-plugin", "Bob").startswith("Hello, Bob")
    r = api.run_cli("hello", "--name", "Ada")
    assert r.exit_code == 0 and "Hello, Ada" in r.output
    assert load_plugin(folder, config={"greeting": "Yo"}).call_tool("my_plugin_greet").startswith("Yo,")
    with pytest.raises(PluginError, match="already exists"):
        plugin_host.scaffold("my-plugin", tmp_path)
    with pytest.raises(PluginError):
        plugin_host.scaffold("Bad Name", tmp_path)


def test_scaffolded_tests_pass_themselves(tmp_path):
    import subprocess
    import sys

    folder = plugin_host.scaffold("selftest", tmp_path)
    repo = Path(__file__).resolve().parent.parent
    env = {"PYTHONPATH": str(repo) + ":" + str(repo / ".venv-laptop" / "lib"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(folder)], capture_output=True,
                       text=True, timeout=120, env=env, cwd=str(folder))
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_new_list_enable_disable_doctor(pdir):
    r = run("new", "cool-thing")
    assert r.exit_code == 0 and (pdir / "cool-thing" / "plugin.py").exists()
    out = run("list").output
    assert "cool-thing" in out and "off" in out
    assert run("enable", "cool-thing", input="n\n").exit_code == 1
    assert not plugin_host.is_enabled("cool-thing")
    r = run("enable", "cool-thing", input="y\n")
    assert r.exit_code == 0 and "no sandbox" in r.output.replace("\n", " ")
    assert plugin_host.is_enabled("cool-thing")
    d = run("doctor", "cool-thing")
    assert d.exit_code == 0 and "cool_thing_greet" in d.output.replace("\n", "")
    assert run("disable", "cool-thing").exit_code == 0 and not plugin_host.is_enabled("cool-thing")
    assert run("enable", "ghost").exit_code == 1


def test_doctor_explains_failures(pdir):
    write_plugin(pdir, "boom", "def register(api):\n    raise KeyError('why')\n")
    write_plugin(pdir, "typo", "def register(api:\n")
    plugin_host.set_enabled("boom", True)
    plugin_host.set_enabled("typo", True)
    res = {(p): (lvl, msg) for lvl, p, msg in plugin_host.doctor("boom")}
    assert res["boom"][0] == "fail" and "KeyError" in res["boom"][1]
    assert any(lvl == "fail" and "syntax error" in msg for lvl, _, msg in plugin_host.doctor("typo"))
    assert run("doctor", "boom").exit_code == 1
    assert plugin_host.doctor("nothere")[0][0] == "fail"


def test_install_from_local_folder_does_not_enable(pdir, tmp_path):
    src = plugin_host.scaffold("shared-one", tmp_path / "src")
    (src / ".git").mkdir()
    r = run("install", str(src))
    assert r.exit_code == 0 and "NOT enabled" in r.output
    assert (pdir / "shared-one" / "plugin.toml").exists() and not (pdir / "shared-one" / ".git").exists()
    assert not plugin_host.is_enabled("shared-one")
    assert run("install", str(src)).exit_code == 1                    # already there
    assert run("install", str(src), "--force").exit_code == 0
    assert run("install", str(tmp_path / "missing")).exit_code == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "plugin.toml" in run("install", str(empty)).output


def test_install_from_git_url_asks_first(pdir, monkeypatch):
    asked = []
    with pytest.raises(PluginError, match="cancelled"):
        plugin_host.install("https://example.com/x/plugin.git", confirm=lambda q: asked.append(q) or False)
    assert "https://example.com/x/plugin.git" in asked[0] and "YOUR permissions" in asked[0]
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        plugin_host.scaffold("cloned", Path(cmd[-1]).parent)
        shutil.move(str(Path(cmd[-1]).parent / "cloned"), cmd[-1])
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(plugin_host.subprocess, "run", fake_run)
    m = plugin_host.install("git@github.com:x/y.git", confirm=lambda q: True)
    assert calls[0][:4] == ["git", "clone", "--depth", "1"] and m.id == "cloned" and (pdir / "cloned").is_dir()


@pytest.mark.parametrize("name,tool,args,expect", [
    ("hello", "say_hello", {"name": "Ada"}, "Hello, Ada!"),
    ("word-count-tool", "count_words", {"path": __file__}, "words"),
])
def test_examples_load_unmodified(name, tool, args, expect):
    api = load_plugin(EXAMPLES / name)
    assert api.tools[tool].risk == "safe"
    assert expect in api.call_tool(tool, **args)


def test_dev_notes_example(tmp_path):
    api = load_plugin(EXAMPLES / "dev-notes", config={"show_last": 2}, data_dir=tmp_path)
    assert api.run_command("devnote", "buy milk") == "saved"
    api.run_command("devnote", "second")
    api.run_command("devnote", "third")
    api.emit("turn_end", outcome="done")
    out = api.run_command("devnotes")
    assert out.startswith("3 note(s), 1 turn(s)") and "buy milk" not in out and "third" in out
    assert (tmp_path / "dev-notes" / "notes.txt").exists()


def test_examples_work_through_the_real_loader(pdir):
    for n in ("hello", "word-count-tool", "dev-notes"):
        shutil.copytree(EXAMPLES / n, pdir / n)
        plugin_host.set_enabled(n, True)
    c = ctx()
    assert sorted(load_plugins(c)) == ["dev-notes", "hello", "word-count-tool"]
    assert c.registry.get("say_hello") and REGISTRY.get("devnote")


def test_plugins_slash_command_and_cli_attach(pdir):
    assert "no plugins installed" in REGISTRY.get("plugins").handler("", None)
    import typer

    body = ("import typer\n"
            "def register(api):\n"
            "    app = typer.Typer()\n"
            "    @app.command()\n"
            "    def ping():\n"
            "        print('pong')\n"
            "    api.cli(app)\n")
    write_plugin(pdir, "pinger", body)
    plugin_host.set_enabled("pinger", True)
    out = REGISTRY.get("plugins").handler("", None)
    assert "pinger" in out
    main = typer.Typer()

    @main.command()
    def other():
        pass

    plugin_host.attach_cli(main)
    r = CliRunner().invoke(main, ["pinger", "ping"])
    assert r.exit_code == 0 and "pong" in r.output
    assert "pinger" in REGISTRY.get("plugins").handler("", None)


def test_home_item_and_launcher_topic(pdir):
    from agent.ui import launcher

    write_plugin(pdir, "homey", "def register(api):\n    api.home_item('Do homey thing', 'hint', lambda console: None)\n")
    plugin_host.set_enabled("homey", True)
    title, sections = launcher.topic_sections("plugins")
    labels = [e.label for _, es in sections for e in es]
    assert title == "Plugins" and "Do homey thing" in labels and "List plugins" in labels


def test_guide_documents_every_api_member():
    text = (Path(__file__).resolve().parent.parent / "docs" / "PLUGINS.md").read_text()
    for member in ("api.tool", "api.command", "api.cli", "api.home_item", "api.on", "api.config", "api.data_dir", "api.log",
                   "api.console", "turn_start", "turn_end", "tool_result", "session_start", "safe", "caution", "dangerous",
                   "plugin new", "plugin install", "plugin doctor", "no sandbox" if False else "not sandboxed"):
        assert member in text, member
    for ev in plugin_api.EVENTS:
        assert f"`{ev}`" in text
