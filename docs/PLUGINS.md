# Writing plugins for nightage

A plugin adds something to nightage without touching its code: a **tool** the model can call, a **slash command** you type in the chat, a **terminal command** (`nightage my-plugin ...`), an entry on the **home screen**, or a **hook** that watches what happens. A plugin is one folder with two small files. If you can write a Python function, you can write a plugin.

> **Read this first: plugins are not sandboxed.** A plugin is ordinary Python that runs inside nightage with *your* permissions. It can read your files, use your network and run programs. Nothing here can stop it. Only enable plugins you wrote or read and trust. nightage never runs a plugin you have not enabled, and says so when you enable one.

Contents: [5-minute quickstart](#5-minute-quickstart) · [Anatomy](#anatomy-of-a-plugin) · [API reference](#api-reference) · [Risk levels](#risk-levels) · [Hooks](#hooks) · [Config and data](#settings-and-storage) · [Testing](#testing-your-plugin) · [Sharing](#packaging-and-sharing) · [Security](#security-model-and-best-practices) · [Troubleshooting](#troubleshooting) · [FAQ](#faq) · [Versioning](#versioning-and-compatibility)

---

## 5-minute quickstart

```bash
nightage plugin new my-plugin     # creates ~/.config/agent/plugins/my-plugin/
nightage plugin enable my-plugin  # shows what it is, asks you to confirm
nightage plugin doctor my-plugin  # a trial run: what does it add? any errors?
```

Start nightage and try the three things the starter plugin adds:

* in the chat, type `/my-plugin Ada` (a slash command);
* ask the model *"greet Ada"*: it can call the `my_plugin_greet` tool;
* in a terminal: `nightage my-plugin hello --name Ada`.

Now open `~/.config/agent/plugins/my-plugin/plugin.py` and change it. The whole file is about 30 lines with comments. Run `nightage plugin doctor my-plugin` after each edit; restart nightage to load the change. The folder also contains `test_plugin.py` (run it with `python -m pytest` inside the folder) and a README.

## Anatomy of a plugin

```
~/.config/agent/plugins/
  my-plugin/
    plugin.toml      the manifest: who/what/which version
    plugin.py        defines  def register(api): ...
    test_plugin.py   optional tests
    helpers.py       optional; import it with  from . import helpers
```

The folder is `~/.config/agent/plugins/`. Set the environment variable `NIGHTAGE_PLUGIN_DIR` (or `AGENT_PLUGIN_DIR`) to use another one.

### plugin.toml

```toml
name = "my-plugin"          # lower-case letters, digits, - and _ ; must start with a letter
version = "0.1.0"
description = "One line about what it does."
author = "You <you@example.com>"
min_nightage = "0.1.0"      # the oldest nightage it works with (optional)
entry = "plugin.py"         # the file with register(api) (optional; this is the default)
repository = "https://github.com/you/my-plugin.git"   # where updates come from (optional; `nightage plugin update`)
# update_url = "..."        # overrides `repository` for updates (optional)
```

### plugin.py

```python
def register(api):
    api.tool(...)      # and/or api.command(...), api.cli(...), api.home_item(...), api.on(...)
```

`register(api)` runs once when nightage starts (after you enabled the plugin). Everything the plugin does goes through `api`.

### Single-file plugins

For something tiny, skip the folder: put `my_thing.py` straight into the plugin folder and start it with a `PLUGIN` dict, which nightage reads without running the file:

```python
"""My tiny plugin."""
PLUGIN = {"name": "my-thing", "version": "0.1.0", "description": "Does a thing."}

def register(api):
    api.command("thing", "Does a thing", lambda args: "done")
```

Single-file plugins have no `plugin.toml`; the `PLUGIN` dict is required (without it, nightage treats the file as an older-style plugin, see the FAQ).

## API reference

`register(api)` receives one object. These are all of its members. Anything not listed here is not part of the API and may change.

### `api.tool(name, description, parameters, handler, risk="caution")`

Gives the **model** a tool it can call.

| Parameter | Meaning |
|---|---|
| `name` | Lower-case letters, digits, `_`; 2-64 characters, starting with a letter. Must not clash with a built-in tool or another plugin's (you get an error naming the clash). Tip: prefix with your plugin name. |
| `description` | Plain sentence telling the model **what it does and when to use it**. This is the most important line you write: the model reads it to decide whether to call your tool. |
| `parameters` | A JSON Schema object describing the arguments (see below). `{}` means "no arguments". |
| `handler` | A function. It is called with the arguments as keyword arguments, `handler(**arguments)`, and returns **text** (anything else is converted with `str`; `None` becomes empty). |
| `risk` | `"safe"`, `"caution"` (default) or `"dangerous"`. See [Risk levels](#risk-levels). |

Parameter schema example:

```python
api.tool(
    "count_words",
    "Count the words in a text file. Use for 'how long is this file?'.",
    {
        "type": "object",
        "properties": {
            "path":  {"type": "string",  "description": "path of the file"},
            "lines": {"type": "boolean", "description": "count lines instead of words"},
            "limit": {"type": "integer", "description": "stop after this many"},
            "mode":  {"type": "string",  "enum": ["fast", "exact"]},
        },
        "required": ["path"],
    },
    lambda path, lines=False, limit=None, mode="fast": ...,
    risk="safe",
)
```

Make every optional argument a function default. The model sometimes leaves arguments out or sends extra ones; a handler that raises is not a disaster: the error text goes back to the model (it can retry) and is logged.

Write for a small local model: short descriptions, few parameters, plain types, results of a few lines. If a result can be large, truncate it.

### `api.command(name, summary, handler, when="instant", usage="")`

Adds a **slash command** `/name` you type in the chat.

* `name`: lower-case letters, digits and `-`. Must not already exist (`/help`, `/status`, ... are taken; a clear error tells you).
* `summary`: one line shown in `/help` and in completions. Plugin commands appear under "Plugins".
* `handler(args)`: `args` is the text after the command (`/note buy milk` gives `"buy milk"`). Return the text to show, or `None`.
* `when`: what happens if the user types it while a turn is running. `"instant"` runs right away (use for anything that just reads or shows), `"step"` applies at the next step, `"deferred"` waits until the turn ends (use for anything that changes state).
* `usage`: shown in help, e.g. `"/note <text>"`.

### `api.cli(typer_app_or_function)`

Adds terminal commands under your plugin's name: `nightage <plugin-name> ...`.

```python
import typer

def register(api):
    app = typer.Typer(help="My plugin")

    @app.command()
    def hello(name: str = "friend"):
        """Say hello."""
        api.console.print(f"Hello, {name}")

    api.cli(app)            # nightage my-plugin hello --name Ada
```

You can also pass a plain function; its parameters become options (`nightage my-plugin my-function --x 1`). Terminal commands appear only for **enabled** plugins and can never replace a built-in command. The CLI uses [typer](https://typer.tiangolo.com/).

### `api.home_item(label, hint, action)`

Adds a line under **Home > Plugins**. `action` is either a function called with a rich console (`lambda console: console.print("hi")`) or a list of nightage arguments (`["my-plugin", "hello"]`).

### `api.on(event, fn)`

Runs `fn(data)` when something happens. See [Hooks](#hooks).

### `api.doctor_check(id, title, fn, group="plugins")`

Adds a check to `nightage doctor` (additive; it does not change `API_VERSION`, so guard with `getattr(api, "doctor_check", None)` if
you also support older nightage). `fn()` takes no arguments and returns a string (ok), or `(status, detail)` / `(status, detail, hint)`
with status `ok`, `warn`, `fail`, `info` or `skip`. It runs with a time limit, an exception becomes a "could not run" warning, and a
plugin check can only report: it can never offer a fix. Example and details: [DOCTOR.md](DOCTOR.md).

### `api.config(key, default=None)`

Reads *your* setting from `config.yaml`. See [Settings and storage](#settings-and-storage).

### `api.data_dir()`

A folder (a `pathlib.Path`) that only your plugin writes to; created on first use.

### `api.log` and `api.console`

* `api.log` is a standard `logging.Logger` named `nightage.plugin.<name>`: `api.log.warning("...")`. Use it for diagnostics rather than `print`.
* `api.console` is a [rich](https://rich.readthedocs.io/) `Console`. Use it for output in terminal commands (`api.console.print("[green]done[/green]")`). Do not use it from tool handlers: return text instead.

### Also available (read-only)

`api.name`, `api.plugin_version` (from the manifest), `api.path` (your plugin's folder, handy for bundled files), `api.version` (the API version, currently `1`).

### Errors

If you ask the API for something it does not allow (a name that is taken, a bad `when`, an unknown event), it raises `PluginError` with a message that says what to change. That makes `register` fail: nightage prints one warning line and goes on without your plugin. `nightage plugin doctor my-plugin` shows the message and the line.

## Risk levels

Every tool declares how risky a call is. This decides what nightage asks the user:

| `risk` | Meaning | What happens |
|---|---|---|
| `"safe"` | Only reads or computes. Cannot change anything or leak anything. | Runs without asking. |
| `"caution"` (**the default**) | Changes something small or reversible, or reads private data. | The user is asked to approve (yes / no / always for this kind of action, depending on their access level). |
| `"dangerous"` | Deletes, sends, spends, publishes, installs, or anything you cannot take back. | The user is asked and must type a full `yes`. |

Guidance: if unsure, leave the default. Use `"safe"` only when the tool truly cannot do harm (`doctor` notes every `safe` tool so you and your users can double-check). A tool that sends a message, writes outside your `data_dir()`, or runs a program is not safe.

What plugins can **not** do: lower the risk of any built-in tool, change the command classifier, edit `safety.safe_commands`, greenlists or any other safety setting, or replace a built-in tool or command. The API has no way to reach them. A risk value that is not exactly one of the three above never becomes `safe` (typos become `caution`).

## Hooks

`api.on(event, fn)` lets a plugin watch. `fn` gets one argument, a **read-only** mapping (you can read `data["outcome"]`, but changing it does nothing and raises). A hook cannot change what happens, and if it raises, the error is logged and ignored: it never interrupts a turn.

| Event | When | `data` keys |
|---|---|---|
| `session_start` | Once, after plugins loaded when nightage starts | `plugins` (names loaded) |
| `turn_start` | A turn (one prompt and the answer) begins | `prompt` |
| `turn_end` | A turn finished | `outcome` (`"done"`, `"interrupted"` or `"error"`) |
| `tool_result` | A tool call finished | `tool`, `arguments`, `result` |

Hooks run on the thread that fires them, so keep them quick and do not block. Use them for logging, counters, notifications and the like.

## Settings and storage

**Settings.** Users configure your plugin in their `config.yaml` under `plugins:` and your plugin name:

```yaml
plugins:
  my-plugin:
    greeting: Hi
    server:
      port: 9000
```

```python
api.config("greeting", "Hello")      # "Hi"  (or "Hello" if unset)
api.config("server.port", 8080)      # 9000  (dots reach into nested values)
```

You only ever see your own block. Document your keys in your README and always give a default so the plugin works with no settings.

**Storage.** `api.data_dir()` is `~/.config/agent/plugins/.data/<name>/` (next to the plugins, not inside your own folder). Keep databases, caches and state there, not in your plugin folder (that is replaced when updating) and not in the user's home.

**Enable/disable lists.** `nightage plugin enable|disable` stores your choice next to the plugins. You can also write `plugins: {enabled: [name], disabled: [name]}` in `config.yaml`; `disabled` always wins. Listing a plugin under `enabled` is your own consent, with no question asked.

## Testing your plugin

`agent.plugin_testing.load_plugin(path)` runs your `register()` through the **real** API rules with throwaway registries, so a test needs no running app and nothing leaks between tests:

```python
from pathlib import Path
from agent.plugin_testing import load_plugin

def test_greet():
    api = load_plugin(Path(__file__).parent, config={"greeting": "Hi"})   # config = your plugin's own settings
    assert api.tools["my_greet"].risk == "safe"
    assert api.call_tool("my_greet", name="Ada").startswith("Hi")
    assert "Hi" in api.run_command("my-plugin", "Ada")
    result = api.run_cli("hello", "--name", "Ada")                       # exit_code, output
    assert result.exit_code == 0
    api.emit("turn_end", outcome="done")                                   # fire a hook
```

The returned `FakeAPI` is the real API plus: `.tools`, `.commands`, `.home_items` (what you registered), `.call_tool(name, **args)`, `.run_command(name, args)`, `.run_cli(*args)`, `.emit(event, **data)`. `load_plugin(path, config=..., data_dir=...)` accepts a folder or a single file. Pass `data_dir=tmp_path` (pytest's) to inspect what your plugin wrote. Run with `python -m pytest` from your plugin folder (nightage must be importable: use its virtual environment).

Fake a network call by monkeypatching the function that makes it; test handlers directly by calling them. Tests must never need the real internet or the real model.

## Packaging and sharing

A plugin is just its folder. To share:

1. Put the folder in a git repository, with `plugin.toml` at the **top** of the repo. Include a README (what it does, the settings it reads, what it touches), a license, and bump `version` for each release.
2. Others install with:
   ```bash
   nightage plugin install https://github.com/you/my-plugin.git   # git clone --depth 1; asks first
   nightage plugin install ./path/to/my-plugin                    # a local folder or .py file
   ```
   Install copies the files and tells the user that the plugin runs with their permissions. It does **not** enable it. The user reads the code, then runs `nightage plugin enable my-plugin`.
3. To update: users run `nightage plugin update my-plugin` (git installs keep an install record, `.source.json`; it shows the changelog and a
   file-level summary and keeps the plugin switched off until the new code is confirmed, see [UPDATING.md](UPDATING.md)). `nightage plugin install <url> --ref <branch-or-tag>` pins a ref. A local-path install has no update source.
4. Dependencies: prefer the standard library. If you need a package, say so in your README; plugins run in nightage's environment, so the user installs it there (`uv pip install ...`). Keep it small: nightage runs on weak laptops and phones too.

## Security model and best practices

**What nightage guarantees**

* Plugins found in the folder do not run until you enable them (`plugin enable` asks). `plugin install` never enables.
* A broken plugin never stops the app: failures are caught per plugin and reported in one line.
* Plugins cannot weaken the safety rules, replace built-in tools or commands, or see config outside their own block.

**What nightage does not do**

* No sandbox, no permission system, no signature checking. Enabling a plugin is exactly as trusting as running a script someone sent you. When enabling, `plugin enable` shows a fingerprint of the files so you can notice a changed plugin; re-read after updating.

**For plugin authors**

* Be honest with `risk`. Default to `"caution"`. Never `"safe"` for anything that writes, sends, deletes or runs commands.
* Do the least: read only what the user asked for; write only into `api.data_dir()`.
* Never log or send secrets, and never read other tools' credentials. Ask users to put keys in your `config.yaml` block or an environment variable, not in code.
* Do not phone home. If the plugin uses the network, say where in the README.
* Validate arguments from the model (it is untrusted input): check paths, limit sizes, reject surprises.
* Return short text. Truncate big results.
* Keep `register()` fast and free of side effects (no network, no heavy imports); do the work inside handlers.

**For users**

* Read the code before enabling; it is a few small files. Prefer plugins from authors you know.
* `nightage plugin disable name` turns one off; deleting its folder removes it.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Plugin does not show up | `nightage plugin list`. It must be in the plugin folder (check `NIGHTAGE_PLUGIN_DIR`) and have a `plugin.toml` with a valid `name`. |
| Listed as `off` | Enable it: `nightage plugin enable name`. |
| `did not load` warning at start | Run `nightage plugin doctor name`: it prints the error and where it happened. |
| "tool name ... already taken" / "/x already exists" | Rename it; prefix names with your plugin name. |
| `needs nightage >= ...` | Update nightage, or lower `min_nightage` if you know it works. |
| Changes not visible | Plugins load at startup: restart nightage. |
| Model never calls the tool | Rewrite its `description` (say when to use it), simplify `parameters`, and check `/plugins` shows it loaded. Small models need clear, short descriptions. |
| `nightage my-plugin ...` not found | The plugin is not enabled, did not call `api.cli(...)`, or the name collides with a built-in command. |
| Hook never fires | Event names are exact (`turn_end`, not `turn-end`); `doctor` lists the hooks registered. |

In the chat, `/plugins` lists what is loaded and why something is not.

## FAQ

**Can a plugin change the model, the prompt or the answers?** Not through this API. It can add tools, commands and watch events. If you need more, open an issue describing the use case.

**Can I use a plugin written for another version?** Check `api.version` and the plugin's `min_nightage`. See below.

**Are older-style plugins still supported?** Yes. A folder with `__init__.py` and `register(ctx)` (no `plugin.toml`) keeps loading automatically the way it always did, with access to the store, vector index and LLM client. That style is for private integrations on your own machine; for anything you share, use the plugin API described here, which is small and stable. A single `.py` file is treated as the new kind only if it has a `PLUGIN = {...}` dict.

**Where is the data of my plugin?** `~/.config/agent/plugins/.data/<name>/`. Removing a plugin keeps its data; delete that folder too if you want it gone.

**Can two plugins talk to each other?** Not officially. Share data through files in your own data dirs, or keep each plugin independent.

**Can a plugin add a web page or GUI panel?** Not in API version 1.

**How do I see logs?** `api.log` messages go to nightage's normal log output.

## Versioning and compatibility

* The plugin API has a version number, `api.version` (currently **1**; the constant is `agent.plugin_api.API_VERSION`).
* Within one API version nothing in this document is removed or changes meaning. New features are added as new optional functions or new events, so version-1 plugins keep working.
* A breaking change bumps the API version and is announced in the release notes; nightage will keep loading version-1 plugins for at least one further release and warn.
* Use `min_nightage` in `plugin.toml` to say which nightage you need; nightage refuses to load a plugin that needs a newer one and says so in `plugin list` and `plugin doctor`.
* Plugins should depend only on what is documented here. Importing nightage internals (`agent.tools`, `agent.ui`, ...) is possible but unsupported and may break at any release.

## Examples

Three small, working plugins live in [`examples/plugins/`](../examples/plugins). Copy one into your plugin folder (or `nightage plugin install examples/plugins/hello`) and enable it:

* `hello`: the smallest plugin, one tool.
* `word-count-tool`: a useful read-only tool with a real parameter schema.
* `dev-notes`: slash commands, a hook and a data folder working together.

## Command summary

```text
nightage plugin new NAME [--path DIR] [--enable]   scaffold a working hello-world plugin
nightage plugin list                               installed plugins: version, enabled, status, what they add
nightage plugin enable NAME [--yes]                turn on (asks first)
nightage plugin disable NAME                       turn off
nightage plugin doctor [NAME]                      explain problems, with a trial run of register()
nightage plugin install SOURCE [--force] [--yes]   copy a folder / .py file, or git clone a URL (asks first)
/plugins                                           in the chat: what is loaded
```
