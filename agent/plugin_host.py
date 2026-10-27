"""Finding, checking, enabling and loading contract plugins (the `plugin.toml` + `register(api)` kind).

The older "directory plugin with register(ctx)" kind (the WhatsApp bridge) is still loaded by agent/plugins.py exactly as
before; this module only deals with plugins that have a manifest. See docs/PLUGINS.md.

Rules that make this safe to leave switched on:
  * a plugin found in the folder is NOT run until the user enabled it (`nightage plugin enable <name>`), or lists it in
    config.yaml under plugins.enabled;
  * every plugin is loaded inside its own try/except: a broken one costs one warning line, never the app;
  * `plugin doctor` explains what is wrong.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import plugin_api
from agent.plugin_api import API_VERSION, PLUGIN_NAME, RUNTIME, PluginAPI, PluginError, Runtime

logger = logging.getLogger("nightage.plugin")

MANIFEST = "plugin.toml"
STATE_FILE = ".state.json"
DATA_SUBDIR = ".data"


# ----------------------------------------------------------------------------------------------- locations


def plugin_dir() -> Path:
    from agent.plugins import plugin_dir as _pd

    return _pd()


def data_root() -> Path:
    return plugin_dir() / DATA_SUBDIR


# ----------------------------------------------------------------------------------------------- manifest


@dataclass
class Manifest:
    id: str
    path: Path                      # the folder, or the single .py file
    kind: str                       # "dir" | "file"
    version: str = "0"
    description: str = ""
    author: str = ""
    min_nightage: str = ""
    entry: Path | None = None
    error: str = ""                 # set when the manifest itself is unusable
    repository: str = ""            # optional: where the plugin's source lives (used by `plugin update`)
    update_url: str = ""            # optional: overrides `repository` for updates


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4])


def _file_plugin_dict(path: Path) -> dict[str, Any] | None:
    """The `PLUGIN = {...}` literal of a single-file plugin, read WITHOUT running the file. None if there is none."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable/invalid python: the legacy loader reports it
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PLUGIN" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                return {}
            return value if isinstance(value, dict) else {}
    return None


def is_contract(entry: Path) -> bool:
    """True when `entry` follows the plugin.toml / PLUGIN contract (so the legacy loader must leave it alone)."""
    if entry.is_dir():
        return (entry / MANIFEST).is_file()
    return entry.suffix == ".py" and _file_plugin_dict(entry) is not None


def read_manifest(entry: Path) -> Manifest:
    """Parse one plugin folder/file. Never raises: problems land in Manifest.error."""
    if entry.is_dir():
        m = Manifest(id=entry.name, path=entry, kind="dir")
        try:
            raw = tomllib.loads((entry / MANIFEST).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            m.error = f"{MANIFEST} cannot be read: {exc}"
            return m
        m.entry = entry / str(raw.get("entry") or "plugin.py")
    else:
        m = Manifest(id=entry.stem, path=entry, kind="file")
        raw = _file_plugin_dict(entry) or {}
        m.entry = entry
    name = str(raw.get("name") or m.id)
    if not PLUGIN_NAME.match(name):
        m.error = f"name {name!r} is not valid: use lower-case letters, digits, - and _ (start with a letter)"
    m.id = name
    m.version = str(raw.get("version") or "0")
    m.description = str(raw.get("description") or "")
    m.author = str(raw.get("author") or "")
    m.min_nightage = str(raw.get("min_nightage") or "")
    m.repository = str(raw.get("repository") or "")
    m.update_url = str(raw.get("update_url") or "")
    return m


def discover() -> list[Manifest]:
    """Every contract plugin in the plugin folder (enabled or not), in a stable order."""
    d = plugin_dir()
    if not d.is_dir():
        return []
    out = []
    for entry in sorted(d.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if is_contract(entry):
            out.append(read_manifest(entry))
    return out


def find(name: str) -> Manifest | None:
    for m in discover():
        if m.id == name or m.path.name == name or m.path.stem == name:
            return m
    return None


# ----------------------------------------------------------------------------------------------- enabled state


def _state_path() -> Path:
    return plugin_dir() / STATE_FILE


def _read_state() -> dict[str, list[str]]:
    try:
        raw = json.loads(_state_path().read_text(encoding="utf-8"))
        return {"enabled": [str(x) for x in raw.get("enabled", [])], "disabled": [str(x) for x in raw.get("disabled", [])]}
    except Exception:  # noqa: BLE001 - missing or damaged state means "nothing enabled yet"
        return {"enabled": [], "disabled": []}


def _write_state(state: dict[str, list[str]]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: sorted(set(v)) for k, v in state.items()}, indent=2), encoding="utf-8")


def _cfg_list(config: dict[str, Any] | None, key: str) -> list[str]:
    block = (config or {}).get("plugins") if isinstance(config, dict) else None
    raw = block.get(key) if isinstance(block, dict) else None
    return [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []


def is_enabled(name: str, config: dict[str, Any] | None = None) -> bool:
    """Enabled = confirmed through `plugin enable` (or listed in config plugins.enabled) and not disabled anywhere.
    config plugins.disabled always wins."""
    state = _read_state()
    if name in _cfg_list(config, "disabled") or name in state["disabled"]:
        return False
    return name in state["enabled"] or name in _cfg_list(config, "enabled")


def set_enabled(name: str, on: bool) -> None:
    state = _read_state()
    state["enabled"] = [n for n in state["enabled"] if n != name]
    state["disabled"] = [n for n in state["disabled"] if n != name]
    state["enabled" if on else "disabled"].append(name)
    _write_state(state)


# ----------------------------------------------------------------------------------------------- loading


def check_compat(m: Manifest) -> str:
    """"" when this plugin may load, else why not."""
    if m.error:
        return m.error
    if m.min_nightage:
        try:
            if _vtuple(plugin_api.nightage_version()) < _vtuple(m.min_nightage):
                return f"needs nightage >= {m.min_nightage} (this is {plugin_api.nightage_version()})"
        except Exception:  # noqa: BLE001
            return f"min_nightage {m.min_nightage!r} is not a version number"
    if m.entry is None or not m.entry.is_file():
        return f"entry file {m.entry.name if m.entry else '?'} is missing"
    return ""


def _import_entry(m: Manifest) -> Any:
    """Import the plugin's entry file as module `nightage_plugin_<id>`; a folder plugin is a package, so
    `from . import helpers` works and two plugins' helper modules can never clash."""
    modname = "nightage_plugin_" + re.sub(r"\W", "_", m.id)
    for stale in [k for k in sys.modules if k == modname or k.startswith(modname + ".")]:
        del sys.modules[stale]
    assert m.entry is not None
    kwargs = {"submodule_search_locations": [str(m.path)]} if m.kind == "dir" else {}
    spec = importlib.util.spec_from_file_location(modname, m.entry, **kwargs)
    if spec is None or spec.loader is None:
        raise PluginError(f"cannot import {m.entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    return module


def load_one(m: Manifest, *, registry: Any, commands: Any, config: dict[str, Any] | None, console: Any,
             runtime: Runtime | None = None, root: Path | None = None,
             api_class: type[PluginAPI] = PluginAPI) -> PluginAPI:
    """Import one plugin and call register(api). Raises on any problem (callers decide how loudly to report it)."""
    rt = runtime or RUNTIME
    why = check_compat(m)
    if why:
        raise PluginError(why)
    rt.forget(m.id, commands)
    module = _import_entry(m)
    register = getattr(module, "register", None)
    if not callable(register):
        raise PluginError("the entry file has no `def register(api):` function")
    api = api_class(m.id, path=m.path, registry=registry, commands=commands, config=config, console=console,
                    data_root=root or data_root(), runtime=rt, plugin_version=m.version)
    rt.provided[m.id] = api.provided
    try:
        register(api)
    except BaseException:
        rt.forget(m.id, commands)                 # keep nothing half-registered
        for t in api.provided.tools:
            registry.unregister(t)
        raise
    return api


def _one_line(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]


def load_contract_plugins(ctx: Any, *, runtime: Runtime | None = None) -> list[str]:
    """Load every ENABLED contract plugin into ctx.registry. Returns the names that loaded. Never raises."""
    rt = runtime or RUNTIME
    loaded: list[str] = []
    try:
        manifests = discover()
    except Exception as exc:  # noqa: BLE001
        logger.warning("plugin discovery failed: %s", exc)
        return loaded
    from agent.ui.commands import REGISTRY

    waiting = 0
    for m in manifests:
        try:
            if not is_enabled(m.id, ctx.config):
                rt.status[m.id] = ("not enabled", "run: nightage plugin enable " + m.id)
                waiting += 1
                continue
            api = load_one(m, registry=ctx.registry, commands=REGISTRY, config=ctx.config, console=ctx.console, runtime=rt)
            rt.status[m.id] = ("loaded", api.provided.summary())
            loaded.append(m.id)
        except BaseException as exc:  # noqa: BLE001 - one broken plugin must never stop the app (even sys.exit in it)
            if isinstance(exc, KeyboardInterrupt):
                raise
            rt.status[m.id] = ("error", _one_line(exc))
            logger.warning("plugin %r failed to load: %s", m.id, exc)
            try:
                ctx.console.print(f"[yellow]plugin {m.id!r} did not load:[/yellow] {_one_line(exc)}  "
                                  f"[dim](nightage plugin doctor {m.id})[/dim]")
            except Exception:  # noqa: BLE001
                pass
    if waiting:
        logger.info("%d plugin(s) installed but not enabled", waiting)
    if not rt.session_started and loaded:
        rt.session_started = True
        rt.emit("session_start", plugins=tuple(loaded))
    return loaded


def load_quietly() -> Any:
    """Load the enabled plugins into a throwaway registry (for the CLI and the home screen, which only need what they
    register elsewhere: terminal commands, home items). Returns the Runtime."""
    from rich.console import Console

    from agent.tools.registry import ToolRegistry

    class _Ctx:
        registry = ToolRegistry()
        config: dict[str, Any] = {}
        console = Console(quiet=True)

    try:
        from agent.config import load_config

        _Ctx.config = load_config()
    except Exception:  # noqa: BLE001
        pass
    load_contract_plugins(_Ctx)
    return RUNTIME


def attach_cli(app: Any) -> None:
    """Add `nightage <plugin> ...` for every enabled plugin that called api.cli(). Called once while the CLI starts;
    never raises, never shadows a built-in command."""
    try:
        import typer.main

        if not any(m for m in discover()):
            return
        load_quietly()
        taken = {c.name for c in app.registered_commands if c.name} | {g.name for g in app.registered_groups if g.name}
        taken |= {getattr(c.callback, "__name__", "").replace("_", "-") for c in app.registered_commands}
        for name, sub in list(RUNTIME.cli_apps.items()):
            if name in taken:
                logger.warning("plugin %s: a built-in command already uses `%s`; its CLI is not added", name, name)
                continue
            app.add_typer(sub, name=name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plugin CLI setup failed: %s", exc)


# ----------------------------------------------------------------------------------------------- listing / doctor


def listing(config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """One row per contract plugin for `plugin list` and /plugins."""
    rows = []
    for m in discover():
        enabled = is_enabled(m.id, config)
        state, detail = RUNTIME.status.get(m.id, ("", ""))
        why = check_compat(m)
        if why:
            state, detail = "problem", why
        elif not enabled:
            state, detail = "off", "not enabled yet" if m.id not in _read_state()["disabled"] else "disabled"
        elif not state:
            state, detail = "enabled", "loads next time the app starts"
        prov = RUNTIME.provided.get(m.id)
        rows.append({"name": m.id, "version": m.version, "enabled": "yes" if enabled else "no", "status": state,
                     "detail": prov.summary() if prov and state == "loaded" else detail, "description": m.description,
                     "author": m.author})
    return rows


def doctor(name: str | None, config: dict[str, Any] | None = None) -> list[tuple[str, str, str]]:
    """Checks as (level, plugin, message); level is ok | warn | fail. Runs the plugin's register() against throwaway
    objects when it is enabled, so you see exactly what it would add (and the real traceback if it breaks)."""
    from agent.tools.registry import ToolRegistry
    from agent.ui.commands import Registry

    out: list[tuple[str, str, str]] = []
    d = plugin_dir()
    if not d.is_dir():
        return [("warn", "-", f"plugin folder {d} does not exist yet (create a plugin with: nightage plugin new my-plugin)")]
    manifests = discover()
    if name:
        m = find(name)
        if m is None:
            return [("fail", name, f"no plugin called {name!r} in {d}; see `nightage plugin list`")]
        manifests = [m]
    if not manifests:
        return [("ok", "-", f"no plugins in {d} (create one with: nightage plugin new my-plugin)")]
    for m in manifests:
        why = check_compat(m)
        if why:
            out.append(("fail", m.id, why))
            continue
        out.append(("ok", m.id, f"manifest fine (version {m.version}, entry {m.entry.name if m.entry else '?'})"))
        if m.kind == "dir" and m.path.name != m.id:
            out.append(("warn", m.id, f"folder is called {m.path.name!r} but the plugin name is {m.id!r}; keep them equal"))
        if not m.description:
            out.append(("warn", m.id, "no description in the manifest"))
        try:
            ast.parse(m.entry.read_text(encoding="utf-8"))                       # type: ignore[union-attr]
        except SyntaxError as exc:
            out.append(("fail", m.id, f"{m.entry.name} has a syntax error at line {exc.lineno}: {exc.msg}"))   # type: ignore[union-attr]
            continue
        if not is_enabled(m.id, config):
            out.append(("warn", m.id, "not enabled, so its code was not run. Enable it: nightage plugin enable " + m.id))
            continue
        try:
            from rich.console import Console

            api = load_one(m, registry=ToolRegistry(), commands=Registry(), config=config, console=Console(quiet=True),
                           runtime=Runtime(), root=Path(tempfile.mkdtemp(prefix="plugin-doctor-")))
            out.append(("ok", m.id, "register() ran fine and adds: " + api.provided.summary()))
            if not any([api.provided.tools, api.provided.commands, api.provided.cli, api.provided.home_items,
                        api.provided.hooks]):
                out.append(("warn", m.id, "register() did not add anything"))
            for t in api.provided.tools:
                tool = api._registry.get(t)
                if tool is not None and tool.risk == "safe":
                    out.append(("ok", m.id, f"note: tool {t} says risk=safe, so it runs without asking. Keep that only if it is read-only"))
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            frames = [f for f in traceback.extract_tb(exc.__traceback__) if str(m.path) in f.filename]
            where = f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
            out.append(("fail", m.id, f"register() failed{where}: {_one_line(exc)}"))
    return out


# ----------------------------------------------------------------------------------------------- scaffold / install


_SCAFFOLD_TOML = '''name = "{name}"
version = "0.1.0"
description = "{desc}"
author = ""
min_nightage = "0.1.0"
# entry = "plugin.py"      # the file that defines register(api); this is the default
'''

_SCAFFOLD_PLUGIN = '''"""{name}: a small plugin. Edit this file, then try it:  nightage plugin doctor {name}

Docs: docs/PLUGINS.md in the nightage project.
"""

from datetime import datetime


def register(api):
    # 1. A tool the model can call. Say clearly in `description` WHEN to use it.
    #    risk: "safe" runs without asking, "caution" asks first (the default), "dangerous" asks and wants a typed yes.
    def greet(name: str = "friend") -> str:
        greeting = api.config("greeting", "Hello")        # config.yaml:  plugins: {{{name}: {{greeting: Hi}}}}
        return f"{{greeting}}, {{name}}! It is {{datetime.now():%H:%M}}."

    api.tool(
        "{ident}_greet",
        "Greet someone by name. Use when the user asks for a greeting.",
        {{
            "type": "object",
            "properties": {{"name": {{"type": "string", "description": "who to greet"}}}},
            "required": [],
        }},
        greet,
        risk="safe",              # this tool only builds a string, so it is safe
    )

    # 2. A slash command: type /{name} in the chat. The handler gets what was typed after the command.
    api.command("{name}", "Say hello from the {name} plugin", lambda args: greet(args.strip() or "friend"),
                usage="/{name} [name]")

    # 3. A terminal command:  nightage {name} hello --name Ada
    import typer

    cli = typer.Typer(help="{name} commands")

    @cli.command()
    def hello(name: str = "friend"):
        """Print a greeting."""
        api.console.print(greet(name))

    api.cli(cli)
'''

_SCAFFOLD_TEST = '''"""Tests for the {name} plugin. Run:  python -m pytest   (from this folder; needs nightage importable)."""

from pathlib import Path

from agent.plugin_testing import load_plugin


def test_tool_works():
    api = load_plugin(Path(__file__).parent)
    assert "Hello, Ada" in api.call_tool("{ident}_greet", name="Ada")


def test_slash_command_works():
    api = load_plugin(Path(__file__).parent)
    assert "Hello, friend" in api.run_command("{name}", "")


def test_cli_works():
    api = load_plugin(Path(__file__).parent)
    result = api.run_cli("hello", "--name", "Ada")
    assert result.exit_code == 0 and "Hello, Ada" in result.output


def test_config_changes_the_greeting():
    api = load_plugin(Path(__file__).parent, config={{"greeting": "Hi"}})
    assert api.call_tool("{ident}_greet").startswith("Hi,")
'''

_SCAFFOLD_README = '''# {name}

{desc}

Enable it with `nightage plugin enable {name}`, then type `/{name}` in the chat.
Tests: `python -m pytest` in this folder. The full guide is docs/PLUGINS.md.
'''


def scaffold(name: str, dest: Path | None = None, description: str = "") -> Path:
    """Write a working hello-world plugin (tool + slash command + CLI + test file). Returns its folder."""
    if not PLUGIN_NAME.match(name):
        raise PluginError("plugin name: lower-case letters, digits, - and _ (start with a letter, at least 2 characters)")
    folder = (dest or plugin_dir()) / name
    if folder.exists():
        raise PluginError(f"{folder} already exists")
    desc = (description or f"My {name} plugin.").replace('"', "'")
    ident = re.sub(r"\W", "_", name)
    folder.mkdir(parents=True)
    (folder / MANIFEST).write_text(_SCAFFOLD_TOML.format(name=name, desc=desc), encoding="utf-8")
    (folder / "plugin.py").write_text(_SCAFFOLD_PLUGIN.format(name=name, ident=ident), encoding="utf-8")
    (folder / "test_plugin.py").write_text(_SCAFFOLD_TEST.format(name=name, ident=ident), encoding="utf-8")
    (folder / "README.md").write_text(_SCAFFOLD_README.format(name=name, desc=desc), encoding="utf-8")
    return folder


def looks_like_git_url(src: str) -> bool:
    return bool(re.match(r"^(https?://|ssh://|git@|git://)", src)) or src.endswith(".git")


def install(src: str, *, confirm: Any, force: bool = False, ref: str = "") -> Manifest:
    """Copy a local plugin folder / single file into the plugin folder, or `git clone --depth 1` a URL there.
    `confirm(question) -> bool` is asked before any remote code is fetched. The plugin is NOT enabled.
    A git install leaves an install record (`.source.json`: url, ref, commit, time) so `nightage plugin update` can find updates;
    `ref` is an optional branch or tag."""
    dest_root = plugin_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    if looks_like_git_url(src):
        if not confirm(f"Download and install a plugin from {src}?\n"
                       "A plugin is ordinary Python that runs with YOUR permissions (no sandbox): it can read your files "
                       "and use your network. Only install code you trust. It stays disabled until you run `plugin enable`."):
            raise PluginError("cancelled")
        tmp = Path(tempfile.mkdtemp(prefix="plugin-clone-"))
        try:
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), src, str(tmp / "repo")]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                raise PluginError("git clone failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1])
            placed = _place(tmp / "repo", dest_root, force)
            try:
                from agent import plugin_update

                plugin_update.write_source(placed.path, url=src, ref=ref, repo=tmp / "repo")
            except Exception:  # noqa: BLE001 - the record is a convenience; the install itself succeeded
                pass
            return placed
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    path = Path(src).expanduser()
    if not path.exists():
        raise PluginError(f"{src} does not exist")
    return _place(path, dest_root, force)


def _place(src: Path, dest_root: Path, force: bool) -> Manifest:
    if src.is_dir() and not (src / MANIFEST).is_file():
        raise PluginError(f"{src} has no {MANIFEST}: it is not a nightage plugin folder")
    if src.is_file() and _file_plugin_dict(src) is None:
        raise PluginError(f"{src.name} has no PLUGIN = {{...}} header: it is not a single-file nightage plugin")
    m = read_manifest(src)
    if m.error:
        raise PluginError(m.error)
    target = dest_root / (m.id if src.is_dir() else m.id + ".py")
    if target.exists():
        if not force:
            raise PluginError(f"a plugin called {m.id!r} is already installed ({target}); use --force to replace it")
        if is_enabled(m.id):
            set_enabled(m.id, False)              # replaced code is new code: it must be enabled again on purpose
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    if src.is_dir():
        shutil.copytree(src, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"))
    else:
        shutil.copy2(src, target)
    return read_manifest(target)


def fingerprint(m: Manifest) -> str:
    """Short hash of a plugin's files (shown when enabling so you can tell later that the code changed)."""
    h = hashlib.sha256()
    files = sorted(p for p in m.path.rglob("*") if p.is_file() and "__pycache__" not in p.parts) if m.path.is_dir() else [m.path]
    for f in files:
        if f.name == ".source.json":              # the install record is bookkeeping, not code
            continue
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]
