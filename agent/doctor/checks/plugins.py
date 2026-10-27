"""Plugins: every plugin's own `doctor` result, then the checks plugins registered with `api.doctor_check(...)`."""

from __future__ import annotations

from agent.doctor.core import (
    Context,
    Result,
    check,
    coerce_plugin_result,
    info,
    run_isolated,
    warn,
)


@check("plugins.manifest", "plugins", "Plugin health", timeout=15)
def manifest(ctx: Context):
    from agent import plugin_host

    try:
        rows = plugin_host.doctor(None, ctx.config)
    except Exception as exc:  # noqa: BLE001
        return warn(f"plugin check failed: {type(exc).__name__}: {exc}")
    if not rows:
        return info("no plugins")
    out = []
    for level, plugin, msg in rows:
        if plugin == "-":
            out.append(Result("info" if level == "ok" else level, msg, id="plugins.folder", title="Plugin folder"))
            continue
        out.append(Result(level, msg, id=f"plugins.{plugin}", title=f"Plugin {plugin}"))
    return out


@check("plugins.checks", "plugins", "Checks added by plugins", timeout=60)
def plugin_checks(ctx: Context):
    from agent import plugin_host
    from agent.plugin_api import RUNTIME

    try:
        if not any(plugin_host.is_enabled(m.id, ctx.config) for m in plugin_host.discover()):
            return None
        plugin_host.load_quietly()
    except Exception as exc:  # noqa: BLE001
        return warn(f"could not load plugins: {type(exc).__name__}: {exc}")
    entries = list(getattr(RUNTIME, "doctor_checks", []))
    out = []
    for plugin, cid, title, fn, group in entries:
        val, err = run_isolated(fn, 8.0)
        rid = f"plugin.{plugin}.{cid}"
        if err:
            r = warn(f"could not run: {err}", f"This check belongs to the {plugin} plugin; tell its author.")
        else:
            r = coerce_plugin_result(val)
        r.id, r.title, r.group = rid, f"{title} ({plugin})", group or "plugins"
        r.fix = None                                   # plugin checks can describe a fix but never run one
        out.append(r)
    return out or None
