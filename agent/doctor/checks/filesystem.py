"""Files and folders: the data folder, the config and state folders, the web token's permissions."""

from __future__ import annotations

from pathlib import Path

from agent.doctor import fixes
from agent.doctor.core import Context, Result, check, fail, info, ok, warn


def config_dir(ctx: Context) -> Path:
    base = ctx.env.get("XDG_CONFIG_HOME")
    return (Path(base) if base else ctx.p.home() / ".config") / "agent"


def state_dir(ctx: Context) -> Path:
    base = ctx.env.get("XDG_STATE_HOME")
    return (Path(base) if base else ctx.p.home() / ".local" / "state") / "nightage"


@check("data", "filesystem", "Data folder", core=True)
def data(ctx: Context):
    data_dir, db = ctx.p.data_paths(ctx.config)
    if ctx.p.writable(data_dir):
        return ok(f"{data_dir} ({'database exists' if db.exists() else 'database will be created'})")
    kw = {}
    try:
        cmd, fn = fixes.chmod_path("u+w", data_dir, home=ctx.p.home(), extra_roots=(Path(data_dir),))
        kw = {"command": cmd, "fix": fn} if ctx.p.exists(data_dir) else {}
    except fixes.FixRefused:
        pass
    return fail(f"{data_dir} is not writable", f"Make it writable (chmod u+w {data_dir}) or set data.db_path to another location.", **kw)


@check("filesystem.dirs", "filesystem", "Config and state folders")
def dirs(ctx: Context):
    out: list[Result] = []
    home = ctx.p.home()
    for label, d, why in (("config", config_dir(ctx), "web token, trusted addresses, plugins"),
                          ("state", state_dir(ctx), "input history, update-check cache")):
        if ctx.p.exists(d):
            if ctx.p.writable(d):
                out.append(ok(f"{label}: {d}", id=f"filesystem.{label}_dir", title=f"{label.title()} folder"))
            else:
                out.append(fail(f"{label}: {d} is not writable ({why})", f"chmod u+w {d}", id=f"filesystem.{label}_dir",
                                title=f"{label.title()} folder"))
        else:
            kw = {}
            try:
                cmd, fn = fixes.make_dir(d, home=home)
                kw = {"command": cmd, "fix": fn}
            except fixes.FixRefused:
                pass
            ok_parent = ctx.p.writable(d)
            r = (info if ok_parent else warn)(f"{label}: {d} does not exist yet ({why}); it is created when needed" + ("" if ok_parent else
                                                                                                                       " but its parent is not writable"), "", **(kw if ok_parent else {}))
            r.id, r.title = f"filesystem.{label}_dir", f"{label.title()} folder"
            out.append(r)
    return out


@check("filesystem.webtoken", "filesystem", "Web access token file")
def webtoken(ctx: Context):
    p = config_dir(ctx) / "web-token"
    mode = ctx.p.file_mode(p)
    if mode is None:
        return info(f"{p} does not exist yet (created the first time you run `nightage web`)")
    if ctx.sysinfo.os_family == "windows":
        return info("file modes do not apply on Windows; keep the file in your user profile")
    if mode & 0o077:
        kw = {}
        try:
            cmd, fn = fixes.chmod_path("600", p, home=ctx.p.home(), extra_roots=(config_dir(ctx),))
            kw = {"command": cmd, "fix": fn}
        except fixes.FixRefused:
            pass
        return warn(f"{p} is readable by other users (mode {mode:04o})", "The token is a password: make it private (mode 600).", **kw)
    return ok(f"{p} (mode {mode:04o})")
