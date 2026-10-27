"""Terminal: size, colour, unicode, how you are connected, and whether copy and paste can reach the clipboard."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, warn


@check("terminal", "terminal", "Terminal", core=True)
def terminal(ctx: Context):
    t = ctx.p.terminal()
    bits = [f"{t['width']} columns", "colour" if t["color"] else "no colour", "unicode" if t["unicode"] else "ASCII only",
            "interactive" if t["tty"] else "not a TTY"]
    if not t["tty"]:
        return warn(", ".join(bits), "Run it in a normal terminal for the interactive screens; output stays plain.")
    if t["width"] < 40:
        return warn(", ".join(bits), "Make the window at least 40 columns wide.")
    return ok(", ".join(bits))


@check("terminal.env", "terminal", "Terminal environment")
def environment(ctx: Context):
    env = ctx.env
    where = []
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        where.append("over ssh")
    if env.get("TMUX"):
        where.append("inside tmux")
    if env.get("STY"):
        where.append("inside screen")
    term = env.get("TERM", "")
    ct = env.get("COLORTERM", "")
    bits = [f"TERM={term or 'unset'}"]
    if ct:
        bits.append(f"COLORTERM={ct}")
    bits += where
    if term in ("", "dumb"):
        return warn(", ".join(bits), "TERM is not set to a real terminal type: colours and key handling will be limited. Try TERM=xterm-256color.")
    if (env.get("TMUX") or env.get("STY")) and not term.startswith(("screen", "tmux", "xterm")):
        return warn(", ".join(bits), "Inside tmux/screen TERM should be screen-256color or tmux-256color.")
    if ct not in ("truecolor", "24bit") and "256" not in term and not env.get("WT_SESSION"):
        return info(", ".join(bits) + "; limited colours (the palette degrades gracefully)")
    return ok(", ".join(bits))


@check("terminal.clipboard", "terminal", "Clipboard")
def clipboard(ctx: Context):
    p, env, si = ctx.p, ctx.env, ctx.sysinfo
    tools = []
    if si.os_family == "macos":
        tools = ["pbcopy"]
    elif si.os_family == "windows":
        tools = ["clip"]
    elif si.os_family == "android-termux":
        tools = ["termux-clipboard-set"]
    elif "wsl" in ctx.tags:
        tools = ["clip.exe", "wl-copy", "xclip"]
    else:
        tools = (["wl-copy"] if env.get("WAYLAND_DISPLAY") else []) + ["xclip", "xsel", "wl-copy"]
    found = next((t for t in tools if p.which(t)), "")
    if found:
        return ok(f"copy works through {found}")
    ssh = bool(env.get("SSH_CONNECTION"))
    hint = {"android-termux": "pkg install termux-api (and install the Termux:API app)", "linux": "install wl-clipboard (Wayland) or xclip (X11)"}.get(
        si.os_family, "install a clipboard helper")
    return info("no clipboard helper found" + ("; over ssh the terminal's own OSC 52 support is used if it has any" if ssh else ""), hint.capitalize() + ".")
