"""Python: version, environment, text encoding."""

from __future__ import annotations

from agent.doctor.core import Context, check, fail, info, ok, warn


@check("python", "python", "Python", core=True)
def python_version(ctx: Context):
    maj, mino, mic = ctx.p.python_version()
    ver = f"{maj}.{mino}.{mic}"
    if (maj, mino) >= (3, 11):
        return ok(ver)
    return fail(f"{ver} (3.11 or newer needed)", "Install Python 3.11+ and reinstall the project.")


@check("python.env", "python", "Python environment")
def python_environment(ctx: Context):
    pi = ctx.p.platform_info()
    where = "virtual environment" if pi.get("venv") == "yes" else "system / user Python"
    return info(f"{pi.get('implementation', 'Python')} {pi.get('python', '?')} ({pi.get('machine', '?')}), {where}")


@check("python.encoding", "python", "Text encoding")
def python_encoding(ctx: Context):
    env = ctx.env
    lang = (env.get("LC_ALL") or env.get("LC_CTYPE") or env.get("LANG") or "").lower()
    if env.get("PYTHONUTF8") == "1" or "utf" in lang:
        return ok(f"UTF-8 ({lang or 'PYTHONUTF8=1'})")
    if not lang and ctx.sysinfo.os_family in ("windows", "macos"):
        return ok("platform default (UTF-8 capable)")
    return warn(f"locale {lang or 'not set'} is not UTF-8: non-English text and symbols may break",
                "Set LANG=C.UTF-8 (or en_US.UTF-8) in your shell profile.")
