"""Termux / Android. UNVERIFIED on a real phone (no device was available); each check reads only documented Termux files and
programs and degrades to an informational line when they are missing."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, warn


@check("storage", "system", "Shared storage", systems=("termux",), core=True)
def storage(ctx: Context):
    if ctx.p.storage_shared():
        return ok("~/storage available")
    return ok("not granted (optional): run `termux-setup-storage` to reach Downloads/Documents")


@check("termux.api", "system", "Termux:API", systems=("termux",))
def api(ctx: Context):
    if ctx.p.which("termux-battery-status"):
        return ok("Termux:API commands are installed (battery, clipboard, notifications)")
    return info("Termux:API is not installed (optional: clipboard, notifications)", "pkg install termux-api, and install the Termux:API app from F-Droid.")


@check("termux.wakelock", "system", "Wake lock", systems=("termux",))
def wakelock(ctx: Context):
    if not ctx.p.which("termux-wake-lock"):
        return info("`termux-wake-lock` not found (comes with the termux-api package)")
    return info("long jobs: run `termux-wake-lock` first, or Android may stop Termux when the screen is off")


@check("termux.battery", "system", "Battery saving", systems=("termux",), timeout=8)
def battery(ctx: Context):
    if ctx.p.which("termux-battery-status"):
        rc, out = ctx.p.run(["termux-battery-status"], timeout=5)
        if rc == 0 and '"percentage"' in out:
            import json

            try:
                d = json.loads(out)
                pct, status = d.get("percentage"), d.get("status")
                if isinstance(pct, int) and pct < 20 and status != "CHARGING":
                    return warn(f"battery {pct}% and not charging", "Plug in before a long model run.")
                return info(f"battery {pct}% ({status}); also exempt Termux from battery optimisation in Android settings")
            except ValueError:
                pass
    return info("Android's battery optimisation can stop Termux in the background: exempt it in Settings > Apps > Termux > Battery")
