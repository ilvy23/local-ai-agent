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
    if _local_ollama(ctx) and ctx.ollama.reachable:
        return info("Ollama runs on this phone: run `termux-wake-lock` (or tap Acquire wakelock in the Termux notification) so Android does "
                    "not suspend it when the screen is off")
    return info("long jobs: run `termux-wake-lock` first, or Android may stop Termux when the screen is off")


def _local_ollama(ctx: Context) -> bool:
    url = ctx.ollama.url or ""
    if not url:
        return True
    from agent.doctor.checks.ollama import is_remote

    return not is_remote(url)


@check("termux.ollama", "system", "Ollama on this phone", systems=("termux",))
def ollama_here(ctx: Context):
    """One line a phone user can act on: is the program installed, is the server up, is the chat model there."""
    oll = ctx.ollama
    if not _local_ollama(ctx):
        return info(f"the models run on {oll.url} (remote), not on this phone")
    binary = ctx.p.which("ollama")
    if not binary:
        return warn("not installed on this phone", "Run `pkg install ollama` (or use a model on another machine: set `ollama.host`).")
    if not oll.reachable:
        return warn("installed, but the server is not running", "Run `nightage ollama start` (it also starts by itself when you open a chat).")
    chat = str(ctx.section("models").get("chat") or "")
    from agent.doctor.probes import model_installed

    if chat and not model_installed(chat, oll.models):
        return warn(f"running, but the chat model {chat} is not downloaded", f"Run `nightage pull {chat}` (shows a percent bar; Ctrl+C is safe, run it again to resume).")
    return ok(f"running, {len(oll.models)} model(s) installed")


@check("termux.phantom", "system", "Android background limits", systems=("termux",))
def phantom(ctx: Context):
    """UNVERIFIED hints (no phone was available): Android 12+ may kill Termux's child processes (the 'phantom process killer')."""
    if not (_local_ollama(ctx) and (ctx.p.which("ollama") or ctx.ollama.reachable)):
        return None
    return info("Android 12+ can silently kill background programs such as the Ollama server (UNVERIFIED on your device)",
                "If the server dies while you chat: (1) keep Termux open and run `termux-wake-lock`; (2) on Android 14+ try Settings > Developer options > "
                "'Disable child process restrictions'; (3) as a last resort, from a PC with adb: "
                "`adb shell settings put global settings_enable_monitor_phantom_procs false` (Android 12-13; changes a system setting, "
                "see docs/TERMUX.md first).")


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
