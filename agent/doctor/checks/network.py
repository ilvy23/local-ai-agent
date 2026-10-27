"""Network: DNS and HTTPS to the places nightage talks to (Ollama's registry, GitHub for updates), proxies. All of it is
skippable with --no-web and runs in parallel threads. Plain GET requests only; nothing identifying is sent."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from agent.doctor.core import Context, check, info, ok, warn

TARGETS = ("ollama.com", "github.com")


@check("web", "network", "Web search", parallel=True, core=True, timeout=6)
def web(ctx: Context):
    if ctx.no_web:
        return ok("skipped")
    okay, where = ctx.p.web_reachable(2.0)
    if okay:
        return ok(f"{where} reachable")
    return warn(f"{where} not reachable", "Chat and code work offline; web research needs internet (or set AGENT_SEARXNG_URL).")


@check("network.dns", "network", "Name lookup (DNS)", network=True, timeout=8)
def dns(ctx: Context):
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as ex:
        res = list(ex.map(ctx.p.resolve, TARGETS))
    bad = [t for t, r in zip(TARGETS, res) if not r[0]]
    if len(bad) == len(TARGETS):
        return warn("cannot look up " + ", ".join(bad), "You seem to be offline (or DNS is broken). Everything local still works.")
    if bad:
        return warn("cannot look up " + ", ".join(bad), "Check your DNS settings or a firewall.")
    return ok(", ".join(f"{t} {r[1]:.0f} ms" for t, r in zip(TARGETS, res)))


@check("network.https", "network", "HTTPS to ollama.com and github.com", network=True, timeout=10)
def https(ctx: Context):
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as ex:
        res = list(ex.map(lambda t: ctx.p.https_get(f"https://{t}/", 4.0), TARGETS))
    bad = [f"{t} ({r[2]})" for t, r in zip(TARGETS, res) if not r[0]]
    if bad:
        return warn("cannot reach " + ", ".join(bad), "Needed only for `ollama pull` and `nightage update`. Check the network, a proxy or a firewall.")
    slow = [t for t, r in zip(TARGETS, res) if r[1] and r[1] > 1500]
    text = ", ".join(f"{t} {r[1]:.0f} ms" for t, r in zip(TARGETS, res))
    return warn(text + " (slow)", "A slow link makes `ollama pull` take long.") if slow else ok(text)


@check("network.proxy", "network", "Proxy settings")
def proxy(ctx: Context):
    env = ctx.env
    seen = {k: env[k] for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy") if env.get(k)}
    if not seen:
        return None
    shown = sorted({re.sub(r"//[^@/]*@", "//", v) for v in seen.values()})      # never print credentials
    msg = "proxy in use: " + ", ".join(shown)
    no = env.get("NO_PROXY") or env.get("no_proxy") or ""
    if not any(x in no for x in ("localhost", "127.0.0.1")):
        return warn(msg + "; NO_PROXY does not list localhost", "Add `NO_PROXY=localhost,127.0.0.1` so the local Ollama is not sent through the proxy.")
    return info(msg)
