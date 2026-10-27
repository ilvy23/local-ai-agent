"""Ollama: is the server there, which one (local / remote / the optional Vulkan one), how new, what is loaded."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

from agent.doctor.core import Context, check, fail, info, ok, skip, warn

MIN_OLLAMA = (0, 4, 0)       # conservative floor: tool calling and /api/ps are older than this; below it nightage is untested


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def is_remote(url: str) -> bool:
    try:
        from agent import llm

        return bool(llm.is_remote(url))
    except Exception:  # noqa: BLE001
        host = (urlparse(url).hostname or "").lower()
        return host not in ("", "localhost", "127.0.0.1", "::1")


@check("ollama", "ollama", "Ollama", core=True, timeout=10)
def server(ctx: Context):
    oll = ctx.ollama
    remote = bool(oll.url) and is_remote(oll.url)
    if oll.reachable:
        return ok(f"{'remote ' if remote else ''}running, version {oll.version or '?'}, {len(oll.models)} model(s) installed")
    if remote:
        return fail(f"remote server {oll.url} not reachable" + (f" ({oll.error})" if oll.error else ""),
                    "Is the other machine on, and Tailscale up on both? On it, Ollama must listen on the network (OLLAMA_HOST=0.0.0.0).")
    return fail("not running" + (f" ({oll.error})" if oll.error else ""),
                "Start it with `ollama serve` or `nightage ollama start` (it also starts by itself when a chat begins, unless `ollama.autostart` is off). If the `ollama` command is "
                "missing, install it from https://ollama.com; on Termux: `pkg install ollama`; or point `ollama.host` at another machine.")


@check("remote_latency", "ollama", "Remote Ollama latency", core=True, timeout=12)
def remote_latency(ctx: Context):
    oll = ctx.ollama
    if not (oll.reachable and oll.url and is_remote(oll.url)):
        return None
    ok_, ms, err = ctx.p.remote_latency(oll.url)
    if not ok_ or ms is None:
        return warn(f"{oll.url}: {err or 'no answer'}", "Check the network / Tailscale connection.")
    if ms > 1500:
        return warn(f"{ms:.0f} ms to {oll.url}", "Very slow link: expect laggy replies; try a closer / direct connection.")
    return ok(f"{ms:.0f} ms to {oll.url}")


@check("ollama.host", "ollama", "Ollama address")
def host(ctx: Context):
    oll = ctx.ollama
    url = oll.url or ""
    cfg_host = str(ctx.section("ollama").get("host") or "").strip()
    env_host = (ctx.env.get("OLLAMA_HOST") or "").strip()
    src = "OLLAMA_HOST" if env_host else "ollama.host in config.yaml" if cfg_host else "the default"
    if not url:
        return info(f"not determined (from {src})")
    remote = is_remote(url)
    note = ""
    if remote and url.startswith("http://"):
        h = urlparse(url).hostname or ""
        private = False
        try:
            a = ipaddress.ip_address(h)
            private = a.is_private or a in ipaddress.ip_network("100.64.0.0/10")
        except ValueError:
            private = h.endswith((".ts.net", ".local", ".lan", ".internal"))
        if not private:
            return warn(f"{url} (remote, from {src}) uses plain http over what looks like a public address",
                        "Keep the server on a private network or VPN (Tailscale); plain http is readable by anyone on the path.")
        note = "; plain http on a private network"
    return info(f"{url} ({'remote' if remote else 'local'}, from {src}){note}")


@check("ollama.binary", "ollama", "Ollama program")
def binary(ctx: Context):
    oll = ctx.ollama
    if oll.url and is_remote(oll.url):
        return skip("the server is remote; the local program is not needed")
    path = ctx.p.which("ollama")
    if not path:
        return warn("the `ollama` command is not on PATH (the server answers, so it runs some other way)",
                    "Only matters for `ollama pull`; install from https://ollama.com or use its container.")
    rc, out = ctx.p.run([path, "--version"], timeout=3)
    line = (out.strip().splitlines() or ["?"])[-1] if rc == 0 else "version unknown"
    return ok(f"{path}; {line[:80]}")


@check("ollama.version", "ollama", "Ollama version", requires=("ollama",))
def version(ctx: Context):
    v = ctx.ollama.version
    if not v:
        return skip("the server did not report a version")
    if _vtuple(v) < MIN_OLLAMA:
        return warn(f"{v} is older than {'.'.join(map(str, MIN_OLLAMA))}: tool calls and some model families may not work",
                    "Update Ollama from https://ollama.com (or your package manager).")
    return ok(v)


@check("ollama.vulkan", "ollama", "Vulkan Ollama (second server)", timeout=6)
def vulkan(ctx: Context):
    env = ctx.env
    wanted = "AGENT_VULKAN_OLLAMA_URL" in env or bool(ctx.p.which("ollama-vk")) or ctx.p.exists(ctx.p.home() / ".local/bin/ollama-vk")
    if not wanted:
        return None
    v = ctx.p.vulkan_server()
    pattern = env.get("AGENT_VULKAN_MODELS")
    routing = "routing off (AGENT_VULKAN_MODELS is empty)" if pattern is not None and not pattern.strip() else "models are routed to it"
    if v["reachable"]:
        return ok(f"{v['url']} running, version {v['version'] or '?'}; {routing}")
    return info(f"{v['url']} is not running; it is started on demand by `ollama-vk` the first time a routed model is used")


@check("ollama.loaded", "ollama", "Models loaded right now", needs_ollama=True)
def loaded(ctx: Context):
    ps = ctx.p.ollama_ps()
    if not ps:
        return info("nothing loaded (models load on first use)")
    rows = []
    for m in ps:
        size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
        where = "all on GPU" if size and vram >= size else "CPU only" if not vram else f"{vram * 100 // max(size, 1)}% on GPU"
        rows.append(f"{m.get('name', '?')} ({size / 1024**3:.1f} GB, {where})")
    partial = [m for m in ps if 0 < int(m.get("size_vram") or 0) < int(m.get("size") or 0)]
    if partial:
        return warn("; ".join(rows), "A model split between GPU and CPU is slow: use a smaller model or a smaller context.")
    return info("; ".join(rows))

