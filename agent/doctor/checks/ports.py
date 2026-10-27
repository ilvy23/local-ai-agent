"""Ports: are the ones nightage uses free, or taken by what you expect?"""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, warn


@check("ports.ollama", "ports", "Port 11434 (Ollama)", requires=())
def ollama_port(ctx: Context):
    from agent.doctor.checks.ollama import is_remote

    oll = ctx.ollama
    if oll.url and is_remote(oll.url):
        return None
    busy = ctx.p.port_open("127.0.0.1", 11434)
    if oll.reachable and busy:
        return ok("in use by Ollama (as expected)")
    if busy:
        return warn("something is listening on 11434, but it does not answer like Ollama",
                    "Another program holds Ollama's port: stop it, or set OLLAMA_HOST / ollama.host to another port.")
    return info("free (Ollama is not running)")


@check("ports.vulkan", "ports", "Port 11436 (Vulkan Ollama)")
def vulkan_port(ctx: Context):
    env = ctx.env
    if "AGENT_VULKAN_OLLAMA_URL" not in env and not ctx.p.which("ollama-vk"):
        return None
    busy = ctx.p.port_open("127.0.0.1", 11436)
    return ok("in use (the Vulkan server is up)") if busy else info("free (the Vulkan server starts on demand)")


@check("ports.web", "ports", "Port 8420 (web interface)")
def web_port(ctx: Context):
    if ctx.p.port_open("127.0.0.1", 8420):
        return info("in use: `nightage web` (or something else) is already listening there; a second one would not start")
    return ok("free")
