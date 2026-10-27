"""Ollama lifecycle: who started the server, and whether a crashed run left one behind (see agent/ollama_local.py)."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok, skip, warn
from agent import ollama_local


@check("ollama.origin", "ollama", "Ollama started by", requires=("ollama",))
def origin(ctx: Context):
    url = ctx.ollama.url or ""
    if url and not ollama_local.is_local(url):
        return skip("the server is on another machine; Nightage never starts or stops it")
    rec = ollama_local.read_record()
    if rec:
        return info("started by Nightage; it is stopped again when the last Nightage window closes")
    return info("an external server (your system service or your own `ollama serve`); Nightage never stops it, it only unloads models on exit")


@check("ollama.leftover", "ollama", "Leftover Ollama from a crashed run")
def leftover(ctx: Context):
    rec = ollama_local.leftover()
    if not rec:
        return ok("none")
    return warn(f"a server Nightage started (process {rec['pid']}) is still running although no Nightage window uses it; it probably ended with a kill that cannot be caught",
                "Run `nightage ollama stop` to free its memory (the next Nightage start also cleans it up).", command="nightage ollama stop")
