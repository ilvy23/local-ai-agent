"""Disk: room for the database, and for the models Ollama keeps."""

from __future__ import annotations

from pathlib import Path

from agent.doctor.core import Context, check, fail, info, ok, warn

GB = 1024 ** 3


def _gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


@check("disk.data", "disk", "Space for your data")
def data(ctx: Context):
    data_dir, _db = ctx.p.data_paths(ctx.config)
    d = ctx.p.disk(data_dir)
    if d is None:
        return info(f"free space of {data_dir} is unknown")
    total, free = d
    if free < 200 * 1024 ** 2:
        return fail(f"only {free / 1024 ** 2:.0f} MB free where {data_dir} lives", "Free some disk space or move data.db_path elsewhere.")
    if free < 2 * GB:
        return warn(f"{_gb(free)} free of {_gb(total)} where {data_dir} lives", "Getting low: the database and logs grow over time.")
    return ok(f"{_gb(free)} free of {_gb(total)}")


def ollama_models_dir(ctx: Context) -> Path | None:
    env = ctx.env.get("OLLAMA_MODELS")
    cands = [Path(env).expanduser()] if env else []
    cands += [ctx.p.home() / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models"), Path("/var/lib/ollama/models"),
              Path("/var/lib/ollama")]
    for c in cands:
        if ctx.p.exists(c):
            return c
    return None


@check("disk.models", "disk", "Space for models", timeout=6)
def models(ctx: Context):
    from agent.doctor.checks.ollama import is_remote

    oll = ctx.ollama
    if oll.reachable and oll.url and is_remote(oll.url):
        return None
    d = ollama_models_dir(ctx)
    if d is None:
        return info("no local Ollama models folder found (looked at OLLAMA_MODELS, ~/.ollama/models and the system locations)")
    du = ctx.p.disk(d)
    size, complete = ctx.p.dir_size(d)
    held = f"{_gb(size)}{'+' if not complete else ''} of models in {d}"
    if du is None:
        return info(held)
    free = du[1]
    if free < 5 * GB:
        return warn(f"{held}; only {_gb(free)} free there", "Not enough room for the next model pull: remove one with `ollama rm <name>` or free space.")
    return ok(f"{held}; {_gb(free)} free")
