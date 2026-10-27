"""Python packages and the command-line tools nightage likes to find."""

from __future__ import annotations

from agent.doctor.core import Context, check, fail, info, ok, warn

REQUIRED_PACKAGES = (
    ("rich", "rich", True), ("typer", "typer", True), ("prompt_toolkit", "prompt_toolkit", True),
    ("httpx", "httpx", True), ("yaml", "pyyaml", True), ("sqlite_vec", "sqlite-vec", False),
)


@check("packages", "packages", "Python packages", core=True)
def packages(ctx: Context):
    missing, optional = [], []
    for module, dist, required in REQUIRED_PACKAGES:
        if not ctx.p.package_available(module):
            (missing if required else optional).append(dist)
    if missing:
        return fail("missing: " + ", ".join(missing), f"Reinstall the project: pip install -e . (needs {', '.join(missing)}).")
    if optional:
        return warn("optional missing: " + ", ".join(optional),
                    f"Without it memory recall falls back to plain keyword search (fine on a phone): pip install {' '.join(optional)}.")
    return ok("all present")


@check("packages.versions", "packages", "Package versions")
def versions(ctx: Context):
    parts = []
    for _module, dist, _req in REQUIRED_PACKAGES:
        v = ctx.p.pkg_version(dist)
        if v:
            parts.append(f"{dist} {v}")
    return info(", ".join(parts) or "versions unknown (running from a source checkout without metadata)")


@check("tools", "packages", "Command-line tools")
def tools(ctx: Context):
    have = {n: bool(ctx.p.which(n)) for n in ("git", "rg", "uv", "ollama")}
    wants = {
        "git": "needed for `nightage update` on checkouts and for git plugins",
        "rg": "ripgrep: faster code search (the agent falls back to grep)",
        "uv": "optional installer / updater",
        "ollama": "the model server (not needed when ollama.host points at another machine)",
    }
    found = [n for n, v in have.items() if v]
    absent = [f"{n} ({wants[n]})" for n, v in have.items() if not v]
    if not have["git"]:
        return warn("found: " + (", ".join(found) or "none") + "; missing: " + "; ".join(absent),
                    "Install git with your package manager (needed for updates and git plugins).")
    return info("found: " + (", ".join(found) or "none") + ("; not found: " + "; ".join(absent) if absent else ""))
