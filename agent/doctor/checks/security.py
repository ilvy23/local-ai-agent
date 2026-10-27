"""Security: the web interface's exposure, trusted addresses, secrets lying around in plain text. Never prints a secret."""

from __future__ import annotations

import ipaddress
import re

from agent.doctor.core import Context, check, info, ok, skip, warn

_SECRET_KEY = re.compile(r"(token|secret|password|passwd|api[_-]?key|apikey|private[_-]?key)", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"(sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def _walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


@check("security.web_auth", "security", "Web interface authentication")
def web_auth(ctx: Context):
    v = (ctx.env.get("AGENT_WEB_AUTH") or "on").strip().lower()
    if v in ("off", "0", "false", "no"):
        return warn("AGENT_WEB_AUTH=off: the web API needs no token", "Unset AGENT_WEB_AUTH unless this is a throw-away local test; anyone who can reach the port can use your agent.")
    return ok("token required for every web API call")


@check("security.web_bind", "security", "Web interface address")
def web_bind(ctx: Context):
    host = (ctx.env.get("AGENT_WEB_HOST") or "").strip()
    if not host:
        if ctx.p.port_open("0.0.0.0", 8420, 0.2) and not ctx.p.port_open("127.0.0.1", 8420, 0.2):
            return info("web interface is listening on a non-loopback address only")
        return ok("not configured: `nightage web` listens on Tailscale (if up) or 127.0.0.1 only")
    if host in ("0.0.0.0", "::", ""):
        off = (ctx.env.get("AGENT_WEB_AUTH") or "on").strip().lower() in ("off", "0", "false", "no")
        return warn(f"AGENT_WEB_HOST={host} listens on every network interface" + (" WITHOUT authentication" if off else ""),
                    "Bind to 127.0.0.1 or your Tailscale address instead.")
    return ok(f"AGENT_WEB_HOST={host}")


@check("security.trusted_ips", "security", "Trusted web addresses")
def trusted_ips(ctx: Context):
    from agent.doctor.checks.filesystem import config_dir

    p = config_dir(ctx) / "web-trusted-ips"
    txt = ctx.p.read_text(p, 100_000)
    if txt is None:
        return skip(f"no trusted-address file ({p.name})")
    entries = [ln.split("#")[0].strip() for ln in txt.splitlines() if ln.split("#")[0].strip()]
    wide = []
    for e in entries:
        try:
            net = ipaddress.ip_network(e, strict=False)
            if net.num_addresses > 2 ** 24 or e in ("0.0.0.0", "::"):
                wide.append(e)
        except ValueError:
            wide.append(f"{e} (not an address)")
    if wide:
        return warn(f"{len(entries)} trusted address(es); too broad or invalid: {', '.join(wide[:3])}",
                    "Trusted addresses skip the token. List single devices only (`nightage web-trust`).")
    return ok(f"{len(entries)} trusted address(es) skip the token") if entries else info("file is empty")


@check("security.secrets", "security", "Secrets in config.yaml")
def secrets(ctx: Context):
    _path, parsed, err = ctx.raw_config
    if err or not parsed:
        return None
    hits = []
    for path, val in _walk(parsed):
        if not isinstance(val, str) or not val.strip():
            continue
        if val.startswith(("$", "env:", "${")):
            continue
        if _SECRET_VALUE.search(val) or (_SECRET_KEY.search(path) and len(val) >= 8 and not val.startswith(("/", "~", "http"))):
            hits.append(path)
    if hits:
        return warn("looks like a secret stored in plain text at: " + ", ".join(hits[:5]) + " (values not shown)",
                    "Put secrets in environment variables instead; config.yaml is easy to share or commit by accident.")
    return ok("no secret-looking values in config.yaml")


@check("security.ollama_exposed", "security", "Ollama network exposure")
def ollama_exposed(ctx: Context):
    oh = (ctx.env.get("OLLAMA_HOST") or "").strip()
    if oh.startswith(("0.0.0.0", "http://0.0.0.0", "[::]", "::")):
        return warn("OLLAMA_HOST listens on every interface; Ollama has no login", "Bind it to 127.0.0.1 or your Tailscale address, or firewall the port.")
    return None
