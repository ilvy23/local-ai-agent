"""Bearer-token protection for the web API.

The server can write files and run commands (behind approvals), and it is meant to be reached from a phone over
Tailscale - so every /api request must carry the access token. The token lives in ~/.config/agent/web-token
(created on first use, mode 0600; print it with `agent web-token`). `AGENT_WEB_TOKEN` overrides it and
`AGENT_WEB_AUTH=off` disables the check (local development only).

Exempt: the static GUI files (the token entry screen has to load) and /api/files/<token>, whose URL is already an
unguessable per-file capability that <audio>/<video>/<img> tags cannot add a header to.
Failed attempts are rate limited per client address.

Trusted addresses (~/.config/agent/web-trusted-ips, one per line; manage with `agent web-trust`) skip the token.
The address is the TCP peer, never a header, so only a device that really holds that tailnet IP gets in (WireGuard
authenticates it); loopback and wildcard addresses can never be trusted.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

from agent import osport

from fastapi import Request
from fastapi.responses import JSONResponse

TOKEN_PATH = osport.config_home() / "agent" / "web-token"
TRUSTED_PATH = TOKEN_PATH.with_name("web-trusted-ips")
MAX_FAILURES = 10
WINDOW_S = 60.0
_failures: dict[str, deque[float]] = defaultdict(deque)


def auth_enabled() -> bool:
    return os.environ.get("AGENT_WEB_AUTH", "on").strip().lower() not in ("off", "0", "false", "no")


def get_token(path: Path | None = None) -> str:
    """The access token: env override, else the file (created with a fresh random token if missing)."""
    env = os.environ.get("AGENT_WEB_TOKEN", "").strip()
    if env:
        return env
    path = path or TOKEN_PATH
    try:
        token = path.read_text().strip()
        if token:
            return token
    except OSError:
        pass
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token


def trusted_ips(path: Path | None = None) -> set[str]:
    try:
        lines = (path or TRUSTED_PATH).read_text().splitlines()
    except OSError:
        return set()
    return {ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()}


def set_trusted(ip: str, on: bool = True, path: Path | None = None) -> set[str]:
    """Add/remove one trusted address; raises ValueError for anything that is not a usable remote IP."""
    addr = ipaddress.ip_address(ip.strip())
    if addr.is_loopback or addr.is_unspecified or addr.is_multicast:
        raise ValueError(f"{addr} cannot be trusted")
    path = path or TRUSTED_PATH
    ips = trusted_ips(path)
    ips.add(str(addr)) if on else ips.discard(str(addr))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("".join(f"{i}\n" for i in sorted(ips)))
    return ips


def _presented(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-agent-token", "").strip()


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limited(client: str, now: float) -> bool:
    q = _failures[client]
    while q and now - q[0] > WINDOW_S:
        q.popleft()
    return len(q) >= MAX_FAILURES


def needs_auth(path: str) -> bool:
    return path.startswith("/api/") and not path.startswith("/api/files/")


async def require_token(request: Request, call_next):
    if not auth_enabled() or not needs_auth(request.url.path):
        return await call_next(request)
    now = time.monotonic()
    client = _client(request)
    if client in trusted_ips():
        return await call_next(request)
    if _limited(client, now):
        return JSONResponse({"detail": "Too many failed attempts. Try again in a minute."}, status_code=429)
    given = _presented(request)
    if given and hmac.compare_digest(given.encode(), get_token().encode()):
        return await call_next(request)
    _failures[client].append(now)
    return JSONResponse({"detail": "Missing or invalid access token."}, status_code=401,
                        headers={"WWW-Authenticate": "Bearer"})
