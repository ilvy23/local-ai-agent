"""Which web addresses the research tools may touch.

The model decides which URL `web_fetch` opens and a hostile page can steer that decision, so a fetch
must never reach this machine or its networks: loopback (the Ollama servers on :11434/:11436, the agent's
own web app), the LAN, link-local/cloud-metadata addresses and the Tailscale range (100.64.0.0/10).
Every redirect hop is checked again (see fetch.fetch_one).

shortcut: the name is resolved here and again by httpx at connect time, so a DNS-rebinding host could
still answer differently the second time; pin the checked IP through a custom transport if that matters.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse

_TAILNET = ipaddress.ip_network("100.64.0.0/10")        # CGNAT range Tailscale hands out; is_private is False for it
_LOCAL_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".home.arpa", ".ts.net")


def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
        or ip.is_unspecified or (isinstance(ip, ipaddress.IPv4Address) and ip in _TAILNET)
    )


def blocked_reason(url: str, resolve: bool = True) -> str:
    """Why `url` must not be fetched, or "" when it is fine. Names that do not resolve are allowed
    here (the fetch itself will fail); `resolve=False` checks literal addresses and names only."""
    try:
        parts = urllib.parse.urlsplit(url.strip())
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return "not a valid address"
    if parts.scheme not in ("http", "https"):
        return f"scheme {parts.scheme or 'none'!r} is not allowed"
    if not host:
        return "no host"
    if parts.username or parts.password:
        return "address contains credentials"
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return "local network name"
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return "private or local address" if _blocked_ip(literal) else ""
    if "." not in host:
        return "single-label host name (local network)"
    if not resolve:
        return ""
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        return ""
    for info in infos:
        try:
            if _blocked_ip(ipaddress.ip_address(info[4][0])):
                return "resolves to a private or local address"
        except ValueError:
            continue
    return ""
