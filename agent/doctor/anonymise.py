"""Make a doctor report safe to paste into a public GitHub issue, then PROVE nothing sensitive is left.

`anonymise(text, identity)` replaces the user name, host name, home folder, IP and MAC addresses, e-mail addresses, URL
credentials and token-shaped strings. `leak_check(text, identity)` is the self-check: it looks for the same things again
(plus the exact values we know about) and returns what is still there, so the caller can refuse to print the report.
The ideas follow scripts/export_public.py's leak check (home paths, key-shaped strings, addresses)."""

from __future__ import annotations

import getpass
import ipaddress
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path

_IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
_IPV6_CANDIDATE = re.compile(r"(?<![\w:.])[0-9A-Fa-f:]{2,39}(?![\w:])")
_MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_URL_CREDS = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^/@\s:]+(?::[^/@\s]*)?@")
_TOKENS = re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{20,}|"
                     r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})")
_LONG_SECRET = re.compile(r"(?<![A-Za-z0-9/+_-])[A-Za-z0-9_-]{40,}(?![A-Za-z0-9/+_-])")
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_TAILNET = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.ts\.net\b", re.IGNORECASE)
_OTHER_HOMES = (
    (re.compile(r"/home/[A-Za-z0-9_.][A-Za-z0-9_.-]*"), "~"),
    (re.compile(r"/Users/[A-Za-z0-9_.][A-Za-z0-9_.-]*"), "~"),
    (re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_. -]+", re.IGNORECASE), "~"),
    (re.compile(r"/run/media/[A-Za-z0-9_.-]+"), "/run/media/<user>"),
)
_KEEP_IPS = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}
_GENERIC = frozenset({"nixos", "ubuntu", "debian", "fedora", "arch", "archlinux", "localhost", "android", "linux", "windows",
                      "macos", "termux", "raspberrypi", "localhost.localdomain", "docker", "container"})


@dataclass
class Identity:
    """What we know about whose machine this is (so it can be searched for literally)."""

    users: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    homes: list[str] = field(default_factory=list)

    @classmethod
    def current(cls) -> Identity:
        users, hosts, homes = [], [], []
        for fn in (getpass.getuser, lambda: os.environ.get("USER", ""), lambda: os.environ.get("LOGNAME", ""),
                   lambda: Path.home().name):
            try:
                v = fn()
                if v:
                    users.append(v)
            except Exception:  # noqa: BLE001
                pass
        for fn in (socket.gethostname, socket.getfqdn, lambda: os.uname().nodename if hasattr(os, "uname") else "",
                   lambda: os.environ.get("HOSTNAME", ""), lambda: os.environ.get("COMPUTERNAME", "")):
            try:
                v = fn()
                if v:
                    hosts.append(v)
                    hosts.append(v.split(".")[0])
            except Exception:  # noqa: BLE001
                pass
        try:
            homes.append(str(Path.home()))
        except Exception:  # noqa: BLE001
            pass
        return cls(_uniq(users), _uniq(hosts), _uniq(homes))


def _uniq(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        i = i.strip()
        if i and i.lower() not in seen and i.lower() not in _GENERIC and i.lower() not in ("root", "user", "nobody"):
            seen.add(i.lower())
            out.append(i)
    return sorted(out, key=len, reverse=True)


def _is_ipv6(s: str) -> bool:
    if s.count(":") < 2 or s in ("::", "::1"):
        return False
    try:
        ipaddress.IPv6Address(s)
        return True
    except ValueError:
        return False


def _hosts(ident: Identity) -> list[str]:
    return [h for h in ident.hosts if len(h) >= 2 and h.lower() not in _GENERIC]


def _word(value: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def anonymise(text: str, identity: Identity | None = None) -> str:
    ident = identity or Identity.current()
    out = text
    for h in sorted(ident.homes, key=len, reverse=True):
        if h and h not in ("/", "~"):
            out = out.replace(h, "~")
    for pat, repl in _OTHER_HOMES:
        out = pat.sub(repl, out)
    out = _URL_CREDS.sub(r"\1", out)
    out = _EMAIL.sub("<email>", out)
    out = _PRIVATE_KEY.sub("<private-key>", out)
    out = _TOKENS.sub("<redacted>", out)
    out = _TAILNET.sub("<tailnet-host>", out)
    for h in _hosts(ident):
        out = _word(h).sub("<host>", out)
    for u in ident.users:
        if len(u) >= 2:
            out = _word(u).sub("<user>", out)
    out = _MAC.sub("<mac>", out)
    out = _IPV6_CANDIDATE.sub(lambda m: "<ip6>" if _is_ipv6(m.group(0)) else m.group(0), out)

    def ip(m: re.Match[str]) -> str:
        v = m.group(1)
        if v in _KEEP_IPS:
            return v
        parts = v.split(".")
        if all(0 <= int(p) <= 255 for p in parts):
            return "<ip>"
        return v
    out = _IPV4.sub(ip, out)
    out = _LONG_SECRET.sub("<redacted>", out)
    return out


def leak_check(text: str, identity: Identity | None = None) -> list[str]:
    """What is still in `text` that should not be (descriptions only, never the values). Empty list = clean."""
    ident = identity or Identity.current()
    problems: list[str] = []
    for u in ident.users:
        if len(u) >= 2 and _word(u).search(text):
            problems.append("the user name")
            break
    for h in _hosts(ident):
        if _word(h).search(text):
            problems.append("the host name")
            break
    for h in ident.homes:
        if h and h not in ("/", "~") and h in text:
            problems.append("the home folder path")
            break
    for pat, _ in _OTHER_HOMES[:3]:
        if pat.search(text):
            problems.append("an absolute home path")
            break
    if any(m.group(1) not in _KEEP_IPS and all(0 <= int(p) <= 255 for p in m.group(1).split(".")) for m in _IPV4.finditer(text)):
        problems.append("an IPv4 address")
    if any(_is_ipv6(m.group(0)) for m in _IPV6_CANDIDATE.finditer(text)):
        problems.append("an IPv6 address")
    if _MAC.search(text):
        problems.append("a MAC address")
    if _EMAIL.search(text):
        problems.append("an e-mail address")
    if _URL_CREDS.search(text):
        problems.append("credentials inside a URL")
    if _TOKENS.search(text) or _PRIVATE_KEY.search(text) or _LONG_SECRET.search(text):
        problems.append("a token- or key-shaped string")
    if _TAILNET.search(text):
        problems.append("a tailnet host name")
    return problems
