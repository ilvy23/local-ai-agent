"""Fetch several pages at once, politely and with hard limits.

The old reader fetched one page after the other (up to 15 s each, so five slow sites meant over a
minute of silence). This runs them in parallel with a short per-page timeout, a size cap, a total
deadline, and skips what can't be read as text (PDF, images) or sits behind a login wall.
"""

from __future__ import annotations

import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import httpx

from agent.research.netguard import blocked_reason

MAX_REDIRECTS = 5
_WALL_STATUS = (401, 402, 403, 429)     # login / paywall / bot wall: reported as "blocked", not as a plain failure
MAX_BYTES = 1_500_000
PAGE_TIMEOUT = 8.0
DEADLINE = 14.0
WORKERS = 6

_UNREADABLE_HOSTS = ("facebook.com", "instagram.com", "x.com", "twitter.com", "linkedin.com", "tiktok.com",
                     "pinterest.", "reddit.com/login", "youtube.com", "youtu.be")
_TEXT_TYPES = ("text/html", "application/xhtml", "text/plain")


@dataclass
class Fetched:
    url: str
    html: str = ""
    status: str = "ok"          # ok | skipped | failed | timeout
    note: str = ""              # why skipped/failed
    seconds: float = 0.0


def _skip_reason(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    path = urllib.parse.urlparse(url).path.lower()
    if any(h in host + path for h in _UNREADABLE_HOSTS):
        return "login or video site"
    if path.endswith((".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mp3", ".exe", ".dmg")):
        return path.rsplit(".", 1)[-1] + " file"
    return ""


def fetch_one(url: str, headers: dict[str, str], timeout: float = PAGE_TIMEOUT, max_bytes: int = MAX_BYTES) -> Fetched:
    started = time.monotonic()
    reason = _skip_reason(url)
    if reason:
        return Fetched(url, status="skipped", note=reason)
    try:
        current = url
        for _hop in range(MAX_REDIRECTS + 1):
            denied = blocked_reason(current)
            if denied:                                            # every hop: a public page may redirect to localhost
                return Fetched(url, status="blocked", note=denied, seconds=time.monotonic() - started)
            with httpx.stream("GET", current, headers=headers, timeout=timeout, follow_redirects=False) as r:
                if r.is_redirect and r.headers.get("location"):
                    current = urllib.parse.urljoin(current, r.headers["location"])
                    continue
                if r.status_code in _WALL_STATUS:
                    return Fetched(url, status="blocked", note=f"http {r.status_code}", seconds=time.monotonic() - started)
                if r.status_code >= 400:
                    return Fetched(url, status="failed", note=f"http {r.status_code}", seconds=time.monotonic() - started)
                ctype = r.headers.get("content-type", "").lower()
                if ctype and not any(t in ctype for t in _TEXT_TYPES):
                    return Fetched(url, status="skipped", note=ctype.split(";")[0] or "not text",
                                   seconds=time.monotonic() - started)
                body = bytearray()
                for chunk in r.iter_bytes():
                    body.extend(chunk)
                    if len(body) >= max_bytes:
                        break
                encoding = r.encoding or "utf-8"
            text = bytes(body).decode(encoding, errors="replace")
            return Fetched(url, html=text, seconds=time.monotonic() - started)
        return Fetched(url, status="failed", note="too many redirects", seconds=time.monotonic() - started)
    except httpx.TimeoutException:
        return Fetched(url, status="timeout", note="timeout", seconds=time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 - one dead site must never fail the search
        return Fetched(url, status="failed", note=type(exc).__name__, seconds=time.monotonic() - started)


def fetch_many(
    urls: list[str], headers: dict[str, str], deadline: float = DEADLINE, workers: int = WORKERS,
    fetch: Callable[[str], Fetched] | None = None,
) -> dict[str, Fetched]:
    """Fetch all `urls` in parallel; whatever has not finished by `deadline` seconds counts as timed out."""
    one = fetch or (lambda u: fetch_one(u, headers))
    results: dict[str, Fetched] = {}
    if not urls:
        return results
    pool = ThreadPoolExecutor(max_workers=max(1, min(workers, len(urls))))
    futures = {pool.submit(one, u): u for u in urls}
    started = time.monotonic()
    try:
        for fut in as_completed(futures, timeout=deadline):
            results[futures[fut]] = fut.result()
    except Exception:  # noqa: BLE001 - as_completed timeout: keep what we have
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    for u in urls:
        results.setdefault(u, Fetched(u, status="timeout", note="timeout", seconds=time.monotonic() - started))
    return results
