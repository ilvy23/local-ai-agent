"""Wait for the model server to come back (it was killed by the OOM killer, or the PC was suspended) instead of giving up."""
from __future__ import annotations

import time
from collections.abc import Callable

import httpx

DEFAULT_WAIT_S = 600
BACKOFF_S = (1, 2, 4, 8, 16, 30)                 # then 30 s for as long as it takes


def _http_probe(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


def wait_for_server(base_url: str, *, max_wait_s: float = DEFAULT_WAIT_S, sleep: Callable[[float], None] = time.sleep,
                    clock: Callable[[], float] = time.monotonic, probe: Callable[[str], bool] | None = None) -> bool:
    """True as soon as GET {base_url}/api/version answers 200; False after `max_wait_s` (<= 0: do not wait at all).
    The first probe is immediate, then 1, 2, 4, 8, 16, 30, 30 ... s apart (the last one cut short so the total stays within the limit)."""
    if max_wait_s <= 0:
        return False
    probe = probe or _http_probe
    start, n = clock(), 0
    while True:
        if probe(base_url):
            return True
        left = max_wait_s - (clock() - start)
        if left <= 0:
            return False
        sleep(min(BACKOFF_S[min(n, len(BACKOFF_S) - 1)], left))
        n += 1
