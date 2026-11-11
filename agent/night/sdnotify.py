"""Tell systemd that the night is alive: READY=1 at the start and WATCHDOG=1 while it makes progress (the unit has `Type=notify` and `WatchdogSec`). A governor that HANGS (not crashes)
stops getting heartbeats and systemd restarts it; one that merely waits for a long test or a model answer is fine, because the watchdog thread only beats while the main loop has
shown a sign of life within `patience_s` (`alive()` is called at every poll of an attempt, every gate wait and after every test run). Without systemd (no NOTIFY_SOCKET) it does nothing."""
from __future__ import annotations

import os
import socket
import threading
import time
from typing import Callable


def send(message: str, path: str | None = None) -> bool:
    path = path if path is not None else os.environ.get("NOTIFY_SOCKET", "")
    if not path:
        return False
    addr = "\0" + path[1:] if path.startswith("@") else path
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.settimeout(2)
            s.connect(addr)
            s.sendall(message.encode())
        return True
    except OSError:
        return False


class Heartbeat:
    def __init__(self, every_s: float = 60.0, patience_s: float = 3000.0, clock: Callable[[], float] = time.monotonic, sender: Callable[[str], bool] = send) -> None:
        self.every_s, self.patience_s, self.clock, self.sender = every_s, patience_s, clock, sender
        self.last = clock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def alive(self, status: str = "") -> None:
        """The main loop calls this whenever it proves it is not stuck."""
        self.last = self.clock()
        if status:
            self.sender("STATUS=" + status[:120])

    def beat(self) -> bool:
        """One heartbeat if the main loop was alive recently; False (and silence) when it has not been for `patience_s`."""
        if self.clock() - self.last > self.patience_s:
            return False
        return self.sender("WATCHDOG=1")

    def __enter__(self) -> "Heartbeat":
        self.sender("READY=1")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self.beat()

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self.sender("STOPPING=1")
