"""Keep the PC awake for the WHOLE night. Each coding attempt already holds an inhibitor while it runs, but between two attempts (tests, git, a health wait)
an idle timer could suspend the machine. One `systemd-inhibit` process for the entire run closes that gap; it ends with the night. It runs `sleep <max_hours>`, never `sleep infinity`: when the governor is killed hard (SIGKILL) nothing stops the child,
and an unbounded one would hold the sleep lock for ever (seen: two such leftovers)."""
from __future__ import annotations

import shutil
import subprocess
from typing import Callable


class Inhibitor:
    def __init__(self, why: str = "nightage overnight coding run", popen: Callable[..., subprocess.Popen] = subprocess.Popen, which: Callable[[str], str | None] = shutil.which,
                 max_hours: float = 14.0) -> None:
        self.why, self.popen, self.which = why, popen, which
        self.max_s = int(max_hours * 3600)             # a bounded sleep: a governor that is killed hard cannot leave the PC unable to sleep forever
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Inhibitor":
        exe = self.which("systemd-inhibit")
        if exe:
            try:
                self.proc = self.popen([exe, "--what=sleep:idle", "--who=nightage", f"--why={self.why}", "--mode=block", "sleep", str(self.max_s)],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            except OSError:
                self.proc = None
        return self

    @property
    def active(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            self.proc = None
