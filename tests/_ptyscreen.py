"""Run `nightage chat` in a pty at a given width, answer the terminal's cursor-position requests, and return the final screen as text
(through the `pyte` terminal emulator). Shared by the narrow-screen tests. `pyte` is optional: tests skip when it is missing
(pip install pyte, or point NIGHTAGE_PYTE_PATH at a folder that has it)."""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKE = Path(__file__).resolve().parent / "_fake_ollama.py"
if os.environ.get("NIGHTAGE_PYTE_PATH"):
    sys.path.append(os.environ["NIGHTAGE_PYTE_PATH"])
pyte = pytest.importorskip("pyte")
pexpect = pytest.importorskip("pexpect")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Fake:
    """The fake Ollama server (tests/_fake_ollama.py) on a free port."""

    def __init__(self, tmp: Path, models: str = "qwen3.5:0.8b,all-minilm", chat_404: bool = False) -> None:
        self.port = free_port()
        env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{self.port}", "FAKE_OLLAMA_MODELS": models, "FAKE_OLLAMA_LOG": str(tmp / "fake.log")}
        env.pop("FAKE_CHAT_404", None)
        if chat_404:
            env["FAKE_CHAT_404"] = "1"
        self.proc = subprocess.Popen([sys.executable, str(FAKE), "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import httpx

        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{self.port}/api/version", timeout=0.3)
                break
            except httpx.HTTPError:
                time.sleep(0.05)

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait()


def run_chat(tmp: Path, port: int, width: int, script: list[tuple[str, float]], *, expect: str = "", rows: int = 30, config: str = "", extra_env: dict | None = None,
             args: tuple[str, ...] = ("chat", "--plain")) -> str:
    """Start the chat, send each (text, seconds to wait after) of `script`, then return the screen. The process is killed at the end."""
    home = tmp / "home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(f"ollama:\n  host: 127.0.0.1:{port}\n  autostart: false\n" + config)
    env = {**os.environ, "NIGHTAGE_HOME": str(home), "XDG_STATE_HOME": str(tmp / "state"), "TERM": "xterm-256color", "COLUMNS": str(width),
           "LINES": str(rows), "PYTHONPATH": f"{ROOT}:{ROOT}/.venv-laptop/lib", "NO_COLOR": "1", **(extra_env or {})}
    screen = pyte.Screen(width, rows)
    stream = pyte.ByteStream(screen)
    child = pexpect.spawn(sys.executable, ["-m", "agent.main", *args], env=env, cwd=str(ROOT), dimensions=(rows, width), timeout=5)

    def pump(seconds: float, until: str = "") -> None:
        end = time.time() + seconds
        while time.time() < end:
            if until and until in "\n".join(screen.display):
                time.sleep(0.4)
                return
            try:
                data = child.read_nonblocking(65536, 0.1)
            except pexpect.TIMEOUT:
                continue
            except pexpect.EOF:
                return
            stream.feed(data)
            if re.search(rb"\x1b\[6n", data):
                child.send(f"\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}R")

    try:
        pump(15.0, until="Enter send")
        for text, wait in script:
            child.send(text)
            pump(wait, until=expect)
    finally:
        child.terminate(force=True)
    return "\n".join(screen.display)
