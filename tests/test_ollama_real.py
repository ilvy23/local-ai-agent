"""One REAL test against the Ollama on this machine: Ctrl+C in `nightage chat` (in a pty) must leave nothing loaded and no server behind.
Skipped when there is no `ollama` binary or no small embedding model installed. Uses a temp NIGHTAGE_HOME and its own port: it never
touches a server you already run."""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import httpx
import pytest

pexpect = pytest.importorskip("pexpect")
ROOT = Path(__file__).resolve().parent.parent
TINY = ("all-minilm", "granite-embedding:30m")


def _installed() -> list[str]:
    try:
        return [m["name"] for m in httpx.get("http://localhost:11434/api/tags", timeout=2).json()["models"]]
    except Exception:  # noqa: BLE001
        return []


@pytest.mark.posix_only
@pytest.mark.parametrize("stop", [True, False])
@pytest.mark.skipif(not shutil.which("ollama"), reason="no ollama binary")
def test_ctrl_c_in_chat_unloads_models_and_stops_the_server_it_started(tmp_path, stop):
    names = _installed()
    tiny = next((n for n in names if n.split(":")[0] in {t.split(":")[0] for t in TINY}), None)
    if tiny is None or not any(not n.startswith(("all-minilm", "granite", "bge", "nomic", "qwen3-embedding", "embeddinggemma")) for n in names):
        pytest.skip("needs the tiny embedding model and one chat model installed on this machine")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(f"ollama:\n  host: 127.0.0.1:{port}\n  autostart: true\n  stop_on_exit: {str(stop).lower()}\n  unload_on_exit: true\n")
    env = {**os.environ, "NIGHTAGE_HOME": str(home), "XDG_STATE_HOME": str(tmp_path / "state"), "HOME": os.environ["HOME"],
           "PYTHONPATH": f"{ROOT}:{ROOT}/.venv-laptop/lib", "TERM": "xterm", "NO_COLOR": "1"}
    for k in ("NIGHTAGE_PROFILE", "NIGHTAGE_NO_AUTOSTART", "NIGHTAGE_OLLAMA_SESSION", "OLLAMA_HOST"):
        env.pop(k, None)
    url = f"http://127.0.0.1:{port}"
    child = pexpect.spawn(sys.executable, ["-m", "agent.main", "chat", "--plain"], env=env, cwd=str(ROOT), dimensions=(30, 90), timeout=60, encoding="utf-8")
    try:
        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                if httpx.get(url + "/api/version", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.3)
        else:
            pytest.fail("autostart did not bring the server up")
        r = httpx.post(url + "/api/embed", json={"model": tiny, "input": "hi", "keep_alive": "10m"}, timeout=60)
        r.raise_for_status()
        assert httpx.get(url + "/api/ps").json()["models"], "the tiny model should be loaded now"
        time.sleep(1.0)
        child.sendintr()
        time.sleep(0.3)
        child.expect(pexpect.EOF, timeout=30)
    finally:
        if child.isalive():
            child.terminate(force=True)
    if not stop:
        try:
            assert httpx.get(url + "/api/ps", timeout=2).json()["models"] == [], "the model must be unloaded even though the server stays"
        finally:
            import signal

            for f in (tmp_path / "state").rglob("server.json"):
                try:
                    os.killpg(json.loads(f.read_text())["pid"], signal.SIGKILL)
                except (OSError, ValueError, KeyError):
                    pass
        return
    deadline = time.time() + 10
    gone = False
    while time.time() < deadline:
        try:
            httpx.get(url + "/api/version", timeout=0.5)
            time.sleep(0.3)
        except httpx.HTTPError:
            gone = True
            break
    assert gone, "the server nightage started should be stopped after Ctrl+C"
