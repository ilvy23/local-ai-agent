"""A fake `ollama` program for tests: `ollama serve` runs a tiny HTTP server with the endpoints Nightage uses.

Environment: OLLAMA_HOST=host:port (where to listen), FAKE_OLLAMA_LOG=<file> (every request is appended as one JSON line),
FAKE_OLLAMA_LOADED=a,b (models that start loaded), FAKE_OLLAMA_PULL_BYTES (total size streamed by /api/pull)."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOADED = [m for m in os.environ.get("FAKE_OLLAMA_LOADED", "").split(",") if m]
LOG = os.environ.get("FAKE_OLLAMA_LOG", "")


def log(entry: dict) -> None:
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        log({"m": "GET", "path": self.path})
        if self.path == "/api/version":
            self._send({"version": "0.0.fake"})
        elif self.path == "/api/tags":
            names = [n for n in os.environ.get("FAKE_OLLAMA_MODELS", "fake:1b").split(",") if n]
            self._send({"models": [{"name": n, "size": 1_000_000} for n in names]})
        elif self.path == "/api/ps":
            self._send({"models": [{"name": n, "size_vram": 0} for n in LOADED]})
        else:
            self._send({}, 404)

    def do_DELETE(self):
        b = self._body()
        log({"m": "DELETE", "path": self.path, "body": b})
        self._send({}, 200 if b.get("model") == "fake:1b" else 404)

    def do_POST(self):
        b = self._body()
        log({"m": "POST", "path": self.path, "body": b})
        if self.path == "/api/show":
            self._send({"capabilities": ["completion", "tools"], "model_info": {}})
        elif self.path in ("/api/embed", "/api/embeddings"):
            self._send({"embeddings": [[0.1] * 8], "embedding": [0.1] * 8})
        elif self.path == "/api/chat":
            if os.environ.get("FAKE_CHAT_404"):
                self._send({"error": f"model '{b.get('model')}' not found"}, 404)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for ln in ({"message": {"role": "assistant", "content": "Hello from the fake."}, "done": False},
                           {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 5, "eval_duration": 1}):
                    self.wfile.write((json.dumps(ln) + "\n").encode())
        elif self.path == "/api/generate":
            if b.get("keep_alive") == 0 and b.get("model") in LOADED:
                LOADED.remove(b["model"])
            self._send({"done": True})
        elif self.path == "/api/pull":
            if b.get("model", "").startswith("missing"):
                self._send({"error": "pull model manifest: file does not exist"}, 404)
                return
            total = int(os.environ.get("FAKE_OLLAMA_PULL_BYTES", "1000000"))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            lines = [{"status": "pulling manifest"}]
            for i in range(0, 101, 20):
                lines.append({"status": "pulling abc123", "digest": "sha256:abc123", "total": total, "completed": total * i // 100})
            lines.append({"status": "pulling def456", "digest": "sha256:def456", "total": 100, "completed": 100})
            lines += [{"status": "verifying sha256 digest"}, {"status": "writing manifest"}, {"status": "success"}]
            for ln in lines:
                self.wfile.write((json.dumps(ln) + "\n").encode())
                self.wfile.flush()
                time.sleep(float(os.environ.get("FAKE_OLLAMA_PULL_DELAY", "0")))
        else:
            self._send({}, 404)


def serve() -> None:
    host, _, port = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").partition(":")
    srv = ThreadingHTTPServer((host or "127.0.0.1", int(port or 11434)), H)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    log({"m": "SERVE", "pid": os.getpid()})
    srv.serve_forever()


if __name__ == "__main__":
    if sys.argv[1:2] == ["serve"]:
        serve()
    elif sys.argv[1:2] == ["--version"]:
        print("ollama version is 0.0.fake")
    else:
        print("fake ollama: unsupported", sys.argv[1:], file=sys.stderr)
        sys.exit(2)
