"""Everything the doctor reads from the machine goes through a `Probes` object, so tests (and the home screen) can inject
fakes. The default probes use short timeouts and never raise. New in the modular doctor: generic helpers (`run`, `which`,
`read_text`, `disk`, `port_open`, `http_get`, `resolve`, ...) and the Ollama calls the deeper checks need."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OllamaInfo:
    reachable: bool = False
    version: str = ""
    models: dict[str, int] = field(default_factory=dict)   # name -> size in bytes (0 = unknown)
    error: str = ""
    url: str = ""


def model_installed(name: str, installed: Any) -> bool:
    """`bge-m3` matches `bge-m3:latest`; a tagged name must match exactly."""
    if not name:
        return False
    names = set(installed)
    return name in names or (":" not in name and f"{name}:latest" in names)


class Probes:
    """Everything that touches the machine. Subclass / replace methods in tests."""

    def __init__(self, timeout: float = 1.5) -> None:
        self.timeout = timeout

    # --- python & packages
    def python_version(self) -> tuple[int, int, int]:
        v = sys.version_info
        return v.major, v.minor, v.micro

    def package_available(self, module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    # --- platform
    def sysinfo(self) -> Any:
        from agent import sysinfo

        return sysinfo.current()

    def storage_shared(self) -> bool:
        """Termux: has `termux-setup-storage` been run (~/storage exists)?"""
        return (Path.home() / "storage").exists()

    def remote_latency(self, url: str) -> tuple[bool, float | None, str]:
        """(reachable, median round trip in ms of GET /api/version, error name) for a REMOTE Ollama; never raises."""
        import statistics
        import time

        try:
            import httpx

            times: list[float] = []
            with httpx.Client(base_url=url, timeout=httpx.Timeout(max(self.timeout, 3.0))) as c:
                for _ in range(3):
                    t0 = time.perf_counter()
                    c.get("/api/version").raise_for_status()
                    times.append((time.perf_counter() - t0) * 1000)
            return True, statistics.median(times), ""
        except Exception as exc:  # noqa: BLE001
            return False, None, type(exc).__name__

    # --- ollama
    def ollama(self) -> OllamaInfo:
        try:
            import httpx

            from agent.llm import current_base_url
        except Exception as exc:  # noqa: BLE001
            return OllamaInfo(error=str(exc))
        base_url = current_base_url()
        info = OllamaInfo(url=base_url)
        try:
            with httpx.Client(base_url=base_url, timeout=httpx.Timeout(self.timeout)) as c:
                r = c.get("/api/version")
                r.raise_for_status()
                info.version = str(r.json().get("version", ""))
                info.reachable = True
                t = c.get("/api/tags")
                t.raise_for_status()
                for m in t.json().get("models", []):
                    if m.get("name"):
                        info.models[m["name"]] = int(m.get("size") or 0)
        except Exception as exc:  # noqa: BLE001 - httpx errors, bad JSON, anything: just report
            if not info.reachable:
                info.error = type(exc).__name__
        return info

    # --- hardware
    def gpus(self) -> list[Any]:
        try:
            from agent import gpumode
            return list(gpumode.detect_gpus())
        except Exception:  # noqa: BLE001
            return []

    def gpu_mode(self, gpus: list[Any]) -> str | None:
        try:
            from agent import gpumode
            env = os.environ.get("AGENT_GPU_MODE", "").strip().lower()
            if env in (gpumode.SINGLE, gpumode.DUAL):
                return env
            return gpumode.auto_mode(gpus)
        except Exception:  # noqa: BLE001
            return None

    def ctx_budget(self, model: str, size: int, gpus: list[Any], config: dict | None) -> dict[str, Any] | None:
        try:
            from agent import ctxbudget
            hw = ctxbudget.hardware_summary(gpus)
            return ctxbudget.explain(model, size or None, hw, config=config)
        except Exception:  # noqa: BLE001
            return None

    # --- files
    def data_paths(self, config: dict | None) -> tuple[Path, Path]:
        """(data dir, db file)."""
        from agent.config import PROJECT_ROOT
        raw = ((config or {}).get("data") or {}).get("db_path") or "data/agent.db"
        db = Path(raw).expanduser()
        if not db.is_absolute():
            db = PROJECT_ROOT / db
        return db.parent, db

    def writable(self, path: Path) -> bool:
        """Can we create files in `path` (or in its nearest existing parent)?"""
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        return p.is_dir() and os.access(p, os.W_OK | os.X_OK)

    def config_file(self) -> tuple[Path, dict | None, str]:
        """(path, parsed mapping or None, error)."""
        from agent.config import DEFAULT_CONFIG_PATH
        path = DEFAULT_CONFIG_PATH
        if not path.exists():
            return path, None, "missing"
        try:
            import yaml
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001
            return path, None, f"unreadable: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}"
        if not isinstance(data, dict):
            return path, None, "not a mapping at the top level"
        return path, data, ""

    def last_session(self, config: dict | None) -> dict[str, Any] | None:
        """{'id', 'title', 'last_active_at'} of the newest session; read-only, never creates the database."""
        _, db = self.data_paths(config)
        if not db.exists():
            return None
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.5)
            try:
                row = con.execute(
                    "SELECT id, title, last_active_at FROM sessions ORDER BY id DESC LIMIT 1"
                ).fetchone()
            finally:
                con.close()
        except sqlite3.Error:
            return None
        return {"id": row[0], "title": row[1] or "", "last_active_at": row[2] or ""} if row else None

    # --- web & terminal
    def web_reachable(self, timeout: float) -> tuple[bool, str]:
        host = "html.duckduckgo.com"
        searx = os.environ.get("AGENT_SEARXNG_URL", "").strip()
        url = searx or f"https://{host}/"
        try:
            import httpx
            httpx.head(url, timeout=httpx.Timeout(timeout), follow_redirects=False)
            return True, searx or host
        except Exception as exc:  # noqa: BLE001
            return False, f"{searx or host}: {type(exc).__name__}"

    def terminal(self) -> dict[str, Any]:
        stream = sys.stdout
        try:
            tty = bool(stream.isatty())
        except (AttributeError, ValueError):
            tty = False
        enc = (getattr(stream, "encoding", None) or "").lower()
        env = os.environ
        return {
            "width": shutil.get_terminal_size((80, 24)).columns,
            "tty": tty,
            "color": tty and not (env.get("NO_COLOR") or env.get("TERM") == "dumb"),
            "unicode": "utf" in enc,
            "term": env.get("TERM", ""),
        }

    # ------------------------------------------------------------------ generic machine access (new)
    def environ(self) -> Mapping[str, str]:
        return os.environ

    def home(self) -> Path:
        return Path.home()

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def run(self, argv: list[str], timeout: float = 4.0) -> tuple[int, str]:
        """(return code, stdout+stderr). (-1, reason) when the program is missing, times out or fails to start."""
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
            return r.returncode, (r.stdout or "") + (r.stderr or "")
        except FileNotFoundError:
            return -1, "not installed"
        except subprocess.TimeoutExpired:
            return -1, f"timed out after {timeout:g} s"
        except (OSError, subprocess.SubprocessError) as exc:
            return -1, type(exc).__name__

    def read_text(self, path: str | Path, limit: int = 1_000_000) -> str | None:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read(limit)
        except OSError:
            return None

    def exists(self, path: str | Path) -> bool:
        try:
            return Path(path).exists()
        except OSError:
            return False

    def file_mode(self, path: str | Path) -> int | None:
        try:
            return Path(path).stat().st_mode & 0o777
        except OSError:
            return None

    def file_size(self, path: str | Path) -> int | None:
        try:
            return Path(path).stat().st_size
        except OSError:
            return None

    def can_access(self, path: str | Path, mode: int) -> bool:
        return os.access(path, mode)

    def disk(self, path: str | Path) -> tuple[int, int] | None:
        """(total bytes, free bytes) of the filesystem holding `path` (nearest existing parent); None if unknown."""
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        try:
            u = shutil.disk_usage(p)
            return u.total, u.free
        except OSError:
            return None

    def dir_size(self, path: str | Path, budget_s: float = 1.5) -> tuple[int, bool]:
        """(bytes under `path`, complete?). Stops after `budget_s` seconds so a huge tree cannot stall the doctor."""
        t0, total = time.monotonic(), 0
        stack = [str(path)]
        while stack:
            if time.monotonic() - t0 > budget_s:
                return total, False
            try:
                with os.scandir(stack.pop()) as it:
                    for e in it:
                        try:
                            if e.is_dir(follow_symlinks=False):
                                stack.append(e.path)
                            elif e.is_file(follow_symlinks=False):
                                total += e.stat(follow_symlinks=False).st_size
                        except OSError:
                            continue
            except OSError:
                continue
        return total, True

    def port_open(self, host: str, port: int, timeout: float = 0.4) -> bool:
        """True when something accepts TCP connections on host:port."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def resolve(self, host: str) -> tuple[bool, float | None, str]:
        t0 = time.perf_counter()
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            return True, (time.perf_counter() - t0) * 1000, ""
        except OSError as exc:
            return False, None, type(exc).__name__

    def https_get(self, url: str, timeout: float = 4.0) -> tuple[bool, float | None, str]:
        """(reachable, ms, error name). Any HTTP answer counts as reachable. A plain GET: nothing identifying is sent."""
        t0 = time.perf_counter()
        try:
            import httpx

            httpx.get(url, timeout=httpx.Timeout(timeout), follow_redirects=False)
            return True, (time.perf_counter() - t0) * 1000, ""
        except Exception as exc:  # noqa: BLE001
            return False, None, type(exc).__name__

    def is_wsl(self) -> bool:
        txt = self.read_text("/proc/sys/kernel/osrelease", 400) or ""
        return "microsoft" in txt.lower() or bool(self.environ().get("WSL_DISTRO_NAME"))

    def platform_info(self) -> dict[str, str]:
        return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                "version": platform.version(), "node": platform.node(), "python": platform.python_version(),
                "implementation": platform.python_implementation(), "executable": sys.executable,
                "venv": "yes" if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else "no"}

    def pkg_version(self, dist: str) -> str:
        try:
            from importlib.metadata import version

            return version(dist)
        except Exception:  # noqa: BLE001
            return ""

    def sqlite_check(self, db: Path, quick: bool = False) -> dict[str, Any]:
        """Read-only look at a database: integrity result, user_version, page stats. Never creates or changes anything."""
        out: dict[str, Any] = {"error": ""}
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
            try:
                out["integrity"] = con.execute("PRAGMA quick_check" if quick else "PRAGMA integrity_check").fetchone()[0]
                out["user_version"] = con.execute("PRAGMA user_version").fetchone()[0]
                out["journal_mode"] = con.execute("PRAGMA journal_mode").fetchone()[0]
                out["tables"] = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            finally:
                con.close()
        except sqlite3.Error as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out

    # ------------------------------------------------------------------ ollama API extras (new)
    def _client(self, timeout: float | None = None) -> Any:
        import httpx

        from agent.llm import current_base_url

        return httpx.Client(base_url=current_base_url(), timeout=httpx.Timeout(timeout or self.timeout))

    def ollama_get(self, path: str, timeout: float | None = None, base_url: str | None = None) -> dict[str, Any] | None:
        try:
            import httpx

            from agent.llm import current_base_url

            with httpx.Client(base_url=base_url or current_base_url(), timeout=httpx.Timeout(timeout or self.timeout)) as c:
                r = c.get(path)
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def ollama_post(self, path: str, body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any] | None:
        try:
            with self._client(timeout) as c:
                r = c.post(path, json=body)
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def ollama_show(self, model: str) -> dict[str, Any] | None:
        return self.ollama_post("/api/show", {"model": model}, timeout=5.0)

    def ollama_ps(self) -> list[dict[str, Any]]:
        d = self.ollama_get("/api/ps", 2.0)
        return list((d or {}).get("models") or [])

    def vulkan_server(self) -> dict[str, Any]:
        """The optional second (Vulkan) Ollama: {'url', 'reachable', 'version'}."""
        try:
            from agent import llm

            url = llm.vulkan_base_url()
        except Exception:  # noqa: BLE001
            url = "http://127.0.0.1:11436"
        d = self.ollama_get("/api/version", 1.5, base_url=url)
        return {"url": url, "reachable": d is not None, "version": str((d or {}).get("version", ""))}

    def ollama_generate(self, model: str, prompt: str, timeout: float = 90.0) -> dict[str, Any]:
        """Stream a short generation: {'ttft_ms', 'tokens', 'tps', 'total_ms', 'error'}."""
        import json as _json

        import httpx

        out: dict[str, Any] = {"error": ""}
        t0 = time.perf_counter()
        try:
            with self._client(timeout) as c, c.stream(
                    "POST", "/api/generate", json={"model": model, "prompt": prompt, "stream": True,
                                                   "options": {"num_predict": 48, "temperature": 0}}) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = _json.loads(line)
                    if "ttft_ms" not in out and d.get("response"):
                        out["ttft_ms"] = (time.perf_counter() - t0) * 1000
                    if d.get("done"):
                        n, dur = int(d.get("eval_count") or 0), int(d.get("eval_duration") or 0)
                        out["tokens"] = n
                        out["tps"] = (n / (dur / 1e9)) if n and dur else None
                        out["load_ms"] = int(d.get("load_duration") or 0) / 1e6
        except (httpx.HTTPError, ValueError) as exc:
            out["error"] = type(exc).__name__
        out["total_ms"] = (time.perf_counter() - t0) * 1000
        return out

    def ollama_embed(self, model: str, text: str = "hello", timeout: float = 60.0) -> dict[str, Any]:
        t0 = time.perf_counter()
        d = self.ollama_post("/api/embed", {"model": model, "input": text}, timeout=timeout)
        vec = ((d or {}).get("embeddings") or [[]])[0] if d else []
        return {"dim": len(vec), "ms": (time.perf_counter() - t0) * 1000, "error": "" if vec else "no embedding returned"}

    def ollama_tool_call(self, model: str, timeout: float = 90.0) -> dict[str, Any]:
        """Ask for a tool call; {'called': bool, 'name', 'args_ok': bool, 'prose': bool, 'ms'}."""
        tool = {"type": "function", "function": {
            "name": "get_weather", "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
        t0 = time.perf_counter()
        d = self.ollama_post("/api/chat", {"model": model, "stream": False, "tools": [tool],
                                           "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                                           "options": {"num_predict": 128, "temperature": 0}}, timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000
        if d is None:
            return {"error": "no answer", "ms": ms}
        msg = d.get("message") or {}
        calls = msg.get("tool_calls") or []
        if calls:
            fn = (calls[0] or {}).get("function") or {}
            args = fn.get("arguments")
            return {"called": True, "name": fn.get("name", ""), "args_ok": isinstance(args, dict) and "city" in args, "ms": ms}
        return {"called": False, "prose": bool(str(msg.get("content") or "").strip()), "ms": ms}
