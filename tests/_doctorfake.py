"""A fully fake machine for the doctor tests: nothing here touches the real system (files only under tmp_path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent import sysinfo
from agent.doctor.core import Context
from agent.doctor.probes import OllamaInfo, Probes

GB = 1024 ** 3


class G:
    def __init__(self, name="RX 7900", total_mb=24576, used_mb=1000, vendor="amd"):
        self.name, self.total_mb, self.used_mb, self.vendor = name, total_mb, used_mb, vendor

    @property
    def free_mb(self):
        return self.total_mb - self.used_mb


class FakeProbes(Probes):
    def __init__(self, tmp: Path, *, ollama: OllamaInfo | None = None, gpus: list | None = None, si: sysinfo.SysInfo | None = None,
                 env: dict[str, str] | None = None, bins: tuple[str, ...] = ("git",), runs: dict | None = None, files: dict | None = None,
                 ports: tuple[int, ...] = (), wsl: bool = False, raw_cfg: dict | None = None, cfg_err: str = "", tty: bool = True,
                 width: int = 100, web=(True, "html.duckduckgo.com"), dns=(True, 20.0, ""), https=(True, 80.0, ""), disk=(500 * GB, 400 * GB),
                 show: dict | None = None, ps: list | None = None, gen: dict | None = None, embed: dict | None = None, tool: dict | None = None):
        super().__init__()
        self.tmp = tmp
        self._o = ollama or OllamaInfo()
        self._g = list(gpus or [])
        self._si = si or sysinfo.SysInfo(ram_total_mb=16 * 1024, ram_free_mb=8192, cpu_count=8, os_family="linux", distro="debian",
                                         init_system="systemd")
        self.env = env or {}
        self.bins, self.runs, self.files, self.open_ports, self.wsl = set(bins), runs or {}, files or {}, set(ports), wsl
        self._cfg, self._cfg_err, self.tty, self.width = raw_cfg or {}, cfg_err, tty, width
        self._web, self._dns, self._https, self._disk = web, dns, https, disk
        self._show, self._ps, self._gen, self._embed, self._tool = show or {}, ps or [], gen or {}, embed or {}, tool or {}
        (tmp / "home").mkdir(exist_ok=True)
        (tmp / "data").mkdir(exist_ok=True)

    # --- machine
    def sysinfo(self): return self._si
    def gpus(self): return self._g
    def gpu_mode(self, gpus): return "single" if gpus else None
    def ollama(self): return self._o
    def environ(self): return self.env
    def home(self): return self.tmp / "home"
    def which(self, name): return f"/usr/bin/{name}" if name in self.bins else None
    def run(self, argv, timeout=4.0): return self.runs.get(tuple(argv), (-1, "not installed"))
    def is_wsl(self): return self.wsl
    def python_version(self): return (3, 12, 1)
    def package_available(self, module): return True
    def pkg_version(self, dist): return "1.0"
    def platform_info(self):
        return {"system": "Linux", "release": "6.1.0", "machine": "x86_64", "version": "#1", "node": "myhost", "python": "3.12.1",
                "implementation": "CPython", "executable": "/usr/bin/python3", "venv": "yes"}

    def _real(self, path) -> bool:
        return str(path).startswith(str(self.tmp))

    def read_text(self, path, limit=1_000_000):
        if str(path) in self.files:
            return self.files[str(path)]
        return super().read_text(path, limit) if self._real(path) else None

    def exists(self, path):
        return str(path) in self.files or (self._real(path) and super().exists(path))

    def file_mode(self, path): return super().file_mode(path) if self._real(path) else None
    def file_size(self, path): return super().file_size(path) if self._real(path) else None
    def can_access(self, path, mode): return True
    def disk(self, path): return self._disk
    def dir_size(self, path, budget_s=1.5): return (3 * GB, True)
    def port_open(self, host, port, timeout=0.4): return port in self.open_ports
    def resolve(self, host): return self._dns
    def https_get(self, url, timeout=4.0): return self._https
    def web_reachable(self, timeout): return self._web
    def data_paths(self, config): return self.tmp / "data", self.tmp / "data" / "agent.db"
    def writable(self, path): return Probes.writable(self, path) if self._real(path) else True
    def config_file(self): return self.tmp / "config.yaml", (None if self._cfg_err else self._cfg), self._cfg_err
    def storage_shared(self): return True
    def remote_latency(self, url): return True, 40.0, ""
    def ctx_budget(self, model, size, gpus, config): return {"override": None, "computed": 32768, "final": 32768, "source": "vram", "ceiling": 95000}
    def terminal(self): return {"width": self.width, "tty": self.tty, "color": self.tty, "unicode": True, "term": "xterm"}
    def ollama_show(self, model): return self._show.get(model)
    def ollama_ps(self): return self._ps
    def vulkan_server(self): return {"url": "http://127.0.0.1:11436", "reachable": False, "version": ""}
    def ollama_generate(self, model, prompt, timeout=90.0): return self._gen or {"error": "", "ttft_ms": 800.0, "tokens": 40, "tps": 12.5, "load_ms": 100.0}
    def ollama_embed(self, model, text="hello", timeout=60.0): return self._embed or {"dim": 1024, "ms": 50.0, "error": ""}
    def ollama_embed_vectors(self, model, texts, timeout=30.0):
        if self._embed.get("error"):
            raise RuntimeError(self._embed["error"])
        import hashlib
        import math
        out = []
        for t in texts:
            v = [0.0] * 32
            for w in "".join(c if c.isalpha() else " " for c in t.lower()).split():
                if w not in {"what", "is", "my", "has", "a", "in"}:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
    def ollama_tool_call(self, model, timeout=90.0): return self._tool or {"called": True, "name": "get_weather", "args_ok": True, "ms": 900.0}


CFG = {"models": {"chat": "qwen3.5:9b", "coding": "qwen3-coder:30b", "embed": "bge-m3"}}
UP = OllamaInfo(True, "0.6.1", {"qwen3.5:9b": 6 * GB, "qwen3-coder:30b": 18 * GB, "bge-m3:latest": GB}, url="http://localhost:11434")


def ctx_for(tmp: Path, cfg: dict | None = None, **kw: Any) -> Context:
    deep, no_web, verbose = kw.pop("deep", False), kw.pop("no_web", False), kw.pop("verbose", False)
    return Context(CFG if cfg is None else cfg, FakeProbes(tmp, **kw), deep=deep, no_web=no_web, verbose=verbose)
