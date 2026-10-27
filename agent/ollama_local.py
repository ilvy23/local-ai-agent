"""Run and manage an Ollama server on THIS machine, so nobody has to know the `ollama` command (phones, laptops, PCs alike).

    status(config)      what is going on: address, answering?, version, binary, did Nightage start it, pid, log file
    start(config)       start `ollama serve` detached (own process group), wait until /api/version answers, remember the pid
    stop(config)        stop the server - only one Nightage started itself
    ensure(config)      start it when the configured host is THIS machine, nothing answers and a local binary exists
    pull(model, ...)    download a model through the streaming /api/pull API with parsed progress (resumable)
    session(config)     what the entry points (chat, code, research, web) wrap their run in: ensure + one cleanup on ANY exit

Rules that keep this safe:
  * never starts anything when `ollama.host` / OLLAMA_HOST points at another machine, and never when something already answers;
  * a server is "ours" only when the pid file written by `start()` names a live process whose command line says ollama; a user's
    own server or a system service (systemd, Windows service, an app) is never stopped by Nightage;
  * `ensure()` is off with `ollama.autostart: false` or NIGHTAGE_NO_AUTOSTART=1.

Cleanup on exit (`session()` / `cleanup()`), one idempotent function reached through every door:
  normal return, exceptions (try/finally), `atexit`, SIGTERM and SIGHUP (closed terminal, killed Termux session) and SIGINT
  (Ctrl+C: the previous handler still runs first, i.e. KeyboardInterrupt, so the program unwinds normally). It
    1. unloads every loaded model (POST /api/generate keep_alive:0 for each model of /api/ps) when `ollama.unload_on_exit` (default on),
    2. stops the server when `ollama.stop_on_exit` (default on) AND Nightage started it AND no other Nightage process is using it,
  inside a short deadline (CLEANUP_SECONDS) so quitting never hangs. A SECOND signal while the cleanup runs skips straight to a forced
  kill of the server we started and exits. The server runs in its own process group (not killed together with the terminal while
  the models are still being unloaded); the pid file stays until the server is really gone, so a later Nightage run finds a server a
  SIGKILL left behind (SIGKILL / power loss / the Android task killer cannot be caught: nothing can run then) and either reuses or
  stops it.

Termux / Android behaviour is UNVERIFIED on a real phone: it is written against documented behaviour and tested with fakes.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from agent import osport

CLEANUP_SECONDS = 3.0          # budget for unloading models; the server stop gets STOP_SECONDS on top
STOP_SECONDS = 2.0
START_SECONDS = 30.0           # a phone can take a while for the first start
ENV_NO_AUTOSTART = "NIGHTAGE_NO_AUTOSTART"
ENV_NO_SESSION = "NIGHTAGE_OLLAMA_SESSION"      # "off": begin() does nothing at all (the test suite sets it, so tests never touch a real server)
REGISTRY = "https://registry.ollama.ai"
BYTES_PER_PARAM_Q4 = 0.6       # rough size of a 4-bit quantised model per parameter (weights only); only used when nothing better is known


# ------------------------------------------------------------------------------------------------------------ paths


def state_dir() -> Path:
    """Where the pid file, the client markers and the server log live ($XDG_STATE_HOME/nightage/ollama)."""
    return osport.state_home() / "nightage" / "ollama"


def pid_path() -> Path:
    return state_dir() / "server.json"


def log_path() -> Path:
    return state_dir() / "server.log"


def clients_dir() -> Path:
    return state_dir() / "clients"


def _block(config: dict | None) -> dict:
    b = (config or {}).get("ollama") if isinstance(config, dict) else None
    return b if isinstance(b, dict) else {}


def base_url(config: dict | None = None) -> str:
    """The server this config talks to: OLLAMA_HOST env > ollama.host > the one the llm module resolved > localhost:11434."""
    from agent import llm

    if os.environ.get("OLLAMA_HOST", "").strip():
        return llm._default_base_url()
    host = str(_block(config).get("host") or "").strip()
    if host:
        return llm.normalize_host(host)
    return llm.current_base_url() if config is None else "http://localhost:11434"


def is_local(url: str) -> bool:
    from agent import llm

    return not llm.is_remote(url)


def find_binary() -> str | None:
    return shutil.which("ollama")


def probe(url: str, timeout: float = 1.5) -> str | None:
    """The server's version string when it answers /api/version, else None. Never raises."""
    try:
        r = httpx.get(url.rstrip("/") + "/api/version", timeout=timeout)
        r.raise_for_status()
        return str(r.json().get("version") or "?")
    except (httpx.HTTPError, ValueError, OSError, AttributeError):
        return None


# ------------------------------------------------------------------------------------------------------------ pid file


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:                                   # a zombie (exited, not yet reaped) is not alive
        with open(f"/proc/{pid}/stat") as fh:
            if fh.read().rsplit(")", 1)[-1].split()[0] == "Z":
                return False
    except (OSError, IndexError):
        pass
    return True


def _looks_like_ollama(pid: int) -> bool:
    """/proc/<pid>/cmdline names ollama; where /proc is missing (macOS, Windows) trust the pid file."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True
    return b"ollama" in cmd.lower()


def read_record() -> dict | None:
    """The pid file of a server Nightage started, only while that process is alive and is an ollama; a stale file is removed."""
    p = pid_path()
    try:
        rec = json.loads(p.read_text())
        pid = int(rec["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if _alive(pid) and _looks_like_ollama(pid):
        rec["pid"] = pid
        return rec
    with contextlib.suppress(OSError):
        p.unlink()
    return None


def _write_record(pid: int, url: str) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    pid_path().write_text(json.dumps({"pid": pid, "started_by": "nightage", "started_at": int(time.time()), "url": url}))


def _forget_record() -> None:
    with contextlib.suppress(OSError):
        pid_path().unlink()


def other_clients() -> list[int]:
    """Live Nightage processes (other than this one) that registered as users of the server; dead markers are removed."""
    live: list[int] = []
    d = clients_dir()
    if not d.is_dir():
        return live
    for f in d.iterdir():
        try:
            pid = int(f.name)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if _alive(pid):
            live.append(pid)
        else:
            with contextlib.suppress(OSError):
                f.unlink()
    return live


def leftover() -> dict | None:
    """The record of a server Nightage started when no Nightage process is using it any more (a run that was killed with SIGKILL), else None."""
    rec = read_record()
    if rec and not other_clients():
        return rec
    return None


def attach() -> None:
    with contextlib.suppress(OSError):
        clients_dir().mkdir(parents=True, exist_ok=True)
        (clients_dir() / str(os.getpid())).write_text("")


def detach() -> None:
    with contextlib.suppress(OSError):
        (clients_dir() / str(os.getpid())).unlink()


# ------------------------------------------------------------------------------------------------------------ status


@dataclass
class Status:
    url: str
    local: bool
    reachable: bool = False
    version: str = ""
    binary: str = ""
    ours: bool = False               # Nightage started the running server
    pid: int = 0
    log: str = ""
    models: int | None = None        # installed models when reachable

    def line(self) -> str:
        where = "on this machine" if self.local else f"on {self.url}"
        if self.reachable:
            who = ", started by Nightage" if self.ours else ""
            return f"Ollama {where}: running (version {self.version}{who})"
        return f"Ollama {where}: not running"

    def fix(self) -> str:
        if self.reachable:
            return ""
        if not self.local:
            return "Is the other machine on, and is its Ollama listening on the network? Check the address in `ollama.host`."
        if not self.binary:
            return "Ollama is not installed here. Termux: `pkg install ollama`. Other systems: https://ollama.com"
        return "Start it with: nightage ollama start"


def status(config: dict | None = None, url: str | None = None, *, timeout: float = 1.5) -> Status:
    url = url or base_url(config)
    st = Status(url=url, local=is_local(url), binary=find_binary() or "", log=str(log_path()))
    ver = probe(url, timeout)
    if ver is not None:
        st.reachable, st.version = True, ver
        try:
            st.models = len(httpx.get(url + "/api/tags", timeout=timeout).json().get("models", []))
        except (httpx.HTTPError, ValueError, OSError):
            pass
    rec = read_record()
    if rec and is_local(url):
        st.ours, st.pid = True, int(rec["pid"])
    return st


# ------------------------------------------------------------------------------------------------------------ start / stop


@dataclass
class Result:
    ok: bool
    action: str          # started | running | remote | disabled | no-binary | failed | stopped | not-ours | not-running
    message: str = ""
    pid: int = 0


def autostart_enabled(config: dict | None) -> bool:
    if os.environ.get(ENV_NO_AUTOSTART, "").strip() not in ("", "0"):
        return False
    return bool(_block(config).get("autostart", True))


def _serve_env(url: str) -> dict[str, str]:
    from urllib.parse import urlparse

    env = dict(os.environ)
    u = urlparse(url)
    if u.hostname:
        env["OLLAMA_HOST"] = f"{u.hostname}:{u.port or 11434}"
    return env


def _log_tail(n: int = 8) -> str:
    try:
        return "\n".join(log_path().read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def start(config: dict | None = None, *, wait: float = START_SECONDS, url: str | None = None) -> Result:
    """Start `ollama serve` in the background (own process group), wait until it answers, record pid + 'started by nightage'."""
    url = url or base_url(config)
    if not is_local(url):
        return Result(False, "remote", f"{url} is another machine; Nightage only starts an Ollama on this one.")
    ver = probe(url)
    if ver is not None:
        return Result(True, "running", f"already running (version {ver})", pid=(read_record() or {}).get("pid", 0))
    exe = find_binary()
    if not exe:
        return Result(False, "no-binary", "the `ollama` command was not found. Termux: `pkg install ollama`. Other systems: https://ollama.com")
    state_dir().mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {}
    if osport.is_windows():
        kw["creationflags"] = 0x00000008 | 0x00000200            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True                           # own process group: not hit by the terminal's Ctrl+C / hangup
    try:
        with open(log_path(), "ab") as logf:
            logf.write(f"\n--- nightage starts: {exe} serve ({time.strftime('%Y-%m-%d %H:%M:%S')})\n".encode())
            logf.flush()
            proc = _spawn([exe, "serve"], stdin=subprocess.DEVNULL, stdout=logf, stderr=logf, env=_serve_env(url), **kw)
    except OSError as exc:
        return Result(False, "failed", f"could not start {exe}: {exc}")
    _write_record(proc.pid, url)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if probe(url, 1.0) is not None:
            return Result(True, "started", f"started (pid {proc.pid}); log: {log_path()}", pid=proc.pid)
        if proc.poll() is not None:
            _forget_record()
            return Result(False, "failed", f"`ollama serve` exited with code {proc.returncode}. Last log lines:\n{_log_tail()}")
        time.sleep(0.25)
    return Result(False, "failed", f"started (pid {proc.pid}) but did not answer within {wait:.0f}s; log: {log_path()}", pid=proc.pid)


_spawn = subprocess.Popen        # tests replace this


def _kill_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if osport.is_windows():
            os.kill(pid, sig)
        else:
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                os.kill(pid, sig)


def stop(config: dict | None = None, *, wait: float = STOP_SECONDS) -> Result:
    """Stop the server Nightage started (never anyone else's). Waits up to `wait` seconds, then kills it."""
    rec = read_record()
    if rec is None:
        if probe(base_url(config)) is not None:
            return Result(False, "not-ours", "this Ollama was not started by Nightage, so Nightage leaves it running. "
                          "Stop it where you started it (e.g. `pkill -f 'ollama serve'`, or the system service).")
        return Result(True, "not-running", "no Nightage-started server is running")
    pid = int(rec["pid"])
    _kill_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    if _alive(pid):
        _kill_group(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        time.sleep(0.1)
    if _alive(pid):
        return Result(False, "failed", f"process {pid} did not exit")
    _forget_record()
    return Result(True, "stopped", f"stopped (pid {pid})", pid=pid)


def ensure(config: dict | None = None) -> Result:
    """Make sure a local server answers when the config wants one: see the module docstring for the rules."""
    url = base_url(config)
    if not is_local(url):
        return Result(True, "remote", f"using {url}")
    if probe(url) is not None:
        return Result(True, "running", "")
    if not autostart_enabled(config):
        return Result(False, "disabled", "Ollama is not running (ollama.autostart is off). Start it with: nightage ollama start")
    return start(config, url=url)


# ------------------------------------------------------------------------------------------------------------ cleanup on exit


@dataclass
class _Session:
    config: dict
    url: str
    cleaning: bool = False
    done: bool = False


_lock = threading.RLock()
_session: _Session | None = None
_prev_handlers: dict[int, Any] = {}
_hooks_installed = False


@dataclass
class CleanupResult:
    unloaded: list[str] = field(default_factory=list)
    stopped: bool = False
    skipped: str = ""


def unload_all(url: str, *, deadline: float) -> list[str]:
    """keep_alive:0 for every model /api/ps lists, stopping at the monotonic `deadline`. Never raises."""
    done: list[str] = []
    try:
        left = max(0.2, deadline - time.monotonic())
        with httpx.Client(base_url=url, timeout=httpx.Timeout(min(1.5, left))) as c:
            names = [m.get("name") or m.get("model") for m in c.get("/api/ps").json().get("models", [])]
            for name in [n for n in names if n]:
                if time.monotonic() >= deadline:
                    break
                try:
                    c.post("/api/generate", json={"model": name, "keep_alive": 0},
                           timeout=max(0.2, min(1.5, deadline - time.monotonic()))).raise_for_status()
                    done.append(name)
                except httpx.HTTPError:
                    continue
    except (httpx.HTTPError, ValueError, OSError, AttributeError):
        pass
    return done


def cleanup(reason: str = "exit") -> CleanupResult:
    """The single idempotent exit routine (see module docstring). Safe to call from anywhere, any number of times."""
    global _session
    with _lock:
        s = _session
        if s is None or s.done or s.cleaning:
            return CleanupResult(skipped="already done")
        s.cleaning = True
    res = CleanupResult()
    try:
        cfg = _block(s.config)
        if (s.config.get("coding") or {}).get("keep_model_loaded"):      # `nightage code --keep-model`: the next run wants it warm
            cfg = {**cfg, "unload_on_exit": False, "stop_on_exit": False}
        t0 = time.monotonic()
        if cfg.get("unload_on_exit", True):
            res.unloaded = unload_all(s.url, deadline=t0 + CLEANUP_SECONDS)
        detach()
        if cfg.get("stop_on_exit", True) and is_local(s.url) and read_record() is not None:
            if other_clients():
                res.skipped = "another Nightage is still using the server"
            else:
                res.stopped = stop(s.config, wait=STOP_SECONDS).action == "stopped"
    finally:
        with _lock:
            s.done, s.cleaning = True, False
            if _session is s:
                _session = None
    return res


def _force_kill() -> None:
    rec = read_record()
    if rec:
        _kill_group(int(rec["pid"]), getattr(signal, "SIGKILL", signal.SIGTERM))
        _forget_record()


def _on_signal(signum: int, frame: Any) -> None:
    s = _session
    if s is not None and s.cleaning:          # a second Ctrl+C / signal while cleaning: hard stop, no more waiting
        _force_kill()
        os._exit(128 + signum)
    if signum == signal.SIGINT:
        prev = _prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)               # Python's default raises KeyboardInterrupt: the entry point's finally cleans up
            return
        raise KeyboardInterrupt
    cleanup(f"signal {signum}")
    with contextlib.suppress(Exception):
        signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)              # die the way the signal wanted, now that the server is cleaned up


def _install_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    atexit.register(lambda: cleanup("atexit"))
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):      # SIGHUP does not exist on Windows
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)
            if prev is signal.SIG_IGN:
                continue                               # e.g. started with nohup: respect it
            _prev_handlers[int(sig)] = prev
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass


def _confirm(question: str, default: bool = True) -> bool | None:
    """A y/n question on the terminal; None when there is no keyboard (callers then print the command to run instead)."""
    try:
        if not sys.stdin.isatty():
            return None
        ans = input(f"? {question} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        return None
    return default if not ans else ans.startswith("y")


def _pull_with_bar(model: str, config: dict) -> bool:
    from agent import ollama_cli

    return ollama_cli.do_pull(model, config=config) == 0


def begin(config: dict | None, *, say: Callable[[str], None] | None = None, require_chat: bool = True) -> Result:
    """Autostart (if due), register this process as a user of the server and install the exit hooks. Idempotent per process."""
    global _session
    config = config or {}
    if os.environ.get(ENV_NO_SESSION, "").strip().lower() == "off":
        return Result(True, "disabled", "")
    res = ensure(config)
    url = base_url(config)
    if say is not None and res.action in ("started", "failed", "disabled", "no-binary"):
        say(("Ollama was not running: " if res.action == "started" else "Ollama: ") + res.message.splitlines()[0])
    with _lock:
        _session = _Session(config=config, url=url)
    if is_local(url):
        attach()
    _install_hooks()
    if res.action in ("running", "started", "remote") and "models" in config:
        from agent import modelpick

        try:
            modelpick.ensure_models(config, url, say=say or (lambda _t: None), confirm=_confirm,
                                    pull=lambda m: _pull_with_bar(m, config), require_chat=require_chat)
        except modelpick.NoModelAvailable:
            cleanup("no model")
            raise
        except Exception as exc:  # noqa: BLE001 - advice must never stop a run that could work
            if say is not None:
                say(f"(model check skipped: {type(exc).__name__})")
    return res


@contextlib.contextmanager
def session(config: dict | None, *, say: Callable[[str], None] | None = None, require_chat: bool = True) -> Iterator[Result]:
    """`with ollama_local.session(config):` around a whole run: autostart on the way in, cleanup on EVERY way out."""
    res = begin(config, say=say, require_chat=require_chat)
    try:
        yield res
    finally:
        cleanup("finally")


# ------------------------------------------------------------------------------------------------------------ pull


@dataclass
class Progress:
    status: str = ""
    total: int = 0                 # bytes over all layers seen so far
    completed: int = 0
    pct: float = 0.0               # never goes backwards
    speed: float = 0.0             # bytes/s, smoothed
    eta: float | None = None       # seconds, None = unknown
    done: bool = False


class PullTracker:
    """Folds the JSON lines of /api/pull into one monotonic percentage, a smoothed speed and an ETA."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.layers: dict[str, tuple[int, int]] = {}
        self.p = Progress()
        self._last: tuple[float, int] | None = None

    def feed(self, ev: dict) -> Progress:
        p = self.p
        p.status = str(ev.get("status") or p.status)
        digest, total = ev.get("digest"), ev.get("total")
        if digest and total:
            self.layers[str(digest)] = (int(total), int(ev.get("completed") or 0))
        p.total = sum(t for t, _ in self.layers.values())
        p.completed = sum(c for _, c in self.layers.values())
        if p.total:
            p.pct = max(p.pct, min(100.0, p.completed * 100.0 / p.total))
        now = self.clock()
        if self._last is None:
            self._last = (now, p.completed)
        elif now - self._last[0] >= 0.5 and p.completed >= self._last[1]:
            inst = (p.completed - self._last[1]) / (now - self._last[0])
            p.speed = inst if p.speed <= 0 else 0.7 * p.speed + 0.3 * inst
            self._last = (now, p.completed)
        p.eta = (p.total - p.completed) / p.speed if p.speed > 1 and p.total > p.completed else None
        if p.status == "success":
            p.done, p.pct, p.eta = True, 100.0, 0.0
        return replace(p)


class PullError(Exception):
    pass


def pull_events(model: str, url: str, *, client: httpx.Client | None = None) -> Iterator[dict]:
    """The raw JSON lines of POST /api/pull (stream). Raises PullError for an `error` line or an HTTP/connection failure."""
    own = client is None
    c = client or httpx.Client(base_url=url, timeout=httpx.Timeout(30.0, read=120.0))
    try:
        with c.stream("POST", "/api/pull", json={"model": model, "stream": True}) as r:
            if r.status_code >= 400:
                r.read()
                try:
                    detail = r.json().get("error", r.text)
                except ValueError:
                    detail = r.text
                raise PullError(f"Ollama answered {r.status_code}: {detail}")
            for line in r.iter_lines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise PullError(str(ev["error"]))
                yield ev
    except httpx.HTTPError as exc:
        raise PullError(f"connection to Ollama failed ({type(exc).__name__}). Is it running? Try: nightage ollama start") from exc
    finally:
        if own:
            c.close()


def pull(model: str, progress: Callable[[Progress], None] | None = None, *, config: dict | None = None, url: str | None = None,
         clock: Callable[[], float] = time.monotonic) -> Progress:
    """Download `model`, calling `progress(Progress)` for every update. Returns the final Progress (done=True). Raises PullError.
    KeyboardInterrupt passes through; run the same pull again to resume (Ollama keeps the finished parts)."""
    tracker = PullTracker(clock)
    for ev in pull_events(model, url or base_url(config)):
        p = tracker.feed(ev)
        if progress:
            progress(p)
    if not tracker.p.done:
        raise PullError("the download ended before Ollama reported success; run the same command again to continue")
    return tracker.p


# ------------------------------------------------------------------------------------------------------------ size / fit


def parse_params(tag: str) -> float | None:
    """Parameter count from a tag like `qwen3.5:4b`, `llama3:70b`, `qwen3.5:0.8b`, `x:270m`; None when the tag says nothing."""
    m = re.search(r"(?:^|[:\-_])(\d+(?:\.\d+)?)([bm])(?:$|[\-_.])", tag.lower())
    if not m:
        return None
    return float(m.group(1)) * (1e9 if m.group(2) == "b" else 1e6)


def manifest_size(model: str, *, timeout: float = 4.0) -> int | None:
    """Real download size from the public registry manifest (layers + config); None when offline or unknown."""
    name, _, tag = model.partition(":")
    path = name if "/" in name else f"library/{name}"
    try:
        r = httpx.get(f"{REGISTRY}/v2/{path}/manifests/{tag or 'latest'}", timeout=timeout,
                      headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        r.raise_for_status()
        d = r.json()
        return int(sum(int(x.get("size", 0)) for x in d.get("layers", [])) + int((d.get("config") or {}).get("size", 0))) or None
    except (httpx.HTTPError, ValueError, OSError, TypeError, AttributeError):
        return None


def estimate_size(model: str, *, fetch: Callable[[str], int | None] | None = manifest_size) -> tuple[int | None, str]:
    """(bytes, source) with source 'registry' | 'catalogue' | 'tag' | '' (unknown)."""
    from agent import embedmodels

    known = embedmodels.describe(model)
    if known.exact and known.download_mb:
        return known.download_mb * 1024 * 1024, "catalogue"
    if fetch is not None:
        size = fetch(model)
        if size:
            return size, "registry"
    params = parse_params(model)
    if params:
        return int(params * BYTES_PER_PARAM_Q4), "tag"
    return None, ""


@dataclass
class Fit:
    verdict: str          # ok | warn | fail | info | unknown
    text: str
    size: int | None = None
    source: str = ""


def disk_free_bytes(url_local: bool = True) -> int | None:
    root = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama")
    probe_dir = root
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    try:
        return shutil.disk_usage(probe_dir).free
    except OSError:
        return None


def check_fit(model: str, *, size: int | None = None, source: str = "", ram_free_mb: int | None = None, ram_total_mb: int | None = None,
              disk_free: int | None = None, gpus: list | None = None) -> Fit:
    """Will `model` fit THIS machine (memory via the doctor's `fit_verdict`, plus free disk)? Used for local servers only."""
    if size is None:
        size, source = estimate_size(model)
    if size is None:
        return Fit("unknown", "the size of this model is unknown, so nothing was checked", None, "")
    if ram_total_mb is None:
        from agent import sysinfo

        si = sysinfo.current()
        ram_free_mb, ram_total_mb = si.ram_free_mb, si.ram_total_mb
    from agent.doctor.checks.models import fit_verdict

    verdict, text = fit_verdict(size, gpus or [], int(ram_free_mb or 0), int(ram_total_mb or 0))
    if disk_free is not None and size * 1.05 > disk_free:
        return Fit("fail", f"needs about {size / 1024 ** 3:.1f} GB of disk but only {disk_free / 1024 ** 3:.1f} GB is free", size, source)
    return Fit(verdict, text, size, source)
