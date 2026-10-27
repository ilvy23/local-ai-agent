"""Finding out that a newer nightage (or plugin) exists, and the one calm line that says so.

What it does, and what it does not:
  * `check_for_updates(config)` asks GitHub's public releases API (a plain GET, 3 s timeout) for the latest release of the
    project and, for a git checkout, asks the remote what its branch points at (`git ls-remote`: nothing is changed on disk).
    For every plugin installed from a git URL it does the same `git ls-remote`. Nothing identifying is sent; GitHub sees your IP
    address like any web request. Local-path plugins have no update source.
  * The result is cached in the state folder (default 24 h, `update.check_interval_hours`). Offline or failing checks are silent
    and are retried after an hour. `start_background_check` runs it in a daemon thread: startup never waits.
  * Opt out with `update.check: false` in config.yaml or `NIGHTAGE_NO_UPDATE_CHECK=1`. The explicit commands (`nightage update`,
    `nightage plugin update`) still work, because you asked for them.
  * Nothing is downloaded or installed here. Installing is `agent/update_cli.py` (framework) and `agent/plugin_update.py` (plugins).

Everything that touches the outside world is injectable (`http`, `run`, `now`, `install`), so the tests use fakes only.
Guide: docs/UPDATING.md.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent import osport
from typing import Any

from agent import brand

CACHE_SCHEMA = 2
DEFAULT_INTERVAL_H = 24.0
OFFLINE_RETRY_S = 3600.0
HTTP_TIMEOUT = 3.0
GIT_TIMEOUT = 8.0
ENV_OPT_OUT = "NIGHTAGE_NO_UPDATE_CHECK"
API_BASE = "https://api.github.com"
DIST_NAME = "agent"          # the distribution name in pyproject.toml (the command is `nightage`)

HttpGet = Callable[[str, float], "tuple[int, Any]"]
Runner = Callable[..., "tuple[int, str]"]


# ----------------------------------------------------------------------------------------------- versions


_VER = re.compile(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.\d+)*\s*(?:[-._+]?(dev|alpha|a|beta|b|rc|c|pre|preview)[-._]?(\d+)?)?"
                  r"(?:[-+.].*)?$", re.IGNORECASE)
_STAGE = {"dev": 0, "alpha": 1, "a": 1, "beta": 2, "b": 2, "pre": 3, "preview": 3, "rc": 4, "c": 4}


def parse_version(v: str) -> tuple[int, int, int, int, int] | None:
    """(major, minor, patch, stage, number); a release has stage 9, so 0.2.0-rc1 < 0.2.0 and 0.2.0 < 0.2.1-dev.
    None when `v` is not a version at all."""
    m = _VER.match(v or "")
    if not m:
        return None
    major, minor, patch, stage, num = m.groups()
    return (int(major), int(minor or 0), int(patch or 0), _STAGE[stage.lower()] if stage else 9, int(num or 0))


def is_prerelease(v: str) -> bool:
    p = parse_version(v)
    return bool(p and p[3] != 9)


def compare(a: str, b: str) -> int:
    """-1, 0, 1 like cmp(a, b); unparseable versions compare equal (we never claim an update we cannot prove)."""
    pa, pb = parse_version(a), parse_version(b)
    if pa is None or pb is None:
        return 0
    return (pa > pb) - (pa < pb)


def is_newer(candidate: str, current: str) -> bool:
    return compare(candidate, current) > 0


def project_root() -> Path:
    from agent.config import PROJECT_ROOT

    return PROJECT_ROOT


def _pyproject_version(root: Path) -> str:
    try:
        import tomllib

        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        proj = data.get("project") or {}
        if proj.get("name") == DIST_NAME:
            return str(proj.get("version") or "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def current_version(root: Path | None = None) -> str:
    """The version running now. A checkout's pyproject.toml wins over the installed metadata, which goes stale after a pull."""
    v = _pyproject_version(root or project_root())
    if v:
        return v
    try:
        from importlib.metadata import version

        return version(DIST_NAME)
    except Exception:  # noqa: BLE001
        return "0.1.0"


# ----------------------------------------------------------------------------------------------- settings


def state_dir() -> Path:
    base = str(osport.state_home())
    return Path(base) / "nightage"


def cache_path() -> Path:
    return state_dir() / "update-check.json"


def _section(config: dict | None) -> dict:
    v = (config or {}).get("update") if isinstance(config, dict) else None
    return v if isinstance(v, dict) else {}


def enabled(config: dict | None = None, env: dict[str, str] | None = None) -> bool:
    """Automatic checks are on unless `update.check: false` or NIGHTAGE_NO_UPDATE_CHECK is set (to anything but 0/false/no)."""
    e = os.environ if env is None else env
    if (e.get(ENV_OPT_OUT) or "").strip().lower() not in ("", "0", "false", "no"):
        return False
    return _section(config).get("check", True) is not False


def interval_s(config: dict | None = None) -> float:
    try:
        h = float(_section(config).get("check_interval_hours", DEFAULT_INTERVAL_H))
    except (TypeError, ValueError):
        h = DEFAULT_INTERVAL_H
    return max(0.0, h) * 3600.0


_SLUG = re.compile(r"(?:github\.com[:/])([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
_PLAIN_SLUG = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")


def parse_slug(text: str) -> tuple[str, str] | None:
    """`owner/repo` from a GitHub URL (https or ssh) or from `owner/repo` itself; None for anything else."""
    t = (text or "").strip()
    m = _SLUG.search(t) or (_PLAIN_SLUG.match(t) if "://" not in t and "@" not in t else None)
    return (m.group(1), m.group(2)) if m else None


# ----------------------------------------------------------------------------------------------- install kind


@dataclass
class InstallInfo:
    kind: str = "unknown"          # git | editable | uv-tool | pip | unknown
    root: str = ""                 # the checkout (git / editable)
    branch: str = ""
    remote_url: str = ""
    has_remote: bool = False
    editable: bool = False
    vcs_url: str = ""              # pip install of git+URL: where from
    vcs_commit: str = ""

    def describe(self) -> str:
        return {"git": "git checkout" + (f" ({self.branch})" if self.branch else "") + ("" if self.has_remote else ", no remote"),
                "editable": "editable install (not a git checkout)", "uv-tool": "uv tool install", "pip": "pip install",
                }.get(self.kind, "unknown install")


def default_runner(argv: list[str], cwd: Path | str | None = None, timeout: float = GIT_TIMEOUT) -> tuple[int, str]:
    """Run a program, never prompting for credentials. (return code, stdout+stderr); -1 when it could not run."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "LC_ALL": "C"}
    try:
        r = subprocess.run(argv, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout, check=False, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return -1, f"{argv[0]} is not installed"
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout:g} s"
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, type(exc).__name__


def _dist_direct_url() -> dict[str, Any]:
    try:
        from importlib.metadata import distribution

        txt = distribution(DIST_NAME).read_text("direct_url.json")
        return json.loads(txt) if txt else {}
    except Exception:  # noqa: BLE001
        return {}


def detect_install(*, root: Path | None = None, run: Runner | None = None, prefix: str | None = None,
                   direct_url: dict[str, Any] | None = None) -> InstallInfo:
    """Which kind of install is running (see docs/UPDATING.md). Pure reads; `run` is for the tests."""
    run = run or default_runner
    root = Path(root) if root else project_root()
    du = _dist_direct_url() if direct_url is None else direct_url
    editable = bool((du.get("dir_info") or {}).get("editable"))
    info = InstallInfo(root=str(root), editable=editable)
    if (root / ".git").exists():
        info.kind = "git"
        rc, out = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, timeout=4)
        info.branch = out.strip() if rc == 0 and out.strip() != "HEAD" else ""
        rc, out = run(["git", "remote", "get-url", "origin"], cwd=root, timeout=4)
        if rc == 0 and out.strip():
            info.has_remote, info.remote_url = True, out.strip()
        return info
    pre = (prefix if prefix is not None else sys.prefix).replace("\\", "/")
    if "/uv/tools/" in pre and not editable:
        info.kind = "uv-tool"
    elif editable:
        info.kind = "editable"
    elif du.get("vcs_info") or _dist_installed():
        info.kind = "pip"
    if du.get("vcs_info"):
        info.vcs_url = str(du.get("url") or "")
        info.vcs_commit = str((du.get("vcs_info") or {}).get("commit_id") or "")
    return info


def _dist_installed() -> bool:
    try:
        from importlib.metadata import version

        version(DIST_NAME)
        return True
    except Exception:  # noqa: BLE001
        return False


def repo_slug(config: dict | None, install: InstallInfo | None = None) -> tuple[str, str] | None:
    """Which GitHub repository to ask: `update.repo` in config.yaml, else the checkout's origin, else brand.REPO_URL."""
    for cand in (str(_section(config).get("repo") or ""), (install.remote_url if install else ""), (install.vcs_url if install else ""),
                 brand.REPO_URL):
        s = parse_slug(cand)
        if s:
            return s
    return None


# ----------------------------------------------------------------------------------------------- results


@dataclass
class FrameworkUpdate:
    current: str = ""
    latest: str = ""
    tag: str = ""
    available: bool = False
    source: str = ""               # "release" | "branch" | ""
    behind: int = 0                # git checkouts: commits the branch is behind (0 = unknown or none)
    branch: str = ""
    head: str = ""                 # the checkout's HEAD when checked
    notes_url: str = ""
    notes: str = ""
    install_kind: str = ""
    repo: str = ""
    error: str = ""
    no_source: bool = False        # nothing to ask (no repository configured)


@dataclass
class PluginUpdate:
    name: str = ""
    current: str = ""
    latest: str = ""
    url: str = ""
    ref: str = ""
    installed_commit: str = ""
    remote_commit: str = ""
    available: bool = False
    note: str = ""                 # "no update source", "offline", ...


@dataclass
class UpdateReport:
    checked_at: float = 0.0
    framework: FrameworkUpdate = field(default_factory=FrameworkUpdate)
    plugins: list[PluginUpdate] = field(default_factory=list)
    offline: bool = False
    from_cache: bool = False
    disabled: bool = False
    schema: int = CACHE_SCHEMA

    @property
    def plugin_updates(self) -> list[PluginUpdate]:
        return [p for p in self.plugins if p.available]

    @property
    def any(self) -> bool:
        return self.framework.available or bool(self.plugin_updates)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UpdateReport:
        fw = FrameworkUpdate(**{k: v for k, v in (d.get("framework") or {}).items() if k in FrameworkUpdate.__dataclass_fields__})
        pl = [PluginUpdate(**{k: v for k, v in p.items() if k in PluginUpdate.__dataclass_fields__}) for p in d.get("plugins") or []]
        return cls(float(d.get("checked_at") or 0), fw, pl, bool(d.get("offline")), False, bool(d.get("disabled")),
                   int(d.get("schema") or 0))


# ----------------------------------------------------------------------------------------------- the checks


def default_http(url: str, timeout: float = HTTP_TIMEOUT) -> tuple[int, Any]:
    """Plain GET -> (status, parsed JSON or None). Raises on network errors. Sends only what HTTP needs plus a fixed User-Agent
    (GitHub's API refuses requests without one); no version, no id, no cookies."""
    import httpx

    r = httpx.get(url, timeout=httpx.Timeout(timeout), follow_redirects=True,
                  headers={"Accept": "application/vnd.github+json", "User-Agent": "nightage-update-check"})
    try:
        data = r.json()
    except ValueError:
        data = None
    return r.status_code, data


def _api(config: dict | None) -> str:
    return str(_section(config).get("api_base") or API_BASE).rstrip("/")


def latest_release(slug: tuple[str, str], config: dict | None, http: HttpGet, *, include_pre: bool = False) -> dict[str, Any]:
    """{'tag', 'version', 'url', 'notes', 'error'} for the newest release (or, without releases, the highest version tag)."""
    owner, repo = slug
    base = f"{_api(config)}/repos/{owner}/{repo}"
    out: dict[str, Any] = {"tag": "", "version": "", "url": "", "notes": "", "error": ""}
    try:
        status, data = http(f"{base}/releases/latest" if not include_pre else f"{base}/releases?per_page=10", HTTP_TIMEOUT)
        if include_pre and status == 200 and isinstance(data, list):
            rels = [r for r in data if isinstance(r, dict) and not r.get("draft") and parse_version(str(r.get("tag_name", "")))]
            data = max(rels, key=lambda r: parse_version(str(r["tag_name"])), default=None)             # type: ignore[arg-type,return-value]
            status = 200 if data else 404
        if status == 200 and isinstance(data, dict) and data.get("tag_name"):
            tag = str(data["tag_name"])
            out.update(tag=tag, version=tag.lstrip("vV"), url=str(data.get("html_url") or ""), notes=str(data.get("body") or ""))
            return out
        if status in (403, 429):
            out["error"] = "GitHub rate limit reached; try again later"
            return out
        if status not in (404, 200):
            out["error"] = f"GitHub answered {status}"
            return out
        status, data = http(f"{base}/tags?per_page=50", HTTP_TIMEOUT)             # no releases: fall back to version tags
        if status == 200 and isinstance(data, list):
            tags = [str(t.get("name", "")) for t in data if isinstance(t, dict) and parse_version(str(t.get("name", "")))]
            tags = [t for t in tags if include_pre or not is_prerelease(t)]
            if tags:
                best = max(tags, key=lambda t: parse_version(t))                  # type: ignore[arg-type,return-value]
                out.update(tag=best, version=best.lstrip("vV"), url=f"https://github.com/{owner}/{repo}/releases/tag/{best}")
        elif status not in (404,):
            out["error"] = f"GitHub answered {status}"
    except Exception as exc:  # noqa: BLE001 - offline, DNS, TLS, timeout ...
        out["error"] = f"offline ({type(exc).__name__})"
    return out


def git_branch_state(install: InstallInfo, run: Runner) -> dict[str, Any]:
    """Compare the checkout's branch with its remote WITHOUT touching the checkout:
    {'head', 'remote', 'behind', 'state': 'same' | 'behind' | 'ahead' | 'diverged' | 'unknown', 'error'}."""
    root = Path(install.root)
    res: dict[str, Any] = {"head": "", "remote": "", "behind": 0, "state": "unknown", "error": ""}
    rc, head = run(["git", "rev-parse", "HEAD"], cwd=root, timeout=4)
    if rc != 0:
        res["error"] = "not a readable git checkout"
        return res
    head = head.strip()
    res["head"] = head
    if not install.has_remote or not install.branch:
        res["error"] = "no remote" if not install.has_remote else "detached HEAD"
        return res
    rc, up = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=root, timeout=4)
    remote_name, branch = "origin", install.branch
    if rc == 0 and "/" in up.strip():
        remote_name, _, branch = up.strip().partition("/")
    rc, out = run(["git", "ls-remote", remote_name, f"refs/heads/{branch}"], cwd=root, timeout=GIT_TIMEOUT)
    if rc != 0:
        res["error"] = f"offline ({out.strip().splitlines()[-1][:60]})" if out.strip() else "offline"
        return res
    line = out.strip().splitlines()[0] if out.strip() else ""
    remote = line.split()[0] if line else ""
    if not remote:
        res["error"] = f"branch {branch} not found on {remote_name}"
        return res
    res["remote"] = remote
    if remote == head:
        res["state"] = "same"
        return res
    has_obj = run(["git", "cat-file", "-e", f"{remote}^{{commit}}"], cwd=root, timeout=4)[0] == 0
    if not has_obj:
        res["state"], res["behind"] = "behind", 0                     # new commits we have never fetched (count unknown until fetch)
        return res
    if run(["git", "merge-base", "--is-ancestor", remote, head], cwd=root, timeout=4)[0] == 0:
        res["state"] = "ahead"                                        # we already contain it
        return res
    if run(["git", "merge-base", "--is-ancestor", head, remote], cwd=root, timeout=4)[0] == 0:
        rc, n = run(["git", "rev-list", "--count", f"{head}..{remote}"], cwd=root, timeout=4)
        res["state"], res["behind"] = "behind", int(n.strip() or 0) if rc == 0 and n.strip().isdigit() else 0
        return res
    res["state"] = "diverged"
    return res


def check_framework(config: dict | None, *, http: HttpGet, run: Runner, install: InstallInfo | None = None,
                    root: Path | None = None) -> FrameworkUpdate:
    install = install or detect_install(root=root, run=run)
    fw = FrameworkUpdate(current=current_version(Path(install.root) if install.root else root), install_kind=install.kind,
                         branch=install.branch)
    slug = repo_slug(config, install)
    if slug:
        fw.repo = f"{slug[0]}/{slug[1]}"
        rel = latest_release(slug, config, http, include_pre=bool(_section(config).get("prerelease")) or is_prerelease(fw.current))
        fw.error = rel["error"]
        if rel["version"]:
            fw.latest, fw.tag, fw.notes_url, fw.notes = rel["version"], rel["tag"], rel["url"], rel["notes"]
            if is_newer(rel["version"], fw.current):
                fw.available, fw.source = True, "release"
    else:
        fw.no_source = True
    if install.kind == "git" and install.has_remote:
        st = git_branch_state(install, run)
        fw.head = st["head"]
        if st["state"] == "behind" and not fw.available:
            fw.available, fw.source, fw.behind = True, "branch", st["behind"]
        elif st["state"] == "behind":
            fw.behind = st["behind"]
        if st["error"] and not fw.error and not fw.latest:
            fw.error = st["error"]
    return fw


def check_plugins(config: dict | None, *, run: Runner) -> list[PluginUpdate]:
    from agent import plugin_update

    return plugin_update.check_all(config, run=run)


def _load_cache(path: Path) -> UpdateReport | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        rep = UpdateReport.from_dict(d)
        return rep if rep.schema == CACHE_SCHEMA else None
    except Exception:  # noqa: BLE001
        return None


def _save_cache(path: Path, rep: UpdateReport) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _fresh(rep: UpdateReport, config: dict | None, now: float) -> bool:
    ttl = OFFLINE_RETRY_S if rep.offline else interval_s(config)
    return 0 <= now - rep.checked_at < ttl


def check_for_updates(config: dict | None = None, force: bool = False, *, http: HttpGet | None = None, run: Runner | None = None,
                      now: float | None = None, install: InstallInfo | None = None, cache: Path | None = None,
                      root: Path | None = None, respect_opt_out: bool = False) -> UpdateReport:
    """The framework and every git-sourced plugin. Never raises; offline just yields `offline=True` and no updates.

    `force` skips the cache. `respect_opt_out=True` (used by the automatic paths) returns an empty `disabled` report when the
    user switched checking off; the explicit commands leave it False."""
    t = time.time() if now is None else now
    if respect_opt_out and not enabled(config):
        return UpdateReport(checked_at=t, disabled=True)
    path = cache or cache_path()
    if not force:
        cached = _load_cache(path)
        if cached and _fresh(cached, config, t) and cached.framework.current == current_version(root):
            cached.from_cache = True
            return cached
    http, run = http or default_http, run or default_runner
    try:
        fw = check_framework(config, http=http, run=run, install=install, root=root)
    except Exception as exc:  # noqa: BLE001
        fw = FrameworkUpdate(current=current_version(root), error=f"{type(exc).__name__}")
    try:
        plugins = check_plugins(config, run=run)
    except Exception:  # noqa: BLE001
        plugins = []
    rep = UpdateReport(checked_at=t, framework=fw, plugins=plugins)
    rep.offline = fw.error.startswith("offline") and not fw.latest
    _save_cache(path, rep)
    return rep


# ----------------------------------------------------------------------------------------------- cache-only reading (no network)


def cached_report(config: dict | None = None, *, cache: Path | None = None, root: Path | None = None,
                  run: Runner | None = None) -> UpdateReport | None:
    """What the last check found, minus anything the user already installed since. No network, no subprocess when the cache is
    missing; None when nothing is known or checking is switched off."""
    if not enabled(config):
        return None
    rep = _load_cache(cache or cache_path())
    if rep is None:
        return None
    cur = current_version(root)
    fw = rep.framework
    if fw.available:
        if fw.source == "release" and not is_newer(fw.latest, cur) or fw.source == "branch" and fw.head and _head_changed(fw.head, root, run):
            fw.available = False
    fw.current = cur
    try:
        from agent import plugin_update

        installed = plugin_update.installed_commits()
        for p in rep.plugins:
            if p.available and installed.get(p.name, p.installed_commit) != p.installed_commit:
                p.available = False
    except Exception:  # noqa: BLE001
        pass
    return rep


def _head_changed(old: str, root: Path | None, run: Runner | None) -> bool:
    rc, out = (run or default_runner)(["git", "rev-parse", "HEAD"], cwd=root or project_root(), timeout=3)
    return rc == 0 and out.strip() != old


def notice_lines(config: dict | None = None, **kw: Any) -> list[str]:
    """The calm lines for the home screen / doctor / --version. [] when there is nothing (or nothing cached yet)."""
    rep = cached_report(config, **kw)
    if rep is None:
        return []
    out = []
    fw = rep.framework
    if fw.available:
        if fw.source == "release":
            out.append(f"Update available: {fw.latest} (you have {fw.current}) - run `{brand.CMD} update`")
        else:
            n = f"{fw.behind} new commit(s)" if fw.behind else "new commits"
            out.append(f"Update available: {n} on {fw.branch or 'the branch'} (you have {fw.current}) - run `{brand.CMD} update`")
    n_pl = len(rep.plugin_updates)
    if n_pl:
        out.append(f"{n_pl} plugin update{'s' if n_pl != 1 else ''} - `{brand.CMD} plugin update`")
    return out


def notice_line(config: dict | None = None, **kw: Any) -> str:
    return "  ".join(notice_lines(config, **kw))


def version_text(version: str | None = None, config: dict | None = None) -> str:
    """For `nightage --version`: the version, plus the calm notice when the cache knows of an update."""
    head = f"{brand.NAME} {version or current_version()}"
    lines = notice_lines(config)
    return "\n".join([head, *lines])


_bg_lock = threading.Lock()
_bg_started = False


def start_background_check(config: dict | None = None, **kw: Any) -> threading.Thread | None:
    """Refresh the cache in a daemon thread (once per process). Returns at once; never raises; nothing when opted out."""
    global _bg_started
    try:
        if not enabled(config):
            return None
        with _bg_lock:
            if _bg_started:
                return None
            _bg_started = True
        t = threading.Thread(target=lambda: check_for_updates(config, **kw), name="nightage-update-check", daemon=True)
        t.start()
        return t
    except Exception:  # noqa: BLE001
        return None


def _reset_background_for_tests() -> None:
    global _bg_started
    _bg_started = False


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M UTC") if ts else "never"
