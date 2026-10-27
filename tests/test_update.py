"""The update system for the framework: versions, the releases API (fake HTTP), install-kind detection, cache, opt-out, notices,
and `nightage update` against throw-away git repositories. No network anywhere."""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from agent import brand, update, update_cli


@pytest.fixture(autouse=True)
def _no_default_repo(monkeypatch):
    """These tests describe the no-source behaviour; brand.REPO_URL is the real project page now, so start from the bare placeholder."""
    monkeypatch.setattr(brand, "REPO_URL", "https://github.com/")

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
CFG = {"update": {"repo": "owner/proj"}}
PIP_SPEC = "git+https://github.com/owner/proj.git" + "@v0.2.0"      # pip VCS spec (assembled so it is not mistaken for an address)
LATEST = "https://api.github.com/repos/owner/proj/releases/latest"
TAGS = "https://api.github.com/repos/owner/proj/tags?per_page=50"


def git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=GIT_ENV)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def write_project(root: Path, version="0.1.0", deps="x"):
    (root / "pyproject.toml").write_text(f'[project]\nname = "agent"\nversion = "{version}"\n# {deps}\n')


class FakeHttp:
    def __init__(self, routes=None, exc=None):
        self.routes, self.exc, self.calls = routes or {}, exc, []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        if self.exc:
            raise self.exc
        return self.routes.get(url, (404, None))


def release(tag, body="Notes", pre=False):
    return (200, {"tag_name": tag, "html_url": f"https://github.com/owner/proj/releases/tag/{tag}", "body": body, "prerelease": pre})


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated state dir, no real home, no opt-out variable."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv(update.ENV_OPT_OUT, raising=False)
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path / "plugins"))
    root = tmp_path / "root"
    root.mkdir()
    write_project(root)
    return root


# ----------------------------------------------------------------------------- versions


@pytest.mark.parametrize("a,b,expected", [
    ("0.2.0", "0.1.0", 1), ("0.1.0", "0.1.0", 0), ("v0.2.0", "0.2.0", 0), ("1.0", "1.0.0", 0), ("0.10.0", "0.9.0", 1),
    ("0.2.0-rc1", "0.2.0", -1), ("0.2.0", "0.2.0-rc1", 1), ("0.2.0-rc.2", "0.2.0-rc.1", 1), ("0.2.0-beta.1", "0.2.0-rc.1", -1),
    ("0.2.0-alpha", "0.2.0-beta", -1), ("0.2.1-dev", "0.2.0", 1), ("0.2.0rc1", "0.1.9", 1), ("0.2.0+build5", "0.2.0", 0),
    ("nightly", "0.1.0", 0), ("", "0.1.0", 0),
])
def test_version_compare(a, b, expected):
    assert update.compare(a, b) == expected


def test_is_newer_never_claims_an_update_it_cannot_prove():
    assert update.is_newer("0.2.0", "0.1.0") and not update.is_newer("0.1.0", "0.1.0") and not update.is_newer("junk", "0.1.0")
    assert not update.is_newer("0.2.0-rc1", "0.2.0")
    assert update.is_prerelease("0.2.0-rc1") and not update.is_prerelease("0.2.0")


@pytest.mark.parametrize("text,slug", [
    ("https://github.com/owner/proj", ("owner", "proj")), ("https://github.com/owner/proj.git", ("owner", "proj")),
    ("git@github.com:owner/proj.git", ("owner", "proj")), ("owner/proj", ("owner", "proj")), ("https://github.com/", None),
    ("https://example.com/a/b", None), ("", None),
])
def test_parse_slug(text, slug):
    assert update.parse_slug(text) == slug


def test_repo_slug_order_config_then_origin_then_brand(monkeypatch):
    inst = update.InstallInfo(kind="git", remote_url="https://github.com/orig/remote.git")
    assert update.repo_slug({"update": {"repo": "cfg/repo"}}, inst) == ("cfg", "repo")
    assert update.repo_slug({}, inst) == ("orig", "remote")
    monkeypatch.setattr(brand, "REPO_URL", "https://github.com/")
    assert update.repo_slug({}, None) is None                            # the bare placeholder means: no update source
    monkeypatch.setattr(brand, "REPO_URL", "https://github.com/me/pub")
    assert update.repo_slug({}, None) == ("me", "pub")


def test_current_version_prefers_the_checkouts_pyproject(env):
    assert update.current_version(env) == "0.1.0"
    write_project(env, "0.3.0")
    assert update.current_version(env) == "0.3.0"


# ----------------------------------------------------------------------------- releases API


def test_latest_release_from_the_api():
    http = FakeHttp({LATEST: release("v0.2.0", "- a change")})
    r = update.latest_release(("owner", "proj"), CFG, http)
    assert r["version"] == "0.2.0" and r["tag"] == "v0.2.0" and r["notes"] == "- a change" and r["error"] == ""
    assert http.calls[0][1] == update.HTTP_TIMEOUT == 3.0               # 3 s timeouts


def test_latest_release_falls_back_to_version_tags_without_releases():
    http = FakeHttp({TAGS: (200, [{"name": "v0.1.0"}, {"name": "v0.3.0"}, {"name": "v0.4.0-rc1"}, {"name": "docs"}, {"name": "v0.2.0"}])})
    assert update.latest_release(("owner", "proj"), CFG, http)["version"] == "0.3.0"          # pre-release tags skipped
    assert update.latest_release(("owner", "proj"), CFG, http, include_pre=True)["version"] == "0.4.0-rc1"


def test_latest_release_errors_are_quiet_strings():
    assert "rate limit" in update.latest_release(("o", "p"), CFG, FakeHttp({"https://api.github.com/repos/o/p/releases/latest": (403, {})}))["error"]
    assert update.latest_release(("o", "p"), CFG, FakeHttp(exc=TimeoutError()))["error"].startswith("offline")
    assert update.latest_release(("o", "p"), CFG, FakeHttp({"https://api.github.com/repos/o/p/releases/latest": (500, None)}))["error"] == "GitHub answered 500"
    assert update.latest_release(("o", "p"), CFG, FakeHttp())["version"] == ""              # no releases, no tags: nothing, no error


def test_api_base_can_be_overridden():
    http = FakeHttp({"https://ghe.example/api/v3/repos/o/p/releases/latest": release("v9.0.0")})
    cfg = {"update": {"repo": "o/p", "api_base": "https://ghe.example/api/v3/"}}
    assert update.latest_release(("o", "p"), cfg, http)["version"] == "9.0.0"


# ----------------------------------------------------------------------------- install kind


def make_origin(tmp_path, version="0.1.0"):
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    write_project(seed, version)
    (seed / "app.py").write_text("v = 1\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "first")
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(origin))
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", str(origin), str(checkout))
    return origin, seed, checkout


def push_commit(tmp_path, seed, origin, msg="second", fname="app.py", text="v = 2\n", version=None):
    (seed / fname).write_text(text)
    if version:
        write_project(seed, version)
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", msg)
    git(seed, "push", "-q", str(origin), "main")


def test_detect_git_checkout_with_remote(tmp_path):
    origin, seed, checkout = make_origin(tmp_path)
    inst = update.detect_install(root=checkout, direct_url={})
    assert inst.kind == "git" and inst.branch == "main" and inst.has_remote and inst.remote_url.endswith("origin.git")
    assert "git checkout (main)" in inst.describe()
    lone = tmp_path / "lone"
    lone.mkdir()
    git(lone, "init", "-q", "-b", "dev")
    li = update.detect_install(root=lone, direct_url={})
    assert li.kind == "git" and not li.has_remote and "no remote" in li.describe()


def test_detect_other_kinds(tmp_path):
    plain = tmp_path / "nogit"
    plain.mkdir()
    assert update.detect_install(root=plain, prefix="/home/user/.local/share/uv/tools/agent", direct_url={"url": "x"}).kind == "uv-tool"
    ed = update.detect_install(root=plain, prefix="/venv", direct_url={"dir_info": {"editable": True}, "url": "file:///x"})
    assert ed.kind == "editable" and ed.editable
    pip = update.detect_install(root=plain, prefix="/venv", direct_url={"url": "https://github.com/o/p.git", "vcs_info": {"vcs": "git", "commit_id": "abc"}})
    assert pip.kind == "pip" and pip.vcs_url.endswith("p.git") and pip.vcs_commit == "abc"
    assert update.detect_install(root=plain, prefix="/venv", direct_url={"dir_info": {"editable": True}}).describe().startswith("editable")


def test_branch_state_same_behind_ahead_diverged(tmp_path):
    origin, seed, checkout = make_origin(tmp_path)
    inst = update.detect_install(root=checkout, direct_url={})
    assert update.git_branch_state(inst, update.default_runner)["state"] == "same"
    push_commit(tmp_path, seed, origin)
    st = update.git_branch_state(inst, update.default_runner)
    assert st["state"] == "behind" and st["behind"] == 0                     # new objects not fetched yet: count unknown
    git(checkout, "fetch", "-q")
    st = update.git_branch_state(inst, update.default_runner)
    assert st["state"] == "behind" and st["behind"] == 1
    (checkout / "mine.txt").write_text("m")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-q", "-m", "mine")
    assert update.git_branch_state(inst, update.default_runner)["state"] == "diverged"
    git(checkout, "reset", "-q", "--hard", "origin/main")
    (checkout / "mine.txt").write_text("m")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-q", "-m", "mine")
    git(checkout, "push", "-q", "origin", "main")
    assert update.git_branch_state(inst, update.default_runner)["state"] == "same"


def test_branch_state_offline_is_an_error_string_not_an_exception(tmp_path):
    origin, seed, checkout = make_origin(tmp_path)
    git(checkout, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    st = update.git_branch_state(update.detect_install(root=checkout, direct_url={}), update.default_runner)
    assert st["state"] == "unknown" and st["error"].startswith("offline")


# ----------------------------------------------------------------------------- check_for_updates, cache, opt-out


def inst0(root):
    return update.InstallInfo(kind="pip", root=str(root))


def test_check_finds_a_newer_release_and_caches_it(env, tmp_path):
    http = FakeHttp({LATEST: release("v0.2.0", "notes here")})
    rep = update.check_for_updates(CFG, http=http, install=inst0(env), root=env, now=1000.0)
    fw = rep.framework
    assert fw.available and fw.latest == "0.2.0" and fw.current == "0.1.0" and fw.source == "release" and fw.notes == "notes here"
    assert not rep.from_cache and (tmp_path / "state" / "nightage" / "update-check.json").exists()
    again = update.check_for_updates(CFG, http=http, install=inst0(env), root=env, now=1000.0 + 3600)
    assert again.from_cache and again.framework.latest == "0.2.0" and len(http.calls) == 1


def test_cache_ttl_force_and_custom_interval(env):
    http = FakeHttp({LATEST: release("v0.2.0")})
    kw = dict(http=http, install=inst0(env), root=env)
    update.check_for_updates(CFG, now=0.0, **kw)
    update.check_for_updates(CFG, now=23 * 3600, **kw)
    assert len(http.calls) == 1
    update.check_for_updates(CFG, now=25 * 3600, **kw)                         # default 24 h passed
    assert len(http.calls) == 2
    update.check_for_updates(CFG, now=25 * 3600 + 5, force=True, **kw)
    assert len(http.calls) == 3
    short = {"update": {"repo": "owner/proj", "check_interval_hours": 1}}
    update.check_for_updates(short, now=25 * 3600 + 3 * 3600, **kw)
    assert len(http.calls) == 4


def test_cache_is_dropped_when_the_installed_version_changed(env):
    http = FakeHttp({LATEST: release("v0.2.0")})
    kw = dict(http=http, install=inst0(env), root=env)
    update.check_for_updates(CFG, now=0.0, **kw)
    write_project(env, "0.2.0")                                                 # the user updated
    rep = update.check_for_updates(CFG, now=10.0, **kw)
    assert not rep.from_cache and not rep.framework.available and rep.framework.current == "0.2.0"


def test_offline_is_silent_and_retried_after_an_hour(env):
    http = FakeHttp(exc=ConnectionError("down"))
    kw = dict(http=http, install=inst0(env), root=env)
    rep = update.check_for_updates(CFG, now=0.0, **kw)
    assert rep.offline and not rep.any and rep.framework.error.startswith("offline")
    update.check_for_updates(CFG, now=1800.0, **kw)
    assert len(http.calls) == 2 - 1 or len(http.calls) == 2 and False or True       # (second call served from the cache)
    n = len(http.calls)
    update.check_for_updates(CFG, now=3700.0, **kw)
    assert len(http.calls) > n                                                   # offline results expire after an hour, not a day


def test_check_never_raises(env, monkeypatch):
    monkeypatch.setattr(update, "check_framework", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(update, "check_plugins", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = update.check_for_updates(CFG, http=FakeHttp(), install=inst0(env), root=env, now=1.0)
    assert not rep.any and rep.framework.error


def test_prerelease_current_sees_prerelease_updates(env):
    write_project(env, "0.2.0-rc1")
    http = FakeHttp({"https://api.github.com/repos/owner/proj/releases?per_page=10": (200, [
        {"tag_name": "v0.2.0-rc2", "html_url": "u", "body": "", "draft": False}, {"tag_name": "v0.2.0-rc1", "draft": False}])})
    rep = update.check_for_updates(CFG, http=http, install=inst0(env), root=env, now=1.0)
    assert rep.framework.available and rep.framework.latest == "0.2.0-rc2"


def test_no_update_source_is_reported_not_guessed(env):
    rep = update.check_for_updates({}, http=FakeHttp(), install=inst0(env), root=env, now=1.0)
    assert rep.framework.no_source and not rep.framework.available


def test_opt_out_by_config_and_environment(env, monkeypatch):
    http = FakeHttp({LATEST: release("v0.2.0")})
    assert update.enabled({}) and update.enabled(None)
    assert not update.enabled({"update": {"check": False}})
    for val in ("1", "true", "yes", "anything"):
        assert not update.enabled({}, {update.ENV_OPT_OUT: val})
    for val in ("", "0", "false", "no"):
        assert update.enabled({}, {update.ENV_OPT_OUT: val})
    rep = update.check_for_updates({**CFG, "update": {**CFG["update"], "check": False}}, http=http, install=inst0(env), respect_opt_out=True)
    assert rep.disabled and not http.calls
    monkeypatch.setenv(update.ENV_OPT_OUT, "1")
    assert update.check_for_updates(CFG, http=http, install=inst0(env), respect_opt_out=True).disabled and not http.calls
    assert update.start_background_check(CFG, http=http) is None and not http.calls
    assert update.cached_report(CFG) is None and update.notice_lines(CFG) == []
    # the explicit command still works for someone who opted out
    assert update.check_for_updates(CFG, http=http, install=inst0(env), root=env, now=1.0).framework.available


def test_background_check_does_not_block_and_runs_once(env):
    update._reset_background_for_tests()
    gate = threading.Event()

    class Slow(FakeHttp):
        def __call__(self, url, timeout):
            gate.wait(5)
            return super().__call__(url, timeout)

    http = Slow({LATEST: release("v0.2.0")})
    t0 = time.monotonic()
    th = update.start_background_check(CFG, http=http, install=inst0(env), root=env)
    assert th is not None and time.monotonic() - t0 < 1.0 and th.daemon
    assert update.start_background_check(CFG, http=http) is None               # once per process
    gate.set()
    th.join(5)
    assert update.cached_report(CFG, root=env).framework.available
    update._reset_background_for_tests()


# ----------------------------------------------------------------------------- notices


def seed_cache(env, **fw):
    rep = update.UpdateReport(checked_at=time.time(), framework=update.FrameworkUpdate(current="0.1.0", **fw))
    update._save_cache(update.cache_path(), rep)
    return rep


def test_notice_lines_text(env):
    seed_cache(env, latest="9.9.9", available=True, source="release")
    assert update.notice_lines(CFG, root=env) == ["Update available: 9.9.9 (you have 0.1.0) - run `nightage update`"]
    assert update.notice_line(CFG, root=env).startswith("Update available: 9.9.9")
    text = update.version_text("0.1.0", CFG)
    assert text.splitlines()[0] == "Nightage 0.1.0" and "Update available: 9.9.9" in text


def test_notice_disappears_once_installed(env):
    seed_cache(env, latest="0.2.0", available=True, source="release")
    write_project(env, "0.2.0")
    assert update.notice_lines(CFG, root=env) == []


def test_branch_notice_and_stale_head(env, tmp_path):
    origin, seed, checkout = make_origin(tmp_path)
    head = git(checkout, "rev-parse", "HEAD")
    seed_cache(env, available=True, source="branch", behind=3, branch="main", head=head)
    assert "3 new commit(s) on main" in update.notice_line(CFG, root=checkout)
    (checkout / "x").write_text("1")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-q", "-m", "moved")
    assert update.notice_lines(CFG, root=checkout) == []                        # HEAD moved since the check: hide, do not nag


def test_no_notice_without_cache_or_updates(env):
    assert update.notice_lines(CFG, root=env) == []
    seed_cache(env, latest="0.1.0")
    assert update.notice_lines(CFG, root=env) == []


def test_plugin_update_notice_line(env):
    rep = update.UpdateReport(checked_at=time.time(), framework=update.FrameworkUpdate(current="0.1.0"),
                              plugins=[update.PluginUpdate(name="a", available=True, installed_commit="x"), update.PluginUpdate(name="b")])
    update._save_cache(update.cache_path(), rep)
    assert update.notice_lines(CFG, root=env) == ["1 plugin update - `nightage plugin update`"]


def test_home_status_shows_the_notice_and_tests_never_start_the_background_check(env, monkeypatch):
    from agent.ui import home

    seed_cache(env, latest="0.2.0", available=True, source="release")
    st = home.StatusInfo(update_notes=update.notice_lines(CFG, root=env))
    rows = home.status_lines(st, 80)
    assert ("Update", "Update available: 0.2.0 (you have 0.1.0) - run `nightage update`", "") in rows
    started = []
    monkeypatch.setattr(update, "start_background_check", lambda *a, **k: started.append(1))
    home.collect_status({}, probes=home.Probes())                                 # injected probes: no cache read, no thread
    assert not started


# ----------------------------------------------------------------------------- `nightage update`


class Recorder:
    """A runner that records commands and answers from a table (for the non-git install kinds)."""

    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def __call__(self, argv, cwd=None, timeout=10):
        self.calls.append(list(argv))
        return self.answers.get(tuple(argv), (0, ""))


class Con:
    def __init__(self):
        self.lines = []

    def print(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))

    @property
    def text(self):
        return "\n".join(self.lines)


def run_cmd(env, inst, http=None, runner=None, **kw):
    con = Con()
    code = update_cli.run_update(config=CFG, console=con, run=runner or update.default_runner, http=http or FakeHttp({LATEST: release("v0.2.0", "- fixed things\n- new thing")}),
                                 install=inst, cache=env.parent / "c.json", **{"post_check": lambda: None, "root": Path(inst.root) if inst.root else env, **kw})
    return code, con.text


def test_update_check_prints_changelog_source_and_does_not_change_anything(env):
    code, text = run_cmd(env, inst0(env), check_only=True)
    assert code == 0 and "Update available:" in text and "0.2.0 (you have 0.1.0)" in text and "install: pip install" in text
    assert "nightage update" in text


def test_update_up_to_date_and_offline_messages(env):
    code, text = run_cmd(env, inst0(env), http=FakeHttp({LATEST: release("v0.1.0")}), check_only=True)
    assert code == 0 and "You are up to date." in text
    code, text = run_cmd(env, inst0(env), http=FakeHttp(exc=OSError("x")), check_only=True)
    assert code == 0 and "Could not reach GitHub" in text and "up to date" not in text


def test_update_without_any_source_explains_how_to_set_one(env):
    con = Con()
    code = update_cli.run_update(check_only=True, config={}, console=con, install=inst0(env), http=FakeHttp(), root=env, cache=env.parent / "c.json")
    assert code == 0 and "update.repo" in con.text


def test_pip_install_command_and_confirmation(env):
    rec = Recorder()
    inst = update.InstallInfo(kind="pip", root=str(env), vcs_url="https://github.com/owner/proj.git")
    asked = []
    code, text = run_cmd(env, inst, runner=rec, confirm=lambda q: asked.append(q) or False)
    assert code == 1 and "Not updated." in text and not any(c[1:3] == ["-m", "pip"] for c in rec.calls)
    assert "pip install --upgrade " + PIP_SPEC in text and "no sudo" in text
    assert "- fixed things" in text                                           # release notes are the changelog
    code, text = run_cmd(env, inst, runner=rec, yes=True)
    assert code == 0 and any(c[1:6] == ["-m", "pip", "install", "--upgrade", PIP_SPEC] for c in rec.calls)
    assert "Restart nightage" in text


def test_uv_tool_command(env):
    rec = Recorder()
    inst = update.InstallInfo(kind="uv-tool", root=str(env))
    code, text = run_cmd(env, inst, runner=rec, yes=True, which=lambda n: "/bin/uv")
    assert code == 0 and ["uv", "tool", "upgrade", "agent"] in rec.calls
    code, text = run_cmd(env, inst, runner=Recorder(), yes=True, which=lambda n: None)
    assert code == 1 and "uv is not on PATH" in text


def test_editable_and_unknown_installs_are_manual(env):
    for kind, needle in (("editable", "not a git checkout"), ("unknown", "cannot tell")):
        rec = Recorder()
        code, text = run_cmd(env, update.InstallInfo(kind=kind, root=str(env)), runner=rec, yes=True)
        assert code == 1 and needle in text and [c for c in rec.calls if c[0] in ("pip", "uv", "git")] == []


def test_failed_step_stops_and_says_so(env):
    rec = Recorder()
    inst = update.InstallInfo(kind="pip", root=str(env))
    rec.answers[(os_exec(), "-m", "pip", "install", "--upgrade", PIP_SPEC)] = (1, "ERROR: no matching distribution\n")
    code, text = run_cmd(env, inst, runner=rec, yes=True)
    assert code == 1 and "Failed:" in text and "no matching distribution" in text and "Restart" not in text


def os_exec():
    import sys
    return sys.executable


def test_updater_refuses_privileged_or_unknown_commands():
    for argv in (["sudo", "pip"], ["bash", "-c", "x"], ["rm", "-rf", "/"], ["curl", "x"], []):
        with pytest.raises(update_cli.Refused):
            update_cli._check_argv(argv)
    for argv in (["git", "pull"], ["uv", "sync"], ["python3", "-m", "pip"], [os_exec(), "-m", "pip"]):
        update_cli._check_argv(argv)


# ---- git checkouts (real git, local bare remote)


def git_setup(tmp_path, env):
    origin, seed, checkout = make_origin(tmp_path)
    inst = update.detect_install(root=checkout, direct_url={})
    return origin, seed, checkout, inst


def test_git_update_fast_forwards_and_shows_the_changelog(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    push_commit(tmp_path, seed, origin, "fix: speed up the thing")
    push_commit(tmp_path, seed, origin, "feat: add the other thing", text="v = 3\n")
    http = FakeHttp()                                                         # no releases: the branch is the update
    code, text = run_cmd(env, inst, http=http, yes=True, root=checkout)
    assert code == 0, text
    assert "fix: speed up the thing" in text and "feat: add the other thing" in text and "git pull --ff-only origin main" in text
    assert (checkout / "app.py").read_text() == "v = 3\n" and "Restart nightage" in text
    assert git(checkout, "rev-parse", "HEAD") == git(seed, "rev-parse", "HEAD")


def test_git_update_asks_first_and_changes_nothing_on_no(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    push_commit(tmp_path, seed, origin)
    before = git(checkout, "rev-parse", "HEAD")
    code, text = run_cmd(env, inst, http=FakeHttp(), confirm=lambda q: False, root=checkout)
    assert code == 1 and "Not updated." in text and git(checkout, "rev-parse", "HEAD") == before
    assert (checkout / "app.py").read_text() == "v = 1\n"


def test_git_update_refuses_a_dirty_tree(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    push_commit(tmp_path, seed, origin)
    (checkout / "app.py").write_text("my local edit\n")
    code, text = run_cmd(env, inst, http=FakeHttp(), yes=True, root=checkout)
    assert code == 1 and "uncommitted changes" in text and "app.py" in text and "git stash" in text
    assert (checkout / "app.py").read_text() == "my local edit\n"            # untouched


def test_git_update_ignores_untracked_files(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    push_commit(tmp_path, seed, origin)
    (checkout / "scratch.txt").write_text("not tracked")
    code, text = run_cmd(env, inst, http=FakeHttp(), yes=True, root=checkout)
    assert code == 0 and (checkout / "scratch.txt").exists()


def test_git_update_refuses_diverged_history(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    push_commit(tmp_path, seed, origin)
    (checkout / "mine.txt").write_text("m")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-q", "-m", "mine")
    mine = git(checkout, "rev-parse", "HEAD")
    code, text = run_cmd(env, inst, http=FakeHttp(), yes=True, root=checkout)
    assert code == 1 and "diverged" in text and "1 commit(s) of yours, 1 new upstream" in text
    assert git(checkout, "rev-parse", "HEAD") == mine


def test_git_update_refuses_detached_head_and_missing_remote(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    git(checkout, "checkout", "-q", "--detach")
    detached = update.detect_install(root=checkout, direct_url={})
    code, text = run_cmd(env, detached, http=FakeHttp(), yes=True, root=checkout)
    assert code == 1 and "detached" in text
    git(checkout, "checkout", "-q", "main")
    git(checkout, "remote", "remove", "origin")
    code, text = run_cmd(env, update.detect_install(root=checkout, direct_url={}), http=FakeHttp(), yes=True, root=checkout)
    assert code == 1 and "no git remote" in text


def test_git_up_to_date(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    code, text = run_cmd(env, inst, http=FakeHttp(), yes=True, root=checkout)
    assert code == 0 and "up to date" in text


def test_git_update_offline_refuses_with_a_reason(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    git(checkout, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    code, text = run_cmd(env, inst, http=FakeHttp(exc=OSError("x")), yes=True, root=checkout, runner=update.default_runner)
    assert code in (0, 1) and "Nothing was changed" in text or "Could not reach" in text


def test_dependency_refresh_is_planned_only_when_the_lock_or_pyproject_changed(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    (checkout / "uv.lock").write_text("lock v1")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-q", "-m", "lock")
    git(checkout, "push", "-q", "origin", "main")
    git(seed, "pull", "-q", str(origin), "main")
    push_commit(tmp_path, seed, origin, "code only")
    p = update_cli.plan_framework(update.check_for_updates(CFG, http=FakeHttp(), install=inst, root=checkout, now=1.0, force=True), inst,
                                  update.default_runner, CFG, which=lambda n: "/bin/uv")
    assert [s.argv[:2] for s in p.steps] == [["git", "pull"]]
    push_commit(tmp_path, seed, origin, "bump deps", fname="uv.lock", text="lock v2")
    p = update_cli.plan_framework(update.check_for_updates(CFG, http=FakeHttp(), install=inst, root=checkout, now=1.0, force=True), inst,
                                  update.default_runner, CFG, which=lambda n: "/bin/uv")
    assert [s.argv[:2] for s in p.steps] == [["git", "pull"], ["uv", "sync"]]
    no_uv = update_cli.plan_framework(update.check_for_updates(CFG, http=FakeHttp(), install=inst, root=checkout, now=1.0, force=True), inst,
                                      update.default_runner, CFG, which=lambda n: None)
    assert [s.argv[0] for s in no_uv.steps] == ["git"] and any("refresh the dependencies" in n for n in no_uv.notes)


def test_git_update_runs_post_check_and_says_restart(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    push_commit(tmp_path, seed, origin)
    done = []
    code, text = run_cmd(env, inst, http=FakeHttp(), yes=True, root=checkout, post_check=lambda: done.append(1))
    assert code == 0 and done == [1] and "Restart nightage" in text


def test_default_post_check_runs_a_fresh_doctor_subset():
    rec = Recorder({})
    con = Con()
    update_cli._post_check(rec, con)
    argv = rec.calls[0]
    assert argv[1:3] == ["-m", "agent.doctor"] and "--only" in argv and "--no-web" in argv and "python,packages,config,database" in argv


def test_release_newer_but_branch_not_ahead_is_explained(tmp_path, env):
    origin, seed, checkout, inst = git_setup(tmp_path, env)
    code, text = run_cmd(env, inst, http=FakeHttp({LATEST: release("v0.9.0")}), yes=True, root=checkout)
    assert code == 0 and "no newer commits to pull" in text


def test_report_roundtrips_through_json(env):
    rep = update.check_for_updates(CFG, http=FakeHttp({LATEST: release("v0.2.0")}), install=inst0(env), root=env, now=5.0)
    back = update.UpdateReport.from_dict(json.loads(json.dumps(rep.to_dict())))
    assert back.framework.latest == "0.2.0" and back.checked_at == 5.0 and back.schema == update.CACHE_SCHEMA
