"""Plugin updates: install record, update check, changelog + file-level summary, and the trust model (updated code is not trusted).
Real git against throw-away local repositories; no network."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from agent import plugin_host, plugin_update, update

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=GIT_ENV)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


class Con:
    def __init__(self):
        self.lines = []

    def print(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))

    @property
    def text(self):
        return "\n".join(self.lines)


@pytest.fixture
def world(tmp_path, monkeypatch):
    pdir = tmp_path / "plugins"
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(pdir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    (seed / "plugin.toml").write_text('name = "remote-plug"\nversion = "0.1.0"\ndescription = "d"\n')
    (seed / "plugin.py").write_text("def register(api):\n    pass\n")
    (seed / "old.py").write_text("x = 1\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "first")
    origin = tmp_path / "plug.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(origin))
    url = f"file://{origin}"

    class W:
        pass

    w = W()
    w.seed, w.origin, w.url, w.pdir, w.tmp = seed, origin, url, pdir, tmp_path

    def install(**kw):
        return plugin_host.install(url, confirm=lambda q: True, **kw)

    def push(msg="second", version="0.2.0"):
        (seed / "plugin.toml").write_text(f'name = "remote-plug"\nversion = "{version}"\ndescription = "d"\n')
        (seed / "plugin.py").write_text("def register(api):\n    api.log.info('v2')\n    pass\n")
        (seed / "new.py").write_text("a = 1\nb = 2\n")
        (seed / "old.py").unlink(missing_ok=True)
        git(seed, "add", "-A")
        git(seed, "commit", "-q", "-m", msg)
        git(seed, "push", "-q", str(origin), "main")
        return git(seed, "rev-parse", "HEAD")

    w.install, w.push = install, push
    return w


def run(names=None, **kw):
    con = Con()
    code = plugin_update.run_plugin_update(names if names is not None else ["remote-plug"], console=con, **kw)
    return code, con.text


def test_git_install_leaves_a_record(world):
    m = world.install()
    rec = json.loads((m.path / ".source.json").read_text())
    assert rec["type"] == "git" and rec["url"] == world.url and rec["commit"] == git(world.seed, "rev-parse", "HEAD")
    assert rec["branch"] == "main" and rec["installed_at"].endswith("Z") and rec["ref"] == ""
    assert not (m.path / ".git").exists()


def test_record_does_not_change_the_fingerprint(world):
    m = world.install()
    fp = plugin_host.fingerprint(m)
    (m.path / ".source.json").write_text(json.dumps({"url": "other", "commit": "zzz"}))
    assert plugin_host.fingerprint(m) == fp


def test_install_with_ref_and_manifest_keys(world):
    git(world.seed, "tag", "v1")
    git(world.seed, "push", "-q", str(world.origin), "v1")
    m = world.install(ref="v1")
    assert plugin_update.read_source(m.path)["ref"] == "v1"
    (m.path / "plugin.toml").write_text(m.path.joinpath("plugin.toml").read_text() + 'repository = "https://example.com/x/y.git"\n')
    assert plugin_host.read_manifest(m.path).repository == "https://example.com/x/y.git"


def test_check_all_reports_available_only_after_a_new_commit(world):
    world.install()
    assert [(p.name, p.available) for p in plugin_update.check_all({})] == [("remote-plug", False)]
    new = world.push()
    (pu,) = plugin_update.check_all({})
    assert pu.available and pu.remote_commit == new and pu.url == world.url


def test_local_installs_have_no_update_source(world, tmp_path):
    src = plugin_host.scaffold("local-one", tmp_path / "src")
    plugin_host.install(str(src), confirm=lambda q: True)
    (pu,) = plugin_update.check_all({})
    assert not pu.available and "no update source" in pu.note
    code, text = run(["local-one"])
    assert code == 1 and "no update source" in text


def test_check_only_changes_nothing(world):
    m = world.install()
    world.push()
    before = plugin_host.fingerprint(m)
    code, text = run(check_only=True)
    assert code == 0 and "update available" in text
    assert plugin_host.fingerprint(plugin_host.find("remote-plug")) == before


def test_offline_source_is_a_quiet_note(world):
    world.install()
    subprocess.run(["rm", "-rf", str(world.origin)])
    (pu,) = plugin_update.check_all({})
    assert not pu.available and pu.note.startswith("offline")


def test_plan_shows_changelog_and_file_level_summary_without_touching_the_plugin(world):
    m = world.install()
    world.push("feat: do the new thing")
    before = plugin_host.fingerprint(m)
    plan = plugin_update.plan_update("remote-plug")
    try:
        assert not plan.problem and plan.old_version == "0.1.0" and plan.new_version == "0.2.0"
        assert any("feat: do the new thing" in line for line in plan.changelog)
        by = {c.path: c for c in plan.changes}
        assert by["new.py"].status == "added" and by["new.py"].plus == 2
        assert by["old.py"].status == "removed" and by["old.py"].minus == 1
        assert by["plugin.py"].status == "changed" and by["plugin.py"].plus >= 1
        assert by["plugin.toml"].status == "changed" and ".source.json" not in by
        assert any(line.strip().startswith("+ new.py") for line in plan.summary_lines())
    finally:
        plan.cleanup()
    assert plugin_host.fingerprint(plugin_host.find("remote-plug")) == before


def test_update_asks_before_replacing(world):
    m = world.install()
    world.push()
    code, text = run(confirm=lambda q: False)
    assert "Not updated." in text and plugin_host.fingerprint(plugin_host.find("remote-plug")) == plugin_host.fingerprint(m) and (m.path / "old.py").exists()


def test_updated_enabled_plugin_is_switched_off_and_needs_the_enable_flow(world):
    m = world.install()
    plugin_host.set_enabled("remote-plug", True)
    old_fp = plugin_host.fingerprint(m)
    new = world.push()
    enable_calls = []
    code, text = run(confirm=lambda q: True, enable_fn=lambda n: enable_calls.append(n) or False)
    assert code == 0 and enable_calls == ["remote-plug"]
    assert not plugin_host.is_enabled("remote-plug")                          # declined: stays off
    now = plugin_host.find("remote-plug")
    assert now.version == "0.2.0" and plugin_host.fingerprint(now) != old_fp and plugin_host.fingerprint(now) in text
    assert "switched OFF" in text and "updated code is new code" in text and "Restart nightage" in text
    assert plugin_update.read_source(now.path)["commit"] == new
    backups = list((world.pdir / ".backups").iterdir())
    assert len(backups) == 1 and (backups[0] / "old.py").exists()
    assert plugin_host.discover() and [x.id for x in plugin_host.discover()] == ["remote-plug"]      # backups are not plugins


def test_yes_skips_the_update_question_but_never_the_enable_question(world):
    world.install()
    plugin_host.set_enabled("remote-plug", True)
    world.push()
    called = []
    code, text = run(yes=True, enable_fn=lambda n: called.append(n) or True)
    assert code == 0 and called == [] and not plugin_host.is_enabled("remote-plug")
    assert "stays off" in text and "nightage plugin enable remote-plug" in text


def test_default_enable_flow_uses_the_real_confirmation(world, monkeypatch):
    import typer

    world.install()
    plugin_host.set_enabled("remote-plug", True)
    world.push()
    asked = []
    monkeypatch.setattr(typer, "confirm", lambda q, default=False, **k: asked.append((q, default)) or False)
    code, text = run(confirm=lambda q: True)
    assert code == 0 and not plugin_host.is_enabled("remote-plug") and asked == [("Enable remote-plug?", False)]
    world.push("third", "0.3.0")
    plugin_host.set_enabled("remote-plug", True)
    monkeypatch.setattr(typer, "confirm", lambda q, default=False, **k: True)
    code, text = run(confirm=lambda q: True)
    assert plugin_host.is_enabled("remote-plug")


def test_a_plugin_that_was_off_stays_off(world):
    world.install()
    world.push()
    called = []
    code, text = run(confirm=lambda q: True, enable_fn=lambda n: called.append(n) or True)
    assert code == 0 and called == [] and not plugin_host.is_enabled("remote-plug") and "stays that way" in text


def test_up_to_date_plugin_is_not_touched(world):
    m = world.install()
    code, text = run(confirm=lambda q: pytest.fail("must not ask"))
    assert code == 0 and "up to date" in text and not (world.pdir / ".backups").exists()
    assert plugin_host.find("remote-plug").path == m.path


def test_update_refuses_a_source_that_renamed_the_plugin(world):
    world.install()
    (world.seed / "plugin.toml").write_text('name = "evil-plug"\nversion = "9.0.0"\n')
    git(world.seed, "add", "-A")
    git(world.seed, "commit", "-q", "-m", "rename")
    git(world.seed, "push", "-q", str(world.origin), "main")
    code, text = run(confirm=lambda q: True)
    assert code == 1 and "refusing to replace one plugin with another" in text
    assert plugin_host.find("remote-plug").version == "0.1.0"


def test_update_refuses_a_source_without_a_manifest(world):
    world.install()
    git(world.seed, "rm", "-q", "plugin.toml")
    git(world.seed, "commit", "-q", "-m", "drop manifest")
    git(world.seed, "push", "-q", str(world.origin), "main")
    code, text = run(confirm=lambda q: True)
    assert code == 1 and "not a nightage plugin" in text


def test_failed_copy_restores_the_old_plugin(world, monkeypatch):
    m = world.install()
    world.push()
    plan = plugin_update.plan_update("remote-plug")
    monkeypatch.setattr(plugin_update.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        plugin_update.apply_plan(plan)
    plan.cleanup()
    assert (m.path / "old.py").exists() and plugin_host.find("remote-plug").version == "0.1.0"


def test_manifest_repository_allows_updates_without_a_record(world):
    m = world.install()
    (m.path / ".source.json").unlink()
    (m.path / "plugin.toml").write_text(f'name = "remote-plug"\nversion = "0.1.0"\nrepository = "{world.url}"\n')
    world.push()
    (pu,) = plugin_update.check_all({})
    assert not pu.available and "installed commit was not recorded" in pu.note
    code, text = run(confirm=lambda q: True, enable_fn=lambda n: False)
    assert code == 0 and plugin_host.find("remote-plug").version == "0.2.0"
    assert plugin_update.read_source(plugin_host.find("remote-plug").path)["url"] == world.url


def test_install_force_switches_an_enabled_plugin_off(world, tmp_path):
    world.install()
    plugin_host.set_enabled("remote-plug", True)
    world.install(force=True)
    assert not plugin_host.is_enabled("remote-plug")


def test_unknown_names_and_missing_arguments(world):
    code, text = run(["nope"])
    assert code == 1 and "No plugin called" in text
    code, text = run([])
    assert code == 1 and "Name a plugin" in text


def test_cache_notice_tracks_installed_commit(world):
    world.install()
    world.push()
    rep = update.UpdateReport(checked_at=__import__("time").time(), framework=update.FrameworkUpdate(current="0.1.0"),
                              plugins=plugin_update.check_all({}))
    update._save_cache(update.cache_path(), rep)
    assert update.notice_lines({}) == ["1 plugin update - `nightage plugin update`"]
    run(confirm=lambda q: True, enable_fn=lambda n: False)
    assert update.notice_lines({}) == []


def test_cli_command_is_registered():
    from typer.testing import CliRunner

    from agent.plugin_cli import plugin_app

    r = CliRunner().invoke(plugin_app, ["update", "--help"])
    assert r.exit_code == 0 and "--check" in r.output and "--yes" in r.output and "--all" in r.output
    assert Path(plugin_update.__file__).exists()
