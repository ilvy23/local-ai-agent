"""`nightage doctor` as a command: options, output at 40/60/100 columns, JSON, --list, --fix confirmations, --report anonymisation."""

import io
import json
import stat

import pytest

from agent import sysinfo
from agent.doctor import anonymise as an
from agent.doctor import cli, core, fixes
from agent.doctor import report as rpt
from agent.doctor.probes import OllamaInfo
from agent.ui import style
from tests._doctorfake import CFG, GB, UP, FakeProbes, G


@pytest.fixture(autouse=True)
def _plugins_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path / "plugins"))


def go(tmp_path, width=100, **kw):
    out, err = io.StringIO(), io.StringIO()
    con = style.make_console(width=width, plain=True, file=out)
    probes = kw.pop("probes", None) or FakeProbes(tmp_path, ollama=kw.pop("ollama", UP), gpus=kw.pop("gpus", [G()]))
    code = cli.run(kw.pop("config", CFG), probes=probes, console=con, plain=True, width=width, out=out, err=err, no_web=kw.pop("no_web", True), **kw)
    return code, out.getvalue(), err.getvalue()


def test_default_run_is_grouped_with_a_summary_line(tmp_path):
    code, out, _ = go(tmp_path)
    for header in ("System", "Python", "Ollama", "Models", "GPU", "Memory", "Disk", "Configuration", "Security"):
        assert f"\n{header}\n" in out
    assert "ok," in out and "warning(s)" in out and "problem(s)" in out
    assert code in (0, 1)


@pytest.mark.parametrize("width", [40, 60, 100])
def test_output_fits_and_is_ascii_in_plain_mode(tmp_path, width):
    code, out, _ = go(tmp_path, width=width, ollama=OllamaInfo(error="x"), verbose=True)
    assert out.isascii()
    assert max(len(line) for line in out.splitlines()) <= width, max(out.splitlines(), key=len)
    assert "problem(s)" in out


def test_only_and_skip(tmp_path):
    _, out, _ = go(tmp_path, only="gpu,memory")
    assert "\nGPU\n" in out and "\nMemory\n" in out and "\nOllama\n" not in out and "\nDisk\n" not in out
    _, out, _ = go(tmp_path, skip="gpu,memory,system,python,packages")
    assert "\nGPU\n" not in out and "\nMemory\n" not in out and "\nOllama\n" in out
    _, out, _ = go(tmp_path, only="disk.data")
    assert "Space for your data" in out and "Space for models" not in out


def test_list_shows_every_check_and_system(tmp_path):
    code, out, _ = go(tmp_path, list_checks=True)
    assert code == 0 and "nixos.nixld" in out and "termux.battery" in out and "deep.generate" in out
    assert "deep" in out and "macos" in out
    _, js, _ = go(tmp_path, list_checks=True, as_json=True)
    rows = json.loads(js)
    assert {r["id"] for r in rows} >= {"python", "gpu.nvidia", "linux.units", "windows.wsl", "deep.recommend"}
    assert next(r for r in rows if r["id"] == "nixos.nixld")["systems"] == ["nixos"]
    assert next(r for r in rows if r["id"] == "deep.generate")["deep"] is True


def test_json_shape_and_statuses(tmp_path):
    code, out, _ = go(tmp_path, as_json=True, ollama=OllamaInfo(error="x"))
    data = json.loads(out)
    assert set(data) >= {"product", "status", "ok", "summary", "meta", "checks"}
    assert data["ok"] is False and code == 1 and data["summary"]["fail"] >= 1
    assert all({"id", "group", "title", "status", "detail", "hint", "command", "ms"} <= set(c) for c in data["checks"])
    assert {c["status"] for c in data["checks"]} <= {"ok", "warn", "fail", "info", "skip"}
    ids = [c["id"] for c in data["checks"]]
    assert len(ids) == len(set(ids))
    nohint = json.loads(go(tmp_path, as_json=True, no_hints=True)[1])
    assert all("hint" not in c for c in nohint["checks"])


def test_exit_codes(tmp_path):
    good = FakeProbes(tmp_path, ollama=UP, gpus=[G()], env={"TERM": "xterm-256color", "COLORTERM": "truecolor"})
    only_warn = go(tmp_path, only="python,gpu", probes=good)
    assert only_warn[0] == 0
    warn_probe = FakeProbes(tmp_path, ollama=UP, gpus=[], si=sysinfo.SysInfo(ram_total_mb=16 * 1024, ram_free_mb=100, os_family="linux"))
    assert go(tmp_path, only="memory.ram", probes=warn_probe)[0] == 0
    assert go(tmp_path, only="memory.ram", probes=warn_probe, strict=True)[0] == 2
    assert go(tmp_path, only="ollama", ollama=OllamaInfo(error="x"))[0] == 1
    assert go(tmp_path, only="ollama", ollama=OllamaInfo(error="x"), strict=True)[0] == 1


def test_no_web_skips_network_and_says_so(tmp_path):
    _, out, _ = go(tmp_path, only="network", no_web=True)
    assert out.count("skipped (--no-web)") == 2
    _, out2, _ = go(tmp_path, only="network", no_web=False)
    assert "ollama.com" in out2 and "skipped (--no-web)" not in out2


def test_deep_is_labelled_and_off_by_default(tmp_path):
    _, plain, _ = go(tmp_path)
    assert "Deep tests" not in plain
    _, deep, _ = go(tmp_path, deep=True)
    assert "Deep tests use your model" in deep and "Deep tests (use the model)" in deep
    assert "Recommended model for this machine" in deep and "ollama pull qwen3.5:" in deep


def test_every_non_deep_check_runs_on_the_fake_machine_without_crashing(tmp_path):
    _, out, _ = go(tmp_path, as_json=True, deep=True)
    data = json.loads(out)
    could_not = [c for c in data["checks"] if "could not run" in c["detail"]]
    assert could_not == [], could_not


# ----------------------------------------------------------------------------- --fix


class RecordingConfirm:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def __call__(self, q):
        self.asked.append(q)
        return self.answers.pop(0) if self.answers else False


def fix_probes(tmp_path):
    cfgdir = tmp_path / "home" / ".config" / "agent"
    cfgdir.mkdir(parents=True)
    tok = cfgdir / "web-token"
    tok.write_text("x")
    tok.chmod(0o666)
    return FakeProbes(tmp_path, ollama=UP, gpus=[G()]), tok


def test_fix_never_runs_without_a_yes(tmp_path):
    probes, tok = fix_probes(tmp_path)
    conf = RecordingConfirm([])                                  # every answer: no
    code, out, _ = go(tmp_path, only="filesystem", probes=probes, fix=True, confirm=conf)
    assert conf.asked and "command: chmod 600" in out and "skipped." in out
    assert stat.S_IMODE(tok.stat().st_mode) == 0o666             # untouched


def test_fix_runs_exactly_what_was_shown_after_a_yes(tmp_path):
    probes, tok = fix_probes(tmp_path)
    conf = RecordingConfirm([True] * 5)
    _, out, _ = go(tmp_path, only="filesystem", probes=probes, fix=True, confirm=conf)
    assert stat.S_IMODE(tok.stat().st_mode) == 0o600
    assert "command: chmod 600" in out and "After the fixes" in out


def test_fix_default_confirm_is_no_without_a_terminal(tmp_path, monkeypatch):
    probes, tok = fix_probes(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))         # not a tty: even a typed y must not count
    go(tmp_path, only="filesystem", probes=probes, fix=True)
    assert stat.S_IMODE(tok.stat().st_mode) == 0o666


def test_fix_is_ignored_with_json(tmp_path):
    probes, tok = fix_probes(tmp_path)
    conf = RecordingConfirm([True])
    _, _, err = go(tmp_path, only="filesystem", probes=probes, fix=True, as_json=True, confirm=conf)
    assert not conf.asked and "ignored" in err and stat.S_IMODE(tok.stat().st_mode) == 0o666


def test_missing_model_fix_is_ollama_pull_only(tmp_path):
    res = core.run_checks(core.Context(CFG, FakeProbes(tmp_path, ollama=OllamaInfo(True, "1", {}, url="http://localhost:11434"))),
                          [core.REGISTRY["models"]])
    chat = next(r for r in res if r.id == "model_chat")
    assert chat.command == "ollama pull qwen3.5:9b"


@pytest.mark.parametrize("argv", [
    ["sudo", "ollama", "pull", "x"], ["ollama", "pull", "x; rm -rf /"], ["ollama", "serve"], ["ollama", "pull", "--insecure"],
    ["chmod", "777", "/tmp/x"], ["chmod", "600", "/etc/shadow"], ["mkdir", "-p", "/etc/x"], ["rm", "-rf", "x"], ["bash", "-c", "x"],
    ["systemctl", "restart", "x"], ["apt", "install", "x"], ["chmod", "600"], [],
])
def test_fixes_refuse_anything_not_user_level(argv, tmp_path):
    with pytest.raises(fixes.FixRefused):
        fixes.user_command(argv, home=tmp_path)


def test_fix_chmod_stays_inside_home(tmp_path):
    inside = tmp_path / "f"
    inside.write_text("x")
    cmd, fn = fixes.user_command(["chmod", "600", str(inside)], home=tmp_path)
    assert cmd == f"chmod 600 {inside}" and fn() == "done"
    link = tmp_path / "link"
    link.symlink_to("/etc/passwd")
    with pytest.raises(fixes.FixRefused):
        fixes.user_command(["chmod", "600", str(link)], home=tmp_path)           # a symlink out of the home is refused


def test_set_config_key_edits_one_line_and_keeps_a_backup(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("# my comment\nprofile: default\nmodels:\n  chat: x\n")
    shown, fn = fixes.set_config_key(cfg, "profile", "phone")
    assert "profile: phone" in shown and "wrote" in fn()
    assert cfg.read_text() == "# my comment\nprofile: phone\nmodels:\n  chat: x\n"
    assert (tmp_path / "config.yaml.bak").read_text().count("profile: default") == 1
    with pytest.raises(fixes.FixRefused):
        fixes.set_config_key(cfg, "Bad Key", "x")


def test_profile_fix_is_offered_on_a_small_phone(tmp_path):
    phone = sysinfo.SysInfo(os_family="android-termux", is_termux=True, ram_total_mb=3000, ram_free_mb=1000)
    res = core.run_checks(core.Context(CFG, FakeProbes(tmp_path, si=phone)), [core.REGISTRY["profile"]])[0]
    assert res.status == "warn" and "profile: phone" in res.command and res.fix is not None


# ----------------------------------------------------------------------------- --report / anonymisation

IDENT = an.Identity(users=["alice"], hosts=["alicebox"], homes=["/home/alice"])


def test_anonymise_replaces_everything_personal():
    text = ("user alice on alicebox, home /home/alice/proj and /home/bob/x and C:\\Users\\Carol\\a, ip 192.168.1.20 and 100.64.0.8, "
            "v6 fe80::1ff:fe23:4567:890a, mac aa:bb:cc:dd:ee:ff, mail alice@example.com, tok ghp_" + "a" * 36 +
            ", url https://alice:hunter2@git.example.com/x, host alicebox.tail1234.ts.net, key sk-" + "b" * 30 + ", local 127.0.0.1")
    out = an.anonymise(text, IDENT)
    for secret in ("alice", "alicebox", "bob", "Carol", "192.168.1.20", "100.64.0.8", "fe80", "aa:bb:cc", "@example.com", "ghp_", "hunter2",
                   "tail1234", "sk-"):
        assert secret not in out, secret
    assert "~/proj" in out and "<ip>" in out and "<ip6>" in out and "<mac>" in out and "<email>" in out and "<host>" in out
    assert "127.0.0.1" in out                                     # loopback is harmless and useful
    assert an.leak_check(out, IDENT) == []


def test_leak_check_finds_what_anonymise_missed():
    assert "the user name" in an.leak_check("hello alice", IDENT)
    assert "the host name" in an.leak_check("on ALICEBOX now", IDENT)
    assert "an IPv4 address" in an.leak_check("ip 10.1.2.3", IDENT)
    assert "an e-mail address" in an.leak_check("someone@example.com", IDENT)
    assert "a token- or key-shaped string" in an.leak_check("sk-" + "c" * 20, IDENT)
    assert "an absolute home path" in an.leak_check("/home/user/x", IDENT)
    assert an.leak_check("Linux 6.1 x86_64, 16 GB RAM, ollama 0.6.1, qwen3.5:9b, 127.0.0.1, version 550.54.14", IDENT) == []


def test_generic_hostnames_are_not_treated_as_secrets():
    ident = an.Identity(users=[], hosts=["nixos"], homes=[])
    assert "nixos" in an.anonymise("Linux (nixos)", ident)           # the distro name must survive


def report_run(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(cli.Identity, "current", classmethod(lambda cls: IDENT))
    remote = OllamaInfo(True, "0.6.1", {"qwen3.5:9b": 1}, url="http://192.168.1.20:11434")
    probes = FakeProbes(tmp_path, ollama=remote, env={"HOSTNAME": "alicebox"})
    return go(tmp_path, probes=probes, make_report=True, **kw)


def test_report_is_anonymised_and_self_checked(tmp_path, monkeypatch):
    code, out, err = report_run(tmp_path, monkeypatch)
    assert out.startswith("# Nightage doctor report") and "## Environment" in out and "| status | check | detail |" in out
    assert "192.168.1.20" not in out and "alice" not in out and "<ip>" in out
    assert "self-checked" in err
    assert an.leak_check(out, IDENT) == []


def test_report_refuses_when_the_self_check_finds_something(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "anonymise", lambda text, ident=None: text)           # a broken anonymiser must not leak
    code, out, err = report_run(tmp_path, monkeypatch)
    assert code == 3 and out == "" and "NOT printed" in err and "an IPv4 address" in err
    assert "192.168.1.20" not in err                                               # the message names the kind, never the value


def test_report_has_no_config_values_or_chat_content(tmp_path, monkeypatch):
    cfg = {"models": {"chat": "qwen3.5:9b"}, "persona": {"name": "SECRET-PERSONA"}, "whatsapp": {"api_token": "tokentokentoken"}}
    monkeypatch.setattr(cli.Identity, "current", classmethod(lambda cls: IDENT))
    _, out, _ = go(tmp_path, config=cfg, make_report=True)
    assert "SECRET-PERSONA" not in out and "tokentokentoken" not in out


def test_verbose_shows_ids_and_timings(tmp_path):
    _, out, _ = go(tmp_path, only="python", verbose=True)
    assert "[python," in out and " ms]" in out


def test_update_note_appears_in_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.update.notice_line", lambda config=None, **kw: "Update available: 9.9.9 (you have 0.1.0) - run `nightage update`")
    _, out, _ = go(tmp_path, only="python")
    assert "Update available: 9.9.9" in out


def test_render_survives_empty_report():
    buf = io.StringIO()
    rpt.render(rpt.Report(), style.make_console(width=40, plain=True, file=buf), plain=True)
    assert "0 ok" in buf.getvalue()
