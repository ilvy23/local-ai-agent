import io
import json
import os
from pathlib import Path

import pytest

from agent.ui import doctor as D
from agent.ui import style


class G:
    def __init__(self, name="RX 7900", total_mb=24576):
        self.name, self.total_mb = name, total_mb


class FakeProbes(D.Probes):
    def __init__(self, tmp_path, *, ollama=None, gpus=None, writable=True, cfg_err="", cfgfile=None, web=True, pkgs_missing=(),
                 py=(3, 12, 1), tty=True, width=100):
        super().__init__()
        self.tmp, self._ollama, self._gpus, self._writable = tmp_path, ollama, gpus if gpus is not None else [G()], writable
        self._cfg_err, self._cfg, self._web, self._missing, self._py = cfg_err, cfgfile or {}, web, set(pkgs_missing), py
        self._tty, self._width = tty, width

    def python_version(self): return self._py
    def package_available(self, module): return module not in self._missing
    def ollama(self): return self._ollama or D.OllamaInfo()
    def gpus(self): return self._gpus
    def gpu_mode(self, gpus): return "single" if gpus else None
    def ctx_budget(self, model, size, gpus, config):
        return {"override": None, "computed": 32768, "final": 32768, "source": "vram", "ceiling": 95000}
    def data_paths(self, config): return self.tmp, self.tmp / "agent.db"
    def writable(self, path): return self._writable
    def config_file(self): return self.tmp / "config.yaml", (None if self._cfg_err else self._cfg), self._cfg_err
    def web_reachable(self, timeout): return self._web, "html.duckduckgo.com"
    def sysinfo(self):
        from agent import sysinfo
        return sysinfo.SysInfo(ram_total_mb=16 * 1024, ram_free_mb=8192, cpu_count=8)
    def storage_shared(self): return True
    def remote_latency(self, url): return True, 42.0, ""
    def terminal(self): return {"width": self._width, "tty": self._tty, "color": self._tty, "unicode": True, "term": "xterm"}


CFG = {"models": {"chat": "qwen3.5:9b", "coding": "qwen3-coder:30b", "embed": "bge-m3"}}
UP = D.OllamaInfo(True, "0.6.1", {"qwen3.5:9b": 6 * 1024**3, "qwen3-coder:30b": 18 * 1024**3, "bge-m3:latest": 1024**3})


def run(tmp_path, cfg=CFG, **kw):
    return D.collect_checks(cfg, probes=FakeProbes(tmp_path, **kw))


def test_all_ok(tmp_path):
    rep = run(tmp_path, ollama=UP)
    assert rep.ok and rep.exit_code == 0 and rep.status == "ok"
    assert [c.status for c in rep.checks if c.id.startswith("model_")] == ["ok"] * 3
    assert "32768" in rep.get("context").detail


def test_missing_chat_model_gives_exact_pull_command(tmp_path):
    info = D.OllamaInfo(True, "0.6.1", {"qwen3-coder:30b": 1, "bge-m3": 1})
    rep = run(tmp_path, ollama=info)
    c = rep.get("model_chat")
    assert c.status == "fail" and c.fix == "Run `ollama pull qwen3.5:9b`."
    assert not rep.ok and rep.exit_code == 1


def test_missing_coding_model_is_only_a_warning(tmp_path):
    info = D.OllamaInfo(True, "1", {"qwen3.5:9b": 1, "bge-m3": 1})
    rep = run(tmp_path, ollama=info)
    assert rep.get("model_coding").status == "warn" and rep.ok


def test_no_ollama(tmp_path):
    rep = run(tmp_path, ollama=D.OllamaInfo(error="ConnectError"))
    c = rep.get("ollama")
    assert c.status == "fail" and "ollama serve" in c.fix
    assert rep.get("model_chat").status == "warn"          # cannot verify, not a second failure


def test_read_only_data_dir(tmp_path):
    rep = run(tmp_path, ollama=UP, writable=False)
    c = rep.get("data")
    assert c.status == "fail" and "chmod" in c.fix


def test_real_writable_probe(tmp_path):
    p = D.Probes()
    assert p.writable(tmp_path / "does" / "not" / "exist")
    if os.geteuid() != 0:
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        try:
            assert not p.writable(ro)
        finally:
            ro.chmod(0o755)


def test_python_packages_config_gpu_web_terminal(tmp_path):
    rep = run(tmp_path, ollama=UP, py=(3, 10, 0), pkgs_missing=("rich", "sqlite_vec"), gpus=[], cfg_err="missing", web=False, tty=False)
    assert rep.get("python").status == "fail"
    assert rep.get("packages").status == "fail" and "rich" in rep.get("packages").detail
    assert rep.get("gpu").status == "warn"
    assert rep.get("config").status == "warn"
    assert rep.get("web").status == "warn"
    assert rep.get("terminal").status == "warn"
    only_opt = run(tmp_path, ollama=UP, pkgs_missing=("sqlite_vec",))
    assert only_opt.get("packages").status == "warn"


def test_bad_config_and_unsafe_safety_keys(tmp_path):
    assert run(tmp_path, ollama=UP, cfg_err="unreadable: boom").get("config").status == "fail"
    bad = {"safety": {"safe_commands": "ls", "max_timeout_s": -1}}
    c = run(tmp_path, ollama=UP, cfgfile=bad).get("config")
    assert c.status == "warn" and "safe_commands" in c.detail and "max_timeout_s" in c.detail
    assert run(tmp_path, ollama=UP, cfgfile={"safety": {"safe_commands": ["ls"], "blocked_patterns": [], "max_timeout_s": 60}}).get("config").status == "ok"


def test_web_check_is_skippable(tmp_path):
    class Boom(FakeProbes):
        def web_reachable(self, timeout): raise AssertionError("must not be called")
    rep = D.collect_checks(CFG, probes=Boom(tmp_path, ollama=UP), check_web=False)
    assert rep.get("web").detail == "skipped"


def test_json_shape(tmp_path):
    buf = io.StringIO()
    con = style.make_console(width=100, plain=True, file=buf)
    rep = D.run_doctor(CFG, as_json=True, probes=FakeProbes(tmp_path, ollama=D.OllamaInfo(error="x")), console=con)
    data = json.loads(buf.getvalue())
    assert set(data) >= {"product", "status", "ok", "summary", "checks"}
    assert data["ok"] is False and data["summary"]["fail"] >= 1
    assert all(set(c) == {"id", "title", "status", "detail", "fix"} for c in data["checks"])
    assert {c["status"] for c in data["checks"]} <= {"ok", "warn", "fail"}
    assert rep.exit_code == 1
    nofix = rep.to_dict(fix_hints=False)
    assert all("fix" not in c for c in nofix["checks"])


@pytest.mark.parametrize("width", [40, 60, 100])
def test_pretty_render_fits_and_is_ascii_in_plain(tmp_path, width):
    buf = io.StringIO()
    con = style.make_console(width=width, plain=True, file=buf)
    D.run_doctor(CFG, probes=FakeProbes(tmp_path, ollama=D.OllamaInfo(error="x")), console=con, plain=True)
    out = buf.getvalue()
    assert out.isascii() and "ollama serve" in out and "problem(s)" in out
    assert max(len(l) for l in out.splitlines()) <= width
    buf2 = io.StringIO()
    D.run_doctor(CFG, fix_hints=False, probes=FakeProbes(tmp_path, ollama=D.OllamaInfo(error="x")),
                 console=style.make_console(width=width, plain=True, file=buf2), plain=True)
    assert "fix:" not in buf2.getvalue()


def test_needs_onboarding(tmp_path):
    p = FakeProbes(tmp_path, ollama=UP)
    assert D.needs_onboarding(CFG, probes=p)                    # no db yet
    (tmp_path / "agent.db").write_text("x")
    assert not D.needs_onboarding(CFG, probes=p)
    assert D.needs_onboarding(CFG, probes=FakeProbes(tmp_path, ollama=D.OllamaInfo(True, "1", {})))   # up but no models
    assert not D.needs_onboarding(CFG, probes=FakeProbes(tmp_path, ollama=D.OllamaInfo()))             # down: not an onboarding case


def test_suggest_model():
    inst = {"bge-m3": 1, "llama3:8b": 5 * 1024**3, "big:70b": 40 * 1024**3, "mid:14b": 9 * 1024**3}
    assert D.suggest_model(inst, {"models": {"chat": "llama3:8b"}}) == "llama3:8b"     # configured one wins
    assert D.suggest_model(inst, {"models": {"chat": "nope"}}) == "mid:14b"            # largest that is not huge, not embedding
    assert D.suggest_model({"bge-m3": 1}, CFG) == ""
    assert D.suggest_model({"big:70b": 40 * 1024**3}, CFG) == "big:70b"


def test_onboarding_report(tmp_path):
    buf = io.StringIO()
    info = D.OllamaInfo(True, "1", {"llama3:8b": 5 * 1024**3, "bge-m3": 1})
    rep = D.run_onboarding(CFG, probes=FakeProbes(tmp_path, ollama=info), console=style.make_console(width=80, plain=True, file=buf), plain=True)
    out = buf.getvalue()
    assert rep.suggestion == "llama3:8b" and len(rep.examples) == 3
    assert "Welcome" in out and "Suggested model: llama3:8b" in out and "Try one of these" in out
    none = D.run_onboarding(CFG, probes=FakeProbes(tmp_path, ollama=D.OllamaInfo(True, "1", {})), print_report=False)
    assert none.suggestion == "" and "ollama pull qwen3.5:9b" in none.examples[0]


def test_model_installed_latest_tag():
    assert D.model_installed("bge-m3", {"bge-m3:latest"})
    assert not D.model_installed("qwen:7b", {"qwen:14b"})
    assert not D.model_installed("", {"x"})


def test_real_probes_do_not_hang_without_ollama(monkeypatch):
    monkeypatch.setattr("agent.llm.DEFAULT_BASE_URL", "http://127.0.0.1:9", raising=True)
    info = D.Probes(timeout=0.5).ollama()
    assert not info.reachable


# ----------------------------------------------------------------------------- platform / profile / remote rows


class _Phone(FakeProbes):
    def __init__(self, tmp_path, *, termux=True, ram=4096, storage=False, latency=(True, 80.0, ""), **kw):
        super().__init__(tmp_path, **kw)
        self._t, self._ram, self._st, self._lat = termux, ram, storage, latency

    def sysinfo(self):
        from agent import sysinfo
        return sysinfo.SysInfo(os_family="android-termux" if self._t else "linux", is_termux=self._t, ram_total_mb=self._ram,
                               ram_free_mb=1200, cpu_count=8)

    def storage_shared(self): return self._st
    def remote_latency(self, url): return self._lat


def test_platform_row_and_phone_suggestion(tmp_path):
    rep = D.collect_checks(CFG, probes=_Phone(tmp_path, ollama=UP))
    plat = rep.get("platform")
    assert plat.status == "ok" and "Termux" in plat.detail and "4 GB" in plat.detail
    prof = rep.get("profile")
    assert prof.status == "warn" and "--profile phone" in prof.fix
    assert rep.get("storage").status == "ok" and "termux-setup-storage" in rep.get("storage").detail


def test_no_suggestion_when_profile_chosen_or_big_machine(tmp_path):
    chosen = _Phone(tmp_path, ollama=UP, cfgfile={"profile": "phone"})
    assert D.collect_checks({**CFG, "profile": "phone"}, probes=chosen).get("profile").status == "ok"
    big = _Phone(tmp_path, termux=False, ram=32 * 1024, ollama=UP)
    rep = D.collect_checks(CFG, probes=big)
    assert rep.get("profile").status == "ok" and rep.get("storage") is None


def test_remote_latency_row(tmp_path):
    remote = D.OllamaInfo(True, "0.6.1", UP.models, url="http://100.64.0.5:11434")
    rep = D.collect_checks(CFG, probes=_Phone(tmp_path, ollama=remote, latency=(True, 85.0, "")))
    assert rep.get("remote_latency").status == "ok" and "85 ms" in rep.get("remote_latency").detail
    assert "remote running" in rep.get("ollama").detail
    slow = D.collect_checks(CFG, probes=_Phone(tmp_path, ollama=remote, latency=(True, 4000.0, "")))
    assert slow.get("remote_latency").status == "warn"
    local = D.collect_checks(CFG, probes=_Phone(tmp_path, ollama=D.OllamaInfo(True, "1", UP.models, url="http://localhost:11434")))
    assert local.get("remote_latency") is None


def test_unreachable_remote_gets_gateway_advice(tmp_path):
    down = D.OllamaInfo(False, "", {}, error="ConnectError", url="http://100.64.0.5:11434")
    c = D.collect_checks(CFG, probes=_Phone(tmp_path, ollama=down)).get("ollama")
    assert c.status == "fail" and "Tailscale" in c.fix and "100.64.0.5" in c.detail
    local = D.collect_checks(CFG, probes=_Phone(tmp_path, ollama=D.OllamaInfo(False, error="x"))).get("ollama")
    assert "pkg install ollama" in local.fix


def test_real_remote_latency_probe_never_raises():
    ok, ms, err = D.Probes(timeout=0.3).remote_latency("http://127.0.0.1:9")
    assert not ok and ms is None and err
