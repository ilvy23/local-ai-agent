"""Things that assume NixOS / systemd / a desktop: each must degrade with a message, not an exception, when the capability is missing."""
import builtins
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from agent import llm, panel, sysinfo
from agent.tools import registry as toolreg


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    yield
    llm.configure({})
    toolreg.configure({})


# ------------------------------------------------------------------ panel: /proc may be unreadable (Android blocks /proc/stat)

def test_panel_survives_unreadable_proc(monkeypatch):
    real_open = builtins.open

    def deny(path, *a, **k):
        if str(path).startswith("/proc/"):
            raise PermissionError(13, "denied", str(path))
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", deny)
    assert panel._cpu_times() == (0, 0)
    pct, _ = panel._cpu_percent((0, 0))
    assert pct == 0.0


def test_panel_mem_and_loadavg_without_proc(monkeypatch):
    monkeypatch.setattr(sysinfo, "read_meminfo", lambda *a, **k: {})
    assert panel._mem() == (0.0, 0.0)
    monkeypatch.setattr(panel.Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("no")))
    monkeypatch.setattr(panel.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError("no")))
    assert panel._loadavg() == "n/a"


def test_panel_mem_reads_fake_meminfo(monkeypatch):
    monkeypatch.setattr(sysinfo, "read_meminfo", lambda *a, **k: {"MemTotal": 4 * 1024 * 1024, "MemAvailable": 1024 * 1024})
    used, total = panel._mem()
    assert round(total) == 4 and round(used) == 3


# ------------------------------------------------------------------ host routes

def test_suspend_without_systemctl_is_a_clear_501(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from agent.web.routes import host
    monkeypatch.setattr(host.shutil, "which", lambda n: None)
    monkeypatch.setattr(host.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn anything"))
    with pytest.raises(HTTPException) as e:
        host.suspend_host()
    assert e.value.status_code == 501 and "systemd" in e.value.detail


def test_ops_systemctl_missing_returns_empty(monkeypatch):
    pytest.importorskip("fastapi")
    from agent.web.routes import ops
    monkeypatch.setattr(ops.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("systemctl")))
    assert ops._systemctl("is-active", "x") == ""


def test_pkill_missing_is_a_noop(monkeypatch):
    pytest.importorskip("fastapi")
    from agent.web.routes import whatsapp_persona as wp
    monkeypatch.setattr(wp.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("pkill")))
    wp._pkill("x")


def test_main_systemctl_user_missing(monkeypatch):
    from agent import main
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("systemctl")))
    assert main._systemctl_user("is-active", "x") == (1, "systemctl unavailable")


def test_vulkan_unit_absent_without_systemctl(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda n: None)
    assert llm._vulkan_unit_exists() is False
    monkeypatch.delenv("AGENT_VULKAN_START_CMD", raising=False)
    monkeypatch.setattr(llm.Path, "home", classmethod(lambda cls: cls("/nonexistent-home")))
    assert llm._vulkan_start_command() is None


def test_autopilot_inhibit_without_systemd_inhibit():
    from agent.codeagent import supervisor
    src = supervisor.Supervisor
    sup = src.__new__(src)
    sup.cfg, sup._which, sup._inhibitor = {}, (lambda n: None), None
    sup._popen = lambda *a, **k: pytest.fail("no inhibitor process without systemd-inhibit")
    sup._inhibit_start()
    assert sup._inhibitor is None


# ------------------------------------------------------------------ safety file stays additive

def test_repoaccess_prefixes_only_grew():
    from agent import repoaccess
    for old in ("/usr/", "/bin/", "/sbin/", "/lib/", "/nix/store/", "/run/current-system/", "/tmp/", "/dev/null", "/dev/stdout",
                "/dev/stderr", "/proc/cpuinfo", "/proc/meminfo"):
        assert old in repoaccess._HARMLESS_PREFIXES
    assert "/data/data/com.termux/files/usr/" in repoaccess._HARMLESS_PREFIXES
    assert "etc" in repoaccess._FORBIDDEN_TOP and "root" in repoaccess._FORBIDDEN_TOP


# ------------------------------------------------------------------ optional dependencies

def test_coverage_gate_skips_without_coverage(monkeypatch):
    import importlib.util
    from agent.coding.verify import guards
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "coverage" else real(name, *a))
    res = guards.coverage_gate(SimpleNamespace(path="."), ["x.py"])
    assert res.ok and "skipped" in res.note


def test_lint_skips_without_ruff(monkeypatch, tmp_path):
    from agent.coding.verify import checks
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: None)
    res = checks.check_lint(tmp_path)
    assert res.ok and "ruff not installed" in res.note


def test_keyword_recall_when_vectors_unavailable():
    from agent.memory import recall
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, content TEXT, active INTEGER DEFAULT 1)")
    con.executemany("INSERT INTO facts (content, active) VALUES (?, ?)", [
        ("The user's dog is called Fluffy", 1), ("Lives in Munich", 1), ("Old dog fact Fluffy", 0), ("Likes tea", 1)])
    store = SimpleNamespace(conn=con)
    vectors = SimpleNamespace(available=False)
    cfg = {"memory": {"recall_k": 3}, "models": {"embed": "x"}}
    hits = recall._semantic_hits(vectors, SimpleNamespace(embed=lambda *a, **k: pytest.fail("no embedding without vectors")), cfg,
                                 "What is my dog called, Fluffy?", store)
    assert hits == ["The user's dog is called Fluffy"]
    assert recall._semantic_hits(vectors, None, cfg, "hi", None) == []
    assert recall._semantic_hits(vectors, None, cfg, "ok", store) == []


# ------------------------------------------------------------------ remote gateway (ollama.host / OLLAMA_HOST)

def test_normalize_host():
    assert llm.normalize_host("100.64.0.5") == "http://100.64.0.5:11434"
    assert llm.normalize_host("pc.tailnet.ts.net:8080") == "http://pc.tailnet.ts.net:8080"
    assert llm.normalize_host("https://pc.example/") == "https://pc.example"
    assert llm.normalize_host("  ") == ""


def test_config_host_and_env_precedence(monkeypatch):
    llm.configure({"ollama": {"host": "100.64.0.5"}})
    assert llm.current_base_url() == "http://100.64.0.5:11434" and llm.is_remote()
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.9:1234")
    assert llm.current_base_url() == "http://10.0.0.9:1234"          # the environment wins
    monkeypatch.delenv("OLLAMA_HOST")
    llm.configure({})
    assert not llm.is_remote() and llm.current_base_url() == llm.DEFAULT_BASE_URL


def test_client_uses_configured_host_and_skips_vulkan(monkeypatch):
    llm.configure({"ollama": {"host": "100.64.0.5:11434", "keep_alive": "2m"}})
    client = llm.OllamaClient()
    assert client._base_url == "http://100.64.0.5:11434"
    assert llm.wants_vulkan("qwen3.5:27b") is False                   # the local Vulkan server is not the gateway's
    assert client.server_url("qwen3.5:27b") == "http://100.64.0.5:11434"
    assert llm.KEEP_ALIVE == "2m"
    llm.configure({})


def test_ctx_budget_ignores_local_ram_for_remote_server():
    from agent import ctxbudget
    hw = ctxbudget.HardwareSummary(gpus=[], total_ram_mb=1500)
    cfg = {"models": {"max_ctx_ceiling": 16384}}
    assert ctxbudget.budget_for("m", 9e9, hw, config=cfg) == 2048                       # local: tiny RAM squeezes it to the floor
    llm.configure({"ollama": {"host": "100.64.0.5"}})
    assert ctxbudget.budget_for("m", 9e9, hw, config=cfg) == 16384                      # remote: the phone's RAM is irrelevant
