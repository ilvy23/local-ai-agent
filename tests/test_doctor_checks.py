"""Individual doctor checks against a fake machine (agent/doctor/checks/*)."""

import sqlite3
import stat
from pathlib import Path

import pytest

from agent import sysinfo
from agent.doctor import core
from agent.doctor.core import Context
from tests._doctorfake import CFG, GB, UP, FakeProbes, G, ctx_for


def one(cid, tmp, cfg=None, **kw):
    """Run one registered check by id and return its single result (or the list)."""
    if cfg is not None:
        kw.setdefault("raw_cfg", cfg)                    # the raw config.yaml the probes return, same content
    ctx = ctx_for(tmp, cfg, **kw)
    core.discover()
    spec = core.REGISTRY[cid]
    res = core.run_checks(ctx, [spec])
    return res[0] if len(res) == 1 else res


# ----------------------------------------------------------------------------- models / memory / gpu


def test_model_fit_verdicts():
    from agent.doctor.checks.models import fit_verdict

    gpu = [G(total_mb=8192, used_mb=500)]
    assert fit_verdict(4 * GB, gpu, 8000, 16000)[0] == "ok"
    st, text = fit_verdict(int(6.5 * GB), [G(total_mb=8192, used_mb=3000)], 8000, 16000)
    assert st == "warn" and "only when it is empty" in text
    st, text = fit_verdict(12 * GB, gpu, 20000, 32000)
    assert st == "warn" and "CPU" in text
    st, text = fit_verdict(30 * GB, [], 6000, 8000)
    assert st == "fail" and "more than this machine has" in text
    assert fit_verdict(5 * GB, [], 16000, 32000)[0] == "ok"
    assert fit_verdict(5 * GB, [], 3000, 8000)[0] == "warn"


def test_models_fit_row_per_model_and_skips_remote(tmp_path):
    res = one("models.fit", tmp_path, ollama=UP, gpus=[G(total_mb=24576)])
    ids = {r.id: r.status for r in res}
    assert ids["models.fit.chat"] == "ok" and ids["models.fit.coding"] == "ok" and "models.fit.embed" in ids
    remote = type(UP)(True, "1", UP.models, url="http://100.64.0.5:11434")
    assert one("models.fit", tmp_path, ollama=remote).status == "skip"


def test_models_configured_keeps_pull_command_and_fix(tmp_path):
    info = type(UP)(True, "1", {"bge-m3": 1, "qwen3-coder:30b": 1}, url="http://localhost:11434")
    res = {r.id: r for r in one("models", tmp_path, ollama=info)}
    chat = res["model_chat"]
    assert chat.status == "fail" and chat.command == "ollama pull qwen3.5:9b" and callable(chat.fix)
    down = {r.id: r for r in one("models", tmp_path)}
    assert down["model_chat"].status == "skip" and "cannot verify" in down["model_chat"].detail


def test_models_capabilities(tmp_path):
    ok_ = one("models.capabilities", tmp_path, ollama=UP, show={"qwen3.5:9b": {"capabilities": ["completion", "tools"]}})
    assert ok_.status == "ok"
    no_tools = one("models.capabilities", tmp_path, ollama=UP, show={"qwen3.5:9b": {"capabilities": ["completion"]}})
    assert no_tools.status == "warn"
    unknown = one("models.capabilities", tmp_path, ollama=UP, show={"qwen3.5:9b": {}})
    assert unknown.status == "info"


def test_memory_checks(tmp_path):
    low = sysinfo.SysInfo(ram_total_mb=8192, ram_free_mb=400)
    assert one("memory.ram", tmp_path, si=low).status == "warn"
    assert one("memory.ram", tmp_path).status == "ok"
    mi = "MemTotal: 8000000 kB\nSwapTotal: 4000000 kB\nSwapFree: 500000 kB\n"
    assert one("memory.swap", tmp_path, files={"/proc/meminfo": mi}).status == "warn"
    assert one("memory.swap", tmp_path, files={"/proc/meminfo": "MemTotal: 1 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"}).status == "info"
    assert one("memory.swap", tmp_path).status == "skip"
    assert one("memory.pressure", tmp_path, files={"/proc/pressure/memory": "some avg10=25.00 avg60=1 avg300=1 total=1\n"}).status == "warn"
    assert one("memory.pressure", tmp_path, files={"/proc/pressure/memory": "some avg10=0.50 avg60=1 avg300=1 total=1\n"}).status == "ok"


def test_memory_headroom_for_cpu_only_and_gpu(tmp_path):
    tight = sysinfo.SysInfo(ram_total_mb=8192, ram_free_mb=5000)
    assert one("memory.headroom", tmp_path, ollama=UP, si=tight).status in ("warn", "fail")
    assert one("memory.headroom", tmp_path, ollama=UP, gpus=[G(total_mb=24576)]).status == "ok"


def test_gpu_nvidia_parses_driver_cuda_and_vram(tmp_path):
    smi = "/usr/bin/nvidia-smi"
    runs = {(smi, "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"):
            (0, "NVIDIA GeForce RTX 4070, 550.54.14, 12282, 800\n"),
            (smi,): (0, "| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4 |")}
    r = one("gpu.nvidia", tmp_path, bins=("git", "nvidia-smi"), runs=runs, gpus=[G(vendor="nvidia")])
    assert r.status == "ok" and "550.54.14" in r.detail and "CUDA 12.4" in r.detail and "800/12282" in r.detail
    missing = one("gpu.nvidia", tmp_path, gpus=[G(vendor="nvidia")])
    assert missing.status == "fail" and "driver" in missing.detail


def test_gpu_nvidia_absent_without_card(tmp_path):
    ctx = ctx_for(tmp_path)
    core.discover()
    assert core.run_checks(ctx, [core.REGISTRY["gpu.nvidia"]]) == []


def test_gpu_vulkan_flags_software_renderer(tmp_path):
    tool = "/usr/bin/vulkaninfo"
    soft = one("gpu.vulkan", tmp_path, bins=("vulkaninfo",), gpus=[G()], runs={(tool, "--summary"): (0, "deviceName = llvmpipe (LLVM 15)\n")})
    assert soft.status == "warn" and "software" in soft.detail
    hw = one("gpu.vulkan", tmp_path, bins=("vulkaninfo",), gpus=[G()], runs={(tool, "--summary"): (0, "deviceName = AMD Radeon RX 7900\n")})
    assert hw.status == "ok"
    assert one("gpu.vulkan", tmp_path, gpus=[G()]).status == "info"


def test_gpu_cpu_only_gets_model_advice(tmp_path):
    r = one("gpu", tmp_path, si=sysinfo.SysInfo(ram_total_mb=4096, ram_free_mb=2000))
    assert r.status == "warn" and "qwen3.5:2b" in r.hint


def test_apple_metal(tmp_path):
    runs = {("system_profiler", "SPDisplaysDataType"): (0, "Chipset Model: Apple M2\n  Metal Support: Metal 3\n")}
    mac = sysinfo.SysInfo(os_family="macos", ram_total_mb=16384, ram_free_mb=8000)
    r = one("gpu.metal", tmp_path, si=mac, runs=runs)
    assert r.status == "ok" and "Apple M2" in r.detail and "unified memory 16 GB" in r.detail


# ----------------------------------------------------------------------------- disk / network / ports


def test_disk_thresholds(tmp_path):
    assert one("disk.data", tmp_path, disk=(500 * GB, 100 * 1024 ** 2)).status == "fail"
    assert one("disk.data", tmp_path, disk=(500 * GB, 1 * GB)).status == "warn"
    assert one("disk.data", tmp_path).status == "ok"


def test_disk_models_dir_and_remote(tmp_path):
    models = tmp_path / "home" / ".ollama" / "models"
    models.mkdir(parents=True)
    r = one("disk.models", tmp_path, disk=(500 * GB, 2 * GB))
    assert r.status == "warn" and "ollama rm" in r.hint and "3.0 GB of models" in r.detail
    assert one("disk.models", tmp_path).status == "ok"


def test_network_checks(tmp_path):
    assert one("network.dns", tmp_path).status == "ok"
    assert one("network.dns", tmp_path, dns=(False, None, "gaierror")).status == "warn"
    assert one("network.https", tmp_path, https=(False, None, "ConnectError")).status == "warn"
    slow = one("network.https", tmp_path, https=(True, 2500.0, ""))
    assert slow.status == "warn" and "slow" in slow.detail


def test_proxy_credentials_never_printed(tmp_path):
    r = one("network.proxy", tmp_path, env={"HTTPS_PROXY": "http://user:secret@proxy.example.com:3128"})
    assert "secret" not in r.detail and "user:" not in r.detail and "proxy.example.com" in r.detail
    assert r.status == "warn" and "NO_PROXY" in r.hint


def test_ports(tmp_path):
    assert one("ports.ollama", tmp_path, ollama=UP, ports=(11434,)).status == "ok"
    assert one("ports.ollama", tmp_path, ports=(11434,)).status == "warn"          # busy, but not answering as Ollama
    assert one("ports.ollama", tmp_path).status == "info"
    assert one("ports.web", tmp_path, ports=(8420,)).status == "info"
    assert one("ports.web", tmp_path).status == "ok"


# ----------------------------------------------------------------------------- terminal / filesystem / database


def test_terminal_env_and_clipboard(tmp_path):
    assert one("terminal.env", tmp_path, env={"TERM": "dumb"}).status == "warn"
    assert one("terminal.env", tmp_path, env={"TERM": "xterm-256color", "COLORTERM": "truecolor", "SSH_TTY": "x"}).status == "ok"
    assert one("terminal.env", tmp_path, env={"TERM": "xterm", "TMUX": "1"}).status == "info"          # works, fewer colours
    assert one("terminal.env", tmp_path, env={"TERM": "vt100", "TMUX": "1"}).status == "warn"
    assert one("terminal.clipboard", tmp_path, bins=("wl-copy",), env={"WAYLAND_DISPLAY": "w"}).status == "ok"
    assert one("terminal.clipboard", tmp_path).status == "info"


def test_webtoken_mode_and_fix_is_chmod_600(tmp_path):
    cfgdir = tmp_path / "home" / ".config" / "agent"
    cfgdir.mkdir(parents=True)
    tok = cfgdir / "web-token"
    tok.write_text("secret")
    tok.chmod(0o644)
    r = one("filesystem.webtoken", tmp_path)
    assert r.status == "warn" and r.command == f"chmod 600 {tok}" and r.fix is not None
    assert r.fix() == "done" and stat.S_IMODE(tok.stat().st_mode) == 0o600
    assert one("filesystem.webtoken", tmp_path).status == "ok"
    assert one("filesystem.webtoken", tmp_path / "x" if False else tmp_path, env={"XDG_CONFIG_HOME": str(tmp_path / "nowhere")}).status == "info"


def test_filesystem_dirs_offer_mkdir_fix(tmp_path):
    res = {r.id: r for r in one("filesystem.dirs", tmp_path)}
    assert res["filesystem.config_dir"].status == "info" and res["filesystem.config_dir"].command.startswith("mkdir -p ")
    res["filesystem.config_dir"].fix()
    assert (tmp_path / "home" / ".config" / "agent").is_dir()


def test_database_checks(tmp_path):
    assert one("database.integrity", tmp_path).status == "skip"
    db = tmp_path / "data" / "agent.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t(x)")
    con.execute("PRAGMA user_version = 3")
    con.commit()
    con.close()
    assert one("database.integrity", tmp_path).status == "ok"
    sch = one("database.schema", tmp_path)
    assert sch.status in ("ok", "info", "warn") and "schema version 3" in sch.detail
    assert one("database.size", tmp_path).status == "ok"
    db.write_bytes(b"this is not a database" * 100)
    bad = one("database.integrity", tmp_path)
    assert bad.status == "fail"


def test_database_check_never_creates_the_file(tmp_path):
    one("database.integrity", tmp_path)
    one("database.schema", tmp_path)
    assert not (tmp_path / "data" / "agent.db").exists()


# ----------------------------------------------------------------------------- config / security


def test_config_unknown_keys_suggest_the_right_one(tmp_path):
    res = one("config.keys", tmp_path, cfg={"modles": {"chat": "x"}, "safety": {}})
    assert res.status == "warn" and "modles (did you mean models?)" in res.detail
    assert one("config.keys", tmp_path, cfg={"models": {}, "safety": {}}).status == "ok"


def test_config_models_sanity(tmp_path):
    assert one("config.models", tmp_path, {"models": {"chat": "bge-m3", "embed": "bge-m3"}}).status == "warn"
    assert one("config.models", tmp_path).status == "ok"


def test_config_safety_flags_broad_safe_commands(tmp_path):
    res = one("config.safety", tmp_path, cfg={"safety": {"safe_commands": ["rm -rf build", "ls"]}})
    assert res.status == "warn" and "rm -rf build" in res.detail
    ok_ = one("config.safety", tmp_path, cfg={"safety": {"safe_commands": ["ls"], "blocked_patterns": ["x"], "max_timeout_s": 60}})
    assert ok_.status == "ok"


def test_security_web_auth_and_bind(tmp_path):
    assert one("security.web_auth", tmp_path, env={"AGENT_WEB_AUTH": "off"}).status == "warn"
    assert one("security.web_auth", tmp_path).status == "ok"
    assert one("security.web_bind", tmp_path, env={"AGENT_WEB_HOST": "0.0.0.0"}).status == "warn"
    assert one("security.web_bind", tmp_path, env={"AGENT_WEB_HOST": "127.0.0.1"}).status == "ok"


def test_security_trusted_ips_counts_and_flags_wide_ranges(tmp_path):
    p = tmp_path / "home" / ".config" / "agent" / "web-trusted-ips"
    p.parent.mkdir(parents=True)
    p.write_text("100.64.0.5\n# a comment\n192.168.1.4\n")
    r = one("security.trusted_ips", tmp_path)
    assert r.status == "ok" and "2 trusted" in r.detail
    p.write_text("0.0.0.0/0\n")
    assert one("security.trusted_ips", tmp_path).status == "warn"
    p.unlink()
    assert one("security.trusted_ips", tmp_path).status == "skip"


def test_security_secrets_are_found_but_never_printed(tmp_path):
    cfg = {"whatsapp": {"api_token": "abcd1234efgh5678"}, "x": {"note": "sk-" + "A" * 24}, "models": {"chat": "q"}}
    r = one("security.secrets", tmp_path, cfg=cfg)
    assert r.status == "warn" and "whatsapp.api_token" in r.detail and "x.note" in r.detail
    assert "abcd1234" not in r.detail and "AAAA" not in r.detail
    assert one("security.secrets", tmp_path, cfg={"models": {"chat": "q"}}).status == "ok"
    assert one("security.secrets", tmp_path, cfg={"a": {"token": "$MY_TOKEN"}}).status == "ok"


def test_ollama_host_warns_on_plain_http_public(tmp_path):
    pub = type(UP)(True, "1", {}, url="http://8.8.8.8:11434")
    assert one("ollama.host", tmp_path, ollama=pub).status == "warn"
    ts = type(UP)(True, "1", {}, url="http://100.64.0.5:11434")
    assert one("ollama.host", tmp_path, ollama=ts).status == "info"
    assert one("security.ollama_exposed", tmp_path, env={"OLLAMA_HOST": "0.0.0.0:11434"}).status == "warn"


def test_ollama_loaded_flags_split_models(tmp_path):
    ps = [{"name": "big:27b", "size": 20 * GB, "size_vram": 8 * GB}]
    assert one("ollama.loaded", tmp_path, ollama=UP, ps=ps).status == "warn"
    assert one("ollama.loaded", tmp_path, ollama=UP, ps=[{"name": "a", "size": GB, "size_vram": GB}]).status == "info"


# ----------------------------------------------------------------------------- system modules


def test_linux_units_failed_service_is_reported(tmp_path):
    runs = {("systemctl", "--user", "is-active", "ollama-vk.service"): (3, "failed\n"),
            ("systemctl", "--user", "is-active", "ollama.service"): (3, "inactive\n"),
            ("systemctl", "--user", "is-active", "nightage-worker.service"): (3, "inactive\n")}
    r = one("linux.units", tmp_path, bins=("systemctl",), runs=runs)
    assert r.status == "warn" and "journalctl --user -u ollama-vk" in r.hint


def test_linux_cgroup_limit(tmp_path):
    r = one("linux.cgroup", tmp_path, files={"/sys/fs/cgroup/memory.max": str(2 * GB)}, si=sysinfo.SysInfo(ram_total_mb=16384, distro="debian"))
    assert r.status == "warn" and "2 GB" in r.detail
    assert one("linux.cgroup", tmp_path, files={"/sys/fs/cgroup/memory.max": "max"}).status == "ok"


def test_nixos_checks_only_on_nixos(tmp_path):
    nix = sysinfo.SysInfo(os_family="linux", distro="nixos", init_system="systemd")
    assert one("nixos.nixld", tmp_path, si=nix).status == "warn"
    assert one("nixos.nixld", tmp_path, si=nix, env={"NIX_LD": "/lib/ld.so"}).status == "ok"
    core.discover()
    assert not core.REGISTRY["nixos.nixld"].applies(ctx_for(tmp_path).tags)          # a Debian machine never runs it
    assert core.REGISTRY["nixos.nixld"].applies(ctx_for(tmp_path, si=nix).tags)


def test_termux_checks(tmp_path):
    phone = sysinfo.SysInfo(os_family="android-termux", is_termux=True, ram_total_mb=4096)
    assert one("termux.api", tmp_path, si=phone).status == "info"
    assert one("termux.api", tmp_path, si=phone, bins=("termux-battery-status",)).status == "ok"
    r = one("termux.battery", tmp_path, si=phone, bins=("termux-battery-status",),
            runs={("termux-battery-status",): (0, '{"percentage": 12, "status": "DISCHARGING"}')})
    assert r.status == "warn" and "12%" in r.detail
    assert one("termux.battery", tmp_path, si=phone).status == "info"
    assert one("termux.wakelock", tmp_path, si=phone, bins=("termux-wake-lock",)).status == "info"


def test_macos_rosetta_warning(tmp_path):
    mac = sysinfo.SysInfo(os_family="macos")

    class Rosetta(FakeProbes):
        def platform_info(self):
            return {**super().platform_info(), "machine": "x86_64"}
    ctx = Context(CFG, Rosetta(tmp_path, si=mac, runs={("sysctl", "-n", "sysctl.proc_translated"): (0, "1\n")}))
    core.discover()
    r = core.run_checks(ctx, [core.REGISTRY["macos.arch"]])[0]
    assert r.status == "warn" and "Rosetta" in r.detail


def test_wsl_data_on_windows_drive(tmp_path):
    class OnMnt(FakeProbes):
        def data_paths(self, config):
            return Path("/mnt/c/data"), Path("/mnt/c/data/agent.db")
    ctx = Context(CFG, OnMnt(tmp_path, wsl=True))
    core.discover()
    r = core.run_checks(ctx, [core.REGISTRY["windows.wsl"]])[0]
    assert r.status == "warn" and "/mnt" in r.detail


# ----------------------------------------------------------------------------- deep


def test_deep_checks_use_the_probes(tmp_path):
    kw = dict(ollama=UP, deep=True)
    assert one("deep.generate", tmp_path, **kw).status == "ok"
    assert "12.5 tokens/s" in one("deep.generate", tmp_path, **kw).detail
    slow = one("deep.generate", tmp_path, gen={"error": "", "ttft_ms": 9000.0, "tokens": 5, "tps": 1.0, "load_ms": 5000.0}, **kw)
    assert slow.status == "warn"
    assert one("deep.generate", tmp_path, gen={"error": "ReadTimeout"}, **kw).status == "fail"
    assert "1024 dimensions" in one("deep.embed", tmp_path, **kw).detail
    assert one("deep.toolcall", tmp_path, **kw).status == "ok"
    prose = one("deep.toolcall", tmp_path, tool={"called": False, "prose": True, "ms": 1.0}, **kw)
    assert prose.status == "warn"


def test_deep_checks_are_skipped_without_ollama(tmp_path):
    assert one("deep.generate", tmp_path, deep=True).status == "skip"


@pytest.mark.parametrize("ram,gpus,expect", [
    (2048, [], "qwen3.5:0.8b"), (8192, [], "qwen3.5:4b"), (16384, [], "qwen3.5:9b"),
    (65536, [G(total_mb=24576)], "qwen3.5:27b"), (16384, [G(total_mb=8192)], "qwen3.5:4b"),
])
def test_recommendation_tiers(tmp_path, ram, gpus, expect):
    r = one("deep.recommend", tmp_path, si=sysinfo.SysInfo(ram_total_mb=ram, ram_free_mb=ram // 2), gpus=gpus, deep=True)
    assert r.hint == f"ollama pull {expect}" and expect in r.detail
    assert any(line.startswith(">") and expect in line for line in r.lines)
    assert sum(1 for line in r.lines if "ollama pull" in line) == 4
    assert r.command == f"ollama pull {expect}" and callable(r.fix)


# ----------------------------------------------------------------------------- plugins (additive API)


def test_plugin_doctor_check_api_and_runner(tmp_path, monkeypatch):
    from agent import plugin_host
    from agent.plugin_api import RUNTIME, PluginError
    from agent.plugin_testing import load_plugin

    pdir = tmp_path / "plugins"
    pdir.mkdir()
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(pdir))
    folder = plugin_host.scaffold("probe-plug", pdir)
    (folder / "plugin.py").write_text(
        "def register(api):\n"
        "    api.doctor_check('disk', 'Disk look', lambda: ('warn', 'nearly full', 'delete things'))\n"
        "    api.doctor_check('boom', 'Broken', lambda: 1 / 0)\n"
        "    api.doctor_check('fine', 'Fine', lambda: 'all good')\n")
    api = load_plugin(folder)
    assert api.provided.doctor_checks == ["disk", "boom", "fine"] and "doctor checks" in api.provided.summary()
    plugin_host.set_enabled("probe-plug", True)
    ctx = ctx_for(tmp_path, {"models": {}})
    core.discover()
    res = {r.id: r for r in core.run_checks(ctx, [core.REGISTRY["plugins.checks"]])}
    assert res["plugin.probe-plug.disk"].status == "warn" and res["plugin.probe-plug.disk"].hint == "delete things"
    assert res["plugin.probe-plug.boom"].status == "warn" and "could not run" in res["plugin.probe-plug.boom"].detail
    assert res["plugin.probe-plug.fine"].status == "ok"
    assert all(r.fix is None for r in res.values())                     # a plugin check can never offer a fix
    with pytest.raises(PluginError):
        api.doctor_check("Bad Id", "t", lambda: "x")
    with pytest.raises(PluginError):
        api.doctor_check("good", "t", "not callable")
    RUNTIME.forget("probe-plug")


def test_plugins_manifest_check_reports_problems(tmp_path, monkeypatch):
    from agent import plugin_host

    pdir = tmp_path / "plugins"
    pdir.mkdir()
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(pdir))
    plugin_host.scaffold("quiet-plug", pdir)
    core.discover()
    res = core.run_checks(ctx_for(tmp_path), [core.REGISTRY["plugins.manifest"]])
    assert any(r.id == "plugins.quiet-plug" and r.status in ("ok", "warn") for r in res)


def test_embed_rows_show_the_label_and_warn_when_large_for_the_free_ram(tmp_path):
    res = {r.id: r for r in one("models", tmp_path, ollama=UP)}
    assert "Multilingual" in res["model_embed"].detail and res["model_embed"].detail.startswith("bge-m3")
    small = sysinfo.SysInfo(ram_total_mb=4096, ram_free_mb=1500, cpu_count=4, os_family="linux", distro="debian", init_system="systemd")
    fit = {r.id: r for r in one("models.fit", tmp_path, ollama=UP, si=small)}
    row = fit["models.fit.embed"]
    assert row.status == "warn" and "bge-m3 needs about" in row.detail
    assert "embed use" in row.hint
    roomy = {r.id: r for r in one("models.fit", tmp_path, ollama=UP, gpus=[G(total_mb=24576)])}
    assert roomy["models.fit.embed"].status in ("ok", "info")


def test_memory_roundtrip_by_meaning_by_keyword_and_with_a_broken_embedder(tmp_path):
    cfg = {"models": {"chat": "qwen3.5:9b", "embed": "bge-m3"}, "memory": {"recall_k": 6, "recall_min_similarity": 0.2}}
    good = one("memory.roundtrip", tmp_path, cfg, ollama=UP)
    assert good.status == "ok" and "recalled by meaning with bge-m3" in good.detail
    down = one("memory.roundtrip", tmp_path, cfg)
    assert down.status == "warn" and "Ollama is not running" in down.detail and "by keyword" in down.detail
    missing = type(UP)(True, "1", {"qwen3.5:9b": 1}, url="http://localhost:11434")
    assert "not installed" in one("memory.roundtrip", tmp_path, cfg, ollama=missing).detail
    broken = one("memory.roundtrip", tmp_path, cfg, ollama=UP, embed={"error": "Ollama answered 500"})
    assert broken.status == "warn" and "Ollama answered 500" in broken.detail
    strict = {**cfg, "memory": {"recall_k": 6, "recall_min_similarity": 0.99}}
    assert "recall_min_similarity" in one("memory.roundtrip", tmp_path, strict, ollama=UP).detail


def test_termux_ollama_checks(tmp_path):
    from agent.doctor.probes import OllamaInfo

    phone = sysinfo.SysInfo(os_family="android-termux", is_termux=True, ram_total_mb=11000)
    down = OllamaInfo(url="http://localhost:11434")
    r = one("termux.ollama", tmp_path, si=phone, ollama=down)
    assert r.status == "warn" and "pkg install ollama" in r.hint                       # no binary
    r = one("termux.ollama", tmp_path, si=phone, ollama=down, bins=("ollama",))
    assert r.status == "warn" and "nightage ollama start" in r.hint                    # binary, no server
    up = OllamaInfo(url="http://localhost:11434", reachable=True, version="0.40.11", models={"all-minilm:latest": 46_000_000})
    r = one("termux.ollama", tmp_path, si=phone, ollama=up, bins=("ollama",), cfg={"models": {"chat": "qwen3.5:4b"}})
    assert r.status == "warn" and "nightage pull qwen3.5:4b" in r.hint                 # server, no chat model
    up.models["qwen3.5:4b"] = 3_400_000_000
    assert one("termux.ollama", tmp_path, si=phone, ollama=up, bins=("ollama",), cfg={"models": {"chat": "qwen3.5:4b"}}).status == "ok"
    remote = OllamaInfo(url="http://100.64.0.5:11434", reachable=True, version="0.40.11")
    assert one("termux.ollama", tmp_path, si=phone, ollama=remote).status == "info"


def test_termux_phantom_note_is_unverified_and_ordered_safest_first(tmp_path):
    from agent.doctor.probes import OllamaInfo

    phone = sysinfo.SysInfo(os_family="android-termux", is_termux=True, ram_total_mb=11000)
    r = one("termux.phantom", tmp_path, si=phone, ollama=OllamaInfo(url="http://localhost:11434"), bins=("ollama",))
    assert r.status == "info" and "UNVERIFIED" in r.detail
    assert r.hint.index("wake-lock") < r.hint.index("Developer options") < r.hint.index("adb")
    assert one("termux.phantom", tmp_path, si=phone, ollama=OllamaInfo(url="http://100.64.0.5:11434", reachable=True)) == []         # remote model: nothing to say


def test_config_models_check(tmp_path):
    from agent.doctor.probes import OllamaInfo

    up = OllamaInfo(url="http://localhost:11434", reachable=True, version="0.40", models={"qwen3.5:0.8b": 1_000_000_000, "all-minilm:latest": 46_000_000})
    r = one("config.roles", tmp_path, ollama=up, cfg={})
    assert r.status == "info" and any(x.startswith("chat: qwen3.5:0.8b") and "installed models" in x for x in r.lines)
    r = one("config.roles", tmp_path, ollama=up, cfg={"models": {"chat": "llama3:8b"}})
    assert r.status == "warn" and "llama3:8b" in r.detail and "nightage models use qwen3.5:0.8b" in r.hint
    r = one("config.roles", tmp_path, ollama=OllamaInfo(url="http://localhost:11434"), cfg={"models": {"chat": "auto"}})
    assert r.status == "info" and not any("NOT installed" in x for x in r.lines)
