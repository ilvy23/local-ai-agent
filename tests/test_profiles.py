"""Profiles: overlay precedence, the phone values, runtime wiring (tools, loop thresholds, remote host) and the --profile option."""
import pytest
import yaml
from typer.testing import CliRunner

from agent import autonomy, config as cfgmod, llm, profiles, sysinfo
from agent.tools import registry as toolreg


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("NIGHTAGE_PROFILE", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("NIGHTAGE_PLAIN", raising=False)
    monkeypatch.setattr(sysinfo, "_cached", sysinfo.SysInfo(ram_total_mb=4000, unicode_ok=True))
    yield
    llm.configure({})
    autonomy.configure({})
    toolreg.configure({})
    profiles.activate("default")


def load(tmp_path, user=None, **env):
    path = tmp_path / "config.yaml"
    if user is not None:
        path.write_text(yaml.safe_dump(user))
    return cfgmod.load_config(path, tmp_path / "data"), path


def test_default_profile_changes_nothing(tmp_path):
    cfg, _ = load(tmp_path, {})
    assert cfg["profile"] == "default"
    assert cfg["models"]["chat"] == cfgmod.DEFAULT_CONFIG["models"]["chat"]
    assert cfg["tools"] == cfgmod.DEFAULT_CONFIG["tools"]
    assert profiles.overlay_for("default") == {}


def test_resolve_name_precedence():
    assert profiles.resolve_name({"profile": "phone"}, {}) == "phone"
    assert profiles.resolve_name({"profile": "phone"}, {"NIGHTAGE_PROFILE": "default"}) == "default"
    assert profiles.resolve_name({}, {"NIGHTAGE_PROFILE": " PHONE "}) == "phone"
    assert profiles.resolve_name({"profile": "bogus"}, {}) == "default"


def test_phone_overlay_applies_and_user_keys_win(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTAGE_PROFILE", "phone")
    cfg, path = load(tmp_path, {"models": {"chat": "my-own-model:7b"}, "memory": {"recall_k": 9}})
    assert cfg["profile"] == "phone"
    assert cfg["models"]["chat"] == "my-own-model:7b"            # explicit user key beats the overlay
    assert cfg["memory"]["recall_k"] == 9
    assert cfg["models"]["background"] == "qwen3.5:0.8b" and cfg["models"]["embed"] == "all-minilm"
    assert cfg["models"]["max_ctx_ceiling"] == 6144               # 4000 MB RAM tier
    assert cfg["ollama"]["unload_on_exit"] is True and cfg["ollama"]["keep_alive"] == "5m"
    assert cfg["tools"]["max_iterations"] == 20 and "deep_research" in cfg["tools"]["disabled"]
    assert cfg["ui"]["mood_emotes"] is False
    assert yaml.safe_load(path.read_text()) == {"models": {"chat": "my-own-model:7b"}, "memory": {"recall_k": 9}}   # never written back


def test_first_run_writes_plain_defaults_but_returns_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTAGE_PROFILE", "phone")
    cfg, path = load(tmp_path)
    assert cfg["models"]["chat"] == "qwen3.5:2b"
    written = yaml.safe_load(path.read_text())
    assert written["models"]["chat"] == cfgmod.DEFAULT_CONFIG["models"]["chat"] and written["profile"] == "default"


def test_model_tiers_and_ctx_ceiling():
    assert profiles.model_tier(2000) == "qwen3.5:0.8b" and profiles.model_tier(4000) == "qwen3.5:2b"
    assert profiles.model_tier(8000) == "qwen3.5:4b" and profiles.model_tier(0) == "qwen3.5:2b"
    ceilings = [profiles.ctx_ceiling(mb, False) for mb in (2000, 3500, 5000, 7000, 12000)]
    assert ceilings == [4096, 6144, 8192, 12288, 16384]
    assert all(4096 <= c <= 16384 for c in ceilings) and profiles.ctx_ceiling(2000, True) == 16384


def test_remote_gateway_keeps_models(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTAGE_PROFILE", "phone")
    cfg, _ = load(tmp_path, {"ollama": {"host": "100.64.0.8"}})
    assert cfg["models"]["chat"] == cfgmod.DEFAULT_CONFIG["models"]["chat"]        # the home PC's models stay
    assert cfg["models"]["max_ctx_ceiling"] == 16384
    assert llm.current_base_url() == "http://100.64.0.8:11434" and llm.is_remote()


def test_phone_loop_thresholds_are_all_stricter_than_the_defaults():
    from agent.codeagent import engine, loopguard, stages
    c = profiles.phone_overlay(sysinfo.SysInfo(ram_total_mb=4000))["coding"]
    pairs = [("loop_abort", engine.LOOP_ABORT), ("family_note", loopguard.FAMILY_NOTE), ("family_abort", loopguard.FAMILY_ABORT),
             ("run_spam_note", loopguard.RUN_SPAM_NOTE), ("run_spam_abort", loopguard.RUN_SPAM_ABORT),
             ("same_tool_note", loopguard.SAME_TOOL_NOTE), ("same_tool_abort", loopguard.SAME_TOOL_ABORT),
             ("alt_note", loopguard.ALT_NOTE), ("alt_abort", loopguard.ALT_ABORT), ("chant_hits", loopguard.CHANT_HITS["content"]),
             ("chant_think_hits", loopguard.CHANT_HITS["thinking"]), ("malformed_abort", engine.MALFORMED_ABORT),
             ("auto_extend_max", engine.AUTO_EXTEND_MAX), ("announce_max", engine.ANNOUNCE_MAX), ("stray_looks", engine.STRAY_LOOKS),
             ("stage_max_steps", stages.STAGE_MAX_STEPS)]
    for key, default in pairs:
        assert c[key] < default, key
    assert c["loop_note"] <= c["loop_repeat"] <= c["loop_abort"]                      # the ladder keeps its order
    assert c["family_note"] < c["family_abort"] and c["same_tool_note"] < c["same_tool_abort"] and c["alt_note"] < c["alt_abort"]


def test_stuck_watcher_thresholds_come_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTAGE_PROFILE", "phone")
    load(tmp_path, {})
    assert (autonomy.FAIL_STREAK, autonomy.IDENTICAL_CALLS) == (2, 2)
    autonomy.configure({})
    assert (autonomy.FAIL_STREAK, autonomy.REPEAT_SAME_ERROR, autonomy.IDENTICAL_CALLS) == (3, 2, 3)      # unset = module defaults
    autonomy.configure({"autonomy": {"fail_streak": "x", "identical_calls": 0}})
    assert autonomy.FAIL_STREAK == 3 and autonomy.IDENTICAL_CALLS == 3                                   # junk ignored


def test_phone_hides_heavy_tools(tmp_path, monkeypatch):
    base = {t.name for t in toolreg.default_registry().list()}
    assert "get_media_info" in base
    monkeypatch.setenv("NIGHTAGE_PROFILE", "phone")
    load(tmp_path, {})
    phone = {t.name for t in toolreg.default_registry().list()}
    assert "get_media_info" not in phone and "read_file" in phone
    assert toolreg.is_disabled("wa_search_messages") and not toolreg.is_disabled("remember")


def test_plain_glyphs_when_terminal_lacks_unicode():
    env = {}
    assert profiles.apply_terminal_defaults(sysinfo.SysInfo(unicode_ok=False), env) and env["NIGHTAGE_PLAIN"] == "1"
    assert not profiles.apply_terminal_defaults(sysinfo.SysInfo(unicode_ok=True), {})
    assert not profiles.apply_terminal_defaults(sysinfo.SysInfo(unicode_ok=False), {"NIGHTAGE_PLAIN": "0x"})


def test_suggest():
    termux = sysinfo.SysInfo(is_termux=True, ram_total_mb=8000)
    assert "--profile phone" in profiles.suggest(termux, {}, {})
    assert "--profile phone" in profiles.suggest(sysinfo.SysInfo(ram_total_mb=4000), {}, {})
    assert profiles.suggest(sysinfo.SysInfo(ram_total_mb=16000), {}, {}) == ""
    assert profiles.suggest(termux, {"profile": "default"}, {}) == ""        # the user chose
    assert profiles.suggest(termux, {}, {"NIGHTAGE_PROFILE": "phone"}) == ""


def test_cli_profile_option(monkeypatch):
    from agent import main
    seen = {}
    monkeypatch.setattr(main, "_home", lambda: seen.setdefault("home", True) and 0)
    monkeypatch.delenv("NIGHTAGE_PROFILE", raising=False)
    runner = CliRunner()
    bad = runner.invoke(main.app, ["--profile", "nope"])
    assert bad.exit_code != 0
    ok = runner.invoke(main.app, ["--profile", "phone"])
    assert ok.exit_code == 0 and seen.get("home")
    import os
    try:
        assert os.environ["NIGHTAGE_PROFILE"] == "phone"
    finally:
        os.environ.pop("NIGHTAGE_PROFILE", None)           # main() sets it behind monkeypatch's back; do not leak it into other tests
