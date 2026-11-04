"""agent.configio: settings are written into the user's own config.yaml - comments, order and `auto` roles survive, the merged
config (shipped defaults, a profile's overlay) is never written, and an empty file is fine."""

from __future__ import annotations

import yaml

from agent import configio


def test_set_keys_edits_in_place_and_keeps_comments_and_auto(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("# mine\nmodels:\n  chat: auto   # let it pick\n  background: auto\npersona:\n  name: Agent\n")
    assert configio.set_keys(p, {"models.chat": "gpt-oss:20b", "persona.name": "Nova"}) == "text"
    text = p.read_text()
    assert text.startswith("# mine\n") and "chat: gpt-oss:20b   # let it pick" in text and "background: auto" in text
    assert yaml.safe_load(text) == {"models": {"chat": "gpt-oss:20b", "background": "auto"}, "persona": {"name": "Nova"}}


def test_new_nested_keys_are_added_where_they_belong(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("models:\n  chat: a\n")
    configio.set_keys(p, {"coding.executor.model": "coder:7b", "models.coding": "coder:7b"})
    assert yaml.safe_load(p.read_text()) == {"models": {"chat": "a", "coding": "coder:7b"},
                                             "coding": {"executor": {"model": "coder:7b"}}}


def test_an_empty_or_missing_file_works(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("")
    configio.set_keys(p, {"models.chat": "m:1"})
    assert yaml.safe_load(p.read_text()) == {"models": {"chat": "m:1"}}
    q = tmp_path / "new.yaml"
    configio.set_keys(q, {"a.b": True})
    assert yaml.safe_load(q.read_text()) == {"a": {"b": True}}


def test_only_the_users_own_values_are_written_never_the_merged_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("profile: default\nmodels:\n  chat: auto\n")
    configio.set_keys(p, {"whatsapp.loop_enabled": False})
    data = yaml.safe_load(p.read_text())
    assert data == {"profile": "default", "models": {"chat": "auto"}, "whatsapp": {"loop_enabled": False}}


def test_a_long_multi_line_value_falls_back_to_a_clean_dump(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("persona:\n  style: short\n")
    assert configio.set_keys(p, {"persona.style": "line one\nline two\n"}) == "dump"
    assert yaml.safe_load(p.read_text())["persona"]["style"] == "line one\nline two\n"


def test_code_model_now_really_changes_the_coding_model(tmp_path, monkeypatch):
    from agent import config as cfgmod
    from agent.coding import config as coding_config
    from agent.coding import models as coding_models

    p = tmp_path / "config.yaml"
    p.write_text("models:\n  coding: old-coder:7b   # mine\n")
    monkeypatch.setattr(cfgmod, "DEFAULT_CONFIG_PATH", p)
    coding_models.set_model("new-coder:14b")
    data = yaml.safe_load(p.read_text())
    assert data["models"]["coding"] == "new-coder:14b" and data["coding"]["executor"]["model"] == "new-coder:14b"
    assert "# mine" in p.read_text()
    assert coding_config.load(data)["executor"]["model"] == "new-coder:14b"


def test_auto_roles_follow_the_chat_model_not_an_installed_shipped_default():
    from agent import config as cfgmod
    from agent import modelpick as mp

    installed = {"gpt-oss:20b": 13 * 2**30, "qwen2.5:7b": 4 * 2**30, "bge-m3:latest": 2**30}
    models = {**cfgmod.DEFAULT_CONFIG["models"], "chat": "gpt-oss:20b"}
    res = mp.resolve(models, {"chat"}, installed, ram_mb=32000)
    assert res.roles["background"].model == "gpt-oss:20b"          # was qwen2.5:7b: a second model behind the chosen one
    assert res.roles["chat"].model == "gpt-oss:20b"


def test_follow_chat_moves_code_model_too():
    from agent import modelpick as mp

    cfg = {"models": {"chat": "a", "background": "a"}, "code": {"model": "a"}}
    changed = mp.follow_chat(cfg, "a", "b")
    assert cfg["code"]["model"] == "b" and changed["code"] == "b"
