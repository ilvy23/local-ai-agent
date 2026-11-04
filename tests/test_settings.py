from __future__ import annotations

import agent.settings as s


def test_current_lists_editable(monkeypatch):
    monkeypatch.setattr(s, "load_config", lambda: {
        "models": {"chat": "gemma2:9b"},
        "memory": {"recall_k": 6}, "background": {"pause_on_game": True, "max_gpu_percent": 40},
    })
    # current() yields (name, category, help, value) - category/help were
    # added for the interactive settings browser, after this test was
    # originally written for a plain 3-tuple.
    rows = {name: value for name, _category, _help, value in s.current()}
    assert rows["chat_model"] == "gemma2:9b"
    assert rows["recall_k"] == 6


def test_set_value_coerces_and_writes(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models:\n  chat: old\n")
    monkeypatch.setattr(s, "DEFAULT_CONFIG_PATH", cfg)
    dotted, coerced = s.set_value("max_gpu_percent", "55")
    assert dotted == "background.max_gpu_percent" and coerced == 55
    import yaml
    assert yaml.safe_load(cfg.read_text())["background"]["max_gpu_percent"] == 55


def test_set_value_bool_and_unknown(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n")
    monkeypatch.setattr(s, "DEFAULT_CONFIG_PATH", cfg)
    _, coerced = s.set_value("pause_on_game", "false")
    assert coerced is False
    import pytest
    # ValueError, not KeyError: KeyError's __str__ wraps the message in an
    # extra repr() layer of quotes, printing literal quote characters at
    # every call site that does console.print(f"...{exc}...") - see
    # set_value's own docstring.
    with pytest.raises(ValueError, match="unknown setting"):
        s.set_value("nope", "x")


def test_switch_chat_model_writes_chat_and_followers_once_and_keeps_deliberate_roles(tmp_path, monkeypatch):
    import yaml

    from agent import config as cfgmod
    from agent import settings as s

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("models:\n  chat: old:1\n  coding: coder:7b\n  background: old:1\n")
    monkeypatch.setattr(s, "DEFAULT_CONFIG_PATH", cfg_file)
    monkeypatch.setitem(cfgmod.LAST_LOAD, "user_models", ["coding", "background", "chat"])
    config = {"models": {"chat": "old:1", "background": "old:1", "coding": "coder:7b", "reasoning": "", "unrestricted": ""}}
    writes = []
    real_write = s._write_paths
    monkeypatch.setattr(s, "_write_paths", lambda values: (writes.append(dict(values)), real_write(values)))
    followed, error = s.switch_chat_model(config, "new:2")
    assert error is None and followed == {"background": "new:2"} and len(writes) == 1
    saved = yaml.safe_load(cfg_file.read_text())["models"]
    assert saved == {"chat": "new:2", "background": "new:2", "coding": "coder:7b"}
    assert config["models"]["chat"] == "new:2" and config["models"]["coding"] == "coder:7b"


def test_write_paths_survives_an_empty_config_file(tmp_path, monkeypatch):
    import yaml

    from agent import settings as s

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")
    monkeypatch.setattr(s, "DEFAULT_CONFIG_PATH", cfg_file)
    s.set_value("chat_model", "m:1")
    assert yaml.safe_load(cfg_file.read_text()) == {"models": {"chat": "m:1"}}


def test_the_suite_never_sees_the_real_config_path():
    """The autouse conftest fixture redirects every copy of DEFAULT_CONFIG_PATH; the real file is never a target."""
    from agent import config as cfgmod
    from agent import settings as s

    assert cfgmod.DEFAULT_CONFIG_PATH.parent.name.startswith("nightage-cfg")
    assert s.DEFAULT_CONFIG_PATH == cfgmod.DEFAULT_CONFIG_PATH
    assert cfgmod.load_config()["models"]["chat"]             # a load with no arguments reads the redirected file
    assert cfgmod.last_load()["path"] == str(cfgmod.DEFAULT_CONFIG_PATH)
