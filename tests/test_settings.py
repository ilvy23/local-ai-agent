from __future__ import annotations

import agent.settings as s


def test_current_lists_editable(monkeypatch):
    monkeypatch.setattr(s, "load_config", lambda: {
        "models": {"chat": "dolphin3:8b"},
        "memory": {"recall_k": 6}, "background": {"pause_on_game": True, "max_gpu_percent": 40},
    })
    rows = dict((n, v) for n, _h, v in s.current())
    assert rows["chat_model"] == "dolphin3:8b"
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
    with pytest.raises(KeyError):
        s.set_value("nope", "x")
