from __future__ import annotations

import companion.governor as gov

CONFIG = {"background": {"pause_on_game": True, "max_gpu_percent": 40}}


def test_pause_reason_game_wins(monkeypatch):
    monkeypatch.setattr(gov, "game_running", lambda: "proton")
    monkeypatch.setattr(gov, "gpu_utilization", lambda: 5)
    assert "game" in gov.pause_reason(CONFIG)


def test_pause_reason_gpu_over_cap(monkeypatch):
    monkeypatch.setattr(gov, "game_running", lambda: None)
    monkeypatch.setattr(gov, "gpu_utilization", lambda: 80)
    assert "GPU" in gov.pause_reason(CONFIG)


def test_pause_reason_clear(monkeypatch):
    monkeypatch.setattr(gov, "game_running", lambda: None)
    monkeypatch.setattr(gov, "gpu_utilization", lambda: 10)
    assert gov.pause_reason(CONFIG) is None


def test_pause_on_game_disabled(monkeypatch):
    monkeypatch.setattr(gov, "game_running", lambda: "proton")
    monkeypatch.setattr(gov, "gpu_utilization", lambda: 10)
    cfg = {"background": {"pause_on_game": False, "max_gpu_percent": 40}}
    assert gov.pause_reason(cfg) is None


def test_wait_until_clear_returns_when_clear(monkeypatch):
    monkeypatch.setattr(gov, "pause_reason", lambda config, **kw: None)
    gov.wait_until_clear(CONFIG)  # must return immediately, no sleep


def test_gpu_utilization_none_when_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gov.shutil, "which", lambda _: None)
    assert gov.gpu_utilization() is None
